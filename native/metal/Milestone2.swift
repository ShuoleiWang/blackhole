import CryptoKit
import Foundation
import Metal

private enum RunnerError: Error, CustomStringConvertible {
    case configuration(String)
    case corpus(String)
    case metal(String)
    case validation(String)

    var description: String {
        switch self {
        case .configuration(let message): return "configuration error: \(message)"
        case .corpus(let message): return "corpus error: \(message)"
        case .metal(let message): return "Metal error: \(message)"
        case .validation(let message): return "validation error: \(message)"
        }
    }
}

private struct Configuration {
    var metallib = ""
    var corpus = ""
    var cpuBenchmark = ""
    var report = ""
    var benchmarkCount = 32_768
    var benchmarkIterations = 7

    static func parse() throws -> Configuration {
        var result = Configuration()
        var arguments = Array(CommandLine.arguments.dropFirst())
        while !arguments.isEmpty {
            let flag = arguments.removeFirst()
            guard !arguments.isEmpty else {
                throw RunnerError.configuration("missing value for \(flag)")
            }
            let value = arguments.removeFirst()
            switch flag {
            case "--metallib": result.metallib = value
            case "--corpus": result.corpus = value
            case "--cpu-benchmark": result.cpuBenchmark = value
            case "--report": result.report = value
            case "--benchmark-count":
                guard let parsed = Int(value), parsed > 0 else {
                    throw RunnerError.configuration("invalid benchmark count: \(value)")
                }
                result.benchmarkCount = parsed
            case "--benchmark-iterations":
                guard let parsed = Int(value), parsed >= 3 else {
                    throw RunnerError.configuration("benchmark iterations must be >= 3")
                }
                result.benchmarkIterations = parsed
            default:
                throw RunnerError.configuration("unknown argument: \(flag)")
            }
        }
        guard !result.metallib.isEmpty,
              !result.corpus.isEmpty,
              !result.cpuBenchmark.isEmpty,
              !result.report.isEmpty else {
            throw RunnerError.configuration(
                "--metallib, --corpus, --cpu-benchmark, and --report are required"
            )
        }
        return result
    }
}

// ABI-identical to the Metal ScaledDD structure (16 bytes, 4-byte alignment).
private struct ScaledDD: Decodable {
    var hi: Float
    var lo: Float
    var exponent: Int32
    var status: UInt32

    init(hi: Float, lo: Float, exponent: Int32, status: UInt32 = 0) {
        self.hi = hi
        self.lo = lo
        self.exponent = exponent
        self.status = status
    }

    init(from decoder: Decoder) throws {
        var values = try decoder.unkeyedContainer()
        hi = try values.decode(Float.self)
        lo = try values.decode(Float.self)
        exponent = try values.decode(Int32.self)
        status = 0
        if !values.isAtEnd {
            throw DecodingError.dataCorruptedError(
                in: values,
                debugDescription: "scaled-DD JSON value must contain exactly three fields"
            )
        }
    }
}

private struct CorpusRecord: Decodable {
    let category: String
    let label: String
    let inverse: [ScaledDD]
    let derivatives: [ScaledDD]
    let covector: [ScaledDD]
    let reference: [Double]
    let referenceScale: [Double]
}

private struct RHSCorpus: Decodable {
    let schema: String
    let encoding: String
    let reference: String
    let realKerrCount: Int
    let adversarialCount: Int
    let records: [CorpusRecord]
}

private struct CPUBenchmark: Decodable {
    let schema: String
    let corpusSHA256: String
    let recordCount: Int
    let repeats: Int
    let bestSeconds: Double
    let recordsPerSecond: Double
    let scope: String
    let checksum: Double

    func dictionary() -> [String: Any] {
        [
            "recordCount": recordCount,
            "schema": schema,
            "corpusSHA256": corpusSHA256,
            "repeats": repeats,
            "bestSeconds": bestSeconds,
            "recordsPerSecond": recordsPerSecond,
            "scope": scope,
            "checksum": checksum,
        ]
    }
}

private struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }

    mutating func unit() -> Double {
        Double(next() >> 11) * 0x1.0p-53
    }
}

private enum ArithmeticOperation: UInt32, CaseIterable {
    case add = 0
    case subtract = 1
    case multiply = 2
    case divide = 3
    case squareRoot = 4

    var name: String {
        switch self {
        case .add: return "add"
        case .subtract: return "subtract"
        case .multiply: return "multiply"
        case .divide: return "divide"
        case .squareRoot: return "sqrt"
        }
    }
}

private struct ArithmeticVector {
    let a: Double
    let b: Double
    let label: String
}

private struct AccuracyMetrics {
    var count = 0
    var exactCount = 0
    var signMismatchCount = 0
    var zeroMismatchCount = 0
    var signMismatchExamples: [String] = []
    var zeroMismatchExamples: [String] = []
    var nonfiniteCount = 0
    var maximumULP = 0.0
    var minimumEffectiveBits = 53.0
    var maximumRelativeError = 0.0
    var maximumConditionNormalizedError = 0.0
    var maximumAbsoluteError = 0.0
    var minimumConditionNormalizedBits = 53.0
    var worstLabel = ""
    var worstComponent = -1
    var worstReference = 0.0
    var worstActual = 0.0

    mutating func record(
        reference: Double,
        actual: Double,
        label: String,
        component: Int = -1,
        normalizationScale: Double? = nil
    ) {
        count += 1
        guard reference.isFinite, actual.isFinite else {
            nonfiniteCount += 1
            return
        }
        let error = abs(actual - reference)
        if error == 0.0 { exactCount += 1 }
        if (reference == 0.0) != (actual == 0.0) {
            zeroMismatchCount += 1
            if zeroMismatchExamples.count < 8 {
                zeroMismatchExamples.append("\(label):component-\(component)")
            }
        }
        if reference != 0.0 && actual != 0.0
            && reference.sign != actual.sign {
            signMismatchCount += 1
            if signMismatchExamples.count < 8 {
                signMismatchExamples.append("\(label):component-\(component)")
            }
        }
        let ulp = reference == 0.0
            ? Double.leastNonzeroMagnitude
            : abs(reference.nextUp - reference)
        let ulpError = error / ulp
        let scale = max(abs(reference), Double.leastNonzeroMagnitude)
        let relativeError = error / scale
        let effectiveBits = error == 0.0
            ? 53.0
            : max(0.0, -log2(relativeError))
        let conditionScale = max(
            normalizationScale ?? abs(reference),
            Double.leastNonzeroMagnitude
        )
        let conditionNormalizedError = error / conditionScale
        let conditionBits = error == 0.0
            ? 53.0
            : max(0.0, -log2(conditionNormalizedError))
        maximumULP = max(maximumULP, ulpError)
        minimumEffectiveBits = min(minimumEffectiveBits, effectiveBits)
        maximumRelativeError = max(maximumRelativeError, relativeError)
        maximumConditionNormalizedError = max(
            maximumConditionNormalizedError,
            conditionNormalizedError
        )
        maximumAbsoluteError = max(maximumAbsoluteError, error)
        minimumConditionNormalizedBits = min(
            minimumConditionNormalizedBits,
            conditionBits
        )
        if relativeError >= maximumRelativeError {
            worstLabel = label
            worstComponent = component
            worstReference = reference
            worstActual = actual
        }
    }

    func dictionary() -> [String: Any] {
        [
            "count": count,
            "exactCount": exactCount,
            "signMismatchCount": signMismatchCount,
            "zeroMismatchCount": zeroMismatchCount,
            "signMismatchExamples": signMismatchExamples,
            "zeroMismatchExamples": zeroMismatchExamples,
            "nonfiniteCount": nonfiniteCount,
            "maximumBinary64ULP": maximumULP,
            "minimumEffectiveBits": minimumEffectiveBits,
            "maximumRelativeError": maximumRelativeError,
            "maximumConditionNormalizedError": maximumConditionNormalizedError,
            "maximumAbsoluteError": maximumAbsoluteError,
            "minimumConditionNormalizedBits": minimumConditionNormalizedBits,
            "worstLabel": worstLabel,
            "worstComponent": worstComponent,
            "worstReference": worstReference,
            "worstActual": worstActual,
        ]
    }
}

private func encodeScaled(_ value: Double) throws -> ScaledDD {
    guard value.isFinite else {
        throw RunnerError.validation("cannot encode non-finite binary64: \(value)")
    }
    if value == 0.0 {
        return ScaledDD(hi: Float(value), lo: 0.0, exponent: 0)
    }
    var exponent: Int32 = 0
    var mantissa = frexp(value, &exponent)
    if abs(Float(mantissa)) >= 1.0 {
        mantissa *= 0.5
        exponent += 1
    }
    let hi = Float(mantissa)
    let lo = Float(mantissa - Double(hi))
    guard abs(hi) >= 0.5, abs(hi) < 1.0,
          lo == 0.0 || abs(lo) >= Float.leastNormalMagnitude else {
        throw RunnerError.validation("scaled-DD encoding invariant failed for \(value)")
    }
    return ScaledDD(hi: hi, lo: lo, exponent: exponent)
}

private func decodeScaled(_ value: ScaledDD) -> Double {
    guard value.status == 0 else { return .nan }
    return scalbn(Double(value.hi) + Double(value.lo), value.exponent)
}

private func quantized(_ value: Double) throws -> Double {
    decodeScaled(try encodeScaled(value))
}

private final class MetalHarness {
    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let arithmeticPipeline: MTLComputePipelineState
    private let rhsPipeline: MTLComputePipelineState
    private let dopriPipeline: MTLComputePipelineState

    init(metallib: URL) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw RunnerError.metal("no default Metal device")
        }
        self.device = device
        let library = try device.makeLibrary(URL: metallib)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw RunnerError.metal("\(name) is missing from metallib")
            }
            return try device.makeComputePipelineState(function: function)
        }
        arithmeticPipeline = try pipeline("scaled_dd_arithmetic")
        rhsPipeline = try pipeline("scaled_dd_hamiltonian_rhs")
        dopriPipeline = try pipeline("scaled_dd_dopri_combine")
        guard let queue = device.makeCommandQueue() else {
            throw RunnerError.metal("could not create command queue")
        }
        self.queue = queue
    }

    private func makeBuffer<T>(_ values: [T]) throws -> MTLBuffer {
        guard !values.isEmpty else {
            throw RunnerError.metal("cannot allocate an empty input buffer")
        }
        let byteCount = values.count * MemoryLayout<T>.stride
        guard let buffer = values.withUnsafeBytes({ raw in
            device.makeBuffer(
                bytes: raw.baseAddress!,
                length: byteCount,
                options: .storageModeShared
            )
        }) else {
            throw RunnerError.metal("could not allocate \(byteCount)-byte input buffer")
        }
        return buffer
    }

    private func makeOutput<T>(count: Int, _: T.Type) throws -> MTLBuffer {
        let byteCount = count * MemoryLayout<T>.stride
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw RunnerError.metal("could not allocate \(byteCount)-byte output buffer")
        }
        return buffer
    }

    private func dispatch(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        count: Int
    ) {
        let width = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
    }

    private func complete(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
    }

    func arithmetic(
        operation: ArithmeticOperation,
        a: [ScaledDD],
        b: [ScaledDD]
    ) throws -> [ScaledDD] {
        precondition(a.count == b.count)
        let aBuffer = try makeBuffer(a)
        let bBuffer = try makeBuffer(b)
        let output = try makeOutput(count: a.count, ScaledDD.self)
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw RunnerError.metal("could not create arithmetic command resources")
        }
        encoder.setComputePipelineState(arithmeticPipeline)
        encoder.setBuffer(aBuffer, offset: 0, index: 0)
        encoder.setBuffer(bBuffer, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var operationRaw = operation.rawValue
        var countRaw = UInt32(a.count)
        encoder.setBytes(&operationRaw, length: 4, index: 3)
        encoder.setBytes(&countRaw, length: 4, index: 4)
        dispatch(encoder: encoder, pipeline: arithmeticPipeline, count: a.count)
        encoder.endEncoding()
        try complete(commandBuffer)
        let pointer = output.contents().bindMemory(to: ScaledDD.self, capacity: a.count)
        return Array(UnsafeBufferPointer(start: pointer, count: a.count))
    }

    func rhs(
        inverse: [ScaledDD],
        derivatives: [ScaledDD],
        covector: [ScaledDD],
        recordCount: Int
    ) throws -> ([ScaledDD], [UInt32]) {
        precondition(inverse.count == recordCount * 16)
        precondition(derivatives.count == recordCount * 64)
        precondition(covector.count == recordCount * 4)
        let inverseBuffer = try makeBuffer(inverse)
        let derivativeBuffer = try makeBuffer(derivatives)
        let covectorBuffer = try makeBuffer(covector)
        let output = try makeOutput(count: recordCount * 8, ScaledDD.self)
        let status = try makeOutput(count: recordCount, UInt32.self)
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw RunnerError.metal("could not create RHS command resources")
        }
        encodeRHS(
            encoder: encoder,
            inverse: inverseBuffer,
            derivatives: derivativeBuffer,
            covector: covectorBuffer,
            output: output,
            status: status,
            recordCount: recordCount
        )
        encoder.endEncoding()
        try complete(commandBuffer)
        let outputPointer = output.contents().bindMemory(
            to: ScaledDD.self,
            capacity: recordCount * 8
        )
        let statusPointer = status.contents().bindMemory(to: UInt32.self, capacity: recordCount)
        return (
            Array(UnsafeBufferPointer(start: outputPointer, count: recordCount * 8)),
            Array(UnsafeBufferPointer(start: statusPointer, count: recordCount))
        )
    }

    private func encodeRHS(
        encoder: MTLComputeCommandEncoder,
        inverse: MTLBuffer,
        derivatives: MTLBuffer,
        covector: MTLBuffer,
        output: MTLBuffer,
        status: MTLBuffer,
        recordCount: Int
    ) {
        encoder.setComputePipelineState(rhsPipeline)
        encoder.setBuffer(inverse, offset: 0, index: 0)
        encoder.setBuffer(derivatives, offset: 0, index: 1)
        encoder.setBuffer(covector, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBuffer(status, offset: 0, index: 4)
        var countRaw = UInt32(recordCount)
        encoder.setBytes(&countRaw, length: 4, index: 5)
        dispatch(encoder: encoder, pipeline: rhsPipeline, count: recordCount)
    }

    func benchmarkRHS(
        inverse: [ScaledDD],
        derivatives: [ScaledDD],
        covector: [ScaledDD],
        recordCount: Int,
        iterations: Int
    ) throws -> [String: Any] {
        let inverseBuffer = try makeBuffer(inverse)
        let derivativeBuffer = try makeBuffer(derivatives)
        let covectorBuffer = try makeBuffer(covector)
        let output = try makeOutput(count: recordCount * 8, ScaledDD.self)
        let status = try makeOutput(count: recordCount, UInt32.self)
        var gpuSamples: [Double] = []
        var wallSamples: [Double] = []
        for iteration in 0...iterations {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw RunnerError.metal("could not create RHS benchmark command resources")
            }
            let start = ProcessInfo.processInfo.systemUptime
            encodeRHS(
                encoder: encoder,
                inverse: inverseBuffer,
                derivatives: derivativeBuffer,
                covector: covectorBuffer,
                output: output,
                status: status,
                recordCount: recordCount
            )
            encoder.endEncoding()
            try complete(commandBuffer)
            let elapsed = ProcessInfo.processInfo.systemUptime - start
            if iteration > 0 {
                wallSamples.append(elapsed)
                let gpu = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
                gpuSamples.append(gpu > 0 ? gpu : elapsed)
            }
        }
        gpuSamples.sort()
        wallSamples.sort()
        let gpuMedian = gpuSamples[gpuSamples.count / 2]
        let wallMedian = wallSamples[wallSamples.count / 2]
        return [
            "recordCount": recordCount,
            "iterations": iterations,
            "gpuMedianSeconds": gpuMedian,
            "wallMedianSeconds": wallMedian,
            "gpuRecordsPerSecond": Double(recordCount) / gpuMedian,
            "wallRecordsPerSecond": Double(recordCount) / wallMedian,
            "scope": "precomputed MetricSample Hamiltonian RHS only",
        ]
    }

    func dopri(
        state: [ScaledDD],
        stages: [ScaledDD],
        steps: [ScaledDD],
        coefficients: [ScaledDD],
        recordCount: Int
    ) throws -> ([ScaledDD], [UInt32]) {
        let stateBuffer = try makeBuffer(state)
        let stageBuffer = try makeBuffer(stages)
        let stepBuffer = try makeBuffer(steps)
        let coefficientBuffer = try makeBuffer(coefficients)
        let output = try makeOutput(count: recordCount * 16, ScaledDD.self)
        let status = try makeOutput(count: recordCount, UInt32.self)
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw RunnerError.metal("could not create DOPRI command resources")
        }
        encoder.setComputePipelineState(dopriPipeline)
        encoder.setBuffer(stateBuffer, offset: 0, index: 0)
        encoder.setBuffer(stageBuffer, offset: 0, index: 1)
        encoder.setBuffer(stepBuffer, offset: 0, index: 2)
        encoder.setBuffer(coefficientBuffer, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setBuffer(status, offset: 0, index: 5)
        var countRaw = UInt32(recordCount)
        encoder.setBytes(&countRaw, length: 4, index: 6)
        dispatch(encoder: encoder, pipeline: dopriPipeline, count: recordCount)
        encoder.endEncoding()
        try complete(commandBuffer)
        let outputPointer = output.contents().bindMemory(
            to: ScaledDD.self,
            capacity: recordCount * 16
        )
        let statusPointer = status.contents().bindMemory(to: UInt32.self, capacity: recordCount)
        return (
            Array(UnsafeBufferPointer(start: outputPointer, count: recordCount * 16)),
            Array(UnsafeBufferPointer(start: statusPointer, count: recordCount))
        )
    }
}

private func arithmeticReference(
    operation: ArithmeticOperation,
    a: Double,
    b: Double
) -> Double {
    switch operation {
    case .add: return a + b
    case .subtract: return a - b
    case .multiply: return a * b
    case .divide: return a / b
    case .squareRoot: return a.squareRoot()
    }
}

private func arithmeticVectors(for operation: ArithmeticOperation) -> [ArithmeticVector] {
    var result: [ArithmeticVector] = []
    let exponents = [-1070, -1020, -900, -200, -104, -100, -1, 0, 1, 100, 200, 900, 1020]
    let perturbations = [0.0, 0x1.0p-23, -0x1.0p-23, 0x1.0p-40, -0x1.0p-40, 0x1.0p-47]
    for exponent in exponents {
        for (index, perturbation) in perturbations.enumerated() {
            switch operation {
            case .add:
                let scale = scalbn(0.75, Int32(exponent))
                result.append(ArithmeticVector(
                    a: scale * (1.0 + perturbation),
                    b: -scale * (1.0 - perturbation * 0.5),
                    label: "scaled-boundary-add-e\(exponent)-p\(index)"
                ))
            case .subtract:
                let scale = scalbn(0.75, Int32(exponent))
                result.append(ArithmeticVector(
                    a: scale * (1.0 + perturbation),
                    b: scale * (1.0 - perturbation * 0.5),
                    label: "scaled-boundary-sub-e\(exponent)-p\(index)"
                ))
            case .multiply:
                let firstExponent = exponent / 2
                result.append(ArithmeticVector(
                    a: scalbn(0.75 + perturbation, Int32(firstExponent)),
                    b: scalbn(0.625 - perturbation * 0.25, Int32(exponent - firstExponent)),
                    label: "scaled-boundary-mul-e\(exponent)-p\(index)"
                ))
            case .divide:
                let firstExponent = exponent / 2
                result.append(ArithmeticVector(
                    a: scalbn(0.75 + perturbation, Int32(firstExponent)),
                    b: scalbn(0.625 - perturbation * 0.25, Int32(firstExponent - exponent)),
                    label: "scaled-boundary-div-e\(exponent)-p\(index)"
                ))
            case .squareRoot:
                let evenExponent = exponent - (exponent & 1)
                result.append(ArithmeticVector(
                    a: scalbn(0.75 + perturbation, Int32(evenExponent)),
                    b: 1.0,
                    label: "scaled-boundary-sqrt-e\(evenExponent)-p\(index)"
                ))
            }
        }
    }
    var rng = SplitMix64(seed: 0x5343414c45444432 ^ UInt64(operation.rawValue))
    while result.count < 4096 {
        let exponent = Int(rng.next() % 2001) - 1000
        let mantissaA = 0.5 + rng.unit() * 0.49
        let mantissaB = 0.5 + rng.unit() * 0.49
        switch operation {
        case .add, .subtract:
            let a = scalbn(mantissaA, Int32(exponent))
            let near = rng.next() & 1 == 0
            let b = near
                ? a * (1.0 + (rng.unit() - 0.5) * 0x1.0p-44)
                : scalbn(mantissaB, Int32(Int(rng.next() % 2001) - 1000))
            result.append(ArithmeticVector(
                a: a,
                b: operation == .add && near ? -b : b,
                label: "scaled-random-\(result.count)"
            ))
        case .multiply:
            let firstExponent = exponent / 2
            result.append(ArithmeticVector(
                a: scalbn(mantissaA, Int32(firstExponent)),
                b: scalbn(mantissaB, Int32(exponent - firstExponent)),
                label: "scaled-random-\(result.count)"
            ))
        case .divide:
            let firstExponent = exponent / 2
            result.append(ArithmeticVector(
                a: scalbn(mantissaA, Int32(firstExponent)),
                b: scalbn(mantissaB, Int32(firstExponent - exponent)),
                label: "scaled-random-\(result.count)"
            ))
        case .squareRoot:
            let evenExponent = exponent - (exponent & 1)
            result.append(ArithmeticVector(
                a: scalbn(mantissaA, Int32(evenExponent)),
                b: 1.0,
                label: "scaled-random-\(result.count)"
            ))
        }
    }
    return result
}

private func validateArithmetic(
    operation: ArithmeticOperation,
    harness: MetalHarness
) throws -> AccuracyMetrics {
    let vectors = arithmeticVectors(for: operation).filter {
        $0.a.isFinite && $0.b.isFinite
    }
    let a = try vectors.map { try encodeScaled($0.a) }
    let b = try vectors.map { try encodeScaled($0.b) }
    let output = try harness.arithmetic(operation: operation, a: a, b: b)
    var metrics = AccuracyMetrics()
    for index in vectors.indices {
        let qa = decodeScaled(a[index])
        let qb = decodeScaled(b[index])
        let reference = arithmeticReference(operation: operation, a: qa, b: qb)
        guard reference.isFinite else { continue }
        metrics.record(
            reference: reference,
            actual: decodeScaled(output[index]),
            label: vectors[index].label
        )
    }
    return metrics
}

private func validateFailClosedArithmetic(harness: MetalHarness) throws -> [String: Any] {
    let one = try encodeScaled(1.0)
    let zero = try encodeScaled(0.0)
    let negative = try encodeScaled(-1.0)
    let malformed = ScaledDD(hi: .nan, lo: 0.0, exponent: 0, status: 0)
    let divideOutput = try harness.arithmetic(
        operation: .divide,
        a: [one],
        b: [zero]
    )[0]
    let sqrtOutput = try harness.arithmetic(
        operation: .squareRoot,
        a: [negative],
        b: [one]
    )[0]
    let malformedOutput = try harness.arithmetic(
        operation: .add,
        a: [malformed],
        b: [one]
    )[0]
    let passed = divideOutput.status == 2
        && sqrtOutput.status == 3
        && malformedOutput.status == 1
        && decodeScaled(divideOutput).isNaN
        && decodeScaled(sqrtOutput).isNaN
        && decodeScaled(malformedOutput).isNaN
    return [
        "passed": passed,
        "divideByZeroStatus": divideOutput.status,
        "negativeSquareRootStatus": sqrtOutput.status,
        "malformedNonfiniteStatus": malformedOutput.status,
    ]
}

private func validateCorpus(_ corpus: RHSCorpus) throws {
    guard corpus.schema == "blackhole-metal-scaled-dd-rhs-corpus-v1" else {
        throw RunnerError.corpus("unsupported schema \(corpus.schema)")
    }
    guard corpus.records.count == corpus.realKerrCount + corpus.adversarialCount,
          corpus.records.filter({ $0.category == "real-kerr" }).count == corpus.realKerrCount,
          corpus.records.filter({ $0.category == "adversarial" }).count == corpus.adversarialCount else {
        throw RunnerError.corpus("record counts or categories do not match header")
    }
    for record in corpus.records {
        guard record.inverse.count == 16,
              record.derivatives.count == 64,
              record.covector.count == 4,
              record.reference.count == 8,
              record.referenceScale.count == 8 else {
            throw RunnerError.corpus("invalid dimensions in \(record.label)")
        }
        let words = [record.inverse, record.derivatives, record.covector].flatMap { $0 }
        guard words.allSatisfy({
            $0.status == 0 && $0.hi.isFinite && $0.lo.isFinite
                && (($0.hi == 0.0 && $0.lo == 0.0) || (abs($0.hi) >= 0.5 && abs($0.hi) < 1.0))
                && ($0.lo == 0.0 || abs($0.lo) >= Float.leastNormalMagnitude)
        }), record.reference.allSatisfy(\.isFinite),
            record.referenceScale.allSatisfy({ $0.isFinite && $0 >= 0.0 }) else {
            throw RunnerError.corpus("noncanonical or nonfinite value in \(record.label)")
        }
    }
}

private func validateRHS(
    corpus: RHSCorpus,
    harness: MetalHarness
) throws -> [String: AccuracyMetrics] {
    let inverse = corpus.records.flatMap(\.inverse)
    let derivatives = corpus.records.flatMap(\.derivatives)
    let covector = corpus.records.flatMap(\.covector)
    let (output, status) = try harness.rhs(
        inverse: inverse,
        derivatives: derivatives,
        covector: covector,
        recordCount: corpus.records.count
    )
    var metrics: [String: AccuracyMetrics] = [
        "real-kerr": AccuracyMetrics(),
        "adversarial": AccuracyMetrics(),
    ]
    for recordIndex in corpus.records.indices {
        let record = corpus.records[recordIndex]
        if status[recordIndex] != 0 {
            throw RunnerError.validation(
                "RHS record \(record.label) returned status \(status[recordIndex])"
            )
        }
        for component in 0..<8 {
            metrics[record.category]!.record(
                reference: record.reference[component],
                actual: decodeScaled(output[recordIndex * 8 + component]),
                label: record.label,
                component: component,
                normalizationScale: record.referenceScale[component]
            )
        }
    }
    return metrics
}

private let fifthCoefficients = [
    35.0 / 384.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
]

private let fourthCoefficients = [
    5179.0 / 57600.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
]

private let fifthStages = [0, 2, 3, 4, 5]
private let fourthStages = [0, 2, 3, 4, 5, 6]

private func dopriReference(
    state: [Double],
    stages: [Double],
    step: Double,
    fifthWeights: [Double],
    fourthWeights: [Double]
) -> [Double] {
    var result = [Double](repeating: 0.0, count: 16)
    for component in 0..<8 {
        var fifthSum = 0.0
        for term in fifthStages.indices {
            fifthSum += fifthWeights[term] * stages[fifthStages[term] * 8 + component]
        }
        var fourthSum = 0.0
        for term in fourthStages.indices {
            fourthSum += fourthWeights[term] * stages[fourthStages[term] * 8 + component]
        }
        let fifth = state[component] + step * fifthSum
        let fourth = state[component] + step * fourthSum
        result[component] = fifth
        result[8 + component] = fifth - fourth
    }
    return result
}

private func validateDOPRI(harness: MetalHarness) throws -> [String: AccuracyMetrics] {
    let count = 4096
    var rng = SplitMix64(seed: 0x444f505249353432)
    var state: [ScaledDD] = []
    var stages: [ScaledDD] = []
    var steps: [ScaledDD] = []
    var references: [[Double]] = []
    let coefficients = try (fifthCoefficients + fourthCoefficients).map(encodeScaled)
    let fifthWeights = coefficients[0..<5].map(decodeScaled)
    let fourthWeights = coefficients[5..<11].map(decodeScaled)
    state.reserveCapacity(count * 8)
    stages.reserveCapacity(count * 56)
    steps.reserveCapacity(count)
    references.reserveCapacity(count)
    for _ in 0..<count {
        let stateRecord = try (0..<8).map { _ -> ScaledDD in
            let exponent = Int(rng.next() % 17) - 8
            let sign = rng.next() & 1 == 0 ? 1.0 : -1.0
            return try encodeScaled(sign * scalbn(0.5 + rng.unit() * 0.49, Int32(exponent)))
        }
        let stageRecord = try (0..<56).map { _ -> ScaledDD in
            let exponent = Int(rng.next() % 25) - 12
            let sign = rng.next() & 1 == 0 ? 1.0 : -1.0
            return try encodeScaled(sign * scalbn(0.5 + rng.unit() * 0.49, Int32(exponent)))
        }
        let step = try encodeScaled(scalbn(0.5 + rng.unit() * 0.49, -5))
        state.append(contentsOf: stateRecord)
        stages.append(contentsOf: stageRecord)
        steps.append(step)
        references.append(dopriReference(
            state: stateRecord.map(decodeScaled),
            stages: stageRecord.map(decodeScaled),
            step: decodeScaled(step),
            fifthWeights: fifthWeights,
            fourthWeights: fourthWeights
        ))
    }
    let (output, status) = try harness.dopri(
        state: state,
        stages: stages,
        steps: steps,
        coefficients: coefficients,
        recordCount: count
    )
    var fifthMetrics = AccuracyMetrics()
    var errorMetrics = AccuracyMetrics()
    for record in 0..<count {
        if status[record] != 0 {
            throw RunnerError.validation("DOPRI record \(record) returned status \(status[record])")
        }
        for component in 0..<16 {
            let targetScale: Double
            if component < 8 {
                targetScale = abs(references[record][component])
            } else {
                // The embedded difference may be much smaller than the state.
                // Normalize its absolute discrepancy to the corresponding
                // fifth-order state magnitude as well as reporting direct
                // relative error on the error estimate itself.
                targetScale = abs(references[record][component - 8])
            }
            if component < 8 {
                fifthMetrics.record(
                    reference: references[record][component],
                    actual: decodeScaled(output[record * 16 + component]),
                    label: "dopri-\(record)",
                    component: component,
                    normalizationScale: targetScale
                )
            } else {
                errorMetrics.record(
                reference: references[record][component],
                actual: decodeScaled(output[record * 16 + component]),
                label: "dopri-\(record)",
                    component: component,
                    normalizationScale: targetScale
                )
            }
        }
    }
    return ["fifthOrderState": fifthMetrics, "embeddedError": errorMetrics]
}

private func repeatedRHSInputs(
    corpus: RHSCorpus,
    count: Int
) -> ([ScaledDD], [ScaledDD], [ScaledDD]) {
    var inverse: [ScaledDD] = []
    var derivatives: [ScaledDD] = []
    var covector: [ScaledDD] = []
    inverse.reserveCapacity(count * 16)
    derivatives.reserveCapacity(count * 64)
    covector.reserveCapacity(count * 4)
    for index in 0..<count {
        let record = corpus.records[index % corpus.records.count]
        inverse.append(contentsOf: record.inverse)
        derivatives.append(contentsOf: record.derivatives)
        covector.append(contentsOf: record.covector)
    }
    return (inverse, derivatives, covector)
}

private func sha256(url: URL) throws -> String {
    let data = try Data(contentsOf: url)
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func run() throws {
    guard MemoryLayout<ScaledDD>.stride == 16,
          MemoryLayout<ScaledDD>.alignment == 4 else {
        throw RunnerError.validation("ScaledDD host ABI does not match Metal")
    }
    let configuration = try Configuration.parse()
    let metallibURL = URL(fileURLWithPath: configuration.metallib)
    let corpusURL = URL(fileURLWithPath: configuration.corpus)
    let cpuBenchmarkURL = URL(fileURLWithPath: configuration.cpuBenchmark)
    let corpus = try JSONDecoder().decode(
        RHSCorpus.self,
        from: Data(contentsOf: corpusURL)
    )
    try validateCorpus(corpus)
    let cpuBenchmark = try JSONDecoder().decode(
        CPUBenchmark.self,
        from: Data(contentsOf: cpuBenchmarkURL)
    )
    guard cpuBenchmark.schema == "blackhole-metal-milestone2-cpu-benchmark-v1",
          cpuBenchmark.corpusSHA256 == (try sha256(url: corpusURL)),
          cpuBenchmark.recordCount > 0,
          cpuBenchmark.repeats >= 3,
          cpuBenchmark.recordsPerSecond.isFinite,
          cpuBenchmark.recordsPerSecond > 0.0 else {
        throw RunnerError.corpus("invalid CPU benchmark sidecar")
    }
    let harness = try MetalHarness(metallib: metallibURL)
    print("Metal device: \(harness.device.name)")

    var arithmeticReport: [String: Any] = [:]
    var arithmeticPassed = true
    for operation in ArithmeticOperation.allCases {
        let metrics = try validateArithmetic(operation: operation, harness: harness)
        let passed = metrics.nonfiniteCount == 0
            && metrics.signMismatchCount == 0
            && metrics.zeroMismatchCount == 0
            && metrics.minimumEffectiveBits >= 40.0
        arithmeticPassed = arithmeticPassed && passed
        arithmeticReport[operation.name] = metrics.dictionary()
        print(String(
            format: "%5@ scaled exponent suite count=%5d minBits=%6.2f sign=%d zero=%d %@",
            operation.name as NSString,
            metrics.count,
            metrics.minimumEffectiveBits,
            metrics.signMismatchCount,
            metrics.zeroMismatchCount,
            passed ? "PASS" : "FAIL"
        ))
    }
    let failClosedReport = try validateFailClosedArithmetic(harness: harness)
    let failClosedPassed = failClosedReport["passed"] as! Bool
    arithmeticPassed = arithmeticPassed && failClosedPassed
    print("fail-closed arithmetic statuses \(failClosedPassed ? "PASS" : "FAIL")")

    let rhsMetrics = try validateRHS(corpus: corpus, harness: harness)
    var rhsReport: [String: Any] = [:]
    var rhsPassed = true
    for category in ["real-kerr", "adversarial"] {
        let metrics = rhsMetrics[category]!
        let passed = metrics.nonfiniteCount == 0
            && metrics.signMismatchCount == 0
            && metrics.zeroMismatchCount == 0
            && metrics.minimumConditionNormalizedBits >= 40.0
        rhsPassed = rhsPassed && passed
        rhsReport[category] = metrics.dictionary()
        print(String(
            format: "%11@ RHS components=%6d resultBits=%6.2f normBits=%6.2f sign=%d %@",
            category as NSString,
            metrics.count,
            metrics.minimumEffectiveBits,
            metrics.minimumConditionNormalizedBits,
            metrics.signMismatchCount,
            passed ? "PASS" : "FAIL"
        ))
    }

    let dopriMetrics = try validateDOPRI(harness: harness)
    let dopriStateMetrics = dopriMetrics["fifthOrderState"]!
    let dopriErrorMetrics = dopriMetrics["embeddedError"]!
    let dopriPassed = dopriStateMetrics.nonfiniteCount == 0
        && dopriStateMetrics.signMismatchCount == 0
        && dopriStateMetrics.minimumEffectiveBits >= 38.0
        && dopriErrorMetrics.nonfiniteCount == 0
        && dopriErrorMetrics.signMismatchCount == 0
        && dopriErrorMetrics.minimumConditionNormalizedBits >= 38.0
    print(String(
        format: "DOPRI state components=%6d minBits=%6.2f errorNormBits=%6.2f sign=%d %@",
        dopriStateMetrics.count,
        dopriStateMetrics.minimumEffectiveBits,
        dopriErrorMetrics.minimumConditionNormalizedBits,
        dopriStateMetrics.signMismatchCount + dopriErrorMetrics.signMismatchCount,
        dopriPassed ? "PASS" : "FAIL"
    ))

    let benchmarkInputs = repeatedRHSInputs(
        corpus: corpus,
        count: configuration.benchmarkCount
    )
    let benchmark = try harness.benchmarkRHS(
        inverse: benchmarkInputs.0,
        derivatives: benchmarkInputs.1,
        covector: benchmarkInputs.2,
        recordCount: configuration.benchmarkCount,
        iterations: configuration.benchmarkIterations
    )
    print(String(
        format: "RHS throughput resident=%9.2f records/s wall=%9.2f records/s CPU=%9.2f speedup=%7.2fx",
        benchmark["gpuRecordsPerSecond"] as! Double,
        benchmark["wallRecordsPerSecond"] as! Double,
        cpuBenchmark.recordsPerSecond,
        (benchmark["wallRecordsPerSecond"] as! Double)
            / cpuBenchmark.recordsPerSecond
    ))

    // Milestone 2's required gate is the repaired arithmetic representation
    // plus the precomputed-metric RHS.  The DOPRI combine is an intentionally
    // fail-visible diagnostic: it was implemented because it is the next
    // boundary, but it is not promoted merely because RHS passed.
    let validationPassed = arithmeticPassed && rhsPassed
    let report: [String: Any] = [
        "schema": "blackhole-metal-scaled-dd-milestone2-v1",
        "productionQualified": false,
        "validationPassed": validationPassed,
        "dopriCombineDiagnosticPassed": dopriPassed,
        "qualificationScope": (
            "scaled float-float arithmetic and precomputed MetricSample Hamiltonian RHS only; "
            + "DOPRI5(4) result/error combination is a non-gating diagnostic"
        ),
        "categoricalSuitability": [
            "componentSignDrift": rhsMetrics.values.reduce(0) { $0 + $1.signMismatchCount },
            "wholeRayFateEvaluated": false,
            "surfaceTopologyEvaluated": false,
            "conclusion": "unresolved until complete adaptive whole rays and surface-event topology are differentially replayed",
        ],
        "subnormalLowWordResolution": (
            "explicit shared binary exponent; nonzero mantissas normalized so low words remain normal FP32"
        ),
        "device": harness.device.name,
        "recommendedMaxWorkingSetSize": harness.device.recommendedMaxWorkingSetSize,
        "toolchain": "com.apple.dt.toolchain.Metal.32023.883",
        "compileContract": [
            "-std=metal3.2",
            "-fmetal-math-mode=safe",
            "-fmetal-math-fp32-functions=precise",
            "-ffp-contract=on (explicit two_product FMA only)",
        ],
        "metallibSHA256": try sha256(url: metallibURL),
        "corpusSHA256": try sha256(url: corpusURL),
        "cpuBenchmarkSHA256": try sha256(url: cpuBenchmarkURL),
        "corpusSchema": corpus.schema,
        "corpusEncoding": corpus.encoding,
        "cpuReference": corpus.reference,
        "corpusCounts": [
            "realKerr": corpus.realKerrCount,
            "adversarial": corpus.adversarialCount,
        ],
        "arithmetic": arithmeticReport,
        "failClosedArithmetic": failClosedReport,
        "hamiltonianRHS": rhsReport,
        "dopriCombine": [
            "fifthOrderState": dopriStateMetrics.dictionary(),
            "embeddedError": dopriErrorMetrics.dictionary(),
        ],
        "throughput": [
            "gpu": benchmark,
            "cpuBinary64": cpuBenchmark.dictionary(),
            "wallSpeedupOverCpuBinary64": (
                (benchmark["wallRecordsPerSecond"] as! Double)
                    / cpuBenchmark.recordsPerSecond
            ),
        ],
        "knownLimits": [
            "Kerr metric construction remains on CPU and was supplied precomputed",
            "no adaptive step acceptance/rejection loop is implemented",
            "no surface probe, root localization, path recording, or fate policy is implemented",
            "RHS component sign parity is not evidence of whole-ray categorical parity",
            "scaled float-float retains about 48 significant bits, fewer than binary64's 53",
            "the DOPRI combine diagnostic currently misses its precision gate and is not suitable for adaptive-step decisions",
        ],
    ]
    let reportData = try JSONSerialization.data(
        withJSONObject: report,
        options: [.prettyPrinted, .sortedKeys]
    )
    try reportData.write(
        to: URL(fileURLWithPath: configuration.report),
        options: .atomic
    )
    print("report: \(configuration.report)")
    guard validationPassed else {
        throw RunnerError.validation("one or more milestone-2 gates failed")
    }
}

do {
    try run()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
