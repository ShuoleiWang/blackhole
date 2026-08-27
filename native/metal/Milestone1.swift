import CryptoKit
import Foundation
import Metal

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

private struct Configuration {
    var metallib = ""
    var report = ""
    var benchmarkCount = 1 << 20
    var benchmarkIterations = 9

    static func parse() throws -> Configuration {
        var configuration = Configuration()
        var arguments = Array(CommandLine.arguments.dropFirst())
        while !arguments.isEmpty {
            let flag = arguments.removeFirst()
            guard !arguments.isEmpty else {
                throw RunnerError.configuration("missing value for \(flag)")
            }
            let value = arguments.removeFirst()
            switch flag {
            case "--metallib": configuration.metallib = value
            case "--report": configuration.report = value
            case "--benchmark-count":
                guard let parsed = Int(value), parsed > 0 else {
                    throw RunnerError.configuration("invalid benchmark count: \(value)")
                }
                configuration.benchmarkCount = parsed
            case "--benchmark-iterations":
                guard let parsed = Int(value), parsed >= 3 else {
                    throw RunnerError.configuration("benchmark iterations must be >= 3")
                }
                configuration.benchmarkIterations = parsed
            default: throw RunnerError.configuration("unknown argument: \(flag)")
            }
        }
        guard !configuration.metallib.isEmpty, !configuration.report.isEmpty else {
            throw RunnerError.configuration("--metallib and --report are required")
        }
        return configuration
    }
}

private enum RunnerError: Error, CustomStringConvertible {
    case configuration(String)
    case metal(String)
    case validation(String)

    var description: String {
        switch self {
        case .configuration(let message): return "configuration error: \(message)"
        case .metal(let message): return "Metal error: \(message)"
        case .validation(let message): return "validation error: \(message)"
        }
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

private struct ValidationVector {
    let a: Double
    let b: Double
    let label: String
}

private struct Metrics {
    var count = 0
    var exactCount = 0
    var maximumULP = 0.0
    var minimumEffectiveBits = 53.0
    var maximumRelativeError = 0.0
    var worstLabel = ""
    var worstReference = 0.0
    var worstActual = 0.0

    mutating func record(reference: Double, actual: Double, label: String) throws {
        guard reference.isFinite, actual.isFinite else {
            throw RunnerError.validation("non-finite result for \(label): expected \(reference), got \(actual)")
        }
        count += 1
        let error = abs(actual - reference)
        if error == 0.0 { exactCount += 1 }
        let ulp = reference == 0.0
            ? Double.leastNonzeroMagnitude
            : abs(reference.nextUp - reference)
        let ulpError = error / ulp
        let scale = max(abs(reference), Double.leastNormalMagnitude)
        let relativeError = error / scale
        let effectiveBits = error == 0.0 ? 53.0 : max(0.0, -log2(relativeError))
        minimumEffectiveBits = min(minimumEffectiveBits, effectiveBits)
        maximumRelativeError = max(maximumRelativeError, relativeError)
        if ulpError > maximumULP {
            maximumULP = ulpError
            worstLabel = label
            worstReference = reference
            worstActual = actual
        }
    }

    func dictionary() -> [String: Any] {
        [
            "count": count,
            "exactCount": exactCount,
            "maximumBinary64ULP": maximumULP,
            "minimumEffectiveBits": minimumEffectiveBits,
            "maximumRelativeError": maximumRelativeError,
            "worstLabel": worstLabel,
            "worstReference": worstReference,
            "worstActual": worstActual,
        ]
    }
}

private final class MetalHarness {
    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    init(metallib: URL) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw RunnerError.metal("no default Metal device")
        }
        self.device = device
        let library = try device.makeLibrary(URL: metallib)
        guard let function = library.makeFunction(name: "dd_arithmetic") else {
            throw RunnerError.metal("dd_arithmetic is missing from metallib")
        }
        pipeline = try device.makeComputePipelineState(function: function)
        guard let queue = device.makeCommandQueue() else {
            throw RunnerError.metal("could not create command queue")
        }
        self.queue = queue
    }

    private func makeBuffer(_ values: [SIMD2<Float>]) throws -> MTLBuffer {
        let byteCount = values.count * MemoryLayout<SIMD2<Float>>.stride
        guard let buffer = values.withUnsafeBytes({ raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: byteCount, options: .storageModeShared)
        }) else {
            throw RunnerError.metal("could not allocate \(byteCount)-byte shared buffer")
        }
        return buffer
    }

    private func encode(
        operation: ArithmeticOperation,
        count: Int,
        a: MTLBuffer,
        b: MTLBuffer,
        output: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw RunnerError.metal("could not create compute encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(b, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var operationRaw = operation.rawValue
        var countRaw = UInt32(count)
        encoder.setBytes(&operationRaw, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&countRaw, length: MemoryLayout<UInt32>.size, index: 4)
        let width = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    func evaluate(
        operation: ArithmeticOperation,
        a: [SIMD2<Float>],
        b: [SIMD2<Float>]
    ) throws -> [SIMD2<Float>] {
        precondition(a.count == b.count)
        let aBuffer = try makeBuffer(a)
        let bBuffer = try makeBuffer(b)
        let byteCount = a.count * MemoryLayout<SIMD2<Float>>.stride
        guard let output = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer() else {
            throw RunnerError.metal("could not allocate validation resources")
        }
        try encode(operation: operation, count: a.count, a: aBuffer, b: bBuffer, output: output, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        let pointer = output.contents().bindMemory(to: SIMD2<Float>.self, capacity: a.count)
        return Array(UnsafeBufferPointer(start: pointer, count: a.count))
    }

    func benchmark(
        operation: ArithmeticOperation,
        a: [SIMD2<Float>],
        b: [SIMD2<Float>],
        iterations: Int
    ) throws -> [String: Any] {
        let aBuffer = try makeBuffer(a)
        let bBuffer = try makeBuffer(b)
        let byteCount = a.count * MemoryLayout<SIMD2<Float>>.stride
        guard let output = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw RunnerError.metal("could not allocate benchmark output")
        }

        var samples: [Double] = []
        for iteration in 0..<(iterations + 1) {
            guard let commandBuffer = queue.makeCommandBuffer() else {
                throw RunnerError.metal("could not create benchmark command buffer")
            }
            let wallStart = ProcessInfo.processInfo.systemUptime
            try encode(operation: operation, count: a.count, a: aBuffer, b: bBuffer, output: output, commandBuffer: commandBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let wallEnd = ProcessInfo.processInfo.systemUptime
            if let error = commandBuffer.error { throw error }
            if iteration > 0 {
                let gpuDuration = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
                samples.append(gpuDuration > 0 ? gpuDuration : wallEnd - wallStart)
            }
        }
        samples.sort()
        let median = samples[samples.count / 2]
        return [
            "elementCount": a.count,
            "iterations": iterations,
            "medianSeconds": median,
            "millionOperationsPerSecond": Double(a.count) / median / 1_000_000.0,
            "timingSource": "MTLCommandBuffer.gpuStartTime/gpuEndTime (wall fallback)",
        ]
    }
}

private func encodeDD(_ value: Double) throws -> SIMD2<Float> {
    let hi = Float(value)
    guard hi.isFinite else {
        throw RunnerError.validation("value is outside finite Float range: \(value)")
    }
    let lo = Float(value - Double(hi))
    return SIMD2<Float>(hi, lo)
}

private func decodeDD(_ value: SIMD2<Float>) -> Double {
    Double(value.x) + Double(value.y)
}

private func quantized(_ value: Double) throws -> Double {
    decodeDD(try encodeDD(value))
}

private func randomFinite(
    rng: inout SplitMix64,
    exponentRange: ClosedRange<Int>,
    positive: Bool = false
) -> Double {
    let exponentWidth = UInt64(exponentRange.upperBound - exponentRange.lowerBound + 1)
    let exponent = exponentRange.lowerBound + Int(rng.next() % exponentWidth)
    let mantissa = 0.5 + rng.unit() * 1.5
    let sign = positive || (rng.next() & 1 == 0) ? 1.0 : -1.0
    return sign * scalbn(mantissa, Int32(exponent))
}

private func validationVectors(for operation: ArithmeticOperation) -> [ValidationVector] {
    var vectors: [ValidationVector] = []
    let perturbations = [
        0.0,
        0x1.0p-12,
        -0x1.0p-12,
        0x1.0p-23,
        -0x1.0p-23,
        0x1.0p-25,
        -0x1.0p-25,
        0x1.0p-40,
        -0x1.0p-40,
    ]

    switch operation {
    case .add, .subtract:
        for exponent in stride(from: -40, through: 40, by: 4) {
            let scale = scalbn(1.0, Int32(exponent))
            for (index, perturbation) in perturbations.enumerated() {
                let a = scale * (1.0 + perturbation)
                let cancellation = scale * (1.0 - perturbation * 0.5 + 0x1.0p-44)
                let b = operation == .add ? -cancellation : cancellation
                vectors.append(ValidationVector(a: a, b: b, label: "cancel-e\(exponent)-p\(index)"))
                vectors.append(ValidationVector(a: a, b: scale * (0.5 + perturbation), label: "same-e\(exponent)-p\(index)"))
            }
        }
    case .multiply, .divide:
        for exponent in stride(from: -40, through: 40, by: 4) {
            for (index, perturbation) in perturbations.enumerated() {
                let a = scalbn(1.25 + perturbation, Int32(exponent / 2))
                let b = scalbn(0.75 - perturbation * 0.5, Int32(-exponent / 2))
                vectors.append(ValidationVector(a: a, b: b, label: "balanced-e\(exponent)-p\(index)"))
                vectors.append(ValidationVector(a: a, b: 1.0 + perturbation, label: "unit-e\(exponent)-p\(index)"))
            }
        }
    case .squareRoot:
        for exponent in stride(from: -40, through: 40, by: 2) {
            for (index, perturbation) in perturbations.enumerated() {
                let a = scalbn(1.0 + perturbation, Int32(exponent))
                vectors.append(ValidationVector(a: a, b: 1.0, label: "sqrt-e\(exponent)-p\(index)"))
            }
        }
    }

    var rng = SplitMix64(seed: 0xddf10a7be5c0ffee ^ UInt64(operation.rawValue))
    while vectors.count < 4096 {
        switch operation {
        case .add, .subtract:
            let a = randomFinite(rng: &rng, exponentRange: -30...30)
            let nearCancellation = rng.next() & 3 == 0
            let b = nearCancellation
                ? a * (1.0 + (rng.unit() - 0.5) * 0x1.0p-38)
                : randomFinite(rng: &rng, exponentRange: -30...30)
            vectors.append(ValidationVector(a: a, b: operation == .add && nearCancellation ? -b : b, label: "random-\(vectors.count)"))
        case .multiply, .divide:
            let a = randomFinite(rng: &rng, exponentRange: -24...24)
            let b = randomFinite(rng: &rng, exponentRange: -24...24)
            vectors.append(ValidationVector(a: a, b: b, label: "random-\(vectors.count)"))
        case .squareRoot:
            let a = randomFinite(rng: &rng, exponentRange: -40...40, positive: true)
            vectors.append(ValidationVector(a: a, b: 1.0, label: "random-\(vectors.count)"))
        }
    }
    return vectors
}

// Metal's FP32 execution may flush subnormal lanes.  Keep an explicit
// diagnostic corpus around the edge where the low word of a float-float
// expansion becomes subnormal.  This suite is intentionally not hidden behind
// the supported-domain gate: its failures quantify a real limitation that a
// future whole-ray design must address with scaling or a different encoding.
private func exponentBoundaryVectors(for operation: ArithmeticOperation) -> [ValidationVector] {
    var vectors: [ValidationVector] = []
    let tinyPerturbations = [0x1.0p-23, 0x1.0p-30, 0x1.0p-40, 0x1.0p-44]

    for exponent in stride(from: -104, through: -88, by: 2) {
        let scale = scalbn(1.0, Int32(exponent))
        for (index, perturbation) in tinyPerturbations.enumerated() {
            switch operation {
            case .add:
                vectors.append(ValidationVector(
                    a: scale * (1.0 + 0x1.0p-23),
                    b: -scale * (1.0 - perturbation),
                    label: "subnormal-low-add-e\(exponent)-p\(index)"
                ))
            case .subtract:
                vectors.append(ValidationVector(
                    a: scale * (1.0 + 0x1.0p-23),
                    b: scale * (1.0 - perturbation),
                    label: "subnormal-low-sub-e\(exponent)-p\(index)"
                ))
            case .multiply:
                vectors.append(ValidationVector(
                    a: scalbn(1.25 + perturbation, Int32(exponent / 2)),
                    b: scalbn(0.75 - perturbation, Int32(exponent - exponent / 2)),
                    label: "subnormal-output-mul-e\(exponent)-p\(index)"
                ))
            case .divide:
                vectors.append(ValidationVector(
                    a: scalbn(1.25 + perturbation, Int32(exponent / 2)),
                    b: scalbn(0.75 - perturbation, Int32(-exponent / 2)),
                    label: "subnormal-output-div-e\(exponent)-p\(index)"
                ))
            case .squareRoot:
                vectors.append(ValidationVector(
                    a: scale * (1.0 + perturbation),
                    b: 1.0,
                    label: "subnormal-input-low-sqrt-e\(exponent)-p\(index)"
                ))
            }
        }
    }

    var rng = SplitMix64(seed: 0x66747a2d626f756e ^ UInt64(operation.rawValue))
    while vectors.count < 1024 {
        switch operation {
        case .add, .subtract:
            let a = randomFinite(rng: &rng, exponentRange: -104...104)
            let nearCancellation = rng.next() & 1 == 0
            let b = nearCancellation
                ? a * (1.0 + (rng.unit() - 0.5) * 0x1.0p-40)
                : randomFinite(rng: &rng, exponentRange: -104...104)
            vectors.append(ValidationVector(
                a: a,
                b: operation == .add && nearCancellation ? -b : b,
                label: "boundary-random-\(vectors.count)"
            ))
        case .multiply, .divide:
            vectors.append(ValidationVector(
                a: randomFinite(rng: &rng, exponentRange: -52...52),
                b: randomFinite(rng: &rng, exponentRange: -52...52),
                label: "boundary-random-\(vectors.count)"
            ))
        case .squareRoot:
            vectors.append(ValidationVector(
                a: randomFinite(rng: &rng, exponentRange: -104...104, positive: true),
                b: 1.0,
                label: "boundary-random-\(vectors.count)"
            ))
        }
    }
    return vectors
}

private func reference(
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

private func validate(
    operation: ArithmeticOperation,
    vectors: [ValidationVector],
    harness: MetalHarness
) throws -> Metrics {
    let a = try vectors.map { try encodeDD($0.a) }
    let b = try vectors.map { try encodeDD($0.b) }
    let output = try harness.evaluate(operation: operation, a: a, b: b)
    var metrics = Metrics()
    for index in vectors.indices {
        let qa = decodeDD(a[index])
        let qb = decodeDD(b[index])
        try metrics.record(
            reference: reference(operation: operation, a: qa, b: qb),
            actual: decodeDD(output[index]),
            label: vectors[index].label
        )
    }
    return metrics
}

private func benchmarkInputs(count: Int) throws -> ([SIMD2<Float>], [SIMD2<Float>]) {
    var a: [SIMD2<Float>] = []
    var b: [SIMD2<Float>] = []
    a.reserveCapacity(count)
    b.reserveCapacity(count)
    var rng = SplitMix64(seed: 0x6d6574616c2d6464)
    for _ in 0..<count {
        a.append(try encodeDD(randomFinite(rng: &rng, exponentRange: -16...16, positive: true)))
        b.append(try encodeDD(randomFinite(rng: &rng, exponentRange: -16...16, positive: true)))
    }
    return (a, b)
}

private func sha256(url: URL) throws -> String {
    let data = try Data(contentsOf: url)
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func run() throws {
    let configuration = try Configuration.parse()
    let metallibURL = URL(fileURLWithPath: configuration.metallib)
    let harness = try MetalHarness(metallib: metallibURL)

    var validationReport: [String: Any] = [:]
    var validationPassed = true
    print("Metal device: \(harness.device.name)")
    for operation in ArithmeticOperation.allCases {
        let metrics = try validate(
            operation: operation,
            vectors: validationVectors(for: operation),
            harness: harness
        )
        let boundaryMetrics = try validate(
            operation: operation,
            vectors: exponentBoundaryVectors(for: operation),
            harness: harness
        )
        validationReport[operation.name] = [
            "supportedDomain": metrics.dictionary(),
            "exponentBoundaryDiagnostic": boundaryMetrics.dictionary(),
        ]
        // Milestone gate, not a production qualification gate.  Float-float is
        // expected to retain roughly 48 significant bits; 40 leaves headroom
        // for deliberately adversarial cancellation while still detecting a
        // silently degraded single-float implementation.
        let passed = metrics.minimumEffectiveBits >= 40.0
            && metrics.maximumULP <= 8192.0
        validationPassed = validationPassed && passed
        print(String(
            format: "%5@  count=%4d exact=%4d maxULP=%9.2f minBits=%6.2f %@",
            operation.name as NSString,
            metrics.count,
            metrics.exactCount,
            metrics.maximumULP,
            metrics.minimumEffectiveBits,
            passed ? "PASS" : "FAIL"
        ))
        print(String(
            format: "      boundary diagnostic: maxULP=%12.2f minBits=%6.2f (not a pass gate)",
            boundaryMetrics.maximumULP,
            boundaryMetrics.minimumEffectiveBits
        ))
    }

    let benchmarkInput = try benchmarkInputs(count: configuration.benchmarkCount)
    var benchmarkReport: [String: Any] = [:]
    for operation in ArithmeticOperation.allCases {
        let result = try harness.benchmark(
            operation: operation,
            a: benchmarkInput.0,
            b: benchmarkInput.1,
            iterations: configuration.benchmarkIterations
        )
        benchmarkReport[operation.name] = result
        let rate = result["millionOperationsPerSecond"] as! Double
        print(String(format: "%5@  throughput=%9.2f million elements/s", operation.name as NSString, rate))
    }

    let report: [String: Any] = [
        "schema": "blackhole-metal-double-double-milestone1-v1",
        "productionQualified": false,
        "qualificationScope": "arithmetic prototype only; no ray, topology, or scientific-policy qualification",
        "supportedValidationDomain": "input exponents approximately [-40,+40], randomized operation exponents constrained to avoid FP32 subnormal low words",
        "knownLimitation": "the exponent-boundary diagnostic exposes Metal FP32 subnormal flushing; a production design requires explicit scaling or another encoding",
        "device": harness.device.name,
        "recommendedMaxWorkingSetSize": harness.device.recommendedMaxWorkingSetSize,
        "metallibSHA256": try sha256(url: metallibURL),
        "validationPassed": validationPassed,
        "validation": validationReport,
        "benchmark": benchmarkReport,
    ]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: configuration.report), options: .atomic)
    print("report: \(configuration.report)")
    guard validationPassed else {
        throw RunnerError.validation("one or more milestone arithmetic gates failed")
    }
}

do {
    try run()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
