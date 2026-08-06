/*
 * Readback harness for the production strong-field WebGPU tracer.
 *
 * The compute entry point below is appended to the production WGSL module and
 * calls the same traceStrongField() function used by fsMain().  This is
 * intentional: a separately ported probe would be an oracle for the port, not
 * evidence about the shader that actually renders the image.
 */

import {
  STRONG_FIELD_UNIFORM_FLOATS,
  strongFieldBinaryTraceFragmentWGSL,
  writeStrongFieldUniformTail,
} from "./strong-field-shaders.js";

export const STRONG_FIELD_GPU_PROBE_SCHEMA =
  "blackhole.strong-field-gpu-probe/v1";
export const STRONG_FIELD_GPU_PROBE_WORKGROUP_SIZE = 64;
export const STRONG_FIELD_GPU_PROBE_OUTPUT_BYTES = 48;

export const STRONG_FIELD_GPU_PROBE_TOLERANCES = Object.freeze({
  escapeDirectionAbsolute: 2e-5,
  frequencyShiftAbsolute: 1e-5,
  frequencyShiftRelative: 1e-4,
  lookbackAbsoluteM: 1e-2,
  lookbackRelative: 1e-4,
  residualAbsolute: 1e-6,
  residualRegressionRelative: 2e-2,
  iterationRegression: 0,
  minimumHorizonDistanceAbsoluteM: 1e-3,
});

export const strongFieldGpuProbeWGSL = /* wgsl */ `${strongFieldBinaryTraceFragmentWGSL}

struct StrongFieldProbeInput {
  // xy: screen coordinates after aspect correction, z: tan(fov / 2).
  screenTanHalfFov: vec4<f32>,
};

struct StrongFieldProbeOutput {
  outcome: u32,
  terminationReason: f32,
  frequencyShift: f32,
  lookback: f32,
  escapeDirectionAndResidual: vec4<f32>,
  iterationsMinimumHorizonPadding: vec4<f32>,
};

@group(1) @binding(0)
var<storage, read> strongFieldProbeInputs: array<StrongFieldProbeInput>;

@group(1) @binding(1)
var<storage, read_write> strongFieldProbeOutputs: array<StrongFieldProbeOutput>;

@compute @workgroup_size(${STRONG_FIELD_GPU_PROBE_WORKGROUP_SIZE})
fn strongFieldProbeMain(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= arrayLength(&strongFieldProbeInputs)) {
    return;
  }
  let probe = strongFieldProbeInputs[index].screenTanHalfFov;
  let result = traceStrongField(probe.xy, probe.z);
  strongFieldProbeOutputs[index].outcome = result.outcome;
  strongFieldProbeOutputs[index].terminationReason = result.terminationReason;
  strongFieldProbeOutputs[index].frequencyShift = result.frequencyShift;
  strongFieldProbeOutputs[index].lookback = result.lookback;
  strongFieldProbeOutputs[index].escapeDirectionAndResidual = vec4<f32>(
    result.escapeDirection,
    result.hamiltonianResidual
  );
  strongFieldProbeOutputs[index].iterationsMinimumHorizonPadding = vec4<f32>(
    result.iterations,
    result.minimumHorizonDistance,
    0.0,
    0.0
  );
}
`;

function finiteNumber(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new TypeError(`${label} must be finite`);
  }
  return number;
}

function finiteVector(value, length, label) {
  if (!value || typeof value.length !== "number" || value.length !== length) {
    throw new TypeError(`${label} must contain exactly ${length} numbers`);
  }
  return Array.from(value, (entry, index) => (
    finiteNumber(entry, `${label}[${index}]`)
  ));
}

function nowMs() {
  return globalThis.performance?.now?.() ?? Date.now();
}

function bytesOf(view) {
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

function fnv1a32(view) {
  let hash = 0x811c9dc5;
  for (const byte of bytesOf(view)) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/**
 * Pack the exact 96-float production uniform ABI without constructing a
 * renderer or loading the celestial texture.  Only trace-relevant fields are
 * required; display controls retain inert defaults.
 */
export function createStrongFieldProbeUniforms(frame, options = {}) {
  if (!frame || typeof frame !== "object") {
    throw new TypeError("Strong-field probe frame is required");
  }
  const resolution = finiteVector(
    options.resolution || [1, 1],
    2,
    "resolution",
  );
  const cameraPos = finiteVector(frame.cameraPos, 3, "cameraPos");
  const forward = finiteVector(frame.forward, 3, "forward");
  const right = finiteVector(frame.right, 3, "right");
  const up = finiteVector(frame.up, 3, "up");
  const observerVelocity = finiteVector(
    frame.observerVelocity || [0, 0, 0],
    3,
    "observerVelocity",
  );
  const data = new Float32Array(STRONG_FIELD_UNIFORM_FLOATS);
  data[0] = resolution[0];
  data[1] = resolution[1];
  data[2] = finiteNumber(frame.time ?? 0, "time");
  data[3] = finiteNumber(frame.massSolar ?? 1, "massSolar");
  data[4] = finiteNumber(frame.accretion ?? 0, "accretion");
  data[5] = finiteNumber(frame.exposure ?? 1, "exposure");
  data[6] = finiteNumber(frame.mode ?? 0, "mode");
  data[7] = finiteNumber(frame.steps, "steps");
  data.set(cameraPos, 8);
  data[11] = finiteNumber(
    frame.cameraRadius ?? Math.hypot(...cameraPos),
    "cameraRadius",
  );
  data.set(forward, 12);
  data[15] = finiteNumber(frame.fov ?? 0.7, "fov");
  data.set(right, 16);
  data[19] = finiteNumber(frame.skyRotation ?? 0, "skyRotation");
  data.set(up, 20);
  data[23] = finiteNumber(frame.diskOuterRadius ?? 18, "diskOuterRadius");
  data[24] = finiteNumber(frame.renderScale ?? 1, "renderScale");
  data[25] = finiteNumber(frame.bloom ?? 0, "bloom");
  data[26] = finiteNumber(frame.motion ?? 0, "motion");
  data[27] = finiteNumber(frame.frame ?? 0, "frame");
  data.set(observerVelocity, 28);
  data[31] = finiteNumber(frame.observerBeta ?? 0, "observerBeta");
  data[32] = finiteNumber(options.outputHDR ?? 0, "outputHDR");
  data[33] = finiteNumber(options.displayP3 ?? 0, "displayP3");
  data[34] = finiteNumber(options.hdrPeak ?? 1, "hdrPeak");
  data[35] = finiteNumber(options.skyRadianceScale ?? 1, "skyRadianceScale");
  writeStrongFieldUniformTail(data.subarray(36), frame);
  return data;
}

/** Create deterministic pixel-centre probes spanning the complete viewport. */
export function createStrongFieldProbeGrid({
  columns = 17,
  rows = 9,
  aspect = 16 / 9,
  fov = 0.7,
} = {}) {
  if (!Number.isInteger(columns) || columns < 1) {
    throw new RangeError("probe grid columns must be a positive integer");
  }
  if (!Number.isInteger(rows) || rows < 1) {
    throw new RangeError("probe grid rows must be a positive integer");
  }
  const finiteAspect = finiteNumber(aspect, "probe grid aspect");
  const finiteFov = finiteNumber(fov, "probe grid fov");
  if (finiteAspect <= 0 || finiteFov <= 0 || finiteFov >= Math.PI) {
    throw new RangeError("probe grid aspect and fov are out of range");
  }
  const tanHalfFov = Math.tan(finiteFov * 0.5);
  return Object.freeze(Array.from({ length: rows * columns }, (_, index) => {
    const row = Math.floor(index / columns);
    const column = index % columns;
    const u = (column + 0.5) / columns;
    const v = (row + 0.5) / rows;
    return Object.freeze({
      id: `r${row}c${column}`,
      screen: Object.freeze([(u * 2 - 1) * finiteAspect, 1 - v * 2]),
      tanHalfFov,
    });
  }));
}

export function encodeStrongFieldProbeInputs(probes) {
  if (!Array.isArray(probes) || probes.length < 1) {
    throw new TypeError("At least one strong-field GPU probe is required");
  }
  const encoded = new Float32Array(probes.length * 4);
  probes.forEach((probe, index) => {
    const screen = finiteVector(probe?.screen, 2, `probes[${index}].screen`);
    const tanHalfFov = finiteNumber(
      probe?.tanHalfFov,
      `probes[${index}].tanHalfFov`,
    );
    if (tanHalfFov <= 0) {
      throw new RangeError(`probes[${index}].tanHalfFov must be positive`);
    }
    encoded.set([screen[0], screen[1], tanHalfFov, 0], index * 4);
  });
  return encoded;
}

export function decodeStrongFieldProbeOutputs(buffer, probes = []) {
  const bytes = buffer instanceof ArrayBuffer
    ? buffer
    : buffer?.buffer?.slice(
      buffer.byteOffset,
      buffer.byteOffset + buffer.byteLength,
    );
  if (!(bytes instanceof ArrayBuffer)) {
    throw new TypeError("Probe readback must be an ArrayBuffer or typed array");
  }
  if (bytes.byteLength % STRONG_FIELD_GPU_PROBE_OUTPUT_BYTES !== 0) {
    throw new RangeError("Probe readback byte length violates the output ABI");
  }
  const view = new DataView(bytes);
  const count = bytes.byteLength / STRONG_FIELD_GPU_PROBE_OUTPUT_BYTES;
  return Object.freeze(Array.from({ length: count }, (_, index) => {
    const offset = index * STRONG_FIELD_GPU_PROBE_OUTPUT_BYTES;
    return Object.freeze({
      index,
      id: probes[index]?.id ?? String(index),
      outcome: view.getUint32(offset, true),
      terminationReason: view.getFloat32(offset + 4, true),
      frequencyShift: view.getFloat32(offset + 8, true),
      lookback: view.getFloat32(offset + 12, true),
      escapeDirection: Object.freeze([
        view.getFloat32(offset + 16, true),
        view.getFloat32(offset + 20, true),
        view.getFloat32(offset + 24, true),
      ]),
      maximumNullResidual: view.getFloat32(offset + 28, true),
      iterations: view.getFloat32(offset + 32, true),
      minimumHorizonDistance: view.getFloat32(offset + 36, true),
    });
  }));
}

function gpuFlags() {
  const usage = globalThis.GPUBufferUsage || {
    MAP_READ: 0x0001,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
  };
  return {
    mapRead: usage.MAP_READ,
    copySrc: usage.COPY_SRC,
    copyDst: usage.COPY_DST,
    uniform: usage.UNIFORM,
    storage: usage.STORAGE,
    readMode: globalThis.GPUMapMode?.READ ?? 0x0001,
  };
}

/**
 * Execute the production trace function as a compute corpus and read every
 * numerical channel back to JavaScript. queueWallTimeMs excludes compilation,
 * buffer creation and mapping; drainQueue=true isolates it from older work.
 */
export async function runStrongFieldGpuProbe(device, {
  uniforms,
  probes,
  label = "Strong-field production tracer probe",
  drainQueue = true,
  pipelineConstants = undefined,
} = {}) {
  if (!device?.queue || typeof device.createBuffer !== "function") {
    throw new TypeError("A live WebGPU device is required");
  }
  if (!(uniforms instanceof Float32Array)) {
    throw new TypeError("Probe uniforms must be a Float32Array");
  }
  if (uniforms.length !== STRONG_FIELD_UNIFORM_FLOATS) {
    throw new RangeError(
      `Probe uniforms must contain ${STRONG_FIELD_UNIFORM_FLOATS} floats`,
    );
  }
  if (![...uniforms].every(Number.isFinite)) {
    throw new TypeError("Probe uniforms must be finite");
  }
  if (
    pipelineConstants != null
    && (
      typeof pipelineConstants !== "object"
      || Array.isArray(pipelineConstants)
      || Object.values(pipelineConstants).some(
        (value) => !Number.isFinite(Number(value)),
      )
    )
  ) {
    throw new TypeError("Probe pipeline constants must be finite numbers");
  }
  const inputs = encodeStrongFieldProbeInputs(probes);
  const flags = gpuFlags();
  const buffers = [];
  let mapped = false;
  const createBuffer = (descriptor) => {
    const buffer = device.createBuffer(descriptor);
    buffers.push(buffer);
    return buffer;
  };
  const uniformBuffer = createBuffer({
    label: `${label} uniforms`,
    size: uniforms.byteLength,
    usage: flags.uniform | flags.copyDst,
  });
  const inputBuffer = createBuffer({
    label: `${label} inputs`,
    size: inputs.byteLength,
    usage: flags.storage | flags.copyDst,
  });
  const outputByteLength = probes.length * STRONG_FIELD_GPU_PROBE_OUTPUT_BYTES;
  const outputBuffer = createBuffer({
    label: `${label} outputs`,
    size: outputByteLength,
    usage: flags.storage | flags.copySrc,
  });
  const readbackBuffer = createBuffer({
    label: `${label} readback`,
    size: outputByteLength,
    usage: flags.mapRead | flags.copyDst,
  });

  try {
    const compilationStartedAtMs = nowMs();
    const shaderModule = device.createShaderModule({
      label: `${label} shader`,
      code: strongFieldGpuProbeWGSL,
    });
    if (typeof shaderModule.getCompilationInfo === "function") {
      const compilation = await shaderModule.getCompilationInfo();
      const errors = compilation.messages.filter((message) => (
        message.type === "error"
      ));
      if (errors.length) {
        throw new Error(errors.map((error) => error.message).join("\n"));
      }
    }
    const descriptor = {
      label: `${label} pipeline`,
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: "strongFieldProbeMain",
        ...(pipelineConstants ? { constants: pipelineConstants } : {}),
      },
    };
    const pipeline = typeof device.createComputePipelineAsync === "function"
      ? await device.createComputePipelineAsync(descriptor)
      : device.createComputePipeline(descriptor);
    const compilationWallTimeMs = nowMs() - compilationStartedAtMs;
    const paramsBindGroup = device.createBindGroup({
      label: `${label} params bind group`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });
    const probeBindGroup = device.createBindGroup({
      label: `${label} storage bind group`,
      layout: pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
      ],
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniforms);
    device.queue.writeBuffer(inputBuffer, 0, inputs);
    if (drainQueue && typeof device.queue.onSubmittedWorkDone === "function") {
      await device.queue.onSubmittedWorkDone();
    }
    const encoder = device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, paramsBindGroup);
    pass.setBindGroup(1, probeBindGroup);
    pass.dispatchWorkgroups(Math.ceil(
      probes.length / STRONG_FIELD_GPU_PROBE_WORKGROUP_SIZE,
    ));
    pass.end();
    encoder.copyBufferToBuffer(
      outputBuffer,
      0,
      readbackBuffer,
      0,
      outputByteLength,
    );

    const submittedAtMs = nowMs();
    device.queue.submit([encoder.finish()]);
    if (typeof device.queue.onSubmittedWorkDone === "function") {
      await device.queue.onSubmittedWorkDone();
    }
    const completedAtMs = nowMs();
    await readbackBuffer.mapAsync(flags.readMode);
    mapped = true;
    const mappedRange = readbackBuffer.getMappedRange();
    const copied = mappedRange.slice(0);
    const mappedAtMs = nowMs();
    const records = decodeStrongFieldProbeOutputs(copied, probes);
    return Object.freeze({
      schema: STRONG_FIELD_GPU_PROBE_SCHEMA,
      probeCount: records.length,
      uniformFingerprint: fnv1a32(uniforms),
      probeFingerprint: fnv1a32(inputs),
      records,
      timing: Object.freeze({
        compilationWallTimeMs,
        queueWallTimeMs: completedAtMs - submittedAtMs,
        readbackMapWallTimeMs: mappedAtMs - completedAtMs,
      }),
    });
  } finally {
    if (mapped) {
      readbackBuffer.unmap();
    }
    for (const buffer of buffers) {
      buffer.destroy?.();
    }
  }
}

function toleranceDelta(a, b, absolute, relative) {
  return absolute + relative * Math.max(Math.abs(a), Math.abs(b));
}

function recordFailure(failures, index, field, baseline, candidate, limit) {
  failures.push(Object.freeze({ index, field, baseline, candidate, limit }));
}

/** Compare an optimized readback against a saved production baseline. */
export function compareStrongFieldGpuProbeRuns(
  baseline,
  candidate,
  tolerances = {},
) {
  if (
    baseline?.schema !== STRONG_FIELD_GPU_PROBE_SCHEMA
    || candidate?.schema !== STRONG_FIELD_GPU_PROBE_SCHEMA
  ) {
    throw new Error("Strong-field GPU probe schema mismatch");
  }
  if (
    baseline.probeFingerprint !== candidate.probeFingerprint
    || baseline.uniformFingerprint !== candidate.uniformFingerprint
  ) {
    throw new Error("Strong-field GPU probe inputs do not match the baseline");
  }
  if (
    !Array.isArray(baseline.records)
    || !Array.isArray(candidate.records)
    || baseline.records.length !== candidate.records.length
  ) {
    throw new Error("Strong-field GPU probe record counts do not match");
  }
  const limits = { ...STRONG_FIELD_GPU_PROBE_TOLERANCES, ...tolerances };
  const failures = [];
  const maxima = {
    escapeDirectionAbsolute: 0,
    frequencyShiftAbsolute: 0,
    lookbackAbsoluteM: 0,
    residualRegression: 0,
    iterationRegression: 0,
    minimumHorizonDistanceAbsoluteM: 0,
  };

  baseline.records.forEach((before, index) => {
    const after = candidate.records[index];
    if (before.outcome !== after.outcome) {
      recordFailure(
        failures,
        index,
        "outcome",
        before.outcome,
        after.outcome,
        0,
      );
    }
    if (before.terminationReason !== after.terminationReason) {
      recordFailure(
        failures,
        index,
        "terminationReason",
        before.terminationReason,
        after.terminationReason,
        0,
      );
    }
    for (let axis = 0; axis < 3; axis += 1) {
      const delta = Math.abs(
        before.escapeDirection[axis] - after.escapeDirection[axis],
      );
      maxima.escapeDirectionAbsolute = Math.max(
        maxima.escapeDirectionAbsolute,
        delta,
      );
      if (!Number.isFinite(delta) || delta > limits.escapeDirectionAbsolute) {
        recordFailure(
          failures,
          index,
          `escapeDirection[${axis}]`,
          before.escapeDirection[axis],
          after.escapeDirection[axis],
          limits.escapeDirectionAbsolute,
        );
      }
    }
    const frequencyDelta = Math.abs(
      before.frequencyShift - after.frequencyShift,
    );
    const frequencyLimit = toleranceDelta(
      before.frequencyShift,
      after.frequencyShift,
      limits.frequencyShiftAbsolute,
      limits.frequencyShiftRelative,
    );
    maxima.frequencyShiftAbsolute = Math.max(
      maxima.frequencyShiftAbsolute,
      frequencyDelta,
    );
    if (!Number.isFinite(frequencyDelta) || frequencyDelta > frequencyLimit) {
      recordFailure(
        failures,
        index,
        "frequencyShift",
        before.frequencyShift,
        after.frequencyShift,
        frequencyLimit,
      );
    }
    const lookbackDelta = Math.abs(before.lookback - after.lookback);
    const lookbackLimit = toleranceDelta(
      before.lookback,
      after.lookback,
      limits.lookbackAbsoluteM,
      limits.lookbackRelative,
    );
    maxima.lookbackAbsoluteM = Math.max(
      maxima.lookbackAbsoluteM,
      lookbackDelta,
    );
    if (!Number.isFinite(lookbackDelta) || lookbackDelta > lookbackLimit) {
      recordFailure(
        failures,
        index,
        "lookback",
        before.lookback,
        after.lookback,
        lookbackLimit,
      );
    }
    const residualRegression = (
      after.maximumNullResidual - before.maximumNullResidual
    );
    const residualLimit = (
      limits.residualAbsolute
      + limits.residualRegressionRelative
        * Math.abs(before.maximumNullResidual)
    );
    maxima.residualRegression = Math.max(
      maxima.residualRegression,
      residualRegression,
    );
    if (
      !Number.isFinite(residualRegression)
      || residualRegression > residualLimit
    ) {
      recordFailure(
        failures,
        index,
        "maximumNullResidual",
        before.maximumNullResidual,
        after.maximumNullResidual,
        residualLimit,
      );
    }
    const iterationRegression = after.iterations - before.iterations;
    maxima.iterationRegression = Math.max(
      maxima.iterationRegression,
      iterationRegression,
    );
    if (
      !Number.isFinite(iterationRegression)
      || iterationRegression > limits.iterationRegression
    ) {
      recordFailure(
        failures,
        index,
        "iterations",
        before.iterations,
        after.iterations,
        limits.iterationRegression,
      );
    }
    const horizonDistanceDelta = Math.abs(
      before.minimumHorizonDistance - after.minimumHorizonDistance,
    );
    maxima.minimumHorizonDistanceAbsoluteM = Math.max(
      maxima.minimumHorizonDistanceAbsoluteM,
      horizonDistanceDelta,
    );
    if (
      !Number.isFinite(horizonDistanceDelta)
      || horizonDistanceDelta > limits.minimumHorizonDistanceAbsoluteM
    ) {
      recordFailure(
        failures,
        index,
        "minimumHorizonDistance",
        before.minimumHorizonDistance,
        after.minimumHorizonDistance,
        limits.minimumHorizonDistanceAbsoluteM,
      );
    }
  });

  return Object.freeze({
    schema: STRONG_FIELD_GPU_PROBE_SCHEMA,
    pass: failures.length === 0,
    compared: baseline.records.length,
    failures: Object.freeze(failures),
    maxima: Object.freeze(maxima),
    baselineQueueWallTimeMs: baseline.timing?.queueWallTimeMs ?? null,
    candidateQueueWallTimeMs: candidate.timing?.queueWallTimeMs ?? null,
  });
}
