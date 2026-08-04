import assert from "node:assert/strict";
import test from "node:test";

import {
  STRONG_FIELD_GPU_PROBE_OUTPUT_BYTES,
  STRONG_FIELD_GPU_PROBE_SCHEMA,
  compareStrongFieldGpuProbeRuns,
  createStrongFieldProbeGrid,
  createStrongFieldProbeUniforms,
  decodeStrongFieldProbeOutputs,
  encodeStrongFieldProbeInputs,
  runStrongFieldGpuProbe,
  strongFieldGpuProbeWGSL,
} from "../src/strong-field-gpu-probe.js";
import {
  STRONG_FIELD_UNIFORM_FLOATS,
  strongFieldBinaryTraceFragmentWGSL,
} from "../src/strong-field-shaders.js";

function frame(overrides = {}) {
  const provider = new Float32Array(44);
  provider.set([0, 0, 0.35, 1e-4], 0);
  provider.set([-5, 0, 0, 0.5], 4);
  provider.set([0, 0, 0.1, 1], 8);
  provider.set([0, 0, 0.1, 0], 12);
  provider.set([5, 0, 0, 0.5], 16);
  provider.set([0, 0, -0.1, 1], 20);
  provider.set([0, 0, -0.1, 0], 24);
  provider.set([0, 0, 0, 0.95], 28);
  provider.set([0, 0, 0, 0], 32);
  provider.set([0, 0, 0.69, 0], 36);
  provider.set([1, 4, 1e-7, 1e6], 40);
  return {
    time: 0,
    steps: 72,
    cameraPos: [0, 0, 40],
    cameraRadius: 40,
    forward: [0, 0, -1],
    fov: 0.7,
    right: [1, 0, 0],
    up: [0, 1, 0],
    observerVelocity: [0, 0, 0],
    observerBeta: 0,
    sceneStrongFieldUniforms: provider,
    sceneStrongIntegrator: [0.018, 0.46, 3.5, 0.08],
    sceneStrongDomain: [96, 240, 0.035, 72],
    sceneStrongDiagnostics: [4, 180, 0.22, 1],
    ...overrides,
  };
}

function record(overrides = {}) {
  return {
    index: 0,
    id: "centre",
    outcome: 2,
    terminationReason: 0,
    escapeDirection: [0.1, 0.2, 0.9746794],
    frequencyShift: 1.02,
    lookback: 84.5,
    maximumNullResidual: 2e-4,
    iterations: 70,
    minimumHorizonDistance: 3.2,
    ...overrides,
  };
}

function run(records, overrides = {}) {
  return {
    schema: STRONG_FIELD_GPU_PROBE_SCHEMA,
    probeFingerprint: "same-probes",
    uniformFingerprint: "same-uniforms",
    records,
    timing: { queueWallTimeMs: 4 },
    ...overrides,
  };
}

test("GPU probe appends a compute entry to the exact production tracer", () => {
  assert.ok(strongFieldGpuProbeWGSL.startsWith(strongFieldBinaryTraceFragmentWGSL));
  assert.match(strongFieldGpuProbeWGSL, /fn strongFieldProbeMain/);
  assert.match(
    strongFieldGpuProbeWGSL,
    /let result = traceStrongField\(probe\.xy, probe\.z\)/,
  );
  for (const field of [
    "outcome",
    "escapeDirection",
    "frequencyShift",
    "lookback",
    "hamiltonianResidual",
    "iterations",
    "minimumHorizonDistance",
    "terminationReason",
  ]) {
    assert.match(strongFieldGpuProbeWGSL, new RegExp(`result\\.${field}`));
  }
});

test("probe uniforms preserve the production 96-float ABI", () => {
  const uniforms = createStrongFieldProbeUniforms(frame(), {
    resolution: [2560, 1440],
  });
  assert.ok(uniforms instanceof Float32Array);
  assert.equal(uniforms.length, STRONG_FIELD_UNIFORM_FLOATS);
  assert.deepEqual(Array.from(uniforms.slice(0, 2)), [2560, 1440]);
  assert.equal(uniforms[7], 72);
  assert.deepEqual(Array.from(uniforms.slice(8, 11)), [0, 0, 40]);
  assert.deepEqual(
    Array.from(uniforms.slice(36, 80)),
    Array.from(frame().sceneStrongFieldUniforms),
  );
  const controls = Array.from(uniforms.slice(80, 84));
  assert.ok(Math.abs(controls[0] - 0.018) < 1e-8);
  assert.ok(Math.abs(controls[1] - 0.46) < 1e-7);
  assert.equal(controls[2], 3.5);
  assert.ok(Math.abs(controls[3] - 0.08) < 1e-8);
});

test("probe grid encodes the same aspect-corrected pixel centres as fsMain", () => {
  const probes = createStrongFieldProbeGrid({
    columns: 3,
    rows: 3,
    aspect: 2,
    fov: 0.6,
  });
  assert.equal(probes.length, 9);
  assert.deepEqual(probes[4].screen, [0, 0]);
  assert.ok(Math.abs(probes[0].screen[0] + 4 / 3) < 1e-15);
  assert.ok(Math.abs(probes[0].screen[1] - 2 / 3) < 1e-15);
  const encoded = encodeStrongFieldProbeInputs(probes);
  assert.equal(encoded.byteLength, probes.length * 16);
  assert.equal(encoded[3], 0);
  assert.ok(Math.abs(encoded[2] - Math.tan(0.3)) < 1e-7);
});

test("readback decoder follows the mixed u32/f32 48-byte output ABI", () => {
  const bytes = new ArrayBuffer(STRONG_FIELD_GPU_PROBE_OUTPUT_BYTES);
  const view = new DataView(bytes);
  view.setUint32(0, 2, true);
  [3, 1.25, 91.5, 0.1, -0.2, 0.97, 3e-4, 88, 2.4].forEach(
    (value, index) => view.setFloat32(4 + index * 4, value, true),
  );
  const [decoded] = decodeStrongFieldProbeOutputs(bytes, [{ id: "ray-a" }]);
  assert.equal(decoded.id, "ray-a");
  assert.equal(decoded.outcome, 2);
  assert.equal(decoded.terminationReason, 3);
  assert.equal(decoded.frequencyShift, 1.25);
  assert.equal(decoded.lookback, 91.5);
  assert.deepEqual(decoded.escapeDirection.map((value) => (
    Number(value.toFixed(2))
  )), [0.1, -0.2, 0.97]);
  assert.ok(Math.abs(decoded.maximumNullResidual - 3e-4) < 1e-10);
  assert.equal(decoded.iterations, 88);
  assert.ok(Math.abs(decoded.minimumHorizonDistance - 2.4) < 1e-6);
});

test("baseline comparison accepts bounded drift and rejects every guarded channel", () => {
  const baseline = run([record()]);
  const acceptable = run([record({
    escapeDirection: [0.100001, 0.2, 0.9746794],
    frequencyShift: 1.02001,
    lookback: 84.505,
    maximumNullResidual: 2.02e-4,
    iterations: 69,
  })]);
  assert.equal(compareStrongFieldGpuProbeRuns(baseline, acceptable).pass, true);

  const guarded = [
    ["outcome", { outcome: 1 }],
    ["escapeDirection[0]", { escapeDirection: [0.2, 0.2, 0.9746794] }],
    ["frequencyShift", { frequencyShift: 1.2 }],
    ["lookback", { lookback: 85 }],
    ["maximumNullResidual", { maximumNullResidual: 4e-4 }],
    ["iterations", { iterations: 71 }],
    ["minimumHorizonDistance", { minimumHorizonDistance: 3.3 }],
  ];
  for (const [field, change] of guarded) {
    const report = compareStrongFieldGpuProbeRuns(
      baseline,
      run([record(change)]),
    );
    assert.equal(report.pass, false, field);
    assert.ok(report.failures.some((failure) => failure.field === field), field);
  }
});

test("baseline comparison refuses mismatched probe or uniform inputs", () => {
  const baseline = run([record()]);
  assert.throws(
    () => compareStrongFieldGpuProbeRuns(
      baseline,
      run([record()], { probeFingerprint: "different" }),
    ),
    /inputs do not match/,
  );
  assert.throws(
    () => compareStrongFieldGpuProbeRuns(
      baseline,
      run([record()], { uniformFingerprint: "different" }),
    ),
    /inputs do not match/,
  );
});

test("probe rejects non-finite pipeline specialization constants before GPU allocation", async () => {
  const device = {
    queue: {},
    createBuffer() {
      throw new Error("must not allocate");
    },
  };
  await assert.rejects(
    runStrongFieldGpuProbe(device, {
      uniforms: createStrongFieldProbeUniforms(frame()),
      probes: createStrongFieldProbeGrid({ columns: 1, rows: 1 }),
      pipelineConstants: { SPACETIME_PHASE_MODE: Number.NaN },
    }),
    /pipeline constants must be finite numbers/,
  );
});
