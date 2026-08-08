import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  STRONG_FIELD_DUAL_DISK_GPU_PROBE_OUTPUT_BYTES,
  STRONG_FIELD_DUAL_DISK_GPU_PROBE_SCHEMA,
  STRONG_FIELD_GPU_PROBE_OUTPUT_BYTES,
  STRONG_FIELD_GPU_PROBE_SCHEMA,
  compareStrongFieldGpuProbeRuns,
  createStrongFieldProbeGrid,
  createStrongFieldProbeUniforms,
  decodeStrongFieldDualDiskProbeOutputs,
  decodeStrongFieldProbeOutputs,
  encodeStrongFieldProbeInputs,
  runStrongFieldGpuProbe,
  strongFieldDualDiskGpuProbeWGSL,
  strongFieldGpuProbeWGSL,
} from "../src/strong-field-gpu-probe.js";
import {
  STRONG_FIELD_ACCRETION_UNIFORM_FLOATS,
  STRONG_FIELD_UNIFORM_FLOATS,
  strongFieldBinaryDualDiskTraceFragmentWGSL,
  strongFieldBinaryTraceFragmentWGSL,
} from "../src/strong-field-shaders.js";

const dualDiskUniforms = Object.freeze([
  1, 0.8, 1, 6,
  0, 1, 0, 3,
  8.5, 0.02, 1, 1,
  0, 1, 0, 3,
  8.5, 0.02, 1, 1,
]);

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

function dualDiskRecord(overrides = {}) {
  return record({
    diskTransferFailure: 0,
    diskRadiance: [2.5, 1.25, 0.5],
    diskTransmittance: 0.125,
    ...overrides,
  });
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

function dualDiskRun(records, overrides = {}) {
  return run(records, {
    schema: STRONG_FIELD_DUAL_DISK_GPU_PROBE_SCHEMA,
    ...overrides,
  });
}

test("GPU probe appends a compute entry to the exact production tracer", () => {
  assert.ok(strongFieldGpuProbeWGSL.startsWith(strongFieldBinaryTraceFragmentWGSL));
  assert.equal(strongFieldGpuProbeWGSL.length, 50531);
  assert.equal(
    createHash("sha256").update(strongFieldGpuProbeWGSL).digest("hex"),
    "b0856b7ff0c1cb10a6cb5fff97a15d7b4e95654ed376125de8c21bb9ce26226f",
  );
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

test("dual-disk GPU probe appends readback to the exact 116-float tracer", () => {
  assert.ok(
    strongFieldDualDiskGpuProbeWGSL.startsWith(
      strongFieldBinaryDualDiskTraceFragmentWGSL,
    ),
  );
  assert.match(strongFieldDualDiskGpuProbeWGSL, /fn strongFieldProbeMain/);
  assert.match(
    strongFieldDualDiskGpuProbeWGSL,
    /let result = traceStrongField\(probe\.xy, probe\.z\)/,
  );
  for (const field of [
    "diskRadiance",
    "diskTransmittance",
    "diskTransferFailure",
  ]) {
    assert.match(
      strongFieldDualDiskGpuProbeWGSL,
      new RegExp(`result\\.${field}`),
    );
    assert.doesNotMatch(strongFieldGpuProbeWGSL, new RegExp(`result\\.${field}`));
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

test("dual-disk probe parameterization preserves 96 floats then appends 20", () => {
  const dualFrame = frame({
    sceneStrongAccretionUniforms: dualDiskUniforms,
  });
  const vacuum = createStrongFieldProbeUniforms(dualFrame, {
    resolution: [2560, 1440],
  });
  const dualDisk = createStrongFieldProbeUniforms(dualFrame, {
    resolution: [2560, 1440],
    variant: "dual-disk",
  });
  assert.equal(vacuum.length, STRONG_FIELD_UNIFORM_FLOATS);
  assert.equal(dualDisk.length, STRONG_FIELD_ACCRETION_UNIFORM_FLOATS);
  assert.deepEqual(
    Array.from(dualDisk.slice(0, STRONG_FIELD_UNIFORM_FLOATS)),
    Array.from(vacuum),
  );
  assert.deepEqual(
    Array.from(dualDisk.slice(STRONG_FIELD_UNIFORM_FLOATS)),
    dualDiskUniforms.map(Math.fround),
  );
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

test("dual-disk readback appends deterministic radiance and transfer fields", () => {
  assert.equal(STRONG_FIELD_DUAL_DISK_GPU_PROBE_OUTPUT_BYTES, 64);
  const bytes = new ArrayBuffer(STRONG_FIELD_DUAL_DISK_GPU_PROBE_OUTPUT_BYTES);
  const view = new DataView(bytes);
  view.setUint32(0, 1, true);
  [
    [4, 0],
    [8, 0.75],
    [12, 72.5],
    [16, 0.1],
    [20, -0.2],
    [24, 0.97],
    [28, 4e-4],
    [32, 63],
    [36, 1.75],
    [40, 1],
    [48, 2.5],
    [52, 1.25],
    [56, 0.5],
    [60, 0.125],
  ].forEach(([offset, value]) => view.setFloat32(offset, value, true));
  const [decoded] = decodeStrongFieldDualDiskProbeOutputs(
    bytes,
    [{ id: "disk-crossing" }],
  );
  assert.equal(decoded.id, "disk-crossing");
  assert.equal(decoded.outcome, 1);
  assert.equal(decoded.diskTransferFailure, 1);
  assert.deepEqual(decoded.diskRadiance, [2.5, 1.25, 0.5]);
  assert.equal(decoded.diskTransmittance, 0.125);
  assert.equal(decoded.iterations, 63);
  assert.equal(decoded.minimumHorizonDistance, 1.75);
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

test("dual-disk baseline comparison guards radiance, transmittance, and failure", () => {
  const baseline = dualDiskRun([dualDiskRecord()]);
  const acceptable = dualDiskRun([dualDiskRecord({
    diskRadiance: [2.5001, 1.25, 0.5],
    diskTransmittance: 0.125005,
  })]);
  assert.equal(compareStrongFieldGpuProbeRuns(baseline, acceptable).pass, true);

  for (const [field, change] of [
    ["diskRadiance[0]", { diskRadiance: [2.6, 1.25, 0.5] }],
    ["diskTransmittance", { diskTransmittance: 0.15 }],
    ["diskTransferFailure", { diskTransferFailure: 1 }],
  ]) {
    const report = compareStrongFieldGpuProbeRuns(
      baseline,
      dualDiskRun([dualDiskRecord(change)]),
    );
    assert.equal(report.pass, false, field);
    assert.ok(report.failures.some((failure) => failure.field === field));
  }
  assert.throws(
    () => compareStrongFieldGpuProbeRuns(baseline, run([record()])),
    /schema mismatch/,
  );
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

test("probe variants reject the wrong uniform ABI before GPU allocation", async () => {
  const device = {
    queue: {},
    createBuffer() {
      throw new Error("must not allocate");
    },
  };
  const probes = createStrongFieldProbeGrid({ columns: 1, rows: 1 });
  const dualFrame = frame({
    sceneStrongAccretionUniforms: dualDiskUniforms,
  });
  const vacuum = createStrongFieldProbeUniforms(dualFrame);
  const dualDisk = createStrongFieldProbeUniforms(dualFrame, {
    variant: "dual-disk",
  });
  await assert.rejects(
    runStrongFieldGpuProbe(device, { uniforms: dualDisk, probes }),
    /Probe uniforms must contain 96 floats/,
  );
  await assert.rejects(
    runStrongFieldGpuProbe(device, {
      uniforms: vacuum,
      probes,
      variant: "dual-disk",
    }),
    /Probe uniforms for dual-disk must contain 116 floats/,
  );
  await assert.rejects(
    runStrongFieldGpuProbe(device, {
      uniforms: vacuum,
      probes,
      variant: "not-a-variant",
    }),
    /Unknown strong-field GPU probe variant/,
  );
});

test("browser corpus exposes an explicit Apple Metal acceptance gate", async () => {
  const source = await readFile(
    new URL("./strong-field-gpu-probe-browser.mjs", import.meta.url),
    "utf8",
  );
  assert.match(source, /requireAdapter/);
  assert.match(source, /REQUIRED_ADAPTER === "apple-metal"/);
  assert.match(source, /adapterIdentity\.includes\("apple"\)/);
  assert.match(source, /adapterIdentity\.includes\("metal"\)/);
  assert.match(source, /Unknown required adapter gate/);
});
