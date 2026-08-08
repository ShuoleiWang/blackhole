import {
  compareStrongFieldGpuProbeRuns,
  createStrongFieldProbeGrid,
  createStrongFieldProbeUniforms,
  runStrongFieldGpuProbe,
} from "../src/strong-field-gpu-probe.js";
import { createStrongFieldOrbitRuntime } from "../src/strong-field-orbit.js";
import {
  TIDAL_TRUNCATION_FACTOR,
  createBinaryAccretionState,
  stableAnnulusWeight,
} from "../src/scenes/binary-accretion-model.js";
import { loadBinaryDynamics } from "../src/scenes/binary-dynamics-adapter.js";

const MANIFEST_URL = new URL(
  "../assets/scenes/binary-sxs-bbh-0001-v2.json",
  import.meta.url,
);
const PROTOCOL_TIME_M = -9210.155252;
const TOTAL_MASS_SOLAR = 10 ** 9.81;
const ACCRETION_RATIO = 10 ** -1.7;
const GRID_COLUMNS = 64;
const GRID_ROWS = 36;
const RASTER = Object.freeze([2560, 1440]);
const DEG = Math.PI / 180;
const REQUIRED_ADAPTER = new URLSearchParams(globalThis.location?.search ?? "")
  .get("requireAdapter");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function normalize(vector) {
  const length = Math.hypot(...vector);
  assert(Number.isFinite(length) && length > 0, "Camera vector is invalid");
  return vector.map((component) => component / length);
}

function scale(vector, factor) {
  return vector.map((component) => component * factor);
}

function cross(first, second) {
  return [
    first[1] * second[2] - first[2] * second[1],
    first[2] * second[0] - first[0] * second[2],
    first[0] * second[1] - first[1] * second[0],
  ];
}

function distance(first, second) {
  return Math.hypot(
    first[0] - second[0],
    first[1] - second[1],
    first[2] - second[2],
  );
}

function cameraFrame(defaults) {
  const phase = 0.58;
  const latitude = (90 - defaults.initialViewingInclinationDeg) * DEG;
  const cosPhase = Math.cos(phase);
  const sinPhase = Math.sin(phase);
  const cosLatitude = Math.cos(latitude);
  const positionUnit = normalize([
    cosLatitude * cosPhase,
    Math.sin(latitude),
    cosLatitude * sinPhase,
  ]);
  const forward = scale(positionUnit, -1);
  const right = normalize([-sinPhase, 0, cosPhase]);
  return Object.freeze({
    cameraPos: Object.freeze(scale(positionUnit, defaults.observerRadiusM)),
    forward: Object.freeze(forward),
    right: Object.freeze(right),
    up: Object.freeze(normalize(cross(forward, right))),
  });
}

function accretionUniforms(geometry, weightA, weightB) {
  const [diskA, diskB] = geometry.disks;
  return new Float32Array([
    1, TIDAL_TRUNCATION_FACTOR, 1, 6,
    0, 1, 0, diskA.innerRadiusM,
    diskA.outerRadiusM, ACCRETION_RATIO, weightA, 1,
    0, 1, 0, diskB.innerRadiusM,
    diskB.outerRadiusM, ACCRETION_RATIO, weightB, 1,
  ]);
}

function summarize(run) {
  let hitCount = 0;
  let failureCount = 0;
  let maximumRadiance = 0;
  let maximumLuminance = 0;
  let whitePlateCount = 0;
  let chromaticHitCount = 0;
  let minimumTransmittance = 1;
  const outcomes = { captured: 0, escaped: 0, unresolved: 0 };
  for (const record of run.records) {
    const scalars = [
      record.frequencyShift,
      record.lookback,
      record.maximumNullResidual,
      record.iterations,
      record.minimumHorizonDistance,
      record.diskTransferFailure,
      record.diskTransmittance,
      ...record.escapeDirection,
      ...record.diskRadiance,
    ];
    assert(scalars.every(Number.isFinite), `Non-finite GPU record ${record.id}`);
    const radiance = Math.max(...record.diskRadiance);
    if (radiance > 1e-7 || record.diskTransmittance < 1 - 1e-7) {
      hitCount += 1;
    }
    const [red, green, blue] = record.diskRadiance;
    const luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
    maximumLuminance = Math.max(maximumLuminance, luminance);
    if (red >= 1 && green >= 1 && blue >= 1) {
      whitePlateCount += 1;
    }
    const minimumChannel = Math.max(Math.min(red, green, blue), 1e-8);
    if (radiance > 1e-7 && radiance / minimumChannel > 1.10) {
      chromaticHitCount += 1;
    }
    maximumRadiance = Math.max(maximumRadiance, radiance);
    minimumTransmittance = Math.min(
      minimumTransmittance,
      record.diskTransmittance,
    );
    if (record.diskTransferFailure > 0.5) {
      failureCount += 1;
    }
    if (record.outcome === 1) outcomes.captured += 1;
    else if (record.outcome === 2) outcomes.escaped += 1;
    else outcomes.unresolved += 1;
  }
  return Object.freeze({
    hitCount,
    failureCount,
    maximumRadiance,
    maximumLuminance,
    fp16Headroom: maximumRadiance > 0 ? 65504 / maximumRadiance : null,
    whitePlateCount,
    chromaticHitCount,
    minimumTransmittance,
    outcomes: Object.freeze(outcomes),
    timing: run.timing,
  });
}

async function main() {
  assert(globalThis.navigator?.gpu, "WebGPU is unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  assert(adapter, "No high-performance WebGPU adapter was returned");
  const exposedAdapterInfo = Object.freeze({
    vendor: adapter.info?.vendor ?? "",
    architecture: adapter.info?.architecture ?? "",
    device: adapter.info?.device ?? "",
    description: adapter.info?.description ?? "",
  });
  const adapterIdentity = [
    exposedAdapterInfo.vendor,
    exposedAdapterInfo.architecture,
    exposedAdapterInfo.device,
    exposedAdapterInfo.description,
  ].join(" ").toLowerCase();
  if (REQUIRED_ADAPTER === "apple-metal") {
    assert(
      adapterIdentity.includes("apple") && adapterIdentity.includes("metal"),
      `Required Apple Metal adapter, received ${adapterIdentity || "private adapter info"}`,
    );
  } else {
    assert(
      REQUIRED_ADAPTER === null,
      `Unknown required adapter gate: ${REQUIRED_ADAPTER}`,
    );
  }
  const device = await adapter.requestDevice();
  try {
    const track = await loadBinaryDynamics(MANIFEST_URL);
    const runtime = createStrongFieldOrbitRuntime({ track });
    const strongFrame = runtime.frameAt(PROTOCOL_TIME_M);
    assert(
      Number(strongFrame.uniforms[1]) === 0,
      "Probe epoch must select the binary specialization",
    );
    const bodies = strongFrame.orbitState.bodies;
    const geometry = createBinaryAccretionState({
      separationM: distance(bodies[0].positionM, bodies[1].positionM),
      massFractions: bodies.map((body) => body.massM),
      maximumOuterRadiusM: 10,
    });
    const weights = geometry.disks.map((disk) => stableAnnulusWeight(
      disk.widthM,
      disk.massFraction,
      0.75,
    ));
    assert(weights.every((weight) => weight > 0.999), "Probe disks are not active");

    const defaults = track.manifest.rendererDefaults;
    const camera = cameraFrame(defaults);
    const fov = defaults.fieldOfViewDeg * DEG;
    const baseFrame = Object.freeze({
      ...camera,
      time: PROTOCOL_TIME_M,
      massSolar: TOTAL_MASS_SOLAR,
      accretion: ACCRETION_RATIO,
      exposure: defaults.exposure,
      mode: 0,
      steps: 288,
      cameraRadius: defaults.observerRadiusM,
      fov,
      diskOuterRadius: Math.max(
        geometry.disks[0].outerRadiusM,
        geometry.disks[1].outerRadiusM,
      ),
      renderScale: 2,
      bloom: 0,
      motion: 0,
      frame: 0,
      observerVelocity: Object.freeze([0, 0, 0]),
      observerBeta: 0,
      sceneStrongFieldUniforms: strongFrame.uniforms,
      sceneStrongIntegrator: Object.freeze([0.010, 0.85, 4.0, 0.05]),
      sceneStrongDomain: Object.freeze([80, 220, 0.04, 32]),
      sceneStrongDiagnostics: Object.freeze([4, 180, 0.055, 1.9]),
    });
    const probes = createStrongFieldProbeGrid({
      columns: GRID_COLUMNS,
      rows: GRID_ROWS,
      aspect: RASTER[0] / RASTER[1],
      fov,
    });

    async function run(label, weightA, weightB) {
      const frame = {
        ...baseFrame,
        sceneStrongAccretionUniforms: accretionUniforms(
          geometry,
          weightA,
          weightB,
        ),
      };
      const uniforms = createStrongFieldProbeUniforms(frame, {
        resolution: RASTER,
        variant: "dual-disk",
      });
      return runStrongFieldGpuProbe(device, {
        label,
        uniforms,
        probes,
        variant: "dual-disk",
        pipelineConstants: { SPACETIME_PHASE_MODE: 0 },
      });
    }

    const both = await run("Dual-disk production probe · both", weights[0], weights[1]);
    const repeat = await run("Dual-disk production probe · repeat", weights[0], weights[1]);
    const diskA = await run("Dual-disk production probe · A", weights[0], 0);
    const diskB = await run("Dual-disk production probe · B", 0, weights[1]);
    const dark = await run("Dual-disk production probe · dark", 0, 0);
    const comparison = compareStrongFieldGpuProbeRuns(both, repeat);
    const summaries = Object.freeze({
      both: summarize(both),
      repeat: summarize(repeat),
      diskA: summarize(diskA),
      diskB: summarize(diskB),
      dark: summarize(dark),
    });

    assert(comparison.pass, "Repeated dual-disk GPU readback drifted");
    assert(summaries.both.hitCount > 0, "Combined disks produced no GPU hits");
    assert(summaries.diskA.hitCount > 0, "Disk A produced no GPU hits");
    assert(summaries.diskB.hitCount > 0, "Disk B produced no GPU hits");
    assert(summaries.both.failureCount === 0, "Combined transfer failed locally");
    assert(summaries.diskA.failureCount === 0, "Disk A transfer failed locally");
    assert(summaries.diskB.failureCount === 0, "Disk B transfer failed locally");
    assert(
      summaries.both.maximumRadiance < 4,
      "Visible-band disk radiance exhausted the declared HDR mastering range",
    );
    assert(
      summaries.both.whitePlateCount === 0,
      "Representative disk rays collapsed into an achromatic white plate",
    );
    assert(
      summaries.both.chromaticHitCount > 0,
      "Visible-band disk rays lost their temperature/frequency colour response",
    );
    assert(
      Math.abs(
        summaries.diskA.maximumRadiance - summaries.diskB.maximumRadiance,
      ) > 1e-3,
      "Fixed observer epoch lost the expected A/B Doppler brightness asymmetry",
    );
    assert(summaries.dark.hitCount === 0, "Dark disks emitted or absorbed light");
    assert(summaries.dark.failureCount === 0, "Dark disks reported transfer failure");
    assert(
      summaries.dark.minimumTransmittance === 1,
      "Dark disks changed transmittance",
    );

    return Object.freeze({
      status: "pass",
      schema: both.schema,
      adapter: exposedAdapterInfo,
      requiredAdapter: REQUIRED_ADAPTER,
      raster: RASTER,
      grid: Object.freeze([GRID_COLUMNS, GRID_ROWS]),
      probeCount: probes.length,
      protocolTimeM: PROTOCOL_TIME_M,
      separationM: geometry.separationM,
      diskOuterRadiiM: Object.freeze(
        geometry.disks.map((disk) => disk.outerRadiusM),
      ),
      comparison,
      summaries,
    });
  } finally {
    device.destroy?.();
  }
}

const resultElement = document.querySelector("#result");
try {
  const result = await main();
  globalThis.__strongFieldDualDiskProbeResult = result;
  document.documentElement.dataset.status = result.status;
  resultElement.textContent = JSON.stringify(result, null, 2);
} catch (error) {
  const result = Object.freeze({
    status: "fail",
    message: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack : null,
  });
  globalThis.__strongFieldDualDiskProbeResult = result;
  document.documentElement.dataset.status = result.status;
  resultElement.textContent = JSON.stringify(result, null, 2);
  console.error(error);
}
