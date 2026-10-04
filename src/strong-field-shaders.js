/*
 * Real-time strong-field binary tracer for the WebGPU production path.
 *
 * Scientific boundary
 * -------------------
 * The inspiral metric is a frame-frozen superposition of two unboosted
 * Kerr-Schild metric contributions (see FROZEN_METRIC_VELOCITY).  It is a
 * strong-field analytic approximation, not a numerical-relativity metric.
 * During merger it blends to one Kerr-Schild remnant whose mass and spin are
 * supplied by the source-backed dynamics track.  The metric provider
 * deliberately accepts (coordinateTime, position), so a later slow-light
 * implementation can advance coordinateTime along the ray without replacing
 * the geodesic integrator.
 *
 * The WebGL2 member of this bundle intentionally remains the existing
 * weak-field preview.  WebGPU/Metal is the production strong-field path; the
 * fallback is labelled as a different physical model rather than constraining
 * WGSL to GLSL parity.
 *
 * Uniform ABI (f32 offsets)
 * -------------------------
 * Shared renderer fields occupy 0..35.  The strong-field tail starts at 36:
 *
 *  36..79 blackhole.strong-field-uniforms/v1 (44 floats), packed by
 *         createStrongFieldSpacetimeProvider().  It carries explicit PN/EOB
 *         positions, velocities, arbitrary spin vectors, remnant state,
 *         attenuation, regularization, and the C2 transition weight.
 *  80 minimum dt M       81 maximum dt M
 *  82 critical distance  83 residual fail threshold
 *  84 escape radius M    85 maximum lookback M
 *  86 capture padding M  87 critical-zone step bonus
 *  88 maximum shown g    89 residual visual scale
 *  90 unresolved level   91 model flags/reserved
 *  92 accumulation index 93 running-average weight
 *  94 history epoch       95 history-reset flag
 */

import { binaryTraceFragmentGLSL } from "./binary-shaders.js";
import { STRONG_FIELD_UNIFORM_ABI } from "./strong-field-spacetime.js";

export const STRONG_FIELD_UNIFORM_FLOATS = 96;
export const STRONG_FIELD_UNIFORM_TAIL_FLOATS = 60;
export const STRONG_FIELD_ACCRETION_UNIFORM_FLOATS = 116;
export const STRONG_FIELD_ACCRETION_UNIFORM_TAIL_FLOATS = 80;
// Classical RK4 steps scale with the distance to the nearest horizon, so far-
// field steps grow geometrically. This ceiling only bounds a single step; the
// per-tier step fraction controls accuracy.
export const STRONG_FIELD_MAXIMUM_STEP_M = 64;
// Sparse tracing for moving frames: fsCoarse traces every
// STRONG_FIELD_SPARSE_STRIDE-th pixel into an rgba32float field that fsMain
// interpolates where it is provably smooth; frames opt in with
// sceneStrongDiagnostics.w = stride.
export const STRONG_FIELD_SPARSE_STRIDE = 4;

export const STRONG_FIELD_UNIFORM_LAYOUT = Object.freeze({
  shared: Object.freeze({ offset: 0, floats: 36 }),
  spacetimeProvider: Object.freeze({
    offset: 36,
    floats: STRONG_FIELD_UNIFORM_ABI.floatCount,
    schema: STRONG_FIELD_UNIFORM_ABI.schema,
  }),
  sceneStrongIntegrator: Object.freeze({ offset: 80, floats: 4 }),
  sceneStrongDomain: Object.freeze({ offset: 84, floats: 4 }),
  sceneStrongDiagnostics: Object.freeze({ offset: 88, floats: 4 }),
  sceneStrongQuality: Object.freeze({ offset: 92, floats: 4 }),
});

export const STRONG_FIELD_ACCRETION_UNIFORM_LAYOUT = Object.freeze({
  ...STRONG_FIELD_UNIFORM_LAYOUT,
  sceneStrongAccretion: Object.freeze({ offset: 96, floats: 20 }),
});

export const STRONG_FIELD_OUTCOMES = Object.freeze({
  unresolved: 0,
  captured: 1,
  escaped: 2,
});

export const STRONG_FIELD_DIAGNOSTIC_MODES = Object.freeze({
  sky: 0,
  outcome: 1,
  lookback: 2,
  frequencyShift: 3,
  hamiltonianResidual: 4,
  integrationCost: 5,
});

const DEFAULT_STRONG_FIELD_INTEGRATOR = Object.freeze([
  0.01, // minimum coordinate-time RK4 step in M
  64, // maximum RK4 step in M
  0.30, // step as a fraction of the distance to the nearest horizon
  0.01, // fail-closed relative energy drift per step
]);

const DEFAULT_STRONG_FIELD_DOMAIN = Object.freeze([
  160, // escape sphere radius in M
  400, // maximum coordinate lookback in M
  0.01, // horizon backstop padding in M (capture is at the photon orbit)
  144, // maximum RK4 steps
]);

const DEFAULT_STRONG_FIELD_DIAGNOSTICS = Object.freeze([
  4, // maximum displayed frequency shift
  180, // logarithmic residual visualisation scale
  0.22, // unresolved diagnostic brightness
  0, // reserved
]);

function finiteVec4(value, name, fallback = null) {
  if (value == null && fallback) {
    return fallback;
  }
  if (
    !value
    || typeof value.length !== "number"
    || value.length !== 4
    || Array.from(value).some((entry) => !Number.isFinite(Number(entry)))
  ) {
    throw new Error(`${name} must contain exactly four finite numbers`);
  }
  return value;
}

function qualityVector(frame) {
  const quality = frame?.strongFieldQuality;
  if (quality == null) {
    return [0, 1, 0, 1];
  }
  const values = [
    quality.accumulationIndex,
    quality.accumulationWeight,
    quality.historyEpoch,
    quality.historyReset ? 1 : 0,
  ];
  if (values.some((value) => !Number.isFinite(Number(value)))) {
    throw new Error("strongFieldQuality must contain finite accumulation state");
  }
  if (
    values[0] < 0
    || values[1] <= 0
    || values[1] > 1
    || values[2] < 0
  ) {
    throw new Error("strongFieldQuality accumulation state is out of range");
  }
  return values;
}

/**
 * Pack the scene-owned part of the renderer uniform buffer.
 *
 * The 44-float provider payload is mandatory and must come from
 * createStrongFieldSpacetimeProvider().frameAt(time).uniforms.  In particular,
 * this writer never constructs Kerr-Schild positions from gauge-dependent SXS
 * horizon centroids, separation, or phase.  Integrator/display controls have
 * conservative M3 Pro defaults and do not alter the spacetime evidence.
 */
export function writeStrongFieldUniformTail(target, frame) {
  if (
    !target
    || typeof target.set !== "function"
    || target.length < STRONG_FIELD_UNIFORM_TAIL_FLOATS
  ) {
    throw new Error(
      `Strong-field uniform tail needs ${STRONG_FIELD_UNIFORM_TAIL_FLOATS} floats`,
    );
  }

  const provider = frame?.sceneStrongFieldUniforms;
  if (
    !provider
    || typeof provider.length !== "number"
    || provider.length !== STRONG_FIELD_UNIFORM_ABI.floatCount
    || Array.from(provider).some((entry) => !Number.isFinite(Number(entry)))
  ) {
    throw new Error(
      `sceneStrongFieldUniforms must contain ${STRONG_FIELD_UNIFORM_ABI.floatCount} finite PN/EOB-provider floats`,
    );
  }
  const integrator = finiteVec4(
    frame?.sceneStrongIntegrator,
    "sceneStrongIntegrator",
    DEFAULT_STRONG_FIELD_INTEGRATOR,
  );
  const domain = finiteVec4(
    frame?.sceneStrongDomain,
    "sceneStrongDomain",
    DEFAULT_STRONG_FIELD_DOMAIN,
  );
  const diagnostics = finiteVec4(
    frame?.sceneStrongDiagnostics,
    "sceneStrongDiagnostics",
    DEFAULT_STRONG_FIELD_DIAGNOSTICS,
  );
  const quality = qualityVector(frame);

  target.set(provider, 0);
  target.set(integrator, 44);
  target.set(domain, 48);
  target.set(diagnostics, 52);
  target.set(quality, 56);
  return target;
}

/**
 * Append the dual-mini-disk emission contract without changing the 44-float
 * spacetime provider or the 96-float vacuum tracer ABI.
 *
 * Layout at absolute offsets 96..115:
 *   control: active, tidal truncation factor, thermal scale, peak optical depth
 *   disk A:  normal.xyz, ISCO; outer radius, Eddington ratio, C2 weight, sign
 *   disk B:  normal.xyz, ISCO; outer radius, Eddington ratio, C2 weight, sign
 */
export function writeStrongFieldAccretionUniformTail(target, frame) {
  if (
    !target
    || typeof target.set !== "function"
    || typeof target.subarray !== "function"
    || target.length < STRONG_FIELD_ACCRETION_UNIFORM_TAIL_FLOATS
  ) {
    throw new Error(
      `Strong-field accretion tail needs ${STRONG_FIELD_ACCRETION_UNIFORM_TAIL_FLOATS} floats`,
    );
  }
  writeStrongFieldUniformTail(
    target.subarray(0, STRONG_FIELD_UNIFORM_TAIL_FLOATS),
    frame,
  );
  const accretion = frame?.sceneStrongAccretionUniforms;
  if (
    !accretion
    || typeof accretion.length !== "number"
    || accretion.length !== 20
    || Array.from(accretion).some((entry) => !Number.isFinite(Number(entry)))
  ) {
    throw new Error(
      "sceneStrongAccretionUniforms must contain exactly 20 finite numbers",
    );
  }
  const active = Number(accretion[0]);
  if (active !== 0 && active !== 1) {
    throw new Error("Strong-field accretion active flag must be 0 or 1");
  }
  const tidalTruncation = Number(accretion[1]);
  const thermalScale = Number(accretion[2]);
  const peakOpticalDepth = Number(accretion[3]);
  if (
    tidalTruncation <= 0
    || tidalTruncation > 1
    || thermalScale <= 0
    || thermalScale > 16
    || peakOpticalDepth < 0
    || peakOpticalDepth > 100
  ) {
    throw new Error("Strong-field accretion control parameters are out of range");
  }
  const transitionWeight = Number(frame.sceneStrongFieldUniforms[1]);
  if (
    transitionWeight > 0
    && (Number(accretion[10]) > 0 || Number(accretion[18]) > 0)
  ) {
    throw new Error(
      "Individual strong-field accretion disks must be dark during merger transition and remnant phases",
    );
  }
  for (const offset of [4, 12]) {
    const normalLength = Math.hypot(
      Number(accretion[offset]),
      Number(accretion[offset + 1]),
      Number(accretion[offset + 2]),
    );
    const innerRadius = Number(accretion[offset + 3]);
    const outerRadius = Number(accretion[offset + 4]);
    const eddingtonRatio = Number(accretion[offset + 5]);
    const weight = Number(accretion[offset + 6]);
    const rotationSign = Number(accretion[offset + 7]);
    if (
      active === 1
      && (
        Math.abs(normalLength - 1) > 1e-5
        || innerRadius <= 0
        || innerRadius > 1.0e5
        || outerRadius < 0
        || outerRadius > 1.0e5
        || eddingtonRatio <= 0
        || eddingtonRatio > 1.0e3
        || weight < 0
        || weight > 1
        || Math.abs(rotationSign) !== 1
        || (weight > 0 && outerRadius <= innerRadius)
      )
    ) {
      throw new Error("Strong-field accretion disk parameters are out of range");
    }
  }
  target.set(accretion, STRONG_FIELD_UNIFORM_TAIL_FLOATS);
  return target;
}

// Analytic, frame-frozen radiative-transfer layer for the dual-mini-disk
// scene. This is deliberately an emission prescription on top of the
// approximate vacuum metric: it neither evolves matter nor claims GRMHD/NR.
// The functions are injected only into the 116-float dual-disk WGSL module, so
// the production 96-float vacuum shader retains its original code path.
const STRONG_FIELD_DUAL_DISK_FUNCTIONS_WGSL = /* wgsl */ `
struct DiskIntersection {
  fraction: f32,
  radius: f32,
  position: vec3<f32>,
  restPosition: vec3<f32>,
  valid: f32,
};

struct DiskTransferSample {
  radiance: vec3<f32>,
  opacity: f32,
  valid: f32,
};

fn emptyDiskIntersection() -> DiskIntersection {
  var result: DiskIntersection;
  result.fraction = 2.0;
  result.radius = 0.0;
  result.position = vec3<f32>(0.0);
  result.restPosition = vec3<f32>(0.0);
  result.valid = 0.0;
  return result;
}

fn bodyRestDisplacement(
  position: vec3<f32>,
  centre: vec3<f32>,
  velocity: vec3<f32>
) -> vec3<f32> {
  let displacement = position - centre;
  let speedSquared = dot(velocity, velocity);
  if (speedSquared <= 1.0e-12) {
    return displacement;
  }
  if (speedSquared >= 0.9999) {
    return vec3<f32>(1.0e19);
  }
  let gamma = inverseSqrt(1.0 - speedSquared);
  return displacement + velocity * (
    (gamma - 1.0) * dot(displacement, velocity) / speedSquared
  );
}

// Locate a crossing on the accepted RK4 chord. Body locations are frozen for
// the complete ray, matching the metric provider's fast-light contract. The
// frozen Kerr-Schild terms are unboosted, so each hole's horizon and ISCO are
// at rest in the coordinate frame and its disk is circular there; the bulk
// velocity enters only the emitter kinematics (Doppler boosting). A small
// positive lower bound prevents an endpoint hit from being counted again at
// the beginning of the next segment.
fn segmentDiskIntersection(
  segmentStart: vec3<f32>,
  segmentEnd: vec3<f32>,
  centre: vec3<f32>,
  normalInput: vec3<f32>,
  innerRadius: f32,
  outerRadius: f32,
  activeWeight: f32
) -> DiskIntersection {
  var result = emptyDiskIntersection();
  if (
    params.sceneDiskControl.x < 0.5
    || activeWeight <= 1.0e-6
    || innerRadius <= 0.0
    || outerRadius <= innerRadius
  ) {
    return result;
  }
  let normalLength = length(normalInput);
  if (!finiteScalar(normalLength) || normalLength < 0.999 || normalLength > 1.001) {
    return result;
  }
  let normal = normalInput / normalLength;
  let restStart = bodyRestDisplacement(segmentStart, centre, FROZEN_METRIC_VELOCITY);
  let restEnd = bodyRestDisplacement(segmentEnd, centre, FROZEN_METRIC_VELOCITY);
  if (!finiteVector(restStart) || !finiteVector(restEnd)) {
    return result;
  }
  let sideStart = dot(restStart, normal);
  let sideEnd = dot(restEnd, normal);
  let denominator = sideStart - sideEnd;
  if (
    !finiteScalar(denominator)
    || abs(denominator) <= 1.0e-7
    || sideStart * sideEnd > 0.0
  ) {
    return result;
  }
  let fraction = sideStart / denominator;
  if (fraction <= 1.0e-5 || fraction > 1.0) {
    return result;
  }
  let restHit = mix(restStart, restEnd, fraction);
  let planarHit = restHit - normal * dot(restHit, normal);
  let radius = length(planarHit);
  if (
    !finiteScalar(radius)
    || radius < innerRadius
    || radius > outerRadius
  ) {
    return result;
  }
  result.fraction = fraction;
  result.radius = radius;
  result.position = mix(segmentStart, segmentEnd, fraction);
  result.restPosition = planarHit;
  result.valid = 1.0;
  return result;
}

// Sparse tracing: whether a chord passes within footprint (M) of a disk
// annulus in the hole's rest frame. The part of the chord inside the slab
// |z| <= footprint is tested against the annulus widened by footprint.
// Non-finite input counts as near (fail closed).
fn chordNearDisk(
  segmentStart: vec3<f32>,
  segmentEnd: vec3<f32>,
  centre: vec3<f32>,
  normalInput: vec3<f32>,
  innerRadius: f32,
  outerRadius: f32,
  activeWeight: f32,
  footprint: f32
) -> bool {
  if (activeWeight <= 1.0e-6 || outerRadius <= innerRadius) {
    return false;
  }
  let normal = safeNormalize(normalInput);
  let start = bodyRestDisplacement(segmentStart, centre, FROZEN_METRIC_VELOCITY);
  let end = bodyRestDisplacement(segmentEnd, centre, FROZEN_METRIC_VELOCITY);
  let sideStart = dot(start, normal);
  let sideEnd = dot(end, normal);
  if (!finiteScalar(sideStart) || !finiteScalar(sideEnd)) {
    return true;
  }
  let rise = sideEnd - sideStart;
  var low = 0.0;
  var high = 1.0;
  if (abs(rise) > 1.0e-9) {
    let lowerCrossing = (-footprint - sideStart) / rise;
    let upperCrossing = (footprint - sideStart) / rise;
    low = max(low, min(lowerCrossing, upperCrossing));
    high = min(high, max(lowerCrossing, upperCrossing));
  } else if (abs(sideStart) > footprint) {
    return false;
  }
  if (low > high) {
    return false;
  }
  let first = mix(start, end, low);
  let last = mix(start, end, high);
  let planarFirst = first - normal * dot(first, normal);
  let chord = (last - normal * dot(last, normal)) - planarFirst;
  let nearestFraction = clamp(
    -dot(planarFirst, chord) / max(dot(chord, chord), 1.0e-12),
    0.0,
    1.0
  );
  let nearest = length(planarFirst + nearestFraction * chord);
  let farthest = max(length(planarFirst), length(planarFirst + chord));
  return nearest <= outerRadius + footprint
    && farthest >= innerRadius - footprint;
}

fn chordNearDisks(
  segmentStart: vec3<f32>,
  segmentEnd: vec3<f32>,
  footprint: f32
) -> bool {
  if (params.sceneDiskControl.x < 0.5) {
    return false;
  }
  return chordNearDisk(
      segmentStart, segmentEnd, params.bodyAPositionMass.xyz,
      params.diskANormalInner.xyz, params.diskANormalInner.w,
      params.diskAOuterAccretionWeight.x, params.diskAOuterAccretionWeight.z,
      footprint
    )
    || chordNearDisk(
      segmentStart, segmentEnd, params.bodyBPositionMass.xyz,
      params.diskBNormalInner.xyz, params.diskBNormalInner.w,
      params.diskBOuterAccretionWeight.x, params.diskBOuterAccretionWeight.z,
      footprint
    );
}

fn visibleBlackbodyLinearSrgbPerBolometric(
  temperatureKelvin: f32
) -> vec3<f32> {
  // Fifteen uniformly spaced samples spanning 380-780 nm. The XYZ weights are the
  // fixed Wyman-Sloan-Shirley approximation to the CIE 1931 2-degree observer.
  // Unlike the previous three-point, unit-luminance colour proxy, this keeps
  // the fraction of bolometric power that actually lands in the visible band.
  let cieSamples = array<vec4<f32>, 15>(
    vec4<f32>(0.38000000, 0.000101768, 0.000126361, 0.003342521),
    vec4<f32>(0.40857143, 0.041103205, 0.002427076, 0.171379214),
    vec4<f32>(0.43714286, 0.345374934, 0.016094588, 1.694938040),
    vec4<f32>(0.46571429, 0.231835560, 0.073791241, 1.489866541),
    vec4<f32>(0.49428571, 0.013837655, 0.256028875, 0.368248028),
    vec4<f32>(0.52285714, 0.091902156, 0.761894395, 0.070519050),
    vec4<f32>(0.55142857, 0.457097020, 0.996550478, 0.007724523),
    vec4<f32>(0.58000000, 0.920460516, 0.872133791, 0.000450337),
    vec4<f32>(0.60857143, 1.014443172, 0.519281599, 0.000013971),
    vec4<f32>(0.63714286, 0.510206266, 0.198321624, 0.000000231),
    vec4<f32>(0.66571429, 0.109492500, 0.046801697, 0.000000002),
    vec4<f32>(0.69428571, 0.010026499, 0.006733690, 0.000000000),
    vec4<f32>(0.72285714, 0.000391779, 0.000589022, 0.000000000),
    vec4<f32>(0.75142857, 0.000006532, 0.000031314, 0.000000000),
    vec4<f32>(0.78000000, 0.000000023, 0.000000506, 0.000000000)
  );
  let temperature = max(temperatureKelvin, 2.0);
  var xyz = vec3<f32>(0.0);
  for (var sampleIndex: i32 = 0; sampleIndex < 15; sampleIndex = sampleIndex + 1) {
    let sample = cieSamples[sampleIndex];
    let wavelength = sample.x;
    let wavelength2 = wavelength * wavelength;
    let wavelength5 = wavelength2 * wavelength2 * wavelength;
    let exponent = min(80.0, 14387.77 / (temperature * wavelength));
    let spectralRadiance = 1.0 / (
      wavelength5 * max(exp(exponent) - 1.0, 1.0e-12)
    );
    xyz = xyz + spectralRadiance * sample.yzw;
  }
  let linearSrgb = vec3<f32>(
     3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z,
    -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z,
     0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z
  );
  let referenceRatio = 6500.0 / temperature;
  let referenceRatio2 = referenceRatio * referenceRatio;
  let bolometricNormalization = (
    referenceRatio2 * referenceRatio2 / 1.3256608705
  );
  // A small channel calibration makes the 6500 K reference neutral in the
  // renderer's D65 linear-sRGB working space without changing temperature
  // ordering or inventing identity colours for the two equal-mass disks.
  return max(
    linearSrgb
      * bolometricNormalization
      * vec3<f32>(0.9662282813, 1.0159099238, 0.9498410897),
    vec3<f32>(0.0)
  );
}

fn smootherstep01(value: f32) -> f32 {
  let x = clamp(value, 0.0, 1.0);
  return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

fn annulusEdgeCoverage(
  radius: f32,
  innerRadius: f32,
  outerRadius: f32,
  bodyMassM: f32
) -> f32 {
  let width = max(outerRadius - innerRadius, 1.0e-6);
  let edgeWidth = min(
    max(0.12 * width, 0.08 * bodyMassM),
    0.45 * width
  );
  let innerCoverage = smootherstep01((radius - innerRadius) / edgeWidth);
  let outerCoverage = smootherstep01((outerRadius - radius) / edgeWidth);
  return innerCoverage * outerCoverage;
}

fn spatialDot(fields: MetricValues, a: vec3<f32>, b: vec3<f32>) -> f32 {
  return dot(a, fields.spatialMetric * b);
}

// Relativistically compose the hole's Eulerian velocity with a circular
// orbital velocity supplied in the hole's instantaneous rest frame. The
// spatial metric defines all local dot products. Invalid/superluminal states
// fail closed in diskTransferAtIntersection().
fn composeEulerianVelocities(
  fields: MetricValues,
  bodyCoordinateVelocity: vec3<f32>,
  orbitalRestVelocity: vec3<f32>
) -> vec3<f32> {
  let bodyVelocity = (
    bodyCoordinateVelocity + fields.shift
  ) / max(fields.lapse, 1.0e-6);
  let bodySpeedSquared = spatialDot(fields, bodyVelocity, bodyVelocity);
  if (bodySpeedSquared <= 1.0e-12) {
    return orbitalRestVelocity;
  }
  if (bodySpeedSquared >= 0.999) {
    return vec3<f32>(1.0e19);
  }
  let projection = spatialDot(
    fields,
    orbitalRestVelocity,
    bodyVelocity
  );
  let parallel = bodyVelocity * (projection / bodySpeedSquared);
  let perpendicular = orbitalRestVelocity - parallel;
  let gammaBody = inverseSqrt(1.0 - bodySpeedSquared);
  let denominator = 1.0 + projection;
  if (!finiteScalar(denominator) || denominator <= 1.0e-5) {
    return vec3<f32>(1.0e19);
  }
  return (bodyVelocity + parallel + perpendicular / gammaBody) / denominator;
}

fn invalidDiskTransfer() -> DiskTransferSample {
  var result: DiskTransferSample;
  result.radiance = vec3<f32>(0.0);
  result.opacity = 1.0;
  result.valid = 0.0;
  return result;
}

fn wrapDiskPatternAngle(angle: f32) -> f32 {
  return TWO_PI * fract(angle / TWO_PI);
}

// Bounded analytic surface modulation. A pure m=2 term with amplitude 0.16 is
// locked to the instantaneous binary axis. Five zero-mean emissivity modes
// have total amplitude 0.09 and use wrapped pattern phases calibrated to the
// Kepler frequency at the zero-torque flux peak. The wrap is continuous under
// every integer mode and prevents the radial phase from accumulating an
// unbounded time*dOmega/dr winding over the 9,330 M protocol. This is a
// deterministic finite-correlation emissivity proxy, not evolved GRMHD matter.
// It changes dissipated flux only; geometry, velocity, optical depth, metric,
// and ray paths remain unchanged. The analytic mean is exactly 1 and the
// multiplier remains in [0.75, 1.25] without a clipping bias.
fn analyticDiskSurfaceStructure(
  restPosition: vec3<f32>,
  normalInput: vec3<f32>,
  binaryAxisInput: vec3<f32>,
  radius: f32,
  innerRadius: f32,
  bodyMassM: f32
) -> f32 {
  let normal = safeNormalize(normalInput);
  let radial = restPosition / max(radius, 1.0e-6);
  var binaryAxis = binaryAxisInput
    - normal * dot(binaryAxisInput, normal);
  if (length(binaryAxis) <= 1.0e-6) {
    binaryAxis = cross(normal, vec3<f32>(1.0, 0.0, 0.0));
    if (length(binaryAxis) <= 1.0e-6) {
      binaryAxis = cross(normal, vec3<f32>(0.0, 0.0, 1.0));
    }
  }
  let axisX = safeNormalize(binaryAxis);
  let axisY = safeNormalize(cross(normal, axisX));
  let azimuth = atan2(dot(radial, axisY), dot(radial, axisX));
  let logarithmicRadius = log(max(radius / max(innerRadius, 1.0e-5), 1.0));
  let tidal = 0.16 * cos(
    2.0 * (azimuth - 0.42 * logarithmicRadius)
  );
  // Novikov-Thorne flux peak (9.5509 m for r_in = 6 m).
  let referenceRadius = 1.5918167 * innerRadius;
  let referenceRadius3 = max(
    referenceRadius * referenceRadius * referenceRadius,
    1.0e-8
  );
  let omegaPeak = sqrt(max(bodyMassM, 1.0e-6) / referenceRadius3);
  let time = params.spacetimeControl.x;
  let phase5 = wrapDiskPatternAngle(0.82 * omegaPeak * time);
  let phase9 = wrapDiskPatternAngle(0.93 * omegaPeak * time);
  let phase14 = wrapDiskPatternAngle(1.04 * omegaPeak * time);
  let phase21 = wrapDiskPatternAngle(1.13 * omegaPeak * time);
  let phase31 = wrapDiskPatternAngle(1.21 * omegaPeak * time);
  let emissivityTexture =
      0.032 * sin(5.0 * (azimuth - phase5) + 1.2 * logarithmicRadius)
    + 0.024 * sin(9.0 * (azimuth - phase9) - 2.1 * logarithmicRadius + 1.7)
    + 0.016 * sin(14.0 * (azimuth - phase14) + 3.6 * logarithmicRadius + 3.1)
    + 0.010 * sin(21.0 * (azimuth - phase21) - 5.4 * logarithmicRadius + 0.8)
    + 0.008 * sin(31.0 * (azimuth - phase31) + 7.1 * logarithmicRadius + 2.4);
  return 1.0 + tidal + emissivityTexture;
}

fn diskTransferAtIntersection(
  intersection: DiskIntersection,
  normalInput: vec3<f32>,
  binaryAxis: vec3<f32>,
  bodyVelocity: vec3<f32>,
  bodyMassM: f32,
  innerRadius: f32,
  outerRadius: f32,
  eddingtonRatio: f32,
  activeWeight: f32,
  rotationSign: f32,
  momentum: vec3<f32>,
  conservedEnergy: f32,
  observerFrequency: f32,
  capturePadding: f32
) -> DiskTransferSample {
  var invalid = invalidDiskTransfer();
  if (intersection.valid < 0.5) {
    invalid.opacity = 0.0;
    return invalid;
  }
  let fields = metricValuesAt(intersection.position);
  // Direct evaluation at the event orders a disk crossing against capture
  // without paying for an additional metric sample on segments with no hit.
  if (
    fields.valid < 0.5
    || fields.horizonDistance <= capturePadding
    || bodyMassM <= 0.0
    || eddingtonRatio <= 0.0
    || activeWeight <= 0.0
  ) {
    return invalid;
  }

  let normal = safeNormalize(normalInput);
  let radial = intersection.restPosition / max(intersection.radius, 1.0e-6);
  var tangent = rotationSign * cross(normal, radial);
  let tangentNormSquared = spatialDot(fields, tangent, tangent);
  if (!finiteScalar(tangentNormSquared) || tangentNormSquared <= 1.0e-10) {
    return invalid;
  }
  tangent = tangent * inverseSqrt(tangentNormSquared);

  // Schwarzschild circular-orbit speed measured by a local static observer.
  // The CPU model supplies r >= 6m; retaining the explicit denominator makes
  // malformed states fail rather than silently manufacture a subluminal disk.
  let orbitalDenominator = intersection.radius - 2.0 * bodyMassM;
  if (orbitalDenominator <= 0.0) {
    return invalid;
  }
  let orbitalSpeedSquared = bodyMassM / orbitalDenominator;
  if (
    !finiteScalar(orbitalSpeedSquared)
    || orbitalSpeedSquared <= 0.0
    || orbitalSpeedSquared >= 0.95
  ) {
    return invalid;
  }
  let orbitalVelocity = tangent * sqrt(orbitalSpeedSquared);
  let emitterVelocity = composeEulerianVelocities(
    fields,
    bodyVelocity,
    orbitalVelocity
  );
  let emitterSpeedSquared = spatialDot(
    fields,
    emitterVelocity,
    emitterVelocity
  );
  if (
    !finiteVector(emitterVelocity)
    || !finiteScalar(emitterSpeedSquared)
    || emitterSpeedSquared < 0.0
    || emitterSpeedSquared >= 0.999
  ) {
    return invalid;
  }
  let emitterGamma = inverseSqrt(1.0 - emitterSpeedSquared);
  let emitterTime = emitterGamma / max(fields.lapse, 1.0e-6);
  let emitterSpatial = emitterGamma * (
    emitterVelocity - fields.shift / max(fields.lapse, 1.0e-6)
  );
  let emitterFrequency = (
    conservedEnergy * emitterTime - dot(momentum, emitterSpatial)
  );
  if (
    !finiteScalar(emitterFrequency)
    || emitterFrequency <= 1.0e-6
    || !finiteScalar(observerFrequency)
    || observerFrequency <= 0.0
  ) {
    return invalid;
  }
  let rawFrequencyShift = observerFrequency / emitterFrequency;
  if (!finiteScalar(rawFrequencyShift) || rawFrequencyShift <= 0.0) {
    return invalid;
  }
  // Bound only the display-oriented RGB chromaticity. The invariant
  // bolometric amplitude below uses the unmodified frequency ratio, so g^4 is
  // never silently replaced by a clipped transfer factor.
  let chromaticFrequencyShift = clamp(rawFrequencyShift, 0.02, 8.0);

  // Novikov-Thorne (Page-Thorne) zero-torque flux around a non-spinning
  // body with the inner edge at its ISCO, normalised to the peak at
  // r = 9.5509 m. x = sqrt(r/m), x0 = sqrt(r_in/m) = sqrt(6).
  let root3 = 1.7320508;
  let ntX = sqrt(max(intersection.radius, innerRadius) / max(bodyMassM, 1.0e-6));
  let ntX0 = sqrt(innerRadius / max(bodyMassM, 1.0e-6));
  let ntBracket = ntX - ntX0 + 0.5 * root3 * log(
    ((ntX + root3) * (ntX0 - root3)) / ((ntX - root3) * (ntX0 + root3))
  );
  let ntX2 = ntX * ntX;
  let fluxShape = max(
    ntBracket / (ntX2 * ntX2 * ntX * (ntX2 - 3.0)),
    0.0
  ) / 1.1458947e-4;
  let edgeCoverage = annulusEdgeCoverage(
    intersection.radius,
    innerRadius,
    outerRadius,
    bodyMassM
  );
  let surfaceStructure = analyticDiskSurfaceStructure(
    intersection.restPosition,
    normal,
    binaryAxis,
    intersection.radius,
    innerRadius,
    bodyMassM
  );
  let structuredFluxShape = fluxShape * surfaceStructure;
  let totalMassSolar = max(params.resolutionTimeMass.w, 1.0);
  let bodyMassSolar = max(totalMassSolar * bodyMassM, 1.0);
  let thermalScale = max(params.sceneDiskControl.z, 1.0e-4);
  // Novikov-Thorne peak effective temperature for Mdot = lambda L_Edd /
  // (0.1 c^2) around the body mass.
  let peakTemperature = 1.086e5 * pow(
    eddingtonRatio * 1.0e8 / bodyMassSolar,
    0.25
  );
  let emittedTemperature = max(
    100.0,
    thermalScale * peakTemperature
      * pow(max(structuredFluxShape, 1.0e-12), 0.25)
  );

  let normalMetricNorm = sqrt(max(spatialDot(fields, normal, normal), 1.0e-10));
  let metricUnitNormal = normal / normalMetricNorm;
  let muEmitter = clamp(
    abs(dot(momentum, metricUnitNormal)) / emitterFrequency,
    0.03,
    1.0
  );
  let peakRadius = 1.5918167 * innerRadius;
  let tauPeak = max(params.sceneDiskControl.w, 0.0);
  let tauFace = tauPeak
    * pow(max(intersection.radius / max(peakRadius, 1.0e-5), 0.1), -0.60);
  let lineOfSightTau = min(tauFace / muEmitter, 30.0);
  let opaqueSurfaceFraction = clamp(
    1.0 - exp(-lineOfSightTau),
    0.0,
    1.0
  );
  // activeWeight and the C2 radial edge are covering fractions. Applying
  // their product once to opacity makes both emission and occultation fade
  // linearly instead of accidentally squaring the transition in optically
  // thin edge pixels.
  let opacity = clamp(
    activeWeight * edgeCoverage * opaqueSurfaceFraction,
    0.0,
    1.0
  );
  // Flux-normalised electron-scattering limb darkening holds for an optically
  // thick atmosphere. A thin slab is already limb-brightened by the 1/mu path
  // length in the opacity above, so its source is isotropic.
  let limbDarkening = mix(
    1.0,
    (1.0 + 2.06 * muEmitter) / 2.373,
    smoothstep(0.5, 2.0, tauFace)
  );
  let g2 = rawFrequencyShift * rawFrequencyShift;
  let bolometricTransfer = g2 * g2;
  let thermalFluxScale = thermalScale * thermalScale
    * thermalScale * thermalScale;
  // The bolometric surface flux retains T_eff^4 proportional to
  // (Mdot / M^2): for an Eddington-scaled rate this is eddingtonRatio /
  // bodyMassSolar, not rate alone. activeWeight and edgeCoverage are applied
  // once through opacity above rather than being multiplied into the source.
  let intrinsicFlux = eddingtonRatio * 1.0e8 / bodyMassSolar
    * structuredFluxShape * thermalFluxScale;
  // The fixed gain anchors a 6500 K visible blackbody to the shared linear-HDR
  // scene scale. The spectral function retains the temperature-dependent
  // visible fraction, so UV-dominated hot disks no longer map all bolometric
  // power into a featureless white surface. control.z remains a physical
  // thermal normalisation with baseline 1, not a per-scene exposure.
  let visibleSceneGain = 1800.0;
  let radiance = visibleBlackbodyLinearSrgbPerBolometric(
    emittedTemperature * chromaticFrequencyShift
  )
    * intrinsicFlux * bolometricTransfer * limbDarkening * visibleSceneGain;
  if (!finiteVector(radiance) || !finiteScalar(opacity)) {
    return invalid;
  }
  var result: DiskTransferSample;
  result.radiance = max(radiance, vec3<f32>(0.0));
  result.opacity = opacity;
  result.valid = 1.0;
  return result;
}

fn applyDiskIntersection(
  ray: RayResult,
  intersection: DiskIntersection,
  normal: vec3<f32>,
  binaryAxis: vec3<f32>,
  bodyVelocity: vec3<f32>,
  bodyMassM: f32,
  innerRadius: f32,
  diskParameters: vec4<f32>,
  momentum: vec3<f32>,
  conservedEnergy: f32,
  observerFrequency: f32,
  capturePadding: f32
) -> RayResult {
  var result = ray;
  if (intersection.valid < 0.5 || result.diskTransmittance <= 1.0e-5) {
    return result;
  }
  let sample = diskTransferAtIntersection(
    intersection,
    normal,
    binaryAxis,
    bodyVelocity,
    bodyMassM,
    innerRadius,
    diskParameters.x,
    diskParameters.y,
    diskParameters.z,
    diskParameters.w,
    momentum,
    conservedEnergy,
    observerFrequency,
    capturePadding
  );
  if (sample.valid < 0.5) {
    // A real geometric interception with invalid local transfer must block the
    // unknown background instead of becoming a fabricated transparent gap.
    result.diskTransmittance = 0.0;
    result.diskTransferFailure = 1.0;
    return result;
  }
  result.diskRadiance = result.diskRadiance
    + result.diskTransmittance * sample.radiance * sample.opacity;
  result.diskTransmittance = result.diskTransmittance * (1.0 - sample.opacity);
  return result;
}

fn accumulateDualDiskEmission(
  ray: RayResult,
  segmentStart: vec3<f32>,
  segmentEnd: vec3<f32>,
  momentumStart: vec3<f32>,
  momentumEnd: vec3<f32>,
  conservedEnergy: f32,
  observerFrequency: f32,
  capturePadding: f32
) -> RayResult {
  var result = ray;
  if (
    params.sceneDiskControl.x < 0.5
    || result.diskTransmittance <= 1.0e-5
  ) {
    return result;
  }
  let hitA = segmentDiskIntersection(
    segmentStart,
    segmentEnd,
    params.bodyAPositionMass.xyz,
    params.diskANormalInner.xyz,
    params.diskANormalInner.w,
    params.diskAOuterAccretionWeight.x,
    params.diskAOuterAccretionWeight.z
  );
  let hitB = segmentDiskIntersection(
    segmentStart,
    segmentEnd,
    params.bodyBPositionMass.xyz,
    params.diskBNormalInner.xyz,
    params.diskBNormalInner.w,
    params.diskBOuterAccretionWeight.x,
    params.diskBOuterAccretionWeight.z
  );
  let momentumA = mix(
    momentumStart,
    momentumEnd,
    clamp(hitA.fraction, 0.0, 1.0)
  );
  let momentumB = mix(
    momentumStart,
    momentumEnd,
    clamp(hitB.fraction, 0.0, 1.0)
  );
  // Apply both intersections in observer-to-source order. The first surface's
  // finite opacity attenuates the second, providing mutual occlusion without a
  // screen-space mask. Invalid entries carry fraction 2 and naturally sort last.
  let firstIsA = hitA.fraction <= hitB.fraction;
  if (firstIsA) {
    result = applyDiskIntersection(
      result, hitA, params.diskANormalInner.xyz,
      params.bodyBPositionMass.xyz - params.bodyAPositionMass.xyz,
      params.bodyAVelocityActive.xyz, params.bodyAPositionMass.w,
      params.diskANormalInner.w, params.diskAOuterAccretionWeight,
      momentumA, conservedEnergy,
      observerFrequency, capturePadding
    );
    result = applyDiskIntersection(
      result, hitB, params.diskBNormalInner.xyz,
      params.bodyAPositionMass.xyz - params.bodyBPositionMass.xyz,
      params.bodyBVelocityActive.xyz, params.bodyBPositionMass.w,
      params.diskBNormalInner.w, params.diskBOuterAccretionWeight,
      momentumB, conservedEnergy,
      observerFrequency, capturePadding
    );
  } else {
    result = applyDiskIntersection(
      result, hitB, params.diskBNormalInner.xyz,
      params.bodyAPositionMass.xyz - params.bodyBPositionMass.xyz,
      params.bodyBVelocityActive.xyz, params.bodyBPositionMass.w,
      params.diskBNormalInner.w, params.diskBOuterAccretionWeight,
      momentumB, conservedEnergy,
      observerFrequency, capturePadding
    );
    result = applyDiskIntersection(
      result, hitA, params.diskANormalInner.xyz,
      params.bodyBPositionMass.xyz - params.bodyAPositionMass.xyz,
      params.bodyAVelocityActive.xyz, params.bodyAPositionMass.w,
      params.diskANormalInner.w, params.diskAOuterAccretionWeight,
      momentumA, conservedEnergy,
      observerFrequency, capturePadding
    );
  }
  return result;
}
`;

function createStrongFieldBinaryTraceFragmentWGSL({ dualDisk = false } = {}) {
  const diskParams = dualDisk ? /* wgsl */ `
  sceneDiskControl: vec4<f32>,
  diskANormalInner: vec4<f32>,
  diskAOuterAccretionWeight: vec4<f32>,
  diskBNormalInner: vec4<f32>,
  diskBOuterAccretionWeight: vec4<f32>,` : "";
  const diskFunctions = dualDisk ? STRONG_FIELD_DUAL_DISK_FUNCTIONS_WGSL : "";
  const diskResultFields = dualDisk ? /* wgsl */ `
  diskRadiance: vec3<f32>,
  diskTransmittance: f32,
  diskTransferFailure: f32,
  // Sparse coarse nodes only: the node's beam passed within its footprint
  // of a disk annulus.
  diskProximity: f32,` : "";
  const sparseDiskFlag = dualDisk ? /* wgsl */ `
  if (
    result.diskTransmittance < 1.0
    || any(result.diskRadiance > vec3<f32>(0.0))
    || result.diskTransferFailure > 0.0
    || result.diskProximity > 0.5
  ) {
    code = code + 4.0;
  }` : "";
  const diskResultInitialization = dualDisk ? /* wgsl */ `
  result.diskRadiance = vec3<f32>(0.0);
  result.diskTransmittance = 1.0;
  result.diskTransferFailure = 0.0;
  result.diskProximity = 0.0;` : "";
  // Disk crossings are located on two chords per RK4 step through the
  // third-order continuous-extension midpoint
  //   y(1/2) = y0 + h (5/24 k1 + 1/6 k2 + 1/6 k3 - 1/24 k4),
  // with the momentum interpolated to the crossing, instead of on the raw
  // step chord. The combination is accumulated stage by stage.
  const diskMidpointDeclaration = dualDisk ? /* wgsl */ `
  var midX = vec3<f32>(0.0);
  var midP = vec3<f32>(0.0);` : "";
  const diskMidpointFromK1 = dualDisk ? /* wgsl */ `
      midX = (5.0 / 24.0) * derivativeX;
      midP = (5.0 / 24.0) * derivativeP;` : "";
  const diskMidpointAddInner = dualDisk ? /* wgsl */ `
      midX = midX + (1.0 / 6.0) * derivativeX;
      midP = midP + (1.0 / 6.0) * derivativeP;` : "";
  const diskStep = dualDisk ? /* wgsl */ `
    let midPosition = stepPosition
      + stepSize * (midX - (1.0 / 24.0) * derivativeX);
    let midMomentum = stepMomentum
      + stepSize * (midP - (1.0 / 24.0) * derivativeP);
    result = accumulateDualDiskEmission(
      result,
      stepPosition,
      midPosition,
      stepMomentum,
      midMomentum,
      conservedEnergy,
      observerQ,
      capturePadding
    );
    result = accumulateDualDiskEmission(
      result,
      midPosition,
      nextPosition,
      midMomentum,
      nextMomentum,
      conservedEnergy,
      observerQ,
      capturePadding
    );
    if (footprintAngle > 0.0 && result.diskProximity < 0.5) {
      let footprint = footprintAngle * (lookback + stepSize);
      if (
        chordNearDisks(stepPosition, midPosition, footprint)
        || chordNearDisks(midPosition, nextPosition, footprint)
      ) {
        result.diskProximity = 1.0;
      }
    }` : "";
  const capturedPhotographicResult = dualDisk
    ? "return result.diskRadiance;"
    : "return vec3<f32>(0.0);";
  const unresolvedPhotographicResult = dualDisk ? /* wgsl */ `return result.diskRadiance
      + result.diskTransmittance * vec3<f32>(0.050, 0.036, 0.024)
      * unresolvedLevel * hatch;`
    : /* wgsl */ `return vec3<f32>(0.050, 0.036, 0.024)
      * unresolvedLevel * hatch;`;
  const escapedPhotographicResult = dualDisk ? /* wgsl */ `return result.diskRadiance
    + result.diskTransmittance * sampleEnvironment(result.escapeDirection)
      * shiftRadiance;`
    : "return sampleEnvironment(result.escapeDirection) * shiftRadiance;";
  return /* wgsl */ `
diagnostic(off, derivative_uniformity);

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const MAX_RK4_STEPS: i32 = 192;
// Outbound rays beyond this radius only move into weaker field, so their RK4
// step fraction grows by FAR_FIELD_GROWTH (capped at FAR_FIELD_MAXIMUM_FRACTION)
// without losing accuracy. Inbound rays keep the tier fraction: a large step
// set from the start-of-step distance would overshoot into the strong field.
const FAR_FIELD_RADIUS_M: f32 = 30.0;
const FAR_FIELD_GROWTH: f32 = 2.0;
const FAR_FIELD_MAXIMUM_FRACTION: f32 = 2.0;
// Sparse tracing: coarse nodes every SPARSE_STRIDE pixels. A pixel is
// interpolated only if the bilinear error bound from the node second
// differences stays below SPARSE_CURVATURE_PIXELS.
const SPARSE_STRIDE: f32 = ${STRONG_FIELD_SPARSE_STRIDE.toFixed(1)};
const SPARSE_CURVATURE_PIXELS: f32 = 0.25;
// A coarse node also stands for the beam around its ray, of this many node
// spacings in angular half-width (a pixel is within 0.71 spacings of a node of
// its cell; the margin absorbs mild lensing distortion of the beam). A disk
// thinner than the node spacing, e.g. seen edge-on, is caught through it.
const SPARSE_FOOTPRINT_SPACINGS: f32 = 2.0;
const RAY_UNRESOLVED: u32 = 0u;
const RAY_CAPTURED: u32 = 1u;
const RAY_ESCAPED: u32 = 2u;
// Render pipelines specialize this override to 0=binary, 1=remnant, or
// 2=transition.  The default keeps the complete provider available to the GPU
// probe and any consumer that does not opt into pipeline specialization.
override SPACETIME_PHASE_MODE: i32 = -1;
// 0 when both binary bodies are non-spinning (as in SXS:BBH:0001). The Kerr
// branch of their terms is then compiled out; although never taken, it
// otherwise costs the binary pipeline ~40% of its speed in register pressure.
override BINARY_SPIN_MODE: i32 = 1;
// Positions are frozen for the whole ray (fast light), so each hole enters the
// metric as an unboosted Kerr-Schild term. A Lorentz-boosted term is a vacuum
// solution only while its centre moves as X0 + v t; frozen, it is not, and its
// trailing null surface moves out to r+ gamma^2 (1 + v)^2, beyond the photon
// orbit for v > 0.2. Body velocities still drive the disk-matter kinematics.
const FROZEN_METRIC_VELOCITY: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

struct Params {
  resolutionTimeMass: vec4<f32>,
  renderControls: vec4<f32>,
  cameraPosRadius: vec4<f32>,
  cameraForwardFov: vec4<f32>,
  cameraRightSkyRotation: vec4<f32>,
  cameraUpDiskOuter: vec4<f32>,
  postDisplayFrame: vec4<f32>,
  observerVelocityBeta: vec4<f32>,
  displayOutput: vec4<f32>,
  spacetimeControl: vec4<f32>,
  bodyAPositionMass: vec4<f32>,
  bodyAVelocityActive: vec4<f32>,
  bodyASpin: vec4<f32>,
  bodyBPositionMass: vec4<f32>,
  bodyBVelocityActive: vec4<f32>,
  bodyBSpin: vec4<f32>,
  remnantPositionMass: vec4<f32>,
  remnantVelocityActive: vec4<f32>,
  remnantSpinBlend: vec4<f32>,
  spacetimeLimits: vec4<f32>,
  sceneStrongIntegrator: vec4<f32>,
  sceneStrongDomain: vec4<f32>,
  sceneStrongDiagnostics: vec4<f32>,
  sceneStrongQuality: vec4<f32>,${diskParams}
};

struct FragmentInput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

// Stable provider boundary for PR6 analytic slow-light and PR7 NR bricks.
// PR5 passes the frame time for every sample (fast-light).
struct SpacetimeProviderInput {
  coordinateTime: f32,
  position: vec3<f32>,
};

struct Dual3 {
  value: f32,
  gradient: vec3<f32>,
};

// One weighted Kerr-Schild term of the frozen metric at a point,
//   g_mu_nu += c l_mu l_nu,   l = (1, n),   |n| = 1,   c = 2 w H,
// with gradients taken at the evaluation point. The Jacobian dn/dx is applied
// in closed form (see termJacobianTranspose); for a spinning term it needs
// jacobianW, spin and inverseNormalization = 1/(r^2 + a^2).
struct KerrSchildTerm {
  c: f32,
  cGradient: vec3<f32>,
  n: vec3<f32>,
  jacobianW: vec3<f32>,
  spin: vec3<f32>,
  inverseNormalization: f32,
  inverseRadius: f32,
  spinning: f32,
  kerrRadius: f32,
  kerrRadiusGradient: vec3<f32>,
  regularized: f32,
};

// The weighted terms of the current frame: A and B are the binary, R is the
// remnant. Phase 0 = binary (A, B), 1 = remnant (R), 2 = transition (A, B, R).
struct SpacetimeTerms {
  a: KerrSchildTerm,
  b: KerrSchildTerm,
  r: KerrSchildTerm,
  weightA: f32,
  weightB: f32,
  weightR: f32,
};

// Horizon and innermost photon-orbit radii of the three terms. They depend
// only on frame uniforms, so a ray computes them once.
struct SpacetimeRadii {
  horizonA: f32,
  photonA: f32,
  horizonB: f32,
  photonB: f32,
  horizonR: f32,
  photonR: f32,
};

// Reduced Hamiltonian H(x,p) = -p_t of the null cone and its flow
// (dx/dt, dp/dt) at one point, from the low-rank form of the inverse metric.
struct LowRankHamiltonian {
  velocity: vec3<f32>,
  momentumRate: vec3<f32>,
  hamiltonian: f32,
  metricValid: f32,
  valid: f32,
};

struct GeodesicSample {
  velocity: vec3<f32>,
  momentumRate: vec3<f32>,
  hamiltonian: f32,
  horizonDistance: f32,
  // Distance inside (<0) or outside (>0) the innermost photon orbit of the
  // nearest active term, and the gradient of that term's Kerr radius.
  photonMargin: f32,
  photonRadialGradient: vec3<f32>,
  // RK4 step scale: distance to the nearest horizon, but never below half the
  // Kerr radius. Ingoing Kerr-Schild data are smooth at the horizon, so the
  // orbit scale r, not r - r+, limits accuracy near prograde photon orbits.
  stepDistance: f32,
  // Distance inside (<0) the unscaled innermost photon orbit of any term
  // present in the metric. During the merger blend the region around the
  // merging holes lies inside the common event horizon, so an unrecoverable
  // sample there is classified as captured.
  failureCaptureMargin: f32,
  // Weighted Kerr-Schild fields summed along their null directions,
  // |sum 2 w H n| (see hover capture).
  alignedField: f32,
  metricValid: f32,
  valid: f32,
};

// Metric values without derivatives, for the camera frame and disk hits.
struct MetricValues {
  gtt: f32,
  shiftCovariant: vec3<f32>,
  spatialMetric: mat3x3<f32>,
  lapse: f32,
  shift: vec3<f32>,
  horizonDistance: f32,
  valid: f32,
};

struct RayResult {
  outcome: u32,
  escapeDirection: vec3<f32>,
  frequencyShift: f32,
  lookback: f32,
  hamiltonianResidual: f32,
  iterations: f32,
  minimumHorizonDistance: f32,
  terminationReason: f32,${diskResultFields}
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tSky: texture_2d<f32>;
@group(0) @binding(2) var skySampler: sampler;
// Sparse tracing: escape directions of every S-th pixel (see fsCoarse).
@group(0) @binding(3) var coarseField: texture_2d<f32>;

fn safeNormalize(value: vec3<f32>) -> vec3<f32> {
  return value * inverseSqrt(max(dot(value, value), 1.0e-18));
}

fn finiteScalar(value: f32) -> bool {
  // NaN is the only IEEE-754 value unequal to itself; the magnitude bound also
  // rejects infinities without relying on optional classification builtins.
  return value == value && abs(value) < 1.0e18;
}

fn finiteVector(value: vec3<f32>) -> bool {
  return all(vec3<bool>(
    finiteScalar(value.x),
    finiteScalar(value.y),
    finiteScalar(value.z)
  ));
}

fn zeroKerrSchildTerm() -> KerrSchildTerm {
  var term: KerrSchildTerm;
  term.c = 0.0;
  term.cGradient = vec3<f32>(0.0);
  term.n = vec3<f32>(0.0);
  term.jacobianW = vec3<f32>(0.0);
  term.spin = vec3<f32>(0.0);
  term.inverseNormalization = 0.0;
  term.inverseRadius = 0.0;
  term.spinning = 0.0;
  term.kerrRadius = 1.0e6;
  term.kerrRadiusGradient = vec3<f32>(0.0);
  term.regularized = 0.0;
  return term;
}

// Horizon radius m(1 + sqrt(1 - chi^2)) and innermost (prograde equatorial)
// circular photon orbit r = 2m[1 + cos(2/3 acos(-chi))] (3m at chi = 0, m at
// chi = 1) of an isolated Kerr hole.
fn kerrRadii(massInput: f32, dimensionlessSpin: vec3<f32>) -> vec2<f32> {
  let mass = max(massInput, 1.0e-5);
  let chi = min(0.998, length(dimensionlessSpin));
  return vec2<f32>(
    mass * (1.0 + sqrt(max(1.0 - chi * chi, 1.0e-5))),
    2.0 * mass * (1.0 + cos((2.0 / 3.0) * acos(-chi)))
  );
}

fn spacetimeRadii() -> SpacetimeRadii {
  let radiiA = kerrRadii(params.bodyAPositionMass.w, params.bodyASpin.xyz);
  let radiiB = kerrRadii(params.bodyBPositionMass.w, params.bodyBSpin.xyz);
  let radiiR = kerrRadii(params.remnantPositionMass.w, params.remnantSpinBlend.xyz);
  var radii: SpacetimeRadii;
  radii.horizonA = radiiA.x;
  radii.photonA = radiiA.y;
  radii.horizonB = radiiB.x;
  radii.photonB = radiiB.y;
  radii.horizonR = radiiR.x;
  radii.photonR = radiiR.y;
  return radii;
}

// One unweighted (c = 2H) Kerr-Schild term. Positions are frozen for the
// whole ray, so the term is unboosted (see FROZEN_METRIC_VELOCITY).
fn kerrSchildTerm(
  position: vec3<f32>,
  centre: vec3<f32>,
  massInput: f32,
  dimensionlessSpin: vec3<f32>,
  activeFlag: f32,
  allowSpin: bool
) -> KerrSchildTerm {
  if (activeFlag < 0.5 || massInput <= 0.0) {
    return zeroKerrSchildTerm();
  }
  let mass = max(massInput, 1.0e-5);
  let spinNorm = length(dimensionlessSpin);
  let radiusFloor = max(
    params.spacetimeLimits.z,
    params.spacetimeControl.w * mass
  );
  let maximumH = max(params.spacetimeLimits.w, 1.0);
  var term = zeroKerrSchildTerm();
  term.regularized = select(0.0, 1.0, spinNorm >= 0.999);

  if (!allowSpin || spinNorm < 1.0e-5) {
    // Schwarzschild: r = |x|, H = m/r, n = x/r, dn/dx = (I - n n^T)/r and
    // dH/dx = -H n/r, all in closed form.
    let displacement = position - centre;
    let radiusSquared = dot(displacement, displacement);
    let inverseCartesian = inverseSqrt(max(radiusSquared, 1.0e-24));
    var radius = radiusSquared * inverseCartesian;
    var inverseRadius = inverseCartesian;
    term.n = displacement * inverseCartesian;
    var clamped = false;
    if (radius < radiusFloor) {
      radius = radiusFloor;
      inverseRadius = 1.0 / radiusFloor;
      clamped = true;
      term.regularized = 1.0;
    }
    term.inverseRadius = inverseRadius;
    term.kerrRadius = radius;
    term.kerrRadiusGradient = term.n;
    var kerrH = mass * inverseRadius;
    var hGradient = -kerrH * inverseRadius * term.n;
    if (clamped) {
      hGradient = vec3<f32>(0.0);
    }
    if (kerrH > maximumH) {
      kerrH = maximumH;
      hGradient = vec3<f32>(0.0);
      term.regularized = 1.0;
    }
    term.c = 2.0 * kerrH;
    term.cGradient = 2.0 * hGradient;
    return term;
  }

  // Kerr: the spheroidal radius solves r^4 - (rho^2 - a^2) r^2 - (a.x)^2 = 0,
  // H = m r^3 / W with W = r^4 + (a.x)^2, and
  // n = [r x + x cross a + a (a.x)/r] / (r^2 + a^2), which is unit exactly.
  // Implicit differentiation gives grad r = r (r^2 x + (a.x) a) / W and
  // grad H = m r^2 [(3 (a.x)^2 - r^4) grad r - 2 r (a.x) a] / W^2.
  let displacement = position - centre;
  let spin = dimensionlessSpin * (mass * min(0.998, spinNorm) / max(spinNorm, 1.0e-12));
  let spinSquared = dot(spin, spin);
  let spinDotPosition = dot(spin, displacement);
  let reduced = dot(displacement, displacement) - spinSquared;
  var radiusSquared = 0.5 * (
    reduced + sqrt(reduced * reduced + 4.0 * spinDotPosition * spinDotPosition)
  );
  var radius = sqrt(max(radiusSquared, 0.0));
  var clamped = false;
  if (radius < radiusFloor) {
    radius = radiusFloor;
    radiusSquared = radius * radius;
    clamped = true;
    term.regularized = 1.0;
  }
  let radiusFourth = radiusSquared * radiusSquared;
  let inverseW = 1.0 / (radiusFourth + spinDotPosition * spinDotPosition);
  let radiusGradient = (radius * inverseW)
    * (radiusSquared * displacement + spinDotPosition * spin);
  var kerrH = mass * radius * radiusSquared * inverseW;
  var hGradient = (mass * radiusSquared * inverseW * inverseW) * (
    (3.0 * spinDotPosition * spinDotPosition - radiusFourth) * radiusGradient
    - (2.0 * radius * spinDotPosition) * spin
  );
  if (clamped) {
    hGradient = vec3<f32>(0.0);
  }
  if (kerrH > maximumH) {
    kerrH = maximumH;
    hGradient = vec3<f32>(0.0);
    term.regularized = 1.0;
  }
  let inverseRadius = 1.0 / radius;
  let inverseNormalization = 1.0 / (radiusSquared + spinSquared);
  term.n = (
    radius * displacement
    + cross(displacement, spin)
    + (spinDotPosition * inverseRadius) * spin
  ) * inverseNormalization;
  if (clamped) {
    // Off the exact Kerr radius n is no longer unit; the low-rank form
    // assumes null l, so restore |n| = 1 (the sample is regularized anyway).
    term.n = safeNormalize(term.n);
  }
  term.jacobianW = displacement
    - (spinDotPosition * inverseRadius * inverseRadius) * spin
    - (2.0 * radius) * term.n;
  term.spin = spin;
  term.inverseNormalization = inverseNormalization;
  term.c = 2.0 * kerrH;
  term.cGradient = 2.0 * hGradient;
  term.inverseRadius = inverseRadius;
  term.spinning = 1.0;
  term.kerrRadius = radius;
  term.kerrRadiusGradient = radiusGradient;
  return term;
}

// J^T v for a term, i.e. the gradient of n . v with v held fixed. From
//   dn/dx^j = [(dr/dx^j) w + r e_j + e_j cross a + (a_j / r) a] / (r^2 + a^2),
//   w = x - (a.x) a / r^2 - 2 r n,
// J^T v = [(w.v) grad r + r v + a cross v + (a.v / r) a] / (r^2 + a^2); for
// a = 0 this is (v - n (n.v)) / r.
fn termJacobianTranspose(
  term: KerrSchildTerm,
  value: vec3<f32>,
  allowSpin: bool
) -> vec3<f32> {
  if (allowSpin && term.spinning > 0.5) {
    return (
      dot(term.jacobianW, value) * term.kerrRadiusGradient
      + term.kerrRadius * value
      + cross(term.spin, value)
      + (dot(term.spin, value) * term.inverseRadius) * term.spin
    ) * term.inverseNormalization;
  }
  return (value - term.n * dot(term.n, value)) * term.inverseRadius;
}

fn scaleTerm(term: KerrSchildTerm, weight: f32) -> KerrSchildTerm {
  var result = term;
  result.c = term.c * weight;
  result.cGradient = term.cGradient * weight;
  return result;
}

// Optional companion attenuation A = 1 - exp[-(r_companion / sigma)^p]
// (disabled by default), with its gradient.
fn companionAttenuation(position: vec3<f32>, companionCentre: vec3<f32>) -> Dual3 {
  let displacement = position - companionCentre;
  let radius = max(length(displacement), 1.0e-6);
  let scale = max(params.spacetimeControl.z, 1.0e-5);
  let power = max(params.spacetimeLimits.y, 2.0);
  let ratio = radius / scale;
  let exponential = exp(-pow(ratio, power));
  var result: Dual3;
  result.value = 1.0 - exponential;
  result.gradient = exponential * power
    * pow(max(ratio, 1.0e-8), power - 1.0) / scale
    * (displacement / radius);
  return result;
}

fn attenuateTerm(term: KerrSchildTerm, attenuation: Dual3) -> KerrSchildTerm {
  var result = term;
  result.c = term.c * attenuation.value;
  result.cGradient = term.cGradient * attenuation.value
    + term.c * attenuation.gradient;
  return result;
}

// The endpoint phase is frame-uniform. Render pipelines specialize it, so the
// inactive Kerr-Schild providers are removed at compile time.
fn spacetimePhase() -> i32 {
  let blend = clamp(params.spacetimeControl.y, 0.0, 1.0);
  if (SPACETIME_PHASE_MODE == 1 || (SPACETIME_PHASE_MODE < 0 && blend == 1.0)) {
    return 1;
  }
  if (SPACETIME_PHASE_MODE == 0 || (SPACETIME_PHASE_MODE < 0 && blend == 0.0)) {
    return 0;
  }
  return 2;
}

// Weighted terms at a point:
//   g = eta + (1 - w)[A_A 2H_A l_A l_A + A_B 2H_B l_B l_B] + w 2H_R l_R l_R,
// with w = C2 smootherstep(mergerBlend) and the optional attenuations A.
fn spacetimeTerms(position: vec3<f32>, phase: i32) -> SpacetimeTerms {
  var terms: SpacetimeTerms;
  terms.a = zeroKerrSchildTerm();
  terms.b = zeroKerrSchildTerm();
  terms.r = zeroKerrSchildTerm();
  terms.weightA = 0.0;
  terms.weightB = 0.0;
  terms.weightR = 0.0;
  let blend = clamp(params.spacetimeControl.y, 0.0, 1.0);
  if (phase == 1) {
    terms.weightR = params.remnantVelocityActive.w;
    terms.r = scaleTerm(
      kerrSchildTerm(
        position,
        params.remnantPositionMass.xyz,
        params.remnantPositionMass.w,
        params.remnantSpinBlend.xyz,
        params.remnantVelocityActive.w,
        true
      ),
      terms.weightR
    );
    return terms;
  }
  var binaryWeight = 1.0;
  if (phase == 2) {
    binaryWeight = select(0.0, 1.0 - blend, 1.0 - blend > 1.0e-6);
    let remnantWeight = select(0.0, blend, blend > 1.0e-6);
    terms.weightR = remnantWeight * params.remnantVelocityActive.w;
    terms.r = scaleTerm(
      kerrSchildTerm(
        position,
        params.remnantPositionMass.xyz,
        params.remnantPositionMass.w,
        params.remnantSpinBlend.xyz,
        params.remnantVelocityActive.w * select(0.0, 1.0, remnantWeight > 0.0),
        true
      ),
      terms.weightR
    );
  }
  let binaryActive = select(0.0, 1.0, binaryWeight > 0.0);
  terms.weightA = binaryWeight * params.bodyAVelocityActive.w;
  terms.weightB = binaryWeight * params.bodyBVelocityActive.w;
  var termA = kerrSchildTerm(
    position,
    params.bodyAPositionMass.xyz,
    params.bodyAPositionMass.w,
    params.bodyASpin.xyz,
    params.bodyAVelocityActive.w * binaryActive,
    BINARY_SPIN_MODE != 0
  );
  var termB = kerrSchildTerm(
    position,
    params.bodyBPositionMass.xyz,
    params.bodyBPositionMass.w,
    params.bodyBSpin.xyz,
    params.bodyBVelocityActive.w * binaryActive,
    BINARY_SPIN_MODE != 0
  );
  if (params.spacetimeLimits.x >= 0.5) {
    let attenuationA = companionAttenuation(position, params.bodyBPositionMass.xyz);
    let attenuationB = companionAttenuation(position, params.bodyAPositionMass.xyz);
    termA = attenuateTerm(termA, attenuationA);
    termB = attenuateTerm(termB, attenuationB);
    terms.weightA = terms.weightA * attenuationA.value;
    terms.weightB = terms.weightB * attenuationB.value;
  }
  terms.a = scaleTerm(termA, binaryWeight * params.bodyAVelocityActive.w);
  terms.b = scaleTerm(termB, binaryWeight * params.bodyBVelocityActive.w);
  return terms;
}

// Future-directed root of the null condition (1 + k) E^2 + 2 m E = P - q,
// chosen in the form that avoids cancellation. sqrtDiscriminant equals
// g^{t nu} p_nu = q / alpha > 0.
fn nullEnergy(
  onePlusK: f32,
  m: f32,
  pq: f32,
  sqrtDiscriminant: f32
) -> f32 {
  return select(
    (sqrtDiscriminant - m) / onePlusK,
    pq / (sqrtDiscriminant + m),
    m > 0.0
  );
}

fn lowRankValidity(
  result: ptr<function, LowRankHamiltonian>,
  capacitanceDeterminant: f32,
  onePlusK: f32,
  discriminant: f32
) {
  // det(I + N C) > 0 is the Lorentzian condition; alpha^2 = 1/(1 + k) >
  // 1e-8 keeps the previous lapse threshold.
  (*result).metricValid = select(
    0.0,
    1.0,
    capacitanceDeterminant > 1.0e-6 && onePlusK > 1.0e-6 && onePlusK < 1.0e8
  );
  // With these bounds every division above is bounded away from zero; the
  // RK4 step still checks the accumulated state for non-finite values.
  (*result).valid = select(
    0.0,
    1.0,
    (*result).metricValid > 0.5 && discriminant > 1.0e-12
  );
}

// Low-rank form of the reduced null Hamiltonian. With g = eta + U C U^T
// (columns l_a, C = diag(c_a)), Woodbury gives g^-1 = eta - L K L^T with
// L_a = (-1, n_a), K = C (I + N C)^-1 and N_ab = n_a . n_b - 1 (N_aa = 0);
// det(I + N C) > 0 exactly when g is Lorentzian. For p = (-E, p_vec) and
// u_a = E + n_a . p_vec the null condition reads -E^2 + |p|^2 - u^T K u = 0,
// a quadratic in E. The flow is
//   dx/dt = (p - sum_a y_a n_a) / sqrt(D),   dp/dt = grad(u^T K u) / (2 sqrt(D))
// with y = K u, z = (I + N C)^-1 u and, for fixed E and p,
//   d(u^T K u) = 2 y^T du + z^T dC z - y^T dN y,
// so only per-term scalar gradients are needed.
fn lowRank1(term: KerrSchildTerm, momentum: vec3<f32>) -> LowRankHamiltonian {
  let v = dot(term.n, momentum);
  let c = term.c;
  let onePlusK = 1.0 + c;
  let m = c * v;
  let pq = dot(momentum, momentum) - c * v * v;
  let discriminant = m * m + onePlusK * pq;
  let sqrtDiscriminant = sqrt(max(discriminant, 1.0e-30));
  let energy = nullEnergy(onePlusK, m, pq, sqrtDiscriminant);
  let u = energy + v;
  let y = c * u;
  let inverseTime = 1.0 / sqrtDiscriminant;
  var result: LowRankHamiltonian;
  result.hamiltonian = energy;
  result.velocity = (momentum - y * term.n) * inverseTime;
  result.momentumRate = (
    2.0 * y * termJacobianTranspose(term, momentum, true)
    + u * u * term.cGradient
  ) * (0.5 * inverseTime);
  lowRankValidity(&result, 1.0, onePlusK, discriminant);
  return result;
}

fn lowRank2(
  termA: KerrSchildTerm,
  termB: KerrSchildTerm,
  momentum: vec3<f32>
) -> LowRankHamiltonian {
  let va = dot(termA.n, momentum);
  let vb = dot(termB.n, momentum);
  let coupling = dot(termA.n, termB.n) - 1.0;
  let ca = termA.c;
  let cb = termB.c;
  // A = [[1, N cb], [N ca, 1]]; A z = r gives z = [r_a - N cb r_b, r_b - N ca r_a] / det.
  let determinant = 1.0 - coupling * coupling * ca * cb;
  let inverseDeterminant = 1.0 / determinant;
  let onesA = (1.0 - coupling * cb) * inverseDeterminant;
  let onesB = (1.0 - coupling * ca) * inverseDeterminant;
  let projectedA = (va - coupling * cb * vb) * inverseDeterminant;
  let projectedB = (vb - coupling * ca * va) * inverseDeterminant;
  let onePlusK = 1.0 + ca * onesA + cb * onesB;
  let m = ca * projectedA + cb * projectedB;
  let pq = dot(momentum, momentum)
    - (ca * projectedA * va + cb * projectedB * vb);
  let discriminant = m * m + onePlusK * pq;
  let sqrtDiscriminant = sqrt(max(discriminant, 1.0e-30));
  let energy = nullEnergy(onePlusK, m, pq, sqrtDiscriminant);
  let za = energy * onesA + projectedA;
  let zb = energy * onesB + projectedB;
  let ya = ca * za;
  let yb = cb * zb;
  let inverseTime = 1.0 / sqrtDiscriminant;
  var result: LowRankHamiltonian;
  result.hamiltonian = energy;
  result.velocity = (momentum - ya * termA.n - yb * termB.n) * inverseTime;
  let couplingGradient = termJacobianTranspose(termA, termB.n, BINARY_SPIN_MODE != 0)
    + termJacobianTranspose(termB, termA.n, BINARY_SPIN_MODE != 0);
  result.momentumRate = (
    2.0 * (
      ya * termJacobianTranspose(termA, momentum, BINARY_SPIN_MODE != 0)
      + yb * termJacobianTranspose(termB, momentum, BINARY_SPIN_MODE != 0)
    )
    + za * za * termA.cGradient
    + zb * zb * termB.cGradient
    - 2.0 * ya * yb * couplingGradient
  ) * (0.5 * inverseTime);
  lowRankValidity(&result, determinant, onePlusK, discriminant);
  return result;
}

fn lowRank3(
  termA: KerrSchildTerm,
  termB: KerrSchildTerm,
  termR: KerrSchildTerm,
  momentum: vec3<f32>
) -> LowRankHamiltonian {
  let v = vec3<f32>(
    dot(termA.n, momentum),
    dot(termB.n, momentum),
    dot(termR.n, momentum)
  );
  let c = vec3<f32>(termA.c, termB.c, termR.c);
  let nab = dot(termA.n, termB.n) - 1.0;
  let nar = dot(termA.n, termR.n) - 1.0;
  let nbr = dot(termB.n, termR.n) - 1.0;
  // A_ij = delta_ij + N_ij c_j; rows are the equations.
  let m01 = nab * c.y;
  let m02 = nar * c.z;
  let m10 = nab * c.x;
  let m12 = nbr * c.z;
  let m20 = nar * c.x;
  let m21 = nbr * c.y;
  let a00 = 1.0 - m12 * m21;
  let a01 = m02 * m21 - m01;
  let a02 = m01 * m12 - m02;
  let a10 = m12 * m20 - m10;
  let a11 = 1.0 - m02 * m20;
  let a12 = m02 * m10 - m12;
  let a20 = m10 * m21 - m20;
  let a21 = m01 * m20 - m21;
  let a22 = 1.0 - m01 * m10;
  let determinant = a00 + m01 * a10 + m02 * a20;
  let inverseDeterminant = 1.0 / determinant;
  let adjugateRow0 = vec3<f32>(a00, a01, a02);
  let adjugateRow1 = vec3<f32>(a10, a11, a12);
  let adjugateRow2 = vec3<f32>(a20, a21, a22);
  let ones = vec3<f32>(
    adjugateRow0.x + adjugateRow0.y + adjugateRow0.z,
    adjugateRow1.x + adjugateRow1.y + adjugateRow1.z,
    adjugateRow2.x + adjugateRow2.y + adjugateRow2.z
  ) * inverseDeterminant;
  let projected = vec3<f32>(
    dot(adjugateRow0, v),
    dot(adjugateRow1, v),
    dot(adjugateRow2, v)
  ) * inverseDeterminant;
  let onePlusK = 1.0 + dot(c, ones);
  let m = dot(c, projected);
  let pq = dot(momentum, momentum) - dot(c * projected, v);
  let discriminant = m * m + onePlusK * pq;
  let sqrtDiscriminant = sqrt(max(discriminant, 1.0e-30));
  let energy = nullEnergy(onePlusK, m, pq, sqrtDiscriminant);
  let z = energy * ones + projected;
  let y = c * z;
  let inverseTime = 1.0 / sqrtDiscriminant;
  var result: LowRankHamiltonian;
  result.hamiltonian = energy;
  result.velocity = (
    momentum - y.x * termA.n - y.y * termB.n - y.z * termR.n
  ) * inverseTime;
  let couplingAB = termJacobianTranspose(termA, termB.n, BINARY_SPIN_MODE != 0)
    + termJacobianTranspose(termB, termA.n, BINARY_SPIN_MODE != 0);
  let couplingAR = termJacobianTranspose(termA, termR.n, BINARY_SPIN_MODE != 0)
    + termJacobianTranspose(termR, termA.n, true);
  let couplingBR = termJacobianTranspose(termB, termR.n, BINARY_SPIN_MODE != 0)
    + termJacobianTranspose(termR, termB.n, true);
  result.momentumRate = (
    2.0 * (
      y.x * termJacobianTranspose(termA, momentum, BINARY_SPIN_MODE != 0)
      + y.y * termJacobianTranspose(termB, momentum, BINARY_SPIN_MODE != 0)
      + y.z * termJacobianTranspose(termR, momentum, true)
    )
    + z.x * z.x * termA.cGradient
    + z.y * z.y * termB.cGradient
    + z.z * z.z * termR.cGradient
    - 2.0 * (
      y.x * y.y * couplingAB
      + y.x * y.z * couplingAR
      + y.y * y.z * couplingBR
    )
  ) * (0.5 * inverseTime);
  lowRankValidity(&result, determinant, onePlusK, discriminant);
  return result;
}

fn addTermGeometry(
  sample: ptr<function, GeodesicSample>,
  term: KerrSchildTerm,
  weight: f32,
  horizonRadius: f32,
  photonRadius: f32
) {
  // Capture geometry uses weight-scaled radii. Near its own centre a
  // Kerr-Schild term with metric weight w acts like a hole of mass w m, so its
  // horizon and innermost photon orbit grow continuously from zero during the
  // merger transition instead of switching on at an arbitrary weight.
  if (weight > 1.0e-4) {
    let w = min(weight, 1.0);
    let horizonDistance = term.kerrRadius - w * horizonRadius;
    (*sample).horizonDistance = min((*sample).horizonDistance, horizonDistance);
    (*sample).stepDistance = min(
      (*sample).stepDistance,
      max(horizonDistance, 0.5 * term.kerrRadius)
    );
    let margin = term.kerrRadius - w * photonRadius;
    if (margin < (*sample).photonMargin) {
      (*sample).photonMargin = margin;
      (*sample).photonRadialGradient = term.kerrRadiusGradient;
    }
  }
  // Failure capture covers every term present in the metric, however small
  // its weight: early in the merger blend the remnant's ring singularity can
  // break the superposition near the centre of mass while w < 1e-4.
  if (weight > 0.0) {
    (*sample).failureCaptureMargin = min(
      (*sample).failureCaptureMargin,
      term.kerrRadius - photonRadius
    );
  }
}

// Reduced 3+1 null Hamiltonian H(x,p) = alpha sqrt(gamma^ij p_i p_j) - beta^i p_i
// = -p_t and its flow at one point, evaluated through the low-rank form of the
// superposed Kerr-Schild metric (one provider evaluation per RK4 stage).
fn evaluateGeodesic(
  position: vec3<f32>,
  momentum: vec3<f32>,
  radii: SpacetimeRadii
) -> GeodesicSample {
  let phase = spacetimePhase();
  let terms = spacetimeTerms(position, phase);
  var flow: LowRankHamiltonian;
  if (phase == 1) {
    flow = lowRank1(terms.r, momentum);
  } else if (phase == 0) {
    flow = lowRank2(terms.a, terms.b, momentum);
  } else {
    flow = lowRank3(terms.a, terms.b, terms.r, momentum);
  }
  var sample: GeodesicSample;
  sample.velocity = flow.velocity;
  sample.momentumRate = flow.momentumRate;
  sample.hamiltonian = flow.hamiltonian;
  sample.horizonDistance = 1.0e6;
  sample.photonMargin = 1.0e6;
  sample.photonRadialGradient = vec3<f32>(0.0);
  sample.stepDistance = 1.0e6;
  sample.failureCaptureMargin = 1.0e6;
  addTermGeometry(&sample, terms.a, terms.weightA, radii.horizonA, radii.photonA);
  addTermGeometry(&sample, terms.b, terms.weightB, radii.horizonB, radii.photonB);
  addTermGeometry(&sample, terms.r, terms.weightR, radii.horizonR, radii.photonR);
  sample.alignedField = length(
    terms.a.c * terms.a.n + terms.b.c * terms.b.n + terms.r.c * terms.r.n
  );
  let activeRegularization = max(
    max(terms.weightA * terms.a.regularized, terms.weightB * terms.b.regularized),
    terms.weightR * terms.r.regularized
  );
  sample.metricValid = select(
    0.0,
    1.0,
    flow.metricValid > 0.5 && activeRegularization < 0.5
  );
  sample.valid = select(
    0.0,
    1.0,
    flow.valid > 0.5 && activeRegularization < 0.5
  );
  return sample;
}

fn termOuter(term: KerrSchildTerm) -> mat3x3<f32> {
  return mat3x3<f32>(
    term.n * (term.c * term.n.x),
    term.n * (term.c * term.n.y),
    term.n * (term.c * term.n.z)
  );
}

// Lapse, shift and spatial metric at a point (no derivatives). The covariant
// metric is g_tt = -1 + sum c_a, g_ti = sum c_a n_a, g_ij = delta_ij + sum
// c_a n_a n_a; the inverse follows from the same capacitance system, with
// alpha = 1/sqrt(1 + k) and beta^i = sum_a (K 1)_a n_a / (1 + k).
fn metricValuesAt(position: vec3<f32>) -> MetricValues {
  let phase = spacetimePhase();
  let terms = spacetimeTerms(position, phase);
  let radii = spacetimeRadii();
  var values: MetricValues;
  values.gtt = -1.0 + terms.a.c + terms.b.c + terms.r.c;
  values.shiftCovariant = terms.a.c * terms.a.n
    + terms.b.c * terms.b.n
    + terms.r.c * terms.r.n;
  values.spatialMetric = mat3x3<f32>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0)
  ) + termOuter(terms.a) + termOuter(terms.b) + termOuter(terms.r);
  var ones = vec3<f32>(1.0, 0.0, 0.0);
  var determinant = 1.0;
  let c = vec3<f32>(terms.a.c, terms.b.c, terms.r.c);
  if (phase == 0) {
    let coupling = dot(terms.a.n, terms.b.n) - 1.0;
    determinant = 1.0 - coupling * coupling * c.x * c.y;
    ones = vec3<f32>(1.0 - coupling * c.y, 1.0 - coupling * c.x, 0.0)
      / determinant;
  } else if (phase == 2) {
    let nab = dot(terms.a.n, terms.b.n) - 1.0;
    let nar = dot(terms.a.n, terms.r.n) - 1.0;
    let nbr = dot(terms.b.n, terms.r.n) - 1.0;
    let m01 = nab * c.y;
    let m02 = nar * c.z;
    let m10 = nab * c.x;
    let m12 = nbr * c.z;
    let m20 = nar * c.x;
    let m21 = nbr * c.y;
    let a00 = 1.0 - m12 * m21;
    let a10 = m12 * m20 - m10;
    let a20 = m10 * m21 - m20;
    determinant = a00 + m01 * a10 + m02 * a20;
    ones = vec3<f32>(
      a00 + (m02 * m21 - m01) + (m01 * m12 - m02),
      a10 + (1.0 - m02 * m20) + (m02 * m10 - m12),
      a20 + (m01 * m20 - m21) + (1.0 - m01 * m10)
    ) / determinant;
  } else {
    ones = vec3<f32>(0.0, 0.0, 1.0);
  }
  let weightedOnes = c * ones;
  let onePlusK = 1.0 + weightedOnes.x + weightedOnes.y + weightedOnes.z;
  values.lapse = inverseSqrt(max(onePlusK, 1.0e-8));
  values.shift = (
    weightedOnes.x * terms.a.n
    + weightedOnes.y * terms.b.n
    + weightedOnes.z * terms.r.n
  ) / max(onePlusK, 1.0e-8);
  var horizonDistance = 1.0e6;
  if (terms.weightA > 1.0e-4) {
    horizonDistance = min(horizonDistance, terms.a.kerrRadius - min(terms.weightA, 1.0) * radii.horizonA);
  }
  if (terms.weightB > 1.0e-4) {
    horizonDistance = min(horizonDistance, terms.b.kerrRadius - min(terms.weightB, 1.0) * radii.horizonB);
  }
  if (terms.weightR > 1.0e-4) {
    horizonDistance = min(horizonDistance, terms.r.kerrRadius - min(terms.weightR, 1.0) * radii.horizonR);
  }
  values.horizonDistance = horizonDistance;
  let activeRegularization = max(
    max(terms.weightA * terms.a.regularized, terms.weightB * terms.b.regularized),
    terms.weightR * terms.r.regularized
  );
  values.valid = select(
    0.0,
    1.0,
    determinant > 1.0e-6
      && onePlusK > 1.0e-6
      && onePlusK < 1.0e8
      && activeRegularization < 0.5
      && finiteScalar(values.lapse)
      && finiteVector(values.shift)
  );
  return values;
}

fn unresolvedResult() -> RayResult {
  var result: RayResult;
  result.outcome = RAY_UNRESOLVED;
  result.escapeDirection = vec3<f32>(0.0);
  result.frequencyShift = 1.0;
  result.lookback = 0.0;
  result.hamiltonianResidual = 1.0;
  result.iterations = 0.0;
  result.minimumHorizonDistance = 1.0e6;
  result.terminationReason = 0.0;${diskResultInitialization}
  return result;
}${diskFunctions}

// The camera is the observer at rest in the frozen coordinates,
// u = partial_t / alpha_s with alpha_s^2 = -g_tt = alpha^2 - beta_k beta^k.
// Vectors orthogonal to u have e^t = beta_j e^j / alpha_s^2, and their
// spatial components carry the metric h_ij = gamma_ij + beta_i beta_j /
// alpha_s^2. (The Eulerian observer of the Kerr-Schild slicing instead falls
// inward at ~2M/r, which shrinks the shadow and puts a Doppler dipole on the
// sky.)
struct StaticObserverFrame {
  metric: mat3x3<f32>,
  shiftCovariant: vec3<f32>,
  lapse: f32,
  valid: f32,
};

fn staticObserverFrame(values: MetricValues) -> StaticObserverFrame {
  let shiftCovariant = values.shiftCovariant;
  let lapseSquared = -values.gtt;
  var frame: StaticObserverFrame;
  frame.valid = select(0.0, 1.0, lapseSquared > 1.0e-6 && values.valid > 0.5);
  frame.lapse = sqrt(max(lapseSquared, 1.0e-6));
  frame.shiftCovariant = shiftCovariant;
  let inverseLapseSquared = 1.0 / (frame.lapse * frame.lapse);
  frame.metric = values.spatialMetric + mat3x3<f32>(
    shiftCovariant * (shiftCovariant.x * inverseLapseSquared),
    shiftCovariant * (shiftCovariant.y * inverseLapseSquared),
    shiftCovariant * (shiftCovariant.z * inverseLapseSquared)
  );
  return frame;
}

fn frameDot(metric: mat3x3<f32>, a: vec3<f32>, b: vec3<f32>) -> f32 {
  return dot(a, metric * b);
}

fn frameNormalize(metric: mat3x3<f32>, value: vec3<f32>) -> vec3<f32> {
  return value * inverseSqrt(max(frameDot(metric, value, value), 1.0e-14));
}

// The camera FOV lives in the observer's local orthonormal spatial frame, not
// in Euclidean coordinate components; metric Gram-Schmidt removes the
// systematic off-axis error that otherwise remains at finite radius.
fn observerCameraDirection(
  metric: mat3x3<f32>,
  screen: vec2<f32>,
  tanHalfFov: f32
) -> vec3<f32> {
  let forward = frameNormalize(metric, params.cameraForwardFov.xyz);
  let rawRight = params.cameraRightSkyRotation.xyz
    - forward * frameDot(metric, forward, params.cameraRightSkyRotation.xyz);
  let right = frameNormalize(metric, rawRight);
  let rawUp = params.cameraUpDiskOuter.xyz
    - forward * frameDot(metric, forward, params.cameraUpDiskOuter.xyz)
    - right * frameDot(metric, right, params.cameraUpDiskOuter.xyz);
  let up = frameNormalize(metric, rawUp);
  return frameNormalize(
    metric,
    forward + tanHalfFov * (screen.x * right + screen.y * up)
  );
}

// Photon-orbit capture. For an isolated Kerr hole no null geodesic has a
// radial turning point inside the innermost (prograde equatorial) circular
// photon orbit, so a ray found there while moving inward cannot escape. Time
// reversal maps Kerr to Kerr with the opposite spin, which has the same
// innermost orbit, so the test also holds for these past-directed rays. In the
// superposed binary metric it is applied to the nearest term.
//
// Hover capture. Around an isolated hole, wherever its Kerr-Schild field 2H is
// at least 2/3 (r <= 3m for Schwarzschild), every photon whose past-directed
// ray still reaches the sky moves at an ingoing Kerr-Schild coordinate speed of
// at least 1/sqrt(3), or 0.55 for Kerr up to chi = 0.95. A slower one is an
// outgoing photon peeling off a trapped surface: traced backward it only
// approaches the horizon, never the sky. During the merger blend such rays
// hover at the blended metric's horizon, outside the weight-scaled capture
// radii, until the step budget runs out; this ends them early with the same
// outcome. In the superposition the field is summed along the terms' null
// directions: aligned terms add like one hole, while between two holes they
// cancel, and there light is slow along the axis (~0.19 at the start of the
// blend) yet escapes. The ray must also be inside the unscaled photon orbit of
// a term present in the metric.
const HOVER_CAPTURE_SPEED: f32 = 0.2;
const HOVER_CAPTURE_FIELD: f32 = 2.0 / 3.0;

fn insidePhotonCapture(
  sample: GeodesicSample,
  backwardVelocity: vec3<f32>
) -> bool {
  let photonOrbitCapture = sample.photonMargin < 0.0
    && dot(sample.photonRadialGradient, backwardVelocity) < 0.0;
  let hoverCapture = sample.failureCaptureMargin < 0.0
    && sample.alignedField >= HOVER_CAPTURE_FIELD
    && length(backwardVelocity) < HOVER_CAPTURE_SPEED;
  return photonOrbitCapture || hoverCapture;
}

fn traceStrongField(screen: vec2<f32>, tanHalfFov: f32) -> RayResult {
  return traceStrongFieldRay(screen, tanHalfFov, 0.0);
}

// footprintAngle > 0 only for sparse coarse nodes: the angular half-width of
// the beam the node stands for (see SPARSE_FOOTPRINT_SPACINGS).
fn traceStrongFieldRay(
  screen: vec2<f32>,
  tanHalfFov: f32,
  footprintAngle: f32
) -> RayResult {
  var result = unresolvedResult();
  var position = params.cameraPosRadius.xyz;
  let radii = spacetimeRadii();
  let observerMetric = metricValuesAt(position);
  if (observerMetric.valid < 0.5) {
    result.terminationReason = 2.0;
    return result;
  }

  // The view direction n points from the camera into the scene. The photon
  // that arrives at the static camera has four-momentum p = E_s (u - n) with
  // unit observed energy E_s; store its covector p_i = beta_i p^t - gamma_ij
  // n^j and integrate Hamilton's equations backward in coordinate time. This
  // arriving-photon convention fixes the sign of frame dragging and of the
  // Doppler/gravitational shifts. The asymptotic energy is -p_t = alpha_s.
  let observerFrame = staticObserverFrame(observerMetric);
  if (observerFrame.valid < 0.5) {
    result.terminationReason = 2.0;
    return result;
  }
  let initialDirection = observerCameraDirection(
    observerFrame.metric,
    screen,
    tanHalfFov
  );
  let staticLapse = observerFrame.lapse;
  let photonTime = (
    1.0 - dot(observerFrame.shiftCovariant, initialDirection) / staticLapse
  ) / staticLapse;
  var momentum = observerFrame.shiftCovariant * photonTime
    - observerMetric.spatialMetric * initialDirection;
  let observerQ = 1.0;
  // The conserved energy E = -p_t (= alpha_s analytically) is taken from the
  // loop's first evaluation, which samples this same point.
  var conservedEnergy = 0.0;

  let minimumStep = clamp(params.sceneStrongIntegrator.x, 0.002, 0.5);
  let maximumStep = clamp(
    params.sceneStrongIntegrator.y,
    minimumStep,
    ${STRONG_FIELD_MAXIMUM_STEP_M.toFixed(1)}
  );
  // Each RK4 step is this fraction of sample.stepDistance, so steps shrink
  // near the holes and grow geometrically in the far field.
  let stepFraction = clamp(params.sceneStrongIntegrator.z, 0.02, 1.0);
  let residualFail = clamp(params.sceneStrongIntegrator.w, 1.0e-4, 0.5);
  let escapeRadius = max(
    params.sceneStrongDomain.x,
    length(position) + 8.0
  );
  let maximumLookback = max(params.sceneStrongDomain.y, 16.0);
  let capturePadding = max(params.sceneStrongDomain.z, 0.0);
  let maximumSteps = clamp(
    i32(params.sceneStrongDomain.w),
    8,
    MAX_RK4_STEPS
  );
  var enteredDomain = false;
  var lookback = 0.0;
  var maximumResidual = 0.0;
  var minimumHorizonDistance = 1.0e6;

  // Classical fourth-order Runge-Kutta, written as a stage machine so that the
  // metric provider is evaluated at exactly one call site per iteration and
  // little state stays live across it. Stage 0 is the start of a step
  // (capture, escape and step-size logic plus k1); stages 1..3 evaluate k2..k4
  // at the usual intermediate states. A failed stage restarts the step from
  // stage 0 with a quarter of the step size.
  var stage = 0;
  var stepCount = 0;
  var stepSize = minimumStep;
  var retryScale = 1.0;
  var stepPosition = position;
  var stepMomentum = momentum;
  var sumX = vec3<f32>(0.0);
  var sumP = vec3<f32>(0.0);
  var evalPosition = position;
  var evalMomentum = momentum;${diskMidpointDeclaration}
  var jets = 0.0;

  for (
    var iteration: i32 = 0;
    iteration < MAX_RK4_STEPS * 4 + 16;
    iteration = iteration + 1
  ) {
    let sample = evaluateGeodesic(evalPosition, evalMomentum, radii);
    jets = jets + 1.0;
    result.iterations = jets;
    var momentumScale = 1.0;
    if (iteration == 0) {
      conservedEnergy = sample.hamiltonian;
      if (!finiteScalar(conservedEnergy) || conservedEnergy <= 1.0e-6) {
        result.terminationReason = 2.0;
        return result;
      }
    }

    if (stage == 0) {
      minimumHorizonDistance = min(
        minimumHorizonDistance,
        sample.horizonDistance
      );
      if (sample.horizonDistance <= capturePadding) {
        result.outcome = RAY_CAPTURED;
        result.lookback = lookback;
        result.hamiltonianResidual = maximumResidual;
        result.minimumHorizonDistance = minimumHorizonDistance;
        return result;
      }
      var deviation = 1.0;
      let kinematicsValid = sample.valid > 0.5
        && sample.hamiltonian > 1.0e-8;
      if (kinematicsValid) {
        deviation = abs(sample.hamiltonian / conservedEnergy - 1.0);
      }
      maximumResidual = max(maximumResidual, deviation);
      if (
        !kinematicsValid
        || !finiteScalar(deviation)
        || deviation > residualFail
      ) {
        // Inside an innermost photon orbit a ray that came from outside cannot
        // escape (and during the merger blend that region is inside the
        // common horizon), so an unrecoverable sample there is a capture.
        if (sample.failureCaptureMargin < 0.0) {
          result.outcome = RAY_CAPTURED;
          result.lookback = lookback;
          result.hamiltonianResidual = maximumResidual;
          result.minimumHorizonDistance = minimumHorizonDistance;
          return result;
        }
        result.terminationReason = select(3.0, 2.0, sample.metricValid < 0.5);
        result.hamiltonianResidual = max(maximumResidual, 1.0);
        result.minimumHorizonDistance = minimumHorizonDistance;
        return result;
      }
      // H is homogeneous of degree one in p, so this rescaling restores the
      // conserved energy exactly without changing the spatial path. The flow
      // was evaluated before rescaling: dx/dt is of degree zero and dp/dt of
      // degree one in p, so only the momentum rate scales.
      momentumScale = conservedEnergy / sample.hamiltonian;
      evalMomentum = evalMomentum * momentumScale;
      stepMomentum = evalMomentum;
    }

    let derivativeX = -sample.velocity;
    let derivativeP = -sample.momentumRate * momentumScale;
    let derivativeValid = sample.valid > 0.5;

    if (stage == 0) {
      if (insidePhotonCapture(sample, derivativeX)) {
        result.outcome = RAY_CAPTURED;
        result.lookback = lookback;
        result.hamiltonianResidual = maximumResidual;
        result.minimumHorizonDistance = minimumHorizonDistance;
        return result;
      }
      let radius = length(stepPosition);
      enteredDomain = enteredDomain || radius < escapeRadius * 0.82;
      if (
        enteredDomain
        && radius >= escapeRadius
        && dot(stepPosition, derivativeX) > 0.0
      ) {
        result.outcome = RAY_ESCAPED;
        // Ingoing Kerr-Schild coordinates follow the ingoing principal null
        // congruence, so an arriving photon's coordinate direction here is
        // already within ~M b^3 / r^4 of its asymptote. No harmonic-gauge tail
        // correction is added.
        result.escapeDirection = safeNormalize(derivativeX);
        result.frequencyShift = clamp(
          observerQ / max(conservedEnergy, 1.0e-5),
          0.02,
          max(params.sceneStrongDiagnostics.x, 1.0)
        );
        result.lookback = lookback;
        result.hamiltonianResidual = maximumResidual;
        result.minimumHorizonDistance = minimumHorizonDistance;
        return result;
      }
      if (stepCount >= maximumSteps) {
        // A ray still circling inside an innermost photon orbit when the
        // budget runs out is falling in slowly (typical of the blended merger
        // metric); elsewhere exhaustion stays unresolved.
        if (sample.failureCaptureMargin < 0.0) {
          result.outcome = RAY_CAPTURED;
          result.lookback = lookback;
          result.hamiltonianResidual = maximumResidual;
          result.minimumHorizonDistance = minimumHorizonDistance;
          return result;
        }
        break;
      }
      if (lookback >= maximumLookback) {
        result.terminationReason = 4.0;
        break;
      }
      if (!derivativeValid) {
        if (sample.failureCaptureMargin < 0.0) {
          result.outcome = RAY_CAPTURED;
          result.lookback = lookback;
          result.hamiltonianResidual = maximumResidual;
          result.minimumHorizonDistance = minimumHorizonDistance;
          return result;
        }
        result.terminationReason = 3.0;
        result.hamiltonianResidual = max(maximumResidual, 1.0);
        result.minimumHorizonDistance = minimumHorizonDistance;
        return result;
      }
      let outbound = radius > FAR_FIELD_RADIUS_M
        && dot(stepPosition, derivativeX) > 0.0;
      let effectiveFraction = select(
        stepFraction,
        min(FAR_FIELD_GROWTH * stepFraction, FAR_FIELD_MAXIMUM_FRACTION),
        outbound
      );
      stepSize = clamp(
        effectiveFraction * max(sample.stepDistance, 0.0) * retryScale,
        minimumStep,
        maximumStep
      );
      stepSize = max(min(stepSize, maximumLookback - lookback), minimumStep);
      sumX = derivativeX;
      sumP = derivativeP;${diskMidpointFromK1}
      evalPosition = stepPosition + 0.5 * stepSize * derivativeX;
      evalMomentum = stepMomentum + 0.5 * stepSize * derivativeP;
      stage = 1;
      continue;
    }

    if (!derivativeValid) {
      // A stage left the valid metric domain. Inside an unscaled innermost
      // photon orbit this is a capture; elsewhere retry the step from its
      // start with a quarter of the size, or give up at the minimum step.
      if (sample.failureCaptureMargin < 0.0) {
        result.outcome = RAY_CAPTURED;
        result.lookback = lookback;
        result.hamiltonianResidual = maximumResidual;
        result.minimumHorizonDistance = minimumHorizonDistance;
        return result;
      }
      if (stepSize <= minimumStep) {
        result.terminationReason = 3.0;
        result.hamiltonianResidual = max(maximumResidual, 1.0);
        result.minimumHorizonDistance = minimumHorizonDistance;
        return result;
      }
      retryScale = retryScale * 0.25;
      evalPosition = stepPosition;
      evalMomentum = stepMomentum;
      stage = 0;
      continue;
    }

    if (stage == 1) {
      sumX = sumX + 2.0 * derivativeX;
      sumP = sumP + 2.0 * derivativeP;${diskMidpointAddInner}
      evalPosition = stepPosition + 0.5 * stepSize * derivativeX;
      evalMomentum = stepMomentum + 0.5 * stepSize * derivativeP;
      stage = 2;
      continue;
    }
    if (stage == 2) {
      sumX = sumX + 2.0 * derivativeX;
      sumP = sumP + 2.0 * derivativeP;${diskMidpointAddInner}
      evalPosition = stepPosition + stepSize * derivativeX;
      evalMomentum = stepMomentum + stepSize * derivativeP;
      stage = 3;
      continue;
    }

    // Stage 3: k4 completes the step.
    let nextPosition = stepPosition
      + (stepSize / 6.0) * (sumX + derivativeX);
    let nextMomentum = stepMomentum
      + (stepSize / 6.0) * (sumP + derivativeP);
    if (!finiteVector(nextPosition) || !finiteVector(nextMomentum)) {
      result.terminationReason = 3.0;
      result.hamiltonianResidual = max(maximumResidual, 1.0);
      result.minimumHorizonDistance = minimumHorizonDistance;
      return result;
    }${diskStep}
    lookback = lookback + stepSize;
    stepCount = stepCount + 1;
    retryScale = 1.0;
    stepPosition = nextPosition;
    stepMomentum = nextMomentum;
    evalPosition = nextPosition;
    evalMomentum = nextMomentum;
    stage = 0;
  }

  // Exhausting the step or lookback budget is unresolved, never silently
  // converted to captured black or a fabricated sky sample.
  result.outcome = RAY_UNRESOLVED;
  if (result.terminationReason < 0.5) {
    result.terminationReason = 1.0;
  }
  result.lookback = lookback;
  result.hamiltonianResidual = maximumResidual;
  result.minimumHorizonDistance = minimumHorizonDistance;
  return result;
}

fn rotateAroundY(direction: vec3<f32>, angle: f32) -> vec3<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec3<f32>(
    c * direction.x + s * direction.z,
    direction.y,
   -s * direction.x + c * direction.z
  );
}

fn sampleEnvironment(direction: vec3<f32>) -> vec3<f32> {
  let d = safeNormalize(
    rotateAroundY(direction, params.cameraRightSkyRotation.w)
  );
  let longitude = atan2(d.z, d.x);
  let latitude = asin(clamp(d.y, -1.0, 1.0));
  let uv = vec2<f32>(
    fract(longitude / TWO_PI + 0.5),
    clamp(0.5 - latitude / PI, 0.00001, 0.99999)
  );
  let dimensions = vec2<f32>(textureDimensions(tSky));
  let texel = vec2<f32>(1.0) / dimensions;
  let resolution = max(
    params.resolutionTimeMass.xy,
    vec2<f32>(1.0)
  );
  let aspect = resolution.x / resolution.y;
  let horizontalFov = 2.0 * atan(
    tan(0.5 * clamp(params.cameraForwardFov.w, 0.02, 2.8))
      * aspect
  );
  let sourceFootprint = max(
    dimensions.x * horizontalFov
      / (TWO_PI * resolution.x),
    0.0
  );
  // Reconstruction depends only on the screen-to-panorama footprint. Do not
  // derive it from per-ray iteration counts or closest-horizon state: those
  // move at sub-pixel boundaries and made bright stars shimmer.
  let footprintPressure = smoothstep(0.62, 1.35, sourceFootprint);
  let radius = clamp(sourceFootprint * 0.72, 0.50, 3.0);
  let centre = textureSampleLevel(
    tSky,
    skySampler,
    uv,
    0.0
  ).rgb;
  let filterWeight = 0.32 * footprintPressure;
  var panorama = centre;
  // Four stable axis taps band-limit the photographic panorama before a
  // low-resolution ray is enlarged by the browser.  Their positions and
  // weights depend only on the screen-to-panorama pixel footprint and the
  // fixed quality tier. Native-resolution pixels below the footprint threshold
  // retain the sharp single-sample path for the 6K source.
  if (filterWeight > 0.01) {
    let filtered = 0.25 * (
      textureSampleLevel(
        tSky,
        skySampler,
        uv + vec2<f32>(radius * texel.x, 0.0),
        0.0
      ).rgb
      + textureSampleLevel(
        tSky,
        skySampler,
        uv - vec2<f32>(radius * texel.x, 0.0),
        0.0
      ).rgb
      + textureSampleLevel(
        tSky,
        skySampler,
        uv + vec2<f32>(0.0, radius * texel.y),
        0.0
      ).rgb
      + textureSampleLevel(
        tSky,
        skySampler,
        uv - vec2<f32>(0.0, radius * texel.y),
        0.0
      ).rgb
    );
    panorama = mix(centre, filtered, filterWeight);
  }
  return max(
    panorama - vec3<f32>(0.0015),
    vec3<f32>(0.0)
  ) * max(params.displayOutput.w, 0.01);
}

fn viridis(valueInput: f32) -> vec3<f32> {
  let value = clamp(valueInput, 0.0, 1.0);
  let c0 = vec3<f32>(0.267, 0.005, 0.329);
  let c1 = vec3<f32>(0.283, 0.141, 0.458);
  let c2 = vec3<f32>(0.254, 0.265, 0.530);
  let c3 = vec3<f32>(0.207, 0.372, 0.553);
  let c4 = vec3<f32>(0.164, 0.471, 0.558);
  let c5 = vec3<f32>(0.128, 0.567, 0.551);
  let c6 = vec3<f32>(0.135, 0.659, 0.518);
  let c7 = vec3<f32>(0.267, 0.749, 0.441);
  let c8 = vec3<f32>(0.478, 0.821, 0.318);
  let c9 = vec3<f32>(0.741, 0.873, 0.150);
  let scaled = value * 9.0;
  let index = i32(floor(scaled));
  let fraction = fract(scaled);
  if (index <= 0) { return mix(c0, c1, fraction); }
  if (index == 1) { return mix(c1, c2, fraction); }
  if (index == 2) { return mix(c2, c3, fraction); }
  if (index == 3) { return mix(c3, c4, fraction); }
  if (index == 4) { return mix(c4, c5, fraction); }
  if (index == 5) { return mix(c5, c6, fraction); }
  if (index == 6) { return mix(c6, c7, fraction); }
  if (index == 7) { return mix(c7, c8, fraction); }
  return mix(c8, c9, fraction);
}

fn outcomeDiagnostic(result: RayResult) -> vec3<f32> {
  if (result.outcome == RAY_CAPTURED) {
    return vec3<f32>(0.02, 0.035, 0.07);
  }
  if (result.outcome == RAY_ESCAPED) {
    return vec3<f32>(0.12, 0.78, 0.50);
  }
  if (result.terminationReason < 1.5) {
    return vec3<f32>(0.95, 0.19, 0.62);
  }
  if (result.terminationReason < 2.5) {
    return vec3<f32>(0.92, 0.10, 0.08);
  }
  if (result.terminationReason < 3.5) {
    return vec3<f32>(1.00, 0.48, 0.05);
  }
  return vec3<f32>(0.76, 0.28, 0.96);
}

// Diagnostic palettes are sRGB-encoded display colours. The scene target is
// linear, and the post pass shows diagnostics without tone mapping, so decode
// them here to reach the screen unchanged.
fn decodeSrgbDisplay(colour: vec3<f32>) -> vec3<f32> {
  let encoded = clamp(colour, vec3<f32>(0.0), vec3<f32>(1.0));
  return select(
    pow((encoded + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)),
    encoded / 12.92,
    encoded <= vec3<f32>(0.04045)
  );
}

fn diagnosticColour(result: RayResult, mode: i32) -> vec3<f32> {
  if (mode == 1) {
    return outcomeDiagnostic(result);
  }
  if (mode == 2) {
    return viridis(result.lookback / max(params.sceneStrongDomain.y, 1.0));
  }
  if (mode == 3) {
    let maximumShift = max(params.sceneStrongDiagnostics.x, 1.0);
    let mapped = log2(max(result.frequencyShift, 0.02))
      / max(log2(maximumShift), 1.0) * 0.5 + 0.5;
    return viridis(mapped);
  }
  if (mode == 4) {
    let scale = max(params.sceneStrongDiagnostics.y, 1.0);
    return viridis(
      clamp(log2(1.0 + result.hamiltonianResidual * scale) / 8.0, 0.0, 1.0)
    );
  }
  // Metric evaluations as a fraction of the active tier's budget: four per
  // RK4 step plus the observer sample.
  let stepBudget = clamp(i32(params.sceneStrongDomain.w), 8, MAX_RK4_STEPS);
  return viridis(result.iterations / (4.0 * f32(stepBudget) + 1.0));
}

fn shadeResult(result: RayResult, pixel: vec2<f32>) -> vec3<f32> {
  let mode = i32(round(params.renderControls.z));
  if (mode >= 1 && mode <= 5) {
    return decodeSrgbDisplay(diagnosticColour(result, mode));
  }
  if (result.outcome == RAY_CAPTURED) {
    ${capturedPhotographicResult}
  }
  if (result.outcome == RAY_UNRESOLVED) {
    // Keep the failure mask visible without leaking a saturated diagnostic
    // colour into the photographic sky mode. Outcome mode above retains the
    // bright reason-coded palette for numerical inspection.
    let hatch = select(
      0.55,
      1.0,
      ((i32(pixel.x) + i32(pixel.y)) & 7) < 3
    );
    let unresolvedLevel = clamp(
      params.sceneStrongDiagnostics.z,
      0.02,
      0.25
    );
    ${unresolvedPhotographicResult}
  }
  // I_nu / nu^3 is invariant. The panorama is broadband rather than spectral,
  // so g^4 is the declared bolometric display approximation, as in the
  // Schwarzschild scene and the stationary reference workbench. A static camera sees
  // every sky photon with the same g = 1 / alpha_static.
  let shiftRadiance = pow(
    clamp(result.frequencyShift, 0.25, 4.0),
    4.0
  );
  ${escapedPhotographicResult}
}

fn radicalInverse(indexInput: u32, base: u32) -> f32 {
  var index = indexInput;
  var factor = 1.0 / f32(base);
  var value = 0.0;
  for (var digitIndex: i32 = 0; digitIndex < 16; digitIndex = digitIndex + 1) {
    if (index == 0u) {
      break;
    }
    value = value + f32(index % base) * factor;
    index = index / base;
    factor = factor / f32(base);
  }
  return value;
}

fn accumulationJitter() -> vec2<f32> {
  // Reset frames are unjittered: every pixel then belongs exclusively to the
  // latest camera. Settled samples use a deterministic, bounded Halton(2,3)
  // prefix. History epochs deliberately do not scramble the sequence: the
  // same static camera/time therefore refines through the same sample set.
  if (params.sceneStrongQuality.w > 0.5) {
    return vec2<f32>(0.0);
  }
  let sampleIndex = u32(clamp(
    params.sceneStrongQuality.x + 1.0,
    1.0,
    1048575.0
  ));
  let sequenceIndex = sampleIndex;
  // The first refinement sample stays close to the centre ray; later samples
  // open gradually to at most +/-0.29 pixel instead of jumping by +/-0.5.
  let jitterAmplitude = mix(
    0.20,
    0.58,
    smoothstep(1.0, 8.0, f32(sampleIndex))
  );
  return jitterAmplitude * vec2<f32>(
    radicalInverse(sequenceIndex, 2u) - 0.5,
    radicalInverse(sequenceIndex, 3u) - 0.5
  );
}

fn screenForPixelCentre(pixelCentre: vec2<f32>) -> vec2<f32> {
  let resolution = max(params.resolutionTimeMass.xy, vec2<f32>(1.0));
  let uv = pixelCentre / resolution;
  return vec2<f32>(
    (uv.x * 2.0 - 1.0) * resolution.x / resolution.y,
    1.0 - uv.y * 2.0
  );
}

fn tracerTanHalfFov() -> f32 {
  return tan(0.5 * clamp(params.cameraForwardFov.w, 0.02, 2.8));
}

// Sparse tracing (moving frames). Coarse node (i, j), stored at texel
// (i + 1, j + 1), traces the ray through full-resolution pixel (i S, j S) and
// stores its escape direction scaled by the frequency shift, with the outcome
// code in w (2 = escaped to the sky; +4 when its beam met a disk).
@fragment
fn fsCoarse(input: FragmentInput) -> @location(0) vec4<f32> {
  let node = floor(input.position.xy) - vec2<f32>(1.0);
  let tanHalfFov = tracerTanHalfFov();
  let resolution = max(params.resolutionTimeMass.xy, vec2<f32>(1.0));
  // Angle of one pixel at the image centre, the largest anywhere.
  let pixelAngle = 2.0 * tanHalfFov / resolution.y;
  let result = traceStrongFieldRay(
    screenForPixelCentre(node * SPARSE_STRIDE + vec2<f32>(0.5)),
    tanHalfFov,
    SPARSE_FOOTPRINT_SPACINGS * SPARSE_STRIDE * pixelAngle
  );
  var code = f32(result.outcome);${sparseDiskFlag}
  return vec4<f32>(result.escapeDirection * result.frequencyShift, code);
}

struct SparseSample {
  direction: vec3<f32>,
  frequencyShift: f32,
  accepted: bool,
};

fn coarseNode(texel: vec2<i32>) -> vec4<f32> {
  return textureLoad(coarseField, texel, 0);
}

// Fail closed: a pixel is interpolated only when the four corners of its cell
// and their axis neighbours all escaped and no beam of theirs met a disk, and
// the bilinear error bound (second difference / 8) is below
// SPARSE_CURVATURE_PIXELS pixels. Everything else is traced.
fn sparseInterpolation(pixel: vec2<f32>, tanHalfFov: f32) -> SparseSample {
  let stride = SPARSE_STRIDE;
  var sample: SparseSample;
  sample.accepted = false;
  sample.direction = vec3<f32>(0.0);
  sample.frequencyShift = 1.0;
  let size = vec2<i32>(textureDimensions(coarseField));
  let cell = floor(pixel / stride);
  let fraction = (pixel - cell * stride) / stride;
  let base = vec2<i32>(cell) + vec2<i32>(1);
  if (any(base < vec2<i32>(1)) || any(base + vec2<i32>(2) >= size)) {
    return sample;
  }
  let c00 = coarseNode(base);
  let c10 = coarseNode(base + vec2<i32>(1, 0));
  let c01 = coarseNode(base + vec2<i32>(0, 1));
  let c11 = coarseNode(base + vec2<i32>(1, 1));
  let corners = vec4<f32>(c00.w, c10.w, c01.w, c11.w);
  if (any(abs(corners - vec4<f32>(2.0)) > vec4<f32>(0.25))) {
    return sample;
  }
  let xm0 = coarseNode(base + vec2<i32>(-1, 0));
  let xp0 = coarseNode(base + vec2<i32>(2, 0));
  let xm1 = coarseNode(base + vec2<i32>(-1, 1));
  let xp1 = coarseNode(base + vec2<i32>(2, 1));
  let ym0 = coarseNode(base + vec2<i32>(0, -1));
  let ym1 = coarseNode(base + vec2<i32>(1, -1));
  let yp0 = coarseNode(base + vec2<i32>(0, 2));
  let yp1 = coarseNode(base + vec2<i32>(1, 2));
  let neighbours = array<f32, 8>(xm0.w, xp0.w, xm1.w, xp1.w, ym0.w, ym1.w, yp0.w, yp1.w);
  for (var index = 0; index < 8; index = index + 1) {
    if (abs(neighbours[index] - 2.0) > 0.25) {
      return sample;
    }
  }
  let curvature = max(
    max(
      max(length(xm0.xyz - 2.0 * c00.xyz + c10.xyz), length(ym0.xyz - 2.0 * c00.xyz + c01.xyz)),
      max(length(c00.xyz - 2.0 * c10.xyz + xp0.xyz), length(ym1.xyz - 2.0 * c10.xyz + c11.xyz))
    ),
    max(
      max(length(xm1.xyz - 2.0 * c01.xyz + c11.xyz), length(c00.xyz - 2.0 * c01.xyz + yp0.xyz)),
      max(length(c01.xyz - 2.0 * c11.xyz + xp1.xyz), length(c10.xyz - 2.0 * c11.xyz + yp1.xyz))
    )
  );
  let resolution = max(params.resolutionTimeMass.xy, vec2<f32>(1.0));
  let pixelAngle = 2.0 * tanHalfFov / resolution.y;
  if (0.125 * curvature > SPARSE_CURVATURE_PIXELS * pixelAngle * length(c00.xyz)) {
    return sample;
  }
  let blended = mix(
    mix(c00.xyz, c10.xyz, fraction.x),
    mix(c01.xyz, c11.xyz, fraction.x),
    fraction.y
  );
  let magnitude = length(blended);
  sample.frequencyShift = magnitude;
  sample.direction = blended / max(magnitude, 1.0e-12);
  sample.accepted = magnitude > 1.0e-6;
  return sample;
}

@fragment
fn fsMain(input: FragmentInput) -> @location(0) vec4<f32> {
  let resolution = max(params.resolutionTimeMass.xy, vec2<f32>(1.0));
  let aspect = resolution.x / resolution.y;
  // Only unjittered photographic frames that requested sparse tracing (the
  // renderer then traced the coarse field first) use the coarse field.
  if (
    params.sceneStrongDiagnostics.w == SPARSE_STRIDE
    && i32(round(params.renderControls.z)) == 0
    && params.sceneStrongQuality.w > 0.5
  ) {
    let sparse = sparseInterpolation(
      floor(input.position.xy),
      tracerTanHalfFov()
    );
    if (sparse.accepted) {
      var interpolated = unresolvedResult();
      interpolated.outcome = RAY_ESCAPED;
      interpolated.escapeDirection = sparse.direction;
      interpolated.frequencyShift = sparse.frequencyShift;
      interpolated.hamiltonianResidual = 0.0;
      let colour = shadeResult(interpolated, input.position.xy);
      return vec4<f32>(max(colour, vec3<f32>(0.0)), 1.0);
    }
  }
  let jitteredUv = input.uv + accumulationJitter() / resolution;
  let screen = vec2<f32>(
    (jitteredUv.x * 2.0 - 1.0) * aspect,
    1.0 - jitteredUv.y * 2.0
  );
  let tanHalfFov = tan(
    0.5 * clamp(params.cameraForwardFov.w, 0.02, 2.8)
  );
  let result = traceStrongField(screen, tanHalfFov);
  let colour = shadeResult(result, input.position.xy);
  return vec4<f32>(max(colour, vec3<f32>(0.0)), 1.0);
}
`;
}

// Metal pipelines specialize the frame-uniform spacetime phase (binary,
// transition, remnant) and whether the binary bodies spin, so the compiler
// drops the inactive Kerr-Schild providers and, for non-spinning bodies, the
// Kerr branch of their terms.
function traceSpecialization(id, phase, binarySpin) {
  return Object.freeze({
    id,
    constants: Object.freeze({
      SPACETIME_PHASE_MODE: phase,
      BINARY_SPIN_MODE: binarySpin,
    }),
  });
}

const STRONG_FIELD_TRACE_SPECIALIZATIONS = Object.freeze([
  traceSpecialization("binary", 0, 0),
  traceSpecialization("binary-spinning", 0, 1),
  traceSpecialization("transition", 2, 0),
  traceSpecialization("transition-spinning", 2, 1),
  traceSpecialization("remnant", 1, 0),
]);

const STRONG_FIELD_SPARSE_COARSE_PASS = Object.freeze({
  entryPoint: "fsCoarse",
  format: "rgba32float",
  stride: STRONG_FIELD_SPARSE_STRIDE,
  binding: 3,
});

// Offsets of the body A and B dimensionless spins in the 44-float packet.
const BODY_SPIN_OFFSETS = Object.freeze([12, 24]);

function selectStrongFieldTraceSpecialization(frame) {
  const uniforms = frame?.sceneStrongFieldUniforms;
  const blend = Number(uniforms?.[1]);
  if (blend === 1) {
    return "remnant";
  }
  // Same threshold as the WGSL Schwarzschild branch (|chi| < 1e-5).
  const spinning = BODY_SPIN_OFFSETS.some((offset) => Math.hypot(
    Number(uniforms?.[offset] ?? 0),
    Number(uniforms?.[offset + 1] ?? 0),
    Number(uniforms?.[offset + 2] ?? 0),
  ) >= 1.0e-5);
  const phase = blend === 0 ? "binary" : "transition";
  return spinning ? `${phase}-spinning` : phase;
}

export const strongFieldBinaryTraceFragmentWGSL =
  createStrongFieldBinaryTraceFragmentWGSL();

export const strongFieldBinaryDualDiskTraceFragmentWGSL =
  createStrongFieldBinaryTraceFragmentWGSL({ dualDisk: true });

export const strongFieldBinaryShaderBundle = Object.freeze({
  id: "binary-strong-field-v1",
  labels: Object.freeze({
    uniforms: "Real-time strong-field binary frame uniforms",
    trace: "WebGPU 3+1 Hamiltonian strong-field binary tracer",
    webglFallback: "Legacy WebGL2 weak-field fast-light fallback",
  }),
  backendPolicy: Object.freeze({
    production: "webgpu",
    webgpuModel: "superposed-kerr-schild-fast-light",
    webgl2Model: "legacy-weak-field-fast-light",
    physicalParityRequired: false,
  }),
  accumulation: Object.freeze({
    mode: "linear-hdr-running-average-v1",
    sampleIndexField: "strongFieldQuality.accumulationIndex",
    resetField: "strongFieldQuality.historyReset",
    jitter: "deterministic-bounded-halton-2-3",
  }),
  wgsl: Object.freeze({
    trace: strongFieldBinaryTraceFragmentWGSL,
    traceSpecializations: STRONG_FIELD_TRACE_SPECIALIZATIONS,
    selectTraceSpecialization: selectStrongFieldTraceSpecialization,
    coarse: STRONG_FIELD_SPARSE_COARSE_PASS,
  }),
  glsl: Object.freeze({
    // Deliberate fallback, not a port of the strong-field provider. It has
    // no diagnostic modes, so its output always takes the photographic path.
    trace: binaryTraceFragmentGLSL,
    diagnosticModes: false,
  }),
  uniforms: Object.freeze({
    requiredFloatCount: STRONG_FIELD_UNIFORM_FLOATS,
    writeWebGPUExtras(tail, frame) {
      writeStrongFieldUniformTail(tail, frame);
    },
    createWebGLExtras(THREE) {
      return {
        uSceneBinaryState: { value: new THREE.Vector4() },
        uSceneBinaryMasses: { value: new THREE.Vector4() },
      };
    },
    writeWebGLExtras(uniforms, frame) {
      const state = finiteVec4(
        frame?.sceneBinaryState,
        "sceneBinaryState",
      );
      const masses = finiteVec4(
        frame?.sceneBinaryMasses,
        "sceneBinaryMasses",
      );
      uniforms.uSceneBinaryState.value.fromArray(state);
      uniforms.uSceneBinaryMasses.value.fromArray(masses);
    },
  }),
});

export const strongFieldBinaryDualDiskShaderBundle = Object.freeze({
  id: "binary-dual-disk-strong-field-v1",
  labels: Object.freeze({
    uniforms: "Strong-field binary frame plus analytic dual thin-disk uniforms",
    trace: "WebGPU 3+1 Hamiltonian tracer with frame-frozen dual thin-disk transfer",
    webglFallback: "Legacy WebGL2 weak-field vacuum fallback without disk parity",
  }),
  backendPolicy: Object.freeze({
    production: "webgpu",
    webgpuModel: "superposed-kerr-schild-fast-light-plus-analytic-thin-disks",
    webgl2Model: "legacy-weak-field-fast-light-vacuum",
    physicalParityRequired: false,
    matterBackreaction: false,
    scientificStatus: "analytic thin-disk transfer with a bounded phenomenological emissivity texture; not GRMHD or NR matter evolution",
  }),
  accumulation: Object.freeze({
    mode: "linear-hdr-running-average-v1",
    sampleIndexField: "strongFieldQuality.accumulationIndex",
    resetField: "strongFieldQuality.historyReset",
    jitter: "deterministic-bounded-halton-2-3",
  }),
  wgsl: Object.freeze({
    trace: strongFieldBinaryDualDiskTraceFragmentWGSL,
    traceSpecializations: STRONG_FIELD_TRACE_SPECIALIZATIONS,
    selectTraceSpecialization: selectStrongFieldTraceSpecialization,
    coarse: STRONG_FIELD_SPARSE_COARSE_PASS,
  }),
  glsl: Object.freeze({
    // Deliberate vacuum fallback. It is surfaced as a different physical model
    // and has no diagnostic modes.
    trace: binaryTraceFragmentGLSL,
    diagnosticModes: false,
  }),
  uniforms: Object.freeze({
    requiredFloatCount: STRONG_FIELD_ACCRETION_UNIFORM_FLOATS,
    writeWebGPUExtras(tail, frame) {
      writeStrongFieldAccretionUniformTail(tail, frame);
    },
    createWebGLExtras(THREE) {
      return {
        uSceneBinaryState: { value: new THREE.Vector4() },
        uSceneBinaryMasses: { value: new THREE.Vector4() },
      };
    },
    writeWebGLExtras(uniforms, frame) {
      const state = finiteVec4(
        frame?.sceneBinaryState,
        "sceneBinaryState",
      );
      const masses = finiteVec4(
        frame?.sceneBinaryMasses,
        "sceneBinaryMasses",
      );
      uniforms.uSceneBinaryState.value.fromArray(state);
      uniforms.uSceneBinaryMasses.value.fromArray(masses);
    },
  }),
});
