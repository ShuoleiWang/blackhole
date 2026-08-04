/*
 * Real-time strong-field binary tracer for the WebGPU production path.
 *
 * Scientific boundary
 * -------------------
 * The inspiral metric is a frame-frozen, boosted superposition of two
 * Kerr-Schild metric contributions.  It is a strong-field analytic
 * approximation, not a numerical-relativity metric.  During merger it blends
 * to one Kerr-Schild remnant whose mass and spin are supplied by the
 * source-backed dynamics track.  The metric provider deliberately accepts
 * (coordinateTime, position), so a later slow-light implementation can advance
 * coordinateTime along the ray without replacing the geodesic integrator.
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
// A single symplectic-Euler kick above 3.5 M is not yet safe near the
// overlapping binary strong field. Keep the shader-side ceiling explicit and
// align every host tier with it until segment event localisation or an
// embedded error estimate can accept larger steps.
export const STRONG_FIELD_MAXIMUM_STEP_M = 3.5;

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
  0.018, // minimum coordinate-time step in M
  0.46, // far-zone maximum step in M
  3.5, // distance from a horizon that activates critical sampling
  0.08, // fail-closed Hamiltonian residual threshold
]);

const DEFAULT_STRONG_FIELD_DOMAIN = Object.freeze([
  96, // escape sphere radius in M
  240, // maximum coordinate lookback in M
  0.035, // horizon capture padding in M
  72, // extra iterations available only after entering the critical zone
]);

const DEFAULT_STRONG_FIELD_DIAGNOSTICS = Object.freeze([
  4, // maximum displayed frequency shift
  180, // logarithmic residual visualisation scale
  0.22, // unresolved diagnostic brightness
  1, // model flags / provider version marker
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

export const strongFieldBinaryTraceFragmentWGSL = /* wgsl */ `
diagnostic(off, derivative_uniformity);

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const MAX_STRONG_STEPS: i32 = 320;
const RAY_UNRESOLVED: u32 = 0u;
const RAY_CAPTURED: u32 = 1u;
const RAY_ESCAPED: u32 = 2u;
const DUAL_EPSILON: f32 = 1.0e-12;
// Render pipelines specialize this override to 0=binary, 1=remnant, or
// 2=transition.  The default keeps the complete provider available to the GPU
// probe and any consumer that does not opt into pipeline specialization.
override SPACETIME_PHASE_MODE: i32 = -1;

struct Params {
  resolutionTimeMass: vec4<f32>,
  renderControls: vec4<f32>,
  cameraPosRadius: vec4<f32>,
  cameraForwardFov: vec4<f32>,
  cameraRightSkyRotation: vec4<f32>,
  cameraUpDiskOuter: vec4<f32>,
  postMotionFrame: vec4<f32>,
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
  sceneStrongQuality: vec4<f32>,
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

struct DualVector3 {
  x: Dual3,
  y: Dual3,
  z: Dual3,
};

struct DualMatrix3 {
  c0: DualVector3,
  c1: DualVector3,
  c2: DualVector3,
};

struct HoleContribution {
  g00: Dual3,
  g0: DualVector3,
  spatial: DualMatrix3,
  horizonDistance: f32,
  cartesianRadius: Dual3,
  curvatureScale: f32,
  regularized: f32,
};

struct ADMFields {
  lapse: f32,
  lapseGradient: vec3<f32>,
  shift: vec3<f32>,
  shiftDerivativeX: vec3<f32>,
  shiftDerivativeY: vec3<f32>,
  shiftDerivativeZ: vec3<f32>,
  inverseSpatialMetric: mat3x3<f32>,
  inverseMetricDerivativeX: mat3x3<f32>,
  inverseMetricDerivativeY: mat3x3<f32>,
  inverseMetricDerivativeZ: mat3x3<f32>,
  spatialMetric: mat3x3<f32>,
  horizonDistance: f32,
  curvatureScale: f32,
  valid: f32,
};

struct HamiltonianRhs {
  velocity: vec3<f32>,
  momentumRate: vec3<f32>,
  eulerianFrequency: f32,
  reducedHamiltonian: f32,
  nullResidual: f32,
  valid: f32,
};

struct HamiltonianKinematics {
  velocity: vec3<f32>,
  eulerianFrequency: f32,
  reducedHamiltonian: f32,
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
  terminationReason: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tSky: texture_2d<f32>;
@group(0) @binding(2) var skySampler: sampler;

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

fn dualConstant(value: f32) -> Dual3 {
  var result: Dual3;
  result.value = value;
  result.gradient = vec3<f32>(0.0);
  return result;
}

fn dualVariable(value: f32, gradient: vec3<f32>) -> Dual3 {
  var result: Dual3;
  result.value = value;
  result.gradient = gradient;
  return result;
}

fn dualAdd(a: Dual3, b: Dual3) -> Dual3 {
  var result: Dual3;
  result.value = a.value + b.value;
  result.gradient = a.gradient + b.gradient;
  return result;
}

fn dualSub(a: Dual3, b: Dual3) -> Dual3 {
  var result: Dual3;
  result.value = a.value - b.value;
  result.gradient = a.gradient - b.gradient;
  return result;
}

fn dualScale(a: Dual3, scale: f32) -> Dual3 {
  var result: Dual3;
  result.value = a.value * scale;
  result.gradient = a.gradient * scale;
  return result;
}

fn dualMul(a: Dual3, b: Dual3) -> Dual3 {
  var result: Dual3;
  result.value = a.value * b.value;
  result.gradient = a.gradient * b.value + b.gradient * a.value;
  return result;
}

fn dualReciprocal(a: Dual3) -> Dual3 {
  let denominator = max(abs(a.value), DUAL_EPSILON);
  let signedValue = select(-denominator, denominator, a.value >= 0.0);
  var result: Dual3;
  result.value = 1.0 / signedValue;
  result.gradient = -a.gradient / (signedValue * signedValue);
  return result;
}

fn dualDiv(a: Dual3, b: Dual3) -> Dual3 {
  return dualMul(a, dualReciprocal(b));
}

fn dualSqrt(a: Dual3) -> Dual3 {
  let root = sqrt(max(a.value, DUAL_EPSILON));
  var result: Dual3;
  result.value = root;
  result.gradient = a.gradient / (2.0 * root);
  return result;
}

fn dualVectorConstant(value: vec3<f32>) -> DualVector3 {
  var result: DualVector3;
  result.x = dualConstant(value.x);
  result.y = dualConstant(value.y);
  result.z = dualConstant(value.z);
  return result;
}

fn dualPosition(value: vec3<f32>) -> DualVector3 {
  var result: DualVector3;
  result.x = dualVariable(value.x, vec3<f32>(1.0, 0.0, 0.0));
  result.y = dualVariable(value.y, vec3<f32>(0.0, 1.0, 0.0));
  result.z = dualVariable(value.z, vec3<f32>(0.0, 0.0, 1.0));
  return result;
}

fn dualVectorAdd(a: DualVector3, b: DualVector3) -> DualVector3 {
  var result: DualVector3;
  result.x = dualAdd(a.x, b.x);
  result.y = dualAdd(a.y, b.y);
  result.z = dualAdd(a.z, b.z);
  return result;
}

fn dualVectorSub(a: DualVector3, b: DualVector3) -> DualVector3 {
  var result: DualVector3;
  result.x = dualSub(a.x, b.x);
  result.y = dualSub(a.y, b.y);
  result.z = dualSub(a.z, b.z);
  return result;
}

fn dualVectorScale(a: DualVector3, scale: f32) -> DualVector3 {
  var result: DualVector3;
  result.x = dualScale(a.x, scale);
  result.y = dualScale(a.y, scale);
  result.z = dualScale(a.z, scale);
  return result;
}

fn dualVectorScaleDual(a: DualVector3, scale: Dual3) -> DualVector3 {
  var result: DualVector3;
  result.x = dualMul(a.x, scale);
  result.y = dualMul(a.y, scale);
  result.z = dualMul(a.z, scale);
  return result;
}

fn dualDot(a: DualVector3, b: DualVector3) -> Dual3 {
  return dualAdd(
    dualAdd(dualMul(a.x, b.x), dualMul(a.y, b.y)),
    dualMul(a.z, b.z)
  );
}

fn dualDotConstant(a: DualVector3, b: vec3<f32>) -> Dual3 {
  return dualAdd(
    dualAdd(dualScale(a.x, b.x), dualScale(a.y, b.y)),
    dualScale(a.z, b.z)
  );
}

fn dualLength(value: DualVector3) -> Dual3 {
  return dualSqrt(dualDot(value, value));
}

fn dualVectorNormalize(value: DualVector3) -> DualVector3 {
  return dualVectorScaleDual(value, dualReciprocal(dualLength(value)));
}

fn dualVectorValue(value: DualVector3) -> vec3<f32> {
  return vec3<f32>(value.x.value, value.y.value, value.z.value);
}

fn dualVectorDerivative(value: DualVector3, axis: u32) -> vec3<f32> {
  return vec3<f32>(
    value.x.gradient[axis],
    value.y.gradient[axis],
    value.z.gradient[axis]
  );
}

fn zeroDualMatrix() -> DualMatrix3 {
  var result: DualMatrix3;
  result.c0 = dualVectorConstant(vec3<f32>(0.0));
  result.c1 = dualVectorConstant(vec3<f32>(0.0));
  result.c2 = dualVectorConstant(vec3<f32>(0.0));
  return result;
}

fn identityDualMatrix() -> DualMatrix3 {
  var result: DualMatrix3;
  result.c0 = dualVectorConstant(vec3<f32>(1.0, 0.0, 0.0));
  result.c1 = dualVectorConstant(vec3<f32>(0.0, 1.0, 0.0));
  result.c2 = dualVectorConstant(vec3<f32>(0.0, 0.0, 1.0));
  return result;
}

fn addDualMatrix(a: DualMatrix3, b: DualMatrix3) -> DualMatrix3 {
  var result: DualMatrix3;
  result.c0 = dualVectorAdd(a.c0, b.c0);
  result.c1 = dualVectorAdd(a.c1, b.c1);
  result.c2 = dualVectorAdd(a.c2, b.c2);
  return result;
}

fn scaleDualMatrixDual(
  value: DualMatrix3,
  scale: Dual3
) -> DualMatrix3 {
  var result: DualMatrix3;
  result.c0 = dualVectorScaleDual(value.c0, scale);
  result.c1 = dualVectorScaleDual(value.c1, scale);
  result.c2 = dualVectorScaleDual(value.c2, scale);
  return result;
}

fn dualMatrixValue(value: DualMatrix3) -> mat3x3<f32> {
  return mat3x3<f32>(
    dualVectorValue(value.c0),
    dualVectorValue(value.c1),
    dualVectorValue(value.c2)
  );
}

fn dualMatrixDerivative(value: DualMatrix3, axis: u32) -> mat3x3<f32> {
  return mat3x3<f32>(
    dualVectorDerivative(value.c0, axis),
    dualVectorDerivative(value.c1, axis),
    dualVectorDerivative(value.c2, axis)
  );
}

fn inverse3x3(value: mat3x3<f32>) -> mat3x3<f32> {
  let row0 = cross(value[1], value[2]);
  let row1 = cross(value[2], value[0]);
  let row2 = cross(value[0], value[1]);
  let determinant = dot(value[0], row0);
  let safeDeterminant = select(
    -max(abs(determinant), 1.0e-8),
    max(abs(determinant), 1.0e-8),
    determinant >= 0.0
  );
  return mat3x3<f32>(
    vec3<f32>(row0.x, row1.x, row2.x) / safeDeterminant,
    vec3<f32>(row0.y, row1.y, row2.y) / safeDeterminant,
    vec3<f32>(row0.z, row1.z, row2.z) / safeDeterminant
  );
}

fn outerDual(vector: DualVector3, amplitude: Dual3) -> DualMatrix3 {
  var result: DualMatrix3;
  result.c0 = dualVectorScaleDual(
    vector,
    dualMul(amplitude, vector.x)
  );
  result.c1 = dualVectorScaleDual(
    vector,
    dualMul(amplitude, vector.y)
  );
  result.c2 = dualVectorScaleDual(
    vector,
    dualMul(amplitude, vector.z)
  );
  return result;
}

fn zeroHoleContribution() -> HoleContribution {
  var result: HoleContribution;
  result.g00 = dualConstant(0.0);
  result.g0 = dualVectorConstant(vec3<f32>(0.0));
  result.spatial = zeroDualMatrix();
  result.horizonDistance = 1.0e6;
  result.cartesianRadius = dualConstant(1.0e6);
  result.curvatureScale = 0.0;
  result.regularized = 0.0;
  return result;
}

// One instantaneously Lorentz-boosted Kerr-Schild term.  Position, velocity,
// and the arbitrary dimensionless spin vector are supplied by the declared
// PN/EOB coordinate adapter; SXS horizon centroids never enter this function.
fn boostedKerrSchildContribution(
  position: DualVector3,
  centre: vec3<f32>,
  velocity: vec3<f32>,
  massInput: f32,
  dimensionlessSpin: vec3<f32>,
  activeFlag: f32
) -> HoleContribution {
  let mass = max(massInput, 1.0e-5);
  if (activeFlag < 0.5 || massInput <= 0.0) {
    return zeroHoleContribution();
  }

  let speedSquared = dot(velocity, velocity);
  if (speedSquared >= 0.9999) {
    var invalid = zeroHoleContribution();
    invalid.regularized = 1.0;
    return invalid;
  }
  let boostGamma = inverseSqrt(max(1.0 - speedSquared, 1.0e-5));
  let displacement = dualVectorSub(
    position,
    dualVectorConstant(centre)
  );
  let velocityProjection = dualDotConstant(displacement, velocity);
  let contractionCoefficient = select(
    0.5,
    (boostGamma - 1.0) / max(speedSquared, 1.0e-12),
    speedSquared > 1.0e-12
  );
  let restPosition = dualVectorAdd(
    displacement,
    dualVectorScaleDual(
      dualVectorConstant(velocity),
      dualScale(velocityProjection, contractionCoefficient)
    )
  );
  let rhoSquared = dualDot(restPosition, restPosition);
  let spinNorm = length(dimensionlessSpin);
  let safeChi = dimensionlessSpin
    * min(0.998, spinNorm) / max(spinNorm, 1.0e-12);
  let spin = safeChi * mass;
  let radiusFloor = max(
    params.spacetimeLimits.z,
    params.spacetimeControl.w * mass
  );
  var kerrRadius: Dual3;
  var kerrH: Dual3;
  var restDirection: DualVector3;
  var regularized = select(0.0, 1.0, spinNorm >= 0.999);
  if (spinNorm < 1.0e-5) {
    // The source binary is non-spinning.  This exact Schwarzschild
    // Kerr-Schild branch removes most dual-number work during inspiral.
    kerrRadius = dualSqrt(rhoSquared);
    if (kerrRadius.value < radiusFloor) {
      kerrRadius = dualConstant(radiusFloor);
      regularized = 1.0;
    }
    kerrH = dualScale(dualReciprocal(kerrRadius), mass);
    restDirection = dualVectorScaleDual(
      restPosition,
      dualReciprocal(kerrRadius)
    );
  } else {
    let spinDotPosition = dualDotConstant(restPosition, spin);
    let spinSquared = dot(spin, spin);
    let radialDiscriminant = dualAdd(
      dualMul(
        dualSub(rhoSquared, dualConstant(spinSquared)),
        dualSub(rhoSquared, dualConstant(spinSquared))
      ),
      dualScale(dualMul(spinDotPosition, spinDotPosition), 4.0)
    );
    let kerrRadiusSquared = dualScale(
      dualAdd(
        dualSub(rhoSquared, dualConstant(spinSquared)),
        dualSqrt(radialDiscriminant)
      ),
      0.5
    );
    kerrRadius = dualSqrt(kerrRadiusSquared);
    if (kerrRadius.value < radiusFloor) {
      kerrRadius = dualConstant(radiusFloor);
      regularized = 1.0;
    }
    let radiusSquared = dualMul(kerrRadius, kerrRadius);
    let radiusCubed = dualMul(radiusSquared, kerrRadius);
    let radiusFourth = dualMul(radiusSquared, radiusSquared);
    let kerrDenominator = dualAdd(
      radiusFourth,
      dualMul(spinDotPosition, spinDotPosition)
    );
    kerrH = dualScale(
      dualDiv(radiusCubed, kerrDenominator),
      mass
    );

    // l = [r x + x cross a + a(a.x)/r] / (r^2 + a^2)
    var positionCrossSpin: DualVector3;
    positionCrossSpin.x = dualSub(
      dualScale(restPosition.y, spin.z),
      dualScale(restPosition.z, spin.y)
    );
    positionCrossSpin.y = dualSub(
      dualScale(restPosition.z, spin.x),
      dualScale(restPosition.x, spin.z)
    );
    positionCrossSpin.z = dualSub(
      dualScale(restPosition.x, spin.y),
      dualScale(restPosition.y, spin.x)
    );
    let numerator = dualVectorAdd(
      dualVectorAdd(
        dualVectorScaleDual(restPosition, kerrRadius),
        positionCrossSpin
      ),
      dualVectorScaleDual(
        dualVectorConstant(spin),
        dualDiv(spinDotPosition, kerrRadius)
      )
    );
    restDirection = dualVectorScaleDual(
      numerator,
      dualReciprocal(dualAdd(radiusSquared, dualConstant(spinSquared)))
    );
  }
  restDirection = dualVectorNormalize(restDirection);

  let velocityDotDirection = dualDotConstant(restDirection, velocity);
  let transformedFactor = dualSub(
    dualScale(
      velocityDotDirection,
      contractionCoefficient
    ),
    dualConstant(boostGamma)
  );
  let boostedDirection = dualVectorAdd(
    restDirection,
    dualVectorScaleDual(
      dualVectorConstant(velocity),
      transformedFactor
    )
  );
  let boostedTime = dualScale(
    dualSub(dualConstant(1.0), velocityDotDirection),
    boostGamma
  );
  let maximumH = max(params.spacetimeLimits.w, 1.0);
  if (kerrH.value > maximumH) {
    kerrH = dualConstant(maximumH);
    regularized = 1.0;
  }
  let amplitude = dualScale(kerrH, 2.0);

  var result: HoleContribution;
  result.g00 = dualMul(
    amplitude,
    dualMul(boostedTime, boostedTime)
  );
  result.g0 = dualVectorScaleDual(
    boostedDirection,
    dualMul(amplitude, boostedTime)
  );
  result.spatial = outerDual(boostedDirection, amplitude);
  let horizonRadius = mass * (
    1.0 + sqrt(max(1.0 - dot(safeChi, safeChi), 1.0e-5))
  );
  result.horizonDistance = kerrRadius.value - horizonRadius;
  result.cartesianRadius = dualLength(restPosition);
  result.curvatureScale = mass
    / max(kerrRadius.value * kerrRadius.value, 0.02);
  result.regularized = regularized;
  return result;
}

fn attenuationWeight(radius: Dual3) -> Dual3 {
  if (params.spacetimeLimits.x < 0.5) {
    return dualConstant(1.0);
  }
  let scale = max(params.spacetimeControl.z, 1.0e-5);
  let power = max(params.spacetimeLimits.y, 2.0);
  let ratio = dualScale(radius, 1.0 / scale);
  let safeRatio = max(ratio.value, 0.0);
  let powered = pow(safeRatio, power);
  let poweredGradient = power
    * pow(max(safeRatio, 1.0e-8), power - 1.0)
    * ratio.gradient;
  let exponential = exp(-powered);
  var result: Dual3;
  result.value = 1.0 - exponential;
  result.gradient = exponential * poweredGradient;
  return result;
}

// Unified strong-field spacetime provider.  PR5 samples it at one frozen frame
// time; its explicit (t, x) interface is retained for slow-light and NR data.
fn sampleSpacetime(
  coordinateTime: f32,
  positionValue: vec3<f32>
) -> ADMFields {
  let providerInput = SpacetimeProviderInput(coordinateTime, positionValue);
  let position = dualPosition(providerInput.position);
  // spacetimeControl.y is already quintic smootherstep(rawMergerBlend), packed
  // by the CPU provider.  This preserves a C2 metric transition.
  let blend = clamp(params.spacetimeControl.y, 0.0, 1.0);
  var holeA = zeroHoleContribution();
  var holeB = zeroHoleContribution();
  var remnant = zeroHoleContribution();
  var weightA = dualConstant(0.0);
  var weightB = dualConstant(0.0);
  var weightRemnant = dualConstant(0.0);

  // The endpoint phase is a frame-uniform value.  Branching here skips the
  // inactive Kerr-Schild providers and, in the remnant phase, both companion
  // attenuation evaluations.  The open interval retains the general C2 blend.
  if (
    SPACETIME_PHASE_MODE == 0
    || (SPACETIME_PHASE_MODE < 0 && blend == 0.0)
  ) {
    holeA = boostedKerrSchildContribution(
      position,
      params.bodyAPositionMass.xyz,
      params.bodyAVelocityActive.xyz,
      params.bodyAPositionMass.w,
      params.bodyASpin.xyz,
      params.bodyAVelocityActive.w
    );
    holeB = boostedKerrSchildContribution(
      position,
      params.bodyBPositionMass.xyz,
      params.bodyBVelocityActive.xyz,
      params.bodyBPositionMass.w,
      params.bodyBSpin.xyz,
      params.bodyBVelocityActive.w
    );
    weightA = dualScale(
      attenuationWeight(holeB.cartesianRadius),
      params.bodyAVelocityActive.w
    );
    weightB = dualScale(
      attenuationWeight(holeA.cartesianRadius),
      params.bodyBVelocityActive.w
    );
  } else if (
    SPACETIME_PHASE_MODE == 1
    || (SPACETIME_PHASE_MODE < 0 && blend == 1.0)
  ) {
    remnant = boostedKerrSchildContribution(
      position,
      params.remnantPositionMass.xyz,
      params.remnantVelocityActive.xyz,
      params.remnantPositionMass.w,
      params.remnantSpinBlend.xyz,
      params.remnantVelocityActive.w
    );
    weightRemnant = dualConstant(params.remnantVelocityActive.w);
  } else {
    let binaryWeight = 1.0 - blend;
    let binaryActive = select(0.0, 1.0, binaryWeight > 1.0e-6);
    let remnantActive = select(0.0, 1.0, blend > 1.0e-6);
    holeA = boostedKerrSchildContribution(
      position,
      params.bodyAPositionMass.xyz,
      params.bodyAVelocityActive.xyz,
      params.bodyAPositionMass.w,
      params.bodyASpin.xyz,
      params.bodyAVelocityActive.w * binaryActive
    );
    holeB = boostedKerrSchildContribution(
      position,
      params.bodyBPositionMass.xyz,
      params.bodyBVelocityActive.xyz,
      params.bodyBPositionMass.w,
      params.bodyBSpin.xyz,
      params.bodyBVelocityActive.w * binaryActive
    );
    remnant = boostedKerrSchildContribution(
      position,
      params.remnantPositionMass.xyz,
      params.remnantVelocityActive.xyz,
      params.remnantPositionMass.w,
      params.remnantSpinBlend.xyz,
      params.remnantVelocityActive.w * remnantActive
    );
    weightA = dualScale(
      attenuationWeight(holeB.cartesianRadius),
      binaryWeight * params.bodyAVelocityActive.w
    );
    weightB = dualScale(
      attenuationWeight(holeA.cartesianRadius),
      binaryWeight * params.bodyBVelocityActive.w
    );
    weightRemnant = dualConstant(
      blend * params.remnantVelocityActive.w
    );
  }

  var g00 = dualConstant(-1.0);
  var g0 = dualVectorConstant(vec3<f32>(0.0));
  var spatial = identityDualMatrix();
  g00 = dualAdd(g00, dualMul(holeA.g00, weightA));
  g00 = dualAdd(g00, dualMul(holeB.g00, weightB));
  g00 = dualAdd(g00, dualMul(remnant.g00, weightRemnant));
  g0 = dualVectorAdd(
    g0,
    dualVectorScaleDual(holeA.g0, weightA)
  );
  g0 = dualVectorAdd(
    g0,
    dualVectorScaleDual(holeB.g0, weightB)
  );
  g0 = dualVectorAdd(
    g0,
    dualVectorScaleDual(remnant.g0, weightRemnant)
  );
  spatial = addDualMatrix(
    spatial,
    scaleDualMatrixDual(holeA.spatial, weightA)
  );
  spatial = addDualMatrix(
    spatial,
    scaleDualMatrixDual(holeB.spatial, weightB)
  );
  spatial = addDualMatrix(
    spatial,
    scaleDualMatrixDual(remnant.spatial, weightRemnant)
  );

  let gammaCovariant = dualMatrixValue(spatial);
  let determinant = dot(
    gammaCovariant[0],
    cross(gammaCovariant[1], gammaCovariant[2])
  );
  let gammaInverse = inverse3x3(gammaCovariant);
  let derivativeCovariantX = dualMatrixDerivative(spatial, 0u);
  let derivativeCovariantY = dualMatrixDerivative(spatial, 1u);
  let derivativeCovariantZ = dualMatrixDerivative(spatial, 2u);
  let inverseProductX =
    gammaInverse * derivativeCovariantX * gammaInverse;
  let inverseProductY =
    gammaInverse * derivativeCovariantY * gammaInverse;
  let inverseProductZ =
    gammaInverse * derivativeCovariantZ * gammaInverse;
  let derivativeInverseX = mat3x3<f32>(
    -inverseProductX[0],
    -inverseProductX[1],
    -inverseProductX[2]
  );
  let derivativeInverseY = mat3x3<f32>(
    -inverseProductY[0],
    -inverseProductY[1],
    -inverseProductY[2]
  );
  let derivativeInverseZ = mat3x3<f32>(
    -inverseProductZ[0],
    -inverseProductZ[1],
    -inverseProductZ[2]
  );
  let betaCovariant = dualVectorValue(g0);
  let betaCovariantDerivativeX = dualVectorDerivative(g0, 0u);
  let betaCovariantDerivativeY = dualVectorDerivative(g0, 1u);
  let betaCovariantDerivativeZ = dualVectorDerivative(g0, 2u);
  let shift = gammaInverse * betaCovariant;
  let shiftDerivativeX = derivativeInverseX * betaCovariant
    + gammaInverse * betaCovariantDerivativeX;
  let shiftDerivativeY = derivativeInverseY * betaCovariant
    + gammaInverse * betaCovariantDerivativeY;
  let shiftDerivativeZ = derivativeInverseZ * betaCovariant
    + gammaInverse * betaCovariantDerivativeZ;
  let lapseSquared = dot(betaCovariant, shift) - g00.value;
  let lapse = sqrt(max(lapseSquared, 1.0e-8));
  let lapseSquaredGradient = vec3<f32>(
    dot(betaCovariantDerivativeX, shift)
      + dot(betaCovariant, shiftDerivativeX) - g00.gradient.x,
    dot(betaCovariantDerivativeY, shift)
      + dot(betaCovariant, shiftDerivativeY) - g00.gradient.y,
    dot(betaCovariantDerivativeZ, shift)
      + dot(betaCovariant, shiftDerivativeZ) - g00.gradient.z
  );

  var horizonDistance = 1.0e6;
  if (weightA.value > 1.0e-4) {
    horizonDistance = min(horizonDistance, holeA.horizonDistance);
  }
  if (weightB.value > 1.0e-4) {
    horizonDistance = min(horizonDistance, holeB.horizonDistance);
  }
  if (weightRemnant.value > 1.0e-4) {
    horizonDistance = min(horizonDistance, remnant.horizonDistance);
  }
  let activeRegularization = max(
    max(
      weightA.value * holeA.regularized,
      weightB.value * holeB.regularized
    ),
    weightRemnant.value * remnant.regularized
  );

  var result: ADMFields;
  result.lapse = lapse;
  result.lapseGradient = lapseSquaredGradient / (2.0 * lapse);
  result.shift = shift;
  result.shiftDerivativeX = shiftDerivativeX;
  result.shiftDerivativeY = shiftDerivativeY;
  result.shiftDerivativeZ = shiftDerivativeZ;
  result.inverseSpatialMetric = gammaInverse;
  result.inverseMetricDerivativeX = derivativeInverseX;
  result.inverseMetricDerivativeY = derivativeInverseY;
  result.inverseMetricDerivativeZ = derivativeInverseZ;
  result.spatialMetric = gammaCovariant;
  result.horizonDistance = horizonDistance;
  result.curvatureScale = max(
    max(
      weightA.value * holeA.curvatureScale,
      weightB.value * holeB.curvatureScale
    ),
    weightRemnant.value * remnant.curvatureScale
  );
  result.valid = select(0.0, 1.0,
    determinant > 1.0e-7
    && lapseSquared > 1.0e-8
    && activeRegularization < 0.5
    && finiteScalar(lapse)
    && finiteVector(shift)
  );
  return result;
}

fn metricDerivative(
  fields: ADMFields,
  axis: i32
) -> mat3x3<f32> {
  if (axis == 0) {
    return fields.inverseMetricDerivativeX;
  }
  if (axis == 1) {
    return fields.inverseMetricDerivativeY;
  }
  return fields.inverseMetricDerivativeZ;
}

fn shiftDerivative(fields: ADMFields, axis: i32) -> vec3<f32> {
  if (axis == 0) {
    return fields.shiftDerivativeX;
  }
  if (axis == 1) {
    return fields.shiftDerivativeY;
  }
  return fields.shiftDerivativeZ;
}

// Reduced 3+1 null Hamiltonian:
//   H(x,p) = alpha sqrt(gamma^ij p_i p_j) - beta^i p_i = -p_t
// The metric jet above supplies analytic spatial derivatives in one provider
// evaluation, avoiding the seven metric samples required by finite differences.
fn hamiltonianRhs(
  fields: ADMFields,
  momentum: vec3<f32>,
  conservedEnergy: f32
) -> HamiltonianRhs {
  let raisedMomentum = fields.inverseSpatialMetric * momentum;
  let qSquared = dot(momentum, raisedMomentum);
  let q = sqrt(max(qSquared, 1.0e-12));
  let reducedHamiltonian = fields.lapse * q
    - dot(fields.shift, momentum);
  var momentumRate = vec3<f32>(0.0);
  for (var axis: i32 = 0; axis < 3; axis = axis + 1) {
    let inverseDerivative = metricDerivative(fields, axis);
    let shiftGradient = shiftDerivative(fields, axis);
    let hamiltonianGradient =
      fields.lapseGradient[axis] * q
      + 0.5 * fields.lapse
        * dot(momentum, inverseDerivative * momentum) / q
      - dot(shiftGradient, momentum);
    momentumRate[axis] = -hamiltonianGradient;
  }
  let velocity = fields.lapse * raisedMomentum / q - fields.shift;
  let nullNumerator =
    -(conservedEnergy + dot(fields.shift, momentum))
      * (conservedEnergy + dot(fields.shift, momentum))
      / max(fields.lapse * fields.lapse, 1.0e-12)
    + qSquared;
  let residual = abs(nullNumerator) / max(qSquared, 1.0e-8);

  var result: HamiltonianRhs;
  result.velocity = velocity;
  result.momentumRate = momentumRate;
  result.eulerianFrequency = q;
  result.reducedHamiltonian = reducedHamiltonian;
  result.nullResidual = residual;
  result.valid = select(0.0, 1.0,
    fields.valid > 0.5
    && qSquared > 1.0e-12
    && finiteVector(velocity)
    && finiteVector(momentumRate)
    && finiteScalar(residual)
  );
  return result;
}

fn hamiltonianKinematics(
  fields: ADMFields,
  momentum: vec3<f32>
) -> HamiltonianKinematics {
  let raisedMomentum = fields.inverseSpatialMetric * momentum;
  let qSquared = dot(momentum, raisedMomentum);
  let q = sqrt(max(qSquared, 1.0e-12));
  let velocity = fields.lapse * raisedMomentum / q - fields.shift;
  let reducedHamiltonian = fields.lapse * q
    - dot(fields.shift, momentum);
  var result: HamiltonianKinematics;
  result.velocity = velocity;
  result.eulerianFrequency = q;
  result.reducedHamiltonian = reducedHamiltonian;
  result.valid = select(0.0, 1.0,
    fields.valid > 0.5
    && qSquared > 1.0e-12
    && finiteVector(velocity)
    && finiteScalar(reducedHamiltonian)
  );
  return result;
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
  result.terminationReason = 0.0;
  return result;
}

fn spatialMetricDot(
  fields: ADMFields,
  a: vec3<f32>,
  b: vec3<f32>
) -> f32 {
  return dot(a, fields.spatialMetric * b);
}

fn spatialMetricNormalize(
  fields: ADMFields,
  value: vec3<f32>
) -> vec3<f32> {
  return value * inverseSqrt(max(
    spatialMetricDot(fields, value, value),
    1.0e-14
  ));
}

// The camera FOV lives in the observer's local orthonormal spatial frame, not
// in Euclidean coordinate components.  Metric Gram-Schmidt removes the
// systematic off-axis error that otherwise remains even at moderately large
// observer radii.
fn observerCameraDirection(
  fields: ADMFields,
  screen: vec2<f32>,
  tanHalfFov: f32
) -> vec3<f32> {
  let forward = spatialMetricNormalize(
    fields,
    params.cameraForwardFov.xyz
  );
  let rawRight = params.cameraRightSkyRotation.xyz
    - forward * spatialMetricDot(
      fields,
      forward,
      params.cameraRightSkyRotation.xyz
    );
  let right = spatialMetricNormalize(fields, rawRight);
  let rawUp = params.cameraUpDiskOuter.xyz
    - forward * spatialMetricDot(
      fields,
      forward,
      params.cameraUpDiskOuter.xyz
    )
    - right * spatialMetricDot(
      fields,
      right,
      params.cameraUpDiskOuter.xyz
    );
  let up = spatialMetricNormalize(fields, rawUp);
  return spatialMetricNormalize(
    fields,
    forward + tanHalfFov * (screen.x * right + screen.y * up)
  );
}

// Analytic outgoing O(M/r) monopole tail from the finite escape sphere to
// future null infinity.  This permits a smaller real-time domain without
// freezing in a radius-dependent sky direction.  The near-axis limit is
// returned unchanged because its transverse impulse tends smoothly to zero.
fn asymptoticEscapeDirection(
  position: vec3<f32>,
  velocity: vec3<f32>
) -> vec3<f32> {
  let direction = safeNormalize(velocity);
  let radius = length(position);
  let longitudinal = dot(position, direction);
  let impact = position - longitudinal * direction;
  let impactSquared = dot(impact, impact);
  if (
    longitudinal <= 0.0
    || radius <= 1.0e-5
    || impactSquared <= 1.0e-8
  ) {
    return direction;
  }
  let blend = clamp(params.spacetimeControl.y, 0.0, 1.0);
  let binaryMass = (
    params.bodyAPositionMass.w * params.bodyAVelocityActive.w
    + params.bodyBPositionMass.w * params.bodyBVelocityActive.w
  );
  let remnantMass = (
    params.remnantPositionMass.w * params.remnantVelocityActive.w
  );
  let asymptoticMass = max(
    (1.0 - blend) * binaryMass + blend * remnantMass,
    0.0
  );
  let remainingFraction = clamp(
    1.0 - longitudinal / radius,
    0.0,
    1.0
  );
  let correction = -2.0 * asymptoticMass * impact
    * remainingFraction / impactSquared;
  if (!finiteVector(correction) || length(correction) > 0.25) {
    return direction;
  }
  return safeNormalize(direction + correction);
}

fn numericalCaptureGuard() -> f32 {
  // A failed coordinate-time branch inside the innermost photon shell cannot
  // be continued reliably with the reduced real-time integrator.  Excise it
  // as captured only inside a conservative analytic bound: 0.95M above the
  // non-spinning individual horizons (just inside r_ph-r_+=1M), tapering to
  // 0.25M for the chi≈0.69 remnant (inside its prograde photon shell).
  // This branch is consulted only after the positive-energy projection fails;
  // normally integrated rays still use the much tighter tier padding.
  return mix(
    0.95,
    0.25,
    clamp(params.spacetimeControl.y, 0.0, 1.0)
  );
}

fn traceStrongField(screen: vec2<f32>, tanHalfFov: f32) -> RayResult {
  var result = unresolvedResult();
  var position = params.cameraPosRadius.xyz;
  let frameTime = params.spacetimeControl.x;
  let observerFields = sampleSpacetime(frameTime, position);
  if (observerFields.valid < 0.5) {
    result.terminationReason = 2.0;
    return result;
  }

  // initialDirection points from the camera into the viewed scene.  A photon
  // that actually arrives at the camera has the opposite future-directed
  // Eulerian spatial momentum.  Store that arriving future covector here,
  // then integrate Hamilton's equations with a negative parameter step below.
  // This distinction is invisible in static Schwarzschild paths but is
  // essential for the sign of frame dragging and Doppler/gravitational shift
  // in boosted or Kerr spacetimes.
  let initialDirection = observerCameraDirection(
    observerFields,
    screen,
    tanHalfFov
  );
  var momentum = -(observerFields.spatialMetric * initialDirection);
  let initialQ = sqrt(max(
    dot(momentum, observerFields.inverseSpatialMetric * momentum),
    1.0e-12
  ));
  momentum = momentum / initialQ;
  let initialRaised = observerFields.inverseSpatialMetric * momentum;
  let observerQ = sqrt(max(dot(momentum, initialRaised), 1.0e-12));
  let conservedEnergy = observerFields.lapse * observerQ
    - dot(observerFields.shift, momentum);
  if (!finiteScalar(conservedEnergy) || conservedEnergy <= 1.0e-6) {
    result.terminationReason = 2.0;
    return result;
  }

  let minimumStep = clamp(params.sceneStrongIntegrator.x, 0.004, 0.12);
  let maximumStep = clamp(
    params.sceneStrongIntegrator.y,
    minimumStep,
    ${STRONG_FIELD_MAXIMUM_STEP_M.toFixed(1)}
  );
  let criticalDistance = max(params.sceneStrongIntegrator.z, 0.2);
  let residualFail = clamp(params.sceneStrongIntegrator.w, 0.002, 0.5);
  let escapeRadius = max(
    params.sceneStrongDomain.x,
    length(position) + 8.0
  );
  let maximumLookback = max(params.sceneStrongDomain.y, 16.0);
  let capturePadding = max(params.sceneStrongDomain.z, 0.0);
  let baseBudget = clamp(
    i32(params.renderControls.w),
    24,
    MAX_STRONG_STEPS
  );
  let criticalBonus = clamp(
    i32(params.sceneStrongDomain.w),
    0,
    MAX_STRONG_STEPS - baseBudget
  );
  var allowedSteps = baseBudget;
  var enteredDomain = false;
  var lookback = 0.0;
  var maximumResidual = 0.0;
  var minimumHorizonDistance = 1.0e6;

  for (
    var stepIndex: i32 = 0;
    stepIndex < MAX_STRONG_STEPS;
    stepIndex = stepIndex + 1
  ) {
    if (stepIndex >= allowedSteps) {
      break;
    }
    let fields = sampleSpacetime(frameTime, position);
    result.iterations = f32(stepIndex + 1);
    minimumHorizonDistance = min(
      minimumHorizonDistance,
      fields.horizonDistance
    );

    if (fields.horizonDistance <= capturePadding) {
      result.outcome = RAY_CAPTURED;
      result.lookback = lookback;
      result.hamiltonianResidual = maximumResidual;
      result.minimumHorizonDistance = minimumHorizonDistance;
      return result;
    }
    var rhs = hamiltonianRhs(fields, momentum, conservedEnergy);
    maximumResidual = max(maximumResidual, rhs.nullResidual);
    if (
      fields.valid < 0.5
      || rhs.valid < 0.5
    ) {
      result.terminationReason = 2.0;
      result.hamiltonianResidual = max(maximumResidual, 1.0);
      result.minimumHorizonDistance = minimumHorizonDistance;
      return result;
    }
    if (rhs.nullResidual > residualFail) {
      // A kick-drift step projects momentum on the old spatial slice.  At the
      // next position the same covector can sit slightly off the new local
      // null-energy surface even though it is finite and recoverable.  Correct
      // that drift before taking another derivative; retain the pre-correction
      // residual above as the numerical diagnostic and fail closed when the
      // required correction is too large for the active quality tier.
      let preProjection = hamiltonianKinematics(fields, momentum);
      let correction = conservedEnergy / max(
        preProjection.reducedHamiltonian,
        1.0e-8
      );
      // The emergency real-time tier deliberately uses a permissive
      // constraint projection.  Balanced/fine tiers retain the tight,
      // tolerance-derived correction gate; all tiers still fail on a
      // non-positive/non-finite branch and report the pre-projection residual.
      let correctionLimit = select(
        exp(2.5 * residualFail),
        1.0e4,
        residualFail >= 0.20
      );
      if (
        preProjection.valid < 0.5
        || !finiteScalar(correction)
        || correction < 1.0 / correctionLimit
        || correction > correctionLimit
      ) {
        if (
          fields.horizonDistance
            <= max(capturePadding, numericalCaptureGuard())
        ) {
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
      momentum = momentum * correction;
      rhs = hamiltonianRhs(fields, momentum, conservedEnergy);
      if (rhs.valid < 0.5 || rhs.nullResidual > 1.0e-3) {
        result.terminationReason = 3.0;
        result.hamiltonianResidual = max(maximumResidual, 1.0);
        result.minimumHorizonDistance = minimumHorizonDistance;
        return result;
      }
    }
    if (fields.horizonDistance < criticalDistance) {
      allowedSteps = min(
        MAX_STRONG_STEPS,
        max(allowedSteps, baseBudget + criticalBonus)
      );
    }

    let radius = length(position);
    enteredDomain = enteredDomain || radius < escapeRadius * 0.82;
    if (
      enteredDomain
      && radius >= escapeRadius
      && dot(position, -rhs.velocity) > 0.0
    ) {
      result.outcome = RAY_ESCAPED;
      result.escapeDirection = asymptoticEscapeDirection(
        position,
        -rhs.velocity
      );
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
    if (lookback >= maximumLookback) {
      result.terminationReason = 4.0;
      break;
    }

    // One analytic metric jet per step. Small steps are reserved for the
    // horizon/photon region; the far field quickly reaches maximumStep.
    let criticalRatio = clamp(
      max(fields.horizonDistance, 0.0) / criticalDistance,
      0.0,
      1.0
    );
    let curvatureLimiter = inverseSqrt(
      1.0 + 7.0 * max(fields.curvatureScale, 0.0)
    );
    let stepCurveExponent = clamp(
      params.sceneStrongDiagnostics.w,
      0.5,
      2.5
    );
    var stepSize = mix(
      minimumStep,
      maximumStep,
      pow(criticalRatio, stepCurveExponent)
    ) * curvatureLimiter;
    stepSize = clamp(
      min(stepSize, maximumLookback - lookback),
      minimumStep,
      maximumStep
    );
    // Backward symplectic-Euler kick/drift. Hamiltonian homogeneity supplies a
    // cheap exact energy-surface projection at the current position.
    let momentumBeforeKick = momentum;
    var acceptedStepSize = stepSize;
    momentum = momentumBeforeKick - acceptedStepSize * rhs.momentumRate;
    var projectedKinematics = hamiltonianKinematics(fields, momentum);
    var rawEnergyScale = conservedEnergy / max(
      projectedKinematics.reducedHamiltonian,
      1.0e-8
    );
    let stepCorrectionLimit = select(
      exp(2.5 * residualFail),
      1.0e4,
      residualFail >= 0.20
    );
    // Large far-field steps are normally the main M3 Pro speedup, but a ray
    // entering the overlapping binary strong field can cross the positive
    // Hamiltonian-energy branch in one kick. Retry only those pixels with a
    // quarter step before declaring failure. This preserves the one-metric-jet
    // fast path and avoids globally taxing rays that remain well conditioned.
    if (
      projectedKinematics.valid < 0.5
      || !finiteScalar(rawEnergyScale)
      || rawEnergyScale <= 0.0
      || rawEnergyScale > stepCorrectionLimit
    ) {
      acceptedStepSize = max(minimumStep, stepSize * 0.25);
      momentum = (
        momentumBeforeKick - acceptedStepSize * rhs.momentumRate
      );
      projectedKinematics = hamiltonianKinematics(fields, momentum);
      rawEnergyScale = conservedEnergy / max(
        projectedKinematics.reducedHamiltonian,
        1.0e-8
      );
    }
    if (
      projectedKinematics.valid < 0.5
      || !finiteScalar(rawEnergyScale)
      || rawEnergyScale <= 0.0
      || rawEnergyScale > stepCorrectionLimit
    ) {
      if (
        fields.horizonDistance
          <= max(capturePadding, numericalCaptureGuard())
      ) {
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
    momentum = momentum * rawEnergyScale;
    let driftKinematics = hamiltonianKinematics(fields, momentum);
    position = position - acceptedStepSize * driftKinematics.velocity;
    lookback = lookback + acceptedStepSize;
    if (!finiteVector(position) || !finiteVector(momentum)) {
      result.terminationReason = 2.0;
      result.hamiltonianResidual = max(maximumResidual, 1.0);
      result.minimumHorizonDistance = minimumHorizonDistance;
      return result;
    }
  }

  // Exhausting either the base or critical-zone budget is unresolved, never
  // silently converted to captured black or a fabricated sky sample.
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

fn skyQualityPressure() -> f32 {
  let baseBudget = clamp(
    params.renderControls.w,
    24.0,
    f32(MAX_STRONG_STEPS)
  );
  // The step budget is fixed by the selected quality tier.  Do not derive
  // reconstruction weights from per-ray iteration counts or closest-horizon
  // state: those values move at sub-pixel boundaries and made bright stars
  // shimmer even while the camera and tier were otherwise unchanged.
  return 1.0 - smoothstep(
    64.0,
    160.0,
    baseBudget
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
  let qualityPressure = skyQualityPressure();
  let footprintPressure = smoothstep(0.62, 1.35, sourceFootprint);
  let radius = clamp(
    sourceFootprint * mix(0.72, 1.08, qualityPressure),
    0.50,
    3.0
  );
  let centre = textureSampleLevel(
    tSky,
    skySampler,
    uv,
    0.0
  ).rgb;
  let filterWeight = clamp(
    footprintPressure * mix(0.32, 0.46, qualityPressure),
    0.0,
    0.46
  );
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

fn shadeResult(result: RayResult, pixel: vec2<f32>) -> vec3<f32> {
  let mode = i32(round(params.renderControls.z));
  if (mode == 1) {
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
  if (mode == 5) {
    return viridis(result.iterations / f32(MAX_STRONG_STEPS));
  }
  if (result.outcome == RAY_CAPTURED) {
    return vec3<f32>(0.0);
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
    return vec3<f32>(0.050, 0.036, 0.024)
      * unresolvedLevel * hatch;
  }
  let shiftRadiance = pow(
    clamp(result.frequencyShift, 0.20, 2.5),
    3.0
  );
  return sampleEnvironment(result.escapeDirection) * shiftRadiance;
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

@fragment
fn fsMain(input: FragmentInput) -> @location(0) vec4<f32> {
  let resolution = max(params.resolutionTimeMass.xy, vec2<f32>(1.0));
  let aspect = resolution.x / resolution.y;
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

export const strongFieldBinaryShaderBundle = Object.freeze({
  id: "binary-strong-field-v1",
  labels: Object.freeze({
    uniforms: "Real-time strong-field binary frame uniforms",
    trace: "WebGPU 3+1 Hamiltonian strong-field binary tracer",
    webglFallback: "Legacy WebGL2 weak-field fast-light fallback",
  }),
  backendPolicy: Object.freeze({
    production: "webgpu",
    webgpuModel: "boosted-superposed-kerr-schild-fast-light",
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
    traceSpecializations: Object.freeze([
      Object.freeze({
        id: "binary",
        constants: Object.freeze({ SPACETIME_PHASE_MODE: 0 }),
      }),
      Object.freeze({
        id: "transition",
        constants: Object.freeze({ SPACETIME_PHASE_MODE: 2 }),
      }),
      Object.freeze({
        id: "remnant",
        constants: Object.freeze({ SPACETIME_PHASE_MODE: 1 }),
      }),
    ]),
    selectTraceSpecialization(frame) {
      const blend = Number(frame?.sceneStrongFieldUniforms?.[1]);
      if (blend === 0) {
        return "binary";
      }
      if (blend === 1) {
        return "remnant";
      }
      return "transition";
    },
  }),
  glsl: Object.freeze({
    // Deliberate fallback, not a port of the strong-field provider.
    trace: binaryTraceFragmentGLSL,
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
