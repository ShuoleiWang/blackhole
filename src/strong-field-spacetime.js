/*
 * CPU reference for the real-time strong-field binary tracer.
 *
 * This module is deliberately renderer-independent.  It defines the numerical
 * contract that the WebGPU implementation must reproduce, while remaining
 * slow and transparent enough to serve as an oracle in tests.
 *
 * Scientific boundary
 * -------------------
 * The binary metric is a time-frozen linear superposition of two instantaneously
 * boosted Kerr-Schild perturbations, smoothly replaced by one remnant
 * Kerr-Schild perturbation.  It is inspired by Combi & Ressler,
 * arXiv:2403.13308, and Combi et al., arXiv:2103.15707.  Unlike a numerical
 * relativity solution it does not solve the Einstein constraints.  Companion
 * attenuation and singularity regularization are numerical prescriptions, not
 * horizons.  The provider therefore identifies itself as
 * "strong-field approximate metric", never as NR ray tracing.
 *
 * Conventions: signature (-,+,+,+), geometric units G=c=M_total=1, and
 *
 *   ds^2 = -alpha^2 dt^2
 *        + gamma_ij (dx^i + beta^i dt)(dx^j + beta^j dt).
 *
 * The coordinate-time null Hamiltonian is
 *
 *   H = alpha sqrt(gamma^ij p_i p_j) - beta^i p_i = -p_t.
 */

export const SPACETIME_PROVIDER_SCHEMA = "blackhole.spacetime-provider/v1";
export const ORBIT_ADAPTER_SCHEMA = "blackhole.pn-eob-orbit-adapter/v1";

export const STRONG_FIELD_UNIFORM_ABI = Object.freeze({
  schema: "blackhole.strong-field-uniforms/v1",
  scalarType: "float32",
  floatCount: 44,
  byteLength: 176,
  vectors: Object.freeze([
    Object.freeze(["timeM", "transitionWeight", "attenuationScaleM", "regularizationRadiusFraction"]),
    Object.freeze(["bodyAPositionX", "bodyAPositionY", "bodyAPositionZ", "bodyAMassM"]),
    Object.freeze(["bodyAVelocityX", "bodyAVelocityY", "bodyAVelocityZ", "bodyAActive"]),
    Object.freeze(["bodyASpinChiX", "bodyASpinChiY", "bodyASpinChiZ", "reserved"]),
    Object.freeze(["bodyBPositionX", "bodyBPositionY", "bodyBPositionZ", "bodyBMassM"]),
    Object.freeze(["bodyBVelocityX", "bodyBVelocityY", "bodyBVelocityZ", "bodyBActive"]),
    Object.freeze(["bodyBSpinChiX", "bodyBSpinChiY", "bodyBSpinChiZ", "reserved"]),
    Object.freeze(["remnantPositionX", "remnantPositionY", "remnantPositionZ", "remnantMassM"]),
    Object.freeze(["remnantVelocityX", "remnantVelocityY", "remnantVelocityZ", "remnantActive"]),
    Object.freeze(["remnantSpinChiX", "remnantSpinChiY", "remnantSpinChiZ", "rawMergerBlend"]),
    Object.freeze(["attenuationEnabled", "attenuationPower", "absoluteRegularizationRadiusM", "maxKerrSchildH"]),
  ]),
});

const ETA = Object.freeze([
  Object.freeze([-1, 0, 0, 0]),
  Object.freeze([0, 1, 0, 0]),
  Object.freeze([0, 0, 1, 0]),
  Object.freeze([0, 0, 0, 1]),
]);
const DEFAULT_ATTENUATION = Object.freeze({
  enabled: true,
  scaleFraction: 0.35,
  minimumScaleM: 1e-3,
  power: 4,
});
const DEFAULT_REGULARIZATION = Object.freeze({
  radiusFraction: 1e-4,
  absoluteRadiusM: 1e-7,
  maxKerrSchildH: 1e6,
});
const METRIC_EPSILON = 1e-13;

function requireCondition(condition, message) {
  if (!condition) {
    throw new Error(`Strong-field spacetime contract violation: ${message}`);
  }
}
function finiteNumber(value, label) {
  requireCondition(
    typeof value === "number" && Number.isFinite(value),
    `${label} must be a finite number`,
  );
  return value;
}

function finiteVector3(value, label) {
  requireCondition(
    Array.isArray(value) && value.length === 3,
    `${label} must be a three-vector`,
  );
  return value.map((component, index) => (
    finiteNumber(component, `${label}[${index}]`)
  ));
}

function cloneMatrix(matrix) {
  return matrix.map((row) => row.slice());
}

function freezeMatrix(matrix) {
  return Object.freeze(matrix.map((row) => Object.freeze(row.slice())));
}

function freezeVector(vector) {
  return Object.freeze(vector.slice());
}

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(vector, factor) {
  return [vector[0] * factor, vector[1] * factor, vector[2] * factor];
}

function dot(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function norm(vector) {
  return Math.hypot(...vector);
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function matrixVector(matrix, vector) {
  return matrix.map((row) => dot(row, vector));
}

function smoothstep5(value) {
  const x = Math.min(1, Math.max(0, value));
  return x * x * x * (x * (x * 6 - 15) + 10);
}

function zeroMatrix(size) {
  return Array.from({ length: size }, () => Array(size).fill(0));
}

function invertSquareMatrix(input, label) {
  const size = input.length;
  requireCondition(
    size > 0 && input.every((row) => row.length === size),
    `${label} must be square`,
  );
  const augmented = input.map((row, rowIndex) => [
    ...row,
    ...Array.from({ length: size }, (_, columnIndex) => (
      rowIndex === columnIndex ? 1 : 0
    )),
  ]);

  for (let column = 0; column < size; column += 1) {
    let pivotRow = column;
    for (let row = column + 1; row < size; row += 1) {
      if (
        Math.abs(augmented[row][column])
          > Math.abs(augmented[pivotRow][column])
      ) {
        pivotRow = row;
      }
    }
    const pivot = augmented[pivotRow][column];
    requireCondition(
      Number.isFinite(pivot) && Math.abs(pivot) > METRIC_EPSILON,
      `${label} is singular or ill-conditioned`,
    );
    [augmented[column], augmented[pivotRow]] = [
      augmented[pivotRow],
      augmented[column],
    ];
    for (let index = 0; index < 2 * size; index += 1) {
      augmented[column][index] /= pivot;
    }
    for (let row = 0; row < size; row += 1) {
      if (row === column) {
        continue;
      }
      const factor = augmented[row][column];
      for (let index = 0; index < 2 * size; index += 1) {
        augmented[row][index] -= factor * augmented[column][index];
      }
    }
  }
  const inverse = augmented.map((row) => row.slice(size));
  requireCondition(
    inverse.flat().every(Number.isFinite),
    `${label} inverse is not finite`,
  );
  return inverse;
}

function isPositiveDefinite3(matrix) {
  // Sylvester's criterion for a real symmetric 3x3 matrix.
  const first = matrix[0][0];
  const second = (
    matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
  );
  const third = (
    matrix[0][0] * (
      matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]
    )
    - matrix[0][1] * (
      matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]
    )
    + matrix[0][2] * (
      matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]
    )
  );
  return first > METRIC_EPSILON
    && second > METRIC_EPSILON
    && third > METRIC_EPSILON;
}

export function decomposeMetric3p1(covariantMetric) {
  requireCondition(
    Array.isArray(covariantMetric)
      && covariantMetric.length === 4
      && covariantMetric.every((row) => (
        Array.isArray(row)
        && row.length === 4
        && row.every(Number.isFinite)
      )),
    "covariant metric must be a finite 4x4 matrix",
  );
  for (let row = 0; row < 4; row += 1) {
    for (let column = row + 1; column < 4; column += 1) {
      requireCondition(
        Math.abs(
          covariantMetric[row][column] - covariantMetric[column][row],
        ) <= 2e-11 * Math.max(
          1,
          Math.abs(covariantMetric[row][column]),
          Math.abs(covariantMetric[column][row]),
        ),
        "covariant metric must be symmetric",
      );
    }
  }

  const spatialMetric = covariantMetric
    .slice(1)
    .map((row) => row.slice(1));
  requireCondition(
    isPositiveDefinite3(spatialMetric),
    "spatial metric is not positive definite",
  );
  const inverseSpatialMetric = invertSquareMatrix(
    spatialMetric,
    "spatial metric",
  );
  const shiftCovariant = covariantMetric[0].slice(1);
  const shift = matrixVector(inverseSpatialMetric, shiftCovariant);
  const lapseSquared = -(
    covariantMetric[0][0] - dot(shiftCovariant, shift)
  );
  requireCondition(
    Number.isFinite(lapseSquared) && lapseSquared > METRIC_EPSILON,
    "metric has no real positive lapse",
  );
  const lapse = Math.sqrt(lapseSquared);
  const inverseMetric = zeroMatrix(4);
  inverseMetric[0][0] = -1 / lapseSquared;
  for (let index = 0; index < 3; index += 1) {
    inverseMetric[0][index + 1] = shift[index] / lapseSquared;
    inverseMetric[index + 1][0] = inverseMetric[0][index + 1];
  }
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      inverseMetric[row + 1][column + 1] = (
        inverseSpatialMetric[row][column]
        - shift[row] * shift[column] / lapseSquared
      );
    }
  }

  return Object.freeze({
    lapse,
    shift: freezeVector(shift),
    shiftCovariant: freezeVector(shiftCovariant),
    spatialMetric: freezeMatrix(spatialMetric),
    inverseSpatialMetric: freezeMatrix(inverseSpatialMetric),
    covariantMetric: freezeMatrix(covariantMetric),
    inverseMetric: freezeMatrix(inverseMetric),
  });
}

function normalizeBlackHole(value, label) {
  requireCondition(
    value && typeof value === "object" && !Array.isArray(value),
    `${label} must be an object`,
  );
  const massM = finiteNumber(value.massM, `${label}.massM`);
  requireCondition(massM > 0, `${label}.massM must be positive`);
  const positionM = finiteVector3(value.positionM, `${label}.positionM`);
  const velocityC = finiteVector3(
    value.velocityC ?? [0, 0, 0],
    `${label}.velocityC`,
  );
  requireCondition(
    dot(velocityC, velocityC) < 1 - 1e-10,
    `${label}.velocityC must be subluminal`,
  );
  const dimensionlessSpin = finiteVector3(
    value.dimensionlessSpin ?? [0, 0, 0],
    `${label}.dimensionlessSpin`,
  );
  requireCondition(
    norm(dimensionlessSpin) < 1,
    `${label}.dimensionlessSpin violates the Kerr bound`,
  );
  return Object.freeze({
    id: String(value.id ?? label),
    massM,
    positionM: freezeVector(positionM),
    velocityC: freezeVector(velocityC),
    dimensionlessSpin: freezeVector(dimensionlessSpin),
  });
}

function normalizeOrbitState(value, timeM, adapter) {
  requireCondition(
    value && typeof value === "object" && !Array.isArray(value),
    "orbit sample must be an object",
  );
  requireCondition(
    Array.isArray(value.bodies) && value.bodies.length === 2,
    "orbit sample must contain exactly two inspiral bodies",
  );
  const bodies = value.bodies.map((body, index) => (
    normalizeBlackHole(body, `bodies[${index}]`)
  ));
  const remnant = normalizeBlackHole(value.remnant, "remnant");
  const mergerBlend = finiteNumber(
    value.mergerBlend,
    "orbit sample mergerBlend",
  );
  requireCondition(
    mergerBlend >= 0 && mergerBlend <= 1,
    "orbit sample mergerBlend must lie in [0,1]",
  );
  return Object.freeze({
    timeM,
    bodies: Object.freeze(bodies),
    remnant,
    mergerBlend,
    provenance: Object.freeze({
      dynamicsModel: adapter.dynamicsModel,
      coordinateFrame: adapter.coordinateFrame,
      source: adapter.source,
      usesSxsGaugeCentroids: false,
    }),
  });
}

export function createPnEobOrbitAdapter({
  dynamicsModel,
  coordinateFrame = "asymptotically-inertial-kerr-schild-com",
  source,
  usesSxsGaugeCentroids,
  sample,
}) {
  requireCondition(
    typeof dynamicsModel === "string" && /(PN|EOB)/i.test(dynamicsModel),
    "dynamicsModel must explicitly identify a PN or EOB model",
  );
  requireCondition(
    typeof coordinateFrame === "string"
      && coordinateFrame.trim().length > 0,
    "coordinateFrame must be explicit",
  );
  requireCondition(
    typeof source === "string" && source.trim().length > 0,
    "orbit adapter source must be explicit",
  );
  requireCondition(
    usesSxsGaugeCentroids === false,
    "SXS horizon-centroid coordinates cannot be used as Kerr-Schild positions",
  );
  requireCondition(typeof sample === "function", "sample must be a function");

  const adapter = {
    schema: ORBIT_ADAPTER_SCHEMA,
    dynamicsModel,
    coordinateFrame,
    source,
    usesSxsGaugeCentroids: false,
    sample(timeM) {
      const validatedTime = finiteNumber(timeM, "orbit timeM");
      return normalizeOrbitState(
        sample(validatedTime),
        validatedTime,
        adapter,
      );
    },
  };
  return Object.freeze(adapter);
}

function normalizeAttenuation(value = {}) {
  const result = {
    ...DEFAULT_ATTENUATION,
    ...value,
  };
  requireCondition(
    typeof result.enabled === "boolean",
    "attenuation.enabled must be boolean",
  );
  for (const key of ["scaleFraction", "minimumScaleM", "power"]) {
    finiteNumber(result[key], `attenuation.${key}`);
  }
  requireCondition(
    result.scaleFraction > 0
      && result.minimumScaleM > 0
      && result.power >= 2,
    "attenuation scales must be positive and power must be >= 2",
  );
  return Object.freeze(result);
}

function normalizeRegularization(value = {}) {
  const result = {
    ...DEFAULT_REGULARIZATION,
    ...value,
  };
  for (const key of [
    "radiusFraction",
    "absoluteRadiusM",
    "maxKerrSchildH",
  ]) {
    finiteNumber(result[key], `regularization.${key}`);
  }
  requireCondition(
    result.radiusFraction > 0
      && result.absoluteRadiusM > 0
      && result.maxKerrSchildH > 1,
    "regularization parameters must be positive",
  );
  return Object.freeze(result);
}

function restKerrSchildTerm(
  restPosition,
  massM,
  dimensionlessSpin,
  regularization,
) {
  const spinParameter = scale(dimensionlessSpin, massM);
  const rhoSquared = dot(restPosition, restPosition);
  const spinSquared = dot(spinParameter, spinParameter);
  const spinPosition = dot(spinParameter, restPosition);
  const discriminant = (
    (rhoSquared - spinSquared) ** 2 + 4 * spinPosition ** 2
  );
  const rawRadiusSquared = Math.max(
    0,
    0.5 * (
      rhoSquared - spinSquared + Math.sqrt(Math.max(0, discriminant))
    ),
  );
  const rawKerrRadius = Math.sqrt(rawRadiusSquared);
  const radiusFloor = Math.max(
    regularization.absoluteRadiusM,
    regularization.radiusFraction * massM,
  );
  const kerrRadius = Math.max(rawKerrRadius, radiusFloor);
  const denominator = kerrRadius * kerrRadius + spinSquared;
  const numerator = add(
    add(
      scale(restPosition, kerrRadius),
      cross(restPosition, spinParameter),
    ),
    scale(spinParameter, spinPosition / kerrRadius),
  );
  let spatialNull = scale(numerator, 1 / denominator);
  const spatialNullNorm = norm(spatialNull);
  if (!(spatialNullNorm > METRIC_EPSILON)) {
    spatialNull = [0, 0, 1];
  } else {
    spatialNull = scale(spatialNull, 1 / spatialNullNorm);
  }
  const rawH = (
    massM * kerrRadius ** 3
    / (kerrRadius ** 4 + spinPosition ** 2)
  );
  const H = Math.min(rawH, regularization.maxKerrSchildH);
  const horizonRadius = massM * (
    1 + Math.sqrt(Math.max(0, 1 - norm(dimensionlessSpin) ** 2))
  );
  return {
    H,
    nullCovector: [1, ...spatialNull],
    cartesianRadius: Math.sqrt(rhoSquared),
    kerrRadius: rawKerrRadius,
    horizonRadius,
    regularized: rawKerrRadius < radiusFloor
      || rawH > regularization.maxKerrSchildH,
  };
}

function boostedKerrSchildTerm(body, positionM, regularization) {
  const displacement = subtract(positionM, body.positionM);
  const velocity = body.velocityC;
  const velocitySquared = dot(velocity, velocity);
  const gamma = 1 / Math.sqrt(1 - velocitySquared);
  const velocityProjection = dot(velocity, displacement);
  const boostFactor = velocitySquared > 1e-24
    ? (gamma - 1) * velocityProjection / velocitySquared
    : 0;
  const restPosition = add(displacement, scale(velocity, boostFactor));
  const rest = restKerrSchildTerm(
    restPosition,
    body.massM,
    body.dimensionlessSpin,
    regularization,
  );
  const restSpatialNull = rest.nullCovector.slice(1);
  const velocityNullProjection = dot(velocity, restSpatialNull);
  const transformedTime = gamma * (1 - velocityNullProjection);
  const transformedFactor = velocitySquared > 1e-24
    ? (
      (gamma - 1) * velocityNullProjection / velocitySquared - gamma
    )
    : 0;
  const transformedSpatial = add(
    restSpatialNull,
    scale(velocity, transformedFactor),
  );
  const nullCovector = [transformedTime, ...transformedSpatial];
  const perturbation = zeroMatrix(4);
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 4; column += 1) {
      perturbation[row][column] = (
        2 * rest.H * nullCovector[row] * nullCovector[column]
      );
    }
  }
  return {
    ...rest,
    nullCovector,
    perturbation,
  };
}

function attenuationWeight(companionRadius, scaleM, attenuation) {
  if (!attenuation.enabled) {
    return 1;
  }
  const ratio = Math.max(0, companionRadius / scaleM);
  return -Math.expm1(-(ratio ** attenuation.power));
}

function evaluateStateAtPosition(
  orbitState,
  positionM,
  attenuation,
  regularization,
) {
  const position = finiteVector3(positionM, "metric positionM");
  const [bodyA, bodyB] = orbitState.bodies;
  const separationM = norm(subtract(bodyA.positionM, bodyB.positionM));
  const attenuationScaleM = Math.max(
    attenuation.minimumScaleM,
    attenuation.scaleFraction * separationM,
  );
  const termA = boostedKerrSchildTerm(bodyA, position, regularization);
  const termB = boostedKerrSchildTerm(bodyB, position, regularization);
  const remnantTerm = boostedKerrSchildTerm(
    orbitState.remnant,
    position,
    regularization,
  );
  const weightA = attenuationWeight(
    termB.cartesianRadius,
    attenuationScaleM,
    attenuation,
  );
  const weightB = attenuationWeight(
    termA.cartesianRadius,
    attenuationScaleM,
    attenuation,
  );
  const transitionWeight = smoothstep5(orbitState.mergerBlend);
  const binaryWeight = 1 - transitionWeight;
  const covariantMetric = cloneMatrix(ETA);
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 4; column += 1) {
      covariantMetric[row][column] += (
        binaryWeight * (
          weightA * termA.perturbation[row][column]
          + weightB * termB.perturbation[row][column]
        )
        + transitionWeight * remnantTerm.perturbation[row][column]
      );
    }
  }
  const fields = decomposeMetric3p1(covariantMetric);
  return Object.freeze({
    ...fields,
    schema: "blackhole.spacetime-3p1-sample/v1",
    model: "boosted-superposed-kerr-schild-to-remnant-kerr",
    scientificStatus: "real-time strong-field approximate metric; not NR",
    timeM: orbitState.timeM,
    positionM: freezeVector(position),
    transitionWeight,
    attenuationScaleM,
    regularized: (
      termA.regularized
      || termB.regularized
      || remnantTerm.regularized
    ),
    constraintSolved: false,
    diagnostics: Object.freeze({
      companionAttenuation: Object.freeze([weightA, weightB]),
      kerrRadiiM: Object.freeze([
        termA.kerrRadius,
        termB.kerrRadius,
        remnantTerm.kerrRadius,
      ]),
      horizonRadiusProxiesM: Object.freeze([
        termA.horizonRadius,
        termB.horizonRadius,
        remnantTerm.horizonRadius,
      ]),
      // These are excision/capture proxies only; no apparent horizon is solved.
      captureDistanceProxiesM: Object.freeze([
        termA.kerrRadius - termA.horizonRadius,
        termB.kerrRadius - termB.horizonRadius,
        remnantTerm.kerrRadius - remnantTerm.horizonRadius,
      ]),
    }),
  });
}

function packUniforms(orbitState, attenuation, regularization) {
  const [bodyA, bodyB] = orbitState.bodies;
  const separationM = norm(subtract(bodyA.positionM, bodyB.positionM));
  const attenuationScaleM = Math.max(
    attenuation.minimumScaleM,
    attenuation.scaleFraction * separationM,
  );
  const output = new Float32Array(STRONG_FIELD_UNIFORM_ABI.floatCount);
  const write = (offset, values) => output.set(values, offset);
  write(0, [
    orbitState.timeM,
    smoothstep5(orbitState.mergerBlend),
    attenuationScaleM,
    regularization.radiusFraction,
  ]);
  write(4, [...bodyA.positionM, bodyA.massM]);
  write(8, [...bodyA.velocityC, 1]);
  write(12, [...bodyA.dimensionlessSpin, 0]);
  write(16, [...bodyB.positionM, bodyB.massM]);
  write(20, [...bodyB.velocityC, 1]);
  write(24, [...bodyB.dimensionlessSpin, 0]);
  write(28, [...orbitState.remnant.positionM, orbitState.remnant.massM]);
  write(32, [...orbitState.remnant.velocityC, 1]);
  write(36, [
    ...orbitState.remnant.dimensionlessSpin,
    orbitState.mergerBlend,
  ]);
  write(40, [
    attenuation.enabled ? 1 : 0,
    attenuation.power,
    regularization.absoluteRadiusM,
    regularization.maxKerrSchildH,
  ]);
  requireCondition(
    [...output].every(Number.isFinite),
    "packed strong-field uniforms are not finite",
  );
  return output;
}

export function createStrongFieldSpacetimeProvider({
  orbitAdapter,
  attenuation: attenuationOptions,
  regularization: regularizationOptions,
}) {
  requireCondition(
    orbitAdapter?.schema === ORBIT_ADAPTER_SCHEMA
      && orbitAdapter.usesSxsGaugeCentroids === false
      && typeof orbitAdapter.sample === "function",
    "orbitAdapter must implement the PN/EOB adapter contract",
  );
  const attenuation = normalizeAttenuation(attenuationOptions);
  const regularization = normalizeRegularization(regularizationOptions);

  function frameAt(timeM) {
    const orbitState = orbitAdapter.sample(
      finiteNumber(timeM, "provider timeM"),
    );
    const uniforms = packUniforms(
      orbitState,
      attenuation,
      regularization,
    );
    return Object.freeze({
      schema: "blackhole.spacetime-frame/v1",
      orbitState,
      uniforms,
      evaluate(positionM) {
        return evaluateStateAtPosition(
          orbitState,
          positionM,
          attenuation,
          regularization,
        );
      },
      evaluateOrUnresolved(positionM) {
        try {
          return Object.freeze({
            outcome: "valid",
            fields: evaluateStateAtPosition(
              orbitState,
              positionM,
              attenuation,
              regularization,
            ),
          });
        } catch (error) {
          return Object.freeze({
            outcome: "unresolved",
            fields: null,
            reason: error instanceof Error ? error.message : String(error),
          });
        }
      },
    });
  }

  return Object.freeze({
    schema: SPACETIME_PROVIDER_SCHEMA,
    model: "boosted-superposed-kerr-schild-fast-light",
    scientificStatus: "strong-field approximate metric; not constraint-solved NR",
    positionsFrozenPerRay: true,
    orbitAdapter,
    attenuation,
    regularization,
    frameAt,
    evaluate(timeM, positionM) {
      return frameAt(timeM).evaluate(positionM);
    },
  });
}

export function evaluateKerrSchild3p1({
  massM,
  positionM,
  centreM = [0, 0, 0],
  velocityC = [0, 0, 0],
  dimensionlessSpin = [0, 0, 0],
  regularization: regularizationOptions,
}) {
  const validatedMass = finiteNumber(massM, "massM");
  requireCondition(validatedMass >= 0, "massM must be non-negative");
  if (validatedMass === 0) {
    return Object.freeze({
      ...decomposeMetric3p1(ETA),
      schema: "blackhole.spacetime-3p1-sample/v1",
      model: "minkowski",
      scientificStatus: "exact analytic vacuum",
      regularized: false,
    });
  }
  const body = normalizeBlackHole({
    id: "single",
    massM: validatedMass,
    positionM: finiteVector3(centreM, "centreM"),
    velocityC: finiteVector3(velocityC, "velocityC"),
    dimensionlessSpin: finiteVector3(
      dimensionlessSpin,
      "dimensionlessSpin",
    ),
  }, "single");
  const regularization = normalizeRegularization(regularizationOptions);
  const term = boostedKerrSchildTerm(
    body,
    finiteVector3(positionM, "positionM"),
    regularization,
  );
  const metric = cloneMatrix(ETA);
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 4; column += 1) {
      metric[row][column] += term.perturbation[row][column];
    }
  }
  return Object.freeze({
    ...decomposeMetric3p1(metric),
    schema: "blackhole.spacetime-3p1-sample/v1",
    model: norm(body.dimensionlessSpin) > 0
      ? "exact-single-kerr-schild"
      : "exact-single-schwarzschild-kerr-schild",
    scientificStatus: "exact analytic vacuum outside regularization",
    regularized: term.regularized,
    kerrRadiusM: term.kerrRadius,
    horizonRadiusM: term.horizonRadius,
  });
}

function validateMomentum(momentumCovariant) {
  const momentum = finiteVector3(momentumCovariant, "momentumCovariant");
  requireCondition(norm(momentum) > 0, "photon momentum must be non-zero");
  return momentum;
}

export function nullHamiltonian(fields, momentumCovariant) {
  const momentum = validateMomentum(momentumCovariant);
  const raised = matrixVector(fields.inverseSpatialMetric, momentum);
  const spatialNormSquared = dot(momentum, raised);
  requireCondition(
    Number.isFinite(spatialNormSquared)
      && spatialNormSquared > METRIC_EPSILON,
    "spatial photon momentum norm is invalid",
  );
  const spatialNorm = Math.sqrt(spatialNormSquared);
  return (
    fields.lapse * spatialNorm - dot(fields.shift, momentum)
  );
}

export function nullHamiltonianResidual(fields, momentumCovariant) {
  const momentum = validateMomentum(momentumCovariant);
  const hamiltonian = nullHamiltonian(fields, momentum);
  const fourMomentumCovariant = [-hamiltonian, ...momentum];
  let residual = 0;
  let scaleSum = 0;
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 4; column += 1) {
      const term = (
        fields.inverseMetric[row][column]
        * fourMomentumCovariant[row]
        * fourMomentumCovariant[column]
      );
      residual += term;
      scaleSum += Math.abs(term);
    }
  }
  const fourMomentumContravariant = matrixVector(
    fields.inverseMetric,
    fourMomentumCovariant,
  );
  return Object.freeze({
    hamiltonian,
    raw: residual,
    normalized: Math.abs(residual) / Math.max(scaleSum, 1e-30),
    fourMomentumCovariant: freezeVector(fourMomentumCovariant),
    fourMomentumContravariant: freezeVector(fourMomentumContravariant),
    futureDirected: fourMomentumContravariant[0] > 0,
  });
}

export function hamiltonianDerivatives(
  provider,
  timeM,
  positionM,
  momentumCovariant,
  {
    absoluteSpaceStepM = 2e-5,
    relativeSpaceStep = 2e-5,
    timeStepM = 2e-5,
  } = {},
) {
  requireCondition(
    provider?.schema === SPACETIME_PROVIDER_SCHEMA,
    "hamiltonian derivatives require a spacetime provider",
  );
  const time = finiteNumber(timeM, "timeM");
  const position = finiteVector3(positionM, "positionM");
  const momentum = validateMomentum(momentumCovariant);
  for (const [value, label] of [
    [absoluteSpaceStepM, "absoluteSpaceStepM"],
    [relativeSpaceStep, "relativeSpaceStep"],
    [timeStepM, "timeStepM"],
  ]) {
    finiteNumber(value, label);
    requireCondition(value > 0, `${label} must be positive`);
  }
  const fields = provider.evaluate(time, position);
  const raised = matrixVector(fields.inverseSpatialMetric, momentum);
  const spatialNorm = Math.sqrt(dot(momentum, raised));
  const dxdt = subtract(
    scale(raised, fields.lapse / spatialNorm),
    fields.shift,
  );
  const gradient = [0, 0, 0];
  for (let axis = 0; axis < 3; axis += 1) {
    const step = Math.max(
      absoluteSpaceStepM,
      relativeSpaceStep * Math.max(1, Math.abs(position[axis])),
    );
    const lower = position.slice();
    const upper = position.slice();
    lower[axis] -= step;
    upper[axis] += step;
    gradient[axis] = (
      nullHamiltonian(provider.evaluate(time, upper), momentum)
      - nullHamiltonian(provider.evaluate(time, lower), momentum)
    ) / (2 * step);
  }
  const dHdt = (
    nullHamiltonian(
      provider.evaluate(time + timeStepM, position),
      momentum,
    )
    - nullHamiltonian(
      provider.evaluate(time - timeStepM, position),
      momentum,
    )
  ) / (2 * timeStepM);
  return Object.freeze({
    hamiltonian: nullHamiltonian(fields, momentum),
    dxdt: freezeVector(dxdt),
    dpdt: freezeVector(scale(gradient, -1)),
    dHdt,
    residual: nullHamiltonianResidual(fields, momentum),
  });
}
