import assert from "node:assert/strict";
import test from "node:test";

import {
  SPACETIME_PROVIDER_SCHEMA,
  createPnEobOrbitAdapter,
  createStrongFieldSpacetimeProvider,
  evaluateKerrSchild3p1,
  nullHamiltonian,
} from "../src/strong-field-spacetime.js";

const SQRT_THREE = Math.sqrt(3);
const SCHWARZSCHILD_CRITICAL_IMPACT = 3 * SQRT_THREE;

function add(a, b) {
  return a.map((value, index) => value + b[index]);
}

function subtract(a, b) {
  return a.map((value, index) => value - b[index]);
}

function scale(vector, factor) {
  return vector.map((value) => value * factor);
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

function normalize(vector) {
  return scale(vector, 1 / norm(vector));
}

function matrixVector(matrix, vector) {
  return matrix.map((row) => dot(row, vector));
}

function metricDot(fields, a, b) {
  return dot(a, matrixVector(fields.spatialMetric, b));
}

function metricNormalize(fields, vector) {
  return scale(vector, 1 / Math.sqrt(metricDot(fields, vector, vector)));
}

// Matches the shader's ADM-metric Gram-Schmidt camera tetrad.
function cameraTetrad(fields, {
  forward = [-1, 0, 0],
  right = [0, 1, 0],
  up = [0, 0, 1],
} = {}) {
  const eForward = metricNormalize(fields, forward);
  const eRight = metricNormalize(
    fields,
    subtract(right, scale(eForward, metricDot(fields, eForward, right))),
  );
  const eUp = metricNormalize(
    fields,
    subtract(
      subtract(up, scale(eForward, metricDot(fields, eForward, up))),
      scale(eRight, metricDot(fields, eRight, up)),
    ),
  );
  return { forward: eForward, right: eRight, up: eUp };
}

function cameraMomentum(fields, slopeX = 0, slopeY = 0, basisOptions) {
  const basis = cameraTetrad(fields, basisOptions);
  const direction = metricNormalize(
    fields,
    add(
      add(basis.forward, scale(basis.right, slopeX)),
      scale(basis.up, slopeY),
    ),
  );
  // The screen direction points from the camera into the scene.  The photon
  // arriving at the camera is future-directed in the opposite local
  // direction; trace its Hamiltonian flow with negative time steps below.
  return scale(matrixVector(fields.spatialMetric, direction), -1);
}

function blackHole(
  id,
  massM,
  positionM,
  {
    velocityC = [0, 0, 0],
    dimensionlessSpin = [0, 0, 0],
  } = {},
) {
  return {
    id,
    massM,
    positionM,
    velocityC,
    dimensionlessSpin,
  };
}

function frozenProviderFor({
  bodies,
  remnant,
  mergerBlend,
  attenuation = { enabled: false },
}) {
  const adapter = createPnEobOrbitAdapter({
    dynamicsModel: "4PN/EOB-compatible deterministic ray oracle",
    coordinateFrame: "asymptotically-inertial-kerr-schild-com",
    source: "independent CPU ray acceptance test",
    usesSxsGaugeCentroids: false,
    sample() {
      return { bodies, remnant, mergerBlend };
    },
  });
  const source = createStrongFieldSpacetimeProvider({
    orbitAdapter: adapter,
    attenuation,
  });
  const frame = source.frameAt(0);
  return Object.freeze({
    schema: SPACETIME_PROVIDER_SCHEMA,
    evaluate(_timeM, positionM) {
      return frame.evaluate(positionM);
    },
  });
}

function singleHoleProvider({
  massM = 1,
  dimensionlessSpin = [0, 0, 0],
}) {
  return frozenProviderFor({
    bodies: [
      blackHole("inactive-A", 0.5, [-20, 0, 0]),
      blackHole("inactive-B", 0.5, [20, 0, 0]),
    ],
    remnant: blackHole("remnant", massM, [0, 0, 0], {
      dimensionlessSpin,
    }),
    mergerBlend: 1,
  });
}

function minkowskiProvider() {
  const fields = evaluateKerrSchild3p1({
    massM: 0,
    positionM: [0, 0, 0],
  });
  return Object.freeze({
    schema: SPACETIME_PROVIDER_SCHEMA,
    evaluate() {
      return fields;
    },
  });
}

function finiteVector(value) {
  return (
    Array.isArray(value)
    && value.length === 3
    && value.every(Number.isFinite)
  );
}

// Independent central-difference implementation of the frozen 3+1
// Hamiltonian equations.  It intentionally does not share the shader's dual
// derivatives or integration update.
function hamiltonianFlow(
  provider,
  position,
  momentum,
  {
    absoluteDifferenceM = 2e-5,
    relativeDifference = 2e-5,
  } = {},
) {
  const fields = provider.evaluate(0, position);
  const raised = matrixVector(fields.inverseSpatialMetric, momentum);
  const q = Math.sqrt(dot(momentum, raised));
  const velocity = subtract(scale(raised, fields.lapse / q), fields.shift);
  const gradient = [0, 0, 0];
  for (let axis = 0; axis < 3; axis += 1) {
    const step = Math.max(
      absoluteDifferenceM,
      relativeDifference * Math.max(1, Math.abs(position[axis])),
    );
    const lower = position.slice();
    const upper = position.slice();
    lower[axis] -= step;
    upper[axis] += step;
    gradient[axis] = (
      nullHamiltonian(provider.evaluate(0, upper), momentum)
      - nullHamiltonian(provider.evaluate(0, lower), momentum)
    ) / (2 * step);
  }
  return {
    fields,
    dxdt: velocity,
    dpdt: scale(gradient, -1),
  };
}

function activeCaptureDistance(fields) {
  const distances = fields.diagnostics?.captureDistanceProxiesM;
  if (!distances) {
    return Infinity;
  }
  return Math.min(...distances.filter(Number.isFinite));
}

function remnantCaptureDistance(fields) {
  return fields.diagnostics?.captureDistanceProxiesM?.[2] ?? Infinity;
}

function binaryCaptureDistance(fields) {
  const distances = fields.diagnostics?.captureDistanceProxiesM;
  return distances ? Math.min(distances[0], distances[1]) : Infinity;
}

function traceFrozenRay({
  provider,
  position: initialPosition,
  momentum: initialMomentum,
  captureDistance = activeCaptureDistance,
  escapeRadiusM = 50,
  maxSteps = 1600,
  minStepM = 0.018,
  maxStepM = 0.45,
  maximumLookbackM = 500,
  capturePaddingM = 0.02,
}) {
  let position = initialPosition.slice();
  let momentum = initialMomentum.slice();
  let lookbackM = 0;
  let enteredDomain = norm(position) < escapeRadiusM * 0.9;
  let maximumRelativeEnergyDrift = 0;
  let initialEnergy;

  try {
    const initialFields = provider.evaluate(0, position);
    initialEnergy = nullHamiltonian(initialFields, momentum);
  } catch {
    return {
      outcome: "unresolved",
      reason: "invalid-initial-domain",
      position,
      momentum,
      lookbackM,
      maximumRelativeEnergyDrift: Infinity,
    };
  }

  for (let stepIndex = 0; stepIndex < maxSteps; stepIndex += 1) {
    try {
      const first = hamiltonianFlow(provider, position, momentum);
      const distance = captureDistance(first.fields);
      // In ingoing Kerr-Schild coordinate time, a past-directed shadow ray
      // approaches the past horizon asymptotically.  Use the same declared
      // just-outside-horizon excision convention as the stationary oracle
      // instead of requiring an impossible finite-time sign crossing.
      if (distance <= capturePaddingM) {
        return {
          outcome: "captured",
          position,
          momentum,
          lookbackM,
          steps: stepIndex,
          maximumRelativeEnergyDrift,
        };
      }
      if (
        !finiteVector(first.dxdt)
        || !finiteVector(first.dpdt)
        || !Number.isFinite(distance)
      ) {
        throw new Error("invalid Hamiltonian flow");
      }

      const radius = norm(position);
      enteredDomain ||= radius < escapeRadiusM * 0.9;
      if (
        enteredDomain
        && radius >= escapeRadiusM
        && dot(position, scale(first.dxdt, -1)) > 0
      ) {
        return {
          outcome: "escaped",
          position,
          momentum,
          direction: normalize(scale(first.dxdt, -1)),
          lookbackM,
          steps: stepIndex,
          maximumRelativeEnergyDrift,
        };
      }
      if (lookbackM >= maximumLookbackM) {
        break;
      }

      const stepM = Math.min(
        maxStepM,
        Math.max(minStepM, 0.12 * Math.max(distance, 0.15)),
        maximumLookbackM - lookbackM,
      );
      const midpointPosition = add(
        position,
        scale(first.dxdt, -0.5 * stepM),
      );
      const midpointMomentum = add(
        momentum,
        scale(first.dpdt, -0.5 * stepM),
      );
      const midpoint = hamiltonianFlow(
        provider,
        midpointPosition,
        midpointMomentum,
      );
      const candidatePosition = add(position, scale(midpoint.dxdt, -stepM));
      let candidateMomentum = add(momentum, scale(midpoint.dpdt, -stepM));
      const candidateFields = provider.evaluate(0, candidatePosition);
      const candidateEnergy = nullHamiltonian(
        candidateFields,
        candidateMomentum,
      );
      const relativeDrift = Math.abs(candidateEnergy - initialEnergy)
        / Math.max(Math.abs(initialEnergy), 1e-12);
      maximumRelativeEnergyDrift = Math.max(
        maximumRelativeEnergyDrift,
        relativeDrift,
      );

      // H is homogeneous in p.  Project only the accumulated integration
      // drift; the pre-projection value above remains the acceptance metric.
      candidateMomentum = scale(
        candidateMomentum,
        initialEnergy / candidateEnergy,
      );
      if (
        !finiteVector(candidatePosition)
        || !finiteVector(candidateMomentum)
      ) {
        throw new Error("non-finite ray state");
      }
      position = candidatePosition;
      momentum = candidateMomentum;
      lookbackM += stepM;
    } catch {
      return {
        outcome: "unresolved",
        reason: "invalid-domain",
        position,
        momentum,
        lookbackM,
        maximumRelativeEnergyDrift,
      };
    }
  }

  return {
    outcome: "unresolved",
    reason: "budget-exhausted",
    position,
    momentum,
    lookbackM,
    steps: maxSteps,
    maximumRelativeEnergyDrift,
  };
}

function signedImpactParameter(provider, cameraPosition, screenSlope) {
  const fields = provider.evaluate(0, cameraPosition);
  const momentum = cameraMomentum(fields, screenSlope);
  const energy = nullHamiltonian(fields, momentum);
  const angularMomentum = cross(cameraPosition, momentum)[2];
  return angularMomentum / energy;
}

function slopeForSignedImpact(
  provider,
  cameraPosition,
  targetImpact,
) {
  let lower = -0.8;
  let upper = 0.8;
  const lowerValue = signedImpactParameter(
    provider,
    cameraPosition,
    lower,
  );
  const upperValue = signedImpactParameter(
    provider,
    cameraPosition,
    upper,
  );
  const ascending = upperValue > lowerValue;
  assert.ok(
    targetImpact >= Math.min(lowerValue, upperValue)
      && targetImpact <= Math.max(lowerValue, upperValue),
    `target impact ${targetImpact} is outside the camera bracket`,
  );
  for (let iteration = 0; iteration < 80; iteration += 1) {
    const midpoint = 0.5 * (lower + upper);
    const value = signedImpactParameter(
      provider,
      cameraPosition,
      midpoint,
    );
    if ((ascending && value < targetImpact) || (!ascending && value > targetImpact)) {
      lower = midpoint;
    } else {
      upper = midpoint;
    }
  }
  return 0.5 * (lower + upper);
}

function equatorialKerrCriticalImpacts(spinA, massM = 1) {
  const dimensionless = spinA / massM;
  const progradeRadius = 2 * massM * (
    1 + Math.cos((2 / 3) * Math.acos(-dimensionless))
  );
  const retrogradeRadius = 2 * massM * (
    1 + Math.cos((2 / 3) * Math.acos(dimensionless))
  );
  const impactAt = (radius) => (
    (
      radius * radius * (radius - 3 * massM)
      + spinA * spinA * (radius + massM)
    )
    / (spinA * (massM - radius))
  );
  return {
    prograde: impactAt(progradeRadius),
    retrograde: impactAt(retrogradeRadius),
  };
}

test("Minkowski Hamiltonian ray is an exact straight line", () => {
  const provider = minkowskiProvider();
  const initialPosition = [-8, 2, 1];
  const direction = normalize([1, 0.15, -0.04]);
  let position = initialPosition.slice();
  let momentum = direction.slice();
  const stepM = 0.75;
  const steps = 40;
  for (let index = 0; index < steps; index += 1) {
    const flow = hamiltonianFlow(provider, position, momentum);
    position = add(position, scale(flow.dxdt, stepM));
    momentum = add(momentum, scale(flow.dpdt, stepM));
  }
  const expected = add(initialPosition, scale(direction, stepM * steps));
  assert.ok(norm(subtract(position, expected)) < 2e-12);
  assert.ok(norm(subtract(momentum, direction)) < 1e-14);
});

test("finite-camera Schwarzschild shadow straddles bcrit=3sqrt(3)M", () => {
  const provider = singleHoleProvider({});
  const cameraPosition = [30, 0, 0];
  const criticalSlope = slopeForSignedImpact(
    provider,
    cameraPosition,
    SCHWARZSCHILD_CRITICAL_IMPACT,
  );
  const measuredImpact = signedImpactParameter(
    provider,
    cameraPosition,
    criticalSlope,
  );
  assert.ok(
    Math.abs(measuredImpact - SCHWARZSCHILD_CRITICAL_IMPACT) < 2e-12,
  );

  const fields = provider.evaluate(0, cameraPosition);
  const inside = traceFrozenRay({
    provider,
    position: cameraPosition,
    momentum: cameraMomentum(fields, criticalSlope * 0.96),
    captureDistance: remnantCaptureDistance,
    escapeRadiusM: 34,
    maxSteps: 1800,
    maxStepM: 0.35,
  });
  const outside = traceFrozenRay({
    provider,
    position: cameraPosition,
    momentum: cameraMomentum(fields, criticalSlope * 1.04),
    captureDistance: remnantCaptureDistance,
    escapeRadiusM: 34,
    maxSteps: 1800,
    maxStepM: 0.35,
  });
  assert.equal(inside.outcome, "captured", JSON.stringify(inside));
  assert.equal(outside.outcome, "escaped", JSON.stringify(outside));
  assert.ok(inside.maximumRelativeEnergyDrift < 0.015);
  assert.ok(outside.maximumRelativeEnergyDrift < 0.015);
});

test("wide binary returns to the 4M/b weak-field deflection scale", () => {
  const provider = frozenProviderFor({
    bodies: [
      blackHole("A", 0.5, [0, 0, -5]),
      blackHole("B", 0.5, [0, 0, 5]),
    ],
    remnant: blackHole("unused-remnant", 0.9516, [0, 0, 0]),
    mergerBlend: 0,
  });
  const impactM = 30;
  const start = [-180, impactM, 0];
  const fields = provider.evaluate(0, start);
  const momentum = cameraMomentum(fields, 0, 0, {
    forward: [1, 0, 0],
    right: [0, 1, 0],
    up: [0, 0, 1],
  });
  const result = traceFrozenRay({
    provider,
    position: start,
    momentum,
    captureDistance: binaryCaptureDistance,
    escapeRadiusM: norm(start) + 1,
    maxSteps: 600,
    minStepM: 0.08,
    maxStepM: 1.2,
    maximumLookbackM: 500,
  });
  assert.equal(result.outcome, "escaped");
  const measured = Math.atan2(-result.direction[1], result.direction[0]);
  const finiteDistanceFactor = 180 / Math.hypot(180, impactM);
  const expected = 4 / impactM * finiteDistanceFactor;
  assert.ok(measured > 0);
  assert.ok(
    Math.abs(measured - expected) / expected < 0.24,
    `measured ${measured}, expected ${expected}`,
  );
  assert.ok(result.maximumRelativeEnergyDrift < 0.01);
});

test("opposite Kerr spins reverse the horizontal capture-boundary shift", () => {
  const cameraPosition = [30, 0, 0];
  const spin = 0.7;
  const positive = singleHoleProvider({
    dimensionlessSpin: [0, 0, spin],
  });
  const negative = singleHoleProvider({
    dimensionlessSpin: [0, 0, -spin],
  });
  const positiveCritical = equatorialKerrCriticalImpacts(spin);
  const negativeCritical = equatorialKerrCriticalImpacts(-spin);
  const positiveEdges = [
    slopeForSignedImpact(
      positive,
      cameraPosition,
      positiveCritical.retrograde,
    ),
    slopeForSignedImpact(
      positive,
      cameraPosition,
      positiveCritical.prograde,
    ),
  ];
  const negativeEdges = [
    slopeForSignedImpact(
      negative,
      cameraPosition,
      negativeCritical.prograde,
    ),
    slopeForSignedImpact(
      negative,
      cameraPosition,
      negativeCritical.retrograde,
    ),
  ];
  const positiveCentre = 0.5 * (positiveEdges[0] + positiveEdges[1]);
  const negativeCentre = 0.5 * (negativeEdges[0] + negativeEdges[1]);
  assert.ok(positiveCentre * negativeCentre < 0);
  assert.ok(Math.abs(positiveCentre + negativeCentre) < 2e-12);
  assert.ok(
    Math.abs(positiveEdges[0] + negativeEdges[0]) < 2e-12,
    JSON.stringify({ positiveEdges, negativeEdges }),
  );
  assert.ok(
    Math.abs(positiveEdges[1] + negativeEdges[1]) < 2e-12,
    JSON.stringify({ positiveEdges, negativeEdges }),
  );

  // A ray between the two positive-side boundaries is outside for +a but
  // inside for -a.  This checks an outcome, not only the analytic edge sign.
  const probeSlope = 0.5 * (positiveEdges[1] + negativeEdges[0]);
  const positiveFields = positive.evaluate(0, cameraPosition);
  const negativeFields = negative.evaluate(0, cameraPosition);
  const positiveRay = traceFrozenRay({
    provider: positive,
    position: cameraPosition,
    momentum: cameraMomentum(positiveFields, probeSlope),
    captureDistance: remnantCaptureDistance,
    escapeRadiusM: 34,
    maxSteps: 1900,
    maxStepM: 0.30,
  });
  const negativeRay = traceFrozenRay({
    provider: negative,
    position: cameraPosition,
    momentum: cameraMomentum(negativeFields, probeSlope),
    captureDistance: remnantCaptureDistance,
    escapeRadiusM: 34,
    maxSteps: 1900,
    maxStepM: 0.30,
  });
  assert.notEqual(positiveRay.outcome, negativeRay.outcome);
  assert.deepEqual(
    new Set([positiveRay.outcome, negativeRay.outcome]),
    new Set(["captured", "escaped"]),
    JSON.stringify({ positiveRay, negativeRay }),
  );
});

test("budget exhaustion and invalid domains remain unresolved", () => {
  const provider = singleHoleProvider({});
  const position = [30, 0, 0];
  const fields = provider.evaluate(0, position);
  const exhausted = traceFrozenRay({
    provider,
    position,
    momentum: cameraMomentum(fields, 0.4),
    captureDistance: remnantCaptureDistance,
    escapeRadiusM: 80,
    maxSteps: 1,
  });
  assert.equal(exhausted.outcome, "unresolved");
  assert.equal(exhausted.reason, "budget-exhausted");

  const invalidProvider = Object.freeze({
    schema: SPACETIME_PROVIDER_SCHEMA,
    evaluate() {
      throw new Error("deliberately invalid metric domain");
    },
  });
  const invalid = traceFrozenRay({
    provider: invalidProvider,
    position: [4, 0, 0],
    momentum: [-1, 0, 0],
    maxSteps: 8,
  });
  assert.equal(invalid.outcome, "unresolved");
  assert.equal(invalid.reason, "invalid-initial-domain");
  assert.notEqual(exhausted.outcome, "captured");
  assert.notEqual(invalid.outcome, "escaped");
});
