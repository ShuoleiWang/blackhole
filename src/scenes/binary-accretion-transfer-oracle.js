/*
 * Renderer-independent CPU oracle for the dual-mini-disk transfer contract.
 *
 * Geometry follows the production fast-light segment convention, while the
 * frequency helper operates in one local orthonormal frame. This is a compact
 * numerical reference for tests, not a second renderer or a GRMHD model.
 */

const SEGMENT_ENDPOINT_EPSILON = 1e-5;
const SEGMENT_PARALLEL_EPSILON = 1e-7;
const NORMAL_LENGTH_TOLERANCE = 1e-3;
const MAXIMUM_OPTICAL_DEPTH = 30;

function finiteNumber(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be a finite number`);
  }
  return value;
}

function positiveFinite(value, label) {
  const number = finiteNumber(value, label);
  if (number <= 0) {
    throw new RangeError(`${label} must be positive`);
  }
  return number;
}

function nonNegativeFinite(value, label) {
  const number = finiteNumber(value, label);
  if (number < 0) {
    throw new RangeError(`${label} must be non-negative`);
  }
  return number;
}

function vector3(value, label) {
  if (
    !(Array.isArray(value) || ArrayBuffer.isView(value))
    || value.length !== 3
  ) {
    throw new TypeError(`${label} must contain exactly three values`);
  }
  return Object.freeze(Array.from(value, (entry, index) => (
    finiteNumber(entry, `${label}[${index}]`)
  )));
}

function nonNegativeVector3(value, label) {
  const result = vector3(value, label);
  if (result.some((entry) => entry < 0)) {
    throw new RangeError(`${label} must be component-wise non-negative`);
  }
  return result;
}

function dot(first, second) {
  return first[0] * second[0]
    + first[1] * second[1]
    + first[2] * second[2];
}

function subtract(first, second) {
  return [
    first[0] - second[0],
    first[1] - second[1],
    first[2] - second[2],
  ];
}

function addScaled(vector, direction, scale) {
  return [
    vector[0] + direction[0] * scale,
    vector[1] + direction[1] * scale,
    vector[2] + direction[2] * scale,
  ];
}

function interpolate(first, second, fraction) {
  return [
    first[0] + (second[0] - first[0]) * fraction,
    first[1] + (second[1] - first[1]) * fraction,
    first[2] + (second[2] - first[2]) * fraction,
  ];
}

function magnitude(value) {
  return Math.hypot(...value);
}

function invalidIntersection(id, reason, failed = false) {
  return Object.freeze({
    id,
    valid: false,
    failed,
    reason,
    fraction: 2,
    radiusM: 0,
    position: Object.freeze([0, 0, 0]),
    restPosition: Object.freeze([0, 0, 0]),
  });
}

function bodyRestDisplacement(position, centre, velocity) {
  const displacement = subtract(position, centre);
  const speedSquared = dot(velocity, velocity);
  if (speedSquared <= 1e-12) {
    return displacement;
  }
  if (speedSquared >= 0.9999) {
    return null;
  }
  const gamma = 1 / Math.sqrt(1 - speedSquared);
  return addScaled(
    displacement,
    velocity,
    (gamma - 1) * dot(displacement, velocity) / speedSquared,
  );
}

/**
 * Locate one annulus crossing on an accepted observer-to-source ray segment.
 * A valid hit includes both endpoints except fraction zero, which prevents the
 * same surface from being counted at the start of the following segment.
 */
export function segmentDiskIntersection({
  id = "disk",
  segmentStart,
  segmentEnd,
  centre = [0, 0, 0],
  velocity = [0, 0, 0],
  normal,
  innerRadiusM,
  outerRadiusM,
  activeWeight = 1,
} = {}) {
  const start = vector3(segmentStart, "segmentStart");
  const end = vector3(segmentEnd, "segmentEnd");
  const diskCentre = vector3(centre, "centre");
  const bodyVelocity = vector3(velocity, "velocity");
  const diskNormal = vector3(normal, "normal");
  const innerRadius = positiveFinite(innerRadiusM, "innerRadiusM");
  const outerRadius = positiveFinite(outerRadiusM, "outerRadiusM");
  const weight = nonNegativeFinite(activeWeight, "activeWeight");
  if (outerRadius <= innerRadius) {
    throw new RangeError("outerRadiusM must exceed innerRadiusM");
  }
  if (weight <= 1e-6) {
    return invalidIntersection(id, "inactive");
  }

  const normalLength = magnitude(diskNormal);
  if (
    normalLength < 1 - NORMAL_LENGTH_TOLERANCE
    || normalLength > 1 + NORMAL_LENGTH_TOLERANCE
  ) {
    return invalidIntersection(id, "non-unit-normal", true);
  }
  const unitNormal = diskNormal.map((entry) => entry / normalLength);
  const restStart = bodyRestDisplacement(start, diskCentre, bodyVelocity);
  const restEnd = bodyRestDisplacement(end, diskCentre, bodyVelocity);
  if (restStart === null || restEnd === null) {
    return invalidIntersection(id, "superluminal-body", true);
  }

  const sideStart = dot(restStart, unitNormal);
  const sideEnd = dot(restEnd, unitNormal);
  const denominator = sideStart - sideEnd;
  if (
    Math.abs(denominator) <= SEGMENT_PARALLEL_EPSILON
    || sideStart * sideEnd > 0
  ) {
    return invalidIntersection(id, "no-plane-crossing");
  }
  const fraction = sideStart / denominator;
  if (fraction <= SEGMENT_ENDPOINT_EPSILON || fraction > 1) {
    return invalidIntersection(id, "outside-open-segment-start");
  }

  const restHit = interpolate(restStart, restEnd, fraction);
  const planarHit = addScaled(restHit, unitNormal, -dot(restHit, unitNormal));
  const radiusM = magnitude(planarHit);
  if (radiusM < innerRadius || radiusM > outerRadius) {
    return invalidIntersection(id, "outside-annulus");
  }
  return Object.freeze({
    id,
    valid: true,
    failed: false,
    reason: null,
    fraction,
    radiusM,
    position: Object.freeze(interpolate(start, end, fraction)),
    restPosition: Object.freeze(planarHit),
  });
}

/** Return a deterministic observer-to-source ordering without mutating input. */
export function sortDiskIntersections(intersections) {
  if (!Array.isArray(intersections)) {
    throw new TypeError("intersections must be an array");
  }
  const decorated = intersections.map((intersection, index) => {
    if (!intersection || typeof intersection !== "object") {
      throw new TypeError(`intersections[${index}] must be an object`);
    }
    finiteNumber(intersection.fraction, `intersections[${index}].fraction`);
    return { intersection, index };
  });
  decorated.sort((first, second) => (
    first.intersection.fraction - second.intersection.fraction
    || first.index - second.index
  ));
  return Object.freeze(decorated.map(({ intersection }) => intersection));
}

function invalidFrequency(reason, emitterFrequency = 0) {
  return Object.freeze({
    valid: false,
    reason,
    emitterFrequency,
    frequencyShift: 0,
    relativeShift: 0,
  });
}

/**
 * Evaluate g = nu_observer / nu_emitter in one local orthonormal frame.
 * photonMomentum is the local covariant spatial momentum used by the WGSL
 * emitter contraction. Subluminal future-directed states must return g > 0.
 */
export function localEmitterFrequencyShift({
  photonEnergy,
  photonMomentum,
  emitterVelocity,
  observerFrequency = photonEnergy,
} = {}) {
  const energy = positiveFinite(photonEnergy, "photonEnergy");
  const momentum = vector3(photonMomentum, "photonMomentum");
  const velocity = vector3(emitterVelocity, "emitterVelocity");
  const observer = positiveFinite(observerFrequency, "observerFrequency");
  const speedSquared = dot(velocity, velocity);
  if (speedSquared >= 1) {
    return invalidFrequency("superluminal-emitter");
  }
  const gamma = 1 / Math.sqrt(1 - speedSquared);
  const emitterFrequency = gamma * (energy - dot(momentum, velocity));
  if (!Number.isFinite(emitterFrequency) || emitterFrequency <= 1e-6) {
    return invalidFrequency("non-positive-emitter-frequency", emitterFrequency);
  }
  const frequencyShift = observer / emitterFrequency;
  if (!Number.isFinite(frequencyShift) || frequencyShift <= 0) {
    return invalidFrequency("non-positive-frequency-shift", emitterFrequency);
  }
  return Object.freeze({
    valid: true,
    reason: null,
    emitterFrequency,
    frequencyShift,
    relativeShift: frequencyShift - 1,
  });
}

/** Convert line-of-sight optical depth into bounded absorptive opacity. */
export function opacityFromOpticalDepth(opticalDepth) {
  const tau = nonNegativeFinite(opticalDepth, "opticalDepth");
  return 1 - Math.exp(-Math.min(tau, MAXIMUM_OPTICAL_DEPTH));
}

function freezeLayer(layer) {
  return Object.freeze({
    ...layer,
    contribution: Object.freeze(layer.contribution),
  });
}

function compositionResult({
  surfaceRadiance,
  backgroundRadiance,
  transmittance,
  failed,
  failureReason,
  orderedIds,
  layers,
}) {
  const finalRadiance = failed
    ? surfaceRadiance
    : surfaceRadiance.map((entry, index) => (
      entry + transmittance * backgroundRadiance[index]
    ));
  return Object.freeze({
    radiance: Object.freeze(finalRadiance),
    surfaceRadiance: Object.freeze(surfaceRadiance),
    transmittance,
    failed,
    failureReason,
    orderedIds: Object.freeze(orderedIds),
    layers: Object.freeze(layers),
  });
}

/**
 * Compose real surface hits from observer to source. A malformed or explicitly
 * invalid transfer sample represents an intercepted but numerically unknown
 * surface: it sets transmittance to zero instead of exposing fake background.
 */
export function composeDiskTransfer(
  samples,
  { backgroundRadiance = [0, 0, 0] } = {},
) {
  if (!Array.isArray(samples)) {
    throw new TypeError("samples must be an array");
  }
  const background = nonNegativeVector3(
    backgroundRadiance,
    "backgroundRadiance",
  );
  const indexed = samples.map((sample, index) => ({ sample, index }));
  for (const { sample, index } of indexed) {
    if (!sample || typeof sample !== "object") {
      throw new TypeError(`samples[${index}] must be an object`);
    }
    if (
      typeof sample.fraction !== "number"
      || !Number.isFinite(sample.fraction)
      || sample.fraction <= 0
      || sample.fraction > 1
    ) {
      return compositionResult({
        surfaceRadiance: [0, 0, 0],
        backgroundRadiance: background,
        transmittance: 0,
        failed: true,
        failureReason: `invalid-fraction:${sample.id ?? index}`,
        orderedIds: [],
        layers: [],
      });
    }
  }
  indexed.sort((first, second) => (
    first.sample.fraction - second.sample.fraction
    || first.index - second.index
  ));

  const surfaceRadiance = [0, 0, 0];
  const orderedIds = [];
  const layers = [];
  let transmittance = 1;
  for (const { sample, index } of indexed) {
    const id = sample.id ?? `sample-${index}`;
    orderedIds.push(id);
    if (sample.valid === false) {
      return compositionResult({
        surfaceRadiance,
        backgroundRadiance: background,
        transmittance: 0,
        failed: true,
        failureReason: `invalid-transfer:${id}`,
        orderedIds,
        layers,
      });
    }

    let radiance;
    let opacity;
    try {
      radiance = nonNegativeVector3(sample.radiance, `${id}.radiance`);
      const hasTau = sample.opticalDepth !== undefined;
      const hasOpacity = sample.opacity !== undefined;
      if (hasTau === hasOpacity) {
        throw new TypeError(
          `${id} must provide exactly one of opticalDepth or opacity`,
        );
      }
      if (hasTau) {
        opacity = opacityFromOpticalDepth(sample.opticalDepth);
      } else {
        opacity = finiteNumber(sample.opacity, `${id}.opacity`);
        if (opacity < 0 || opacity > 1) {
          throw new RangeError(`${id}.opacity must lie in [0, 1]`);
        }
      }
    } catch (error) {
      return compositionResult({
        surfaceRadiance,
        backgroundRadiance: background,
        transmittance: 0,
        failed: true,
        failureReason: `invalid-transfer:${id}:${error.message}`,
        orderedIds,
        layers,
      });
    }

    const incomingTransmittance = transmittance;
    const contribution = radiance.map(
      (entry) => incomingTransmittance * opacity * entry,
    );
    for (let channel = 0; channel < 3; channel += 1) {
      surfaceRadiance[channel] += contribution[channel];
    }
    transmittance *= 1 - opacity;
    layers.push(freezeLayer({
      id,
      fraction: sample.fraction,
      opacity,
      incomingTransmittance,
      outgoingTransmittance: transmittance,
      contribution,
    }));
  }

  return compositionResult({
    surfaceRadiance,
    backgroundRadiance: background,
    transmittance,
    failed: false,
    failureReason: null,
    orderedIds,
    layers,
  });
}
