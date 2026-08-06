/*
 * Runtime PN/EOB-compatible orbit adapter for the strong-field tracer.
 *
 * SXS supplies waveform phase and source events.  It does not supply physical
 * Kerr-Schild body positions: sample.separationM and sample.orbitalPhaseRad are
 * deliberately never read here because both were derived from gauge-dependent
 * apparent-horizon centroids.
 *
 * The coordinate trajectory uses an explicit frequency-domain,
 * quasi-circular EOB-like prescription:
 *
 *   Omega = 1/2 |d arg(h_22) / dt|
 *   x     = (M Omega)^(2/3)
 *   r     = M / x
 *
 * This is the exact Kepler relation in the test-mass Schwarzschild circular
 * limit and the leading PN relation at wide separation.  It is not a complete
 * calibrated EOB Hamiltonian.  The common-horizon-to-waveform-peak interval is
 * a C2 quintic removal trajectory; after the peak the individual-hole terms
 * have zero metric weight.
 */

import {
  STRONG_FIELD_UNIFORM_ABI,
  createPnEobOrbitAdapter,
  createStrongFieldSpacetimeProvider,
} from "./strong-field-spacetime.js";

export const STRONG_FIELD_ORBIT_MODEL = Object.freeze({
  schema: "blackhole.strong-field-orbit-runtime/v1",
  dynamicsModel: "waveform-anchored quasi-circular PN/EOB-like v1",
  coordinateFrame: "asymptotically-inertial-kerr-schild-com",
  radiusRelation: "r/M=(M Omega)^(-2/3)",
  positionSource: "analytic frequency-radius relation, not SXS centroids",
  phaseSource: "unwrapped SXS h22 phase and pinned source events",
  timeAlignment: "source protocol alignment; waveform retarded time and horizon coordinate time remain distinct",
});

const TWO_PI = 2 * Math.PI;

function requireCondition(condition, message) {
  if (!condition) {
    throw new Error(`Strong-field orbit contract violation: ${message}`);
  }
}

function finiteNumber(value, label) {
  requireCondition(
    typeof value === "number" && Number.isFinite(value),
    `${label} must be a finite number`,
  );
  return value;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function smoothstep5(value) {
  const x = clamp(value, 0, 1);
  return x * x * x * (x * (x * 6 - 15) + 10);
}

function eventTime(manifest, name) {
  const time = manifest?.events?.[name]?.tProtocolM;
  return finiteNumber(time, `manifest.events.${name}.tProtocolM`);
}

function finiteVector3(value, label) {
  requireCondition(
    Array.isArray(value)
      && value.length === 3
      && value.every(Number.isFinite),
    `${label} must be a finite three-vector`,
  );
  return value.slice();
}

function mapSxsVectorToRenderer(vector) {
  // Right-handed map: source x -> renderer x, source y -> -renderer z,
  // source orbital angular momentum +z -> renderer +y.
  return [vector[0], vector[2], -vector[1]];
}

function unwrapNearest(angle, prediction) {
  return angle + TWO_PI * Math.round((prediction - angle) / TWO_PI);
}

function median(values) {
  const sorted = values.slice().sort((first, second) => first - second);
  return sorted[Math.floor(sorted.length / 2)];
}

function sampleLinear(times, values, requestedTime) {
  const time = clamp(requestedTime, times[0], times[times.length - 1]);
  let lower = 0;
  let upper = times.length - 1;
  while (upper - lower > 1) {
    const middle = Math.floor((lower + upper) / 2);
    if (times[middle] <= time) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  const duration = times[upper] - times[lower];
  const weight = duration > 0 ? (time - times[lower]) / duration : 0;
  return values[lower] + (values[upper] - values[lower]) * weight;
}

function derivative(values, spacing) {
  return values.map((_, index) => {
    if (index === 0) {
      return (values[1] - values[0]) / spacing;
    }
    if (index === values.length - 1) {
      return (
        values[index] - values[index - 1]
      ) / spacing;
    }
    return (
      values[index + 1] - values[index - 1]
    ) / (2 * spacing);
  });
}

function smoothRobust(values, radius = 2) {
  const medianFiltered = values.map((_, index) => {
    const lower = Math.max(0, index - radius);
    const upper = Math.min(values.length, index + radius + 1);
    return median(values.slice(lower, upper));
  });
  return medianFiltered.map((_, index) => {
    let weighted = 0;
    let weightSum = 0;
    for (
      let offset = -radius;
      offset <= radius;
      offset += 1
    ) {
      const sampleIndex = clamp(
        index + offset,
        0,
        medianFiltered.length - 1,
      );
      const weight = radius + 1 - Math.abs(offset);
      weighted += medianFiltered[sampleIndex] * weight;
      weightSum += weight;
    }
    return weighted / weightSum;
  });
}

function fillUnreliablePhase(
  rawPhase,
  reliable,
  spacing,
  maximumWaveformAngularFrequency,
) {
  const count = rawPhase.length;
  const unwrapped = Array(count).fill(null);
  const firstReliable = reliable.findIndex(Boolean);
  requireCondition(
    firstReliable >= 0,
    "waveform never rises above the declared amplitude floor",
  );
  unwrapped[firstReliable] = rawPhase[firstReliable];

  let lastReliable = firstReliable;
  let lastSlope = 0;
  for (let index = firstReliable + 1; index < count; index += 1) {
    if (!reliable[index]) {
      continue;
    }
    const elapsed = (index - lastReliable) * spacing;
    const prediction = unwrapped[lastReliable] + lastSlope * elapsed;
    const candidate = unwrapNearest(rawPhase[index], prediction);
    if (lastReliable !== firstReliable) {
      lastSlope = clamp(
        (candidate - unwrapped[lastReliable]) / elapsed,
        -maximumWaveformAngularFrequency,
        maximumWaveformAngularFrequency,
      );
    } else {
      lastSlope = clamp(
        (candidate - unwrapped[lastReliable]) / elapsed,
        -maximumWaveformAngularFrequency,
        maximumWaveformAngularFrequency,
      );
    }
    unwrapped[index] = candidate;
    lastReliable = index;
  }

  // Fill gaps between reliable samples by interpolation.  The unwrap above
  // uses a slope prediction, so an arbitrarily long low-amplitude interval
  // does not force the next reliable sample onto the nearest wrong 2pi branch.
  let left = firstReliable;
  for (let index = firstReliable + 1; index < count; index += 1) {
    if (unwrapped[index] === null) {
      continue;
    }
    for (let gap = left + 1; gap < index; gap += 1) {
      const weight = (gap - left) / (index - left);
      unwrapped[gap] = (
        unwrapped[left]
        + (unwrapped[index] - unwrapped[left]) * weight
      );
    }
    left = index;
  }

  const trailingSlope = left > firstReliable
    ? clamp(
      (unwrapped[left] - unwrapped[left - 1]) / spacing,
      -maximumWaveformAngularFrequency,
      maximumWaveformAngularFrequency,
    )
    : 0;
  for (let index = left + 1; index < count; index += 1) {
    unwrapped[index] = (
      unwrapped[left] + trailingSlope * (index - left) * spacing
    );
  }

  let secondReliable = firstReliable + 1;
  while (
    secondReliable < count
    && unwrapped[secondReliable] === null
  ) {
    secondReliable += 1;
  }
  const leadingSlope = secondReliable < count
    ? (unwrapped[secondReliable] - unwrapped[firstReliable])
      / ((secondReliable - firstReliable) * spacing)
    : trailingSlope;
  for (let index = firstReliable - 1; index >= 0; index -= 1) {
    unwrapped[index] = (
      unwrapped[firstReliable]
      - leadingSlope * (firstReliable - index) * spacing
    );
  }
  requireCondition(
    unwrapped.every(Number.isFinite),
    "unwrapped waveform phase is not finite",
  );
  return unwrapped;
}

function quinticBoundary({
  startValue,
  startRate,
  startAcceleration,
  endValue,
  endRate,
  endAcceleration,
  duration,
}) {
  requireCondition(duration > 0, "quintic duration must be positive");
  const coefficients = [
    startValue,
    duration * startRate,
    0.5 * duration * duration * startAcceleration,
  ];
  const valueResidual = endValue
    - coefficients[0] - coefficients[1] - coefficients[2];
  const rateResidual = duration * endRate
    - coefficients[1] - 2 * coefficients[2];
  const accelerationResidual = duration * duration * endAcceleration
    - 2 * coefficients[2];
  coefficients.push(
    10 * valueResidual - 4 * rateResidual
      + 0.5 * accelerationResidual,
    -15 * valueResidual + 7 * rateResidual
      - accelerationResidual,
    6 * valueResidual - 3 * rateResidual
      + 0.5 * accelerationResidual,
  );

  return Object.freeze({
    evaluate(timeSinceStart) {
      const u = clamp(timeSinceStart / duration, 0, 1);
      const [a0, a1, a2, a3, a4, a5] = coefficients;
      const value = a0 + u * (
        a1 + u * (a2 + u * (a3 + u * (a4 + u * a5)))
      );
      const rate = (
        a1 + u * (
          2 * a2 + u * (3 * a3 + u * (4 * a4 + u * 5 * a5))
        )
      ) / duration;
      const acceleration = (
        2 * a2 + u * (
          6 * a3 + u * (12 * a4 + u * 20 * a5)
        )
      ) / (duration * duration);
      return Object.freeze({ value, rate, acceleration });
    },
  });
}

function validateTrack(track) {
  requireCondition(
    track
      && typeof track.sampleAt === "function"
      && Number.isFinite(track.firstTimeM)
      && Number.isFinite(track.finalTimeM)
      && track.firstTimeM < track.finalTimeM,
    "track must expose sampleAt(), firstTimeM, and finalTimeM",
  );
  const sample = track.sampleAt(track.firstTimeM);
  requireCondition(
    sample?.waveform
      && Number.isFinite(sample.waveform.h22Real)
      && Number.isFinite(sample.waveform.h22Imag)
      && Number.isFinite(sample.waveform.amplitude),
    "track samples must expose a finite complex h22 waveform",
  );
}

function physicalSystemFromManifest(manifest) {
  const sourceBodies = manifest?.physicalSystem?.bodies;
  requireCondition(
    Array.isArray(sourceBodies) && sourceBodies.length === 2,
    "manifest must contain exactly two source bodies",
  );
  const bodies = sourceBodies.map((body, index) => {
    const massM = finiteNumber(
      body.massFraction,
      `manifest body ${index} massFraction`,
    );
    requireCondition(massM > 0, "body masses must be positive");
    const sourceSpin = finiteVector3(
      body.dimensionlessSpin,
      `manifest body ${index} dimensionlessSpin`,
    );
    return Object.freeze({
      id: String(body.id ?? (index === 0 ? "A" : "B")),
      massM,
      dimensionlessSpin: Object.freeze(
        mapSxsVectorToRenderer(sourceSpin),
      ),
    });
  });
  const remnantSource = manifest?.physicalSystem?.remnant;
  requireCondition(remnantSource, "manifest remnant is missing");
  const remnant = Object.freeze({
    id: "R",
    massM: finiteNumber(
      remnantSource.massFraction,
      "manifest remnant massFraction",
    ),
    dimensionlessSpin: Object.freeze(mapSxsVectorToRenderer(
      finiteVector3(
        remnantSource.dimensionlessSpin,
        "manifest remnant dimensionlessSpin",
      ),
    )),
  });
  requireCondition(
    remnant.massM > 0,
    "remnant mass must be positive",
  );
  return Object.freeze({
    bodies: Object.freeze(bodies),
    remnant,
    totalInspiralMassM: bodies[0].massM + bodies[1].massM,
  });
}

function buildWaveformModel(track, manifest, options) {
  const firstTimeM = track.firstTimeM;
  const commonHorizonTimeM = eventTime(
    manifest,
    "commonApparentHorizonFirst",
  );
  const waveformPeakTimeM = eventTime(manifest, "waveformPeak");
  requireCondition(
    firstTimeM < commonHorizonTimeM
      && commonHorizonTimeM < waveformPeakTimeM
      && waveformPeakTimeM <= track.finalTimeM,
    "source events are not ordered inside the track",
  );

  const requestedCount = options.phaseSampleCount ?? Math.max(
    4097,
    2 * (Number.isInteger(track.sampleCount) ? track.sampleCount : 2048) + 1,
  );
  const sampleCount = clamp(Math.round(requestedCount), 1025, 16385);
  const spacing = (
    waveformPeakTimeM - firstTimeM
  ) / (sampleCount - 1);
  const times = Array.from(
    { length: sampleCount },
    (_, index) => firstTimeM + index * spacing,
  );
  const waveform = times.map((timeM) => track.sampleAt(timeM).waveform);
  const amplitudes = waveform.map((sample) => sample.amplitude);
  const peakAmplitude = Math.max(...amplitudes);
  const amplitudeFloorFraction = (
    options.amplitudeFloorFraction ?? 1e-3
  );
  requireCondition(
    amplitudeFloorFraction > 0 && amplitudeFloorFraction < 1,
    "amplitudeFloorFraction must lie in (0,1)",
  );
  const amplitudeFloor = Math.max(
    options.absoluteAmplitudeFloor ?? 1e-8,
    peakAmplitude * amplitudeFloorFraction,
  );
  const reliable = amplitudes.map((amplitude) => (
    Number.isFinite(amplitude) && amplitude >= amplitudeFloor
  ));
  const rawPhase = waveform.map((sample) => (
    Math.atan2(sample.h22Imag, sample.h22Real)
  ));
  const maximumOrbitalOmegaM = (
    options.maximumOrbitalOmegaM ?? 0.24
  );
  const minimumOrbitalOmegaM = (
    options.minimumOrbitalOmegaM ?? 0.001
  );
  requireCondition(
    minimumOrbitalOmegaM > 0
      && minimumOrbitalOmegaM < maximumOrbitalOmegaM
      && maximumOrbitalOmegaM < 0.5,
    "orbital frequency limits are invalid",
  );
  requireCondition(
    2 * maximumOrbitalOmegaM * spacing < 0.8 * Math.PI,
    "phase grid does not resolve the declared maximum h22 frequency",
  );
  const unwrappedWaveform = fillUnreliablePhase(
    rawPhase,
    reliable,
    spacing,
    2 * maximumOrbitalOmegaM,
  );
  const waveformDerivative = derivative(unwrappedWaveform, spacing);
  const reliableDerivatives = waveformDerivative.filter((_, index) => (
    reliable[index]
  ));
  const phaseOrientation = median(reliableDerivatives) >= 0 ? 0.5 : -0.5;
  const waveformOrbitalPhase = unwrappedWaveform.map(
    (phase) => phaseOrientation * phase,
  );
  const rawOmega = derivative(waveformOrbitalPhase, spacing).map(
    (omega) => clamp(
      Math.abs(omega),
      minimumOrbitalOmegaM,
      maximumOrbitalOmegaM,
    ),
  );
  const frequencySmoothingRadius = (
    options.frequencySmoothingRadius ?? 2
  );
  requireCondition(
    Number.isInteger(frequencySmoothingRadius)
      && frequencySmoothingRadius >= 0
      && frequencySmoothingRadius <= 16,
    "frequencySmoothingRadius must be an integer in [0,16]",
  );
  const omega = smoothRobust(rawOmega, frequencySmoothingRadius)
    .map((value) => clamp(
      value,
      minimumOrbitalOmegaM,
      maximumOrbitalOmegaM,
    ));

  // Integrate the filtered positive frequency to make phase and velocity use
  // the same dynamics.  A constant is then fixed by the SXS waveform phase at
  // the common-horizon event; no centroid position or separation enters.
  const integratedPhase = Array(sampleCount).fill(0);
  for (let index = 1; index < sampleCount; index += 1) {
    integratedPhase[index] = (
      integratedPhase[index - 1]
      + 0.5 * (omega[index - 1] + omega[index]) * spacing
    );
  }
  const commonWaveformPhase = sampleLinear(
    times,
    waveformOrbitalPhase,
    commonHorizonTimeM,
  );
  const phaseOffset = (
    commonWaveformPhase
    - sampleLinear(times, integratedPhase, commonHorizonTimeM)
  );
  const orientationAtPeakRad = options.orientationAtPeakRad ?? 0;
  const waveformPeakPhase = sampleLinear(
    times,
    waveformOrbitalPhase,
    waveformPeakTimeM,
  );
  const orientationOffset = orientationAtPeakRad - waveformPeakPhase;
  const phase = integratedPhase.map(
    (value) => value + phaseOffset + orientationOffset,
  );

  const totalMassM = physicalSystemFromManifest(
    manifest,
  ).totalInspiralMassM;
  const minimumSeparationM = options.minimumSeparationM ?? 2.6;
  const maximumSeparationM = options.maximumSeparationM ?? 30;
  requireCondition(
    minimumSeparationM > 2
      && maximumSeparationM > minimumSeparationM,
    "separation limits are invalid",
  );
  const maximumInspiralSpeedC = (
    options.maximumInspiralSpeedC ?? 0.18
  );
  requireCondition(
    maximumInspiralSpeedC > 0
      && Math.hypot(
        maximumOrbitalOmegaM * minimumSeparationM,
        maximumInspiralSpeedC,
      ) < 0.95,
    "frequency/separation limits permit a superluminal relative trajectory",
  );
  const separation = omega.map((value) => {
    const x = (totalMassM * value) ** (2 / 3);
    return clamp(
      totalMassM / Math.max(x, 1e-12),
      minimumSeparationM,
      maximumSeparationM,
    );
  });
  const separationRate = smoothRobust(
    derivative(separation, spacing),
    2,
  ).map((value) => clamp(
    value,
    -maximumInspiralSpeedC,
    0,
  ));
  const separationAcceleration = derivative(separationRate, spacing);
  const omegaRate = derivative(omega, spacing);

  const common = {
    separationM: sampleLinear(
      times,
      separation,
      commonHorizonTimeM,
    ),
    separationRateC: sampleLinear(
      times,
      separationRate,
      commonHorizonTimeM,
    ),
    separationAcceleration: sampleLinear(
      times,
      separationAcceleration,
      commonHorizonTimeM,
    ),
    phaseRad: sampleLinear(times, phase, commonHorizonTimeM),
    omegaM: sampleLinear(times, omega, commonHorizonTimeM),
    omegaRate: sampleLinear(times, omegaRate, commonHorizonTimeM),
  };
  const peakOmega = sampleLinear(times, omega, waveformPeakTimeM);
  const peakSeparation = clamp(
    totalMassM / Math.max(
      (totalMassM * peakOmega) ** (2 / 3),
      1e-12,
    ),
    minimumSeparationM,
    maximumSeparationM,
  );
  const transitionDuration = waveformPeakTimeM - commonHorizonTimeM;
  const separationTransition = quinticBoundary({
    startValue: common.separationM,
    startRate: common.separationRateC,
    startAcceleration: common.separationAcceleration,
    endValue: peakSeparation,
    endRate: 0,
    endAcceleration: 0,
    duration: transitionDuration,
  });
  const phaseTransition = quinticBoundary({
    startValue: common.phaseRad,
    startRate: common.omegaM,
    startAcceleration: common.omegaRate,
    endValue: orientationAtPeakRad,
    endRate: peakOmega,
    endAcceleration: 0,
    duration: transitionDuration,
  });

  function kinematicsAt(requestedTimeM) {
    const timeM = clamp(
      requestedTimeM,
      firstTimeM,
      track.finalTimeM,
    );
    let separationState;
    let phaseState;
    let rawMergerBlend;
    if (timeM <= commonHorizonTimeM) {
      separationState = {
        value: sampleLinear(times, separation, timeM),
        rate: sampleLinear(times, separationRate, timeM),
        acceleration: sampleLinear(
          times,
          separationAcceleration,
          timeM,
        ),
      };
      phaseState = {
        value: sampleLinear(times, phase, timeM),
        rate: sampleLinear(times, omega, timeM),
        acceleration: sampleLinear(times, omegaRate, timeM),
      };
      rawMergerBlend = 0;
    } else if (timeM < waveformPeakTimeM) {
      const elapsed = timeM - commonHorizonTimeM;
      separationState = separationTransition.evaluate(elapsed);
      phaseState = phaseTransition.evaluate(elapsed);
      rawMergerBlend = elapsed / transitionDuration;
    } else {
      separationState = {
        value: peakSeparation,
        rate: 0,
        acceleration: 0,
      };
      phaseState = {
        value: orientationAtPeakRad
          + peakOmega * (timeM - waveformPeakTimeM),
        rate: peakOmega,
        acceleration: 0,
      };
      rawMergerBlend = 1;
    }
    const sourceSample = track.sampleAt(timeM);
    return Object.freeze({
      timeM,
      separationM: separationState.value,
      separationRateC: separationState.rate,
      separationAcceleration: separationState.acceleration,
      orbitalPhaseRad: phaseState.value,
      orbitalOmegaM: phaseState.rate,
      orbitalOmegaRate: phaseState.acceleration,
      rawMergerBlend: clamp(rawMergerBlend, 0, 1),
      transitionWeight: smoothstep5(rawMergerBlend),
      waveformAmplitude: sourceSample.waveform.amplitude,
      waveformPhaseReliable: (
        sourceSample.waveform.amplitude >= amplitudeFloor
      ),
      phaseSource: "unwrapped h22; centroid phase unused",
      radiusSource: STRONG_FIELD_ORBIT_MODEL.radiusRelation,
    });
  }

  return Object.freeze({
    firstTimeM,
    finalTimeM: track.finalTimeM,
    commonHorizonTimeM,
    waveformPeakTimeM,
    amplitudeFloor,
    minimumOrbitalOmegaM,
    maximumOrbitalOmegaM,
    minimumSeparationM,
    maximumSeparationM,
    phaseOrientation,
    phaseSampleCount: sampleCount,
    kinematicsAt,
  });
}

export function createStrongFieldOrbitRuntime({
  track,
  manifest = track?.manifest,
  model: modelOptions = {},
  provider: providerOptions = {},
}) {
  validateTrack(track);
  requireCondition(
    manifest && typeof manifest === "object",
    "manifest is required",
  );
  const physicalSystem = physicalSystemFromManifest(manifest);
  const waveformModel = buildWaveformModel(
    track,
    manifest,
    modelOptions,
  );
  const [bodyA, bodyB] = physicalSystem.bodies;
  const totalMassM = physicalSystem.totalInspiralMassM;

  function orbitSample(timeM) {
    const kinematics = waveformModel.kinematicsAt(timeM);
    const phase = kinematics.orbitalPhaseRad;
    const radial = [Math.cos(phase), 0, -Math.sin(phase)];
    const tangent = [-Math.sin(phase), 0, -Math.cos(phase)];
    const relativePosition = radial.map(
      (component) => component * kinematics.separationM,
    );
    const relativeVelocity = radial.map((component, index) => (
      component * kinematics.separationRateC
      + tangent[index]
        * kinematics.separationM
        * kinematics.orbitalOmegaM
    ));
    const scaleA = -bodyB.massM / totalMassM;
    const scaleB = bodyA.massM / totalMassM;
    const mapBody = (body, scaleFactor) => Object.freeze({
      id: body.id,
      massM: body.massM,
      positionM: relativePosition.map(
        (component) => component * scaleFactor,
      ),
      velocityC: relativeVelocity.map(
        (component) => component * scaleFactor,
      ),
      dimensionlessSpin: body.dimensionlessSpin.slice(),
    });
    return Object.freeze({
      bodies: Object.freeze([
        mapBody(bodyA, scaleA),
        mapBody(bodyB, scaleB),
      ]),
      remnant: Object.freeze({
        id: physicalSystem.remnant.id,
        massM: physicalSystem.remnant.massM,
        positionM: [0, 0, 0],
        velocityC: [0, 0, 0],
        dimensionlessSpin: (
          physicalSystem.remnant.dimensionlessSpin.slice()
        ),
      }),
      mergerBlend: kinematics.rawMergerBlend,
    });
  }

  const orbitAdapter = createPnEobOrbitAdapter({
    dynamicsModel: STRONG_FIELD_ORBIT_MODEL.dynamicsModel,
    coordinateFrame: STRONG_FIELD_ORBIT_MODEL.coordinateFrame,
    source: [
      manifest?.source?.simulation ?? "declared waveform source",
      "h22 phase/events + analytic frequency-radius trajectory",
    ].join("; "),
    usesSxsGaugeCentroids: false,
    sample: orbitSample,
  });
  const spacetimeProvider = createStrongFieldSpacetimeProvider({
    orbitAdapter,
    ...providerOptions,
  });

  function frameAt(requestedTimeM) {
    const timeM = clamp(
      finiteNumber(requestedTimeM, "frame timeM"),
      track.firstTimeM,
      track.finalTimeM,
    );
    const frame = spacetimeProvider.frameAt(timeM);
    requireCondition(
      frame.uniforms.length === STRONG_FIELD_UNIFORM_ABI.floatCount,
      "spacetime provider returned the wrong uniform ABI",
    );
    return Object.freeze({
      ...frame,
      kinematics: waveformModel.kinematicsAt(timeM),
    });
  }

  return Object.freeze({
    schema: STRONG_FIELD_ORBIT_MODEL.schema,
    model: STRONG_FIELD_ORBIT_MODEL,
    manifest,
    physicalSystem,
    waveformModel,
    orbitAdapter,
    spacetimeProvider,
    sampleAt(timeM) {
      return orbitAdapter.sample(
        clamp(timeM, track.firstTimeM, track.finalTimeM),
      );
    },
    frameAt,
  });
}
