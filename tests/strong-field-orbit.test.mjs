import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  createDynamicsTrack,
} from "../src/scenes/binary-dynamics-adapter.js";
import {
  STRONG_FIELD_ORBIT_MODEL,
  createStrongFieldOrbitRuntime,
} from "../src/strong-field-orbit.js";
import {
  STRONG_FIELD_UNIFORM_ABI,
} from "../src/strong-field-spacetime.js";

const root = new URL("../", import.meta.url);
const bundledManifest = JSON.parse(
  await readFile(
    new URL("assets/scenes/binary-sxs-bbh-0001-v2.json", root),
    "utf8",
  ),
);
const bundledPayload = JSON.parse(
  await readFile(
    new URL(
      "assets/scenes/binary-sxs-bbh-0001-v2.samples.json",
      root,
    ),
    "utf8",
  ),
);

function close(actual, expected, tolerance, label = "value") {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected}, received ${actual}`,
  );
}

function finiteVector(vector) {
  assert.ok(vector.every(Number.isFinite));
}

function syntheticManifest() {
  return {
    source: {
      simulation: "synthetic:h22-only",
    },
    events: {
      commonApparentHorizonFirst: {
        tProtocolM: -10,
      },
      waveformPeak: {
        tProtocolM: 0,
      },
    },
    physicalSystem: {
      bodies: [
        {
          id: "A",
          massFraction: 0.6,
          dimensionlessSpin: [0.01, 0.02, 0.03],
        },
        {
          id: "B",
          massFraction: 0.4,
          dimensionlessSpin: [-0.02, 0.01, -0.04],
        },
      ],
      remnant: {
        massFraction: 0.95,
        dimensionlessSpin: [0.01, -0.02, 0.7],
      },
    },
  };
}

function syntheticTrack({
  centroidSeparation = 20,
  centroidPhaseOffset = 0,
  unreliableStart = Infinity,
  unreliableEnd = -Infinity,
  noisyUnreliablePhase = false,
} = {}) {
  const firstTimeM = -100;
  const finalTimeM = 20;
  function physicalPhase(timeM) {
    return (
      0.02 * timeM
      + 0.00005 * timeM * timeM
      + 0.12 * Math.tanh((timeM + 45) * 4)
    );
  }
  return {
    manifest: syntheticManifest(),
    firstTimeM,
    finalTimeM,
    sampleCount: 1201,
    sampleAt(requestedTimeM) {
      const timeM = Math.min(
        finalTimeM,
        Math.max(firstTimeM, requestedTimeM),
      );
      const unreliable = (
        timeM >= unreliableStart && timeM <= unreliableEnd
      );
      const phase = unreliable && noisyUnreliablePhase
        ? 900 * timeM
        : -2 * physicalPhase(timeM);
      const amplitude = unreliable ? 1e-12 : (
        timeM <= 0 ? 1 : Math.exp(-0.08 * timeM)
      );
      return {
        tM: timeM,
        separationM: centroidSeparation + 0.1 * Math.sin(timeM),
        orbitalPhaseRad: physicalPhase(timeM) + centroidPhaseOffset,
        renderTopologyBlend: Math.min(
          1,
          Math.max(0, (timeM + 10) / 10),
        ),
        waveform: {
          h22Real: amplitude * Math.cos(phase),
          h22Imag: amplitude * Math.sin(phase),
          amplitude,
        },
      };
    },
  };
}

function bodyDistance(sample) {
  return Math.hypot(
    ...sample.bodies[0].positionM.map(
      (component, index) => (
        component - sample.bodies[1].positionM[index]
      ),
    ),
  );
}

test("bundled SXS runtime emits the provider ABI from h22/events", () => {
  const track = createDynamicsTrack(
    bundledManifest,
    bundledPayload,
  );
  const runtime = createStrongFieldOrbitRuntime({ track });
  assert.equal(runtime.schema, STRONG_FIELD_ORBIT_MODEL.schema);
  assert.equal(
    runtime.orbitAdapter.usesSxsGaugeCentroids,
    false,
  );
  assert.match(runtime.orbitAdapter.source, /h22 phase\/events/);

  for (const timeM of [
    track.firstTimeM,
    -1000,
    -100,
    -20,
    bundledManifest.events.commonApparentHorizonFirst.tProtocolM,
    -3,
    0,
    track.finalTimeM,
  ]) {
    const frame = runtime.frameAt(timeM);
    const sample = frame.orbitState;
    assert.equal(
      frame.uniforms.length,
      STRONG_FIELD_UNIFORM_ABI.floatCount,
    );
    finiteVector([...frame.uniforms]);
    finiteVector(sample.bodies[0].positionM);
    finiteVector(sample.bodies[0].velocityC);
    finiteVector(sample.bodies[1].positionM);
    finiteVector(sample.bodies[1].velocityC);
    assert.ok(frame.kinematics.orbitalOmegaM > 0);
    assert.ok(frame.kinematics.separationM >= 2.6);

    const centreOfMassPosition = [0, 1, 2].map((axis) => (
      sample.bodies[0].massM * sample.bodies[0].positionM[axis]
      + sample.bodies[1].massM * sample.bodies[1].positionM[axis]
    ));
    const centreOfMassVelocity = [0, 1, 2].map((axis) => (
      sample.bodies[0].massM * sample.bodies[0].velocityC[axis]
      + sample.bodies[1].massM * sample.bodies[1].velocityC[axis]
    ));
    assert.ok(Math.hypot(...centreOfMassPosition) < 2e-14);
    assert.ok(Math.hypot(...centreOfMassVelocity) < 2e-14);
  }

  const peak = runtime.frameAt(0);
  close(peak.uniforms[1], 1, 0, "C2 provider transition");
  close(
    peak.orbitState.remnant.dimensionlessSpin[1],
    bundledManifest.physicalSystem.remnant.dimensionlessSpin[2],
    0,
    "source z spin mapped to renderer y",
  );
});

test("centroid separation and phase mutations cannot change physical orbit", () => {
  const firstTrack = syntheticTrack({
    centroidSeparation: 2,
    centroidPhaseOffset: -500,
  });
  const secondTrack = syntheticTrack({
    centroidSeparation: 8000,
    centroidPhaseOffset: 1200,
  });
  const first = createStrongFieldOrbitRuntime({
    track: firstTrack,
    manifest: syntheticManifest(),
  });
  const second = createStrongFieldOrbitRuntime({
    track: secondTrack,
    manifest: syntheticManifest(),
  });

  for (const timeM of [-100, -70, -30, -10, -5, 0, 15]) {
    assert.deepEqual(
      Array.from(first.frameAt(timeM).uniforms),
      Array.from(second.frameAt(timeM).uniforms),
      `gauge-centroid mutation leaked into uniforms at ${timeM} M`,
    );
    assert.deepEqual(
      first.sampleAt(timeM).bodies,
      second.sampleAt(timeM).bodies,
      `gauge-centroid mutation leaked into bodies at ${timeM} M`,
    );
  }
});

test("runtime never reads centroid-derived sample fields", () => {
  const base = syntheticTrack();
  const guardedTrack = {
    ...base,
    sampleAt(timeM) {
      const sample = base.sampleAt(timeM);
      return {
        ...sample,
        get separationM() {
          throw new Error("centroid separation was read");
        },
        get orbitalPhaseRad() {
          throw new Error("centroid phase was read");
        },
      };
    },
  };
  const runtime = createStrongFieldOrbitRuntime({
    track: guardedTrack,
    manifest: syntheticManifest(),
  });
  assert.doesNotThrow(() => runtime.frameAt(-35));
  assert.doesNotThrow(() => runtime.frameAt(-5));
});

test("low-amplitude phase gaps are bridged and frequency is bounded", () => {
  const track = syntheticTrack({
    unreliableStart: -62,
    unreliableEnd: -34,
    noisyUnreliablePhase: true,
  });
  const runtime = createStrongFieldOrbitRuntime({
    track,
    manifest: syntheticManifest(),
    model: {
      minimumOrbitalOmegaM: 0.005,
      maximumOrbitalOmegaM: 0.08,
      amplitudeFloorFraction: 1e-4,
    },
  });

  let previousPhase = -Infinity;
  for (let timeM = -90; timeM <= -20; timeM += 0.5) {
    const kinematics = runtime.frameAt(timeM).kinematics;
    assert.ok(kinematics.orbitalPhaseRad >= previousPhase);
    previousPhase = kinematics.orbitalPhaseRad;
    assert.ok(kinematics.orbitalOmegaM >= 0.005 - 1e-14);
    assert.ok(kinematics.orbitalOmegaM <= 0.08 + 1e-14);
    assert.ok(Number.isFinite(kinematics.separationM));
  }
  assert.equal(
    runtime.frameAt(-50).kinematics.waveformPhaseReliable,
    false,
  );
  assert.equal(
    runtime.frameAt(-25).kinematics.waveformPhaseReliable,
    true,
  );
});

test("quasi-circular radius and velocity use one consistent frequency", () => {
  const runtime = createStrongFieldOrbitRuntime({
    track: syntheticTrack(),
    manifest: syntheticManifest(),
  });
  for (const timeM of [-90, -60, -30, -15]) {
    const frame = runtime.frameAt(timeM);
    const kinematics = frame.kinematics;
    const totalMass = 1;
    const x = (totalMass * kinematics.orbitalOmegaM) ** (2 / 3);
    close(
      kinematics.separationM,
      totalMass / x,
      2e-6,
      "frequency-radius relation",
    );
    close(
      bodyDistance(frame.orbitState),
      kinematics.separationM,
      2e-14,
      "body separation",
    );

    const first = frame.orbitState.bodies[0];
    const second = frame.orbitState.bodies[1];
    const relativeVelocity = first.velocityC.map(
      (component, index) => component - second.velocityC[index],
    );
    const relativePosition = first.positionM.map(
      (component, index) => component - second.positionM[index],
    );
    const radialSpeed = (
      relativePosition.reduce(
        (sum, component, index) => (
          sum + component * relativeVelocity[index]
        ),
        0,
      ) / kinematics.separationM
    );
    close(
      Math.abs(radialSpeed),
      Math.abs(kinematics.separationRateC),
      2e-13,
      "radial speed",
    );
  }
});

test("body velocities are derivatives of the emitted positions", () => {
  const runtime = createStrongFieldOrbitRuntime({
    track: syntheticTrack(),
    manifest: syntheticManifest(),
  });
  const step = 2e-4;
  for (const timeM of [-70, -20, -7, -3]) {
    const lower = runtime.sampleAt(timeM - step);
    const centre = runtime.sampleAt(timeM);
    const upper = runtime.sampleAt(timeM + step);
    for (let body = 0; body < 2; body += 1) {
      for (let axis = 0; axis < 3; axis += 1) {
        const numerical = (
          upper.bodies[body].positionM[axis]
          - lower.bodies[body].positionM[axis]
        ) / (2 * step);
        close(
          centre.bodies[body].velocityC[axis],
          numerical,
          3e-5,
          `body ${body} velocity axis ${axis}`,
        );
      }
    }
  }
});

test("common-horizon and peak joins preserve value, rate, acceleration", () => {
  const runtime = createStrongFieldOrbitRuntime({
    track: syntheticTrack(),
    manifest: syntheticManifest(),
  });
  const events = [-10, 0];
  const epsilon = 1e-5;
  for (const eventTime of events) {
    const lower = runtime.frameAt(eventTime - epsilon).kinematics;
    const exact = runtime.frameAt(eventTime).kinematics;
    const upper = runtime.frameAt(eventTime + epsilon).kinematics;
    for (const key of [
      "separationM",
      "separationRateC",
      "separationAcceleration",
      "orbitalPhaseRad",
      "orbitalOmegaM",
      "orbitalOmegaRate",
    ]) {
      assert.ok(
        Math.abs(lower[key] - exact[key]) < 2e-3,
        `${key} is discontinuous below ${eventTime} M`,
      );
      assert.ok(
        Math.abs(upper[key] - exact[key]) < 2e-3,
        `${key} is discontinuous above ${eventTime} M`,
      );
    }
  }
  close(
    runtime.frameAt(-10).uniforms[1],
    0,
    0,
    "transition begins at zero",
  );
  close(
    runtime.frameAt(0).uniforms[1],
    1,
    0,
    "transition ends at one",
  );
  assert.ok(runtime.frameAt(-10 + 1e-4).uniforms[1] < 1e-10);
  assert.ok(1 - runtime.frameAt(-1e-4).uniforms[1] < 1e-10);
});

test("runtime clamps external time without producing NaN", () => {
  const track = syntheticTrack();
  const runtime = createStrongFieldOrbitRuntime({
    track,
    manifest: syntheticManifest(),
  });
  const first = runtime.frameAt(-1e9);
  const last = runtime.frameAt(1e9);
  close(first.orbitState.timeM, track.firstTimeM, 0);
  close(last.orbitState.timeM, track.finalTimeM, 0);
  finiteVector([...first.uniforms]);
  finiteVector([...last.uniforms]);
});

test("unsafe frequency and smoothing configurations fail closed", () => {
  const track = syntheticTrack();
  assert.throws(
    () => createStrongFieldOrbitRuntime({
      track,
      manifest: syntheticManifest(),
      model: {
        maximumOrbitalOmegaM: 0.49,
      },
    }),
    /superluminal relative trajectory/,
  );
  assert.throws(
    () => createStrongFieldOrbitRuntime({
      track,
      manifest: syntheticManifest(),
      model: {
        frequencySmoothingRadius: 2.5,
      },
    }),
    /frequencySmoothingRadius/,
  );
  assert.throws(
    () => createStrongFieldOrbitRuntime({
      track,
      manifest: syntheticManifest(),
      model: {
        amplitudeFloorFraction: 1,
      },
    }),
    /amplitudeFloorFraction/,
  );
});
