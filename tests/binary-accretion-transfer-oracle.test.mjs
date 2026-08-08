import assert from "node:assert/strict";
import test from "node:test";

import {
  composeDiskTransfer,
  localEmitterFrequencyShift,
  opacityFromOpticalDepth,
  segmentDiskIntersection,
  sortDiskIntersections,
} from "../src/scenes/binary-accretion-transfer-oracle.js";

function close(actual, expected, tolerance = 1e-12, label = "value") {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected}, received ${actual}`,
  );
}

function closeVector(actual, expected, tolerance = 1e-12) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    close(actual[index], expected[index], tolerance, `channel ${index}`);
  }
}

test("segment-plane intersections sort from observer to source", () => {
  const segment = {
    segmentStart: [2, 0, 2],
    segmentEnd: [2, 0, -2],
    normal: [0, 0, 1],
    innerRadiusM: 1,
    outerRadiusM: 3,
  };
  const back = segmentDiskIntersection({
    ...segment,
    id: "back",
    centre: [0, 0, -1],
  });
  const front = segmentDiskIntersection({
    ...segment,
    id: "front",
    centre: [0, 0, 1],
  });
  const innerHole = segmentDiskIntersection({
    ...segment,
    id: "inner-hole",
    segmentStart: [0, 0, 2],
    segmentEnd: [0, 0, -2],
    centre: [0, 0, 0],
  });

  assert.equal(front.valid, true);
  assert.equal(back.valid, true);
  close(front.fraction, 0.25);
  close(back.fraction, 0.75);
  close(front.radiusM, 2);
  close(back.radiusM, 2);
  assert.deepEqual(
    sortDiskIntersections([back, innerHole, front]).map(({ id }) => id),
    ["front", "back", "inner-hole"],
  );
  assert.equal(innerHole.fraction, 2);
});

test("segment oracle excludes holes, parallel rays, and repeated start hits", () => {
  const base = {
    centre: [0, 0, 0],
    normal: [0, 0, 1],
    innerRadiusM: 1,
    outerRadiusM: 3,
  };
  assert.equal(segmentDiskIntersection({
    ...base,
    segmentStart: [0, 0, 1],
    segmentEnd: [0, 0, -1],
  }).reason, "outside-annulus");
  assert.equal(segmentDiskIntersection({
    ...base,
    segmentStart: [2, 0, 1],
    segmentEnd: [2, 0, 1],
  }).reason, "no-plane-crossing");
  assert.equal(segmentDiskIntersection({
    ...base,
    segmentStart: [2, 0, 0],
    segmentEnd: [2, 0, -1],
  }).reason, "outside-open-segment-start");
});

test("local emitter frequency remains positive and reverses Doppler sign", () => {
  const speed = 0.3;
  const approaching = localEmitterFrequencyShift({
    photonEnergy: 1,
    photonMomentum: [1, 0, 0],
    emitterVelocity: [speed, 0, 0],
  });
  const receding = localEmitterFrequencyShift({
    photonEnergy: 1,
    photonMomentum: [1, 0, 0],
    emitterVelocity: [-speed, 0, 0],
  });

  assert.equal(approaching.valid, true);
  assert.equal(receding.valid, true);
  assert.ok(approaching.emitterFrequency > 0);
  assert.ok(receding.emitterFrequency > 0);
  assert.ok(approaching.relativeShift > 0);
  assert.ok(receding.relativeShift < 0);
  close(approaching.frequencyShift * receding.frequencyShift, 1);
  close(
    approaching.frequencyShift,
    Math.sqrt((1 + speed) / (1 - speed)),
  );
});

test("tau conversion and sorted front-to-back composition are analytic", () => {
  close(opacityFromOpticalDepth(0), 0);
  close(opacityFromOpticalDepth(Math.log(2)), 0.5);
  close(
    opacityFromOpticalDepth(100),
    1 - Math.exp(-30),
  );

  const result = composeDiskTransfer([
    {
      id: "back",
      fraction: 0.75,
      radiance: [0, 8, 0],
      opacity: 0.25,
    },
    {
      id: "front",
      fraction: 0.25,
      radiance: [10, 0, 0],
      opticalDepth: Math.log(2),
    },
  ], {
    backgroundRadiance: [0, 0, 4],
  });

  assert.equal(result.failed, false);
  assert.deepEqual(result.orderedIds, ["front", "back"]);
  close(result.transmittance, 0.375);
  closeVector(result.surfaceRadiance, [5, 1, 0]);
  closeVector(result.radiance, [5, 1, 1.5]);
  close(result.layers[0].incomingTransmittance, 1);
  close(result.layers[1].incomingTransmittance, 0.5);
});

test("invalid local transfer fails closed instead of exposing background", () => {
  const invalidFrequency = localEmitterFrequencyShift({
    photonEnergy: 1,
    photonMomentum: [1, 0, 0],
    emitterVelocity: [1, 0, 0],
  });
  assert.equal(invalidFrequency.valid, false);
  assert.equal(invalidFrequency.frequencyShift, 0);
  assert.equal(invalidFrequency.reason, "superluminal-emitter");

  const result = composeDiskTransfer([
    {
      id: "front",
      fraction: 0.2,
      radiance: [10, 0, 0],
      opacity: 0.5,
    },
    {
      id: "unknown-back",
      fraction: 0.7,
      valid: false,
    },
  ], {
    backgroundRadiance: [0, 20, 30],
  });
  assert.equal(result.failed, true);
  assert.equal(result.failureReason, "invalid-transfer:unknown-back");
  assert.equal(result.transmittance, 0);
  closeVector(result.radiance, [5, 0, 0]);
});

test("non-finite transfer data is rejected or made opaque fail-closed", () => {
  assert.throws(
    () => opacityFromOpticalDepth(Number.NaN),
    /finite number/,
  );
  assert.throws(
    () => segmentDiskIntersection({
      segmentStart: [2, 0, Number.POSITIVE_INFINITY],
      segmentEnd: [2, 0, -1],
      normal: [0, 0, 1],
      innerRadiusM: 1,
      outerRadiusM: 3,
    }),
    /finite number/,
  );

  const malformed = composeDiskTransfer([{
    id: "nan-emission",
    fraction: 0.5,
    radiance: [Number.NaN, 1, 1],
    opacity: 0.5,
  }], {
    backgroundRadiance: [4, 5, 6],
  });
  assert.equal(malformed.failed, true);
  assert.equal(malformed.transmittance, 0);
  closeVector(malformed.radiance, [0, 0, 0]);
  assert.match(malformed.failureReason, /^invalid-transfer:nan-emission:/);
});
