import assert from "node:assert/strict";
import test from "node:test";

import {
  CAMERA_GLARE_STRENGTH,
  cieDisabilityGlarePsf,
  glareLevelCount,
  glareLevelWeights,
} from "../src/glare.js";
import {
  strongFieldBinaryDualDiskShaderBundle,
  strongFieldBinaryShaderBundle,
} from "../src/strong-field-shaders.js";
import { supernovaIaShaderBundle } from "../src/supernova-ia-shaders.js";

// Energy of the PSF between two angles in the small-angle image plane
// (2 pi theta dtheta), by log-spaced quadrature.
function psfEnergy(fromDeg, toDeg) {
  const steps = 20000;
  let energy = 0;
  for (let i = 0; i < steps; i += 1) {
    const a = fromDeg * (toDeg / fromDeg) ** (i / steps);
    const b = fromDeg * (toDeg / fromDeg) ** ((i + 1) / steps);
    const theta = Math.sqrt(a * b);
    energy += cieDisabilityGlarePsf(theta) * 2 * Math.PI * (theta * Math.PI / 180) * ((b - a) * Math.PI / 180);
  }
  return energy;
}

test("the glare PSF is the CIE 146:2002 general disability-glare function", () => {
  // Young observer (A = 25 yr, 1 + (A / 62.5)^4 = 1.0256), pigmentation 0.5.
  const age = 1 + (25 / 62.5) ** 4;
  for (const theta of [0.1, 0.3, 1, 3, 10, 30]) {
    const expected = 10 / theta ** 3 + (5 / theta ** 2 + 0.05 / theta) * age + 0.00125;
    assert.ok(Math.abs(cieDisabilityGlarePsf(theta) / expected - 1) < 1e-12, `theta ${theta}`);
  }
  // The function is valid from 0.1 deg; smaller angles belong to the core.
  assert.equal(cieDisabilityGlarePsf(0.01), cieDisabilityGlarePsf(0.1));
});

test("pyramid levels carry the PSF energy from 0.1 deg to the frame diagonal", () => {
  const width = 3456;
  const height = 2234;
  const pixelAngleDeg = 44 / height;
  // Levels 2..8 (scale 1/4 to 1/256).
  const levels = glareLevelCount(width, height);
  assert.equal(levels, 7);
  const weights = glareLevelWeights({ pixelAngleDeg, levels, width, height });
  assert.equal(weights.length, levels);
  assert.ok(weights.every((w) => w >= 0));
  const total = weights.reduce((sum, w) => sum + w, 0);
  const expected = psfEnergy(0.1, Math.hypot(width, height) * pixelAngleDeg);
  assert.ok(Math.abs(total / expected - 1) < 0.01, `${total} vs ${expected}`);
  // A young eye scatters ~27% of a point source beyond 0.1 deg.
  assert.ok(total > 0.22 && total < 0.32, `scattered ${total}`);
});

test("camera glare matches a good lens and the black-hole scenes do not use it", () => {
  // Veiling glare beyond 1 deg: ~7.8% for the eye, ~2% for a good camera
  // lens (ISO 9358 veiling-glare index).
  const beyondOneDegree = psfEnergy(1, 60);
  assert.ok(beyondOneDegree > 0.06 && beyondOneDegree < 0.09, `eye ${beyondOneDegree}`);
  const camera = CAMERA_GLARE_STRENGTH * beyondOneDegree;
  assert.ok(camera > 0.015 && camera < 0.03, `camera ${camera}`);
  assert.equal(supernovaIaShaderBundle.glare.strength, CAMERA_GLARE_STRENGTH);
  // The black-hole scenes keep their unmodified display path.
  assert.equal(strongFieldBinaryShaderBundle.glare, undefined);
  assert.equal(strongFieldBinaryDualDiskShaderBundle.glare, undefined);
});
