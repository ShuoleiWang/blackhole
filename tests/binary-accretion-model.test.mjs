import assert from "node:assert/strict";
import test from "node:test";

import {
  SCHWARZSCHILD_ISCO_FACTOR,
  TIDAL_TRUNCATION_FACTOR,
  analyticDiskSurfaceStructureWeight,
  annulusEdgeCoverage,
  createBinaryAccretionState,
  eggletonRocheLobeFraction,
  stableAnnulusWeight,
  thinDiskPeakTemperatureK,
  visibleBlackbodyLinearSrgbPerBolometric,
  zeroTorqueThinDiskFluxShape,
  zeroTorqueThinDiskTemperatureShape,
} from "../src/scenes/binary-accretion-model.js";

function close(actual, expected, tolerance = 1e-12) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${expected}, received ${actual}`,
  );
}

test("equal-mass binary at 30 M has two tidally truncated stable disks", () => {
  const state = createBinaryAccretionState({
    separationM: 30,
    massFractions: [0.5, 0.5],
  });
  const equalMassRocheFraction = 0.49 / (0.6 + Math.log(2));
  close(eggletonRocheLobeFraction(1), equalMassRocheFraction);
  const rocheRadiusM = 30 * equalMassRocheFraction;
  const outerRadiusM = TIDAL_TRUNCATION_FACTOR * rocheRadiusM;

  assert.equal(state.phase, "dual-stable");
  assert.equal(state.disks.length, 2);
  for (const disk of state.disks) {
    assert.equal(disk.massRatio, 1);
    assert.equal(disk.innerRadiusM, SCHWARZSCHILD_ISCO_FACTOR * 0.5);
    close(disk.rocheLobeRadiusM, rocheRadiusM);
    close(disk.outerRadiusM, outerRadiusM);
    close(disk.widthM, outerRadiusM - disk.innerRadiusM);
    assert.equal(disk.stable, true);
  }
});

test("shrinking separation monotonically truncates disks through all phases", () => {
  const states = [30, 10, 5].map((separationM) => (
    createBinaryAccretionState({
      separationM,
      massFractions: [0.8, 0.2],
    })
  ));

  assert.deepEqual(
    states.map(({ phase }) => phase),
    ["dual-stable", "tidally-marginal", "disrupted"],
  );
  for (let diskIndex = 0; diskIndex < 2; diskIndex += 1) {
    assert.ok(
      states[0].disks[diskIndex].outerRadiusM
        > states[1].disks[diskIndex].outerRadiusM,
    );
    assert.ok(
      states[1].disks[diskIndex].outerRadiusM
        > states[2].disks[diskIndex].outerRadiusM,
    );
  }
});

test("outer cap and Schwarzschild ISCO independently bound each annulus", () => {
  const capped = createBinaryAccretionState({
    separationM: 100,
    massFractions: [0.5, 0.5],
    maximumOuterRadiusM: 7,
  });
  assert.deepEqual(capped.disks.map((disk) => disk.outerRadiusM), [7, 7]);
  assert.deepEqual(capped.disks.map((disk) => disk.innerRadiusM), [3, 3]);

  const disrupted = createBinaryAccretionState({
    separationM: 5,
    massFractions: [0.5, 0.5],
  });
  assert.equal(disrupted.phase, "disrupted");
  for (const disk of disrupted.disks) {
    assert.ok(disk.outerRadiusM < disk.innerRadiusM);
    assert.equal(disk.widthM, 0);
    assert.equal(disk.stable, false);
  }
});

test("reciprocal mass ratios make body swapping exactly symmetric", () => {
  const first = createBinaryAccretionState({
    separationM: 24,
    massFractions: [0.8, 0.2],
  });
  const swapped = createBinaryAccretionState({
    separationM: 24,
    massFractions: [0.2, 0.8],
  });

  close(first.disks[0].massRatio, 1 / first.disks[1].massRatio);
  for (const field of [
    "massFraction",
    "massRatio",
    "innerRadiusM",
    "rocheLobeRadiusM",
    "outerRadiusM",
    "widthM",
  ]) {
    close(first.disks[0][field], swapped.disks[1][field]);
    close(first.disks[1][field], swapped.disks[0][field]);
  }
  assert.equal(first.disks[0].stable, swapped.disks[1].stable);
  assert.equal(first.disks[1].stable, swapped.disks[0].stable);
});

test("peak temperature follows the fourth root of accretion rate over mass", () => {
  const reference = thinDiskPeakTemperatureK({
    totalMassSolar: 1e8,
    massFraction: 0.5,
    accretionRatio: 0.01,
  });
  const changed = thinDiskPeakTemperatureK({
    totalMassSolar: 4e8,
    massFraction: 0.5,
    accretionRatio: 0.08,
  });
  close(changed / reference, Math.pow(2, 0.25));
});

test("visible blackbody response follows temperature without assigning false colours", () => {
  const warm = visibleBlackbodyLinearSrgbPerBolometric(3000);
  const neutral = visibleBlackbodyLinearSrgbPerBolometric(6500);
  const hot = visibleBlackbodyLinearSrgbPerBolometric(22500);
  const hotter = visibleBlackbodyLinearSrgbPerBolometric(50000);

  for (const response of [warm, neutral, hot, hotter]) {
    assert.equal(response.length, 3);
    assert.ok(response.every((channel) => Number.isFinite(channel) && channel >= 0));
  }
  for (const channel of neutral) {
    close(channel, 1, 2e-7);
  }
  assert.ok(warm[0] > warm[1] && warm[1] > warm[2]);
  assert.ok(hot[2] > hot[1] && hot[1] > hot[0]);
  assert.ok(hot[2] / hot[0] > neutral[2] / neutral[0]);
  assert.ok(hotter[2] / hotter[0] > hot[2] / hot[0]);

  const luminance = ([red, green, blue]) => (
    0.2126 * red + 0.7152 * green + 0.0722 * blue
  );
  assert.ok(luminance(hot) < luminance(neutral));
  assert.ok(luminance(hotter) < luminance(hot));
});

test("15-sample response agrees with the independent CIE 1931 1 nm corpus", () => {
  // Frozen from the official CIE 1931 2-degree 1 nm CSV (380-780 nm),
  // SHA-256 fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1
  // (MD5 17cca777db64b17170f06f67ce9d3ab7). Each reference integrates
  // Planck B_lambda through XYZ, converts to linear sRGB, divides by T^4,
  // then normalizes each channel to the independently integrated 6500 K value.
  // https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer
  const officialOneNanometreGoldens = new Map([
    [3000, [0.367920286308, 0.186046536806, 0.057001949142]],
    [6500, [1, 1, 1]],
    [10000, [0.643734146709, 0.779076407371, 1.064747566743]],
    [22500, [0.122993723286, 0.175031007950, 0.318075886332]],
    [50000, [0.015216099557, 0.022883788078, 0.046154004184]],
  ]);
  for (const [temperatureK, expected] of officialOneNanometreGoldens) {
    const actual = visibleBlackbodyLinearSrgbPerBolometric(temperatureK);
    for (let channel = 0; channel < 3; channel += 1) {
      const relativeError = Math.abs(actual[channel] - expected[channel])
        / expected[channel];
      assert.ok(
        relativeError < 0.01,
        `${temperatureK} K channel ${channel} relative error ${relativeError}`,
      );
    }
  }
});

test("analytic disk surface structure is unit-mean and strictly low amplitude", () => {
  const sampleCount = 4096;
  for (const timeM of [-9210.155252, -1000, 0, 120]) {
    for (const radiusM of [3.01, 49 * 3 / 36, 5.64]) {
      let minimum = Infinity;
      let maximum = -Infinity;
      let sum = 0;
      for (let sample = 0; sample < sampleCount; sample += 1) {
        const value = analyticDiskSurfaceStructureWeight({
          azimuth: 2 * Math.PI * sample / sampleCount,
          radiusM,
          innerRadiusM: 3,
          bodyMassM: 0.5,
          timeM,
        });
        minimum = Math.min(minimum, value);
        maximum = Math.max(maximum, value);
        sum += value;
      }
      assert.ok(minimum >= 0.75 - 1e-12, `minimum ${minimum}`);
      assert.ok(maximum <= 1.25 + 1e-12, `maximum ${maximum}`);
      close(sum / sampleCount, 1, 2e-14);
    }
  }
});

test("wrapped Kepler-calibrated emissivity phases remain continuous", () => {
  const innerRadiusM = 3;
  const bodyMassM = 0.5;
  const referenceRadiusM = 49 * innerRadiusM / 36;
  const omegaPeak = Math.sqrt(bodyMassM / referenceRadiusM ** 3);
  const firstPhaseWrapM = 2 * Math.PI / (0.82 * omegaPeak);
  const epsilonM = 1e-7;
  const before = analyticDiskSurfaceStructureWeight({
    azimuth: 0.73,
    radiusM: 4.6,
    innerRadiusM,
    bodyMassM,
    timeM: firstPhaseWrapM - epsilonM,
  });
  const after = analyticDiskSurfaceStructureWeight({
    azimuth: 0.73,
    radiusM: 4.6,
    innerRadiusM,
    bodyMassM,
    timeM: firstPhaseWrapM + epsilonM,
  });
  assert.ok(Math.abs(after - before) < 1e-7);
});

test("zero-torque flux is non-negative and dark at or within the inner edge", () => {
  const innerRadiusM = 3;
  for (const radiusM of [-1, 0, 1.5, innerRadiusM]) {
    assert.equal(
      zeroTorqueThinDiskFluxShape(radiusM, innerRadiusM),
      0,
    );
    assert.equal(
      zeroTorqueThinDiskTemperatureShape(radiusM, innerRadiusM),
      0,
    );
  }
  for (const radiusM of [3.01, 4, 6, 12, 30, 300, 3e6]) {
    const flux = zeroTorqueThinDiskFluxShape(radiusM, innerRadiusM);
    assert.ok(flux >= 0, `flux must be non-negative at ${radiusM} M`);
    close(
      zeroTorqueThinDiskTemperatureShape(radiusM, innerRadiusM),
      flux ** 0.25,
    );
  }
});

test("zero-torque flux has its analytic maximum at 49 r_in / 36", () => {
  const innerRadiusM = 4;
  const peakRadiusM = 49 * innerRadiusM / 36;
  const peakFlux = zeroTorqueThinDiskFluxShape(
    peakRadiusM,
    innerRadiusM,
  );
  const expectedPeakFlux = (36 / 49) ** 3 * (1 - 6 / 7);
  close(peakFlux, expectedPeakFlux);
  assert.ok(
    peakFlux > zeroTorqueThinDiskFluxShape(
      peakRadiusM * 0.999,
      innerRadiusM,
    ),
  );
  assert.ok(
    peakFlux > zeroTorqueThinDiskFluxShape(
      peakRadiusM * 1.001,
      innerRadiusM,
    ),
  );
});

test("far-field disk temperature approaches the r^-3/4 law", () => {
  const innerRadiusM = 3;
  const firstRadiusM = innerRadiusM * 1e8;
  const radiusRatio = 16;
  const first = zeroTorqueThinDiskTemperatureShape(
    firstRadiusM,
    innerRadiusM,
  );
  const second = zeroTorqueThinDiskTemperatureShape(
    firstRadiusM * radiusRatio,
    innerRadiusM,
  );
  close(
    second / first,
    radiusRatio ** (-0.75),
    3e-6,
  );
});

test("stable annulus fade has exact endpoints and C2-smooth joins", () => {
  const massFraction = 0.4;
  const fadeWidthPerBodyM = 0.75;
  const transitionWidthM = massFraction * fadeWidthPerBodyM;
  const weightAt = (normalizedWidth) => stableAnnulusWeight(
    normalizedWidth * transitionWidthM,
    massFraction,
    fadeWidthPerBodyM,
  );

  assert.equal(weightAt(0), 0);
  assert.equal(weightAt(1), 1);
  assert.equal(weightAt(2), 1);
  close(weightAt(0.5), 0.5);

  const step = 1e-4;
  const firstDerivativeAtZero = (weightAt(step) - weightAt(0)) / step;
  const secondDerivativeAtZero = (
    weightAt(2 * step) - 2 * weightAt(step) + weightAt(0)
  ) / step ** 2;
  const firstDerivativeAtOne = (
    weightAt(1) - weightAt(1 - step)
  ) / step;
  const secondDerivativeAtOne = (
    weightAt(1) - 2 * weightAt(1 - step) + weightAt(1 - 2 * step)
  ) / step ** 2;
  assert.ok(Math.abs(firstDerivativeAtZero) < 1e-5);
  assert.ok(Math.abs(firstDerivativeAtOne) < 1e-5);
  assert.ok(Math.abs(secondDerivativeAtZero) < 0.01);
  assert.ok(Math.abs(secondDerivativeAtOne) < 0.01);
});

test("photospheric annulus coverage is bounded and C2 at both edges", () => {
  const options = Object.freeze({
    innerRadiusM: 3,
    outerRadiusM: 5.64,
    bodyMassM: 0.5,
  });
  const coverageAt = (radiusM) => annulusEdgeCoverage({
    ...options,
    radiusM,
  });
  assert.equal(coverageAt(2.9), 0);
  assert.equal(coverageAt(options.innerRadiusM), 0);
  assert.equal(coverageAt(options.outerRadiusM), 0);
  assert.equal(coverageAt(5.8), 0);
  assert.equal(coverageAt((options.innerRadiusM + options.outerRadiusM) / 2), 1);
  for (let index = 0; index <= 100; index += 1) {
    const radius = options.innerRadiusM
      + index * (options.outerRadiusM - options.innerRadiusM) / 100;
    const coverage = coverageAt(radius);
    assert.ok(coverage >= 0 && coverage <= 1);
  }

  const step = 1e-6;
  for (const [edge, direction] of [
    [options.innerRadiusM, 1],
    [options.outerRadiusM, -1],
  ]) {
    const f0 = coverageAt(edge);
    const f1 = coverageAt(edge + direction * step);
    const f2 = coverageAt(edge + direction * 2 * step);
    assert.ok(Math.abs((f1 - f0) / step) < 1e-6);
    assert.ok(Math.abs((f2 - 2 * f1 + f0) / step ** 2) < 0.01);
  }
});

test("thin-disk shapes and annulus fade reject invalid inputs", () => {
  for (const value of [Number.NaN, Number.POSITIVE_INFINITY, "4"]) {
    assert.throws(
      () => zeroTorqueThinDiskFluxShape(value, 3),
      /finite number/,
    );
    assert.throws(
      () => zeroTorqueThinDiskTemperatureShape(value, 3),
      /finite number/,
    );
  }
  for (const innerRadiusM of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.throws(
      () => zeroTorqueThinDiskFluxShape(4, innerRadiusM),
      /finite positive/,
    );
  }
  for (const args of [
    [-1, 0.5],
    [Number.NaN, 0.5],
    [1, 0],
    [1, Number.POSITIVE_INFINITY],
    [1, 0.5, 0],
    [1, 0.5, Number.NaN],
  ]) {
    assert.throws(() => stableAnnulusWeight(...args));
  }
  for (const options of [
    undefined,
    { radiusM: 4, innerRadiusM: 0, outerRadiusM: 5, bodyMassM: 0.5 },
    { radiusM: 4, innerRadiusM: 3, outerRadiusM: 3, bodyMassM: 0.5 },
    { radiusM: 4, innerRadiusM: 3, outerRadiusM: 5, bodyMassM: 0 },
    { radiusM: Number.NaN, innerRadiusM: 3, outerRadiusM: 5, bodyMassM: 0.5 },
  ]) {
    assert.throws(() => annulusEdgeCoverage(options));
  }
});

test("model rejects malformed and non-positive physical inputs", () => {
  for (const q of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, "1"]) {
    assert.throws(() => eggletonRocheLobeFraction(q), /finite positive/);
  }

  for (const options of [
    undefined,
    { separationM: 0, massFractions: [0.5, 0.5] },
    { separationM: Number.NaN, massFractions: [0.5, 0.5] },
    { separationM: 30, massFractions: [1] },
    { separationM: 30, massFractions: [0.5, 0.5, 0] },
    { separationM: 30, massFractions: [0, 1] },
    { separationM: 30, massFractions: [0.6, 0.6] },
    { separationM: 30, massFractions: [0.5, Number.POSITIVE_INFINITY] },
    { separationM: 30, massFractions: [0.5, 0.5], maximumOuterRadiusM: -1 },
  ]) {
    assert.throws(() => createBinaryAccretionState(options));
  }

  for (const options of [
    undefined,
    { totalMassSolar: 0, massFraction: 0.5, accretionRatio: 0.1 },
    { totalMassSolar: 1e8, massFraction: 0, accretionRatio: 0.1 },
    { totalMassSolar: 1e8, massFraction: 0.5, accretionRatio: 0 },
    { totalMassSolar: 1e8, massFraction: 0.5, accretionRatio: Number.NaN },
  ]) {
    assert.throws(() => thinDiskPeakTemperatureK(options), /finite positive/);
  }

  for (const temperature of [
    0,
    -1,
    Number.NaN,
    Number.POSITIVE_INFINITY,
    "6500",
  ]) {
    assert.throws(
      () => visibleBlackbodyLinearSrgbPerBolometric(temperature),
      /finite positive/,
    );
  }
});
