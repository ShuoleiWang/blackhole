import assert from "node:assert/strict";
import test from "node:test";

import {
  blackbodyColorTable,
  blackbodyLinearSrgbPerBolometric,
  blackbodyXyzPerBolometric,
} from "../src/transients/blackbody-color.js";
import { solveHomologousDiffusion } from "../src/transients/homologous-diffusion.js";
import { DAY_S, SOLAR_MASS_G } from "../src/transients/physical-constants.js";
import { createPresentationTimeline } from "../src/transients/presentation-timeline.js";

function chromaticity(temperature) {
  const [x, y, z] = blackbodyXyzPerBolometric(temperature);
  return [x / (x + y + z), y / (x + y + z)];
}

test("blackbody luminous efficacy and chromaticity match the CIE Planckian locus", () => {
  // 683 lm/W times the luminous fraction: ~93 lm/W for the Sun's 5778 K, a
  // maximum of ~95 lm/W near 6600 K.
  const efficacy = (temperature) => 683 * blackbodyXyzPerBolometric(temperature)[1];
  assert.ok(Math.abs(efficacy(5778) - 93) < 2);
  assert.ok(efficacy(6600) > efficacy(5000) && efficacy(6600) > efficacy(8000));
  // Planckian-locus chromaticities (CIE 1931) to the accuracy of the fit.
  for (const [temperature, x, y] of [
    [2500, 0.4770, 0.4137],
    [4000, 0.3805, 0.3768],
    [10000, 0.2807, 0.2884],
  ]) {
    const [cx, cy] = chromaticity(temperature);
    assert.ok(Math.abs(cx - x) < 0.004 && Math.abs(cy - y) < 0.004, `${temperature} K`);
  }
  // The infinite-temperature limit is the Rayleigh-Jeans blue-white.
  const [xInf, yInf] = chromaticity(1e9);
  assert.ok(Math.abs(xInf - 0.2399) < 0.004 && Math.abs(yInf - 0.2342) < 0.004);
});

test("6500 K is neutral and the GPU table stores colour and luminance losslessly", () => {
  const neutral = blackbodyLinearSrgbPerBolometric(6500);
  assert.ok(Math.abs(neutral[0] / neutral[1] - 1) < 1e-9);
  assert.ok(Math.abs(neutral[2] / neutral[1] - 1) < 1e-9);
  const table = blackbodyColorTable({ minLog10Temperature: 3, maxLog10Temperature: 8, entries: 101 });
  assert.equal(table.data.length, 404);
  for (let index = 0; index < table.entries; index += 1) {
    const temperature = 10 ** (3 + 5 * index / 100);
    const rgb = blackbodyLinearSrgbPerBolometric(temperature);
    const luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    assert.ok(Math.abs(table.data[4 * index + 3] - Math.log10(luminance)) < 1e-5);
    assert.ok(Math.abs(table.data[4 * index + 1] * luminance - rgb[1]) <= 1e-5 * Math.abs(rgb[1]) + 1e-30);
  }
  // Hotter is bluer.
  const blueRatio = (index) => table.data[4 * index + 2] / table.data[4 * index];
  assert.ok(blueRatio(100) > blueRatio(40) && blueRatio(40) > blueRatio(10));
});

function uniformSphere({ cells = 120, opacity = 0.1, nickelFraction = 0.6, stepsPerDecade = 240 } = {}) {
  const mass = SOLAR_MASS_G;
  const kineticEnergy = 1.2e51;
  const vmax = Math.sqrt(10 * kineticEnergy / (3 * mass)) / 1e5;
  const edges = Float64Array.from({ length: cells + 1 }, (_, i) => vmax * i / cells);
  const factor = mass / ((4 * Math.PI / 3) * (vmax * 1e5) ** 3);
  const density = new Float64Array(cells).fill(factor);
  const vNi = vmax * Math.cbrt(nickelFraction);
  const xNi = Array.from({ length: cells }, (_, i) => {
    if (edges[i + 1] <= vNi) return 1;
    if (edges[i] >= vNi) return 0;
    return (vNi ** 3 - edges[i] ** 3) / (edges[i + 1] ** 3 - edges[i] ** 3);
  });
  const heatingPerNickel = (t) => (3.9e10 - 6.78e9) * Math.exp(-t / (8.8 * DAY_S))
    + 6.78e9 * Math.exp(-t / (111.3 * DAY_S));
  const samples = Array.from({ length: 237 }, (_, k) => (1 + 0.25 * k) * DAY_S);
  const result = solveHomologousDiffusion({
    velocityEdgesKmS: edges,
    densityFactor: density,
    opacity: () => opacity,
    heating: (i, t) => xNi[i] * heatingPerNickel(t),
    initialEnergyDensity: () => 0,
    startTimeS: 100,
    endTimeS: 60 * DAY_S,
    sampleTimesS: samples,
    stepsPerDecade,
  });
  return { result, samples, heating: (t) => nickelFraction * mass * heatingPerNickel(t) };
}

test("homologous diffusion conserves the Katz integral and obeys Arnett's rule", () => {
  const { result, samples, heating } = uniformSphere();
  const katz = result.katz;
  const predicted = katz.initialTimeEnergy + katz.heatingIntegral - katz.lossIntegral;
  assert.ok(Math.abs(katz.finalTimeEnergy - predicted) < 2e-4 * katz.heatingIntegral);
  let peakIndex = 0;
  for (let k = 1; k < samples.length; k += 1) {
    if (result.luminosityErgS[k] > result.luminosityErgS[peakIndex]) peakIndex = k;
  }
  const ratio = result.luminosityErgS[peakIndex] / heating(samples[peakIndex]);
  // Arnett's rule holds to ~10-20% for centrally concentrated 56Ni.
  assert.ok(ratio > 0.8 && ratio < 1.1, `L_peak / Q(t_peak) = ${ratio}`);
  // Arnett's effective diffusion time for this sphere is 9.5 d; the peak
  // follows within ~1.2 tau_m.
  const peakDays = samples[peakIndex] / DAY_S;
  assert.ok(peakDays > 9 && peakDays < 13.5, `peak at ${peakDays} d`);
  // Late times: luminosity approaches the instantaneous deposition.
  const late = samples.length - 1;
  assert.ok(Math.abs(result.luminosityErgS[late] / heating(samples[late]) - 1) < 0.12);
  // Temperatures and the photosphere are positive and the photosphere recedes.
  assert.ok(result.temperatureK.every((row) => row.every((value) => value >= 0)));
  assert.ok(result.photosphereVelocityKmS[late] < result.photosphereVelocityKmS[20]);
});

test("homologous diffusion light curve is converged in time step and resolution", () => {
  const reference = uniformSphere({ cells: 160, stepsPerDecade: 480 }).result;
  const coarse = uniformSphere({ cells: 120, stepsPerDecade: 240 }).result;
  for (const k of [36, 76, 116, 196]) {
    const a = coarse.luminosityErgS[k];
    const b = reference.luminosityErgS[k];
    assert.ok(Math.abs(a / b - 1) < 0.02, `sample ${k}: ${a} vs ${b}`);
  }
});

test("pure radiative cooling of an initially hot envelope is monotone and bounded", () => {
  const cells = 64;
  const edges = Float64Array.from({ length: cells + 1 }, (_, i) => 20000 * i / cells);
  const density = new Float64Array(cells).fill(1e8);
  const samples = [10, 100, 1000, 1e4, 1e5];
  const result = solveHomologousDiffusion({
    velocityEdgesKmS: edges,
    densityFactor: density,
    opacity: () => 0.2,
    heating: () => 0,
    initialEnergyDensity: () => 1e16,
    startTimeS: 1,
    endTimeS: 1e5,
    sampleTimesS: samples,
  });
  for (let k = 1; k < samples.length; k += 1) {
    assert.ok(result.internalEnergyErg[k] < result.internalEnergyErg[k - 1]);
  }
  const katz = result.katz;
  assert.ok(Math.abs(katz.finalTimeEnergy - (katz.initialTimeEnergy - katz.lossIntegral))
    < 1e-3 * katz.initialTimeEnergy);
});


test("presentation timeline maps progress to time monotonically across linear and log segments", () => {
  const timeline = createPresentationTimeline({
    segments: [
      { kind: "linear", start: -120, end: 0, wallSeconds: 12 },
      { kind: "linear", start: 0, end: 4, wallSeconds: 8 },
      { kind: "log", start: 4, end: 1.3e7, wallSeconds: 20 },
    ],
    endHoldSeconds: 2,
    loop: true,
  });
  assert.equal(timeline.totalWallSeconds, 40);
  let previous = -Infinity;
  for (let k = 0; k <= 400; k += 1) {
    const time = timeline.timeAtProgress(k / 400);
    assert.ok(time >= previous);
    previous = time;
    assert.ok(Math.abs(timeline.progressAtTime(time) - k / 400) < 1e-9);
  }
  assert.equal(timeline.timeAtProgress(0.3), 0);
  assert.equal(timeline.timeAtProgress(0.5), 4);
  // Inside the log segment log10 t advances uniformly with wall time.
  const decades = Math.log10(1.3e7 / 4);
  const t1 = timeline.timeAtProgress(0.5 + 10 / 40);
  assert.ok(Math.abs(Math.log10(t1 / 4) - decades / 2) < 1e-9);
  // Physical rate = t ln(end/start) / wall in a log segment.
  assert.ok(Math.abs(timeline.rateAt(1000) - 1000 * Math.log(1.3e7 / 4) / 20) < 1e-6);
  assert.equal(timeline.rateAt(-60), 10);
});

test("presentation timeline advances, holds at the end and loops", () => {
  const timeline = createPresentationTimeline({
    segments: [
      { kind: "linear", start: 0, end: 10, wallSeconds: 10 },
    ],
    endHoldSeconds: 1.5,
    loop: true,
  });
  let step = timeline.advance(2, 1, 2);
  assert.equal(step.time, 4);
  assert.equal(step.holding, false);
  step = timeline.advance(9, 5, 1);
  assert.equal(step.time, 10);
  assert.equal(step.holding, true);
  step = timeline.advance(10, 1, 1, step.holdElapsed);
  assert.equal(step.holding, true);
  step = timeline.advance(10, 1, 1, step.holdElapsed);
  assert.equal(step.wrapped, true);
  assert.equal(step.time, 0);
  assert.throws(() => createPresentationTimeline({ segments: [{ kind: "log", start: 0, end: 1, wallSeconds: 1 }] }));
  assert.throws(() => createPresentationTimeline({
    segments: [
      { kind: "linear", start: 0, end: 1, wallSeconds: 1 },
      { kind: "linear", start: 2, end: 3, wallSeconds: 1 },
    ],
  }));
});
