// Grey flux-limited radiation diffusion in spherically symmetric, homologously
// expanding, radiation-dominated ejecta. The 56Ni/56Co-powered Type Ia light
// curve comes from this solver, so the rendered photosphere, its temperature
// and the bolometric light curve are mutually consistent.
//
// A cell i spans velocities [v_i, v_{i+1}] (r = v t) and keeps its mass:
// rho(v, t) = f(v) / t^3. With E the radiation energy density,
//   d(E t^4)/dt = t^4 ( -div F + rho q ),
//   F = -(c lambda(R) / (kappa rho)) dE/dr,   R = |dE/dr| / (kappa rho E),
// where lambda is the Levermore-Pomraning (1981) limiter
// (2 + R) / (6 + 3R + R^2), so F -> -(c/(3 kappa rho)) dE/dr in the optically
// thick limit and |F| -> c E when free streaming. The outer boundary is a
// Marshak (vacuum) condition F = (c/2) E_surface. Each log-spaced step is
// backward Euler (tridiagonal) with the limiter and opacity lagged by one
// fixed-point iteration; positivity and the Katz integral
//   t E_int(t) = int_0^t t' (Q - L) dt'
// are preserved to the time-step error (see the tests).

import {
  RADIATION_CONSTANT_CGS,
  SPEED_OF_LIGHT_CM_S,
  STEFAN_BOLTZMANN_CGS,
} from "./physical-constants.js";

const KM_S_TO_CM_S = 1e5;

function limiter(ratio) {
  return (2 + ratio) / (6 + 3 * ratio + ratio * ratio);
}

function solveTridiagonal(lower, diagonal, upper, rhs, out) {
  const n = diagonal.length;
  const cPrime = new Float64Array(n);
  const dPrime = new Float64Array(n);
  cPrime[0] = upper[0] / diagonal[0];
  dPrime[0] = rhs[0] / diagonal[0];
  for (let i = 1; i < n; i += 1) {
    const denominator = diagonal[i] - lower[i] * cPrime[i - 1];
    cPrime[i] = i < n - 1 ? upper[i] / denominator : 0;
    dPrime[i] = (rhs[i] - lower[i] * dPrime[i - 1]) / denominator;
  }
  out[n - 1] = dPrime[n - 1];
  for (let i = n - 2; i >= 0; i -= 1) {
    out[i] = dPrime[i] - cPrime[i] * out[i + 1];
  }
  return out;
}

export function radiationTemperature(energyDensity) {
  return Math.sqrt(Math.sqrt(Math.max(energyDensity, 0) / RADIATION_CONSTANT_CGS));
}

/**
 * @param {object} options
 * @param {ArrayLike<number>} options.velocityEdgesKmS cell edges, N + 1 increasing values
 * @param {ArrayLike<number>} options.densityFactor f_i = rho t^3 per cell (g s^3 cm^-3)
 * @param {(cell: number, temperatureK: number, timeS: number) => number} options.opacity grey kappa, cm^2/g
 * @param {(cell: number, timeS: number) => number} options.heating thermalised heating, erg g^-1 s^-1
 * @param {(cell: number, timeS: number) => number} options.initialEnergyDensity E at the start time, erg/cm^3
 * @param {number} options.startTimeS
 * @param {number} options.endTimeS
 * @param {ArrayLike<number>} options.sampleTimesS increasing output times inside [start, end]
 * @param {number} [options.stepsPerDecade]
 */
export function solveHomologousDiffusion({
  velocityEdgesKmS,
  densityFactor,
  opacity,
  heating,
  initialEnergyDensity,
  startTimeS,
  endTimeS,
  sampleTimesS,
  stepsPerDecade = 240,
}) {
  const cells = densityFactor.length;
  if (velocityEdgesKmS.length !== cells + 1 || cells < 3) {
    throw new RangeError("Diffusion grid needs N >= 3 cells and N + 1 edges");
  }
  for (let i = 0; i < cells; i += 1) {
    if (!(velocityEdgesKmS[i + 1] > velocityEdgesKmS[i]) || !(densityFactor[i] >= 0)) {
      throw new RangeError("Diffusion grid edges must increase and densities be non-negative");
    }
  }
  if (!(startTimeS > 0) || !(endTimeS > startTimeS)) {
    throw new RangeError("Diffusion time range must be positive and increasing");
  }
  const samples = Array.from(sampleTimesS, Number);
  if (samples.some((time, index) => !(time >= startTimeS && time <= endTimeS)
    || (index > 0 && !(time > samples[index - 1])))) {
    throw new RangeError("Sample times must increase inside the solved time range");
  }

  const edges = Float64Array.from(velocityEdgesKmS, (v) => v * KM_S_TO_CM_S);
  const centres = new Float64Array(cells);
  const shellVolumeFactor = new Float64Array(cells);
  for (let i = 0; i < cells; i += 1) {
    const inner3 = edges[i] ** 3;
    const outer3 = edges[i + 1] ** 3;
    centres[i] = Math.cbrt(0.5 * (inner3 + outer3));
    shellVolumeFactor[i] = (4 * Math.PI / 3) * (outer3 - inner3);
  }
  const mass = Float64Array.from(densityFactor, (f, i) => f * shellVolumeFactor[i]);

  let energy = new Float64Array(cells);
  for (let i = 0; i < cells; i += 1) {
    energy[i] = Math.max(initialEnergyDensity(i, startTimeS), 0);
  }
  const lower = new Float64Array(cells);
  const diagonal = new Float64Array(cells);
  const upper = new Float64Array(cells);
  const rhs = new Float64Array(cells);
  const next = new Float64Array(cells);
  const kappaRho = new Float64Array(cells);

  const temperatureSamples = [];
  const luminositySamples = new Float64Array(samples.length);
  const photosphereVelocity = new Float64Array(samples.length);
  const effectiveTemperature = new Float64Array(samples.length);
  const depositionSamples = new Float64Array(samples.length);
  const internalEnergySamples = new Float64Array(samples.length);
  let katzLossIntegral = 0;
  let katzHeatingIntegral = 0;

  function outerCoefficient(time, lastEnergy, lastKappaRho) {
    // Marshak boundary through the half cell between the last centre and the
    // outer edge: F = E_last / (dr / D + 2 / c).
    const halfWidth = (edges[cells] - centres[cells - 1]) * time;
    const gradientRatio = 1 / Math.max(lastKappaRho * halfWidth, 1e-300);
    const diffusion = SPEED_OF_LIGHT_CM_S * limiter(gradientRatio) / Math.max(lastKappaRho, 1e-300);
    return 1 / (halfWidth / diffusion + 2 / SPEED_OF_LIGHT_CM_S);
  }

  function luminosityAt(time, state, kr) {
    const area = 4 * Math.PI * (edges[cells] * time) ** 2;
    return area * outerCoefficient(time, state[cells - 1], kr[cells - 1]) * state[cells - 1];
  }

  function updateKappaRho(time, state) {
    const time3 = time ** 3;
    for (let i = 0; i < cells; i += 1) {
      const rho = densityFactor[i] / time3;
      kappaRho[i] = Math.max(opacity(i, radiationTemperature(state[i]), time), 0) * rho;
    }
  }

  function totalHeating(time) {
    let total = 0;
    for (let i = 0; i < cells; i += 1) {
      total += mass[i] * heating(i, time);
    }
    return total;
  }

  function internalEnergy(time, state) {
    let total = 0;
    const time3 = time ** 3;
    for (let i = 0; i < cells; i += 1) {
      total += state[i] * shellVolumeFactor[i] * time3;
    }
    return total;
  }

  function recordSample(sampleIndex, time, state) {
    updateKappaRho(time, state);
    const temperatures = new Float32Array(cells);
    for (let i = 0; i < cells; i += 1) {
      temperatures[i] = radiationTemperature(state[i]);
    }
    temperatureSamples.push(temperatures);
    const luminosity = luminosityAt(time, state, kappaRho);
    luminositySamples[sampleIndex] = luminosity;
    depositionSamples[sampleIndex] = totalHeating(time);
    internalEnergySamples[sampleIndex] = internalEnergy(time, state);
    // Photosphere: radial optical depth 2/3 from outside.
    let tau = 0;
    let photosphere = 0;
    for (let i = cells - 1; i >= 0; i -= 1) {
      const width = (edges[i + 1] - edges[i]) * time;
      const cellTau = kappaRho[i] * width;
      if (tau + cellTau >= 2 / 3) {
        const fraction = (2 / 3 - tau) / Math.max(cellTau, 1e-300);
        photosphere = edges[i + 1] - fraction * (edges[i + 1] - edges[i]);
        break;
      }
      tau += cellTau;
    }
    photosphereVelocity[sampleIndex] = photosphere / KM_S_TO_CM_S;
    const radius = Math.max(photosphere * time, 1);
    effectiveTemperature[sampleIndex] = photosphere > 0
      ? Math.sqrt(Math.sqrt(luminosity / (4 * Math.PI * radius * radius * STEFAN_BOLTZMANN_CGS)))
      : 0;
  }

  const growth = 10 ** (1 / stepsPerDecade);
  let time = startTimeS;
  let sampleIndex = 0;
  while (sampleIndex < samples.length && samples[sampleIndex] <= startTimeS) {
    recordSample(sampleIndex, startTimeS, energy);
    sampleIndex += 1;
  }
  updateKappaRho(time, energy);
  let previousLuminosity = luminosityAt(time, energy, kappaRho);
  let previousHeating = totalHeating(time);

  while (time < endTimeS * (1 - 1e-12)) {
    let nextTime = Math.min(time * growth, endTimeS);
    // Land exactly on sample times.
    if (sampleIndex < samples.length && samples[sampleIndex] < nextTime) {
      nextTime = samples[sampleIndex];
    }
    const step = nextTime - time;
    const adiabatic = (time / nextTime) ** 4;
    next.set(energy);
    for (let iteration = 0; iteration < 2; iteration += 1) {
      updateKappaRho(nextTime, next);
      const time2 = nextTime * nextTime;
      const time3 = time2 * nextTime;
      for (let i = 0; i < cells; i += 1) {
        const volume = shellVolumeFactor[i] * time3;
        let alphaInner = 0;
        let alphaOuter = 0;
        if (i > 0) {
          const distance = (centres[i] - centres[i - 1]) * nextTime;
          const kr = 0.5 * (kappaRho[i] + kappaRho[i - 1]);
          const faceEnergy = Math.max(0.5 * (next[i] + next[i - 1]), 1e-300);
          const ratio = Math.abs(next[i] - next[i - 1]) / (distance * Math.max(kr, 1e-300) * faceEnergy);
          const diffusion = SPEED_OF_LIGHT_CM_S * limiter(ratio) / Math.max(kr, 1e-300);
          alphaInner = 4 * Math.PI * edges[i] ** 2 * time2 * diffusion / (distance * volume);
        }
        if (i < cells - 1) {
          const distance = (centres[i + 1] - centres[i]) * nextTime;
          const kr = 0.5 * (kappaRho[i] + kappaRho[i + 1]);
          const faceEnergy = Math.max(0.5 * (next[i] + next[i + 1]), 1e-300);
          const ratio = Math.abs(next[i + 1] - next[i]) / (distance * Math.max(kr, 1e-300) * faceEnergy);
          const diffusion = SPEED_OF_LIGHT_CM_S * limiter(ratio) / Math.max(kr, 1e-300);
          alphaOuter = 4 * Math.PI * edges[i + 1] ** 2 * time2 * diffusion / (distance * volume);
        } else {
          alphaOuter = 4 * Math.PI * edges[cells] ** 2 * time2
            * outerCoefficient(nextTime, next[i], kappaRho[i]) / volume;
        }
        lower[i] = -step * alphaInner;
        upper[i] = i < cells - 1 ? -step * alphaOuter : 0;
        diagonal[i] = 1 + step * (alphaInner + alphaOuter);
        rhs[i] = energy[i] * adiabatic
          + step * (densityFactor[i] / time3) * heating(i, nextTime);
      }
      solveTridiagonal(lower, diagonal, upper, rhs, next);
      for (let i = 0; i < cells; i += 1) {
        next[i] = Math.max(next[i], 0);
      }
    }
    energy.set(next);
    updateKappaRho(nextTime, energy);
    const luminosity = luminosityAt(nextTime, energy, kappaRho);
    const heatingNow = totalHeating(nextTime);
    // Trapezoidal Katz integrals for diagnostics.
    katzLossIntegral += 0.5 * step * (time * previousLuminosity + nextTime * luminosity);
    katzHeatingIntegral += 0.5 * step * (time * previousHeating + nextTime * heatingNow);
    previousLuminosity = luminosity;
    previousHeating = heatingNow;
    time = nextTime;
    while (sampleIndex < samples.length && Math.abs(samples[sampleIndex] - time) <= 1e-9 * time) {
      recordSample(sampleIndex, time, energy);
      sampleIndex += 1;
    }
  }
  while (sampleIndex < samples.length) {
    recordSample(sampleIndex, time, energy);
    sampleIndex += 1;
  }

  return Object.freeze({
    cells,
    velocityEdgesKmS: Float64Array.from(velocityEdgesKmS),
    cellMassG: mass,
    sampleTimesS: Float64Array.from(samples),
    temperatureK: temperatureSamples,
    luminosityErgS: luminositySamples,
    depositionErgS: depositionSamples,
    internalEnergyErg: internalEnergySamples,
    photosphereVelocityKmS: photosphereVelocity,
    effectiveTemperatureK: effectiveTemperature,
    katz: Object.freeze({
      startTimeS,
      endTimeS: time,
      initialTimeEnergy: startTimeS * internalEnergy(startTimeS, Float64Array.from(
        { length: cells },
        (_, i) => Math.max(initialEnergyDensity(i, startTimeS), 0),
      )),
      finalTimeEnergy: time * internalEnergy(time, energy),
      heatingIntegral: katzHeatingIntegral,
      lossIntegral: katzLossIntegral,
    }),
  });
}
