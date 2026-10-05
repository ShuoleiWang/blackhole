// Shader tables of a homologous radiation-diffusion solution (see
// transient-wgsl.js, transientEjectaWGSL). For every sample time of the
// solution the solver grid is resampled onto `bins` velocity bins centred at
// (bin + 0.5) / bins * vmax:
//   thermal   = (log10 T_rad, d/dlog10 t, log10(kappa rho t^3), d/dlog10 t)
//   radiative = (log10 tau_out, d/dlog10 t, log10(j_dep t^3), d/dlog10 t)
// with tau_out the radial optical depth from the surface and j_dep = rho q /
// (4 pi) the emissivity of locally deposited power. The time derivatives let
// the shader evaluate each sample at its own retarded time.

import { STEFAN_BOLTZMANN_CGS } from "./physical-constants.js";

function clamp(value, low, high) {
  return Math.min(Math.max(value, low), high);
}

export function tabulateDiffusionProfiles({
  diffusion,
  velocityEdgesKmS,
  densityFactor,
  opacity,
  heating,
  sampleTimesS,
  bins,
  maximumVelocityKmS,
}) {
  const cells = densityFactor.length;
  const edges = velocityEdgesKmS;
  const cellWidthCm = (i) => (edges[i + 1] - edges[i]) * 1e5;
  const shellMass = Float64Array.from(densityFactor, (f, i) => (
    f * (4 * Math.PI / 3) * ((edges[i + 1] * 1e5) ** 3 - (edges[i] * 1e5) ** 3)
  ));
  const tables = diffusion.temperatureK.map((temperatures, k) => {
    const time = sampleTimesS[k];
    // Radial optical depth from the surface to each cell centre at time t.
    const tauAtCentre = new Float64Array(cells);
    let tau = 0;
    for (let i = cells - 1; i >= 0; i -= 1) {
      const kr = opacity(i, temperatures[i], time) * densityFactor[i] / time ** 3;
      const half = 0.5 * kr * cellWidthCm(i) * time;
      tauAtCentre[i] = tau + half;
      tau += 2 * half;
    }
    // Deposited power escaping directly (thin regions) and the remaining
    // thermal flux, which alone sets the photospheric T_eff used for the
    // atmosphere cap.
    let nebularLuminosity = 0;
    for (let i = 0; i < cells; i += 1) {
      nebularLuminosity += shellMass[i] * heating(i, time) * Math.exp(-tauAtCentre[i]);
    }
    const logT = new Float64Array(bins);
    const logAbs = new Float64Array(bins);
    const logTau = new Float64Array(bins);
    const logEmissivity = new Float64Array(bins);
    for (let b = 0; b < bins; b += 1) {
      const velocity = (b + 0.5) / bins * maximumVelocityKmS;
      // Cell whose centre velocity brackets the bin centre.
      let low = 0;
      while (low < cells - 1 && 0.5 * (edges[low + 1] + edges[low + 2]) <= velocity) low += 1;
      const lowCentre = 0.5 * (edges[low] + edges[low + 1]);
      const high = Math.min(low + 1, cells - 1);
      const highCentre = 0.5 * (edges[high] + edges[high + 1]);
      const f = high === low ? 0 : clamp((velocity - lowCentre) / (highCentre - lowCentre), 0, 1);
      const lerpLog = (a, c) => (1 - f) * Math.log10(Math.max(a, 1e-300)) + f * Math.log10(Math.max(c, 1e-300));
      logT[b] = lerpLog(Math.max(temperatures[low], 10), Math.max(temperatures[high], 10));
      logAbs[b] = lerpLog(
        opacity(low, temperatures[low], time) * densityFactor[low],
        opacity(high, temperatures[high], time) * densityFactor[high],
      );
      logTau[b] = lerpLog(tauAtCentre[low], tauAtCentre[high]);
      // Emissivity of deposited power: j t^3 = f q / (4 pi).
      logEmissivity[b] = lerpLog(
        densityFactor[low] * heating(low, time) / (4 * Math.PI) + 1e-300,
        densityFactor[high] * heating(high, time) / (4 * Math.PI) + 1e-300,
      );
    }
    const photosphere = diffusion.photosphereVelocityKmS[k] * 1e5 * time;
    const thermalLuminosity = Math.max(diffusion.luminosityErgS[k] - nebularLuminosity, 0.05 * diffusion.luminosityErgS[k]);
    const thermalTemperature = photosphere > 0
      ? (thermalLuminosity / (4 * Math.PI * photosphere * photosphere * STEFAN_BOLTZMANN_CGS)) ** 0.25
      : 0;
    return { logT, logAbs, logTau, logEmissivity, nebularLuminosity, thermalTemperature };
  });
  return Object.freeze({
    bins,
    maximumVelocityKmS,
    sampleTimesS: Float64Array.from(sampleTimesS),
    log10SampleTimes: Float64Array.from(sampleTimesS, (time) => Math.log10(time)),
    tables,
    diffusion,
  });
}

// Interpolated tables at time t (clamped to the solved range), packed as the
// shader's per-bin vec4 pairs, plus the photosphere summary and the outermost
// velocity that still absorbs (tau_out > 0.02, i.e. dims the light behind it
// by more than 2%) or emits deposited power above 1e-5 of the brightest bin.
export function diffusionProfilesAt(profiles, timeS) {
  const { bins, maximumVelocityKmS: vmax, sampleTimesS, log10SampleTimes, tables, diffusion } = profiles;
  const count = sampleTimesS.length;
  const log10T = Math.log10(clamp(timeS, sampleTimesS[0], sampleTimesS[count - 1]));
  const position = (log10T - log10SampleTimes[0])
    / (log10SampleTimes[count - 1] - log10SampleTimes[0]) * (count - 1);
  const k = clamp(Math.floor(position), 0, count - 2);
  const f = clamp(position - k, 0, 1);
  const a = tables[k];
  const b = tables[k + 1];
  const dLog = log10SampleTimes[k + 1] - log10SampleTimes[k];
  const thermal = new Float32Array(bins * 4);
  const radiative = new Float32Array(bins * 4);
  for (let bin = 0; bin < bins; bin += 1) {
    thermal[4 * bin] = (1 - f) * a.logT[bin] + f * b.logT[bin];
    thermal[4 * bin + 1] = (b.logT[bin] - a.logT[bin]) / dLog;
    thermal[4 * bin + 2] = (1 - f) * a.logAbs[bin] + f * b.logAbs[bin];
    thermal[4 * bin + 3] = (b.logAbs[bin] - a.logAbs[bin]) / dLog;
    radiative[4 * bin] = (1 - f) * a.logTau[bin] + f * b.logTau[bin];
    radiative[4 * bin + 1] = (b.logTau[bin] - a.logTau[bin]) / dLog;
    radiative[4 * bin + 2] = (1 - f) * a.logEmissivity[bin] + f * b.logEmissivity[bin];
    radiative[4 * bin + 3] = (b.logEmissivity[bin] - a.logEmissivity[bin]) / dLog;
  }
  let maximumEmissivity = -Infinity;
  for (let bin = 0; bin < bins; bin += 1) {
    maximumEmissivity = Math.max(maximumEmissivity, radiative[4 * bin + 2]);
  }
  let activeBin = bins - 1;
  while (activeBin > 0
    && radiative[4 * activeBin] < Math.log10(0.02)
    && radiative[4 * activeBin + 2] < maximumEmissivity - 5) {
    activeBin -= 1;
  }
  const activeVelocityKmS = Math.min(vmax, (activeBin + 2) / bins * vmax);
  const lerp = (array) => (1 - f) * array[k] + f * array[k + 1];
  const luminosity = 10 ** ((1 - f) * Math.log10(Math.max(diffusion.luminosityErgS[k], 1))
    + f * Math.log10(Math.max(diffusion.luminosityErgS[k + 1], 1)));
  return {
    thermal,
    radiative,
    luminosity,
    photosphereVelocityKmS: lerp(diffusion.photosphereVelocityKmS),
    effectiveTemperatureK: (1 - f) * a.thermalTemperature + f * b.thermalTemperature,
    nebularLuminosity: (1 - f) * a.nebularLuminosity + f * b.nebularLuminosity,
    activeVelocityKmS,
    log10Time: log10T,
  };
}
