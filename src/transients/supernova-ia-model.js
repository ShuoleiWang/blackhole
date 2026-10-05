// Physical model of the Type Ia supernova scene: a double-degenerate binary
// with dynamically unstable mass transfer, a helium-shell detonation that
// triggers a converging-shock carbon/oxygen core detonation (the "D6" double
// detonation), and the homologous 56Ni-powered ejecta. Every rendered
// quantity comes from this module; the shader only transports light.
//
// Calibration (see docs/transient-physics.md): explosion and ejecta after Boos
// et al. 2021 (d2e5_m100) and Shen et al. 2021; companion interaction after
// Tanikawa, Nomoto & Leung 2018; light curve and colours against the normal
// SN Ia template SN 2011fe (Pereira et al. 2013; Burns et al. 2014).
//
// Units: km, s, K, erg, solar masses. The binary orbits in the x-z plane with
// angular momentum along +y; the scene origin is the binary centre of mass.

import {
  DAY_S,
  GRAVITATIONAL_CONSTANT_CGS,
  SOLAR_LUMINOSITY_ERG_S,
  SOLAR_MASS_G,
  SPEED_OF_LIGHT_CM_S,
  SPEED_OF_LIGHT_KM_S,
  STEFAN_BOLTZMANN_CGS,
} from "./physical-constants.js";
import { blackbodyLinearSrgbPerBolometric } from "./blackbody-color.js";
import { diffusionProfilesAt, tabulateDiffusionProfiles } from "./diffusion-profiles.js";
import { solveHomologousDiffusion } from "./homologous-diffusion.js";
import { createPresentationTimeline } from "./presentation-timeline.js";
import {
  ballisticStream,
  eggletonRocheLobe,
  gravitationalInspiralTimeS,
  innerLagrangePoint,
  keplerAngularFrequency,
  rochePotential,
} from "./roche-binary.js";
import { solveWhiteDwarfStructure } from "./white-dwarf-structure.js";

// Mass fractions by velocity (km/s): 56Ni, stable Fe-group, intermediate-mass
// elements (Si/S/Ca/Ar), O/C/Mg/Ne, He (Boos et al. 2021 d2e5_m100; Shen et
// al. 2018/2021). Linear between rows.
const COMPOSITION_TABLE = Object.freeze([
  [0, 0.87, 0.10, 0.03, 0.00, 0.00],
  [8000, 0.85, 0.10, 0.05, 0.00, 0.00],
  [11000, 0.10, 0.02, 0.78, 0.10, 0.00],
  [12000, 0.00, 0.00, 0.90, 0.10, 0.00],
  [15000, 0.00, 0.00, 0.90, 0.10, 0.00],
  [16500, 0.00, 0.00, 0.40, 0.60, 0.00],
  [22000, 0.00, 0.00, 0.35, 0.65, 0.00],
  [23000, 0.00, 0.00, 0.20, 0.80, 0.00],
  [24000, 0.00, 0.00, 0.20, 0.80, 0.00],
  [24600, 0.00, 0.00, 0.25, 0.00, 0.75],
  [40000, 0.00, 0.00, 0.25, 0.00, 0.75],
]);

export const SUPERNOVA_IA_CANONICAL_MODEL = Object.freeze({
  scenario: "dynamically driven double-degenerate double detonation (D6)",
  primaryMassSolar: 1.0,
  heliumShellMassSolar: 0.016,
  donorMassSolar: 0.6,
  primaryTemperatureK: 20000,
  donorTemperatureK: 12000,
  limbDarkening: 0.45,
  gravityDarkeningExponent: 0.25,
  // Direct-impact accretion in the dynamical phase is super-Eddington by
  // orders of magnitude (G M Mdot / R >> L_Edd), so the emergent flux of the
  // impact region saturates at the local Eddington flux g c / kappa_es of the
  // He-rich shell (kappa_es = 0.2 cm^2/g): T ~ 1e6 K and ~4e36 erg/s over
  // the impact spot; the rest is advected into the shell (Guillochon et al.
  // 2010). The stream and the donor absorb and re-radiate half of the light
  // of the spot and the accretor that falls on them (Bond albedo 0.5).
  streamHalfWidthKm: 450,
  impactAngularRadiusDeg: 14,
  impactOpacity: 0.2,
  irradiationAbsorbedFraction: 0.5,
  // Helium detonation reaches the antipode after 2.0 s; the converging shock
  // ignites the C/O core at 0.4 R on the antipode side at 2.0 s; the core
  // detonation runs at 12,500 km/s (Boos et al. 2021; Gronow et al. 2021).
  heliumAntipodeTimeS: 2.0,
  coreIgnitionTimeS: 2.0,
  coreIgnitionRadiusFraction: 0.4,
  coreDetonationSpeedKmS: 12500,
  surfaceAccelerationTimeS: 0.1,
  // The core detonation drives the outermost layers to the ejecta edge: the
  // helium-ash plus core kick equals the stretched edge velocity
  // vmax s(theta) (below), so the fireball meets the homologous ejecta at the
  // hand-off in every direction.
  heliumAshSpeedKmS: 5000,
  // Breakout photospheres (Piro, Chang & Weinberg 2010 scalings). Behind the
  // helium front T = 1e7 K (dt / 0.01 s)^-1/2 (~5e40 erg over the sweep).
  // Each element of the core-detonation breakout flashes at 6e6 K for
  // ~0.01 s and fades as (dt / 0.01 s)^-1/3 e^(-dt / 0.1 s): ~1e21 erg/cm^2,
  // i.e. the ~1e40 erg radiation content of the breakout shell over the
  // whole surface. The fireball then follows T_eff = 1e6 K (t / 1 s)^-0.46
  // and, from 10 s, blends in log t onto the diffusion solution's
  // photospheric temperature at the hand-off (homologousStartS), so radius,
  // temperature and luminosity are continuous there.
  heliumFlashTemperatureK: 1e7,
  coreFlashTemperatureK: 6e6,
  flashDecayS: 0.1,
  fireballTemperatureK: 1e6,
  fireballCoolingIndex: 0.46,
  fireballMatchStartS: 10,
  // The fastest core ash leaving the primary's facing surface strikes the
  // donor after crossing the gap a - R1 - R2 (~3.3 s). Its facing hemisphere
  // flashes like a breakout, T = 3e6 K (dt / 0.01 s)^-1/2, and settles at the
  // shock-heated ~1e5 K until the fireball engulfs it (Tanikawa, Nomoto &
  // Leung 2018; Bauer, White & Bildsten 2019).
  donorShockTemperatureK: 3e6,
  donorPostShockTemperatureK: 1e5,
  // Homologous ejecta: exponential density with v_e = sqrt(E_k / 6 M).
  ejectaMassSolar: 1.0,
  kineticEnergyErg: 1.15e51,
  maximumVelocityKmS: 32000,
  // Effective grey optical opacities (cm^2/g). Line-expansion opacity rises
  // with ionisation and collapses on recombination (Pinto & Eastman 2000):
  // Fe-group 0.18 above 1e4 K, 0.05 at 7500 K, 0.02 below 5000 K; Si/S/Ca
  // 0.04 below 8500 K rising to 0.4 above 1.1e4 K (O 0.03 -> 0.3); He ash
  // 0.03. Calibrated so the diffusion light curve reproduces the normal SN Ia
  // template (bolometric peak 1.1e43 erg/s at 17 d, v_ph 9500 km/s at peak,
  // L(5 d) 7e41, L(33 d) 4.5e42 erg/s; Pereira et al. 2013).
  opacityIronGroupHot: 0.18,
  opacityIronGroupMid: 0.05,
  opacityIronGroupCool: 0.02,
  opacityIntermediate: 0.04,
  opacityOxygen: 0.03,
  opacityIntermediateHot: 0.4,
  intermediateHotLowK: 8500,
  intermediateHotHighK: 11000,
  opacityHelium: 0.03,
  recombinationLowK: 5000,
  recombinationHighK: 7500,
  ionisationHotK: 10000,
  gammaRayOpacity: 0.025,
  positronFraction: 0.032,
  // Homology and the diffusion start (s after the core detonation).
  homologousStartS: 100,
  initialThermalEnergyErg: 4e48,
  finalTimeS: 150 * DAY_S,
  // Companion shadow (Tanikawa et al. 2018; Kasen 2010).
  shadowHalfAngleDeg: 44,
  shadowInteriorDensity: 0.2,
  shadowRimDensity: 3,
  shadowRimWidthDeg: 5,
  // Ejecta faster by 20% towards the impact pole, 15% slower at the antipode.
  impactPoleStretch: 0.2,
  antipodeStretch: -0.15,
  bMaximumDays: 18,
});

const NICKEL_LIFETIME_S = 8.764 * DAY_S;
const COBALT_LIFETIME_S = 111.4 * DAY_S;
const NICKEL_HEATING = 3.9e10; // erg g^-1 s^-1 (Nadyozhin 1994)
const COBALT_HEATING = 6.8e9;
const GRID_CELLS = 192;
const PROFILE_BINS = 64;
const TABLE_TIMES = 288;
const GAMMA_DIRECTIONS = 8;

function clamp(value, low, high) {
  return Math.min(Math.max(value, low), high);
}

function smoothstep(edge0, edge1, x) {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function exponentialEnclosedFraction(x) {
  return 1 - Math.exp(-x) * (1 + x + 0.5 * x * x);
}

function compositionAt(velocityKmS) {
  const rows = COMPOSITION_TABLE;
  if (velocityKmS <= rows[0][0]) return rows[0].slice(1);
  for (let i = 1; i < rows.length; i += 1) {
    if (velocityKmS <= rows[i][0]) {
      const f = (velocityKmS - rows[i - 1][0]) / (rows[i][0] - rows[i - 1][0]);
      return rows[i].slice(1).map((value, k) => rows[i - 1][k + 1] + f * (value - rows[i - 1][k + 1]));
    }
  }
  return rows[rows.length - 1].slice(1);
}

// Heating per gram of initial 56Ni (Bateman solution; Nadyozhin 1994).
export function nickelCobaltHeating(timeS) {
  return NICKEL_HEATING * Math.exp(-timeS / NICKEL_LIFETIME_S)
    + COBALT_HEATING * COBALT_LIFETIME_S / (COBALT_LIFETIME_S - NICKEL_LIFETIME_S)
      * (Math.exp(-timeS / COBALT_LIFETIME_S) - Math.exp(-timeS / NICKEL_LIFETIME_S));
}

function cobaltPower(timeS) {
  return COBALT_HEATING * COBALT_LIFETIME_S / (COBALT_LIFETIME_S - NICKEL_LIFETIME_S)
    * (Math.exp(-timeS / COBALT_LIFETIME_S) - Math.exp(-timeS / NICKEL_LIFETIME_S));
}

// Colour temperature from B-V (Ballesteros 2012, EPL 97, 34008).
export function colourTemperatureFromBV(bv) {
  return 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62));
}

// Intrinsic B-V of a normal SN Ia versus days after explosion: the observed
// track before and after B maximum (Burns et al. 2014; Stritzinger et al.
// 2018; first light B-V ~ +0.45) and the Lira law 0.725 - 0.0118 (t_V - 60)
// for 30-90 d after V maximum (Phillips et al. 1999).
export function supernovaIaBV(daysSinceExplosion, bMaximumDays = 18) {
  const phase = daysSinceExplosion - bMaximumDays;
  const track = [
    [-17.5, 0.45], [-16, 0.3], [-12, 0.1], [-10, 0.0], [-5, -0.1], [0, -0.03],
    [5, 0.15], [10, 0.45], [15, 0.8], [20, 1.0], [27.5, 1.1], [31.5, 1.08],
  ];
  if (phase <= track[0][0]) return track[0][1];
  for (let i = 1; i < track.length; i += 1) {
    if (phase <= track[i][0]) {
      const f = (phase - track[i - 1][0]) / (track[i][0] - track[i - 1][0]);
      return track[i - 1][1] + f * (track[i][1] - track[i - 1][1]);
    }
  }
  const tV = clamp(phase - 1.5, 30, 90);
  return 0.725 - 0.0118 * (tV - 60);
}

function luminousFraction(temperature) {
  const rgb = blackbodyLinearSrgbPerBolometric(temperature);
  return Math.max(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2], 1e-300);
}

// Photometric luminance (cd/m^2) of a blackbody surface.
function surfaceLuminance(temperature) {
  return 0.683 * STEFAN_BOLTZMANN_CGS * temperature ** 4 / Math.PI * luminousFraction(temperature);
}

export function createSupernovaIaModel(config = SUPERNOVA_IA_CANONICAL_MODEL) {
  // ---- Binary ---------------------------------------------------------------
  const primaryStructure = solveWhiteDwarfStructure(config.primaryMassSolar);
  const donorStructure = solveWhiteDwarfStructure(config.donorMassSolar);
  const primaryRadiusKm = primaryStructure.radiusKm;
  const donorRadiusKm = donorStructure.radiusKm;
  const massRatio = config.donorMassSolar / config.primaryMassSolar;
  const totalMass = config.primaryMassSolar + config.donorMassSolar;
  const mu = config.donorMassSolar / totalMass;
  const separationKm = donorRadiusKm / eggletonRocheLobe(massRatio);
  const omega = keplerAngularFrequency(totalMass, separationKm);
  const periodS = 2 * Math.PI / omega;
  const orbitalSpeedKmS = omega * separationKm;
  const primarySpeedKmS = orbitalSpeedKmS * mu;
  const donorSpeedKmS = orbitalSpeedKmS * (1 - mu);
  const lagrangeX = innerLagrangePoint(mu);
  const lagrangePotential = rochePotential(lagrangeX, 0, 0, mu);
  const stream = ballisticStream({ mu, accretorRadius: primaryRadiusKm / separationKm, soundSpeed: 0.02, points: 16 });
  const streamHalfWidth = config.streamHalfWidthKm / separationKm;
  const impact = stream.points[stream.points.length - 1];
  const impactDirection = (() => {
    const v = [impact[0] + mu, impact[1], impact[2]];
    const n = Math.hypot(...v);
    return v.map((c) => c / n);
  })();
  // Angle of the impact point ahead of the line of centres (for the docs).
  const impactLeadDeg = Math.atan2(-impactDirection[2], impactDirection[0]) * 180 / Math.PI;
  const poleGravity = (() => {
    const y = donorRadiusKm / separationKm * 0.97;
    const d1 = [1, y, 0];
    const d2 = [0, y, 0];
    const r1 = Math.hypot(...d1);
    const r2 = Math.hypot(...d2);
    const gx = (1 - mu) * d1[0] / r1 ** 3 + mu * d2[0] / r2 ** 3 - (1 - mu);
    const gy = (1 - mu) * d1[1] / r1 ** 3 + mu * d2[1] / r2 ** 3;
    return Math.hypot(gx, gy);
  })();
  const donorBoundingRadiusKm = (1 - mu - lagrangeX + 0.02) * separationKm * 1.05;
  // Eddington-limited impact spot (see the config).
  const primarySurfaceGravity = GRAVITATIONAL_CONSTANT_CGS * config.primaryMassSolar * SOLAR_MASS_G
    / (primaryRadiusKm * 1e5) ** 2;
  const impactFlux = primarySurfaceGravity * SPEED_OF_LIGHT_CM_S / config.impactOpacity;
  const impactTemperatureK = (impactFlux / STEFAN_BOLTZMANN_CGS) ** 0.25;
  const impactLuminosity = impactFlux * 2 * Math.PI * (primaryRadiusKm * 1e5) ** 2
    * (1 - Math.cos(config.impactAngularRadiusDeg * Math.PI / 180));
  const primaryLuminosity = 4 * Math.PI * (primaryRadiusKm * 1e5) ** 2 * STEFAN_BOLTZMANN_CGS
    * config.primaryTemperatureK ** 4;
  // Irradiation is reprocessed light, not an additional source.
  const quiescentLuminosity = primaryLuminosity
    + 4 * Math.PI * (donorRadiusKm * 1e5) ** 2 * STEFAN_BOLTZMANN_CGS * config.donorTemperatureK ** 4
    + impactLuminosity;

  // ---- Detonations ----------------------------------------------------------
  // Ejecta faster towards the helium-ignition (impact) pole: v' = s v with
  // s(theta) = 1 + a cos(theta) + b (theta from the ignition direction).
  const stretchA = 0.5 * (config.impactPoleStretch - config.antipodeStretch);
  const stretchB = 0.5 * (config.impactPoleStretch + config.antipodeStretch);
  // Core kick of the surface layer at polar cosine c from the ignition point.
  const coreKickAt = (cosTheta) => Math.max(
    config.maximumVelocityKmS * (1 + stretchA * cosTheta + stretchB) - config.heliumAshSpeedKmS,
    0,
  );
  const heliumAngularSpeed = Math.PI / config.heliumAntipodeTimeS;
  const coreIgnitionTimeS = config.coreIgnitionTimeS;
  // Ejecta clock: half-way through the core burn (burning ends by ~3 s).
  const explosionTimeS = coreIgnitionTimeS
    + 0.5 * (1 + config.coreIgnitionRadiusFraction) * primaryRadiusKm / config.coreDetonationSpeedKmS;

  // ---- Ejecta ---------------------------------------------------------------
  const ejectaMassG = config.ejectaMassSolar * SOLAR_MASS_G;
  const ve = Math.sqrt(config.kineticEnergyErg / (6 * ejectaMassG)) / 1e5;
  const vmax = config.maximumVelocityKmS;
  const edges = Float64Array.from({ length: GRID_CELLS + 1 }, (_, i) => vmax * i / GRID_CELLS);
  const density = new Float64Array(GRID_CELLS);
  const composition = [];
  let nickelMassG = 0;
  for (let i = 0; i < GRID_CELLS; i += 1) {
    const x0 = edges[i] / ve;
    const x1 = edges[i + 1] / ve;
    // The exponential tail beyond vmax (0.2% of M_ej) is left out: folding it
    // into the last cell would make an artificial dense, hot edge shell.
    const shellMass = (exponentialEnclosedFraction(x1) - exponentialEnclosedFraction(x0)) * ejectaMassG;
    const volume = (4 * Math.PI / 3) * ((edges[i + 1] * 1e5) ** 3 - (edges[i] * 1e5) ** 3);
    density[i] = shellMass / volume;
    const [ni, stable, ime, oxygen, helium] = compositionAt(0.5 * (edges[i] + edges[i + 1]));
    composition.push({ nickel: ni, ironGroup: ni + stable, intermediate: ime, oxygen, helium });
    nickelMassG += ni * shellMass;
  }

  function opacity(cell, temperatureK) {
    const c = composition[cell];
    // Fe-group line opacity rises with ionisation (Fe II -> Fe III/IV,
    // Pinto & Eastman 2000) and collapses when Fe recombines below ~5000 K.
    const ironGroup = config.opacityIronGroupCool
      + (config.opacityIronGroupMid - config.opacityIronGroupCool)
        * smoothstep(config.recombinationLowK, config.recombinationHighK, temperatureK)
      + (config.opacityIronGroupHot - config.opacityIronGroupMid)
        * smoothstep(config.recombinationHighK, config.ionisationHotK, temperatureK);
    const hot = smoothstep(config.intermediateHotLowK, config.intermediateHotHighK, temperatureK);
    const intermediate = config.opacityIntermediate + (config.opacityIntermediateHot - config.opacityIntermediate) * hot;
    const oxygen = config.opacityOxygen + (config.opacityIntermediateHot - config.opacityIntermediate) * 0.75 * hot;
    const line = c.ironGroup * ironGroup
      + c.intermediate * intermediate
      + c.oxygen * oxygen
      + c.helium * config.opacityHelium;
    // Fully ionised Z/A = 1/2 matter: electron scattering 0.2 cm^2/g.
    const electron = 0.2 * smoothstep(2e4, 8e4, temperatureK);
    return Math.max(line, electron);
  }

  // Gamma-ray escape probability per cell from column masses along Gauss-
  // Legendre directions (scale as t^-2 in homologous flow).
  const gaussNodes = [-0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425, 0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565];
  const gaussWeights = [0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834, 0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363];
  const gammaColumn = [];
  const outerCm = edges[GRID_CELLS] * 1e5;
  for (let i = 0; i < GRID_CELLS; i += 1) {
    const r = 0.5 * (edges[i] + edges[i + 1]) * 1e5;
    const columns = [];
    for (let k = 0; k < GAMMA_DIRECTIONS; k += 1) {
      const muDir = gaussNodes[k];
      const pathLength = -r * muDir + Math.sqrt(outerCm * outerCm - r * r * (1 - muDir * muDir));
      let column = 0;
      const steps = 64;
      for (let j = 0; j < steps; j += 1) {
        const s = (j + 0.5) / steps * pathLength;
        const radius = Math.sqrt(r * r + s * s + 2 * r * s * muDir);
        const cell = Math.min(GRID_CELLS - 1, Math.floor(radius / outerCm * GRID_CELLS));
        column += density[cell] * pathLength / steps;
      }
      columns.push(column);
    }
    gammaColumn.push(columns);
  }

  function gammaDeposition(cell, timeS) {
    let escape = 0;
    for (let k = 0; k < GAMMA_DIRECTIONS; k += 1) {
      escape += 0.5 * gaussWeights[k] * Math.exp(-config.gammaRayOpacity * gammaColumn[cell][k] / (timeS * timeS));
    }
    return 1 - escape;
  }

  function heating(cell, timeS) {
    const nickel = composition[cell].nickel;
    if (nickel <= 0) return 0;
    const positrons = config.positronFraction * cobaltPower(timeS);
    return nickel * ((nickelCobaltHeating(timeS) - positrons) * gammaDeposition(cell, timeS) + positrons);
  }

  let gridMass = 0;
  for (let i = 0; i < GRID_CELLS; i += 1) {
    gridMass += density[i] * (4 * Math.PI / 3) * ((edges[i + 1] * 1e5) ** 3 - (edges[i] * 1e5) ** 3);
  }
  const specificThermal = config.initialThermalEnergyErg / gridMass;
  const t0 = config.homologousStartS;
  const sampleTimes = Array.from({ length: TABLE_TIMES }, (_, k) => (
    t0 * (config.finalTimeS / t0) ** (k / (TABLE_TIMES - 1))
  ));
  const diffusion = solveHomologousDiffusion({
    velocityEdgesKmS: edges,
    densityFactor: density,
    opacity,
    heating,
    initialEnergyDensity: (cell, time) => specificThermal * density[cell] / time ** 3,
    startTimeS: t0,
    endTimeS: config.finalTimeS,
    sampleTimesS: sampleTimes,
    stepsPerDecade: 160,
  });
  const profiles = tabulateDiffusionProfiles({
    diffusion,
    velocityEdgesKmS: edges,
    densityFactor: density,
    opacity,
    heating,
    sampleTimesS: sampleTimes,
    bins: PROFILE_BINS,
    maximumVelocityKmS: vmax,
  });

  function profileAt(ejectaTimeS) {
    return diffusionProfilesAt(profiles, ejectaTimeS);
  }

  // ---- Explosion surface (before homology) ---------------------------------
  function surfaceKick(elapsed, speed) {
    if (elapsed <= 0) return 0;
    const tau = config.surfaceAccelerationTimeS;
    return speed * (elapsed - tau * (1 - Math.exp(-elapsed / tau)));
  }

  // Fireball photosphere: T_fb (t / 1 s)^-n, blended in log t from
  // fireballMatchStartS onto the same power law anchored to the diffusion
  // solution's photospheric temperature at the hand-off t0. The stretched
  // surface has (1 + b)^2 + a^2 / 3 times the area of the sphere, so T is
  // lowered to radiate the same luminosity.
  const stretchAreaFactor = (1 + stretchB) ** 2 + stretchA ** 2 / 3;
  const handoffTemperatureK = profileAt(t0).effectiveTemperatureK * stretchAreaFactor ** -0.25;
  function fireballTemperature(ejectaTime) {
    const time = Math.max(ejectaTime, 0.05);
    const early = config.fireballTemperatureK * time ** -config.fireballCoolingIndex;
    const matched = handoffTemperatureK * (time / t0) ** -config.fireballCoolingIndex;
    const weight = smoothstep(Math.log(config.fireballMatchStartS), Math.log(t0), Math.log(time));
    return Math.exp((1 - weight) * Math.log(early) + weight * Math.log(matched));
  }

  function surfaceTemperature(heliumElapsed, coreElapsed, ejectaTime) {
    let temperature = config.primaryTemperatureK;
    if (heliumElapsed > 0) {
      temperature = Math.max(
        temperature,
        config.heliumFlashTemperatureK / Math.sqrt(Math.max(heliumElapsed, 0.01) / 0.01),
      );
    }
    if (coreElapsed > 0) {
      const flash = config.coreFlashTemperatureK * (Math.max(coreElapsed, 0.01) / 0.01) ** (-1 / 3)
        * Math.exp(-coreElapsed / config.flashDecayS);
      temperature = Math.max(temperature, flash, fireballTemperature(ejectaTime));
    }
    return temperature;
  }

  function explosionSurfaceState(timeS) {
    const directions = 96;
    let area = 0;
    let luminosity = 0;
    let maximumRadius = primaryRadiusKm;
    let maximumTemperature = config.primaryTemperatureK;
    const elements = [];
    const coreIgnition = [-config.coreIgnitionRadiusFraction * primaryRadiusKm, 0, 0];
    for (let k = 0; k < directions; k += 1) {
      const cosTheta = 1 - 2 * (k + 0.5) / directions;
      const theta = Math.acos(cosTheta);
      const heliumTime = theta / heliumAngularSpeed;
      const surfacePoint = [primaryRadiusKm * cosTheta, primaryRadiusKm * Math.sin(theta), 0];
      const coreTime = coreIgnitionTimeS + Math.hypot(
        surfacePoint[0] - coreIgnition[0],
        surfacePoint[1] - coreIgnition[1],
      ) / config.coreDetonationSpeedKmS;
      const radius = primaryRadiusKm
        + surfaceKick(timeS - heliumTime, config.heliumAshSpeedKmS)
        + surfaceKick(timeS - coreTime, coreKickAt(cosTheta));
      const temperature = surfaceTemperature(timeS - heliumTime, timeS - coreTime, timeS - explosionTimeS);
      const cellArea = 4 * Math.PI * radius * radius / directions;
      area += cellArea;
      luminosity += cellArea * 1e10 * STEFAN_BOLTZMANN_CGS * temperature ** 4;
      maximumRadius = Math.max(maximumRadius, radius);
      maximumTemperature = Math.max(maximumTemperature, temperature);
      elements.push([surfaceLuminance(temperature), cellArea]);
    }
    // Area-weighted median surface luminance: the camera exposes for the
    // surface that covers most of the star, so a flash covering a small part
    // of it overexposes instead of blacking out the rest.
    elements.sort((a, b) => a[0] - b[0]);
    let cumulative = 0;
    let medianLuminance = elements[elements.length - 1][0];
    for (const [value, cellArea] of elements) {
      cumulative += cellArea;
      if (cumulative >= 0.5 * area) {
        medianLuminance = value;
        break;
      }
    }
    return {
      luminosity,
      maximumRadiusKm: maximumRadius,
      // Radius of the sphere of equal area (the framing and disk measure).
      meanRadiusKm: Math.sqrt(area / (4 * Math.PI)),
      effectiveTemperatureK: (luminosity / (area * 1e10 * STEFAN_BOLTZMANN_CGS)) ** 0.25,
      maximumTemperatureK: maximumTemperature,
      medianLuminance,
    };
  }

  // ---- Timeline ---------------------------------------------------------------
  const firstTime = -2.5 * periodS;
  const homologousSceneTime = explosionTimeS + t0;
  const timeline = createPresentationTimeline({
    segments: [
      { id: "binary", kind: "linear", start: firstTime, end: -1, wallSeconds: 10 },
      // The helium sweep at ~1/3 speed, then the core detonation, breakout and
      // donor strike slowed ~7x so the explosion itself can be followed.
      { id: "detonation", kind: "linear", start: -1, end: 1.9, wallSeconds: 9 },
      { id: "breakout", kind: "linear", start: 1.9, end: 3.4, wallSeconds: 10 },
      { id: "expansion", kind: "log", start: 3.4, end: explosionTimeS + config.finalTimeS, wallSeconds: 26 },
    ],
    endHoldSeconds: 2.5,
    loop: true,
  });

  function colourTemperature(ejectaTimeS, effectiveTemperature) {
    const days = ejectaTimeS / DAY_S;
    const observed = colourTemperatureFromBV(supernovaIaBV(days, config.bMaximumDays));
    // Before first light (<~0.3 d) the photosphere radiates like a blackbody.
    const weight = smoothstep(0.1, 0.5, days);
    return (1 - weight) * effectiveTemperature + weight * observed;
  }

  function bolometricLuminosity(timeS) {
    if (timeS < 0) return quiescentLuminosity;
    const ejectaTime = timeS - explosionTimeS;
    if (ejectaTime < t0) {
      return Math.max(explosionSurfaceState(timeS).luminosity, quiescentLuminosity);
    }
    return profileAt(ejectaTime).luminosity;
  }

  function regimeKey(timeS) {
    if (timeS < 0) return "supernovaIa.regime.inspiral";
    if (timeS < coreIgnitionTimeS) return "supernovaIa.regime.heliumDetonation";
    const ejectaTime = timeS - explosionTimeS;
    if (ejectaTime < t0) return "supernovaIa.regime.coreDetonation";
    if (ejectaTime < 0.5 * DAY_S) return "supernovaIa.regime.fireball";
    if (ejectaTime < 25 * DAY_S) return "supernovaIa.regime.radioactive";
    if (ejectaTime < 80 * DAY_S) return "supernovaIa.regime.decline";
    return "supernovaIa.regime.nebular";
  }

  const lightCurveTimes = [];
  const lightCurveLog10 = [];
  for (let k = 0; k <= 480; k += 1) {
    const time = timeline.timeAtProgress(k / 480);
    lightCurveTimes.push(time);
    lightCurveLog10.push(Math.log10(Math.max(bolometricLuminosity(time), 1)));
  }

  // ---- State at a time ---------------------------------------------------------
  function orbitalPhase(timeS) {
    return omega * Math.min(timeS, 0);
  }

  function fromCorotating(v, phase) {
    const c = Math.cos(phase);
    const s = Math.sin(phase);
    return [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]];
  }

  const primaryAtExplosion = fromCorotating([-mu * separationKm, 0, 0], 0);
  const donorAtExplosion = fromCorotating([(1 - mu) * separationKm, 0, 0], 0);
  // The binary is held at its t = 0 configuration through the detonations
  // (14 degrees of orbit are neglected) and released when the core
  // detonation unbinds the accretor; the donor keeps its Roche shape until the
  // ejecta strike it.
  const releaseTimeS = explosionTimeS;
  // Orbital velocities at t = 0 (angular momentum +y): the ejecta inherit the
  // primary's (the D6 "slingshot"), the released donor keeps its own.
  const primaryVelocity = fromCorotating([0, 0, primarySpeedKmS], 0);
  const donorVelocity = fromCorotating([0, 0, -donorSpeedKmS], 0);
  const ignitionDirection = fromCorotating(impactDirection, 0);
  const shadowAxis = (() => {
    const v = donorAtExplosion.map((c, i) => c - primaryAtExplosion[i]);
    const n = Math.hypot(...v);
    return v.map((c) => c / n);
  })();
  // The ejecta strike the donor when the opaque explosion surface facing it
  // (the same helium-ash and core-ash kinematics the tracer draws) reaches
  // the donor's near surface; the two stars separate at their orbital speeds.
  const facingCosine = clamp(shadowAxis.reduce((sum, c, i) => sum + c * ignitionDirection[i], 0), -1, 1);
  const facingHeliumTimeS = Math.acos(facingCosine) / heliumAngularSpeed;
  const facingCoreTimeS = coreIgnitionTimeS + Math.hypot(...shadowAxis.map(
    (c, i) => primaryRadiusKm * (c + config.coreIgnitionRadiusFraction * ignitionDirection[i]),
  )) / config.coreDetonationSpeedKmS;
  const donorShockTimeS = (() => {
    const relativeSpeed = primarySpeedKmS + donorSpeedKmS;
    const reach = (time) => primaryRadiusKm
      + surfaceKick(time - facingHeliumTimeS, config.heliumAshSpeedKmS)
      + surfaceKick(time - facingCoreTimeS, coreKickAt(facingCosine))
      - (Math.hypot(separationKm, relativeSpeed * Math.max(time - releaseTimeS, 0)) - donorRadiusKm);
    let low = 0;
    let high = 20;
    for (let k = 0; k < 60; k += 1) {
      const mid = 0.5 * (low + high);
      if (reach(mid) >= 0) high = mid;
      else low = mid;
    }
    return high;
  })();
  function evaluate(timeS) {
    const time = clamp(timeS, timeline.firstTime, timeline.finalTime);
    const phase = orbitalPhase(time);
    const exploded = time >= 0;
    const primaryCentre = exploded
      ? primaryAtExplosion.map((c, i) => c + primaryVelocity[i] * Math.max(time - releaseTimeS, 0))
      : fromCorotating([-mu * separationKm, 0, 0], phase);
    const donorCentre = exploded
      ? donorAtExplosion.map((c, i) => c + donorVelocity[i] * Math.max(time - releaseTimeS, 0))
      : fromCorotating([(1 - mu) * separationKm, 0, 0], phase);
    const ejectaTime = time - explosionTimeS;
    const homologous = ejectaTime >= t0;
    let profile = null;
    let surface = null;
    let luminosity;
    let effectiveTemperatureK;
    let colourTemperatureK;
    let photosphereVelocityKmS = null;
    let visibleRadiusKm;
    if (homologous) {
      profile = profileAt(ejectaTime);
      luminosity = profile.luminosity;
      photosphereVelocityKmS = profile.photosphereVelocityKmS;
      const coreRadius = 9000 * ejectaTime;
      // Equal-area radius of the stretched photosphere.
      const photosphereRadius = photosphereVelocityKmS * ejectaTime * Math.sqrt(stretchAreaFactor);
      visibleRadiusKm = Math.max(photosphereRadius, coreRadius);
      effectiveTemperatureK = photosphereVelocityKmS > 0
        ? profile.effectiveTemperatureK
        : (luminosity / (4 * Math.PI * (visibleRadiusKm * 1e5) ** 2 * STEFAN_BOLTZMANN_CGS)) ** 0.25;
      colourTemperatureK = colourTemperature(ejectaTime, effectiveTemperatureK);
    } else if (exploded) {
      surface = explosionSurfaceState(time);
      luminosity = Math.max(surface.luminosity, quiescentLuminosity);
      effectiveTemperatureK = surface.effectiveTemperatureK;
      colourTemperatureK = effectiveTemperatureK;
      visibleRadiusKm = surface.meanRadiusKm;
    } else {
      luminosity = quiescentLuminosity;
      effectiveTemperatureK = config.primaryTemperatureK;
      colourTemperatureK = effectiveTemperatureK;
      visibleRadiusKm = separationKm;
    }
    // Camera scale follows the photosphere (or the nebular Fe core) with the
    // same 1.6 R framing as the fireball, so it is continuous at the hand-off.
    let scaleKm;
    if (homologous) {
      const viewVelocity = Math.max(1.6 * Math.max(photosphereVelocityKmS, 8000) * Math.sqrt(stretchAreaFactor), 12000);
      scaleKm = Math.max(separationKm, viewVelocity * ejectaTime);
    } else if (exploded) {
      scaleKm = Math.max(separationKm, 1.6 * surface.meanRadiusKm);
    } else {
      scaleKm = separationKm;
    }
    const focusWeight = exploded ? smoothstep(0, 3, time) : 0;
    const focusKm = primaryCentre.map((c) => c * focusWeight);
    // The camera's exposure reference (cd/m^2): the quiescent accretor's
    // surface, the area-median surface of the exploding star, or the mean
    // surface luminance L / (4 pi^2 R^2) of the photospheric disk.
    let referenceLuminance;
    if (homologous) {
      referenceLuminance = 0.683 * luminosity / (4 * Math.PI * Math.PI * (visibleRadiusKm * 1e5) ** 2)
        * luminousFraction(colourTemperatureK);
    } else if (exploded) {
      referenceLuminance = surface.medianLuminance;
    } else {
      referenceLuminance = surfaceLuminance(config.primaryTemperatureK);
    }
    // Hemisphere-blind donor temperature (fallback renderer and readouts);
    // the tracer resolves the facing-hemisphere shock itself.
    const donorTemperatureK = time > donorShockTimeS
      ? Math.max(config.donorTemperatureK, config.donorPostShockTemperatureK)
      : config.donorTemperatureK;
    return {
      timeS: time,
      exploded,
      homologous,
      phase,
      ejectaTimeS: ejectaTime,
      scaleKm,
      focusKm,
      primaryCentre,
      donorCentre,
      donorTemperatureK,
      profile,
      surface,
      luminosity,
      effectiveTemperatureK,
      colourTemperatureK,
      photosphereVelocityKmS,
      visibleRadiusKm,
      referenceLuminance,
      regimeKey: regimeKey(time),
    };
  }

  /**
   * Pack the Type Ia uniform tail (after the 36-float renderer prefix).
   * @param {object} sample from evaluate()
   * @param {object} view { scaleKm, exposureLog2, quality, steps, mode }
   */
  function packUniforms(sample, view, layout, floatCount) {
    const out = new Float32Array(floatCount - 36);
    const put = (name, values) => {
      const offset = layout[name] - 36;
      for (let i = 0; i < values.length; i += 1) out[offset + i] = values[i];
    };
    const L = view.scaleKm;
    const toUnits = (v) => v.map((c) => c / L);
    put("sceneScale", [L, sample.timeS, view.exposureLog2, SPEED_OF_LIGHT_KM_S / L]);
    put("quality", view.quality);
    put("steps", [view.steps[0], view.steps[1], view.steps[2], view.mode ?? 0]);
    put("primaryPositionRadius", [...toUnits(sample.primaryCentre), primaryRadiusKm / L]);
    put("primaryState", [config.primaryTemperatureK, config.limbDarkening, sample.homologous ? 0 : 1, impactTemperatureK]);
    put("hotSpot", [...impactDirection, Math.cos(config.impactAngularRadiusDeg * Math.PI / 180)]);
    put("donorPositionRadius", [...toUnits(sample.donorCentre), donorBoundingRadiusKm / L]);
    put("donorState", [config.donorTemperatureK, mu, separationKm / L, lagrangePotential]);
    put("orbit", [Math.cos(sample.phase), Math.sin(sample.phase), config.gravityDarkeningExponent, sample.timeS < donorShockTimeS ? 1 : 0]);
    put("donorMotion", [poleGravity, sample.homologous ? 0 : 1, donorRadiusKm / L, lagrangeX]);
    put("donorShock", [donorShockTimeS, config.donorShockTemperatureK, config.donorPostShockTemperatureK, 0]);
    // Irradiating sources in units of 1e36 erg/s (float32 range): the
    // Lambertian impact spot (radiant intensity L / pi along its normal)
    // before the explosion, and the accretor as a point source of its
    // current luminosity.
    const accretorLuminosity = sample.homologous ? 0 : (sample.exploded ? sample.surface.luminosity : primaryLuminosity);
    put("irradiation", [
      sample.exploded ? 0 : impactLuminosity / Math.PI / 1e36,
      config.irradiationAbsorbedFraction,
      accretorLuminosity / 1e36,
      0,
    ]);
    put("streamState", [config.donorTemperatureK, 60, stream.points.length, sample.exploded ? 0 : 1]);
    stream.points.forEach((point, index) => {
      put(`streamPoint${index}`, [point[0], point[1], point[2], streamHalfWidth * (1 + 0.8 * index / (stream.points.length - 1))]);
    });
    const explosionActive = sample.exploded && !sample.homologous;
    put("explosion", [...ignitionDirection, Math.max(sample.timeS, 0)]);
    const coreIgnitionPoint = sample.primaryCentre.map(
      (c, i) => c - config.coreIgnitionRadiusFraction * primaryRadiusKm * ignitionDirection[i],
    );
    put("explosionFronts", [heliumAngularSpeed, coreIgnitionTimeS, config.coreDetonationSpeedKmS / L, explosionActive ? 1 : 0]);
    put("coreIgnition", [...toUnits(coreIgnitionPoint), explosionTimeS]);
    put("explosionSurface", [config.surfaceAccelerationTimeS, config.heliumAshSpeedKmS / L, vmax / L, primaryRadiusKm / L]);
    // The velocity stretch shapes the fireball as well as the ejecta.
    put("ejectaAsymmetry", [...ignitionDirection, stretchA]);
    put("ejectaZones", [11000 / vmax, 16500 / vmax, 24300 / vmax, stretchB]);
    put("fireball", [explosionActive ? fireballTemperature(sample.ejectaTimeS) : 0, 0, 0, 0]);
    if (sample.homologous) {
      const centre = primaryAtExplosion.map((c, i) => c + primaryVelocity[i] * (sample.timeS - releaseTimeS));
      put("ejectaCentre", [...toUnits(centre), sample.ejectaTimeS]);
      put("ejectaDrift", [...primaryVelocity.map((c) => c / L), vmax]);
      put("ejectaAxis", [...shadowAxis, sample.profile.log10Time]);
      const half = config.shadowHalfAngleDeg * Math.PI / 180;
      const rimWidth = Math.sin(half) * config.shadowRimWidthDeg * Math.PI / 180;
      put("ejectaCone", [Math.cos(half), config.shadowInteriorDensity, rimWidth, 1]);
      put("ejectaPhotosphere", [
        sample.effectiveTemperatureK,
        sample.colourTemperatureK / Math.max(sample.effectiveTemperatureK, 1),
        sample.colourTemperatureK,
        config.shadowRimDensity,
      ]);
      put("ejectaBounds", [sample.profile.activeVelocityKmS, 0, 0, 0]);
      out.set(sample.profile.thermal, layout.ejectaProfile0 - 36);
      out.set(sample.profile.radiative, layout.ejectaRadiative0 - 36);
    }
    return out;
  }

  const binary = Object.freeze({
    primaryRadiusKm,
    donorRadiusKm,
    separationKm,
    periodS,
    orbitalSpeedKmS,
    primarySpeedKmS,
    donorSpeedKmS,
    massRatio,
    mu,
    lagrangeX,
    lagrangePotential,
    impactLeadDeg,
    streamReachesAccretor: stream.reachedAccretor,
    impactTemperatureK,
    impactLuminosityErgS: impactLuminosity,
    inspiralTimeS: gravitationalInspiralTimeS(config.primaryMassSolar, config.donorMassSolar, separationKm),
    primaryCentralDensity: primaryStructure.centralDensity,
  });
  const explosion = Object.freeze({
    heliumAngularSpeed,
    coreIgnitionTimeS,
    explosionTimeS,
    homologousSceneTime,
    exponentialVelocityKmS: ve,
    nickelMassSolar: nickelMassG / SOLAR_MASS_G,
  });

  return Object.freeze({
    config,
    binary,
    explosion,
    diffusion,
    timeline,
    lightCurve: Object.freeze({
      timesS: Float64Array.from(lightCurveTimes),
      log10LuminosityErgS: Float64Array.from(lightCurveLog10),
    }),
    bolometricLuminosity,
    evaluate,
    packUniforms,
    solarLuminosity: SOLAR_LUMINOSITY_ERG_S,
  });
}
