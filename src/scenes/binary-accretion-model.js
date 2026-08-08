/*
 * Analytic geometry and temperature scales for two tidally truncated,
 * Schwarzschild thin disks. Radii use the initial total mass as M = 1.
 * This module is renderer-independent so the shader-facing scene state can be
 * checked against a small deterministic CPU contract.
 */

export const SCHWARZSCHILD_ISCO_FACTOR = 6;
export const TIDAL_TRUNCATION_FACTOR = 0.8;

const REFERENCE_TEMPERATURE_K = 1.43e5;
const REFERENCE_MASS_SOLAR = 1e8;

// Fifteen uniformly spaced samples spanning 380-780 nm. The XYZ weights are the
// compact Wyman-Sloan-Shirley analytic approximation to the CIE 1931 2-degree
// colour-matching functions, evaluated once at fixed wavelengths. Keeping the
// table here gives the production WGSL a small deterministic CPU oracle.
const VISIBLE_CIE_SAMPLES = Object.freeze([
  [0.38000000, 0.000101768, 0.000126361, 0.003342521],
  [0.40857143, 0.041103205, 0.002427076, 0.171379214],
  [0.43714286, 0.345374934, 0.016094588, 1.694938040],
  [0.46571429, 0.231835560, 0.073791241, 1.489866541],
  [0.49428571, 0.013837655, 0.256028875, 0.368248028],
  [0.52285714, 0.091902156, 0.761894395, 0.070519050],
  [0.55142857, 0.457097020, 0.996550478, 0.007724523],
  [0.58000000, 0.920460516, 0.872133791, 0.000450337],
  [0.60857143, 1.014443172, 0.519281599, 0.000013971],
  [0.63714286, 0.510206266, 0.198321624, 0.000000231],
  [0.66571429, 0.109492500, 0.046801697, 0.000000002],
  [0.69428571, 0.010026499, 0.006733690, 0.000000000],
  [0.72285714, 0.000391779, 0.000589022, 0.000000000],
  [0.75142857, 0.000006532, 0.000031314, 0.000000000],
  [0.78000000, 0.000000023, 0.000000506, 0.000000000],
].map(Object.freeze));

const CIE_REFERENCE_Y_6500K = 1.3256608705;
const CIE_D65_CHANNEL_CALIBRATION = Object.freeze([
  0.9662282813,
  1.0159099238,
  0.9498410897,
]);

function finiteNumber(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be a finite number`);
  }
  return value;
}

function positiveFinite(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${label} must be a finite positive number`);
  }
  return value;
}

/**
 * Eggleton's volume-equivalent Roche-lobe radius as a fraction of binary
 * separation, with q = body mass / companion mass.
 */
export function eggletonRocheLobeFraction(q) {
  const massRatio = positiveFinite(q, "q");
  const qOneThird = Math.cbrt(massRatio);
  const qTwoThirds = qOneThird * qOneThird;
  return (
    0.49 * qTwoThirds
    / (0.6 * qTwoThirds + Math.log1p(qOneThird))
  );
}

/**
 * Dimensionless zero-torque Schwarzschild thin-disk flux profile. The inner
 * boundary is dark; outside it the Newtonian profile is
 * (r_in / r)^3 * (1 - sqrt(r_in / r)).
 */
export function zeroTorqueThinDiskFluxShape(radiusM, innerRadiusM) {
  const radius = finiteNumber(radiusM, "radiusM");
  const innerRadius = positiveFinite(innerRadiusM, "innerRadiusM");
  if (radius <= innerRadius) {
    return 0;
  }
  const inverseRadius = innerRadius / radius;
  return (
    inverseRadius ** 3
    * (1 - Math.sqrt(inverseRadius))
  );
}

/** Temperature profile corresponding to the dimensionless flux above. */
export function zeroTorqueThinDiskTemperatureShape(radiusM, innerRadiusM) {
  return zeroTorqueThinDiskFluxShape(radiusM, innerRadiusM) ** 0.25;
}

/**
 * C2 fade for a tidally truncated annulus. The transition width scales with
 * body mass so equal dimensionless annuli receive the same stability weight.
 */
export function stableAnnulusWeight(
  widthM,
  massFraction,
  fadeWidthPerBodyM = 0.75,
) {
  const width = finiteNumber(widthM, "widthM");
  if (width < 0) {
    throw new RangeError("widthM must be non-negative");
  }
  const fraction = positiveFinite(massFraction, "massFraction");
  const fadePerBody = positiveFinite(
    fadeWidthPerBodyM,
    "fadeWidthPerBodyM",
  );
  const normalizedWidth = width / (fadePerBody * fraction);
  if (normalizedWidth <= 0) {
    return 0;
  }
  if (normalizedWidth >= 1) {
    return 1;
  }
  return (
    normalizedWidth ** 3
    * (normalizedWidth * (normalizedWidth * 6 - 15) + 10)
  );
}

function smootherstep01(value) {
  const normalized = Math.min(Math.max(value, 0), 1);
  return (
    normalized ** 3
    * (normalized * (normalized * 6 - 15) + 10)
  );
}

/** C2 photospheric covering fraction mirrored by the dual-disk WGSL. */
export function annulusEdgeCoverage({
  radiusM,
  innerRadiusM,
  outerRadiusM,
  bodyMassM,
} = {}) {
  const radius = finiteNumber(radiusM, "radiusM");
  const inner = positiveFinite(innerRadiusM, "innerRadiusM");
  const outer = positiveFinite(outerRadiusM, "outerRadiusM");
  const bodyMass = positiveFinite(bodyMassM, "bodyMassM");
  if (outer <= inner) {
    throw new RangeError("outerRadiusM must be greater than innerRadiusM");
  }
  const width = outer - inner;
  const edgeWidth = Math.min(
    Math.max(0.12 * width, 0.08 * bodyMass),
    0.45 * width,
  );
  return (
    smootherstep01((radius - inner) / edgeWidth)
    * smootherstep01((outer - radius) / edgeWidth)
  );
}

function wrapAngle(angle) {
  const twoPi = 2 * Math.PI;
  return ((angle % twoPi) + twoPi) % twoPi;
}

/**
 * Bounded analytic emissivity structure mirrored by the dual-disk WGSL.
 *
 * The m=2 term follows the instantaneous binary axis. The remaining modes use
 * wrapped pattern phases calibrated to the Kepler frequency at the zero-torque
 * flux peak; they are an emissivity snapshot proxy, not evolved GRMHD matter.
 */
export function analyticDiskSurfaceStructureWeight({
  azimuth,
  radiusM,
  innerRadiusM,
  bodyMassM,
  timeM,
} = {}) {
  const angle = finiteNumber(azimuth, "azimuth");
  const radius = positiveFinite(radiusM, "radiusM");
  const innerRadius = positiveFinite(innerRadiusM, "innerRadiusM");
  const bodyMass = positiveFinite(bodyMassM, "bodyMassM");
  const time = finiteNumber(timeM, "timeM");
  const logarithmicRadius = Math.log(Math.max(radius / innerRadius, 1));
  const referenceRadius = (49 / 36) * innerRadius;
  const omegaPeak = Math.sqrt(bodyMass / referenceRadius ** 3);
  const tidal = 0.16 * Math.cos(
    2 * (angle - 0.42 * logarithmicRadius),
  );
  const modes = [
    [5, 0.82, 0.032, 1.2, 0],
    [9, 0.93, 0.024, -2.1, 1.7],
    [14, 1.04, 0.016, 3.6, 3.1],
    [21, 1.13, 0.010, -5.4, 0.8],
    [31, 1.21, 0.008, 7.1, 2.4],
  ];
  const texture = modes.reduce((sum, [mode, speed, amplitude, radial, phase]) => {
    const patternPhase = wrapAngle(speed * omegaPeak * time);
    return sum + amplitude * Math.sin(
      mode * (angle - patternPhase) + radial * logarithmicRadius + phase,
    );
  }, 0);
  return 1 + tidal + texture;
}

function createDisk(
  id,
  massFraction,
  companionMassFraction,
  separationM,
  maximumOuterRadiusM,
) {
  const massRatio = massFraction / companionMassFraction;
  const innerRadiusM = SCHWARZSCHILD_ISCO_FACTOR * massFraction;
  const rocheLobeRadiusM = (
    separationM * eggletonRocheLobeFraction(massRatio)
  );
  const outerRadiusM = Math.min(
    maximumOuterRadiusM,
    TIDAL_TRUNCATION_FACTOR * rocheLobeRadiusM,
  );
  const widthM = Math.max(outerRadiusM - innerRadiusM, 0);

  return Object.freeze({
    id,
    massFraction,
    massRatio,
    innerRadiusM,
    rocheLobeRadiusM,
    outerRadiusM,
    widthM,
    stable: widthM > 0,
  });
}

/**
 * Construct the two local disk annuli for a binary at one instantaneous
 * separation. A marginal state retains one stable annulus; a disrupted state
 * has no positive-width annulus.
 */
export function createBinaryAccretionState({
  separationM,
  massFractions,
  maximumOuterRadiusM = 10,
} = {}) {
  const separation = positiveFinite(separationM, "separationM");
  const maximumOuterRadius = positiveFinite(
    maximumOuterRadiusM,
    "maximumOuterRadiusM",
  );
  if (!Array.isArray(massFractions) || massFractions.length !== 2) {
    throw new TypeError("massFractions must contain exactly two values");
  }
  const firstMass = positiveFinite(massFractions[0], "massFractions[0]");
  const secondMass = positiveFinite(massFractions[1], "massFractions[1]");
  if (Math.abs(firstMass + secondMass - 1) > 1e-6) {
    throw new RangeError("massFractions must sum to one initial total mass");
  }
  const disks = Object.freeze([
    createDisk("A", firstMass, secondMass, separation, maximumOuterRadius),
    createDisk("B", secondMass, firstMass, separation, maximumOuterRadius),
  ]);
  const stableCount = disks.filter((disk) => disk.stable).length;
  const phase = stableCount === 2
    ? "dual-stable"
    : stableCount === 1
      ? "tidally-marginal"
      : "disrupted";

  return Object.freeze({
    separationM: separation,
    massFractions: Object.freeze([firstMass, secondMass]),
    maximumOuterRadiusM: maximumOuterRadius,
    disks,
    phase,
  });
}

/**
 * Peak thin-disk colour-temperature scale. The accretion ratio is Eddington-
 * normalized, so the physical scaling is T_peak proportional to (Mdot/M^2)^1/4
 * and therefore to (accretionRatio/M)^1/4 for each body.
 */
export function thinDiskPeakTemperatureK({
  totalMassSolar,
  massFraction,
  accretionRatio,
} = {}) {
  const totalMass = positiveFinite(totalMassSolar, "totalMassSolar");
  const fraction = positiveFinite(massFraction, "massFraction");
  const accretion = positiveFinite(accretionRatio, "accretionRatio");
  const bodyMassSolar = totalMass * fraction;
  return REFERENCE_TEMPERATURE_K * Math.pow(
    accretion * REFERENCE_MASS_SOLAR / bodyMassSolar,
    0.25,
  );
}

/**
 * Visible linear-sRGB response per unit bolometric blackbody flux.
 *
 * The previous renderer normalised three visible samples to unit luminance,
 * which incorrectly put nearly all UV-dominated disk power into the visible
 * image. This integrates a fixed CIE XYZ corpus, converts it to linear sRGB,
 * and divides by T^4. The result is normalised so a 6500 K blackbody maps to
 * neutral unit RGB; hotter emitters stay blue but contribute a smaller visible
 * fraction instead of becoming a featureless white plate.
 */
export function visibleBlackbodyLinearSrgbPerBolometric(temperatureK) {
  const temperature = positiveFinite(temperatureK, "temperatureK");
  const xyz = [0, 0, 0];
  for (const [wavelengthMicron, xBar, yBar, zBar] of VISIBLE_CIE_SAMPLES) {
    const wavelength2 = wavelengthMicron * wavelengthMicron;
    const wavelength5 = wavelength2 * wavelength2 * wavelengthMicron;
    const exponent = Math.min(
      80,
      14387.77 / (temperature * wavelengthMicron),
    );
    const spectrum = 1 / (wavelength5 * Math.max(Math.expm1(exponent), 1e-12));
    xyz[0] += spectrum * xBar;
    xyz[1] += spectrum * yBar;
    xyz[2] += spectrum * zBar;
  }

  const [x, y, z] = xyz;
  const linearSrgb = [
    3.2406 * x - 1.5372 * y - 0.4986 * z,
    -0.9689 * x + 1.8758 * y + 0.0415 * z,
    0.0557 * x - 0.2040 * y + 1.0570 * z,
  ];
  const bolometricNormalization = (
    (6500 / temperature) ** 4 / CIE_REFERENCE_Y_6500K
  );
  const calibrated = linearSrgb.map((channel, index) => Math.max(
    0,
    channel
      * bolometricNormalization
      * CIE_D65_CHANNEL_CALIBRATION[index],
  ));
  if (!calibrated.every(Number.isFinite)) {
    throw new RangeError("temperatureK produced a non-finite visible spectrum");
  }
  return Object.freeze(calibrated);
}
