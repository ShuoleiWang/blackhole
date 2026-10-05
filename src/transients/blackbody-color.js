// Visible colour of thermal emission for the transient scenes.
//
// The CIE 1931 2-degree colour-matching functions use the Wyman, Sloan &
// Shirley (2013) multi-lobe piecewise-Gaussian fit (JCGT 2(2)), accurate to
// about 1% of peak. A spectrum is integrated over 360-830 nm at 1 nm and
// converted to linear sRGB (IEC 61966-2-1). Colours are expressed per unit
// bolometric radiance, so scene radiance is (sigma T^4 / pi) * colour(T): the
// luminance channel Y is the luminous fraction (683 lm/W = 1). As in the
// dual-disk emission model, a channel calibration makes a 6500 K blackbody
// neutral in the D65 working space without changing its luminance.

import {
  SPEED_OF_LIGHT_CM_S,
  STEFAN_BOLTZMANN_CGS,
} from "./physical-constants.js";

const PLANCK_CGS = 6.62607015e-27;
const BOLTZMANN_CGS = 1.380649e-16;
const SECOND_RADIATION_CONSTANT_CM_K = PLANCK_CGS * SPEED_OF_LIGHT_CM_S / BOLTZMANN_CGS;
const FIRST_RADIATION_CONSTANT = 2 * PLANCK_CGS * SPEED_OF_LIGHT_CM_S ** 2;
const LAMBDA_MIN_NM = 360;
const LAMBDA_MAX_NM = 830;

function lobe(lambdaNm, mean, sigmaBelow, sigmaAbove) {
  const sigma = lambdaNm < mean ? sigmaBelow : sigmaAbove;
  const x = (lambdaNm - mean) / sigma;
  return Math.exp(-0.5 * x * x);
}

export function cie1931(lambdaNm) {
  return [
    1.056 * lobe(lambdaNm, 599.8, 37.9, 31.0)
      + 0.362 * lobe(lambdaNm, 442.0, 16.0, 26.7)
      - 0.065 * lobe(lambdaNm, 501.1, 20.4, 26.2),
    0.821 * lobe(lambdaNm, 568.8, 46.9, 40.5)
      + 0.286 * lobe(lambdaNm, 530.9, 16.3, 31.1),
    1.217 * lobe(lambdaNm, 437.0, 11.8, 36.0)
      + 0.681 * lobe(lambdaNm, 459.0, 26.0, 13.8),
  ];
}

// Planck spectral radiance B_lambda in erg s^-1 cm^-2 sr^-1 cm^-1.
export function planckLambda(lambdaNm, temperatureK) {
  const lambdaCm = lambdaNm * 1e-7;
  const exponent = SECOND_RADIATION_CONSTANT_CM_K / (lambdaCm * temperatureK);
  if (exponent > 700) {
    return 0;
  }
  return FIRST_RADIATION_CONSTANT / lambdaCm ** 5 / Math.expm1(exponent);
}

export function xyzToLinearSrgb([x, y, z]) {
  return [
    3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
    -0.969266 * x + 1.8760108 * y + 0.041556 * z,
    0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
  ];
}

// XYZ of a blackbody per unit bolometric radiance, optionally multiplied by a
// transmission T(lambda) (e.g. line blanketing), still normalised to the
// unattenuated bolometric radiance sigma T^4 / pi.
export function blackbodyXyzPerBolometric(temperatureK, transmission = null) {
  if (!(temperatureK > 0)) {
    throw new RangeError("Blackbody temperature must be positive");
  }
  const xyz = [0, 0, 0];
  for (let lambda = LAMBDA_MIN_NM; lambda <= LAMBDA_MAX_NM; lambda += 1) {
    const weight = (lambda === LAMBDA_MIN_NM || lambda === LAMBDA_MAX_NM) ? 0.5 : 1;
    const radiance = planckLambda(lambda, temperatureK)
      * (transmission ? transmission(lambda) : 1) * weight * 1e-7;
    const cmf = cie1931(lambda);
    xyz[0] += radiance * cmf[0];
    xyz[1] += radiance * cmf[1];
    xyz[2] += radiance * cmf[2];
  }
  const bolometric = STEFAN_BOLTZMANN_CGS * temperatureK ** 4 / Math.PI;
  return xyz.map((component) => component / bolometric);
}

const NEUTRAL_REFERENCE_K = 6500;
const referenceRgb = xyzToLinearSrgb(blackbodyXyzPerBolometric(NEUTRAL_REFERENCE_K));
const referenceLuminance = blackbodyXyzPerBolometric(NEUTRAL_REFERENCE_K)[1];
export const NEUTRAL_CHANNEL_CALIBRATION = Object.freeze(
  referenceRgb.map((channel) => referenceLuminance / channel),
);

// Linear-sRGB colour per unit bolometric radiance (luminance = luminous
// fraction), calibrated so that 6500 K is neutral.
export function blackbodyLinearSrgbPerBolometric(temperatureK, transmission = null) {
  const rgb = xyzToLinearSrgb(blackbodyXyzPerBolometric(temperatureK, transmission));
  return rgb.map((channel, index) => channel * NEUTRAL_CHANNEL_CALIBRATION[index]);
}

// GPU table over log10 T. Each texel stores the colour divided by its
// luminous fraction (O(1) chromaticity weights) and log10 of the luminous
// fraction, so a 32-bit table resolves both 10^-20 and 0.14 without loss.
// Negative channels (outside the sRGB gamut, e.g. very red spectra) are kept:
// the shader clamps only the final radiance.
export function blackbodyColorTable({
  minLog10Temperature = 2.5,
  maxLog10Temperature = 9.0,
  entries = 512,
  transmission = null,
} = {}) {
  if (!(maxLog10Temperature > minLog10Temperature) || !Number.isInteger(entries) || entries < 2) {
    throw new RangeError("Invalid blackbody colour table range");
  }
  const data = new Float32Array(entries * 4);
  for (let index = 0; index < entries; index += 1) {
    const log10T = minLog10Temperature
      + (maxLog10Temperature - minLog10Temperature) * index / (entries - 1);
    const temperature = 10 ** log10T;
    const rgb = blackbodyLinearSrgbPerBolometric(temperature, transmission);
    const luminance = Math.max(
      0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2],
      1e-38,
    );
    data[4 * index] = rgb[0] / luminance;
    data[4 * index + 1] = rgb[1] / luminance;
    data[4 * index + 2] = rgb[2] / luminance;
    data[4 * index + 3] = Math.log10(luminance);
  }
  return Object.freeze({
    minLog10Temperature,
    maxLog10Temperature,
    entries,
    data,
  });
}
