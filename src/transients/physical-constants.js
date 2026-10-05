// CGS physical constants shared by the transient scenes (CODATA 2018 / IAU
// 2015 nominal values).

export const SPEED_OF_LIGHT_CM_S = 2.99792458e10;
export const SPEED_OF_LIGHT_KM_S = 2.99792458e5;
export const GRAVITATIONAL_CONSTANT_CGS = 6.6743e-8;
export const STEFAN_BOLTZMANN_CGS = 5.670374419e-5;
// Radiation constant a = 4 sigma / c.
export const RADIATION_CONSTANT_CGS = 4 * STEFAN_BOLTZMANN_CGS / SPEED_OF_LIGHT_CM_S;
export const SOLAR_MASS_G = 1.98841e33;
// IAU 2015 nominal solar luminosity.
export const SOLAR_LUMINOSITY_ERG_S = 3.828e33;
export const SOLAR_RADIUS_KM = 6.957e5;
export const ASTRONOMICAL_UNIT_KM = 1.495978707e8;
export const DAY_S = 86400;
// G Msun / c^2 in km.
export const GRAVITATIONAL_RADIUS_KM_PER_SOLAR_MASS = (
  GRAVITATIONAL_CONSTANT_CGS * SOLAR_MASS_G / SPEED_OF_LIGHT_CM_S ** 2 / 1e5
);
export const KM_TO_CM = 1e5;
