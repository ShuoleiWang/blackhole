// Zero-temperature white-dwarf structure: the Chandrasekhar (1939) ideal
// degenerate-electron equation of state in hydrostatic equilibrium,
//   P = A f(x),  rho = B x^3,  x = p_F / (m_e c),
//   f(x) = x (2x^2 - 3) sqrt(1 + x^2) + 3 asinh(x),  f'(x) = 8 x^4 / sqrt(1 + x^2),
// with A = pi m_e^4 c^5 / (3 h^3) and B = mu_e m_u (m_e c / hbar)^3 / (3 pi^2).
// A shooting search on the central density matches the requested mass.

import { GRAVITATIONAL_CONSTANT_CGS, SOLAR_MASS_G } from "./physical-constants.js";

const A_EOS = 6.00228e22; // dyn cm^-2
const B_PER_MU_E = 9.73940e5; // g cm^-3

function integrate(centralDensity, muE, steps) {
  const B = B_PER_MU_E * muE;
  let x = Math.cbrt(centralDensity / B);
  let r = 0;
  let m = 0;
  // Start slightly off centre with the Taylor expansion.
  const dr0 = 1e3; // cm
  r = dr0;
  m = (4 / 3) * Math.PI * dr0 ** 3 * centralDensity;
  const radii = [0];
  const masses = [0];
  const densities = [centralDensity];
  // Adaptive step in radius: a fixed fraction of the local scale height.
  let guard = 0;
  while (x > 1e-4 && guard < steps) {
    guard += 1;
    const rho = B * x ** 3;
    const fPrime = 8 * x ** 4 / Math.sqrt(1 + x * x);
    const dxdr = -GRAVITATIONAL_CONSTANT_CGS * m * rho / (r * r * A_EOS * fPrime);
    const scale = Math.min(Math.abs(x / dxdr), r);
    const dr = Math.max(0.002 * scale, 1e3);
    // RK2 (midpoint).
    const xm = x + 0.5 * dr * dxdr;
    if (xm <= 0) {
      break;
    }
    const rm = r + 0.5 * dr;
    const mm = m + 0.5 * dr * 4 * Math.PI * r * r * rho;
    const rhom = B * xm ** 3;
    const fPrimeM = 8 * xm ** 4 / Math.sqrt(1 + xm * xm);
    const dxdrM = -GRAVITATIONAL_CONSTANT_CGS * mm * rhom / (rm * rm * A_EOS * fPrimeM);
    const xNext = x + dr * dxdrM;
    m += dr * 4 * Math.PI * rm * rm * rhom;
    r += dr;
    if (xNext <= 0) {
      radii.push(r);
      masses.push(m);
      densities.push(0);
      x = 0;
      break;
    }
    x = xNext;
    radii.push(r);
    masses.push(m);
    densities.push(B * x ** 3);
  }
  return { radii, masses, densities };
}

/**
 * @param {number} massSolar target total mass (Msun), below the Chandrasekhar limit
 * @param {number} [muE] electrons per nucleon^-1 (2 for C/O or He)
 */
export function solveWhiteDwarfStructure(massSolar, muE = 2) {
  if (!(massSolar > 0.05 && massSolar < 1.42)) {
    throw new RangeError("White-dwarf structure mass must lie in (0.05, 1.42) Msun");
  }
  const target = massSolar * SOLAR_MASS_G;
  let low = Math.log(1e4);
  let high = Math.log(5e10);
  let model = null;
  for (let iteration = 0; iteration < 80; iteration += 1) {
    const mid = 0.5 * (low + high);
    model = integrate(Math.exp(mid), muE, 200000);
    const mass = model.masses[model.masses.length - 1];
    if (mass < target) {
      low = mid;
    } else {
      high = mid;
    }
    if (Math.abs(mass / target - 1) < 1e-6) {
      break;
    }
  }
  const radii = Float64Array.from(model.radii, (r) => r / 1e5);
  const masses = Float64Array.from(model.masses, (m) => m / SOLAR_MASS_G);
  return Object.freeze({
    massSolar: masses[masses.length - 1],
    radiusKm: radii[radii.length - 1],
    centralDensity: model.densities[0],
    radiiKm: radii,
    enclosedMassSolar: masses,
    density: Float64Array.from(model.densities),
  });
}
