// Close-binary geometry for mass-transferring white-dwarf systems.
//
// Units: masses in solar masses, lengths in km, times in s. The orbit is
// circular in the x-z plane with angular momentum along +y (the convention of
// the binary scenes). In the co-rotating frame the primary sits at
// x = -a q/(1+q), the donor at x = +a/(1+q).

import {
  GRAVITATIONAL_CONSTANT_CGS,
  SOLAR_MASS_G,
  SOLAR_RADIUS_KM,
  SPEED_OF_LIGHT_CM_S,
} from "./physical-constants.js";

const G_KM3_PER_MSUN_S2 = GRAVITATIONAL_CONSTANT_CGS * SOLAR_MASS_G / 1e15;
const CHANDRASEKHAR_MASS = 1.44;
const MASS_P = 0.00057;

// Zero-temperature white-dwarf mass-radius relation (Eggleton, quoted by
// Verbunt & Rappaport 1988, ApJ 332, 193).
export function whiteDwarfRadiusKm(massSolar) {
  if (!(massSolar > 0 && massSolar < CHANDRASEKHAR_MASS)) {
    throw new RangeError("White-dwarf mass must lie in (0, 1.44) Msun");
  }
  const ratio = massSolar / CHANDRASEKHAR_MASS;
  const degenerate = Math.sqrt(ratio ** (-2 / 3) - ratio ** (2 / 3));
  const correction = (1 + 3.5 * (massSolar / MASS_P) ** (-2 / 3) + MASS_P / massSolar) ** (-2 / 3);
  return 0.0114 * degenerate * correction * SOLAR_RADIUS_KM;
}

// Eggleton (1983) volume-equivalent Roche-lobe radius over separation for the
// star of mass ratio q = M_star / M_other.
export function eggletonRocheLobe(q) {
  const q13 = Math.cbrt(q);
  const q23 = q13 * q13;
  return 0.49 * q23 / (0.6 * q23 + Math.log(1 + q13));
}

export function keplerAngularFrequency(totalMassSolar, separationKm) {
  return Math.sqrt(G_KM3_PER_MSUN_S2 * totalMassSolar / separationKm ** 3);
}

// Peters (1964) circular-orbit inspiral time from separation a.
export function gravitationalInspiralTimeS(massA, massB, separationKm) {
  const c = SPEED_OF_LIGHT_CM_S / 1e5;
  const g = G_KM3_PER_MSUN_S2;
  return (5 / 256) * c ** 5 * separationKm ** 4 / (g ** 3 * massA * massB * (massA + massB));
}

// Dimensionless Roche potential in units of G (M1 + M2) / a with lengths in
// units of a, co-rotating frame, primary (mass fraction 1 - mu) at
// x = -mu, donor (mass fraction mu) at x = 1 - mu, rotation about y.
export function rochePotential(x, y, z, mu) {
  const r1 = Math.hypot(x + mu, y, z);
  const r2 = Math.hypot(x - 1 + mu, y, z);
  return -(1 - mu) / r1 - mu / r2 - 0.5 * (x * x + z * z);
}

// Inner Lagrange point L1 on the x axis (lengths in units of a).
export function innerLagrangePoint(mu) {
  const force = (x) => (
    -(1 - mu) * (x + mu) / Math.abs(x + mu) ** 3
    - mu * (x - 1 + mu) / Math.abs(x - 1 + mu) ** 3
    + x
  );
  let low = -mu + 1e-9;
  let high = 1 - mu - 1e-9;
  // force > 0 near the primary side when pulled towards the donor; bisect.
  let fLow = force(low);
  for (let iteration = 0; iteration < 200; iteration += 1) {
    const mid = 0.5 * (low + high);
    const fMid = force(mid);
    if ((fMid > 0) === (fLow > 0)) {
      low = mid;
      fLow = fMid;
    } else {
      high = mid;
    }
  }
  return 0.5 * (low + high);
}

/**
 * Ballistic mass-transfer stream from L1 in the co-rotating frame (Lubow &
 * Shu 1975): a test particle released at L1 with a small sonic offset,
 * integrated with RK4 under gravity, centrifugal and Coriolis forces until it
 * reaches the accretor's surface. Returns points in units of a (x, y, z) and
 * the dimensionless speed in the co-rotating frame.
 */
export function ballisticStream({
  mu,
  accretorRadius,
  soundSpeed = 0.02,
  points = 64,
  maximumTime = 6,
}) {
  const l1 = innerLagrangePoint(mu);
  const primaryX = -mu;
  const donorX = 1 - mu;
  function acceleration(state) {
    const [x, y, z, vx, vy, vz] = state;
    const d1 = Math.hypot(x - primaryX, y, z);
    const d2 = Math.hypot(x - donorX, y, z);
    const g1 = (1 - mu) / d1 ** 3;
    const g2 = mu / d2 ** 3;
    // Rotation vector along +y (unit angular frequency): centrifugal
    // Omega^2 (x, 0, z); Coriolis -2 Omega x v = -2 (vz, 0, -vx).
    const ax = -g1 * (x - primaryX) - g2 * (x - donorX) + x - 2 * vz;
    const ay = -g1 * y - g2 * y;
    const az = -g1 * z - g2 * z + z + 2 * vx;
    return [vx, vy, vz, ax, ay, az];
  }
  let state = [l1 - 1e-4, 0, 0, -soundSpeed, 0, 0];
  const dt = 1e-3;
  const trajectory = [];
  let time = 0;
  while (time < maximumTime) {
    const d1 = Math.hypot(state[0] - primaryX, state[1], state[2]);
    trajectory.push([state[0], state[1], state[2], Math.hypot(state[3], state[4], state[5]), time]);
    if (d1 <= accretorRadius) {
      break;
    }
    const k1 = acceleration(state);
    const k2 = acceleration(state.map((value, i) => value + 0.5 * dt * k1[i]));
    const k3 = acceleration(state.map((value, i) => value + 0.5 * dt * k2[i]));
    const k4 = acceleration(state.map((value, i) => value + dt * k3[i]));
    state = state.map((value, i) => value + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    time += dt;
  }
  // Resample uniformly in arc length.
  const lengths = [0];
  for (let i = 1; i < trajectory.length; i += 1) {
    const [x0, y0, z0] = trajectory[i - 1];
    const [x1, y1, z1] = trajectory[i];
    lengths.push(lengths[i - 1] + Math.hypot(x1 - x0, y1 - y0, z1 - z0));
  }
  const total = lengths[lengths.length - 1];
  const resampled = [];
  let cursor = 0;
  for (let k = 0; k < points; k += 1) {
    const target = total * k / (points - 1);
    while (cursor < lengths.length - 2 && lengths[cursor + 1] < target) cursor += 1;
    const span = Math.max(lengths[cursor + 1] - lengths[cursor], 1e-12);
    const f = Math.min(Math.max((target - lengths[cursor]) / span, 0), 1);
    const a = trajectory[cursor];
    const b = trajectory[cursor + 1] ?? a;
    resampled.push([
      a[0] + f * (b[0] - a[0]),
      a[1] + f * (b[1] - a[1]),
      a[2] + f * (b[2] - a[2]),
      a[3] + f * (b[3] - a[3]),
      a[4] + f * (b[4] - a[4]),
    ]);
  }
  const last = trajectory[trajectory.length - 1];
  return Object.freeze({
    l1,
    points: resampled,
    reachedAccretor: Math.hypot(last[0] - primaryX, last[1], last[2]) <= accretorRadius * 1.0001,
    arcLength: total,
  });
}
