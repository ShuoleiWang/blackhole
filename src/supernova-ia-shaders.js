// WebGPU tracer for the Type Ia supernova scene: a double-degenerate binary
// (Roche-lobe-filling donor, ballistic stream, accretor with an impact hot
// spot), the helium-shell and core detonation breakouts, and the homologous
// radioactively powered ejecta. Space is flat (the white dwarfs' surface
// potential is GM/(Rc^2) ~ 3e-4); relativistic Doppler and light-travel time
// of the 0.03-0.1c ejecta are kept. Radiance is cd/m^2 before exposure.

import { CAMERA_GLARE_STRENGTH, GLARE_OBSERVER } from "./glare.js";
import { blackbodyColorTable } from "./transients/blackbody-color.js";
import { SUPERNOVA_IA_CANONICAL_MODEL } from "./transients/supernova-ia-model.js";
import { transientFallbackUniforms, TRANSIENT_SPHERES_FALLBACK_GLSL } from "./transients/transient-fallback.js";
import { createTransientTableResources } from "./transients/transient-resources.js";
import {
  TRANSIENT_BASE_WGSL,
  TRANSIENT_COMMON_WGSL,
  TRANSIENT_PARAMS_PREFIX_WGSL,
  transientBlackbodyConstantsWGSL,
  transientEjectaWGSL,
} from "./transients/transient-wgsl.js";

// WGSL float literal (always carries a decimal point).
function wgslFloat(value) {
  const [mantissa, exponent] = Number(value).toExponential().split("e");
  return `${mantissa.includes(".") ? mantissa : `${mantissa}.0`}e${Number(exponent)}`;
}

export const SUPERNOVA_IA_STREAM_POINTS = 16;
// Moving photographic frames trace one node per 4x4 pixels first and
// interpolate radiance where it is provably smooth.
export const SUPERNOVA_IA_SPARSE_STRIDE = 4;
export const SUPERNOVA_IA_EJECTA_BINS = 64;

// vec4 slots after the 36-float renderer prefix (slot 0 = sceneScale).
const SLOT_NAMES = Object.freeze([
  "sceneScale",
  "quality",
  "steps",
  "primaryPositionRadius",
  "primaryState",
  "hotSpot",
  "donorPositionRadius",
  "donorState",
  "orbit",
  "donorMotion",
  "donorShock",
  "irradiation",
  "streamState",
  ...Array.from({ length: SUPERNOVA_IA_STREAM_POINTS }, (_, index) => `streamPoint${index}`),
  "explosion",
  "explosionFronts",
  "coreIgnition",
  "explosionSurface",
  "fireball",
  "ejectaCentre",
  "ejectaDrift",
  "ejectaAxis",
  "ejectaCone",
  "ejectaZones",
  "ejectaPhotosphere",
  "ejectaAsymmetry",
  "ejectaBounds",
  ...Array.from({ length: SUPERNOVA_IA_EJECTA_BINS }, (_, index) => `ejectaProfile${index}`),
  ...Array.from({ length: SUPERNOVA_IA_EJECTA_BINS }, (_, index) => `ejectaRadiative${index}`),
]);

export const SUPERNOVA_IA_UNIFORM_LAYOUT = Object.freeze(Object.fromEntries(
  SLOT_NAMES.map((name, index) => [name, 36 + 4 * index]),
));
export const SUPERNOVA_IA_UNIFORM_FLOATS = 36 + 4 * SLOT_NAMES.length;

export const SUPERNOVA_IA_BLACKBODY_TABLE = blackbodyColorTable({
  minLog10Temperature: 2.5,
  maxLog10Temperature: 9.5,
  entries: 512,
});

const SUPERNOVA_IA_WGSL = /* wgsl */ `
struct Params {
${TRANSIENT_PARAMS_PREFIX_WGSL}
  // accumulation index, weight, epoch, reset
  quality: vec4<f32>,
  // ejecta steps, optical-depth step, surface bisection iterations, mode
  steps: vec4<f32>,
  primaryPositionRadius: vec4<f32>,
  // T_eff (K), linear limb darkening u, visible (0/1), hot-spot T (K)
  primaryState: vec4<f32>,
  // hot-spot centre direction (co-rotating frame), cos(angular radius)
  hotSpot: vec4<f32>,
  // donor centre (scene units), bounding radius
  donorPositionRadius: vec4<f32>,
  // T_pole (K), donor mass fraction mu, separation a (scene units), Roche L1 potential
  donorState: vec4<f32>,
  // cos phase, sin phase, gravity-darkening exponent, Roche shape (1) or relaxed sphere (0)
  orbit: vec4<f32>,
  // |grad Phi| at the donor pole (Roche units), visible, relaxed-sphere radius, L1 x (units of a)
  donorMotion: vec4<f32>,
  // ejecta strike time (s after helium ignition), facing-hemisphere flash
  // temperature (K), post-shock temperature (K), reserved
  donorShock: vec4<f32>,
  // impact-spot radiant intensity L/pi along its normal (1e36 erg/s/sr; 0
  // after ignition), absorbed fraction of incident flux, accretor luminosity
  // (1e36 erg/s, as a point source), reserved
  irradiation: vec4<f32>,
  // stream floor temperature (K, gas leaving L1), central absorption (per
  // unit a), point count, visible
  streamState: vec4<f32>,
  streamPoints: array<vec4<f32>, ${SUPERNOVA_IA_STREAM_POINTS}>,
  // helium ignition direction (world), time since helium ignition (s)
  explosion: vec4<f32>,
  // He-front angular speed (rad/s), core ignition time (s), core front speed (scene units/s), active
  explosionFronts: vec4<f32>,
  // core ignition point (world, scene units), ejecta clock origin (s after
  // helium ignition)
  coreIgnition: vec4<f32>,
  // acceleration time (s), He-kick speed and ejecta edge speed vmax (scene
  // units/s), accretor radius (scene units); the core kick of a surface
  // element is vmax s(n) - v_He with the ejecta velocity stretch s(n)
  explosionSurface: vec4<f32>,
  // fireball photospheric temperature (K) at this time, reserved
  fireball: vec4<f32>,
  // ejecta centre at the reference time (scene units), reference time since explosion (s)
  ejectaCentre: vec4<f32>,
  // ejecta drift (scene units/s), maximum velocity (km/s)
  ejectaDrift: vec4<f32>,
  // shadow-cone axis (unit), reference log10 t
  ejectaAxis: vec4<f32>,
  // cos cone half-angle, density factor inside, cos edge width, active
  ejectaCone: vec4<f32>,
  // composition boundaries as v/vmax: end of the 56Ni core, end of the IME
  // layer, start of the helium-detonation ash; reserved
  ejectaZones: vec4<f32>,
  // photospheric T_eff (K), colour/effective temperature ratio, nebular
  // colour temperature (K), shadow-cone rim density factor
  ejectaPhotosphere: vec4<f32>,
  // impact (helium-ignition) axis (unit), velocity stretch amplitude a in
  // s(theta) = 1 + a cos(theta) + b; b in ejectaZones.w
  ejectaAsymmetry: vec4<f32>,
  // active velocity (km/s), reserved
  ejectaBounds: vec4<f32>,
  ejectaProfile: array<vec4<f32>, ${SUPERNOVA_IA_EJECTA_BINS}>,
  ejectaRadiative: array<vec4<f32>, ${SUPERNOVA_IA_EJECTA_BINS}>,
};

struct FragmentInput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tSky: texture_2d<f32>;
@group(0) @binding(2) var skySampler: sampler;
// Sparse radiance nodes (see fsCoarse); bound only when the scene traces
// sparsely.
@group(0) @binding(3) var coarseField: texture_2d<f32>;
@group(0) @binding(4) var blackbodyTable: texture_2d<f32>;

const SPARSE_STRIDE: f32 = ${SUPERNOVA_IA_SPARSE_STRIDE.toFixed(1)};
// A pixel is interpolated only if the bilinear error bound (second
// difference / 8) stays below 0.4% of the local display value, or below an
// absolute floor of 0.004 (1/250 of display white) in dark regions.
const SPARSE_RELATIVE_TOLERANCE: f32 = 0.004;
const SPARSE_ABSOLUTE_FLOOR: f32 = 0.004;

${transientBlackbodyConstantsWGSL(SUPERNOVA_IA_BLACKBODY_TABLE)}
${TRANSIENT_BASE_WGSL}${TRANSIENT_COMMON_WGSL}

fn ejectaDensityFactor(direction: vec3<f32>) -> f32 {
  if (params.ejectaCone.w < 0.5) {
    return 1.0;
  }
  // The surviving companion shields a cone behind it (Tanikawa et al. 2018;
  // Kasen 2010): a partially refilled hole with a shocked, denser rim.
  let cosine = dot(direction, params.ejectaAxis.xyz);
  let width = max(params.ejectaCone.z, 1.0e-4);
  let inside = smoothstep(params.ejectaCone.x - width, params.ejectaCone.x + width, cosine);
  let edge = (cosine - params.ejectaCone.x) / width;
  let rim = (params.ejectaPhotosphere.w - 1.0) * exp(-edge * edge);
  return mix(1.0, params.ejectaCone.y, inside) + rim;
}

// One polar sector: the photosphere has a single effective temperature.
fn ejectaEffectiveTemperature(sector: f32) -> f32 {
  return params.ejectaPhotosphere.x;
}

// Material launched towards the helium-ignition (impact) side ends ~20%
// faster, towards the antipode ~15% slower (Boos et al. 2021; Shen et al.
// 2021): s(theta) = 1 + a cos(theta) + b.
fn ejectaVelocityStretch(direction: vec3<f32>) -> f32 {
  return 1.0 + params.ejectaAsymmetry.w * dot(direction, params.ejectaAsymmetry.xyz) + params.ejectaZones.w;
}

${transientEjectaWGSL({ bins: SUPERNOVA_IA_EJECTA_BINS, sectors: 1 })}

struct SurfaceHit {
  distance: f32,
  radiance: vec3<f32>,
  log10Temperature: f32,
  // 0 none, 1 accretor, 2 donor
  kind: i32,
};

fn noHit() -> SurfaceHit {
  var hit: SurfaceHit;
  hit.distance = 1.0e30;
  hit.radiance = vec3<f32>(0.0);
  hit.log10Temperature = 0.0;
  hit.kind = 0;
  return hit;
}

fn limbDarkening(mu: f32, coefficient: f32) -> f32 {
  return 1.0 - coefficient * (1.0 - clamp(mu, 0.0, 1.0));
}

// World direction into the co-rotating frame (rotation about +y by -phase).
fn toCorotating(v: vec3<f32>) -> vec3<f32> {
  let c = params.orbit.x;
  let s = params.orbit.y;
  return vec3<f32>(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
}

fn fromCorotating(v: vec3<f32>) -> vec3<f32> {
  let c = params.orbit.x;
  let s = params.orbit.y;
  return vec3<f32>(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

// ---- Accretor before the explosion ---------------------------------------
fn shadeAccretorQuiescent(ray: Ray, exposureLog2: f32) -> SurfaceHit {
  var hit = noHit();
  let centre = params.primaryPositionRadius.xyz;
  let interval = sphereInterval(ray, centre, params.primaryPositionRadius.w);
  if (interval.x >= interval.y) {
    return hit;
  }
  let position = ray.origin + interval.x * ray.direction;
  let normal = safeNormalize(position - centre);
  let mu = dot(normal, -ray.direction);
  var temperature = params.primaryState.x;
  // Stream impact hot spot, fixed in the co-rotating frame.
  let spot = dot(toCorotating(normal), params.hotSpot.xyz);
  let spotWeight = smoothstep(params.hotSpot.w - 0.02, min(params.hotSpot.w + 0.02, 1.0), spot);
  temperature = mix(temperature, params.primaryState.w, spotWeight);
  hit.distance = interval.x;
  hit.radiance = exposedBlackbody(temperature, exposureLog2) * limbDarkening(mu, params.primaryState.y);
  hit.log10Temperature = log2(temperature) * 0.30102999566;
  hit.kind = 1;
  return hit;
}

// ---- Accretor during the detonations -------------------------------------
// Surface burn times: the helium detonation runs along the shell from the
// ignition point at angular speed omega; the core detonation starts at the
// converging-shock point and reaches the surface point after |R n - x_C| / v_C.
fn surfaceBurnTimes(direction: vec3<f32>) -> vec2<f32> {
  let angle = acos(clamp(dot(direction, params.explosion.xyz), -1.0, 1.0));
  let heliumTime = angle / max(params.explosionFronts.x, 1.0e-6);
  let surfacePoint = params.primaryPositionRadius.xyz + direction * params.explosionSurface.w;
  let coreTime = params.explosionFronts.y
    + length(surfacePoint - params.coreIgnition.xyz) / max(params.explosionFronts.z, 1.0e-9);
  return vec2<f32>(heliumTime, coreTime);
}

// Displacement after a kick at speed v applied at elapsed time s, reaching
// v over the acceleration time tau: v (s - tau (1 - exp(-s/tau))).
fn kickDisplacement(elapsed: f32, speed: f32) -> f32 {
  if (elapsed <= 0.0) {
    return 0.0;
  }
  let tau = max(params.explosionSurface.x, 1.0e-4);
  return speed * (elapsed - tau * (1.0 - exp(-elapsed / tau)));
}

// The core detonation drives each surface element to the stretched ejecta
// edge, so the fireball has the ejecta's shape at the hand-off.
fn coreKickSpeed(direction: vec3<f32>) -> f32 {
  return max(params.explosionSurface.z * ejectaVelocityStretch(direction) - params.explosionSurface.y, 0.0);
}

fn explodingSurfaceRadius(direction: vec3<f32>) -> f32 {
  let times = surfaceBurnTimes(direction);
  let t = params.explosion.w;
  return params.explosionSurface.w
    + kickDisplacement(t - times.x, params.explosionSurface.y)
    + kickDisplacement(t - times.y, coreKickSpeed(direction));
}

// Photospheric temperature of a surface element (Piro, Chang & Weinberg
// 2010 scalings; constants from SUPERNOVA_IA_CANONICAL_MODEL): the helium
// front leaves a thin brilliant arc cooling as T = T_He (dt / 0.01 s)^-1/2;
// the core-detonation breakout flashes at T_core for ~0.01 s and fades as
// (dt / 0.01 s)^-1/3 e^(-dt / t_flash), after which the fireball has the
// model's photospheric temperature (fireball.x). A rise over the
// light-crossing time R/c keeps the fronts resolved.
const HELIUM_FLASH_TEMPERATURE: f32 = ${wgslFloat(SUPERNOVA_IA_CANONICAL_MODEL.heliumFlashTemperatureK)};
const CORE_FLASH_TEMPERATURE: f32 = ${wgslFloat(SUPERNOVA_IA_CANONICAL_MODEL.coreFlashTemperatureK)};
const FLASH_DECAY_TIME: f32 = ${wgslFloat(SUPERNOVA_IA_CANONICAL_MODEL.flashDecayS)};
fn explodingSurfaceTemperature(direction: vec3<f32>, radius: f32) -> f32 {
  let times = surfaceBurnTimes(direction);
  let t = params.explosion.w;
  var temperature = params.primaryState.x;
  let riseTime = max(params.explosionSurface.w / params.sceneScale.w, 1.0e-4);
  if (t > times.x) {
    let elapsed = max(t - times.x, 0.01);
    let rise = smoothstep(0.0, riseTime, t - times.x);
    temperature = max(temperature, rise * HELIUM_FLASH_TEMPERATURE * inverseSqrt(elapsed / 0.01));
  }
  if (t > times.y) {
    let elapsed = max(t - times.y, 0.01);
    let rise = smoothstep(0.0, riseTime, t - times.y);
    let flash = CORE_FLASH_TEMPERATURE * pow(elapsed / 0.01, -0.3333333) * exp(-elapsed / FLASH_DECAY_TIME);
    temperature = max(temperature, rise * max(flash, params.fireball.x));
  }
  return temperature;
}

fn shadeAccretorExploding(ray: Ray, exposureLog2: f32) -> SurfaceHit {
  var hit = noHit();
  let centre = params.primaryPositionRadius.xyz;
  let t = params.explosion.w;
  let maximumCoreKick = params.explosionSurface.z
    * (1.0 + abs(params.ejectaAsymmetry.w) + max(params.ejectaZones.w, 0.0)) - params.explosionSurface.y;
  let maximumRadius = params.explosionSurface.w
    + kickDisplacement(t, params.explosionSurface.y)
    + kickDisplacement(t - params.explosionFronts.y, maximumCoreKick);
  let interval = sphereInterval(ray, centre, maximumRadius * 1.001);
  if (interval.x >= interval.y) {
    return hit;
  }
  // Star-shaped surface r = R(n): march for the first sign change of
  // |x - c| - R(n), then bisect.
  let marchSteps = 32;
  let iterations = i32(params.steps.z);
  var previousS = interval.x;
  var previousF = 1.0;
  var found = false;
  var low = 0.0;
  var high = 0.0;
  for (var index: i32 = 0; index <= marchSteps; index = index + 1) {
    let s = interval.x + (interval.y - interval.x) * f32(index) / f32(marchSteps);
    let offset = ray.origin + s * ray.direction - centre;
    let radius = length(offset);
    let f = radius - explodingSurfaceRadius(offset / max(radius, 1.0e-9));
    if (index > 0 && previousF > 0.0 && f <= 0.0) {
      low = previousS;
      high = s;
      found = true;
      break;
    }
    previousS = s;
    previousF = f;
  }
  if (!found) {
    return hit;
  }
  for (var index: i32 = 0; index < iterations; index = index + 1) {
    let mid = 0.5 * (low + high);
    let offset = ray.origin + mid * ray.direction - centre;
    let radius = length(offset);
    if (radius - explodingSurfaceRadius(offset / max(radius, 1.0e-9)) > 0.0) {
      low = mid;
    } else {
      high = mid;
    }
  }
  let s = 0.5 * (low + high);
  let offset = ray.origin + s * ray.direction - centre;
  let radius = length(offset);
  let direction = offset / max(radius, 1.0e-9);
  // Surface normal from the gradient of |x - c| - R(n).
  let tangentA = safeNormalize(cross(direction, select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(direction.y) > 0.9)));
  let tangentB = cross(direction, tangentA);
  let epsilon = 0.01;
  let dA = (explodingSurfaceRadius(safeNormalize(direction + epsilon * tangentA))
    - explodingSurfaceRadius(safeNormalize(direction - epsilon * tangentA))) / (2.0 * epsilon * radius);
  let dB = (explodingSurfaceRadius(safeNormalize(direction + epsilon * tangentB))
    - explodingSurfaceRadius(safeNormalize(direction - epsilon * tangentB))) / (2.0 * epsilon * radius);
  let normal = safeNormalize(direction - dA * tangentA - dB * tangentB);
  let mu = dot(normal, -ray.direction);
  let temperature = explodingSurfaceTemperature(direction, radius);
  // The breakout layer moves outward; its Doppler factor is kept.
  let times = surfaceBurnTimes(direction);
  let tau = max(params.explosionSurface.x, 1.0e-4);
  let heliumSpeed = select(0.0, params.explosionSurface.y * (1.0 - exp(-(t - times.x) / tau)), t > times.x);
  let coreSpeed = select(0.0, coreKickSpeed(direction) * (1.0 - exp(-(t - times.y) / tau)), t > times.y);
  let beta = direction * (heliumSpeed + coreSpeed) / params.sceneScale.w;
  let doppler = dopplerFactor(beta, -ray.direction);
  hit.distance = s;
  hit.radiance = exposedBlackbody(doppler * temperature, exposureLog2) * limbDarkening(mu, params.primaryState.y);
  hit.log10Temperature = log2(doppler * temperature) * 0.30102999566;
  hit.kind = 1;
  return hit;
}

// ---- Irradiation -----------------------------------------------------------
// Incidence factor of a source of angular radius asin(sinAlpha): cos(theta)
// while the whole source is above the local horizon, zero once it has set,
// and a quadratic blend (continuous in value and slope) across the penumbra.
fn penumbraIncidence(cosine: f32, sinAlpha: f32) -> f32 {
  if (cosine >= sinAlpha) {
    return cosine;
  }
  if (cosine <= -sinAlpha) {
    return 0.0;
  }
  let x = cosine + sinAlpha;
  return x * x / (4.0 * max(sinAlpha, 1.0e-4));
}

// Flux (units of 1e18 erg s^-1 cm^-2, float32 range) falling on a surface
// element at position with the given normal: the impact spot as a
// Lambertian emitter (radiant intensity I0 cos(theta) along its normal) and
// the accretor as a point source of its current luminosity. For a thin tube
// the flux is averaged over its circumference (1/pi of the normal-incidence
// flux) instead of using a normal.
fn irradiationFlux(position: vec3<f32>, normal: vec3<f32>, tube: bool) -> f32 {
  // Scene units -> 1e9 cm; 1e36 erg/s over (1e9 cm)^2 is 1e18 erg/s/cm^2.
  let gigacmPerUnit = params.sceneScale.x * 1.0e-4;
  let accretorCentre = params.primaryPositionRadius.xyz;
  var flux = 0.0;
  if (params.irradiation.z > 0.0) {
    let toAccretor = accretorCentre - position;
    let distance = max(length(toAccretor), params.primaryPositionRadius.w);
    let sinAlpha = min(params.primaryPositionRadius.w / distance, 1.0);
    let incidence = select(penumbraIncidence(dot(normal, toAccretor) / distance, sinAlpha), 0.3183099, tube);
    if (incidence > 0.0) {
      let distanceGcm = distance * gigacmPerUnit;
      flux = flux + params.irradiation.z / (4.0 * PI * distanceGcm * distanceGcm) * incidence;
    }
  }
  if (params.irradiation.x > 0.0) {
    let spotNormal = fromCorotating(params.hotSpot.xyz);
    let toPoint = position - (accretorCentre + spotNormal * params.primaryPositionRadius.w);
    let distance = max(length(toPoint), 1.0e-6);
    let direction = toPoint / distance;
    let emission = dot(spotNormal, direction);
    // Spot radius R1 sin(angular radius), from its cosine in hotSpot.w.
    let spotRadius = params.primaryPositionRadius.w * sqrt(max(1.0 - params.hotSpot.w * params.hotSpot.w, 0.0));
    let incidence = select(penumbraIncidence(-dot(normal, direction), min(spotRadius / distance, 1.0)), 0.3183099, tube);
    if (emission > 0.0 && incidence > 0.0) {
      let distanceGcm = distance * gigacmPerUnit;
      flux = flux + params.irradiation.x * emission * incidence / (distanceGcm * distanceGcm);
    }
  }
  return flux;
}

// Re-radiated temperature: T^4 = T_int^4 + f_abs F / sigma, with F in
// 1e18 erg s^-1 cm^-2 (1e18 / sigma = 1.7635e22 K^4).
fn irradiatedTemperature(intrinsic: f32, flux: f32) -> f32 {
  let t2 = intrinsic * intrinsic;
  return sqrt(sqrt(t2 * t2 + params.irradiation.y * flux * 1.7635e22));
}

// ---- Roche-lobe-filling donor --------------------------------------------
// Dimensionless Roche potential (units of G M / a, lengths in a) in the
// co-rotating frame: accretor (1 - mu) at x = -mu, donor mu at x = 1 - mu.
fn rochePotential(p: vec3<f32>) -> f32 {
  let mu = params.donorState.y;
  let r1 = length(p - vec3<f32>(-mu, 0.0, 0.0));
  let r2 = length(p - vec3<f32>(1.0 - mu, 0.0, 0.0));
  return -(1.0 - mu) / r1 - mu / r2 - 0.5 * (p.x * p.x + p.z * p.z);
}

fn rocheGradient(p: vec3<f32>) -> vec3<f32> {
  let mu = params.donorState.y;
  let d1 = p - vec3<f32>(-mu, 0.0, 0.0);
  let d2 = p - vec3<f32>(1.0 - mu, 0.0, 0.0);
  let r1 = length(d1);
  let r2 = length(d2);
  return (1.0 - mu) * d1 / (r1 * r1 * r1) + mu * d2 / (r2 * r2 * r2) - vec3<f32>(p.x, 0.0, p.z);
}

// Released donor after the ejecta strike: the explosion surface reaches the
// sub-primary point at donorShock.x and sweeps to the limb at the combined
// helium-ash and core-ash speed; the struck hemisphere flashes as
// T_flash (dt / 0.01 s)^-1/2 and settles at the post-shock temperature, while
// the shadowed hemisphere stays quiescent. Flux (T^4) blends across the
// terminator.
fn shockedDonorTemperature(normal: vec3<f32>) -> f32 {
  let quiescent = params.donorState.x;
  let toPrimary = safeNormalize(params.primaryPositionRadius.xyz - params.donorPositionRadius.xyz);
  let cosFacing = dot(normal, toPrimary);
  let delay = params.donorMotion.z * (1.0 - cosFacing)
    / max(params.explosionSurface.y + params.explosionSurface.z, 1.0e-9);
  let sinceStrike = params.explosion.w - params.donorShock.x - delay;
  let facing = smoothstep(-0.15, 0.35, cosFacing);
  if (sinceStrike <= 0.0 || facing <= 0.0) {
    return quiescent;
  }
  let flash = params.donorShock.y * inverseSqrt(max(sinceStrike, 0.01) / 0.01);
  let shocked = max(max(flash, params.donorShock.z), quiescent);
  let ratio = quiescent / shocked;
  let ratio4 = ratio * ratio * ratio * ratio;
  return shocked * pow(mix(ratio4, 1.0, facing), 0.25);
}

fn shadeDonor(ray: Ray, exposureLog2: f32) -> SurfaceHit {
  var hit = noHit();
  if (params.donorMotion.y < 0.5) {
    return hit;
  }
  let centre = params.donorPositionRadius.xyz;
  if (params.orbit.w < 0.5) {
    // After the explosion the released donor relaxes to a sphere.
    let interval = sphereInterval(ray, centre, params.donorMotion.z);
    if (interval.x >= interval.y) {
      return hit;
    }
    let position = ray.origin + interval.x * ray.direction;
    let normal = safeNormalize(position - centre);
    let mu = dot(normal, -ray.direction);
    let temperature = irradiatedTemperature(shockedDonorTemperature(normal), irradiationFlux(position, normal, false));
    hit.distance = interval.x;
    hit.radiance = exposedBlackbody(temperature, exposureLog2) * limbDarkening(mu, params.primaryState.y);
    hit.log10Temperature = log2(temperature) * 0.30102999566;
    hit.kind = 2;
    return hit;
  }
  let interval = sphereInterval(ray, centre, params.donorPositionRadius.w);
  if (interval.x >= interval.y) {
    return hit;
  }
  let separation = params.donorState.z;
  let potentialL1 = params.donorState.w;
  let mu = params.donorState.y;
  // March in the co-rotating frame (units of a) from the bounding-sphere
  // entry; the donor is the region x > x_L1 with Phi < Phi_L1. After the
  // release the Roche figure moves with the donor until the ejecta strike it.
  let rocheShift = centre - fromCorotating(vec3<f32>((1.0 - mu) * separation, 0.0, 0.0));
  let origin = toCorotating(ray.origin - rocheShift) / separation;
  let direction = toCorotating(ray.direction);
  var s = interval.x / separation;
  let sEnd = interval.y / separation;
  let lagrangeX = params.donorMotion.w;
  var inside = false;
  var sPrevious = s;
  for (var index: i32 = 0; index < 48; index = index + 1) {
    let p = origin + s * direction;
    let f = rochePotential(p) - potentialL1;
    if (f < 0.0 && p.x > lagrangeX) {
      inside = true;
      break;
    }
    // |f| / |grad Phi| is a conservative distance estimate to the level set.
    let gradient = length(rocheGradient(p));
    sPrevious = s;
    s = s + clamp(abs(f) / max(gradient, 1.0e-4), 2.0e-4, 0.05);
    if (s > sEnd) {
      break;
    }
  }
  if (!inside) {
    return hit;
  }
  var low = sPrevious;
  var high = s;
  for (var index: i32 = 0; index < 12; index = index + 1) {
    let mid = 0.5 * (low + high);
    if (rochePotential(origin + mid * direction) - potentialL1 < 0.0) {
      high = mid;
    } else {
      low = mid;
    }
  }
  let p = origin + high * direction;
  let gradient = rocheGradient(p);
  let gravity = length(gradient);
  let normal = fromCorotating(gradient / max(gravity, 1.0e-9));
  let cosView = dot(normal, -ray.direction);
  // von Zeipel gravity darkening, T ~ g^beta, plus the re-radiated part of
  // the light from the impact spot and the accretor.
  let intrinsic = params.donorState.x
    * pow(max(gravity / max(params.donorMotion.x, 1.0e-9), 1.0e-3), params.orbit.z);
  let position = ray.origin + high * separation * ray.direction;
  let temperature = irradiatedTemperature(intrinsic, irradiationFlux(position, normal, false));
  hit.distance = high * separation;
  hit.radiance = exposedBlackbody(temperature, exposureLog2) * limbDarkening(cosView, params.primaryState.y);
  hit.log10Temperature = log2(temperature) * 0.30102999566;
  hit.kind = 2;
  return hit;
}

// ---- Mass-transfer stream -------------------------------------------------
// A thin tube along the ballistic trajectory (co-rotating frame, units of a)
// with a Gaussian cross-section of half-width w; each segment adds its
// integrated optical depth tau = alpha_0 w sqrt(pi) / sin(theta) exp(-d^2/w^2).
// The gas leaves L1 at the donor's temperature and re-radiates the absorbed
// light of the impact spot and the accretor, capped at the spot temperature;
// segments rarely overlap along a ray, so their emission is combined with the
// total opacity.
fn streamEmission(ray: Ray, limit: f32, exposureLog2: f32) -> vec4<f32> {
  if (params.streamState.w < 0.5) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let separation = params.donorState.z;
  // The stream is fixed in the co-rotating frame about the centre of mass
  // (the scene origin).
  let origin = toCorotating(ray.origin) / separation;
  let direction = toCorotating(ray.direction);
  let count = i32(params.streamState.z);
  var tau = 0.0;
  var emission = vec3<f32>(0.0);
  for (var index: i32 = 0; index < ${SUPERNOVA_IA_STREAM_POINTS - 1}; index = index + 1) {
    if (index + 1 >= count) {
      break;
    }
    let a = params.streamPoints[index];
    let b = params.streamPoints[index + 1];
    let segment = b.xyz - a.xyz;
    let segmentLength = length(segment);
    if (segmentLength <= 1.0e-9) {
      continue;
    }
    let axis = segment / segmentLength;
    // Closest approach between the ray and the segment line.
    let w0 = origin - a.xyz;
    let bDot = dot(direction, axis);
    let denominator = max(1.0 - bDot * bDot, 1.0e-6);
    let sRay = (bDot * dot(w0, axis) - dot(w0, direction)) / denominator;
    let sSegment = clamp((dot(w0, axis) - bDot * dot(w0, direction)) / denominator, 0.0, segmentLength);
    if (sRay < 0.0 || sRay * separation > limit) {
      continue;
    }
    let closest = w0 + sRay * direction - sSegment * axis;
    let halfWidth = mix(a.w, b.w, sSegment / segmentLength);
    let d2 = dot(closest, closest) / (halfWidth * halfWidth);
    let sinTheta = sqrt(denominator);
    let segmentTau = params.streamState.y * halfWidth * 1.7724539 / max(sinTheta, 0.05) * exp(-d2);
    if (segmentTau > 1.0e-7) {
      let world = fromCorotating((a.xyz + sSegment * axis) * separation);
      var temperature = min(
        irradiatedTemperature(params.streamState.x, irradiationFlux(world, vec3<f32>(0.0), true)),
        params.primaryState.w
      );
      // Gas reaching the impact region is part of the spot.
      let spotNormal = fromCorotating(params.hotSpot.xyz);
      let spotRadius = params.primaryPositionRadius.w * sqrt(max(1.0 - params.hotSpot.w * params.hotSpot.w, 0.0));
      if (length(world - (params.primaryPositionRadius.xyz + spotNormal * params.primaryPositionRadius.w)) < spotRadius) {
        temperature = params.primaryState.w;
      }
      emission = emission + segmentTau * exposedBlackbody(temperature, exposureLog2);
      tau = tau + segmentTau;
    }
  }
  if (tau <= 1.0e-6) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let opacity = 1.0 - exp(-tau);
  return vec4<f32>(emission / tau * opacity, 1.0 - opacity);
}

fn compositionColour(x: f32) -> vec3<f32> {
  // v/vmax bands: 56Ni-rich core (gold), intermediate-mass elements Si/S/Ca
  // (green), unburned O/C (blue), helium-detonation ash (magenta).
  let zones = params.ejectaZones;
  if (x >= zones.z) { return vec3<f32>(0.85, 0.25, 0.85); }
  if (x >= zones.y) { return vec3<f32>(0.25, 0.45, 0.95); }
  if (x >= zones.x) { return vec3<f32>(0.30, 0.80, 0.35); }
  return vec3<f32>(0.95, 0.72, 0.20);
}

// Full shading of one pixel: colour plus an object signature in w (surface
// kind + 4 if ejecta contribute + 8 if the stream does), used to keep sparse
// interpolation within one kind of object.
fn shadePixel(uv: vec2<f32>, subpixel: vec2<f32>, marchJitter: f32, mode: i32) -> vec4<f32> {
  let ray = cameraRay(uv, subpixel);
  let exposureLog2 = params.sceneScale.z;

  // Nearest opaque surface.
  var surface = noHit();
  if (params.primaryState.z > 0.5) {
    if (params.explosionFronts.w > 0.5) {
      surface = shadeAccretorExploding(ray, exposureLog2);
    } else {
      surface = shadeAccretorQuiescent(ray, exposureLog2);
    }
  }
  let donor = shadeDonor(ray, exposureLog2);
  if (donor.kind != 0 && donor.distance < surface.distance) {
    surface = donor;
  }

  var geometry: EjectaGeometry;
  geometry.centre = params.ejectaCentre.xyz;
  geometry.drift = params.ejectaDrift.xyz;
  geometry.referenceTime = params.ejectaCentre.w;
  geometry.maximumVelocity = params.ejectaDrift.w;
  geometry.axis = params.ejectaAxis.xyz;
  geometry.referenceLog10Time = params.ejectaAxis.w;
  geometry.colourRatio = params.ejectaPhotosphere.y;
  geometry.nebularColourTemperature = params.ejectaPhotosphere.z;
  geometry.activeVelocity = params.ejectaBounds.x;
  var ejecta: EjectaResult;
  ejecta.radiance = vec3<f32>(0.0);
  ejecta.transmittance = 1.0;
  ejecta.weight = 0.0;
  ejecta.weightedLog10Temperature = 0.0;
  ejecta.weightedVelocity = 0.0;
  ejecta.weightedSector = 0.0;
  if (geometry.referenceTime > 0.0) {
    ejecta = marchEjecta(
      ray,
      geometry,
      exposureLog2,
      i32(params.steps.x),
      params.steps.y,
      surface.distance,
      marchJitter
    );
  }
  let stream = streamEmission(ray, surface.distance, exposureLog2);

  if (mode == 3) {
    // Composition false colour.
    var colour = vec3<f32>(0.0);
    if (ejecta.weight > 1.0e-4) {
      colour = compositionColour(ejecta.weightedVelocity / ejecta.weight) * ejecta.weight;
    }
    var behind = vec3<f32>(0.0);
    if (surface.kind == 1) { behind = vec3<f32>(0.55, 0.62, 0.80); }
    if (surface.kind == 2) { behind = vec3<f32>(0.80, 0.70, 0.45); }
    behind = mix(behind, vec3<f32>(0.95, 0.35, 0.20), 1.0 - stream.w);
    return vec4<f32>(decodeSrgbDisplay(colour + ejecta.transmittance * behind), 0.0);
  }
  if (mode == 2) {
    // Temperature false colour: emission-weighted log10 T over 3..8.
    var log10T = surface.log10Temperature;
    if (ejecta.weight > 0.02) {
      log10T = mix(surface.log10Temperature, ejecta.weightedLog10Temperature / ejecta.weight, clamp(ejecta.weight, 0.0, 1.0));
    }
    if (ejecta.weight <= 0.02 && surface.kind == 0) {
      return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    return vec4<f32>(decodeSrgbDisplay(viridis((log10T - 3.0) / 5.0)), 0.0);
  }

  var colour = ejecta.radiance;
  var behind = surface.radiance;
  if (surface.kind == 0) {
    behind = skyLuminance(ray.direction) * exp2(exposureLog2);
  }
  // The stream lies in front of the surfaces it does not intersect.
  behind = stream.rgb + stream.w * behind;
  colour = colour + ejecta.transmittance * behind;
  let signature = f32(surface.kind)
    + select(0.0, 4.0, ejecta.weight > 0.01)
    + select(0.0, 8.0, stream.w < 0.999);
  return vec4<f32>(clamp(colour, vec3<f32>(0.0), vec3<f32>(HALF_MAX)), signature);
}

// Coarse node (i, j), stored at texel (i + 1, j + 1), shades pixel
// (i S, j S) with a fixed march offset so the node field is smooth.
@fragment
fn fsCoarse(input: FragmentInput) -> @location(0) vec4<f32> {
  let node = floor(input.position.xy) - vec2<f32>(1.0);
  let resolution = max(params.resolutionTimeMass.xy, vec2<f32>(1.0));
  let pixel = node * SPARSE_STRIDE + vec2<f32>(0.5);
  return shadePixel(pixel / resolution, vec2<f32>(0.0), 0.5, 0);
}

fn coarseNode(texel: vec2<i32>) -> vec4<f32> {
  return textureLoad(coarseField, texel, 0);
}

fn channelMaximum(v: vec3<f32>) -> f32 {
  return max(max(abs(v.x), abs(v.y)), abs(v.z));
}

// Fail closed: interpolate only when the four cell corners and their eight
// axis neighbours show the same objects and the bilinear error bound is small.
// Returns the colour in rgb and 1 in w when accepted, else w = 0.
fn sparseRadiance(pixel: vec2<f32>) -> vec4<f32> {
  let size = vec2<i32>(textureDimensions(coarseField));
  let cell = floor(pixel / SPARSE_STRIDE);
  let fraction = (pixel - cell * SPARSE_STRIDE) / SPARSE_STRIDE;
  let base = vec2<i32>(cell) + vec2<i32>(1);
  if (any(base < vec2<i32>(1)) || any(base + vec2<i32>(2) >= size)) {
    return vec4<f32>(0.0);
  }
  let c00 = coarseNode(base);
  let c10 = coarseNode(base + vec2<i32>(1, 0));
  let c01 = coarseNode(base + vec2<i32>(0, 1));
  let c11 = coarseNode(base + vec2<i32>(1, 1));
  let xm0 = coarseNode(base + vec2<i32>(-1, 0));
  let xp0 = coarseNode(base + vec2<i32>(2, 0));
  let xm1 = coarseNode(base + vec2<i32>(-1, 1));
  let xp1 = coarseNode(base + vec2<i32>(2, 1));
  let ym0 = coarseNode(base + vec2<i32>(0, -1));
  let ym1 = coarseNode(base + vec2<i32>(1, -1));
  let yp0 = coarseNode(base + vec2<i32>(0, 2));
  let yp1 = coarseNode(base + vec2<i32>(1, 2));
  let signature = c00.w;
  let signatures = array<f32, 11>(c10.w, c01.w, c11.w, xm0.w, xp0.w, xm1.w, xp1.w, ym0.w, ym1.w, yp0.w, yp1.w);
  for (var index = 0; index < 11; index = index + 1) {
    if (abs(signatures[index] - signature) > 0.25) {
      return vec4<f32>(0.0);
    }
  }
  let curvature = max(
    max(
      max(channelMaximum(xm0.rgb - 2.0 * c00.rgb + c10.rgb), channelMaximum(ym0.rgb - 2.0 * c00.rgb + c01.rgb)),
      max(channelMaximum(c00.rgb - 2.0 * c10.rgb + xp0.rgb), channelMaximum(ym1.rgb - 2.0 * c10.rgb + c11.rgb))
    ),
    max(
      max(channelMaximum(xm1.rgb - 2.0 * c01.rgb + c11.rgb), channelMaximum(c00.rgb - 2.0 * c01.rgb + yp0.rgb)),
      max(channelMaximum(c01.rgb - 2.0 * c11.rgb + xp1.rgb), channelMaximum(c10.rgb - 2.0 * c11.rgb + yp1.rgb))
    )
  );
  let local = max(channelMaximum(0.25 * (c00.rgb + c10.rgb + c01.rgb + c11.rgb)), 0.0);
  if (0.125 * curvature > max(SPARSE_RELATIVE_TOLERANCE * local, SPARSE_ABSOLUTE_FLOOR)) {
    return vec4<f32>(0.0);
  }
  let colour = mix(mix(c00.rgb, c10.rgb, fraction.x), mix(c01.rgb, c11.rgb, fraction.x), fraction.y);
  return vec4<f32>(colour, 1.0);
}

@fragment
fn fsMain(input: FragmentInput) -> @location(0) vec4<f32> {
  let mode = i32(round(params.renderControls.z));
  // Only unjittered photographic frames that requested sparse tracing use
  // the coarse field (the renderer traced it first).
  if (params.steps.w == SPARSE_STRIDE && params.quality.w > 0.5 && (mode == 0 || mode == 1)) {
    let sparse = sparseRadiance(floor(input.position.xy));
    if (sparse.w > 0.5) {
      return vec4<f32>(sparse.rgb, 1.0);
    }
  }
  let shade = shadePixel(
    input.uv,
    jitterForSample(params.quality.x, params.quality.w),
    pixelJitter(input.position.xy, params.quality.x),
    mode
  );
  return vec4<f32>(shade.rgb, 1.0);
}
`;

export const supernovaIaTraceFragmentWGSL = SUPERNOVA_IA_WGSL;



function finitePacket(frame) {
  const packet = frame?.sceneTransientUniforms;
  const tailFloats = SUPERNOVA_IA_UNIFORM_FLOATS - 36;
  if (!packet || packet.length !== tailFloats || !Array.from(packet).every(Number.isFinite)) {
    throw new Error(`sceneTransientUniforms must contain ${tailFloats} finite floats for the Type Ia scene`);
  }
  return packet;
}

function transientQualityVector(frame) {
  const quality = frame?.strongFieldQuality;
  if (quality == null) {
    return [0, 1, 0, 1];
  }
  return [
    Number(quality.accumulationIndex) || 0,
    Number(quality.accumulationWeight) || 1,
    Number(quality.historyEpoch) || 0,
    quality.historyReset ? 1 : 0,
  ];
}

export const supernovaIaShaderBundle = Object.freeze({
  id: "supernova-ia-double-detonation-v1",
  labels: Object.freeze({
    uniforms: "Type Ia supernova scene uniforms",
    trace: "WebGPU double-detonation Type Ia supernova emission-absorption tracer",
    webglFallback: "Reduced WebGL2 photosphere fallback without volumetric transport",
  }),
  backendPolicy: Object.freeze({
    production: "webgpu",
    webgpuModel: "flat-space emission-absorption transport through a grey radiation-diffusion solution",
    webgl2Model: "reduced-limb-darkened-spheres",
    physicalParityRequired: false,
  }),
  accumulation: Object.freeze({
    mode: "linear-hdr-running-average-v1",
    sampleIndexField: "strongFieldQuality.accumulationIndex",
    resetField: "strongFieldQuality.historyReset",
    jitter: "deterministic-bounded-halton-2-3",
  }),
  // Camera veiling glare (src/glare.js) on photographic frames.
  glare: Object.freeze({ observer: GLARE_OBSERVER.model, strength: CAMERA_GLARE_STRENGTH }),
  wgsl: Object.freeze({
    trace: supernovaIaTraceFragmentWGSL,
    coarse: Object.freeze({
      entryPoint: "fsCoarse",
      format: "rgba32float",
      stride: SUPERNOVA_IA_SPARSE_STRIDE,
      binding: 3,
      sharesTraceResources: true,
    }),
  }),
  glsl: Object.freeze({
    trace: TRANSIENT_SPHERES_FALLBACK_GLSL,
    diagnosticModes: false,
  }),
  uniforms: Object.freeze({
    requiredFloatCount: SUPERNOVA_IA_UNIFORM_FLOATS,
    writeWebGPUExtras(tail, frame) {
      const packet = finitePacket(frame);
      if (tail.length < packet.length) {
        throw new Error("Type Ia uniform tail is too short");
      }
      tail.set(packet, 0);
      tail.set(transientQualityVector(frame), SUPERNOVA_IA_UNIFORM_LAYOUT.quality - 36);
      const steps = frame.sceneTransientSteps;
      if (steps) {
        tail.set(steps.slice(0, 3), SUPERNOVA_IA_UNIFORM_LAYOUT.steps - 36);
      }
      // Sparse stride in steps.w, identical to the renderer's opt-in field.
      tail[SUPERNOVA_IA_UNIFORM_LAYOUT.steps - 36 + 3] = Number(frame.sceneStrongDiagnostics?.[3]) || 0;
    },
    ...transientFallbackUniforms("Type Ia"),
  }),
  resources: createTransientTableResources([
    {
      name: "blackbody colour",
      binding: 4,
      uniform: "uSceneBlackbodyTable",
      width: SUPERNOVA_IA_BLACKBODY_TABLE.entries,
      height: 1,
      data: SUPERNOVA_IA_BLACKBODY_TABLE.data,
    },
  ]),
});
