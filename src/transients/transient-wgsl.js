// WGSL shared by the transient scenes. Radiance inside the trace pass is
// photometric luminance in cd/m^2 with linear-sRGB chromaticity; the scene's
// auto-exposure scale maps it to the shared HDR post pass. Bindings: 0 params,
// 1 sky panorama, 2 sampler, 3 the blackbody table (see blackbody-color.js),
// 4+ scene tables.

// Common Params prefix; the renderer writes the first 40 floats.
export const TRANSIENT_PARAMS_PREFIX_WGSL = /* wgsl */ `
  resolutionTimeMass: vec4<f32>,
  renderControls: vec4<f32>,
  cameraPosRadius: vec4<f32>,
  cameraForwardFov: vec4<f32>,
  cameraRightSkyRotation: vec4<f32>,
  cameraUpDiskOuter: vec4<f32>,
  postDisplayFrame: vec4<f32>,
  observerVelocityBeta: vec4<f32>,
  displayOutput: vec4<f32>,
  // x: km per scene length unit, y: physical time (s), z: log2 exposure
  // (display value per cd/m^2), w: speed of light in scene units per second.
  sceneScale: vec4<f32>,`;

// Helpers that the strong-field tracer defines identically: a scene built on
// that tracer includes TRANSIENT_COMMON_WGSL without this block.
export const TRANSIENT_BASE_WGSL = /* wgsl */ `
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;

fn safeNormalize(value: vec3<f32>) -> vec3<f32> {
  let lengthSquared = dot(value, value);
  if (lengthSquared <= 1.0e-30) {
    return vec3<f32>(0.0, 0.0, 1.0);
  }
  return value * inverseSqrt(lengthSquared);
}

fn radicalInverse(indexInput: u32, base: u32) -> f32 {
  var index = indexInput;
  var factor = 1.0 / f32(base);
  var value = 0.0;
  for (var digit: i32 = 0; digit < 16; digit = digit + 1) {
    if (index == 0u) {
      break;
    }
    value = value + f32(index % base) * factor;
    index = index / base;
    factor = factor / f32(base);
  }
  return value;
}

fn rotateAroundY(direction: vec3<f32>, angle: f32) -> vec3<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec3<f32>(c * direction.x + s * direction.z, direction.y, -s * direction.x + c * direction.z);
}

fn viridis(valueInput: f32) -> vec3<f32> {
  let value = clamp(valueInput, 0.0, 1.0);
  let c0 = vec3<f32>(0.267, 0.005, 0.329);
  let c1 = vec3<f32>(0.283, 0.141, 0.458);
  let c2 = vec3<f32>(0.254, 0.265, 0.530);
  let c3 = vec3<f32>(0.207, 0.372, 0.553);
  let c4 = vec3<f32>(0.164, 0.471, 0.558);
  let c5 = vec3<f32>(0.128, 0.567, 0.551);
  let c6 = vec3<f32>(0.135, 0.659, 0.518);
  let c7 = vec3<f32>(0.267, 0.749, 0.441);
  let c8 = vec3<f32>(0.478, 0.821, 0.318);
  let c9 = vec3<f32>(0.741, 0.873, 0.150);
  let scaled = value * 9.0;
  let index = i32(floor(scaled));
  let fraction = fract(scaled);
  if (index <= 0) { return mix(c0, c1, fraction); }
  if (index == 1) { return mix(c1, c2, fraction); }
  if (index == 2) { return mix(c2, c3, fraction); }
  if (index == 3) { return mix(c3, c4, fraction); }
  if (index == 4) { return mix(c4, c5, fraction); }
  if (index == 5) { return mix(c5, c6, fraction); }
  if (index == 6) { return mix(c6, c7, fraction); }
  if (index == 7) { return mix(c7, c8, fraction); }
  return mix(c8, c9, fraction);
}

fn decodeSrgbDisplay(colour: vec3<f32>) -> vec3<f32> {
  let encoded = clamp(colour, vec3<f32>(0.0), vec3<f32>(1.0));
  return select(
    pow((encoded + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)),
    encoded / 12.92,
    encoded <= vec3<f32>(0.04045)
  );
}
`;

export const TRANSIENT_COMMON_WGSL = /* wgsl */ `
const STEFAN_BOLTZMANN: f32 = 5.670374419e-5;
// Photometric luminance (cd/m^2) of bolometric radiance 1 erg s^-1 cm^-2 sr^-1
// times the luminous fraction: 683 lm/W * 1e-3 W m^-2 sr^-1.
const LUMINANCE_PER_CGS_RADIANCE: f32 = 0.683;
// Calibration of the photographic panorama: its median pixel is taken as
// V = 22 mag/arcsec^2 (~1.7e-4 cd/m^2), the dark-sky surface brightness away
// from the Galactic plane. Only a sky-referenced exposure can show it next to
// a stellar surface ~10^13 times brighter.
const SKY_LUMINANCE_PER_PANORAMA_UNIT: f32 = 1.2e-3;
const HALF_MAX: f32 = 60000.0;

struct Ray {
  origin: vec3<f32>,
  direction: vec3<f32>,
};

// Deterministic, bounded Halton(2,3) sub-pixel jitter for progressive
// accumulation; reset samples are unjittered (see the binary tracer).
fn jitterForSample(sampleIndex: f32, reset: f32) -> vec2<f32> {
  if (reset > 0.5) {
    return vec2<f32>(0.0);
  }
  let index = u32(clamp(sampleIndex + 1.0, 1.0, 1048575.0));
  let amplitude = mix(0.20, 0.58, smoothstep(1.0, 8.0, f32(index)));
  return amplitude * vec2<f32>(
    radicalInverse(index, 2u) - 0.5,
    radicalInverse(index, 3u) - 0.5
  );
}

// Interleaved-gradient noise (Jimenez 2014) in [0, 1), decorrelated per
// accumulation sample.
fn pixelJitter(pixel: vec2<f32>, sampleIndex: f32) -> f32 {
  let p = pixel + vec2<f32>(5.588238, 3.7) * sampleIndex;
  return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

fn cameraRay(uv: vec2<f32>, jitter: vec2<f32>) -> Ray {
  let resolution = max(params.resolutionTimeMass.xy, vec2<f32>(1.0));
  let aspect = resolution.x / resolution.y;
  let jittered = uv + jitter / resolution;
  let screen = vec2<f32>((jittered.x * 2.0 - 1.0) * aspect, 1.0 - jittered.y * 2.0);
  let tanHalfFov = tan(0.5 * clamp(params.cameraForwardFov.w, 0.02, 2.8));
  let forward = safeNormalize(params.cameraForwardFov.xyz);
  let right = safeNormalize(params.cameraRightSkyRotation.xyz - forward * dot(forward, params.cameraRightSkyRotation.xyz));
  let up = cross(right, forward);
  var ray: Ray;
  ray.origin = params.cameraPosRadius.xyz;
  ray.direction = safeNormalize(forward + tanHalfFov * (screen.x * right + screen.y * up));
  return ray;
}

// Entry and exit distances along a ray, (1, 0) when the sphere is missed.
fn sphereInterval(ray: Ray, centre: vec3<f32>, radius: f32) -> vec2<f32> {
  let offset = ray.origin - centre;
  let b = dot(offset, ray.direction);
  let c = dot(offset, offset) - radius * radius;
  let discriminant = b * b - c;
  if (discriminant <= 0.0 || radius <= 0.0) {
    return vec2<f32>(1.0, 0.0);
  }
  let root = sqrt(discriminant);
  return vec2<f32>(max(-b - root, 0.0), -b + root);
}

// Photographic panorama as sky luminance (cd/m^2) in linear sRGB.
fn skyLuminance(direction: vec3<f32>) -> vec3<f32> {
  let d = safeNormalize(rotateAroundY(direction, params.cameraRightSkyRotation.w));
  let uv = vec2<f32>(
    fract(atan2(d.z, d.x) / TWO_PI + 0.5),
    clamp(0.5 - asin(clamp(d.y, -1.0, 1.0)) / PI, 0.00001, 0.99999)
  );
  let panorama = textureSampleLevel(tSky, skySampler, uv, 0.0).rgb;
  return max(panorama - vec3<f32>(0.0015), vec3<f32>(0.0))
    * max(params.displayOutput.w, 0.01) * SKY_LUMINANCE_PER_PANORAMA_UNIT;
}

// Blackbody table: texel = (rgb / luminous fraction, log10 luminous fraction)
// over log10 T in [BLACKBODY_LOG10_T_MIN, BLACKBODY_LOG10_T_MAX].
fn blackbodyEntry(log10Temperature: f32) -> vec4<f32> {
  let count = f32(textureDimensions(blackbodyTable).x);
  let position = clamp(
    (log10Temperature - BLACKBODY_LOG10_T_MIN) / (BLACKBODY_LOG10_T_MAX - BLACKBODY_LOG10_T_MIN),
    0.0,
    1.0
  ) * (count - 1.0);
  let lower = floor(position);
  let fraction = position - lower;
  let a = textureLoad(blackbodyTable, vec2<i32>(i32(lower), 0), 0);
  let b = textureLoad(blackbodyTable, vec2<i32>(min(i32(lower) + 1, i32(count) - 1), 0), 0);
  return mix(a, b, fraction);
}

// Luminance (cd/m^2, linear-sRGB chromaticity) of a blackbody surface at
// temperature T.
fn blackbodyLuminance(temperature: f32) -> vec3<f32> {
  let t = max(temperature, 1.0);
  let entry = blackbodyEntry(log2(t) * 0.30102999566);
  // sigma T^4 / pi * luminous fraction, computed in log space so that neither
  // 10^9 K nor 300 K overflows or underflows f32.
  let log10Luminance = log2(STEFAN_BOLTZMANN * LUMINANCE_PER_CGS_RADIANCE / PI) * 0.30102999566
    + 4.0 * log2(t) * 0.30102999566 + entry.w;
  return entry.rgb * exp2(log10Luminance * 3.32192809489);
}

// Same, pre-multiplied by the exposure scale in log space so the result is a
// display-range value even for 10^9 K sources.
fn exposedBlackbody(temperature: f32, exposureLog2: f32) -> vec3<f32> {
  let t = max(temperature, 1.0);
  let entry = blackbodyEntry(log2(t) * 0.30102999566);
  let log2Value = log2(STEFAN_BOLTZMANN * LUMINANCE_PER_CGS_RADIANCE / PI)
    + 4.0 * log2(t) + entry.w * 3.32192809489 + exposureLog2;
  return entry.rgb * exp2(min(log2Value, 16.0));
}

// Relativistic Doppler factor D = 1 / (gamma (1 - beta . n_photon)) of an
// emitter with velocity beta (units of c) for a photon travelling along
// photonDirection towards the camera.
fn dopplerFactor(beta: vec3<f32>, photonDirection: vec3<f32>) -> f32 {
  let betaSquared = min(dot(beta, beta), 0.9999);
  let gamma = inverseSqrt(1.0 - betaSquared);
  return 1.0 / (gamma * (1.0 - dot(beta, photonDirection)));
}
`;

export function transientBlackbodyConstantsWGSL(table) {
  return /* wgsl */ `
const BLACKBODY_LOG10_T_MIN: f32 = ${table.minLog10Temperature.toFixed(4)};
const BLACKBODY_LOG10_T_MAX: f32 = ${table.maxLog10Temperature.toFixed(4)};
`;
}

// Homologously expanding ejecta, rendered by emission-absorption transport
// through the radiation-diffusion solution. The scene evaluates the solution
// on the CPU at the displayed time and passes, per polar sector and velocity
// bin (index sector * EJECTA_BINS + bin, velocity (bin + 0.5)/EJECTA_BINS vmax):
//   ejectaProfile  = (log10 T_rad, d/dlog10 t, log10(kappa rho t^3), d/dlog10 t)
//   ejectaRadiative = (log10 tau_out, d/dlog10 t, log10(j_dep t^3), d/dlog10 t)
// with tau_out the radial optical depth to the surface and j_dep = rho q/(4 pi)
// the emissivity of locally deposited radioactive power (erg s^-1 cm^-3 sr^-1).
// Each sample uses its own retarded time (light-travel delay) and Doppler
// factor D of the local expansion velocity: the comoving grey source B(T) is
// seen as B(D T) with absorption kappa rho / D.
//
// Source function. Deep layers radiate at the diffusion temperature T_rad.
// Near the photosphere a coarse radial grid cannot resolve the steep
// temperature drop, so T is capped by the grey Eddington atmosphere that
// carries the same flux: T^4 = (3/4) T_eff^4 (tau_out + 2/3). Above the
// photosphere the gas is thin and deposited radioactive energy escapes
// directly with weight exp(-tau_out) (nebular emission), so the rendered
// brightness follows the light curve in both limits. Spectral line
// blanketing is represented by a colour temperature T_col = r T (r from the
// observed B-V track) that sets chromaticity and luminous fraction while the
// bolometric radiance stays sigma T^4 / pi.
export function transientEjectaWGSL({ bins, sectors }) {
  return /* wgsl */ `
const EJECTA_BINS: i32 = ${bins};
const EJECTA_SECTORS: i32 = ${sectors};

struct EjectaSample {
  log10Temperature: f32,
  absorption: f32,
  log10OpticalDepth: f32,
  log10Emissivity: f32,
};

// Profile at velocity fraction x = v / vmax, fractional polar sector, and
// log10 of the emission time relative to the reference time.
fn ejectaProfileAt(x: f32, sector: f32, deltaLog10Time: f32) -> EjectaSample {
  let binPosition = clamp(x * f32(EJECTA_BINS) - 0.5, 0.0, f32(EJECTA_BINS - 1));
  let binLow = i32(floor(binPosition));
  let binHigh = min(binLow + 1, EJECTA_BINS - 1);
  let binFraction = binPosition - f32(binLow);
  let sectorPosition = clamp(sector, 0.0, f32(EJECTA_SECTORS - 1));
  let sectorLow = i32(floor(sectorPosition));
  let sectorHigh = min(sectorLow + 1, EJECTA_SECTORS - 1);
  let sectorFraction = sectorPosition - f32(sectorLow);
  let i00 = sectorLow * EJECTA_BINS + binLow;
  let i01 = sectorLow * EJECTA_BINS + binHigh;
  let i10 = sectorHigh * EJECTA_BINS + binLow;
  let i11 = sectorHigh * EJECTA_BINS + binHigh;
  let profile = mix(
    mix(params.ejectaProfile[i00], params.ejectaProfile[i01], binFraction),
    mix(params.ejectaProfile[i10], params.ejectaProfile[i11], binFraction),
    sectorFraction
  );
  let radiative = mix(
    mix(params.ejectaRadiative[i00], params.ejectaRadiative[i01], binFraction),
    mix(params.ejectaRadiative[i10], params.ejectaRadiative[i11], binFraction),
    sectorFraction
  );
  var sample: EjectaSample;
  sample.log10Temperature = profile.x + profile.y * deltaLog10Time;
  sample.absorption = profile.z + profile.w * deltaLog10Time;
  sample.log10OpticalDepth = radiative.x + radiative.y * deltaLog10Time;
  sample.log10Emissivity = radiative.z + radiative.w * deltaLog10Time;
  // Beyond the outermost bin centre the optical depth to the surface falls
  // to zero at vmax.
  let outerCentre = 1.0 - 0.5 / f32(EJECTA_BINS);
  if (x > outerCentre) {
    sample.log10OpticalDepth = sample.log10OpticalDepth
      + log2(max((1.0 - x) / (1.0 - outerCentre), 1.0e-6)) * 0.30102999566;
  }
  return sample;
}

struct EjectaResult {
  radiance: vec3<f32>,
  transmittance: f32,
  // Emission-weighted false-colour quantities for diagnostic modes.
  weightedLog10Temperature: f32,
  weightedSector: f32,
  weightedVelocity: f32,
  weight: f32,
};

// Ejecta geometry: centre at the reference emission time (scene units), its
// drift velocity (scene units per second), the reference emission time (s,
// since explosion), vmax (km/s), the polar axis for sectors and asymmetry,
// the colour-temperature ratio of the photosphere and the colour temperature
// of nebular emission.
struct EjectaGeometry {
  centre: vec3<f32>,
  drift: vec3<f32>,
  referenceTime: f32,
  maximumVelocity: f32,
  axis: vec3<f32>,
  referenceLog10Time: f32,
  colourRatio: f32,
  nebularColourTemperature: f32,
  // Outermost velocity (km/s) that still absorbs or emits appreciably; the
  // march is bounded to it.
  activeVelocity: f32,
};

// Hooks implemented by the scene:
//   fn ejectaDensityFactor(direction: vec3<f32>) -> f32   (1 = isotropic)
//   fn ejectaVelocityStretch(direction: vec3<f32>) -> f32 (1 = spherical)
//   fn ejectaEffectiveTemperature(sector: f32) -> f32     (photospheric T_eff
//     of the diffusion solution in that polar sector, for the atmosphere cap)
// A stretch s(n) maps the spherical profile to v' = s v along n, keeping the
// mass per solid angle (density / s^3).

// Blackbody with bolometric radiance sigma T^4 / pi but the chromaticity and
// luminous fraction of a blackbody at the colour temperature.
fn exposedColourBlackbody(temperature: f32, colourTemperature: f32, exposureLog2: f32) -> vec3<f32> {
  let t = max(temperature, 1.0);
  let entry = blackbodyEntry(log2(max(colourTemperature, 1.0)) * 0.30102999566);
  let log2Value = log2(STEFAN_BOLTZMANN * LUMINANCE_PER_CGS_RADIANCE / PI)
    + 4.0 * log2(t) + entry.w * 3.32192809489 + exposureLog2;
  return entry.rgb * exp2(min(log2Value, 16.0));
}

// limit: distance to the nearest opaque surface (the march stops there);
// jitter in [0, 1) offsets the first sample (stratified per pixel) so coarse
// steps through thin gas dither instead of banding.
fn marchEjecta(
  ray: Ray,
  geometry: EjectaGeometry,
  exposureLog2: f32,
  maximumSteps: i32,
  opticalDepthStep: f32,
  limit: f32,
  jitter: f32
) -> EjectaResult {
  var result: EjectaResult;
  result.radiance = vec3<f32>(0.0);
  result.transmittance = 1.0;
  result.weightedLog10Temperature = 0.0;
  result.weightedSector = 0.0;
  result.weightedVelocity = 0.0;
  result.weight = 0.0;
  if (geometry.referenceTime <= 0.0 || geometry.maximumVelocity <= 0.0) {
    return result;
  }
  let kmPerUnit = params.sceneScale.x;
  let lightSpeed = params.sceneScale.w;
  let cmPerUnit = kmPerUnit * 1.0e5;
  let maximumStretch = 1.25;
  // Only shells inside the active velocity are marched; the bounding sphere
  // holds their retarded extent (the near side is seen later, larger by up
  // to 1 / (1 - beta)).
  let boundVelocity = min(geometry.activeVelocity, geometry.maximumVelocity);
  let betaMax = min(maximumStretch * boundVelocity / 299792.458, 0.9);
  let centreDistance = length(geometry.centre - ray.origin);
  let outerRadius = maximumStretch * boundVelocity * geometry.referenceTime / kmPerUnit / (1.0 - betaMax);
  let interval = sphereInterval(ray, geometry.centre, outerRadius * 1.02);
  if (interval.x >= interval.y) {
    return result;
  }
  let sEnd = min(interval.y, limit);
  let minimumStep = (interval.y - interval.x) / f32(max(maximumSteps, 8)) * 0.02;
  let maximumStep = (interval.y - interval.x) / 48.0;
  var s = interval.x + jitter * maximumStep;
  // Nebular chromaticity varies slowly: one table fetch per ray.
  let nebularEntry = blackbodyEntry(log2(max(geometry.nebularColourTemperature, 1.0)) * 0.30102999566);
  let nebularLog2Base = log2(LUMINANCE_PER_CGS_RADIANCE) + nebularEntry.w * 3.32192809489 + exposureLog2;
  for (var step: i32 = 0; step < maximumSteps; step = step + 1) {
    // Below transmittance 1e-3 the remaining contribution is < 0.1%.
    if (s >= sEnd || result.transmittance < 1.0e-3) {
      break;
    }
    let position = ray.origin + s * ray.direction;
    // Emission time of light from this point that reaches the camera together
    // with light from the centre emitted at referenceTime.
    let emissionTime = max(geometry.referenceTime + (centreDistance - s) / lightSpeed, 1.0e-3);
    let centre = geometry.centre + geometry.drift * (emissionTime - geometry.referenceTime);
    let offset = position - centre;
    let velocityKmS = offset * (kmPerUnit / emissionTime);
    let speed = length(velocityKmS);
    let direction = velocityKmS / max(speed, 1.0e-6);
    let stretch = ejectaVelocityStretch(direction);
    let x = speed / (geometry.maximumVelocity * stretch);
    var stepLength = maximumStep;
    if (speed < boundVelocity * stretch) {
      let cosPolar = abs(dot(direction, geometry.axis));
      let sector = acos(clamp(cosPolar, 0.0, 1.0)) / (0.5 * PI) * f32(EJECTA_SECTORS - 1);
      let log10Time = log2(emissionTime) * 0.30102999566;
      let deltaLog10Time = log10Time - geometry.referenceLog10Time;
      let sample = ejectaProfileAt(x, sector, deltaLog10Time);
      let beta = (velocityKmS + geometry.drift * kmPerUnit) / 299792.458;
      let doppler = dopplerFactor(beta, -ray.direction);
      let densityFactor = ejectaDensityFactor(direction) / (stretch * stretch * stretch);
      // kappa rho = 10^absorption / t^3 (cm^-1).
      let log10Time3 = 3.0 * log10Time;
      let absorption = exp2((sample.absorption - log10Time3) * 3.32192809489) * densityFactor / doppler;
      let absorptionPerUnit = absorption * cmPerUnit;
      // Deeper along the ray the source contributes with weight
      // exp(-tau_ray), so the optical-depth step can grow.
      let rayOpticalDepth = -log(max(result.transmittance, 1.0e-30));
      stepLength = clamp(
        opticalDepthStep * (1.0 + 0.5 * rayOpticalDepth) / max(absorptionPerUnit, 1.0e-30),
        minimumStep,
        maximumStep
      );
      let tau = absorptionPerUnit * stepLength;
      let opacity = 1.0 - exp(-tau);
      // Eddington cap: T^4 <= (3/4) T_eff^4 (tau + 2/3). A step that jumps
      // from outside to far below an unresolved photosphere (early on it is a
      // skin ~10^-6 of the radius thick) emits from its Eddington-Barbier
      // depth, one optical depth below where the ray entered.
      let opticalDepth = exp2(sample.log10OpticalDepth * 3.32192809489) * densityFactor;
      // Along a ray at cosine mu to the radial direction one optical depth
      // lies at radial depth ~mu, which yields grey limb darkening
      // I ~ mu + 2/3.
      let cosRadial = abs(dot(ray.direction, direction));
      let rayDepth = cosRadial * (rayOpticalDepth + 1.0);
      let log2Teff = log2(max(ejectaEffectiveTemperature(sector), 1.0));
      let log2Cap = log2Teff + 0.25 * log2(0.75 * (min(opticalDepth, rayDepth) + 0.6666667));
      let log2Temperature = min(sample.log10Temperature * 3.32192809489, log2Cap);
      let temperature = doppler * exp2(log2Temperature);
      let emitted = exposedColourBlackbody(temperature, geometry.colourRatio * temperature, exposureLog2);
      let contribution = result.transmittance * opacity;
      // Nebular emission of locally deposited power, escaping with
      // exp(-tau_out); j D^3 per unit observer path (I_nu / nu^3 invariant,
      // bolometric D^4 with the D^-1 path factor). Negligible below
      // tau_out ~ 25.
      if (opticalDepth < 25.0) {
        let log2Nebular = (sample.log10Emissivity - log10Time3) * 3.32192809489
          + log2(densityFactor) + 3.0 * log2(doppler) - opticalDepth * 1.44269504
          + log2(stepLength * cmPerUnit) + nebularLog2Base;
        result.radiance = result.radiance + result.transmittance * nebularEntry.rgb * exp2(min(log2Nebular, 16.0));
      }
      result.radiance = result.radiance + contribution * emitted;
      result.weightedLog10Temperature = result.weightedLog10Temperature
        + contribution * (log2(max(temperature, 1.0)) * 0.30102999566);
      result.weightedSector = result.weightedSector + contribution * sector;
      result.weightedVelocity = result.weightedVelocity + contribution * x;
      result.weight = result.weight + contribution;
      result.transmittance = result.transmittance * (1.0 - opacity);
    }
    s = s + stepLength;
  }
  return result;
}
`;
}
