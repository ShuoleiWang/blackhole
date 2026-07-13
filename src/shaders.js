/*
 * Shared shader interface for the WebGPU and WebGL2 renderers.
 *
 * Geometry is expressed in gravitational radii with G = c = M = 1.  The
 * solar mass parameter affects the Novikov-Thorne-like temperature scale; it
 * does not resize the dimensionless Schwarzschild solution.  The supplied time
 * is already coordinate time in M.
 *
 * WGSL Params layout (Float32 offsets, std140-compatible because every field is
 * a vec4; minimum binding size 144 bytes).  The renderer keeps a 40-float
 * allocation; the remaining tail floats carry display and sky calibration.
 *
 *   0  resolution.x       1  resolution.y
 *   2  timeGeometricM     3  massSolar
 *   4  accretionRatio     5  exposureMultiplier
 *   6  displayMode        7  effectiveTraceSteps
 *   8  cameraPos.x        9  cameraPos.y
 *  10  cameraPos.z       11  cameraRadius
 *  12  cameraForward.x   13  cameraForward.y
 *  14  cameraForward.z   15  verticalFovRadians
 *  16  cameraRight.x     17  cameraRight.y
 *  18  cameraRight.z     19  skyRotationRadians
 *  20  cameraUp.x        21  cameraUp.y
 *  22  cameraUp.z        23  diskOuterRadius (in M)
 *  24  renderScale       25  bloomStrength
 *  26  motionState       27  frameIndex
 *  28  observerVelocity.x (unit tangent in the local static tetrad)
 *  29  observerVelocity.y
 *  30  observerVelocity.z
 *  31  observerBeta      (v/c; circular orbit uses 1/sqrt(R - 2))
 *  32  extendedHDR       33  outputDisplayP3
 *  34  HDRpeak/SDRwhite  35  panorama radiance calibration
 *
 * WGSL bindings:
 *   trace: binding 0 Params, binding 1 equirectangular sky texture,
 *          binding 2 repeat-U/clamp-V linear sampler.
 *   post:  binding 0 Params, binding 1 linear-HDR scene texture,
 *          binding 2 clamp linear sampler.
 * Entry points are vsMain and fsMain.
 *
 * GLSL uses the matching scalar/vector uniforms requested by the renderer and
 * samplers tSky (trace) and tScene (post).  All trace outputs are linear HDR;
 * exposure, ACES, the optional Hubble colour/PSF treatment, bloom, and dithering
 * happen only in the post pass.
 */

export const fullscreenVertexWGSL = /* wgsl */ `
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );

  let p = positions[vertexIndex];
  var output: VertexOutput;
  output.position = vec4<f32>(p, 0.0, 1.0);
  // WebGPU sampled textures use a top-left origin.  Keeping v = 0 at the top
  // makes the trace and post passes agree without a hidden render-target flip.
  output.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return output;
}
`;

export const traceFragmentWGSL = /* wgsl */ `
diagnostic(off, derivative_uniformity);

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const MAX_STEPS: i32 = 384;
const ISCO: f32 = 6.0;
const PHOTON_IMPACT: f32 = 5.196152422706632;

struct Params {
  resolutionTimeMass: vec4<f32>,
  renderControls: vec4<f32>,
  cameraPosRadius: vec4<f32>,
  cameraForwardFov: vec4<f32>,
  cameraRightSkyRotation: vec4<f32>,
  cameraUpDiskOuter: vec4<f32>,
  postMotionFrame: vec4<f32>,
  observerVelocityBeta: vec4<f32>,
  displayOutput: vec4<f32>,
};

struct FragmentInput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tSky: texture_2d<f32>;
@group(0) @binding(2) var skySampler: sampler;

fn safeNormalize(v: vec3<f32>) -> vec3<f32> {
  return v * inverseSqrt(max(dot(v, v), 1.0e-18));
}

fn schwarzschildForce(u: f32) -> f32 {
  // Exact Schwarzschild null-geodesic equation in the ray plane, G=c=M=1.
  return -u + 3.0 * u * u;
}

fn hash31(value: vec3<f32>) -> f32 {
  var p = fract(value * 0.1031);
  p = p + vec3<f32>(dot(p, p.yzx + vec3<f32>(33.33)));
  return fract((p.x + p.y) * p.z);
}

fn valueNoise3(value: vec3<f32>) -> f32 {
  let cell = floor(value);
  var f = fract(value);
  f = f * f * (vec3<f32>(3.0) - 2.0 * f);

  let n000 = hash31(cell + vec3<f32>(0.0, 0.0, 0.0));
  let n100 = hash31(cell + vec3<f32>(1.0, 0.0, 0.0));
  let n010 = hash31(cell + vec3<f32>(0.0, 1.0, 0.0));
  let n110 = hash31(cell + vec3<f32>(1.0, 1.0, 0.0));
  let n001 = hash31(cell + vec3<f32>(0.0, 0.0, 1.0));
  let n101 = hash31(cell + vec3<f32>(1.0, 0.0, 1.0));
  let n011 = hash31(cell + vec3<f32>(0.0, 1.0, 1.0));
  let n111 = hash31(cell + vec3<f32>(1.0, 1.0, 1.0));

  let nx00 = mix(n000, n100, f.x);
  let nx10 = mix(n010, n110, f.x);
  let nx01 = mix(n001, n101, f.x);
  let nx11 = mix(n011, n111, f.x);
  return mix(mix(nx00, nx10, f.y), mix(nx01, nx11, f.y), f.z);
}

fn diskTurbulence(position: vec3<f32>) -> f32 {
  var p = position;
  var result = 0.0;
  var amplitude = 0.5714286;
  for (var octave = 0; octave < 3; octave = octave + 1) {
    result = result + amplitude * valueNoise3(p);
    p = p * 2.07 + vec3<f32>(7.1, 13.7, 3.9);
    amplitude = amplitude * 0.5;
  }
  return result;
}

fn rotateAroundY(direction: vec3<f32>, angle: f32) -> vec3<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec3<f32>(
    c * direction.x + s * direction.z,
    direction.y,
   -s * direction.x + c * direction.z
  );
}

fn cubeStarCoordinates(direction: vec3<f32>) -> vec3<f32> {
  let d = safeNormalize(direction);
  let a = abs(d);
  var uv = vec2<f32>(0.0);
  var face = 0.0;
  if (a.x >= a.y && a.x >= a.z) {
    if (d.x >= 0.0) {
      uv = vec2<f32>(-d.z, d.y) / a.x;
      face = 0.0;
    } else {
      uv = vec2<f32>(d.z, d.y) / a.x;
      face = 1.0;
    }
  } else if (a.y >= a.z) {
    if (d.y >= 0.0) {
      uv = vec2<f32>(d.x, -d.z) / a.y;
      face = 2.0;
    } else {
      uv = vec2<f32>(d.x, d.z) / a.y;
      face = 3.0;
    }
  } else if (d.z >= 0.0) {
    uv = vec2<f32>(d.x, d.y) / a.z;
    face = 4.0;
  } else {
    uv = vec2<f32>(-d.x, d.y) / a.z;
    face = 5.0;
  }
  return vec3<f32>(uv * 0.5 + vec2<f32>(0.5), face);
}

fn planckChromaticity(temperatureKelvin: f32) -> vec3<f32> {
  // Three visible-band samples of Planck's law.  The calibration makes 6500 K
  // close to neutral while retaining the physically meaningful colour shift.
  let wavelengthsMicron = vec3<f32>(0.640, 0.530, 0.460);
  let lambda2 = wavelengthsMicron * wavelengthsMicron;
  let lambda5 = lambda2 * lambda2 * wavelengthsMicron;
  let exponent = min(
    vec3<f32>(80.0),
    vec3<f32>(14387.77 / max(temperatureKelvin, 100.0)) / wavelengthsMicron
  );
  var spectrum = vec3<f32>(1.0) / (lambda5 * max(exp(exponent) - vec3<f32>(1.0), vec3<f32>(1.0e-12)));
  spectrum = spectrum * vec3<f32>(1.2320, 1.0, 0.9367);
  let luminance = max(dot(spectrum, vec3<f32>(0.2126, 0.7152, 0.0722)), 1.0e-8);
  return spectrum / luminance;
}

fn proceduralStars(
  direction: vec3<f32>,
  observerShift: f32,
  criticalWeight: f32
) -> vec3<f32> {
  // A direction-locked sub-pixel stellar layer supplements the 6K photograph.
  // It is evaluated only after the null geodesic escapes, so every point is
  // lensed, duplicated and stretched by exactly the same map as the panorama.
  let mapped = cubeStarCoordinates(direction);
  var result = vec3<f32>(0.0);
  for (var layer = 0; layer < 2; layer = layer + 1) {
    var grid = 192.0;
    var threshold = 0.976;
    var coreRadius = 0.040;
    var minimumRadiance = 0.18;
    var maximumRadiance = 8.0;
    if (layer == 1) {
      grid = 384.0;
      threshold = 0.990;
      coreRadius = 0.055;
      minimumRadiance = 0.055;
      maximumRadiance = 2.2;
    }

    let coordinates = mapped.xy * grid;
    let cell = floor(coordinates);
    let local = fract(coordinates);
    let seed = vec3<f32>(
      cell + vec2<f32>(f32(layer) * 17.0, f32(layer) * 29.0),
      mapped.z * 37.0 + f32(layer) * 11.0
    );
    let random = vec3<f32>(
      hash31(seed),
      hash31(seed + vec3<f32>(19.17, 7.31, 3.13)),
      hash31(seed + vec3<f32>(43.71, 31.97, 17.53))
    );
    if (random.z > threshold) {
      let centre = vec2<f32>(0.5) + (random.xy - vec2<f32>(0.5)) * 0.5;
      let delta = local - centre;
      let distanceToStar = length(delta);
      let radialDirection = delta / max(distanceToStar, 1.0e-6);
      let footprint = 0.5 * (
        abs(dot(radialDirection, dpdx(coordinates)))
        + abs(dot(radialDirection, dpdy(coordinates)))
      );
      let regularAaWidth = min(
        clamp(footprint, 0.006, 0.045),
        0.90 * coreRadius
      );
      // The fixed 2x2 geodesic coverage resolves the critical curve itself.
      // Keeping this source-plane edge narrow prevents a stellar point from
      // becoming a soft radial ribbon after extreme tangential magnification.
      let criticalAaWidth = min(0.014, 0.35 * coreRadius);
      let antialiasWidth = mix(
        regularAaWidth,
        criticalAaWidth,
        clamp(criticalWeight, 0.0, 1.0)
      );
      // Constant-brightness stellar core with only a one-pixel analytic edge.
      // Unlike the old smooth radial blob, this stays a thin arc when the
      // lensing Jacobian becomes highly anisotropic near a critical curve.
      let core = 1.0 - smoothstep(
        coreRadius - antialiasWidth,
        coreRadius + antialiasWidth,
        distanceToStar
      );
      let rank = clamp((random.z - threshold) / (1.0 - threshold), 0.0, 1.0);
      // A compact, energy-light PSF skirt gives bright stars a readable HDR
      // hierarchy without softening their lensed image.  It is suppressed on
      // the critical curve, where even a tiny source-plane halo becomes a wide
      // tangential streak and the dedicated 2x2 coverage already resolves it.
      let haloRadius = coreRadius * mix(3.2, 2.2, rank);
      let halo = pow(max(1.0 - distanceToStar / haloRadius, 0.0), 3.0);
      let psf = core + 0.10 * rank * halo
              * (1.0 - clamp(criticalWeight, 0.0, 1.0));
      let stellarTemperature = mix(
        3000.0,
        11000.0,
        pow(hash31(seed + vec3<f32>(91.7, 53.1, 27.9)), 0.75)
      ) * observerShift;
      let radiance = mix(minimumRadiance, maximumRadiance, pow(rank, 0.42));
      result = result + planckChromaticity(stellarTemperature)
               * radiance * psf;
    }
  }
  return result;
}

fn skyLuminance(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn sampleSkyLevelZero(uv: vec2<f32>) -> vec3<f32> {
  return max(
    textureSampleLevel(tSky, skySampler, uv, 0.0).rgb,
    vec3<f32>(0.0)
  );
}

fn filterSkyPanorama(
  uv: vec2<f32>,
  centre: vec3<f32>,
  criticalWeight: f32
) -> vec3<f32> {
  // sampleEnvironment is reached after a per-pixel variable-length ray loop,
  // where quad derivatives are non-portable.  Use the already-computed smooth
  // critical-curve coverage instead and explicitly sample a small isotropic
  // footprint.  Ordinary rays still take only the authored centre sample.
  let weight = 0.62 * smoothstep(
    0.08,
    0.92,
    clamp(criticalWeight, 0.0, 1.0)
  );
  if (weight <= 0.001) {
    return centre;
  }
  let dimensions = vec2<f32>(textureDimensions(tSky));
  let texel = vec2<f32>(1.0) / dimensions;
  let radius = mix(1.0, 10.0, clamp(criticalWeight, 0.0, 1.0));
  let filtered = 0.25 * (
    sampleSkyLevelZero(uv + vec2<f32>(radius * texel.x, 0.0))
    + sampleSkyLevelZero(uv - vec2<f32>(radius * texel.x, 0.0))
    + sampleSkyLevelZero(uv + vec2<f32>(0.0, radius * texel.y))
    + sampleSkyLevelZero(uv - vec2<f32>(0.0, radius * texel.y))
  );
  return mix(centre, filtered, weight);
}

fn suppressBakedStarPsf(
  uv: vec2<f32>,
  centre: vec3<f32>,
  criticalWeight: f32
) -> vec3<f32> {
  let centreY = skyLuminance(centre);
  if (criticalWeight <= 0.001 || centreY < 0.008) {
    return centre;
  }

  let texel = vec2<f32>(1.0) / vec2<f32>(textureDimensions(tSky));
  let offset = 2.5 * texel;
  let pairX = 0.5 * (
    sampleSkyLevelZero(uv + vec2<f32>(offset.x, 0.0))
    + sampleSkyLevelZero(uv - vec2<f32>(offset.x, 0.0))
  );
  let pairY = 0.5 * (
    sampleSkyLevelZero(uv + vec2<f32>(0.0, offset.y))
    + sampleSkyLevelZero(uv - vec2<f32>(0.0, offset.y))
  );
  let pairXY = 0.5 * (pairX + pairY);
  let compactPeak = max(min(
    centreY - skyLuminance(pairX),
    centreY - skyLuminance(pairY)
  ), 0.0);
  let localY = max(skyLuminance(pairXY), 0.012);
  let starMask = smoothstep(0.004, 0.025, compactPeak)
               * smoothstep(0.35, 1.25, compactPeak / localY);
  let positiveHighFrequency = max(centre - pairXY, vec3<f32>(0.0));
  return max(
    centre - 0.90 * criticalWeight * starMask * positiveHighFrequency,
    vec3<f32>(0.0)
  );
}

fn sampleEnvironment(
  direction: vec3<f32>,
  observerShift: f32,
  criticalWeight: f32
) -> vec3<f32> {
  let d = safeNormalize(rotateAroundY(direction, params.cameraRightSkyRotation.w));
  let longitude = atan2(d.z, d.x);
  let latitude = asin(clamp(d.y, -1.0, 1.0));
  let uv = vec2<f32>(
    fract(longitude / TWO_PI + 0.5),
    clamp(0.5 - latitude / PI, 0.00001, 0.99999)
  );

  // Lensing itself preserves surface brightness.  The procedural stars have
  // a shifted blackbody spectrum; observerShift^4 is the common bolometric
  // gravitational/kinematic transfer for both sky components.
  let rawPanorama = sampleSkyLevelZero(uv);
  let filteredPanorama = filterSkyPanorama(uv, rawPanorama, criticalWeight);
  let panorama = suppressBakedStarPsf(
    uv,
    filteredPanorama,
    clamp(criticalWeight, 0.0, 1.0)
  );
  let shift2 = observerShift * observerShift;
  let calibratedPanorama = max(panorama - vec3<f32>(0.002), vec3<f32>(0.0));
  return (
    calibratedPanorama * max(params.displayOutput.w, 0.01)
    + proceduralStars(d, observerShift, criticalWeight)
  ) * shift2 * shift2;
}

fn diskNoiseField(
  advectedPhase: f32,
  radius: f32,
  generation: f32
) -> vec3<f32> {
  let seed = vec3<f32>(
    generation,
    generation * 0.7548777 + 19.31,
    generation * 1.3247180 - 7.17
  );
  let phaseOffset = TWO_PI * hash31(seed + vec3<f32>(3.1, 11.7, 29.3));
  let offsetAngle = TWO_PI * hash31(seed + vec3<f32>(41.9, 5.3, 17.1));
  let offsetLength = 29.0 * hash31(seed + vec3<f32>(7.7, 31.1, 13.9));
  let fieldOffset = offsetLength * vec2<f32>(cos(offsetAngle), sin(offsetAngle));
  let seedZ = 23.0 * hash31(seed + vec3<f32>(53.1, 2.9, 37.7));
  let phase = advectedPhase + phaseOffset;
  let materialPosition = radius * vec2<f32>(cos(phase), sin(phase)) + fieldOffset;

  // Domain-warped Cartesian cells model finite MRI eddies.  Unlike a noise
  // field indexed mainly by radius, these cells cannot close into concentric
  // contour rings; differential rotation naturally stretches them into short
  // orbit-aligned clouds as they age.
  let warp = vec2<f32>(
    2.0 * valueNoise3(vec3<f32>(materialPosition * 0.12, seedZ + 1.7)) - 1.0,
    2.0 * valueNoise3(vec3<f32>(materialPosition * 0.12 + vec2<f32>(9.3, -6.1), seedZ + 7.9)) - 1.0
  );
  let warpedPosition = materialPosition + 1.8 * warp;
  let cloud = 2.0 * valueNoise3(vec3<f32>(
    warpedPosition * 0.28,
    seedZ + 13.1
  )) - 1.0;
  let strandNoise = 2.0 * valueNoise3(vec3<f32>(
    warpedPosition * 0.72 + 0.45 * warp,
    seedZ + 29.7
  )) - 1.0;
  let fine = 2.0 * valueNoise3(vec3<f32>(
    warpedPosition * 1.75 + 0.72 * warp,
    seedZ + 47.3
  )) - 1.0;
  return vec3<f32>(cloud, strandNoise, fine);
}

fn accretionDiskSample(
  hitPosition: vec3<f32>,
  diskNormal: vec3<f32>,
  traceDirection: vec3<f32>,
  lambdaZ: f32,
  observerShift: f32,
  travelDelay: f32
) -> vec4<f32> {
  let height = dot(hitPosition, diskNormal);
  let planarPosition = hitPosition - height * diskNormal;
  let radius = length(planarPosition);
  let outerRadius = max(params.cameraUpDiskOuter.w, ISCO + 0.5);
  let x = ISCO / max(radius, ISCO);
  let fluxShapeRaw = x * x * x * max(1.0 - sqrt(x), 0.0);
  let xPeak = 36.0 / 49.0;
  let fluxPeak = xPeak * xPeak * xPeak * (1.0 - sqrt(xPeak));
  let innerFade = smoothstep(ISCO, ISCO + 0.35, radius);
  let outerFade = 1.0 - smoothstep(outerRadius * 0.82, outerRadius, radius);
  let fluxShape = (fluxShapeRaw / fluxPeak) * innerFade * outerFade;

  var referenceAxis = vec3<f32>(0.0, 1.0, 0.0);
  if (abs(dot(referenceAxis, diskNormal)) > 0.92) {
    referenceAxis = vec3<f32>(1.0, 0.0, 0.0);
  }
  let diskX = safeNormalize(cross(referenceAxis, diskNormal));
  let diskY = cross(diskNormal, diskX);
  let azimuth = atan2(dot(planarPosition, diskY), dot(planarPosition, diskX));
  let omega = inverseSqrt(max(radius * radius * radius, 1.0e-8));

  let massSolar = max(params.resolutionTimeMass.w, 1.0);
  let timeGeometric = params.resolutionTimeMass.z;

  // MRI turbulence has a finite coherence time.  Cross-fading two advected
  // generations prevents Kepler shear from growing without bound while both
  // fields still obey d(phi)/dt = Omega and use the ray's retarded time.
  let retardedTime = timeGeometric - travelDelay;
  let coherenceTime = 32.0;
  let generation = floor(retardedTime / coherenceTime);
  let generationBlend = fract(retardedTime / coherenceTime);
  let ageA = retardedTime - generation * coherenceTime;
  let ageB = ageA - coherenceTime;
  let fieldA = diskNoiseField(
    azimuth - omega * ageA,
    radius,
    generation
  );
  let fieldB = diskNoiseField(
    azimuth - omega * ageB,
    radius,
    generation + 1.0
  );
  let blendWeight = smoothstep(0.0, 1.0, generationBlend);
  let varianceCompensation = min(inverseSqrt(
    (1.0 - blendWeight) * (1.0 - blendWeight)
    + blendWeight * blendWeight
  ), 1.25);
  let turbulence = mix(fieldA, fieldB, blendWeight) * varianceCompensation;
  let cloud = turbulence.x;
  let strandNoise = turbulence.y;
  let fine = turbulence.z;

  let filamentRidge = smoothstep(
    0.05,
    0.82,
    0.62 * strandNoise + 0.38 * cloud
  );
  // Surface density keeps the turbulent hierarchy.  The artificial m=2 wave is
  // removed, and the smaller high-frequency weights below keep the effective
  // temperature smoother than the density filaments; T responds to the local
  // dissipation through the fourth root further down.
  let densityContrast = clamp(
    0.62 * cloud + 0.30 * strandNoise + 0.12 * fine
    + 0.16 * (filamentRidge - 0.42),
    -0.85,
    0.85
  );
  let localHeating = exp(clamp(
    0.45 * cloud + 0.30 * strandNoise + 0.14 * fine
    + 0.12 * (filamentRidge - 0.42),
    -0.50,
    0.58
  ));

  // A marginally optically thick surface layer keeps foreground absorption
  // while allowing low-density MRI lanes and secondary images to retain depth.
  let tauMean = 1.45
              * pow(max(radius / 8.17, 0.1), -0.62)
              * innerFade * outerFade;
  let tauFace = tauMean * exp(0.75 * densityContrast);
  let covering = mix(
    0.82,
    1.0,
    smoothstep(-0.75, 0.75, densityContrast)
  );

  let accretion = max(params.renderControls.x, 1.0e-6);
  // The UI ratio is L/L_Edd with Mdot_Edd = L_Edd/(0.1 c^2).  This
  // normalisation gives about 4500 K for the default 6.46e9 Msun,
  // 6.3e-5-Edd disk; temperature still follows (Mdot/M)^(1/4).
  let peakTemperature = 1.43e5 * pow(accretion * 1.0e8 / massSolar, 0.25);
  let emittedTemperature = max(
    600.0,
    peakTemperature * pow(max(fluxShape * localHeating, 1.0e-8), 0.25)
  );

  // Electron scattering hardens hot, optically thick zones without changing
  // their bolometric flux (planckChromaticity is luminance-normalised).
  let spectralHardening = 1.0
    + 0.15 * smoothstep(8000.0, 30000.0, emittedTemperature)
           * smoothstep(0.25, 2.0, tauFace);

  // Exact circular-orbit frequency transfer in Schwarzschild.  The numerator
  // contains the moving observer's gravitational + SR Doppler factor.
  let emitterUt = inverseSqrt(max(1.0 - 3.0 / radius, 1.0e-5));
  let orbitalDenominator = max(1.0 - omega * lambdaZ, 0.015);
  let transferDenominator = emitterUt * orbitalDenominator;
  let g = clamp(observerShift / transferDenominator, 0.04, 8.0);
  let observedTemperature = emittedTemperature * spectralHardening * g;
  let g2 = g * g;
  let bolometricTransfer = g2 * g2;

  // Convert the traced static-tetrad ray angle to the circular emitter frame.
  // This controls both line-of-sight optical depth and thick-slab limb darkening.
  let localLapse = sqrt(max(1.0 - 2.0 / radius, 1.0e-5));
  let emitterEnergyOverStatic = localLapse * emitterUt * orbitalDenominator;
  let muEmit = clamp(
    abs(dot(traceDirection, diskNormal)) / max(emitterEnergyOverStatic, 1.0e-4),
    0.03,
    1.0
  );
  let tauLineOfSight = min(tauFace / muEmit, 20.0);
  let opacity = clamp(covering * (1.0 - exp(-tauLineOfSight)), 0.0, 1.0);
  let thickLimb = (1.0 + 2.06 * muEmit) / 3.06;
  let limbDarkening = mix(1.0, thickLimb, smoothstep(0.25, 1.5, tauFace));

  // The colour is luminance-normalised, so g^4 is applied exactly once.
  let radiance = max(fluxShape, 0.0)
               * localHeating * accretion * bolometricTransfer * limbDarkening;
  // Absolute luminosity cannot be represented by a display texture; 6500 is a
  // single global camera calibration, while all radial and frequency ratios
  // above remain physical.
  let source = planckChromaticity(observedTemperature) * radiance * 6500.0;
  return vec4<f32>(source * opacity, opacity);
}

fn aberratePastRay(comovingDirection: vec3<f32>) -> vec3<f32> {
  let beta = clamp(params.observerVelocityBeta.w, 0.0, 0.95);
  let velocityLength = length(params.observerVelocityBeta.xyz);
  if (beta <= 1.0e-6 || velocityLength <= 1.0e-6) {
    return safeNormalize(comovingDirection);
  }

  let velocityDirection = params.observerVelocityBeta.xyz / velocityLength;
  let mu = dot(comovingDirection, velocityDirection);
  let gamma = inverseSqrt(max(1.0 - beta * beta, 1.0e-6));
  let perpendicular = comovingDirection - mu * velocityDirection;
  let denominator = max(gamma * (1.0 - beta * mu), 1.0e-6);
  // Lorentz-transform k' = (-1,n') from the comoving tetrad to the local
  // Schwarzschild static tetrad, then renormalise its past-directed time part.
  return safeNormalize((perpendicular + gamma * (mu - beta) * velocityDirection) / denominator);
}

fn observerFrequencyShift(comovingDirection: vec3<f32>, staticLapse: f32) -> f32 {
  let beta = clamp(params.observerVelocityBeta.w, 0.0, 0.95);
  let velocityLength = length(params.observerVelocityBeta.xyz);
  if (beta <= 1.0e-6 || velocityLength <= 1.0e-6) {
    return 1.0 / staticLapse;
  }
  let velocityDirection = params.observerVelocityBeta.xyz / velocityLength;
  let mu = dot(comovingDirection, velocityDirection);
  let gamma = inverseSqrt(max(1.0 - beta * beta, 1.0e-6));
  // E_comoving / E_infinity for the same null ray.
  return 1.0 / max(staticLapse * gamma * (1.0 - beta * mu), 1.0e-6);
}

fn traceSchwarzschild(
  comovingRay: vec3<f32>,
  criticalWeight: f32
) -> vec3<f32> {
  let suppliedRadius = params.cameraPosRadius.w;
  let rawCameraLength = length(params.cameraPosRadius.xyz);
  let cameraRadius = max(select(rawCameraLength, suppliedRadius, suppliedRadius > 2.001), 2.001);
  let cameraPosition = safeNormalize(params.cameraPosRadius.xyz) * cameraRadius;
  let radialBasis = cameraPosition / cameraRadius;
  let lapseSquared = max(1.0 - 2.0 / cameraRadius, 1.0e-6);
  let staticLapse = sqrt(lapseSquared);
  let observerShift = observerFrequencyShift(comovingRay, staticLapse);
  let ray = aberratePastRay(comovingRay);

  let radialDirection = dot(ray, radialBasis);
  let tangentVector = ray - radialDirection * radialBasis;
  let tangentLength = length(tangentVector);

  if (tangentLength < 1.0e-6) {
    if (radialDirection < 0.0) {
      return vec3<f32>(0.0);
    }
    return sampleEnvironment(radialBasis, observerShift, criticalWeight);
  }

  let tangentBasis = tangentVector / tangentLength;
  let impact = cameraRadius * tangentLength / staticLapse;
  let environmentFilterWeight = max(
    criticalWeight,
    0.68 * (1.0 - smoothstep(0.30, 2.40, abs(impact - PHOTON_IMPACT)))
  );
  var inverseRadius = 1.0 / cameraRadius;
  var inverseRadiusDerivative = -inverseRadius * staticLapse * radialDirection / tangentLength;
  var planeAngle = 0.0;
  var directionOnPlane = radialBasis;

  let diskNormal = safeNormalize(vec3<f32>(0.0, 1.0, 0.0));
  let outerRadius = max(params.cameraUpDiskOuter.w, ISCO + 0.5);
  // For a backwards ray n, the future-directed photon arriving at the camera
  // has local spatial momentum -n; hence the minus sign in Lz/E.
  let lambdaZ = -dot(cross(cameraPosition, ray), diskNormal) / staticLapse;

  var previousDiskSide = dot(directionOnPlane, diskNormal);
  var travelDelay = 0.0;
  var accumulatedRadiance = vec3<f32>(0.0);
  var throughput = 1.0;
  var activeSteps = clamp(i32(params.renderControls.w + 0.5), 1, MAX_STEPS);
  if (abs(impact - PHOTON_IMPACT) < 0.45) {
    activeSteps = MAX_STEPS;
  }
  let stepAngle = 0.030;
  let horizonInverseRadius = 0.5 / (1.0 + 2.0e-4);

  for (var step = 0; step < MAX_STEPS; step = step + 1) {
    if (step >= activeSteps) {
      break;
    }

    let previousU = inverseRadius;
    let previousV = inverseRadiusDerivative;
    let previousAngle = planeAngle;
    let previousDirection = directionOnPlane;

    // Störmer-Verlet update of u'' = -u + 3u^2.
    let halfV = previousV + 0.5 * stepAngle * schwarzschildForce(previousU);
    let nextU = previousU + stepAngle * halfV;
    let nextV = halfV + 0.5 * stepAngle * schwarzschildForce(nextU);
    let nextAngle = previousAngle + stepAngle;
    let nextDirection = cos(nextAngle) * radialBasis + sin(nextAngle) * tangentBasis;
    let nextDiskSide = dot(nextDirection, diskNormal);

    var boundaryFraction = 2.0;
    var crossedHorizon = false;
    var escaped = false;
    if (nextU >= horizonInverseRadius && nextU != previousU) {
      boundaryFraction = clamp((horizonInverseRadius - previousU) / (nextU - previousU), 0.0, 1.0);
      crossedHorizon = true;
    }
    if (nextU <= 0.0 && nextU != previousU) {
      let escapeFraction = clamp(previousU / (previousU - nextU), 0.0, 1.0);
      if (escapeFraction < boundaryFraction) {
        boundaryFraction = escapeFraction;
        crossedHorizon = false;
        escaped = true;
      }
    }

    let midpointU = max(0.5 * (previousU + max(nextU, 1.0e-6)), 1.0e-5);
    let midpointLapse = max(1.0 - 2.0 * midpointU, 1.0e-5);
    let delayIncrement = stepAngle / max(impact * midpointU * midpointU * midpointLapse, 1.0e-7);

    if (
      previousDiskSide * nextDiskSide <= 0.0
      && abs(previousDiskSide - nextDiskSide) > 1.0e-7
    ) {
      let diskFraction = clamp(
        previousDiskSide / (previousDiskSide - nextDiskSide),
        0.0,
        1.0
      );
      if (diskFraction > 1.0e-5 && diskFraction < boundaryFraction) {
        let hitU = mix(previousU, nextU, diskFraction);
        if (hitU > 0.0) {
          let hitRadius = 1.0 / hitU;
          let hitAngle = mix(previousAngle, nextAngle, diskFraction);
          let hitDirection = safeNormalize(mix(previousDirection, nextDirection, diskFraction));
          let hitPosition = hitDirection * hitRadius;
          let hitHeight = dot(hitPosition, diskNormal);
          let hitPlanarRadius = length(hitPosition - hitHeight * diskNormal);
          if (hitPlanarRadius >= ISCO && hitPlanarRadius <= outerRadius) {
            let hitDerivative = mix(previousV, nextV, diskFraction);
            let tangentAtHit = -sin(hitAngle) * radialBasis + cos(hitAngle) * tangentBasis;
            let hitLapse = sqrt(max(1.0 - 2.0 * hitU, 1.0e-5));
            let tangentialDirection = clamp(impact * hitU * hitLapse, 0.0, 1.0);
            let radialMagnitude = sqrt(max(1.0 - tangentialDirection * tangentialDirection, 0.0));
            let radialSign = select(1.0, -1.0, hitDerivative > 0.0);
            let traceDirectionAtHit = safeNormalize(
              radialSign * radialMagnitude * hitDirection
              + tangentialDirection * tangentAtHit
            );
            let diskSample = accretionDiskSample(
              hitPosition,
              diskNormal,
              traceDirectionAtHit,
              lambdaZ,
              observerShift,
              travelDelay + diskFraction * delayIncrement
            );
            accumulatedRadiance = accumulatedRadiance + throughput * diskSample.rgb;
            throughput = throughput * (1.0 - diskSample.a);
            if (throughput < 0.02) {
              return accumulatedRadiance;
            }
          }
        }
      }
    }

    if (crossedHorizon) {
      return accumulatedRadiance;
    }
    if (escaped) {
      let escapeAngle = mix(previousAngle, nextAngle, boundaryFraction);
      let escapeDirection = cos(escapeAngle) * radialBasis + sin(escapeAngle) * tangentBasis;
      return accumulatedRadiance
           + throughput * sampleEnvironment(
               escapeDirection,
               observerShift,
               environmentFilterWeight
             );
    }

    inverseRadius = nextU;
    inverseRadiusDerivative = nextV;
    planeAngle = nextAngle;
    directionOnPlane = nextDirection;
    previousDiskSide = nextDiskSide;
    travelDelay = travelDelay + delayIncrement;
  }

  // Critical rays receive 384 angular steps. Captured rays contribute only
  // radiation accumulated at real disk crossings; unresolved outward rays use
  // their current asymptotic direction rather than a painted photon ring.
  if (impact < PHOTON_IMPACT && radialDirection < 0.0 && inverseRadiusDerivative >= 0.0) {
    return accumulatedRadiance;
  }
  return accumulatedRadiance
       + throughput * sampleEnvironment(
           directionOnPlane,
           observerShift,
           environmentFilterWeight
         );
}

fn cameraRayForScreen(screen: vec2<f32>, tanHalfFov: f32) -> vec3<f32> {
  return safeNormalize(
    params.cameraForwardFov.xyz
    + tanHalfFov * screen.x * params.cameraRightSkyRotation.xyz
    + tanHalfFov * screen.y * params.cameraUpDiskOuter.xyz
  );
}

fn initialImpact(comovingRay: vec3<f32>) -> f32 {
  let cameraRadius = max(params.cameraPosRadius.w, 2.001);
  let radialBasis = safeNormalize(params.cameraPosRadius.xyz);
  let staticLapse = sqrt(max(1.0 - 2.0 / cameraRadius, 1.0e-6));
  let staticRay = aberratePastRay(comovingRay);
  let radialComponent = dot(staticRay, radialBasis);
  return cameraRadius * sqrt(max(1.0 - radialComponent * radialComponent, 0.0))
       / staticLapse;
}

@fragment
fn fsMain(input: FragmentInput) -> @location(0) vec4<f32> {
  let resolution = max(params.resolutionTimeMass.xy, vec2<f32>(1.0));
  let aspect = resolution.x / resolution.y;
  let screen = vec2<f32>(
    (input.uv.x * 2.0 - 1.0) * aspect,
    1.0 - input.uv.y * 2.0
  );
  let tanHalfFov = tan(0.5 * clamp(params.cameraForwardFov.w, 0.02, 2.8));
  let centreRay = cameraRayForScreen(screen, tanHalfFov);
  let centreImpact = initialImpact(centreRay);
  let impactFootprint = max(fwidth(centreImpact), 1.0e-5);
  let criticalDistancePixels = abs(centreImpact - PHOTON_IMPACT)
                             / impactFootprint;
  let criticalWeight = 1.0 - smoothstep(
    3.0,
    8.0,
    criticalDistancePixels
  );

  var color = vec3<f32>(0.0);
  if (abs(centreImpact - PHOTON_IMPACT) < 8.0 * impactFootprint) {
    // Fixed 2x2 coverage only in the narrow critical band.  It resolves a
    // sub-pixel point source into a stable, thin lensing arc without paying
    // four geodesics for the rest of the image or introducing temporal noise.
    let pixelSpan = 2.0 / resolution.y;
    let offsets = array<vec2<f32>, 4>(
      vec2<f32>(-0.25, -0.25),
      vec2<f32>( 0.25, -0.25),
      vec2<f32>(-0.25,  0.25),
      vec2<f32>( 0.25,  0.25)
    );
    for (var sampleIndex = 0; sampleIndex < 4; sampleIndex = sampleIndex + 1) {
      color = color + 0.25 * traceSchwarzschild(
        cameraRayForScreen(screen + offsets[sampleIndex] * pixelSpan, tanHalfFov),
        criticalWeight
      );
    }
  } else {
    color = traceSchwarzschild(centreRay, 0.0);
  }
  return vec4<f32>(max(color, vec3<f32>(0.0)), 1.0);
}
`;

export const postFragmentWGSL = /* wgsl */ `
struct Params {
  resolutionTimeMass: vec4<f32>,
  renderControls: vec4<f32>,
  cameraPosRadius: vec4<f32>,
  cameraForwardFov: vec4<f32>,
  cameraRightSkyRotation: vec4<f32>,
  cameraUpDiskOuter: vec4<f32>,
  postMotionFrame: vec4<f32>,
  observerVelocityBeta: vec4<f32>,
  displayOutput: vec4<f32>,
};

struct FragmentInput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tScene: texture_2d<f32>;
@group(0) @binding(2) var sceneSampler: sampler;

fn sampleScene(uv: vec2<f32>) -> vec3<f32> {
  return max(
    textureSampleLevel(tScene, sceneSampler, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb,
    vec3<f32>(0.0)
  );
}

fn luminance(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn brightPart(color: vec3<f32>, threshold: f32) -> vec3<f32> {
  let light = luminance(color);
  let knee = max(0.25 * threshold, 1.0e-5);
  let soft = clamp(light - threshold + knee, 0.0, 2.0 * knee);
  let softContribution = soft * soft / (4.0 * knee);
  let contribution = max(light - threshold, softContribution);
  return color * contribution / max(light, 1.0e-5);
}

fn acesFitted(color: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  let positive = max(color, vec3<f32>(0.0));
  let mapped = clamp((positive * (a * positive + vec3<f32>(b))) /
                     (positive * (c * positive + vec3<f32>(d)) + vec3<f32>(e)),
                     vec3<f32>(0.0), vec3<f32>(1.0));
  // Re-introduce part of the scene-linear chromaticity after the per-channel
  // fit.  This keeps hot Doppler highlights coloured instead of driving all
  // three channels to identical display white, while the gamut scale prevents
  // clipping from becoming a second hard shoulder.
  let sourceLight = max(luminance(positive), 1.0e-5);
  let mappedLight = luminance(mapped);
  var huePreserved = positive * (mappedLight / sourceLight);
  let hueMaximum = max(max(huePreserved.r, huePreserved.g), huePreserved.b);
  huePreserved = huePreserved / max(hueMaximum, 1.0);
  let chromaWeight = 0.34 * smoothstep(0.10, 0.85, mappedLight);
  return clamp(
    mix(mapped, huePreserved, chromaWeight),
    vec3<f32>(0.0),
    vec3<f32>(1.0)
  );
}

fn linearSrgbToDisplayP3(color: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    0.82259287 * color.r + 0.17753395 * color.g,
    0.03319951 * color.r + 0.96678350 * color.g,
    0.01708535 * color.r + 0.07239572 * color.g + 0.91030148 * color.b
  );
}

fn encodeSrgbTransfer(color: vec3<f32>) -> vec3<f32> {
  let positive = max(color, vec3<f32>(0.0));
  let low = positive * 12.92;
  let high = 1.055 * pow(positive, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
  return select(high, low, positive <= vec3<f32>(0.0031308));
}

fn extendedHdrShoulder(color: vec3<f32>, peak: f32) -> vec3<f32> {
  let safePeak = max(peak, 1.01);
  let positive = max(color, vec3<f32>(0.0));
  let brightest = max(max(positive.r, positive.g), positive.b);
  if (brightest <= 1.0) {
    return positive;
  }
  let headroom = safePeak - 1.0;
  let excess = brightest - 1.0;
  // A rational shoulder has unit slope at SDR white and approaches the panel
  // peak more slowly than the previous exponential.  Scaling all channels by
  // the same factor preserves both hue and fine gradients on the approaching
  // side of the disk.
  let mappedBrightest = 1.0
    + headroom * excess / (excess + headroom);
  return positive * (mappedBrightest / brightest);
}

fn hashPixel(pixel: vec2<f32>, frame: f32) -> f32 {
  let p = vec3<f32>(pixel, frame + 1.0);
  return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
}

@fragment
fn fsMain(input: FragmentInput) -> @location(0) vec4<f32> {
  let resolution = max(params.resolutionTimeMass.xy, vec2<f32>(1.0));
  let texel = vec2<f32>(1.0) / resolution;
  let mode = clamp(params.renderControls.z, 0.0, 1.0);
  let exposure = max(params.renderControls.y, 0.0);
  // Keep the native ray-traced sample untouched: a real Hubble PSF is far
  // below one pixel at this field of view.  Only bright sources receive a
  // restrained, additive telescope halo.
  var color = sampleScene(input.uv) * exposure;
  let bloomStrength = max(params.postMotionFrame.y, 0.0);
  if (bloomStrength > 1.0e-5) {
    let cssScale = max(params.postMotionFrame.x, 0.5);
    let bloomRadius = texel * cssScale * mix(2.0, 3.0, mode);
    let bloomSamples = brightPart(sampleScene(input.uv + vec2<f32>( bloomRadius.x, 0.0)) * exposure, 1.0)
                     + brightPart(sampleScene(input.uv + vec2<f32>(-bloomRadius.x, 0.0)) * exposure, 1.0)
                     + brightPart(sampleScene(input.uv + vec2<f32>(0.0,  bloomRadius.y)) * exposure, 1.0)
                     + brightPart(sampleScene(input.uv + vec2<f32>(0.0, -bloomRadius.y)) * exposure, 1.0)
                     + 0.5 * (
                         brightPart(sampleScene(input.uv + bloomRadius) * exposure, 1.0)
                         + brightPart(sampleScene(input.uv - bloomRadius) * exposure, 1.0)
                         + brightPart(sampleScene(input.uv + vec2<f32>(bloomRadius.x, -bloomRadius.y)) * exposure, 1.0)
                         + brightPart(sampleScene(input.uv + vec2<f32>(-bloomRadius.x, bloomRadius.y)) * exposure, 1.0)
                       );
    color = color + bloomSamples * (bloomStrength / 24.0);
  }

  // Hubble mode changes only the display colour response and PSF.  It never
  // changes the geodesic, shadow, disk intersection, or frequency transfer.
  let light = luminance(color);
  let hubbleColour = mix(vec3<f32>(light), color, 1.06)
                    * vec3<f32>(1.018, 1.0, 0.985);
  color = max(mix(color, hubbleColour, mode), vec3<f32>(0.0));

  let extendedHdr = params.displayOutput.x > 0.5;
  let displayP3 = params.displayOutput.y > 0.5;
  if (extendedHdr) {
    color = extendedHdrShoulder(color, params.displayOutput.z);
    if (displayP3) {
      color = max(linearSrgbToDisplayP3(color), vec3<f32>(0.0));
    }
    // WebGPU's predefined canvas colour spaces consume their encoded RGB
    // values even for rgba16float.  Values above encoded 1.0 are handed to
    // macOS EDR instead of being clipped by the extended tone-mapping mode.
    color = encodeSrgbTransfer(color);
    return vec4<f32>(color, 1.0);
  }

  color = acesFitted(color);
  if (displayP3) {
    color = max(linearSrgbToDisplayP3(color), vec3<f32>(0.0));
  }
  color = encodeSrgbTransfer(color);

  let dither = (hashPixel(input.position.xy, params.postMotionFrame.w) - 0.5) / 255.0;
  color = clamp(color + vec3<f32>(dither), vec3<f32>(0.0), vec3<f32>(1.0));
  return vec4<f32>(color, 1.0);
}
`;

export const fullscreenVertexGLSL = /* glsl */ `
precision highp float;

varying vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const traceFragmentGLSL = /* glsl */ `
precision highp float;
precision highp int;

uniform vec2 uResolution;
uniform float uTime;
uniform float uMassSolar;
uniform float uAccretion;
uniform float uExposure;
uniform float uMode;
uniform float uSteps;
uniform vec3 uCameraPos;
uniform float uCameraRadius;
uniform vec3 uForward;
uniform float uFov;
uniform vec3 uRight;
uniform float uSkyRotation;
uniform vec3 uUp;
uniform float uDiskOuterRadius;
uniform float uRenderScale;
uniform float uBloom;
uniform float uMotion;
uniform float uFrame;
uniform vec3 uObserverVelocity;
uniform float uObserverBeta;
uniform float uSkyRadianceScale;
uniform sampler2D tSky;

varying vec2 vUv;

const float PI = 3.14159265358979323846;
const float TWO_PI = 6.28318530717958647692;
const int MAX_STEPS = 384;
const float ISCO = 6.0;
const float PHOTON_IMPACT = 5.196152422706632;

vec3 safeNormalize(vec3 v) {
  return v * inversesqrt(max(dot(v, v), 1.0e-18));
}

float schwarzschildForce(float u) {
  return -u + 3.0 * u * u;
}

float hash31(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

float valueNoise3(vec3 value) {
  vec3 cell = floor(value);
  vec3 f = fract(value);
  f = f * f * (3.0 - 2.0 * f);

  float n000 = hash31(cell + vec3(0.0, 0.0, 0.0));
  float n100 = hash31(cell + vec3(1.0, 0.0, 0.0));
  float n010 = hash31(cell + vec3(0.0, 1.0, 0.0));
  float n110 = hash31(cell + vec3(1.0, 1.0, 0.0));
  float n001 = hash31(cell + vec3(0.0, 0.0, 1.0));
  float n101 = hash31(cell + vec3(1.0, 0.0, 1.0));
  float n011 = hash31(cell + vec3(0.0, 1.0, 1.0));
  float n111 = hash31(cell + vec3(1.0, 1.0, 1.0));

  float nx00 = mix(n000, n100, f.x);
  float nx10 = mix(n010, n110, f.x);
  float nx01 = mix(n001, n101, f.x);
  float nx11 = mix(n011, n111, f.x);
  return mix(mix(nx00, nx10, f.y), mix(nx01, nx11, f.y), f.z);
}

float diskTurbulence(vec3 p) {
  float result = 0.0;
  float amplitude = 0.5714286;
  for (int octave = 0; octave < 3; ++octave) {
    result += amplitude * valueNoise3(p);
    p = p * 2.07 + vec3(7.1, 13.7, 3.9);
    amplitude *= 0.5;
  }
  return result;
}

vec3 rotateAroundY(vec3 d, float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return vec3(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
}

vec3 cubeStarCoordinates(vec3 direction) {
  vec3 d = safeNormalize(direction);
  vec3 a = abs(d);
  vec2 faceUv;
  float face;
  if (a.x >= a.y && a.x >= a.z) {
    if (d.x >= 0.0) {
      faceUv = vec2(-d.z, d.y) / a.x;
      face = 0.0;
    } else {
      faceUv = vec2(d.z, d.y) / a.x;
      face = 1.0;
    }
  } else if (a.y >= a.z) {
    if (d.y >= 0.0) {
      faceUv = vec2(d.x, -d.z) / a.y;
      face = 2.0;
    } else {
      faceUv = vec2(d.x, d.z) / a.y;
      face = 3.0;
    }
  } else if (d.z >= 0.0) {
    faceUv = vec2(d.x, d.y) / a.z;
    face = 4.0;
  } else {
    faceUv = vec2(-d.x, d.y) / a.z;
    face = 5.0;
  }
  return vec3(faceUv * 0.5 + 0.5, face);
}

vec3 planckChromaticity(float temperatureKelvin) {
  vec3 wavelengthsMicron = vec3(0.640, 0.530, 0.460);
  vec3 lambda2 = wavelengthsMicron * wavelengthsMicron;
  vec3 lambda5 = lambda2 * lambda2 * wavelengthsMicron;
  vec3 exponent = min(vec3(80.0), vec3(14387.77 / max(temperatureKelvin, 100.0)) / wavelengthsMicron);
  vec3 spectrum = vec3(1.0) / (lambda5 * max(exp(exponent) - vec3(1.0), vec3(1.0e-12)));
  spectrum *= vec3(1.2320, 1.0, 0.9367);
  float light = max(dot(spectrum, vec3(0.2126, 0.7152, 0.0722)), 1.0e-8);
  return spectrum / light;
}

vec3 proceduralStars(
  vec3 direction,
  float observerShift,
  float criticalWeight
) {
  vec3 mapped = cubeStarCoordinates(direction);
  vec3 result = vec3(0.0);
  for (int layer = 0; layer < 2; ++layer) {
    float grid = 192.0;
    float threshold = 0.976;
    float coreRadius = 0.040;
    float minimumRadiance = 0.18;
    float maximumRadiance = 8.0;
    if (layer == 1) {
      grid = 384.0;
      threshold = 0.990;
      coreRadius = 0.055;
      minimumRadiance = 0.055;
      maximumRadiance = 2.2;
    }

    vec2 coordinates = mapped.xy * grid;
    vec2 cell = floor(coordinates);
    vec2 local = fract(coordinates);
    float layerIndex = float(layer);
    vec3 seed = vec3(
      cell + vec2(layerIndex * 17.0, layerIndex * 29.0),
      mapped.z * 37.0 + layerIndex * 11.0
    );
    vec3 random = vec3(
      hash31(seed),
      hash31(seed + vec3(19.17, 7.31, 3.13)),
      hash31(seed + vec3(43.71, 31.97, 17.53))
    );
    if (random.z > threshold) {
      vec2 centre = vec2(0.5) + (random.xy - 0.5) * 0.5;
      vec2 delta = local - centre;
      float distanceToStar = length(delta);
      vec2 radialDirection = delta / max(distanceToStar, 1.0e-6);
      float footprint = 0.5 * (
        abs(dot(radialDirection, dFdx(coordinates)))
        + abs(dot(radialDirection, dFdy(coordinates)))
      );
      float regularAaWidth = min(
        clamp(footprint, 0.006, 0.045),
        0.90 * coreRadius
      );
      float criticalAaWidth = min(0.014, 0.35 * coreRadius);
      float antialiasWidth = mix(
        regularAaWidth,
        criticalAaWidth,
        clamp(criticalWeight, 0.0, 1.0)
      );
      float core = 1.0 - smoothstep(
        coreRadius - antialiasWidth,
        coreRadius + antialiasWidth,
        distanceToStar
      );
      float rank = clamp((random.z - threshold) / (1.0 - threshold), 0.0, 1.0);
      float haloRadius = coreRadius * mix(3.2, 2.2, rank);
      float halo = pow(max(1.0 - distanceToStar / haloRadius, 0.0), 3.0);
      float psf = core + 0.10 * rank * halo
                * (1.0 - clamp(criticalWeight, 0.0, 1.0));
      float stellarTemperature = mix(
        3000.0,
        11000.0,
        pow(hash31(seed + vec3(91.7, 53.1, 27.9)), 0.75)
      ) * observerShift;
      float radiance = mix(minimumRadiance, maximumRadiance, pow(rank, 0.42));
      result += planckChromaticity(stellarTemperature)
                * radiance * psf;
    }
  }
  return result;
}

float skyLuminance(vec3 color) {
  return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 sampleSkyLevelZero(vec2 skyUv) {
  return max(textureLod(tSky, skyUv, 0.0).rgb, vec3(0.0));
}

vec3 filterSkyPanorama(
  vec2 skyUv,
  vec3 centre,
  float criticalWeight
) {
  float weight = 0.62 * smoothstep(
    0.08,
    0.92,
    clamp(criticalWeight, 0.0, 1.0)
  );
  if (weight <= 0.001) {
    return centre;
  }
  vec2 dimensions = vec2(textureSize(tSky, 0));
  vec2 texel = 1.0 / dimensions;
  float radius = mix(1.0, 10.0, clamp(criticalWeight, 0.0, 1.0));
  vec3 filtered = 0.25 * (
    sampleSkyLevelZero(skyUv + vec2(radius * texel.x, 0.0))
    + sampleSkyLevelZero(skyUv - vec2(radius * texel.x, 0.0))
    + sampleSkyLevelZero(skyUv + vec2(0.0, radius * texel.y))
    + sampleSkyLevelZero(skyUv - vec2(0.0, radius * texel.y))
  );
  return mix(centre, filtered, weight);
}

vec3 suppressBakedStarPsf(
  vec2 skyUv,
  vec3 centre,
  float criticalWeight
) {
  float centreY = skyLuminance(centre);
  if (criticalWeight <= 0.001 || centreY < 0.008) {
    return centre;
  }

  vec2 texel = 1.0 / vec2(textureSize(tSky, 0));
  vec2 offset = 2.5 * texel;
  vec3 pairX = 0.5 * (
    sampleSkyLevelZero(skyUv + vec2(offset.x, 0.0))
    + sampleSkyLevelZero(skyUv - vec2(offset.x, 0.0))
  );
  vec3 pairY = 0.5 * (
    sampleSkyLevelZero(skyUv + vec2(0.0, offset.y))
    + sampleSkyLevelZero(skyUv - vec2(0.0, offset.y))
  );
  vec3 pairXY = 0.5 * (pairX + pairY);
  float compactPeak = max(min(
    centreY - skyLuminance(pairX),
    centreY - skyLuminance(pairY)
  ), 0.0);
  float localY = max(skyLuminance(pairXY), 0.012);
  float starMask = smoothstep(0.004, 0.025, compactPeak)
                 * smoothstep(0.35, 1.25, compactPeak / localY);
  vec3 positiveHighFrequency = max(centre - pairXY, vec3(0.0));
  return max(
    centre - 0.90 * criticalWeight * starMask * positiveHighFrequency,
    vec3(0.0)
  );
}

vec3 sampleEnvironment(
  vec3 direction,
  float observerShift,
  float criticalWeight
) {
  vec3 d = safeNormalize(rotateAroundY(direction, uSkyRotation));
  float longitude = atan(d.z, d.x);
  float latitude = asin(clamp(d.y, -1.0, 1.0));
  vec2 skyUv = vec2(
    fract(longitude / TWO_PI + 0.5),
    clamp(0.5 - latitude / PI, 0.00001, 0.99999)
  );
  vec3 rawPanorama = sampleSkyLevelZero(skyUv);
  vec3 filteredPanorama = filterSkyPanorama(skyUv, rawPanorama, criticalWeight);
  vec3 panorama = suppressBakedStarPsf(
    skyUv,
    filteredPanorama,
    clamp(criticalWeight, 0.0, 1.0)
  );
  float shift2 = observerShift * observerShift;
  vec3 calibratedPanorama = max(panorama - vec3(0.002), vec3(0.0));
  return (
    calibratedPanorama * max(uSkyRadianceScale, 0.01)
    + proceduralStars(d, observerShift, criticalWeight)
  ) * shift2 * shift2;
}

vec3 diskNoiseField(
  float advectedPhase,
  float radius,
  float generation
) {
  vec3 seed = vec3(
    generation,
    generation * 0.7548777 + 19.31,
    generation * 1.3247180 - 7.17
  );
  float phaseOffset = TWO_PI * hash31(seed + vec3(3.1, 11.7, 29.3));
  float offsetAngle = TWO_PI * hash31(seed + vec3(41.9, 5.3, 17.1));
  float offsetLength = 29.0 * hash31(seed + vec3(7.7, 31.1, 13.9));
  vec2 fieldOffset = offsetLength * vec2(cos(offsetAngle), sin(offsetAngle));
  float seedZ = 23.0 * hash31(seed + vec3(53.1, 2.9, 37.7));
  float phase = advectedPhase + phaseOffset;
  vec2 materialPosition = radius * vec2(cos(phase), sin(phase)) + fieldOffset;
  vec2 warp = vec2(
    2.0 * valueNoise3(vec3(materialPosition * 0.12, seedZ + 1.7)) - 1.0,
    2.0 * valueNoise3(vec3(materialPosition * 0.12 + vec2(9.3, -6.1), seedZ + 7.9)) - 1.0
  );
  vec2 warpedPosition = materialPosition + 1.8 * warp;
  float cloud = 2.0 * valueNoise3(vec3(
    warpedPosition * 0.28,
    seedZ + 13.1
  )) - 1.0;
  float strandNoise = 2.0 * valueNoise3(vec3(
    warpedPosition * 0.72 + 0.45 * warp,
    seedZ + 29.7
  )) - 1.0;
  float fine = 2.0 * valueNoise3(vec3(
    warpedPosition * 1.75 + 0.72 * warp,
    seedZ + 47.3
  )) - 1.0;
  return vec3(cloud, strandNoise, fine);
}

vec4 accretionDiskSample(
  vec3 hitPosition,
  vec3 diskNormal,
  vec3 traceDirection,
  float lambdaZ,
  float observerShift,
  float travelDelay
) {
  float height = dot(hitPosition, diskNormal);
  vec3 planarPosition = hitPosition - height * diskNormal;
  float radius = length(planarPosition);
  float outerRadius = max(uDiskOuterRadius, ISCO + 0.5);
  float x = ISCO / max(radius, ISCO);
  float fluxShapeRaw = x * x * x * max(1.0 - sqrt(x), 0.0);
  float xPeak = 36.0 / 49.0;
  float fluxPeak = xPeak * xPeak * xPeak * (1.0 - sqrt(xPeak));
  float innerFade = smoothstep(ISCO, ISCO + 0.35, radius);
  float outerFade = 1.0 - smoothstep(outerRadius * 0.82, outerRadius, radius);
  float fluxShape = (fluxShapeRaw / fluxPeak) * innerFade * outerFade;

  vec3 referenceAxis = vec3(0.0, 1.0, 0.0);
  if (abs(dot(referenceAxis, diskNormal)) > 0.92) {
    referenceAxis = vec3(1.0, 0.0, 0.0);
  }
  vec3 diskX = safeNormalize(cross(referenceAxis, diskNormal));
  vec3 diskY = cross(diskNormal, diskX);
  float azimuth = atan(dot(planarPosition, diskY), dot(planarPosition, diskX));
  float omega = inversesqrt(max(radius * radius * radius, 1.0e-8));
  float massSolar = max(uMassSolar, 1.0);
  float timeGeometric = uTime;
  float retardedTime = timeGeometric - travelDelay;
  const float coherenceTime = 32.0;
  float generation = floor(retardedTime / coherenceTime);
  float generationBlend = fract(retardedTime / coherenceTime);
  float ageA = retardedTime - generation * coherenceTime;
  float ageB = ageA - coherenceTime;
  vec3 fieldA = diskNoiseField(
    azimuth - omega * ageA,
    radius,
    generation
  );
  vec3 fieldB = diskNoiseField(
    azimuth - omega * ageB,
    radius,
    generation + 1.0
  );
  float blendWeight = smoothstep(0.0, 1.0, generationBlend);
  float varianceCompensation = min(inversesqrt(
    (1.0 - blendWeight) * (1.0 - blendWeight)
    + blendWeight * blendWeight
  ), 1.25);
  vec3 turbulence = mix(fieldA, fieldB, blendWeight) * varianceCompensation;
  float cloud = turbulence.x;
  float strandNoise = turbulence.y;
  float fine = turbulence.z;
  float filamentRidge = smoothstep(
    0.05,
    0.82,
    0.62 * strandNoise + 0.38 * cloud
  );
  float densityContrast = clamp(
    0.62 * cloud + 0.30 * strandNoise + 0.12 * fine
    + 0.16 * (filamentRidge - 0.42),
    -0.85,
    0.85
  );
  float localHeating = exp(clamp(
    0.45 * cloud + 0.30 * strandNoise + 0.14 * fine
    + 0.12 * (filamentRidge - 0.42),
    -0.50,
    0.58
  ));
  float tauMean = 1.45
                * pow(max(radius / 8.17, 0.1), -0.62)
                * innerFade * outerFade;
  float tauFace = tauMean * exp(0.75 * densityContrast);
  float covering = mix(
    0.82,
    1.0,
    smoothstep(-0.75, 0.75, densityContrast)
  );

  float accretion = max(uAccretion, 1.0e-6);
  float peakTemperature = 1.43e5 * pow(accretion * 1.0e8 / massSolar, 0.25);
  float emittedTemperature = max(
    600.0,
    peakTemperature * pow(max(fluxShape * localHeating, 1.0e-8), 0.25)
  );
  float spectralHardening = 1.0
    + 0.15 * smoothstep(8000.0, 30000.0, emittedTemperature)
           * smoothstep(0.25, 2.0, tauFace);
  float emitterUt = inversesqrt(max(1.0 - 3.0 / radius, 1.0e-5));
  float orbitalDenominator = max(1.0 - omega * lambdaZ, 0.015);
  float transferDenominator = emitterUt * orbitalDenominator;
  float g = clamp(observerShift / transferDenominator, 0.04, 8.0);
  float observedTemperature = emittedTemperature * spectralHardening * g;
  float g2 = g * g;
  float localLapse = sqrt(max(1.0 - 2.0 / radius, 1.0e-5));
  float emitterEnergyOverStatic = localLapse * emitterUt * orbitalDenominator;
  float muEmit = clamp(
    abs(dot(traceDirection, diskNormal)) / max(emitterEnergyOverStatic, 1.0e-4),
    0.03,
    1.0
  );
  float tauLineOfSight = min(tauFace / muEmit, 20.0);
  float opacity = clamp(covering * (1.0 - exp(-tauLineOfSight)), 0.0, 1.0);
  float thickLimb = (1.0 + 2.06 * muEmit) / 3.06;
  float limbDarkening = mix(1.0, thickLimb, smoothstep(0.25, 1.5, tauFace));
  float radiance = max(fluxShape, 0.0)
                 * localHeating * accretion * g2 * g2 * limbDarkening;
  vec3 source = planckChromaticity(observedTemperature) * radiance * 6500.0;
  return vec4(source * opacity, opacity);
}

vec3 aberratePastRay(vec3 comovingDirection) {
  float beta = clamp(uObserverBeta, 0.0, 0.95);
  float velocityLength = length(uObserverVelocity);
  if (beta <= 1.0e-6 || velocityLength <= 1.0e-6) {
    return safeNormalize(comovingDirection);
  }
  vec3 velocityDirection = uObserverVelocity / velocityLength;
  float mu = dot(comovingDirection, velocityDirection);
  float gamma = inversesqrt(max(1.0 - beta * beta, 1.0e-6));
  vec3 perpendicular = comovingDirection - mu * velocityDirection;
  float denominator = max(gamma * (1.0 - beta * mu), 1.0e-6);
  return safeNormalize((perpendicular + gamma * (mu - beta) * velocityDirection) / denominator);
}

float observerFrequencyShift(vec3 comovingDirection, float staticLapse) {
  float beta = clamp(uObserverBeta, 0.0, 0.95);
  float velocityLength = length(uObserverVelocity);
  if (beta <= 1.0e-6 || velocityLength <= 1.0e-6) {
    return 1.0 / staticLapse;
  }
  vec3 velocityDirection = uObserverVelocity / velocityLength;
  float mu = dot(comovingDirection, velocityDirection);
  float gamma = inversesqrt(max(1.0 - beta * beta, 1.0e-6));
  return 1.0 / max(staticLapse * gamma * (1.0 - beta * mu), 1.0e-6);
}

vec3 traceSchwarzschild(vec3 comovingRay, float criticalWeight) {
  float rawCameraLength = length(uCameraPos);
  float cameraRadius = max(uCameraRadius > 2.001 ? uCameraRadius : rawCameraLength, 2.001);
  vec3 cameraPosition = safeNormalize(uCameraPos) * cameraRadius;
  vec3 radialBasis = cameraPosition / cameraRadius;
  float lapseSquared = max(1.0 - 2.0 / cameraRadius, 1.0e-6);
  float staticLapse = sqrt(lapseSquared);
  float observerShift = observerFrequencyShift(comovingRay, staticLapse);
  vec3 ray = aberratePastRay(comovingRay);

  float radialDirection = dot(ray, radialBasis);
  vec3 tangentVector = ray - radialDirection * radialBasis;
  float tangentLength = length(tangentVector);
  if (tangentLength < 1.0e-6) {
    if (radialDirection < 0.0) {
      return vec3(0.0);
    }
    return sampleEnvironment(radialBasis, observerShift, criticalWeight);
  }

  vec3 tangentBasis = tangentVector / tangentLength;
  float impact = cameraRadius * tangentLength / staticLapse;
  float environmentFilterWeight = max(
    criticalWeight,
    0.68 * (1.0 - smoothstep(0.30, 2.40, abs(impact - PHOTON_IMPACT)))
  );
  float inverseRadius = 1.0 / cameraRadius;
  float inverseRadiusDerivative = -inverseRadius * staticLapse * radialDirection / tangentLength;
  float planeAngle = 0.0;
  vec3 directionOnPlane = radialBasis;

  vec3 diskNormal = vec3(0.0, 1.0, 0.0);
  float outerRadius = max(uDiskOuterRadius, ISCO + 0.5);
  float lambdaZ = -dot(cross(cameraPosition, ray), diskNormal) / staticLapse;
  float previousDiskSide = dot(directionOnPlane, diskNormal);
  float travelDelay = 0.0;
  vec3 accumulatedRadiance = vec3(0.0);
  float throughput = 1.0;
  int activeSteps = clamp(int(uSteps + 0.5), 1, MAX_STEPS);
  if (abs(impact - PHOTON_IMPACT) < 0.45) {
    activeSteps = MAX_STEPS;
  }
  const float stepAngle = 0.030;
  const float horizonInverseRadius = 0.5 / (1.0 + 2.0e-4);

  for (int step = 0; step < MAX_STEPS; ++step) {
    if (step >= activeSteps) {
      break;
    }
    float previousU = inverseRadius;
    float previousV = inverseRadiusDerivative;
    float previousAngle = planeAngle;
    vec3 previousDirection = directionOnPlane;

    float halfV = previousV + 0.5 * stepAngle * schwarzschildForce(previousU);
    float nextU = previousU + stepAngle * halfV;
    float nextV = halfV + 0.5 * stepAngle * schwarzschildForce(nextU);
    float nextAngle = previousAngle + stepAngle;
    vec3 nextDirection = cos(nextAngle) * radialBasis + sin(nextAngle) * tangentBasis;
    float nextDiskSide = dot(nextDirection, diskNormal);

    float boundaryFraction = 2.0;
    bool crossedHorizon = false;
    bool escaped = false;
    if (nextU >= horizonInverseRadius && nextU != previousU) {
      boundaryFraction = clamp((horizonInverseRadius - previousU) / (nextU - previousU), 0.0, 1.0);
      crossedHorizon = true;
    }
    if (nextU <= 0.0 && nextU != previousU) {
      float escapeFraction = clamp(previousU / (previousU - nextU), 0.0, 1.0);
      if (escapeFraction < boundaryFraction) {
        boundaryFraction = escapeFraction;
        crossedHorizon = false;
        escaped = true;
      }
    }

    float midpointU = max(0.5 * (previousU + max(nextU, 1.0e-6)), 1.0e-5);
    float midpointLapse = max(1.0 - 2.0 * midpointU, 1.0e-5);
    float delayIncrement = stepAngle / max(impact * midpointU * midpointU * midpointLapse, 1.0e-7);

    if (
      previousDiskSide * nextDiskSide <= 0.0
      && abs(previousDiskSide - nextDiskSide) > 1.0e-7
    ) {
      float diskFraction = clamp(
        previousDiskSide / (previousDiskSide - nextDiskSide),
        0.0,
        1.0
      );
      if (diskFraction > 1.0e-5 && diskFraction < boundaryFraction) {
        float hitU = mix(previousU, nextU, diskFraction);
        if (hitU > 0.0) {
          float hitRadius = 1.0 / hitU;
          float hitAngle = mix(previousAngle, nextAngle, diskFraction);
          vec3 hitDirection = safeNormalize(mix(previousDirection, nextDirection, diskFraction));
          vec3 hitPosition = hitDirection * hitRadius;
          float hitHeight = dot(hitPosition, diskNormal);
          float hitPlanarRadius = length(hitPosition - hitHeight * diskNormal);
          if (hitPlanarRadius >= ISCO && hitPlanarRadius <= outerRadius) {
            float hitDerivative = mix(previousV, nextV, diskFraction);
            vec3 tangentAtHit = -sin(hitAngle) * radialBasis + cos(hitAngle) * tangentBasis;
            float hitLapse = sqrt(max(1.0 - 2.0 * hitU, 1.0e-5));
            float tangentialDirection = clamp(impact * hitU * hitLapse, 0.0, 1.0);
            float radialMagnitude = sqrt(max(1.0 - tangentialDirection * tangentialDirection, 0.0));
            float radialSign = hitDerivative > 0.0 ? -1.0 : 1.0;
            vec3 traceDirectionAtHit = safeNormalize(
              radialSign * radialMagnitude * hitDirection
              + tangentialDirection * tangentAtHit
            );
            vec4 diskSample = accretionDiskSample(
              hitPosition,
              diskNormal,
              traceDirectionAtHit,
              lambdaZ,
              observerShift,
              travelDelay + diskFraction * delayIncrement
            );
            accumulatedRadiance += throughput * diskSample.rgb;
            throughput *= 1.0 - diskSample.a;
            if (throughput < 0.02) {
              return accumulatedRadiance;
            }
          }
        }
      }
    }

    if (crossedHorizon) {
      return accumulatedRadiance;
    }
    if (escaped) {
      float escapeAngle = mix(previousAngle, nextAngle, boundaryFraction);
      vec3 escapeDirection = cos(escapeAngle) * radialBasis + sin(escapeAngle) * tangentBasis;
      return accumulatedRadiance
           + throughput * sampleEnvironment(
               escapeDirection,
               observerShift,
               environmentFilterWeight
             );
    }

    inverseRadius = nextU;
    inverseRadiusDerivative = nextV;
    planeAngle = nextAngle;
    directionOnPlane = nextDirection;
    previousDiskSide = nextDiskSide;
    travelDelay += delayIncrement;
  }

  if (impact < PHOTON_IMPACT && radialDirection < 0.0 && inverseRadiusDerivative >= 0.0) {
    return accumulatedRadiance;
  }
  return accumulatedRadiance
       + throughput * sampleEnvironment(
           directionOnPlane,
           observerShift,
           environmentFilterWeight
         );
}

vec3 cameraRayForScreen(vec2 screen, float tanHalfFov) {
  return safeNormalize(
    uForward + tanHalfFov * screen.x * uRight + tanHalfFov * screen.y * uUp
  );
}

float initialImpact(vec3 comovingRay) {
  float cameraRadius = max(uCameraRadius, 2.001);
  vec3 radialBasis = safeNormalize(uCameraPos);
  float staticLapse = sqrt(max(1.0 - 2.0 / cameraRadius, 1.0e-6));
  vec3 staticRay = aberratePastRay(comovingRay);
  float radialComponent = dot(staticRay, radialBasis);
  return cameraRadius * sqrt(max(1.0 - radialComponent * radialComponent, 0.0))
       / staticLapse;
}

void main() {
  vec2 resolution = max(uResolution, vec2(1.0));
  float aspect = resolution.x / resolution.y;
  vec2 screen = vec2((vUv.x * 2.0 - 1.0) * aspect, vUv.y * 2.0 - 1.0);
  float tanHalfFov = tan(0.5 * clamp(uFov, 0.02, 2.8));
  vec3 centreRay = cameraRayForScreen(screen, tanHalfFov);
  float centreImpact = initialImpact(centreRay);
  float impactFootprint = max(fwidth(centreImpact), 1.0e-5);
  float criticalDistancePixels = abs(centreImpact - PHOTON_IMPACT)
                               / impactFootprint;
  float criticalWeight = 1.0 - smoothstep(
    3.0,
    8.0,
    criticalDistancePixels
  );
  vec3 color;
  if (abs(centreImpact - PHOTON_IMPACT) < 8.0 * impactFootprint) {
    float pixelSpan = 2.0 / resolution.y;
    color = 0.25 * (
      traceSchwarzschild(
        cameraRayForScreen(screen + vec2(-0.25, -0.25) * pixelSpan, tanHalfFov),
        criticalWeight
      )
      + traceSchwarzschild(
          cameraRayForScreen(screen + vec2( 0.25, -0.25) * pixelSpan, tanHalfFov),
          criticalWeight
        )
      + traceSchwarzschild(
          cameraRayForScreen(screen + vec2(-0.25,  0.25) * pixelSpan, tanHalfFov),
          criticalWeight
        )
      + traceSchwarzschild(
          cameraRayForScreen(screen + vec2( 0.25,  0.25) * pixelSpan, tanHalfFov),
          criticalWeight
        )
    );
  } else {
    color = traceSchwarzschild(centreRay, 0.0);
  }
  gl_FragColor = vec4(max(color, vec3(0.0)), 1.0);
}
`;

export const postFragmentGLSL = /* glsl */ `
precision highp float;
precision highp int;

uniform vec2 uResolution;
uniform float uTime;
uniform float uMassSolar;
uniform float uAccretion;
uniform float uExposure;
uniform float uMode;
uniform float uSteps;
uniform vec3 uCameraPos;
uniform float uCameraRadius;
uniform vec3 uForward;
uniform float uFov;
uniform vec3 uRight;
uniform float uSkyRotation;
uniform vec3 uUp;
uniform float uDiskOuterRadius;
uniform float uRenderScale;
uniform float uBloom;
uniform float uMotion;
uniform float uFrame;
uniform vec3 uObserverVelocity;
uniform float uObserverBeta;
uniform sampler2D tScene;

varying vec2 vUv;

vec3 sampleScene(vec2 sceneUv) {
  return max(textureLod(tScene, clamp(sceneUv, vec2(0.0), vec2(1.0)), 0.0).rgb, vec3(0.0));
}

float luminance(vec3 color) {
  return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 brightPart(vec3 color, float threshold) {
  float light = luminance(color);
  float knee = max(0.25 * threshold, 1.0e-5);
  float soft = clamp(light - threshold + knee, 0.0, 2.0 * knee);
  float softContribution = soft * soft / (4.0 * knee);
  float contribution = max(light - threshold, softContribution);
  return color * contribution / max(light, 1.0e-5);
}

vec3 acesFitted(vec3 color) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  vec3 positive = max(color, 0.0);
  vec3 mapped = clamp(
    (positive * (a * positive + b)) / (positive * (c * positive + d) + e),
    0.0,
    1.0
  );
  float sourceLight = max(luminance(positive), 1.0e-5);
  float mappedLight = luminance(mapped);
  vec3 huePreserved = positive * (mappedLight / sourceLight);
  float hueMaximum = max(max(huePreserved.r, huePreserved.g), huePreserved.b);
  huePreserved /= max(hueMaximum, 1.0);
  float chromaWeight = 0.34 * smoothstep(0.10, 0.85, mappedLight);
  return clamp(mix(mapped, huePreserved, chromaWeight), 0.0, 1.0);
}

vec3 encodeSrgbTransfer(vec3 color) {
  vec3 positive = max(color, 0.0);
  vec3 low = positive * 12.92;
  vec3 high = 1.055 * pow(positive, vec3(1.0 / 2.4)) - 0.055;
  return mix(high, low, lessThanEqual(positive, vec3(0.0031308)));
}

float hashPixel(vec2 pixel, float frame) {
  return fract(sin(dot(vec3(pixel, frame + 1.0), vec3(12.9898, 78.233, 37.719))) * 43758.5453);
}

void main() {
  vec2 resolution = max(uResolution, vec2(1.0));
  vec2 texel = 1.0 / resolution;
  float mode = clamp(uMode, 0.0, 1.0);
  float exposure = max(uExposure, 0.0);
  vec3 color = sampleScene(vUv) * exposure;
  float bloomStrength = max(uBloom, 0.0);
  if (bloomStrength > 1.0e-5) {
    float cssScale = max(uRenderScale, 0.5);
    vec2 bloomRadius = texel * cssScale * mix(2.0, 3.0, mode);
    vec3 bloomSamples = brightPart(sampleScene(vUv + vec2( bloomRadius.x, 0.0)) * exposure, 1.0)
                      + brightPart(sampleScene(vUv + vec2(-bloomRadius.x, 0.0)) * exposure, 1.0)
                      + brightPart(sampleScene(vUv + vec2(0.0,  bloomRadius.y)) * exposure, 1.0)
                      + brightPart(sampleScene(vUv + vec2(0.0, -bloomRadius.y)) * exposure, 1.0)
                      + 0.5 * (
                          brightPart(sampleScene(vUv + bloomRadius) * exposure, 1.0)
                          + brightPart(sampleScene(vUv - bloomRadius) * exposure, 1.0)
                          + brightPart(sampleScene(vUv + vec2(bloomRadius.x, -bloomRadius.y)) * exposure, 1.0)
                          + brightPart(sampleScene(vUv + vec2(-bloomRadius.x, bloomRadius.y)) * exposure, 1.0)
                        );
    color += bloomSamples * (bloomStrength / 24.0);
  }

  float light = luminance(color);
  vec3 hubbleColour = mix(vec3(light), color, 1.06) * vec3(1.018, 1.0, 0.985);
  color = max(mix(color, hubbleColour, mode), 0.0);
  color = acesFitted(color);
  color = encodeSrgbTransfer(color);
  float dither = (hashPixel(gl_FragCoord.xy, uFrame) - 0.5) / 255.0;
  color = clamp(color + dither, 0.0, 1.0);
  gl_FragColor = vec4(color, 1.0);
}
`;
