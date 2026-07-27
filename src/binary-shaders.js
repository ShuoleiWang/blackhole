/*
 * Experimental binary-black-hole trace shaders.
 *
 * This scene deliberately does not reuse the exact Schwarzschild ray-plane
 * integrator.  Its frame-frozen, two-centre potential is a weak-field/fast-
 * light preview; it is not a solved numerical-relativity spacetime.  See
 * docs/binary-model.md for the model boundary and the future transfer-map
 * contract.
 *
 * Shared offsets 0..35 keep the renderer ABI used by shaders.js.  The binary
 * scene owns two final vec4 values:
 *
 *   36 separation in initial-total-mass units
 *   37 orbital phase in radians
 *   38 phenomenological common-remnant blend [0, 1]
 *   39 reserved
 *   40 body A mass fraction
 *   41 body B mass fraction
 *   42 remnant mass fraction
 *   43 reserved
 */

export const binaryTraceFragmentWGSL = /* wgsl */ `
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const MAX_STEPS: i32 = 512;

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
  sceneBinaryState: vec4<f32>,
  sceneBinaryMasses: vec4<f32>,
};

struct FragmentInput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tSky: texture_2d<f32>;
@group(0) @binding(2) var skySampler: sampler;

fn safeNormalize(value: vec3<f32>) -> vec3<f32> {
  return value * inverseSqrt(max(dot(value, value), 1.0e-18));
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

fn hash31(value: vec3<f32>) -> f32 {
  var p = fract(value * 0.1031);
  p = p + vec3<f32>(dot(p, p.yzx + vec3<f32>(33.33)));
  return fract((p.x + p.y) * p.z);
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

fn analyticStars(direction: vec3<f32>) -> vec3<f32> {
  // A sparse direction-locked layer keeps lensing legible in dark portions of
  // the panorama.  It is sampled only after the ray has escaped, so the points
  // follow the same lens map as the photographic sky.
  let mapped = cubeStarCoordinates(direction);
  let grid = 256.0;
  let coordinates = mapped.xy * grid;
  let cell = floor(coordinates);
  let local = fract(coordinates);
  let seed = vec3<f32>(cell, mapped.z * 41.0);
  let rank = hash31(seed + vec3<f32>(17.1, 31.7, 9.3));
  if (rank < 0.994) {
    return vec3<f32>(0.0);
  }
  let centre = vec2<f32>(
    0.24 + 0.52 * hash31(seed + vec3<f32>(7.3, 19.1, 3.7)),
    0.24 + 0.52 * hash31(seed + vec3<f32>(29.9, 5.1, 13.7))
  );
  let radius = length(local - centre);
  let core = 1.0 - smoothstep(0.075, 0.18, radius);
  let temperature = hash31(seed + vec3<f32>(43.1, 11.9, 23.3));
  let colour = mix(
    vec3<f32>(1.0, 0.66, 0.38),
    vec3<f32>(0.64, 0.78, 1.0),
    temperature
  );
  let radiance = mix(0.20, 3.2, pow((rank - 0.994) / 0.006, 0.55));
  return colour * radiance * core;
}

fn sampleEnvironment(direction: vec3<f32>, nearLensWeight: f32) -> vec3<f32> {
  let d = safeNormalize(rotateAroundY(direction, params.cameraRightSkyRotation.w));
  let longitude = atan2(d.z, d.x);
  let latitude = asin(clamp(d.y, -1.0, 1.0));
  let uv = vec2<f32>(
    fract(longitude / TWO_PI + 0.5),
    clamp(0.5 - latitude / PI, 0.00001, 0.99999)
  );

  var panorama = max(
    textureSampleLevel(tSky, skySampler, uv, 0.0).rgb - vec3<f32>(0.0015),
    vec3<f32>(0.0)
  );
  if (nearLensWeight > 0.02) {
    let texel = vec2<f32>(1.0) / vec2<f32>(textureDimensions(tSky));
    let radius = mix(1.0, 4.0, clamp(nearLensWeight, 0.0, 1.0));
    let filtered = 0.25 * (
      textureSampleLevel(tSky, skySampler, uv + vec2<f32>(radius * texel.x, 0.0), 0.0).rgb
      + textureSampleLevel(tSky, skySampler, uv - vec2<f32>(radius * texel.x, 0.0), 0.0).rgb
      + textureSampleLevel(tSky, skySampler, uv + vec2<f32>(0.0, radius * texel.y), 0.0).rgb
      + textureSampleLevel(tSky, skySampler, uv - vec2<f32>(0.0, radius * texel.y), 0.0).rgb
    );
    panorama = mix(panorama, max(filtered - vec3<f32>(0.0015), vec3<f32>(0.0)), 0.34);
  }
  // Vacuum lensing preserves source surface brightness; there is no invented
  // caustic glow or luminous merger material in this scene.
  return panorama * max(params.displayOutput.w, 0.01) + analyticStars(d);
}

fn potentialGradient(
  position: vec3<f32>,
  centreA: vec3<f32>,
  centreB: vec3<f32>,
  remnantBlend: f32,
  massA: f32,
  massB: f32,
  remnantMass: f32
) -> vec3<f32> {
  let sideWeight = 1.0 - remnantBlend;
  let remnantWeight = remnantBlend;
  let offsetA = position - centreA;
  let offsetB = position - centreB;
  let offsetR = position;
  let radiusA = max(length(offsetA), 0.05);
  let radiusB = max(length(offsetB), 0.05);
  let radiusR = max(length(offsetR), 0.05);
  let sideGradient = massA * offsetA / (radiusA * radiusA * radiusA)
                   + massB * offsetB / (radiusB * radiusB * radiusB);
  let remnantGradient = remnantMass
                      * offsetR / (radiusR * radiusR * radiusR);
  return sideWeight * sideGradient + remnantWeight * remnantGradient;
}

fn bendingAcceleration(
  position: vec3<f32>,
  direction: vec3<f32>,
  centreA: vec3<f32>,
  centreB: vec3<f32>,
  remnantBlend: f32,
  massA: f32,
  massB: f32,
  remnantMass: f32
) -> vec3<f32> {
  let gradient = potentialGradient(
    position,
    centreA,
    centreB,
    remnantBlend,
    massA,
    massB,
    remnantMass
  );
  let transverse = gradient - direction * dot(direction, gradient);
  let acceleration = -2.0 * transverse;
  let magnitude = length(acceleration);
  return acceleration * min(1.0, 6.0 / max(magnitude, 1.0e-6));
}

fn captureBoundaryDistance(
  position: vec3<f32>,
  centreA: vec3<f32>,
  centreB: vec3<f32>,
  remnantBlend: f32,
  massA: f32,
  massB: f32,
  remnantMass: f32
) -> f32 {
  let binaryDistance = min(
    length(position - centreA) - 2.0 * massA,
    length(position - centreB) - 2.0 * massB
  );
  let remnantDistance = length(position) - 2.0 * remnantMass;
  // This SDF morph is only a capture boundary for the preview.  It is not a
  // computed apparent/event horizon and must not be interpreted as one.
  let topologyBlend = smoothstep(0.12, 0.88, remnantBlend);
  return mix(binaryDistance, remnantDistance, topologyBlend);
}

fn traceBinary(initialDirection: vec3<f32>) -> vec3<f32> {
  let separation = max(params.sceneBinaryState.x, 0.0);
  let orbitalPhase = params.sceneBinaryState.y;
  let remnantBlend = clamp(params.sceneBinaryState.z, 0.0, 1.0);
  let massA = max(params.sceneBinaryMasses.x, 1.0e-6);
  let massB = max(params.sceneBinaryMasses.y, 1.0e-6);
  let remnantMass = max(params.sceneBinaryMasses.z, 1.0e-6);
  let totalBinaryMass = max(massA + massB, 1.0e-6);
  let axis = vec3<f32>(cos(orbitalPhase), 0.0, sin(orbitalPhase));
  let centreA = -separation * (massB / totalBinaryMass) * axis;
  let centreB =  separation * (massA / totalBinaryMass) * axis;

  var position = params.cameraPosRadius.xyz;
  var direction = safeNormalize(initialDirection);
  let cameraRadius = max(params.cameraPosRadius.w, length(position));
  let requestedSteps = i32(clamp(params.renderControls.w, 1.0, f32(MAX_STEPS)));
  var previousRadius = length(position);
  var approached = false;
  var minimumCaptureDistance = 1.0e6;

  for (var index = 0; index < MAX_STEPS; index = index + 1) {
    if (index >= requestedSteps) {
      break;
    }

    let radius = length(position);
    let captureDistance = captureBoundaryDistance(
      position,
      centreA,
      centreB,
      remnantBlend,
      massA,
      massB,
      remnantMass
    );
    minimumCaptureDistance = min(minimumCaptureDistance, captureDistance);
    if (captureDistance <= 0.0) {
      return vec3<f32>(0.0);
    }

    approached = approached || radius < previousRadius;
    if (
      approached
      && radius > 1.035 * cameraRadius
      && dot(position, direction) > 0.0
    ) {
      let nearLensWeight = 1.0 - smoothstep(
        0.15,
        3.0,
        minimumCaptureDistance
      );
      return sampleEnvironment(direction, nearLensWeight);
    }
    previousRadius = radius;

    var nearestMassRadius = 1.0e3;
    if (remnantBlend < 0.999) {
      nearestMassRadius = min(
        length(position - centreA) / massA,
        length(position - centreB) / massB
      );
    }
    if (remnantBlend > 0.001) {
      nearestMassRadius = min(
        nearestMassRadius,
        length(position) / remnantMass
      );
    }
    var stepLength = clamp(0.045 + 0.020 * nearestMassRadius, 0.045, 0.58);
    stepLength = min(
      stepLength,
      clamp(0.34 * captureDistance, 0.018, 0.58)
    );

    let acceleration0 = bendingAcceleration(
      position,
      direction,
      centreA,
      centreB,
      remnantBlend,
      massA,
      massB,
      remnantMass
    );
    let midpointDirection = safeNormalize(
      direction + 0.5 * stepLength * acceleration0
    );
    let midpoint = position + 0.5 * stepLength * midpointDirection;
    let accelerationMid = bendingAcceleration(
      midpoint,
      midpointDirection,
      centreA,
      centreB,
      remnantBlend,
      massA,
      massB,
      remnantMass
    );
    direction = safeNormalize(direction + stepLength * accelerationMid);
    position = position + stepLength * midpointDirection;
  }

  // Reaching the real-time step budget is not an escape condition.  Returning
  // sky radiance here would turn unresolved near-critical paths into fake
  // bright arcs, so unresolved rays use the same conservative black sentinel
  // as captured rays.
  return vec3<f32>(0.0);
}

fn cameraRay(screen: vec2<f32>, tanHalfFov: f32) -> vec3<f32> {
  return safeNormalize(
    params.cameraForwardFov.xyz
    + tanHalfFov * screen.x * params.cameraRightSkyRotation.xyz
    + tanHalfFov * screen.y * params.cameraUpDiskOuter.xyz
  );
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
  let colour = traceBinary(cameraRay(screen, tanHalfFov));
  return vec4<f32>(max(colour, vec3<f32>(0.0)), 1.0);
}
`;

export const binaryTraceFragmentGLSL = /* glsl */ `
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
uniform vec4 uSceneBinaryState;
uniform vec4 uSceneBinaryMasses;
uniform sampler2D tSky;

varying vec2 vUv;

const float PI = 3.14159265358979323846;
const float TWO_PI = 6.28318530717958647692;
const int MAX_STEPS = 512;

vec3 safeNormalize(vec3 value) {
  return value * inversesqrt(max(dot(value, value), 1.0e-18));
}

vec3 rotateAroundY(vec3 direction, float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return vec3(
    c * direction.x + s * direction.z,
    direction.y,
   -s * direction.x + c * direction.z
  );
}

float hash31(vec3 value) {
  vec3 p = fract(value * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
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

vec3 analyticStars(vec3 direction) {
  vec3 mapped = cubeStarCoordinates(direction);
  float grid = 256.0;
  vec2 coordinates = mapped.xy * grid;
  vec2 cell = floor(coordinates);
  vec2 local = fract(coordinates);
  vec3 seed = vec3(cell, mapped.z * 41.0);
  float rank = hash31(seed + vec3(17.1, 31.7, 9.3));
  if (rank < 0.994) {
    return vec3(0.0);
  }
  vec2 centre = vec2(
    0.24 + 0.52 * hash31(seed + vec3(7.3, 19.1, 3.7)),
    0.24 + 0.52 * hash31(seed + vec3(29.9, 5.1, 13.7))
  );
  float radius = length(local - centre);
  float core = 1.0 - smoothstep(0.075, 0.18, radius);
  float temperature = hash31(seed + vec3(43.1, 11.9, 23.3));
  vec3 colour = mix(
    vec3(1.0, 0.66, 0.38),
    vec3(0.64, 0.78, 1.0),
    temperature
  );
  float radiance = mix(0.20, 3.2, pow((rank - 0.994) / 0.006, 0.55));
  return colour * radiance * core;
}

vec3 sampleEnvironment(vec3 direction, float nearLensWeight) {
  vec3 d = safeNormalize(rotateAroundY(direction, uSkyRotation));
  float longitude = atan(d.z, d.x);
  float latitude = asin(clamp(d.y, -1.0, 1.0));
  vec2 skyUv = vec2(
    fract(longitude / TWO_PI + 0.5),
    clamp(0.5 - latitude / PI, 0.00001, 0.99999)
  );

  vec3 panorama = max(
    textureLod(tSky, skyUv, 0.0).rgb - vec3(0.0015),
    vec3(0.0)
  );
  if (nearLensWeight > 0.02) {
    vec2 texel = 1.0 / vec2(textureSize(tSky, 0));
    float radius = mix(1.0, 4.0, clamp(nearLensWeight, 0.0, 1.0));
    vec3 filtered = 0.25 * (
      textureLod(tSky, skyUv + vec2(radius * texel.x, 0.0), 0.0).rgb
      + textureLod(tSky, skyUv - vec2(radius * texel.x, 0.0), 0.0).rgb
      + textureLod(tSky, skyUv + vec2(0.0, radius * texel.y), 0.0).rgb
      + textureLod(tSky, skyUv - vec2(0.0, radius * texel.y), 0.0).rgb
    );
    panorama = mix(panorama, max(filtered - vec3(0.0015), vec3(0.0)), 0.34);
  }
  return panorama * max(uSkyRadianceScale, 0.01) + analyticStars(d);
}

vec3 potentialGradient(
  vec3 position,
  vec3 centreA,
  vec3 centreB,
  float remnantBlend,
  float massA,
  float massB,
  float remnantMass
) {
  float sideWeight = 1.0 - remnantBlend;
  float remnantWeight = remnantBlend;
  vec3 offsetA = position - centreA;
  vec3 offsetB = position - centreB;
  vec3 offsetR = position;
  float radiusA = max(length(offsetA), 0.05);
  float radiusB = max(length(offsetB), 0.05);
  float radiusR = max(length(offsetR), 0.05);
  vec3 sideGradient = massA * offsetA / (radiusA * radiusA * radiusA)
                    + massB * offsetB / (radiusB * radiusB * radiusB);
  vec3 remnantGradient = remnantMass
                       * offsetR / (radiusR * radiusR * radiusR);
  return sideWeight * sideGradient + remnantWeight * remnantGradient;
}

vec3 bendingAcceleration(
  vec3 position,
  vec3 direction,
  vec3 centreA,
  vec3 centreB,
  float remnantBlend,
  float massA,
  float massB,
  float remnantMass
) {
  vec3 gradient = potentialGradient(
    position,
    centreA,
    centreB,
    remnantBlend,
    massA,
    massB,
    remnantMass
  );
  vec3 transverse = gradient - direction * dot(direction, gradient);
  vec3 acceleration = -2.0 * transverse;
  float magnitude = length(acceleration);
  return acceleration * min(1.0, 6.0 / max(magnitude, 1.0e-6));
}

float captureBoundaryDistance(
  vec3 position,
  vec3 centreA,
  vec3 centreB,
  float remnantBlend,
  float massA,
  float massB,
  float remnantMass
) {
  float binaryDistance = min(
    length(position - centreA) - 2.0 * massA,
    length(position - centreB) - 2.0 * massB
  );
  float remnantDistance = length(position) - 2.0 * remnantMass;
  float topologyBlend = smoothstep(0.12, 0.88, remnantBlend);
  return mix(binaryDistance, remnantDistance, topologyBlend);
}

vec3 traceBinary(vec3 initialDirection) {
  float separation = max(uSceneBinaryState.x, 0.0);
  float orbitalPhase = uSceneBinaryState.y;
  float remnantBlend = clamp(uSceneBinaryState.z, 0.0, 1.0);
  float massA = max(uSceneBinaryMasses.x, 1.0e-6);
  float massB = max(uSceneBinaryMasses.y, 1.0e-6);
  float remnantMass = max(uSceneBinaryMasses.z, 1.0e-6);
  float totalBinaryMass = max(massA + massB, 1.0e-6);
  vec3 axis = vec3(cos(orbitalPhase), 0.0, sin(orbitalPhase));
  vec3 centreA = -separation * (massB / totalBinaryMass) * axis;
  vec3 centreB =  separation * (massA / totalBinaryMass) * axis;

  vec3 position = uCameraPos;
  vec3 direction = safeNormalize(initialDirection);
  float cameraRadius = max(uCameraRadius, length(position));
  int requestedSteps = int(clamp(uSteps, 1.0, float(MAX_STEPS)));
  float previousRadius = length(position);
  bool approached = false;
  float minimumCaptureDistance = 1.0e6;

  for (int index = 0; index < MAX_STEPS; ++index) {
    if (index >= requestedSteps) {
      break;
    }

    float radius = length(position);
    float captureDistance = captureBoundaryDistance(
      position,
      centreA,
      centreB,
      remnantBlend,
      massA,
      massB,
      remnantMass
    );
    minimumCaptureDistance = min(minimumCaptureDistance, captureDistance);
    if (captureDistance <= 0.0) {
      return vec3(0.0);
    }

    approached = approached || radius < previousRadius;
    if (
      approached
      && radius > 1.035 * cameraRadius
      && dot(position, direction) > 0.0
    ) {
      float nearLensWeight = 1.0 - smoothstep(
        0.15,
        3.0,
        minimumCaptureDistance
      );
      return sampleEnvironment(direction, nearLensWeight);
    }
    previousRadius = radius;

    float nearestMassRadius = 1.0e3;
    if (remnantBlend < 0.999) {
      nearestMassRadius = min(
        length(position - centreA) / massA,
        length(position - centreB) / massB
      );
    }
    if (remnantBlend > 0.001) {
      nearestMassRadius = min(
        nearestMassRadius,
        length(position) / remnantMass
      );
    }
    float stepLength = clamp(
      0.045 + 0.020 * nearestMassRadius,
      0.045,
      0.58
    );
    stepLength = min(
      stepLength,
      clamp(0.34 * captureDistance, 0.018, 0.58)
    );

    vec3 acceleration0 = bendingAcceleration(
      position,
      direction,
      centreA,
      centreB,
      remnantBlend,
      massA,
      massB,
      remnantMass
    );
    vec3 midpointDirection = safeNormalize(
      direction + 0.5 * stepLength * acceleration0
    );
    vec3 midpoint = position + 0.5 * stepLength * midpointDirection;
    vec3 accelerationMid = bendingAcceleration(
      midpoint,
      midpointDirection,
      centreA,
      centreB,
      remnantBlend,
      massA,
      massB,
      remnantMass
    );
    direction = safeNormalize(direction + stepLength * accelerationMid);
    position += stepLength * midpointDirection;
  }

  return vec3(0.0);
}

vec3 cameraRay(vec2 screen, float tanHalfFov) {
  return safeNormalize(
    uForward
    + tanHalfFov * screen.x * uRight
    + tanHalfFov * screen.y * uUp
  );
}

void main() {
  vec2 resolution = max(uResolution, vec2(1.0));
  float aspect = resolution.x / resolution.y;
  vec2 screen = vec2(
    (vUv.x * 2.0 - 1.0) * aspect,
    vUv.y * 2.0 - 1.0
  );
  float tanHalfFov = tan(0.5 * clamp(uFov, 0.02, 2.8));
  vec3 colour = traceBinary(cameraRay(screen, tanHalfFov));
  gl_FragColor = vec4(max(colour, vec3(0.0)), 1.0);
}
`;

export const binaryApproxShaderBundle = Object.freeze({
  id: "binary-approx-v1",
  labels: Object.freeze({
    uniforms: "Binary preview frame uniforms",
    trace: "Binary weak-field fast-light lens preview",
  }),
  wgsl: Object.freeze({
    trace: binaryTraceFragmentWGSL,
  }),
  glsl: Object.freeze({
    trace: binaryTraceFragmentGLSL,
  }),
  uniforms: Object.freeze({
    requiredFloatCount: 44,
    writeWebGPUExtras(tail, frame) {
      if (!frame.sceneBinaryState || frame.sceneBinaryState.length !== 4) {
        throw new Error("Binary scene frame is missing sceneBinaryState");
      }
      if (!frame.sceneBinaryMasses || frame.sceneBinaryMasses.length !== 4) {
        throw new Error("Binary scene frame is missing sceneBinaryMasses");
      }
      tail.set(frame.sceneBinaryState, 0);
      tail.set(frame.sceneBinaryMasses, 4);
    },
    createWebGLExtras(THREE) {
      return {
        uSceneBinaryState: { value: new THREE.Vector4() },
        uSceneBinaryMasses: { value: new THREE.Vector4() },
      };
    },
    writeWebGLExtras(uniforms, frame) {
      if (!frame.sceneBinaryState || frame.sceneBinaryState.length !== 4) {
        throw new Error("Binary scene frame is missing sceneBinaryState");
      }
      if (!frame.sceneBinaryMasses || frame.sceneBinaryMasses.length !== 4) {
        throw new Error("Binary scene frame is missing sceneBinaryMasses");
      }
      uniforms.uSceneBinaryState.value.fromArray(frame.sceneBinaryState);
      uniforms.uSceneBinaryMasses.value.fromArray(frame.sceneBinaryMasses);
    },
  }),
});
