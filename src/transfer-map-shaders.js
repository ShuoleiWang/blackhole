/*
 * Stationary transfer-map consumer.
 *
 * The CPU loader has already authenticated and validated every 32-byte record.
 * WebGPU receives the canonical 32-byte records as a read-only storage buffer.
 * WebGL2 receives two immutable RGBA32F planes:
 *   primary = escapeDirection.xyz, frequencyShiftG
 *   metrics = coordinateLookbackTimeM, nullResidual, projectionErrorPx,
 *             outcome + 256*target + 65536*validityMask
 *
 * Every output fragment selects the nearest detector texel. Ray directions
 * are deliberately never interpolated: deflection is singular at the photon
 * separatrix, so linear direction blending has no defensible global error
 * bound. Only outcome=escaped is allowed to sample the sky.
 */

export const transferMapTraceFragmentWGSL = /* wgsl */ `
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const OUTCOME_ESCAPED: f32 = 0.0;
const OUTCOME_CAPTURED: f32 = 1.0;

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
  sceneTransferState: vec4<f32>,
};

struct FragmentInput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

struct TransferSample {
  primary: vec4<f32>,
  metrics: vec3<f32>,
  state: u32,
};

struct TransferRecord {
  primary: vec4<f32>,
  metrics: vec3<f32>,
  state: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tSky: texture_2d<f32>;
@group(0) @binding(2) var skySampler: sampler;
@group(0) @binding(3) var<storage, read> sceneTransferRecords: array<TransferRecord>;

fn safeNormalize(value: vec3<f32>) -> vec3<f32> {
  return value * inverseSqrt(max(dot(value, value), 1.0e-18));
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
  let mapped = cubeStarCoordinates(direction);
  let coordinates = mapped.xy * 256.0;
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

fn sampleEnvironment(direction: vec3<f32>, frequencyShift: f32) -> vec3<f32> {
  let d = safeNormalize(direction);
  // Canonical ICRS: +Z is the north celestial pole and longitude is measured
  // from +X toward +Y.
  let longitude = atan2(d.y, d.x);
  let latitude = asin(clamp(d.z, -1.0, 1.0));
  let skyUv = vec2<f32>(
    fract(longitude / TWO_PI + 0.5),
    clamp(0.5 - latitude / PI, 0.00001, 0.99999)
  );
  let panorama = max(
    textureSampleLevel(tSky, skySampler, skyUv, 0.0).rgb - vec3<f32>(0.0015),
    vec3<f32>(0.0)
  );
  // I_nu / nu^3 is invariant. The panorama is broadband rather than spectral,
  // so g^4 is the declared bolometric display approximation.
  let bolometricShift = pow(clamp(frequencyShift, 0.25, 4.0), 4.0);
  return (
    panorama * max(params.displayOutput.w, 0.01) + analyticStars(d)
  ) * bolometricShift;
}

fn loadTransfer(pixel: vec2<i32>) -> TransferSample {
  let width = u32(params.sceneTransferState.y + 0.5);
  let index = u32(pixel.y) * width + u32(pixel.x);
  let record = sceneTransferRecords[index];
  var sample: TransferSample;
  sample.primary = record.primary;
  sample.metrics = record.metrics;
  sample.state = record.state;
  return sample;
}

fn isEscaped(sample: TransferSample) -> bool {
  return (sample.state & 255u) == 0u;
}

fn transferAt(mapUv: vec2<f32>) -> TransferSample {
  let size = vec2<i32>(params.sceneTransferState.yz + vec2<f32>(0.5));
  let maximum = size - vec2<i32>(1);
  let coordinate = mapUv * vec2<f32>(size) - vec2<f32>(0.5);
  let nearest = clamp(
    vec2<i32>(floor(coordinate + vec2<f32>(0.5))),
    vec2<i32>(0),
    maximum
  );
  return loadTransfer(nearest);
}

fn mapUvForCanvas(uv: vec2<f32>) -> vec3<f32> {
  let resolution = max(params.resolutionTimeMass.xy, vec2<f32>(1.0));
  let mapSize = params.sceneTransferState.yz;
  let canvasAspect = resolution.x / resolution.y;
  let mapAspect = mapSize.x / mapSize.y;
  var mapped = uv;
  var inside = true;
  if (canvasAspect > mapAspect) {
    let widthFraction = mapAspect / canvasAspect;
    mapped.x = (uv.x - 0.5) / widthFraction + 0.5;
    inside = mapped.x >= 0.0 && mapped.x <= 1.0;
  } else {
    let heightFraction = canvasAspect / mapAspect;
    mapped.y = (uv.y - 0.5) / heightFraction + 0.5;
    inside = mapped.y >= 0.0 && mapped.y <= 1.0;
  }
  return vec3<f32>(clamp(mapped, vec2<f32>(0.0), vec2<f32>(1.0)), select(0.0, 1.0, inside));
}

fn outcomeColour(outcome: u32) -> vec3<f32> {
  if (outcome == 0u) {
    return vec3<f32>(0.10, 0.72, 0.92);
  }
  if (outcome == 1u) {
    return vec3<f32>(0.015, 0.018, 0.022);
  }
  if (outcome == 2u) {
    return vec3<f32>(0.95, 0.61, 0.10);
  }
  if (outcome == 3u) {
    return vec3<f32>(0.67, 0.28, 0.90);
  }
  if (outcome == 4u) {
    return vec3<f32>(0.92, 0.15, 0.14);
  }
  return vec3<f32>(0.95, 0.12, 0.65);
}

@fragment
fn fsMain(input: FragmentInput) -> @location(0) vec4<f32> {
  let mapped = mapUvForCanvas(input.uv);
  if (mapped.z < 0.5) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let sample = transferAt(mapped.xy);
  if (params.sceneTransferState.x > 0.5) {
    return vec4<f32>(outcomeColour(sample.state & 255u), 1.0);
  }
  if (!isEscaped(sample)) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let colour = sampleEnvironment(sample.primary.xyz, sample.primary.w);
  return vec4<f32>(max(colour, vec3<f32>(0.0)), 1.0);
}
`;

export const transferMapTraceFragmentGLSL = /* glsl */ `
precision highp float;
precision highp int;

uniform vec2 uResolution;
uniform float uSkyRadianceScale;
uniform vec4 uSceneTransferState;
uniform sampler2D tSky;
uniform sampler2D uSceneTransferPrimary;
uniform sampler2D uSceneTransferMetrics;

varying vec2 vUv;

const float PI = 3.14159265358979323846;
const float TWO_PI = 6.28318530717958647692;
const float OUTCOME_ESCAPED = 0.0;
const float OUTCOME_CAPTURED = 1.0;

struct TransferSample {
  vec4 primary;
  vec4 metrics;
};

vec3 safeNormalize(vec3 value) {
  return value * inversesqrt(max(dot(value, value), 1.0e-18));
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
  vec2 coordinates = mapped.xy * 256.0;
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

vec3 sampleEnvironment(vec3 direction, float frequencyShift) {
  vec3 d = safeNormalize(direction);
  // Canonical ICRS: +Z is north; right ascension advances from +X toward +Y.
  float longitude = atan(d.y, d.x);
  float latitude = asin(clamp(d.z, -1.0, 1.0));
  vec2 skyUv = vec2(
    fract(longitude / TWO_PI + 0.5),
    clamp(0.5 - latitude / PI, 0.00001, 0.99999)
  );
  vec3 panorama = max(
    textureLod(tSky, skyUv, 0.0).rgb - vec3(0.0015),
    vec3(0.0)
  );
  float bolometricShift = pow(clamp(frequencyShift, 0.25, 4.0), 4.0);
  return (
    panorama * max(uSkyRadianceScale, 0.01) + analyticStars(d)
  ) * bolometricShift;
}

TransferSample loadTransfer(ivec2 pixel) {
  TransferSample transferValue;
  transferValue.primary = texelFetch(uSceneTransferPrimary, pixel, 0);
  transferValue.metrics = texelFetch(uSceneTransferMetrics, pixel, 0);
  return transferValue;
}

bool isEscaped(TransferSample transferValue) {
  return abs(mod(transferValue.metrics.w, 256.0) - OUTCOME_ESCAPED) < 0.25;
}

TransferSample transferAt(vec2 mapUv) {
  ivec2 size = textureSize(uSceneTransferPrimary, 0);
  ivec2 maximum = size - ivec2(1);
  vec2 coordinate = mapUv * vec2(size) - 0.5;
  ivec2 nearest = clamp(ivec2(floor(coordinate + 0.5)), ivec2(0), maximum);
  return loadTransfer(nearest);
}

vec3 mapUvForCanvas(vec2 uv) {
  vec2 resolution = max(uResolution, vec2(1.0));
  vec2 mapSize = vec2(textureSize(uSceneTransferPrimary, 0));
  float canvasAspect = resolution.x / resolution.y;
  float mapAspect = mapSize.x / mapSize.y;
  vec2 mapped = uv;
  bool inside = true;
  if (canvasAspect > mapAspect) {
    float widthFraction = mapAspect / canvasAspect;
    mapped.x = (uv.x - 0.5) / widthFraction + 0.5;
    inside = mapped.x >= 0.0 && mapped.x <= 1.0;
  } else {
    float heightFraction = canvasAspect / mapAspect;
    mapped.y = (uv.y - 0.5) / heightFraction + 0.5;
    inside = mapped.y >= 0.0 && mapped.y <= 1.0;
  }
  return vec3(clamp(mapped, 0.0, 1.0), inside ? 1.0 : 0.0);
}

vec3 outcomeColour(float outcome) {
  if (abs(outcome - OUTCOME_ESCAPED) < 0.25) {
    return vec3(0.10, 0.72, 0.92);
  }
  if (abs(outcome - OUTCOME_CAPTURED) < 0.25) {
    return vec3(0.015, 0.018, 0.022);
  }
  if (abs(outcome - 2.0) < 0.25) {
    return vec3(0.95, 0.61, 0.10);
  }
  if (abs(outcome - 3.0) < 0.25) {
    return vec3(0.67, 0.28, 0.90);
  }
  if (abs(outcome - 4.0) < 0.25) {
    return vec3(0.92, 0.15, 0.14);
  }
  return vec3(0.95, 0.12, 0.65);
}

void main() {
  // Three.js UVs use a bottom-left origin; transfer-map rows are top-left.
  vec3 mapped = mapUvForCanvas(vec2(vUv.x, 1.0 - vUv.y));
  if (mapped.z < 0.5) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }
  TransferSample transferValue = transferAt(mapped.xy);
  if (uSceneTransferState.x > 0.5) {
    gl_FragColor = vec4(
      outcomeColour(mod(transferValue.metrics.w, 256.0)),
      1.0
    );
    return;
  }
  if (!isEscaped(transferValue)) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }
  vec3 colour = sampleEnvironment(
    transferValue.primary.xyz,
    transferValue.primary.w
  );
  gl_FragColor = vec4(max(colour, vec3(0.0)), 1.0);
}
`;

function configureWebGLTexture(THREE, texture) {
  texture.flipY = false;
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  if (THREE.NoColorSpace) {
    texture.colorSpace = THREE.NoColorSpace;
  }
  texture.needsUpdate = true;
}

export function createTransferMapShaderBundle(dataset) {
  if (
    !(dataset?.primary instanceof Float32Array)
    || !(dataset?.metrics instanceof Float32Array)
    || !(dataset?.records instanceof Uint8Array)
    || !Number.isInteger(dataset.width)
    || !Number.isInteger(dataset.height)
    || dataset.primary.length !== dataset.width * dataset.height * 4
    || dataset.metrics.length !== dataset.width * dataset.height * 4
    || dataset.records.byteLength !== dataset.width * dataset.height * 32
  ) {
    throw new Error("Transfer-map shader bundle requires validated RGBA32F planes");
  }

  return Object.freeze({
    id: "stationary-transfer-map-reference-v1",
    labels: Object.freeze({
      uniforms: "Stationary transfer-map frame uniforms",
      trace: "Validated Schwarzschild transfer-map compositor",
    }),
    wgsl: Object.freeze({
      trace: transferMapTraceFragmentWGSL,
    }),
    glsl: Object.freeze({
      trace: transferMapTraceFragmentGLSL,
    }),
    uniforms: Object.freeze({
      requiredFloatCount: 44,
      writeWebGPUExtras(tail, frame) {
        tail.set(
          frame.sceneTransferState
          || [0, dataset.width, dataset.height, 0],
          0,
        );
      },
      createWebGLExtras(THREE) {
        return {
          uSceneTransferState: { value: new THREE.Vector4() },
        };
      },
      writeWebGLExtras(uniforms, frame) {
        uniforms.uSceneTransferState.value.fromArray(
          frame.sceneTransferState || [0, 0, 0, 0],
        );
      },
    }),
    resources: Object.freeze({
      createWebGPU(device) {
        let recordBuffer;
        try {
          recordBuffer = device.createBuffer({
            label: "Transfer map · canonical 32-byte ray records",
            size: dataset.records.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          device.queue.writeBuffer(recordBuffer, 0, dataset.records);
        } catch (error) {
          recordBuffer?.destroy();
          throw error;
        }
        return Object.freeze({
          entries: Object.freeze([
            Object.freeze({ binding: 3, resource: { buffer: recordBuffer } }),
          ]),
          dispose() {
            recordBuffer.destroy();
          },
        });
      },
      createWebGL(THREE) {
        const primaryTexture = new THREE.DataTexture(
          dataset.primary,
          dataset.width,
          dataset.height,
          THREE.RGBAFormat,
          THREE.FloatType,
        );
        const metricsTexture = new THREE.DataTexture(
          dataset.metrics,
          dataset.width,
          dataset.height,
          THREE.RGBAFormat,
          THREE.FloatType,
        );
        configureWebGLTexture(THREE, primaryTexture);
        configureWebGLTexture(THREE, metricsTexture);
        return Object.freeze({
          uniforms: Object.freeze({
            uSceneTransferPrimary: { value: primaryTexture },
            uSceneTransferMetrics: { value: metricsTexture },
          }),
          dispose() {
            primaryTexture.dispose();
            metricsTexture.dispose();
          },
        });
      },
    }),
  });
}
