// Veiling glare of the observer's eye for the self-luminous transient scenes.
//
// A display cannot reproduce a stellar surface ~10^9 times brighter than
// daylight, so, as in a real photograph or to the eye, what reads as
// "dazzling" is overexposure plus the light the optics scatter around a bright
// source. This pass adds that scattered light with the CIE 146:2002 general
// disability-glare function of a young observer,
//   L_veil / E_glare = 10 / theta^3 + (5 / theta^2 + 0.1 p / theta)
//                      (1 + (A / 62.5)^4) + 0.0025 p        [sr^-1],
// theta in degrees (0.1 to 100 deg), age A = 25 yr, pigmentation p = 0.5.
// It is a model of the observer, not of the source: no astrophysical quantity
// changes, and diagnostic false-colour frames never receive it. A bundle sets
// the strength relative to this eye (CAMERA_GLARE_STRENGTH for a camera
// lens); frame.sceneGlare may override it (false or 0 disables it).
//
// The PSF is approximated by a Gaussian pyramid: the scene is box-averaged
// over 4x4 pixels into level 2 (scale 1/4) and downsampled by a 13-tap filter
// (Jimenez 2014) into levels k = 3..8. Each level contributes a near-Gaussian
// kernel of width GLARE_LEVEL_SIGMA 2^k full-resolution pixels and carries the
// PSF energy of the octave annulus around 1.2 sigma_k; level 2 also takes the
// energy from 0.1 deg inward of it and the last level everything outward to
// the frame diagonal. Against the exact function this reproduces the halo to
// ~10% rms in log over 0.1-6 deg. The levels are recombined by tent
// upsampling into a quarter-resolution field that the post pass adds to the
// unscattered image before tone mapping.

// Pyramid levels k = GLARE_FIRST_LEVEL..GLARE_LAST_LEVEL (scale 2^-k).
export const GLARE_FIRST_LEVEL = 2;
const GLARE_LAST_LEVEL = 8;
// Width of one pyramid level's kernel in full-resolution pixels per 2^k,
// measured from the GPU impulse response of this downsample/upsample chain
// (1.06, 1.19, then 1.22 for levels 3 and up; energy conserved to 0.3%).
export const GLARE_LEVEL_SIGMA = 1.22;
const OCTAVE_CENTRE = 1.2;

export const GLARE_OBSERVER = Object.freeze({
  model: "CIE 146:2002 general disability glare",
  ageYears: 25,
  pigmentation: 0.5,
  minimumAngleDeg: 0.1,
});

// A young eye scatters ~7.8% of the light of a point source beyond 1 deg; a
// good camera lens has a veiling-glare index of ~2% (ISO 9358). The transient
// scenes render a camera, so they use the CIE shape at this fraction.
export const CAMERA_GLARE_STRENGTH = 0.3;

/** CIE 146:2002 disability-glare PSF (sr^-1), theta in degrees. */
export function cieDisabilityGlarePsf(thetaDeg, observer = GLARE_OBSERVER) {
  const theta = Math.max(thetaDeg, observer.minimumAngleDeg);
  const age = 1 + (observer.ageYears / 62.5) ** 4;
  return 10 / theta ** 3
    + (5 / theta ** 2 + 0.1 * observer.pigmentation / theta) * age
    + 0.0025 * observer.pigmentation;
}

/**
 * Energy fraction of the PSF carried by each pyramid level.
 * @param {object} options
 * @param {number} options.pixelAngleDeg angular size of one full-resolution pixel
 * @param {number} options.levels number of pyramid levels, from GLARE_FIRST_LEVEL
 * @param {number} options.width full-resolution width in pixels
 * @param {number} options.height full-resolution height in pixels
 */
export function glareLevelWeights({ pixelAngleDeg, levels, width, height, observer = GLARE_OBSERVER }) {
  if (!(pixelAngleDeg > 0) || !(levels >= 1)) {
    throw new RangeError("Glare weights need a positive pixel angle and at least one level");
  }
  const pixelSolidAngle = (pixelAngleDeg * Math.PI / 180) ** 2;
  const minimumRadius = Math.max(observer.minimumAngleDeg / pixelAngleDeg, 1);
  const maximumRadius = Math.hypot(width, height);
  const perPixel = (radius) => cieDisabilityGlarePsf(radius * pixelAngleDeg, observer) * pixelSolidAngle;
  const annulusEnergy = (inner, outer) => {
    if (!(outer > inner)) return 0;
    const samples = 96;
    const ratio = Math.log(outer / inner);
    let energy = 0;
    for (let i = 0; i < samples; i += 1) {
      const radius = inner * Math.exp(ratio * (i + 0.5) / samples);
      energy += perPixel(radius) * 2 * Math.PI * radius * radius * ratio / samples;
    }
    return energy;
  };
  const weights = new Float32Array(levels);
  for (let index = 0; index < levels; index += 1) {
    const k = GLARE_FIRST_LEVEL + index;
    const centre = OCTAVE_CENTRE * GLARE_LEVEL_SIGMA * 2 ** k;
    const inner = index === 0 ? minimumRadius : Math.max(centre / Math.SQRT2, minimumRadius);
    const outer = index === levels - 1 ? maximumRadius : Math.min(centre * Math.SQRT2, maximumRadius);
    weights[index] = annulusEnergy(inner, outer);
  }
  return weights;
}

// Levels from GLARE_FIRST_LEVEL while the level is at least 2 pixels across.
export function glareLevelCount(width, height) {
  let levels = 0;
  while (
    GLARE_FIRST_LEVEL + levels <= GLARE_LAST_LEVEL
    && Math.min(width, height) / 2 ** (GLARE_FIRST_LEVEL + levels) >= 2
  ) {
    levels += 1;
  }
  return levels;
}

const GLARE_VERTEX_WGSL = /* wgsl */ `
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vsGlare(@builtin(vertex_index) index: u32) -> VertexOutput {
  let x = f32((index << 1u) & 2u);
  let y = f32(index & 2u);
  var output: VertexOutput;
  output.position = vec4<f32>(2.0 * x - 1.0, 1.0 - 2.0 * y, 0.0, 1.0);
  output.uv = vec2<f32>(x, y);
  return output;
}
`;

const GLARE_DOWNSAMPLE_WGSL = /* wgsl */ `${GLARE_VERTEX_WGSL}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;

fn tap(uv: vec2<f32>, texel: vec2<f32>, offset: vec2<f32>) -> vec3<f32> {
  return textureSampleLevel(source, linearSampler, uv + texel * offset, 0.0).rgb;
}

// Full-resolution source to level 2: four bilinear taps, each at the shared
// corner of a 2x2 block, average the 4x4 block exactly.
@fragment
fn fsDownsampleBox(input: VertexOutput) -> @location(0) vec4<f32> {
  let texel = 1.0 / vec2<f32>(textureDimensions(source));
  let average = 0.25 * (
    tap(input.uv, texel, vec2<f32>(-1.0, -1.0)) + tap(input.uv, texel, vec2<f32>(1.0, -1.0))
    + tap(input.uv, texel, vec2<f32>(-1.0, 1.0)) + tap(input.uv, texel, vec2<f32>(1.0, 1.0))
  );
  return vec4<f32>(max(average, vec3<f32>(0.0)), 1.0);
}

// 13-tap downsample (Jimenez 2014): five overlapping 2x2 box averages.
@fragment
fn fsDownsample(input: VertexOutput) -> @location(0) vec4<f32> {
  let texel = 1.0 / vec2<f32>(textureDimensions(source));
  let uv = input.uv;
  let a = tap(uv, texel, vec2<f32>(-2.0, -2.0));
  let b = tap(uv, texel, vec2<f32>(0.0, -2.0));
  let c = tap(uv, texel, vec2<f32>(2.0, -2.0));
  let d = tap(uv, texel, vec2<f32>(-1.0, -1.0));
  let e = tap(uv, texel, vec2<f32>(1.0, -1.0));
  let f = tap(uv, texel, vec2<f32>(-2.0, 0.0));
  let g = tap(uv, texel, vec2<f32>(0.0, 0.0));
  let h = tap(uv, texel, vec2<f32>(2.0, 0.0));
  let i = tap(uv, texel, vec2<f32>(-1.0, 1.0));
  let j = tap(uv, texel, vec2<f32>(1.0, 1.0));
  let k = tap(uv, texel, vec2<f32>(-2.0, 2.0));
  let l = tap(uv, texel, vec2<f32>(0.0, 2.0));
  let m = tap(uv, texel, vec2<f32>(2.0, 2.0));
  let result = (d + e + i + j) * 0.125 + (a + c + k + m) * 0.03125
    + (b + f + h + l) * 0.0625 + g * 0.125;
  return vec4<f32>(max(result, vec3<f32>(0.0)), 1.0);
}
`;

const GLARE_TENT_WGSL = /* wgsl */ `
fn tent(image: texture_2d<f32>, uv: vec2<f32>) -> vec3<f32> {
  let texel = 1.0 / vec2<f32>(textureDimensions(image));
  var sum = textureSampleLevel(image, linearSampler, uv, 0.0).rgb * 4.0;
  sum = sum + 2.0 * (
    textureSampleLevel(image, linearSampler, uv + vec2<f32>(texel.x, 0.0), 0.0).rgb
    + textureSampleLevel(image, linearSampler, uv - vec2<f32>(texel.x, 0.0), 0.0).rgb
    + textureSampleLevel(image, linearSampler, uv + vec2<f32>(0.0, texel.y), 0.0).rgb
    + textureSampleLevel(image, linearSampler, uv - vec2<f32>(0.0, texel.y), 0.0).rgb
  );
  sum = sum
    + textureSampleLevel(image, linearSampler, uv + texel, 0.0).rgb
    + textureSampleLevel(image, linearSampler, uv - texel, 0.0).rgb
    + textureSampleLevel(image, linearSampler, uv + vec2<f32>(texel.x, -texel.y), 0.0).rgb
    + textureSampleLevel(image, linearSampler, uv + vec2<f32>(-texel.x, texel.y), 0.0).rgb;
  return sum / 16.0;
}
`;

const GLARE_UPSAMPLE_WGSL = /* wgsl */ `${GLARE_VERTEX_WGSL}
@group(0) @binding(0) var lower: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var level: texture_2d<f32>;
// weight of this level, 1 if a coarser level is accumulated, reserved
@group(0) @binding(3) var<uniform> control: vec4<f32>;
${GLARE_TENT_WGSL}
@fragment
fn fsUpsample(input: VertexOutput) -> @location(0) vec4<f32> {
  var sum = textureSampleLevel(level, linearSampler, input.uv, 0.0).rgb * control.x;
  if (control.y > 0.5) {
    sum = sum + tent(lower, input.uv);
  }
  return vec4<f32>(sum, 1.0);
}
`;

export const GLARE_SHADER_SOURCES = Object.freeze({
  downsample: GLARE_DOWNSAMPLE_WGSL,
  upsample: GLARE_UPSAMPLE_WGSL,
});

/**
 * GPU glare pass. resize() allocates the pyramid, setSources() binds the
 * textures the post pass may read, encode() writes the quarter-resolution glare
 * field (glareView) that the post pass adds.
 */
export class GlarePass {
  constructor(device, format) {
    this.device = device;
    this.format = format;
    this.sampler = device.createSampler({
      label: "Glare pyramid sampler",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
      magFilter: "linear",
      minFilter: "linear",
    });
    const pipeline = (label, code, entryPoint) => {
      const module = device.createShaderModule({ label: `${label} shader`, code });
      return device.createRenderPipeline({
        label,
        layout: "auto",
        vertex: { module, entryPoint: "vsGlare" },
        fragment: { module, entryPoint, targets: [{ format }] },
        primitive: { topology: "triangle-list" },
      });
    };
    this.sourcePipeline = pipeline("Glare source downsample", GLARE_DOWNSAMPLE_WGSL, "fsDownsampleBox");
    this.downsamplePipeline = pipeline("Glare downsample", GLARE_DOWNSAMPLE_WGSL, "fsDownsample");
    this.upsamplePipeline = pipeline("Glare upsample", GLARE_UPSAMPLE_WGSL, "fsUpsample");
    this.levels = 0;
    this.textures = [];
    this.weightsKey = "";
    this.levelControls = [];
    this.sourceBindGroups = [];
  }

  // Quarter-resolution glare field read by the post pass.
  get glareView() {
    return this.textures[0]?.upView ?? null;
  }

  resize(width, height) {
    this.destroyTargets();
    this.width = width;
    this.height = height;
    this.levels = glareLevelCount(width, height);
    const { device, format } = this;
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    for (let index = 0; index < this.levels; index += 1) {
      const k = GLARE_FIRST_LEVEL + index;
      const size = [Math.max(1, Math.floor(width / 2 ** k)), Math.max(1, Math.floor(height / 2 ** k)), 1];
      const down = device.createTexture({ label: `Glare downsample level ${k}`, size, format, usage });
      const up = device.createTexture({ label: `Glare upsample level ${k}`, size, format, usage });
      this.textures.push({ down, up, downView: down.createView(), upView: up.createView() });
      if (!this.levelControls[index]) {
        this.levelControls[index] = device.createBuffer({
          label: `Glare level ${k} control`,
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
      }
    }
    this.chainBindGroups = [];
    for (let index = 1; index < this.levels; index += 1) {
      this.chainBindGroups.push(this.device.createBindGroup({
        label: `Glare downsample to level ${GLARE_FIRST_LEVEL + index}`,
        layout: this.downsamplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.textures[index - 1].downView },
          { binding: 1, resource: this.sampler },
        ],
      }));
    }
    this.upsampleBindGroups = [];
    for (let index = 0; index < this.levels; index += 1) {
      const coarser = index < this.levels - 1 ? this.textures[index + 1].upView : this.textures[index].downView;
      this.upsampleBindGroups.push(this.device.createBindGroup({
        label: `Glare upsample level ${GLARE_FIRST_LEVEL + index}`,
        layout: this.upsamplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: coarser },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: this.textures[index].downView },
          { binding: 3, resource: { buffer: this.levelControls[index] } },
        ],
      }));
    }
    this.weightsKey = "";
  }

  // Views the post pass may read (trace target, accumulation histories).
  setSources(views) {
    this.sourceBindGroups = views.map((view, index) => this.device.createBindGroup({
      label: `Glare source ${index}`,
      layout: this.sourcePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: view },
        { binding: 1, resource: this.sampler },
      ],
    }));
  }

  // Level weights for the frame's field of view (radians, vertical).
  updateWeights(verticalFov, strength = 1) {
    const key = `${verticalFov.toFixed(6)}:${this.width}x${this.height}:${strength}`;
    if (key === this.weightsKey || this.levels === 0) {
      return;
    }
    this.weightsKey = key;
    const weights = glareLevelWeights({
      pixelAngleDeg: verticalFov * 180 / Math.PI / this.height,
      levels: this.levels,
      width: this.width,
      height: this.height,
    });
    for (let index = 0; index < this.levels; index += 1) {
      this.device.queue.writeBuffer(
        this.levelControls[index],
        0,
        new Float32Array([strength * weights[index], index < this.levels - 1 ? 1 : 0, 0, 0]),
      );
    }
    this.weights = weights;
  }

  encode(encoder, sourceIndex) {
    if (this.levels === 0 || !this.sourceBindGroups[sourceIndex]) {
      return false;
    }
    const draw = (view, pipeline, bindGroup) => {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
    };
    draw(this.textures[0].downView, this.sourcePipeline, this.sourceBindGroups[sourceIndex]);
    for (let index = 1; index < this.levels; index += 1) {
      draw(this.textures[index].downView, this.downsamplePipeline, this.chainBindGroups[index - 1]);
    }
    for (let index = this.levels - 1; index >= 0; index -= 1) {
      draw(this.textures[index].upView, this.upsamplePipeline, this.upsampleBindGroups[index]);
    }
    return true;
  }

  destroyTargets() {
    for (const level of this.textures) {
      level.down.destroy();
      level.up.destroy();
    }
    this.textures = [];
    this.sourceBindGroups = [];
  }

  dispose() {
    this.destroyTargets();
    for (const buffer of this.levelControls) {
      buffer.destroy();
    }
    this.levelControls = [];
  }
}
