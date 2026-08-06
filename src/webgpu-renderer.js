import {
  fullscreenVertexWGSL,
  traceFragmentWGSL,
  postFragmentWGSL,
} from "./shaders.js";

const HDR_FORMAT = "rgba16float";
const UNIFORM_FLOATS = 40;
const ULTRA_SKY_DIMENSION = 16000;
const ORIGINAL_SKY_DIMENSIONS = Object.freeze([
  Object.freeze({ token: "gaia-edr3-16k", width: 16000, height: 8000 }),
  Object.freeze({ token: "milky-way-360-6k", width: 6000, height: 3000 }),
]);
const PROGRESSIVE_ACCUMULATION_MODE = "linear-hdr-running-average-v1";
const DYNAMIC_PROGRESSIVE_PHASES = new Set(["interactive", "realtime"]);
const WEBGPU_FALLBACK_REASONS = Object.freeze({
  "device-lost": "WebGPU 设备连接丢失",
  "render-error": "WebGPU 渲染异常",
});

export function webGLRecoveryUrl(href, reason) {
  if (!Object.hasOwn(WEBGPU_FALLBACK_REASONS, reason)) {
    throw new Error(`Unsupported WebGPU recovery reason ${JSON.stringify(reason)}`);
  }
  const url = new URL(href);
  url.searchParams.set("renderer", "webgl");
  url.searchParams.set("fallback", `webgpu-${reason}`);
  return url.href;
}

export function webGPUFallbackDescription(token) {
  const prefix = "webgpu-";
  const reason = typeof token === "string" && token.startsWith(prefix)
    ? token.slice(prefix.length)
    : "";
  return WEBGPU_FALLBACK_REASONS[reason] || "";
}

function monotonicNowMs() {
  return globalThis.performance?.now?.() ?? Date.now();
}

/*
 * WebGPU queue.submit() is intentionally asynchronous.  Without an explicit
 * gate, requestAnimationFrame can enqueue many expensive ray-trace passes
 * while Metal is still executing the first one.  Apart from input latency,
 * that also makes RAF cadence a dangerously optimistic proxy for GPU time.
 *
 * This small queue-agnostic contract keeps exactly one submission in flight
 * and exposes each completed wall-time sample once to the quality scheduler.
 */
export class WebGPUFrameSubmissionGate {
  constructor(now = monotonicNowMs) {
    if (typeof now !== "function") {
      throw new TypeError("WebGPU submission gate clock must be a function");
    }
    this.now = now;
    this.inFlight = false;
    this.closed = false;
    this.generation = 0;
    this.submittedAtMs = null;
    this.lastQueueCompletionAtMs = null;
    this.lastCompletedFrameTimeMs = null;
    this.pendingCompletedFrameTimeMs = null;
    this.lastCompletionError = null;
    this.onFailure = null;
    this.externalWorkDepth = 0;
  }

  get readyForFrame() {
    return (
      !this.closed
      && !this.inFlight
      && this.externalWorkDepth === 0
    );
  }

  canSubmitFrame() {
    return this.readyForFrame;
  }

  submit(queue, commandBuffers) {
    if (!this.readyForFrame) {
      return false;
    }
    if (
      !queue
      || typeof queue.submit !== "function"
      || typeof queue.onSubmittedWorkDone !== "function"
    ) {
      throw new TypeError(
        "WebGPU submission gate requires submit() and onSubmittedWorkDone()",
      );
    }
    if (!Array.isArray(commandBuffers) || commandBuffers.length === 0) {
      throw new TypeError("WebGPU submission requires at least one command buffer");
    }

    const generation = ++this.generation;
    const submittedAtMs = this.now();
    this.inFlight = true;
    this.submittedAtMs = submittedAtMs;
    this.lastCompletionError = null;
    try {
      queue.submit(commandBuffers);
    } catch (error) {
      this.inFlight = false;
      this.submittedAtMs = null;
      throw error;
    }

    let completion;
    try {
      completion = queue.onSubmittedWorkDone();
    } catch (error) {
      // The work was already submitted, so remain closed to further frames
      // rather than risk violating the one-in-flight invariant.
      this.lastCompletionError = error;
      this.closed = true;
      throw error;
    }
    Promise.resolve(completion).then(
      () => this.complete(generation, submittedAtMs),
      (error) => this.fail(generation, error),
    );
    return true;
  }

  complete(generation, submittedAtMs) {
    if (this.closed || generation !== this.generation) {
      return;
    }
    const completedAtMs = this.now();
    const elapsedMs = Math.max(completedAtMs - submittedAtMs, 0.001);
    this.inFlight = false;
    this.submittedAtMs = null;
    this.lastQueueCompletionAtMs = completedAtMs;
    this.lastCompletedFrameTimeMs = elapsedMs;
    this.pendingCompletedFrameTimeMs = elapsedMs;
  }

  fail(generation, error) {
    if (this.closed || generation !== this.generation) {
      return;
    }
    this.inFlight = false;
    this.closed = true;
    this.submittedAtMs = null;
    this.lastCompletionError = error;
    try {
      this.onFailure?.(error);
    } catch (callbackError) {
      console.error("WebGPU submission failure callback threw", callbackError);
    }
  }

  consumeCompletedFrameTimeMs() {
    const sample = this.pendingCompletedFrameTimeMs;
    this.pendingCompletedFrameTimeMs = null;
    return sample;
  }

  /*
   * copyExternalImageToTexture() and similar resource uploads share the same
   * GPUQueue as traced frames. A frame submitted behind a large sky upload
   * would otherwise report upload + render wall time to the quality governor.
   * External scopes pause new frame submissions until their own
   * onSubmittedWorkDone() boundary has completed. A frame that was already in
   * flight remains independently timed because its completion promise was
   * registered before the later upload.
   */
  beginExternalQueueWork() {
    if (this.closed) {
      return () => {};
    }
    this.externalWorkDepth += 1;
    let active = true;
    return () => {
      if (!active) {
        return;
      }
      active = false;
      if (this.externalWorkDepth > 0) {
        this.externalWorkDepth -= 1;
      }
    };
  }

  close() {
    this.closed = true;
    this.inFlight = false;
    this.externalWorkDepth = 0;
    this.submittedAtMs = null;
    this.generation += 1;
    this.pendingCompletedFrameTimeMs = null;
    this.onFailure = null;
  }
}

export const progressiveAccumulationFragmentWGSL = /* wgsl */ `
struct FragmentInput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

struct AccumulationParams {
  weightResetIndexEpoch: vec4<f32>,
};

@group(0) @binding(0) var previousFrame: texture_2d<f32>;
@group(0) @binding(1) var rawTrace: texture_2d<f32>;
@group(0) @binding(2) var<uniform> accumulation: AccumulationParams;

@fragment
fn fsMain(input: FragmentInput) -> @location(0) vec4<f32> {
  let dimensions = textureDimensions(rawTrace);
  let pixel = clamp(
    vec2<i32>(input.position.xy),
    vec2<i32>(0),
    vec2<i32>(dimensions) - vec2<i32>(1)
  );
  let raw = textureLoad(rawTrace, pixel, 0);
  let reset = accumulation.weightResetIndexEpoch.y > 0.5;
  if (reset) {
    return raw;
  }
  let previous = textureLoad(previousFrame, pixel, 0);
  let weight = clamp(accumulation.weightResetIndexEpoch.x, 0.0, 1.0);
  return mix(previous, raw, weight);
}
`;

function shaderBundleFrom(options) {
  return options?.shaderBundle || {};
}

function traceSpecializationsFrom(bundle) {
  const declarations = bundle.wgsl?.traceSpecializations;
  if (declarations == null) {
    return Object.freeze([
      Object.freeze({ id: "default", constants: undefined }),
    ]);
  }
  if (!Array.isArray(declarations) || declarations.length < 1) {
    throw new Error("WGSL trace specializations must be a non-empty array");
  }
  const ids = new Set();
  return Object.freeze(declarations.map((declaration, index) => {
    const id = String(declaration?.id || "");
    if (!id || ids.has(id)) {
      throw new Error(`WGSL trace specialization ${index} has an invalid id`);
    }
    ids.add(id);
    const constants = declaration.constants;
    if (
      !constants
      || typeof constants !== "object"
      || Array.isArray(constants)
      || Object.values(constants).some((value) => !Number.isFinite(Number(value)))
    ) {
      throw new Error(`WGSL trace specialization ${id} has invalid constants`);
    }
    return Object.freeze({ id, constants: Object.freeze({ ...constants }) });
  }));
}

function progressiveAccumulationFrom(options, bundle) {
  const declaration = options?.progressiveAccumulation ?? bundle.accumulation;
  if (declaration == null && bundle.id !== "binary-strong-field-v1") {
    return null;
  }
  const mode = declaration?.mode ?? (
    bundle.id === "binary-strong-field-v1"
      ? PROGRESSIVE_ACCUMULATION_MODE
      : null
  );
  if (mode !== PROGRESSIVE_ACCUMULATION_MODE) {
    throw new Error(
      `Unsupported WebGPU progressive accumulation mode ${JSON.stringify(mode)}`,
    );
  }
  return Object.freeze({ mode });
}

function finiteHistoryValue(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error(`Progressive history ${name} must be finite`);
  }
  return number;
}

function appendHistoryValues(target, name, value) {
  if (value == null) {
    target.push(`${name}=null`);
    return;
  }
  if (typeof value === "number") {
    target.push(`${name}=${finiteHistoryValue(value, name)}`);
    return;
  }
  if (Array.isArray(value) || ArrayBuffer.isView(value)) {
    target.push(`${name}=[`);
    for (let index = 0; index < value.length; index += 1) {
      target.push(String(finiteHistoryValue(value[index], `${name}[${index}]`)));
    }
    target.push("]");
    return;
  }
  throw new Error(`Progressive history ${name} has an unsupported value`);
}

export function progressiveHistorySignature(frame) {
  if (!frame || typeof frame !== "object" || Array.isArray(frame)) {
    throw new TypeError("Progressive history requires a frame object");
  }
  const values = [];
  for (const name of [
    "time",
    "massSolar",
    "accretion",
    "exposure",
    "mode",
    "steps",
    "cameraPos",
    "cameraRadius",
    "forward",
    "fov",
    "right",
    "skyRotation",
    "up",
    "diskOuterRadius",
    "renderScale",
    "bloom",
    "motion",
    "observerVelocity",
    "observerBeta",
    "sceneStrongFieldUniforms",
    "sceneStrongIntegrator",
    "sceneStrongDomain",
    "sceneStrongDiagnostics",
    "sceneBinaryState",
    "sceneBinaryMasses",
  ]) {
    appendHistoryValues(values, name, frame[name]);
  }
  return values.join("|");
}

export function progressiveFrameState(frame) {
  const quality = frame?.strongFieldQuality;
  if (quality == null) {
    return Object.freeze({
      accumulationIndex: 0,
      accumulationWeight: 1,
      historyEpoch: 0,
      historyReset: true,
    });
  }
  const accumulationIndex = finiteHistoryValue(
    quality.accumulationIndex,
    "accumulationIndex",
  );
  const accumulationWeight = finiteHistoryValue(
    quality.accumulationWeight,
    "accumulationWeight",
  );
  const historyEpoch = finiteHistoryValue(
    quality.historyEpoch,
    "historyEpoch",
  );
  if (
    !Number.isInteger(accumulationIndex)
    || accumulationIndex < 0
    || !Number.isInteger(historyEpoch)
    || historyEpoch < 0
    || accumulationWeight <= 0
    || accumulationWeight > 1
  ) {
    throw new Error("Progressive history state is outside its valid range");
  }
  const expectedWeight = 1 / (accumulationIndex + 1);
  if (Math.abs(accumulationWeight - expectedWeight) > 1e-7) {
    throw new Error(
      "Progressive accumulation weight must equal 1/(sampleIndex+1)",
    );
  }
  const historyReset = quality.historyReset === true;
  if (historyReset && accumulationIndex !== 0) {
    throw new Error("A progressive history reset must restart at sample zero");
  }
  return Object.freeze({
    accumulationIndex,
    accumulationWeight,
    historyEpoch,
    historyReset,
  });
}

function isDynamicProgressiveFrame(frame) {
  return (
    Number(frame?.motion) !== 0
    || DYNAMIC_PROGRESSIVE_PHASES.has(
      frame?.strongFieldQuality?.convergencePhase,
    )
  );
}

function uniformFloatCount(bundle) {
  const requested = Number(bundle.uniforms?.requiredFloatCount ?? UNIFORM_FLOATS);
  if (!Number.isInteger(requested) || requested < UNIFORM_FLOATS || requested % 4 !== 0) {
    throw new Error(
      `Scene shader uniform size must be a multiple of four and at least ${UNIFORM_FLOATS} floats`,
    );
  }
  return requested;
}

function createSceneResources(bundle, device) {
  const create = bundle.resources?.createWebGPU;
  if (!create) {
    return null;
  }
  const resources = create(device);
  if (
    !resources
    || !Array.isArray(resources.entries)
    || typeof resources.dispose !== "function"
  ) {
    resources?.dispose?.();
    throw new Error(
      "Scene WebGPU resources must provide bind-group entries and dispose()",
    );
  }
  const bindings = new Set();
  for (const entry of resources.entries) {
    if (
      !Number.isInteger(entry?.binding)
      || entry.binding < 3
      || !entry.resource
      || bindings.has(entry.binding)
    ) {
      resources.dispose();
      throw new Error(
        "Scene WebGPU resource bindings must be unique integers starting at 3",
      );
    }
    bindings.add(entry.binding);
  }
  return resources;
}

function isApplePlatform() {
  const platform = navigator.userAgentData?.platform || navigator.platform || "";
  return /Mac|iPhone|iPad/i.test(platform);
}

function finiteLimit(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.floor(number) : fallback;
}

function selectedSkyUrl(urls, requireUltra = false) {
  const source = typeof urls === "string" ? { high: urls } : urls;
  const selected = requireUltra ? source.ultra : source.high;
  if (!selected) {
    throw new Error(
      requireUltra
        ? "The explicit Gaia 16K sky source is unavailable"
        : "The required ESO 6K sky source is unavailable",
    );
  }
  return selected;
}

function assertOriginalSkyDimensions(url, width, height) {
  const normalizedUrl = String(url).toLowerCase();
  const expected = ORIGINAL_SKY_DIMENSIONS.find(({ token }) => (
    normalizedUrl.includes(token)
  ));
  if (expected && (width !== expected.width || height !== expected.height)) {
    throw new Error(
      `${url} decoded at ${width}×${height}; expected the original ${expected.width}×${expected.height} pixels`,
    );
  }
}

async function loadBitmap(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Sky texture request failed (${response.status})`);
  }
  const blob = await response.blob();
  try {
    return await createImageBitmap(blob, { colorSpaceConversion: "none" });
  } catch (bitmapError) {
    // Chromium can reject very large PNGs in the ImageBitmap decoder even
    // when the GPU advertises a 16K texture limit.  HTMLImageElement uses the
    // browser's streaming image decoder and is also a valid external-image
    // source for copyExternalImageToTexture.
    const objectUrl = URL.createObjectURL(blob);
    const image = new Image();
    image.decoding = "async";
    image.src = objectUrl;
    try {
      await image.decode();
      image.close = () => URL.revokeObjectURL(objectUrl);
      return image;
    } catch (imageError) {
      image.src = "";
      URL.revokeObjectURL(objectUrl);
      throw new AggregateError(
        [bitmapError, imageError],
        `Unable to decode celestial panorama ${url}`,
      );
    }
  }
}

async function uploadSkyTexture(
  device,
  bitmap,
  url,
  beginExternalQueueWork = undefined,
) {
  assertOriginalSkyDimensions(url, bitmap.width, bitmap.height);
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  let texture;
  let thrownError;
  let finishExternalQueueWork = null;
  try {
    texture = device.createTexture({
      label: `Celestial sphere · ${url}`,
      size: [bitmap.width, bitmap.height, 1],
      format: "rgba8unorm-srgb",
      mipLevelCount: 1,
      usage: GPUTextureUsage.TEXTURE_BINDING
        | GPUTextureUsage.COPY_DST
        | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    finishExternalQueueWork = beginExternalQueueWork?.() ?? null;
    device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture },
      [bitmap.width, bitmap.height],
    );
    await device.queue.onSubmittedWorkDone();
  } catch (error) {
    thrownError = error;
  } finally {
    finishExternalQueueWork?.();
  }
  const validationError = await device.popErrorScope();
  const memoryError = await device.popErrorScope();
  const uploadError = thrownError || validationError || memoryError;
  if (uploadError) {
    texture?.destroy();
    throw uploadError;
  }
  return texture;
}

async function loadSkyTexture(
  device,
  urls,
  requireUltra = false,
  beginExternalQueueWork = undefined,
) {
  const maxTextureDimension = finiteLimit(device.limits.maxTextureDimension2D, 4096);
  const url = selectedSkyUrl(urls, requireUltra);
  let bitmap;
  try {
    bitmap = await loadBitmap(url);
    if (bitmap.width > maxTextureDimension || bitmap.height > maxTextureDimension) {
      throw new Error(
        `${bitmap.width}×${bitmap.height} exceeds the device ${maxTextureDimension}px texture limit`,
      );
    }
    const texture = await uploadSkyTexture(
      device,
      bitmap,
      url,
      beginExternalQueueWork,
    );
    return { texture, width: bitmap.width, height: bitmap.height, url };
  } finally {
    bitmap?.close?.();
  }
}

function adapterLabel(adapter) {
  let info = {};
  try {
    info = adapter.info || {};
  } catch (error) {
    console.info("WebGPU adapter details are privacy-restricted by the browser.", error);
  }
  const pieces = [info.vendor, info.architecture, info.device, info.description]
    .filter(Boolean)
    .map((item) => String(item).trim())
    .filter((item, index, all) => all.indexOf(item) === index);

  if (pieces.length) {
    return pieces.join(" · ");
  }

  return isApplePlatform() ? "Apple GPU · Metal" : "High-performance GPU";
}

async function requestCompatibleAdapter() {
  let highPerformanceError;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter) {
      return adapter;
    }
  } catch (error) {
    highPerformanceError = error;
  }

  console.info(
    "A high-performance WebGPU adapter was not available; retrying with the browser default.",
    highPerformanceError,
  );
  return navigator.gpu.requestAdapter();
}

async function requestCompatibleDevice(adapter) {
  const adapterTextureLimit = finiteLimit(adapter.limits.maxTextureDimension2D, 4096);
  if (adapterTextureLimit < ULTRA_SKY_DIMENSION) {
    return {
      device: await adapter.requestDevice(),
      requestedUltraLimit: false,
      limitFallbackReason: "adapter-limit",
    };
  }

  try {
    return {
      device: await adapter.requestDevice({
        requiredLimits: { maxTextureDimension2D: ULTRA_SKY_DIMENSION },
      }),
      requestedUltraLimit: true,
      limitFallbackReason: "",
    };
  } catch (error) {
    // Some browser/Metal combinations advertise the native 16K limit but do
    // not allow a page to raise the device's conservative default limit.
    console.info("Unable to request the 16K WebGPU texture limit; using the default device limits.", error);
    return {
      device: await adapter.requestDevice(),
      requestedUltraLimit: false,
      limitFallbackReason: error instanceof Error ? error.message : String(error),
    };
  }
}

export class WebGPURenderer {
  static async create(canvas, skyUrl, options = undefined) {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available");
    }

    const adapter = await requestCompatibleAdapter();
    if (!adapter) {
      throw new Error("No WebGPU adapter was returned");
    }

    // WebGPU devices otherwise expose only the conservative default limit
    // even when Metal reports native 16K textures.  Request the real limit so
    // the scientific Gaia sky is not silently downgraded to the 6K fallback.
    const negotiation = await requestCompatibleDevice(adapter);
    const { device } = negotiation;
    const context = canvas.getContext("webgpu");
    if (!context) {
      throw new Error("Unable to create a WebGPU canvas context");
    }

    const instance = new WebGPURenderer(
      canvas,
      context,
      adapter,
      device,
      negotiation,
      options,
    );
    try {
      await instance.init(skyUrl);
      return instance;
    } catch (error) {
      instance.dispose();
      throw error;
    }
  }

  constructor(canvas, context, adapter, device, negotiation, options = undefined) {
    this.canvas = canvas;
    this.context = context;
    this.adapter = adapter;
    this.device = device;
    this.preferredFormat = navigator.gpu.getPreferredCanvasFormat();
    this.format = this.preferredFormat;
    this.backend = isApplePlatform() ? "WebGPU · Metal" : "WebGPU · GPU";
    this.gpu = adapterLabel(adapter);
    this.requestedUltraLimit = negotiation.requestedUltraLimit;
    this.limitFallbackReason = negotiation.limitFallbackReason;
    this.maxRenderDimension = finiteLimit(device.limits.maxTextureDimension2D, 4096);
    this.outputHDR = false;
    this.displayP3 = false;
    this.hdrPeak = 1;
    this.outputDescription = "sRGB 标准动态范围";
    this.skyDetail = "银河背景待载入";
    this.skyRadianceScale = 0.55;
    this.shaderBundle = shaderBundleFrom(options);
    this.progressiveAccumulation = progressiveAccumulationFrom(
      options,
      this.shaderBundle,
    );
    this.uniformData = new Float32Array(uniformFloatCount(this.shaderBundle));
    this.width = 1;
    this.height = 1;
    this.traceTexture = null;
    this.traceView = null;
    this.tracePipelines = null;
    this.traceBindGroups = null;
    this.traceDefaultSpecialization = "default";
    this.postBindGroup = null;
    this.accumulationBuffer = null;
    this.accumulationData = new Float32Array(4);
    this.accumulationTextures = [];
    this.accumulationViews = [];
    this.accumulationBindGroups = [];
    this.progressivePostBindGroups = [];
    this.accumulationReadIndex = 0;
    this.progressiveHistoryValid = false;
    this.progressiveHistorySignature = null;
    this.progressiveHistoryEpoch = null;
    this.submissionGate = new WebGPUFrameSubmissionGate();
    this.submissionGate.onFailure = (error) => {
      if (!this.disposed && !this.lost) {
        this.pendingRuntimeError = error;
        this.onError?.(error);
      }
    };
    this.disposed = false;
    this.lost = false;
    this.deviceLossInfo = null;
    this.pendingRuntimeError = null;
    this.outputFallbackReason = "";
    this.resizeWasClamped = false;
    this.handleUncapturedError = (event) => {
      console.error("Uncaptured WebGPU validation error", event.error);
      if (!this.disposed && !this.lost) {
        const error = event.error || new Error("Uncaptured WebGPU validation error");
        this.pendingRuntimeError = error;
        try {
          this.onError?.(error);
        } catch (callbackError) {
          console.error("WebGPU validation-error callback threw", callbackError);
        }
      }
    };
    device.addEventListener?.("uncapturederror", this.handleUncapturedError);
  }

  get hdrMode() {
    if (this.outputHDR) {
      return matchMedia("(dynamic-range: high)").matches
        ? "HDR · P3 · FP16"
        : "P3 扩展 · 屏幕 SDR";
    }
    return this.displayP3 ? "Display‑P3 · SDR" : "sRGB · SDR";
  }

  get readyForFrame() {
    return (
      !this.lost
      && !this.disposed
      && this.submissionGate.readyForFrame
    );
  }

  canSubmitFrame() {
    return this.readyForFrame;
  }

  consumeCompletedFrameTimeMs() {
    return this.submissionGate.consumeCompletedFrameTimeMs();
  }

  beginResourceQueueWork() {
    return this.submissionGate.beginExternalQueueWork();
  }

  get lastQueueCompletionAtMs() {
    return this.submissionGate.lastQueueCompletionAtMs;
  }

  get lastCompletedFrameTimeMs() {
    return this.submissionGate.lastCompletedFrameTimeMs;
  }

  configureOutput() {
    const common = {
      device: this.device,
      alphaMode: "opaque",
    };
    const hdrDisabled = new URLSearchParams(location.search).get("hdr") === "0";
    this.outputHDR = false;
    this.displayP3 = false;
    this.hdrPeak = 1;
    this.outputFallbackReason = hdrDisabled ? "disabled-by-query" : "";

    if (!hdrDisabled) {
      try {
        this.context.configure({
          ...common,
          format: HDR_FORMAT,
          colorSpace: "display-p3",
          toneMapping: { mode: "extended" },
        });
        const applied = this.context.getConfiguration?.();
        if (!applied || (
          applied.format !== HDR_FORMAT
          || applied.colorSpace !== "display-p3"
          || applied.toneMapping?.mode !== "extended"
        )) {
          throw new Error("Browser did not retain the requested extended HDR canvas configuration");
        }
        this.format = HDR_FORMAT;
        this.outputHDR = true;
        this.displayP3 = true;
        // Relative to SDR diffuse white.  The WebGPU canvas compositor maps
        // values above 1.0 into the active macOS display's HDR headroom.
        this.hdrPeak = 4;
        this.outputDescription = "16 位浮点 Display‑P3 扩展 HDR（高光最高 4× SDR 白）";
        return;
      } catch (error) {
        console.info("Extended WebGPU HDR unavailable; trying wide-gamut SDR.", error);
        this.outputFallbackReason = error instanceof Error ? error.message : String(error);
        this.context.unconfigure?.();
      }
    }

    try {
      this.format = this.preferredFormat;
      this.context.configure({
        ...common,
        format: this.format,
        colorSpace: "display-p3",
      });
      const applied = this.context.getConfiguration?.();
      if (!applied || applied.format !== this.format || applied.colorSpace !== "display-p3") {
        throw new Error("Browser did not retain Display-P3 output");
      }
      this.displayP3 = true;
      this.outputDescription = "Display‑P3 标准动态范围";
      return;
    } catch (error) {
      console.info("Display-P3 canvas unavailable; using sRGB SDR.", error);
      this.context.unconfigure?.();
    }

    this.format = this.preferredFormat;
    try {
      this.context.configure({
        ...common,
        format: this.format,
        colorSpace: "srgb",
      });
    } catch (error) {
      // Baseline WebGPU implementations may predate colorSpace.  The preferred
      // format without optional canvas members is the final compatibility path.
      console.info("Explicit sRGB canvas configuration unavailable; using baseline WebGPU output.", error);
      this.context.unconfigure?.();
      this.context.configure({ ...common, format: this.format });
    }
    this.outputDescription = "sRGB 标准动态范围";
  }

  reportCapabilities() {
    let features = [];
    try {
      features = [...this.device.features].sort();
    } catch (error) {
      console.info("WebGPU feature enumeration is unavailable.", error);
    }
    this.capabilities = Object.freeze({
      api: "webgpu",
      backend: this.backend,
      adapter: this.gpu,
      adapterMaxTextureDimension2D: finiteLimit(this.adapter.limits.maxTextureDimension2D, 0),
      deviceMaxTextureDimension2D: this.maxRenderDimension,
      requestedUltraTextureLimit: this.requestedUltraLimit,
      limitFallbackReason: this.limitFallbackReason,
      features: Object.freeze(features),
      canvasFormat: this.format,
      canvasColorSpace: this.displayP3 ? "display-p3" : "srgb",
      canvasToneMapping: this.outputHDR ? "extended" : "standard",
      screenDynamicRange: matchMedia("(dynamic-range: high)").matches ? "high" : "standard",
      skyTexture: this.skyTexture
        ? `${this.skyTextureWidth}×${this.skyTextureHeight}`
        : "unavailable",
      skyUrl: this.skyUrl || "",
      outputFallbackReason: this.outputFallbackReason,
      progressiveAccumulation: this.progressiveAccumulation?.mode || "disabled",
      maxFramesInFlight: 1,
    });
    console.info("Black-hole renderer capabilities", this.capabilities);
  }

  async init(skyUrl) {
    const { device } = this;
    this.configureOutput();

    this.uniformBuffer = device.createBuffer({
      label: this.shaderBundle.labels?.uniforms || "Schwarzschild frame uniforms",
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.skySampler = device.createSampler({
      label: "Linear celestial sampler",
      addressModeU: "repeat",
      addressModeV: "clamp-to-edge",
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
    });
    this.postSampler = device.createSampler({
      label: "Clamped HDR post-process sampler",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
      magFilter: "linear",
      minFilter: "linear",
    });

    const skyMode = new URLSearchParams(location.search).get("sky");
    const blockForUltra = skyMode === "ultra";
    const {
      texture,
      width: skyTextureWidth,
      height: skyTextureHeight,
      url: selectedSkyUrl,
    } = await loadSkyTexture(
      device,
      skyUrl,
      blockForUltra,
      () => this.beginResourceQueueWork(),
    );
    this.skyRadianceScale = /gaia-edr3/i.test(selectedSkyUrl) ? 0.16 : 0.55;
    this.skyTexture = texture;
    this.skyTextureWidth = skyTextureWidth;
    this.skyTextureHeight = skyTextureHeight;
    this.skyDetail = `${skyTextureWidth}×${skyTextureHeight} 原始全景 · 解析恒星层`;
    this.skyUrl = selectedSkyUrl;

    const vertexModule = device.createShaderModule({
      label: "Fullscreen triangle vertex shader",
      code: fullscreenVertexWGSL,
    });
    const traceModule = device.createShaderModule({
      label: this.shaderBundle.labels?.trace || "Schwarzschild null-geodesic tracer",
      code: this.shaderBundle.wgsl?.trace || traceFragmentWGSL,
    });
    const postModule = device.createShaderModule({
      label: "HDR telescope post-process",
      code: postFragmentWGSL,
    });
    const accumulationModule = this.progressiveAccumulation
      ? device.createShaderModule({
        label: "Strong-field linear HDR progressive accumulation",
        code: progressiveAccumulationFragmentWGSL,
      })
      : null;

    const compilation = await Promise.all([
      traceModule.getCompilationInfo(),
      postModule.getCompilationInfo(),
      ...(accumulationModule
        ? [accumulationModule.getCompilationInfo()]
        : []),
    ]);
    const errors = compilation.flatMap((info) => info.messages.filter((message) => message.type === "error"));
    if (errors.length) {
      throw new Error(errors.map((error) => error.message).join("\n"));
    }

    const traceSpecializations = traceSpecializationsFrom(this.shaderBundle);
    const tracePipelineEntries = await Promise.all(
      traceSpecializations.map(async ({ id, constants }) => [
        id,
        await device.createRenderPipelineAsync({
          label: id === "default"
            ? "Relativistic trace pipeline"
            : `Relativistic trace pipeline · ${id}`,
          layout: "auto",
          vertex: { module: vertexModule, entryPoint: "vsMain" },
          fragment: {
            module: traceModule,
            entryPoint: "fsMain",
            ...(constants ? { constants } : {}),
            targets: [{ format: HDR_FORMAT }],
          },
          primitive: { topology: "triangle-list" },
        }),
      ]),
    );
    this.tracePipelines = Object.fromEntries(tracePipelineEntries);
    this.traceDefaultSpecialization = traceSpecializations[0].id;
    this.tracePipeline = this.tracePipelines[this.traceDefaultSpecialization];

    this.postPipeline = await device.createRenderPipelineAsync({
      label: "Telescope display pipeline",
      layout: "auto",
      vertex: { module: vertexModule, entryPoint: "vsMain" },
      fragment: {
        module: postModule,
        entryPoint: "fsMain",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });
    if (accumulationModule) {
      this.accumulationPipeline = await device.createRenderPipelineAsync({
        label: "Strong-field progressive accumulation pipeline",
        layout: "auto",
        vertex: { module: vertexModule, entryPoint: "vsMain" },
        fragment: {
          module: accumulationModule,
          entryPoint: "fsMain",
          targets: [{ format: HDR_FORMAT }],
        },
        primitive: { topology: "triangle-list" },
      });
      this.accumulationBuffer = device.createBuffer({
        label: "Strong-field progressive accumulation parameters",
        size: this.accumulationData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }

    this.sceneResourceState = createSceneResources(this.shaderBundle, device);
    this.traceBindGroups = this.createTraceBindGroups(this.skyTexture);
    this.traceBindGroup = this.traceBindGroups[this.traceDefaultSpecialization];

    // GPUDevice.lost cannot be cancelled. Keep only this detachable lifecycle
    // cell in the promise closure so dispose() releases the renderer and all
    // user callbacks even if the browser never resolves the device promise.
    const deviceLossLifecycle = { renderer: this };
    this.deviceLossLifecycle = deviceLossLifecycle;
    const handleDeviceLoss = (info) => {
      const target = deviceLossLifecycle.renderer;
      deviceLossLifecycle.renderer = null;
      if (!target || target.disposed || target.lost) {
        return;
      }
      target.lost = true;
      target.deviceLossInfo = info;
      target.submissionGate.close();
      try {
        target.onLost?.(info);
      } catch (error) {
        console.error("WebGPU device-loss callback threw", error);
      }
    };
    Promise.resolve(device.lost).then(
      handleDeviceLoss,
      (error) => handleDeviceLoss({
        reason: "unknown",
        message: error instanceof Error ? error.message : String(error),
      }),
    );
    this.reportCapabilities();
  }

  createTraceBindGroup(texture, pipeline = this.tracePipeline, label = "Trace resources") {
    return this.device.createBindGroup({
      label,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: texture.createView() },
        { binding: 2, resource: this.skySampler },
        ...(this.sceneResourceState?.entries || []),
      ],
    });
  }

  createTraceBindGroups(texture) {
    const pipelines = this.tracePipelines || {
      [this.traceDefaultSpecialization]: this.tracePipeline,
    };
    return Object.fromEntries(Object.entries(pipelines).map(([id, pipeline]) => [
      id,
      this.createTraceBindGroup(texture, pipeline, `Trace resources · ${id}`),
    ]));
  }

  traceResourcesForFrame(frame) {
    const selector = this.shaderBundle?.wgsl?.selectTraceSpecialization;
    const selected = typeof selector === "function"
      ? String(selector(frame))
      : this.traceDefaultSpecialization;
    const pipeline = this.tracePipelines?.[selected];
    const bindGroup = this.traceBindGroups?.[selected];
    if (pipeline && bindGroup) {
      return { pipeline, bindGroup };
    }
    return { pipeline: this.tracePipeline, bindGroup: this.traceBindGroup };
  }

  resize(width, height) {
    const requestedWidth = Math.max(1, Math.floor(Number.isFinite(width) ? width : 1));
    const requestedHeight = Math.max(1, Math.floor(Number.isFinite(height) ? height : 1));
    const limitScale = Math.min(
      1,
      this.maxRenderDimension / requestedWidth,
      this.maxRenderDimension / requestedHeight,
    );
    const nextWidth = Math.max(1, Math.floor(requestedWidth * limitScale));
    const nextHeight = Math.max(1, Math.floor(requestedHeight * limitScale));
    if (limitScale < 1 && !this.resizeWasClamped) {
      this.resizeWasClamped = true;
      console.info(
        `Render target ${requestedWidth}×${requestedHeight} exceeds the WebGPU limit; `
        + `using ${nextWidth}×${nextHeight}.`,
      );
    }
    if (nextWidth === this.width && nextHeight === this.height && this.traceTexture) {
      return;
    }

    this.width = nextWidth;
    this.height = nextHeight;
    this.canvas.width = nextWidth;
    this.canvas.height = nextHeight;

    this.traceTexture?.destroy();
    this.traceTexture = this.device.createTexture({
      label: "Linear HDR ray-trace target",
      size: [nextWidth, nextHeight, 1],
      format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.traceView = this.traceTexture.createView();
    this.postBindGroup = this.createPostBindGroup(
      this.traceView,
      "Post-process resources",
    );
    this.recreateProgressiveTargets(nextWidth, nextHeight);
  }

  createPostBindGroup(textureView, label) {
    return this.device.createBindGroup({
      label,
      layout: this.postPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: textureView },
        { binding: 2, resource: this.postSampler },
      ],
    });
  }

  destroyProgressiveTargets() {
    for (const texture of this.accumulationTextures) {
      texture.destroy();
    }
    this.accumulationTextures = [];
    this.accumulationViews = [];
    this.accumulationBindGroups = [];
    this.progressivePostBindGroups = [];
  }

  recreateProgressiveTargets(width, height) {
    this.destroyProgressiveTargets();
    this.invalidateProgressiveHistory();
    if (!this.progressiveAccumulation) {
      return;
    }
    for (let index = 0; index < 2; index += 1) {
      const texture = this.device.createTexture({
        label: `Strong-field accumulated HDR history ${index}`,
        size: [width, height, 1],
        format: HDR_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.accumulationTextures.push(texture);
      this.accumulationViews.push(texture.createView());
    }
    this.accumulationBindGroups = this.accumulationViews.map(
      (view, index) => this.device.createBindGroup({
        label: `Strong-field accumulation read history ${index}`,
        layout: this.accumulationPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: view },
          { binding: 1, resource: this.traceView },
          { binding: 2, resource: { buffer: this.accumulationBuffer } },
        ],
      }),
    );
    this.progressivePostBindGroups = this.accumulationViews.map(
      (view, index) => this.createPostBindGroup(
        view,
        `Strong-field accumulated post-process ${index}`,
      ),
    );
    this.accumulationReadIndex = 0;
  }

  invalidateProgressiveHistory() {
    this.progressiveHistoryValid = false;
    this.progressiveHistorySignature = null;
    this.progressiveHistoryEpoch = null;
  }

  writeUniforms(frame) {
    const data = this.uniformData;
    data.fill(0);
    data[0] = this.width;
    data[1] = this.height;
    data[2] = frame.time;
    data[3] = frame.massSolar;
    data[4] = frame.accretion;
    data[5] = frame.exposure;
    data[6] = frame.mode;
    data[7] = frame.steps;
    data.set(frame.cameraPos, 8);
    data[11] = frame.cameraRadius;
    data.set(frame.forward, 12);
    data[15] = frame.fov;
    data.set(frame.right, 16);
    data[19] = frame.skyRotation;
    data.set(frame.up, 20);
    data[23] = frame.diskOuterRadius;
    data[24] = frame.renderScale;
    data[25] = frame.bloom;
    data[26] = frame.motion;
    data[27] = frame.frame;
    data.set(frame.observerVelocity, 28);
    data[31] = frame.observerBeta;
    data[32] = this.outputHDR ? 1 : 0;
    data[33] = this.displayP3 ? 1 : 0;
    data[34] = this.hdrPeak;
    data[35] = this.skyRadianceScale;
    this.shaderBundle.uniforms?.writeWebGPUExtras?.(
      data.subarray(UNIFORM_FLOATS - 4),
      frame,
    );
    this.device.queue.writeBuffer(this.uniformBuffer, 0, data);
  }

  prepareProgressiveFrame(frame) {
    if (!this.progressiveAccumulation) {
      return {
        frame,
        state: null,
        signature: null,
        bypassAccumulation: false,
      };
    }
    const requested = progressiveFrameState(frame);
    const signature = progressiveHistorySignature(frame);
    const dynamicFrame = isDynamicProgressiveFrame(frame);
    const forcedReset = (
      requested.historyReset
      || !this.progressiveHistoryValid
      || signature !== this.progressiveHistorySignature
      || requested.historyEpoch !== this.progressiveHistoryEpoch
      || dynamicFrame
    );
    const state = forcedReset
      ? {
        accumulationIndex: 0,
        accumulationWeight: 1,
        historyEpoch: requested.historyEpoch,
        historyReset: true,
      }
      : requested;
    return {
      frame: {
        ...frame,
        // The trace shader uses the shared frame slot as its deterministic
        // sub-pixel sample index. Dynamic/reset frames always restart at zero.
        frame: state.accumulationIndex,
        strongFieldQuality: {
          ...(frame.strongFieldQuality || {}),
          ...state,
        },
      },
      state,
      signature,
      // A dynamic sample is already forced to sample zero and cannot seed
      // reusable history. The reset accumulation shader would only copy the
      // FP16 trace target into another same-sized FP16 target before the same
      // post pipeline reads it, so post can read traceView directly instead.
      bypassAccumulation: dynamicFrame,
    };
  }

  render(frame) {
    const progressiveTargetsReady = (
      !this.progressiveAccumulation
      || (
        this.accumulationViews.length === 2
        && this.progressivePostBindGroups.length === 2
      )
    );
    if (
      !this.canSubmitFrame()
      || !this.traceView
      || !this.postBindGroup
      || !progressiveTargetsReady
    ) {
      return false;
    }

    const progressive = this.prepareProgressiveFrame(frame);
    this.writeUniforms(progressive.frame);
    const traceResources = this.traceResourcesForFrame(progressive.frame);
    const encoder = this.device.createCommandEncoder({ label: "Black-hole frame" });
    const tracePass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.traceView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    tracePass.setPipeline(traceResources.pipeline);
    tracePass.setBindGroup(0, traceResources.bindGroup);
    tracePass.draw(3);
    tracePass.end();

    let postBindGroup = this.postBindGroup;
    if (progressive.state && !progressive.bypassAccumulation) {
      const writeIndex = 1 - this.accumulationReadIndex;
      this.accumulationData[0] = progressive.state.accumulationWeight;
      this.accumulationData[1] = progressive.state.historyReset ? 1 : 0;
      this.accumulationData[2] = progressive.state.accumulationIndex;
      this.accumulationData[3] = progressive.state.historyEpoch;
      this.device.queue.writeBuffer(
        this.accumulationBuffer,
        0,
        this.accumulationData,
      );
      const accumulationPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.accumulationViews[writeIndex],
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      accumulationPass.setPipeline(this.accumulationPipeline);
      accumulationPass.setBindGroup(
        0,
        this.accumulationBindGroups[this.accumulationReadIndex],
      );
      accumulationPass.draw(3);
      accumulationPass.end();
      postBindGroup = this.progressivePostBindGroups[writeIndex];
    }

    const postPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    postPass.setPipeline(this.postPipeline);
    postPass.setBindGroup(0, postBindGroup);
    postPass.draw(3);
    postPass.end();
    const submitted = this.submissionGate.submit(
      this.device.queue,
      [encoder.finish()],
    );
    if (!submitted) {
      return false;
    }
    if (progressive.state) {
      if (progressive.bypassAccumulation) {
        // The displayed trace is current, but it must never become temporal
        // history. The first static sample will take the regular reset path,
        // populate an accumulation target, and only then mark history valid.
        this.invalidateProgressiveHistory();
      } else {
        this.accumulationReadIndex = 1 - this.accumulationReadIndex;
        this.progressiveHistoryValid = true;
        this.progressiveHistorySignature = progressive.signature;
        this.progressiveHistoryEpoch = progressive.state.historyEpoch;
      }
    }
    return true;
  }

  dispose() {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    if (this.deviceLossLifecycle) {
      this.deviceLossLifecycle.renderer = null;
      this.deviceLossLifecycle = null;
    }
    this.device.removeEventListener?.(
      "uncapturederror",
      this.handleUncapturedError,
    );
    this.onLost = null;
    this.onError = null;
    this.deviceLossInfo = null;
    this.pendingRuntimeError = null;
    this.submissionGate.close();
    this.sceneResourceState?.dispose();
    this.sceneResourceState = null;
    this.destroyProgressiveTargets();
    this.traceTexture?.destroy();
    this.skyTexture?.destroy();
    this.accumulationBuffer?.destroy();
    this.uniformBuffer?.destroy();
    this.context.unconfigure?.();
    this.device.destroy?.();
  }
}
