import {
  fullscreenVertexWGSL,
  traceFragmentWGSL,
  postFragmentWGSL,
} from "./shaders.js";

const HDR_FORMAT = "rgba16float";
const UNIFORM_FLOATS = 40;
const ULTRA_SKY_DIMENSION = 16000;

function isApplePlatform() {
  const platform = navigator.userAgentData?.platform || navigator.platform || "";
  return /Mac|iPhone|iPad/i.test(platform);
}

function finiteLimit(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.floor(number) : fallback;
}

function skyCandidates(urls, maxTextureDimension, includeUltra = true) {
  const source = typeof urls === "string" ? { high: urls, fallback: urls } : urls;
  const candidates = includeUltra && maxTextureDimension >= ULTRA_SKY_DIMENSION
    ? [source.ultra, source.high, source.fallback]
    : maxTextureDimension >= 6000
      ? [source.high, source.fallback]
      : [source.fallback];
  return [...new Set(candidates.filter(Boolean))];
}

function scheduleBackgroundTask(callback) {
  if (typeof requestIdleCallback === "function") {
    requestIdleCallback(callback, { timeout: 2500 });
  } else {
    setTimeout(callback, 1200);
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
      URL.revokeObjectURL(objectUrl);
      throw new AggregateError(
        [bitmapError, imageError],
        `Unable to decode celestial panorama ${url}`,
      );
    }
  }
}

async function uploadSkyTexture(device, bitmap, url) {
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  let texture;
  let thrownError;
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
    device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture },
      [bitmap.width, bitmap.height],
    );
    await device.queue.onSubmittedWorkDone();
  } catch (error) {
    thrownError = error;
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

async function loadBestSkyTexture(device, urls, includeUltra = true) {
  const maxTextureDimension = finiteLimit(device.limits.maxTextureDimension2D, 4096);
  let lastError;
  for (const url of skyCandidates(urls, maxTextureDimension, includeUltra)) {
    let bitmap;
    try {
      bitmap = await loadBitmap(url);
      if (bitmap.width > maxTextureDimension || bitmap.height > maxTextureDimension) {
        throw new Error(
          `${bitmap.width}×${bitmap.height} exceeds the device ${maxTextureDimension}px texture limit`,
        );
      }
      const texture = await uploadSkyTexture(device, bitmap, url);
      return { texture, width: bitmap.width, height: bitmap.height, url };
    } catch (error) {
      lastError = error;
      console.info(`Sky candidate ${url} unavailable; trying the next size.`, error);
    } finally {
      bitmap?.close?.();
    }
  }
  throw lastError || new Error("No celestial panorama could be loaded");
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
  static async create(canvas, skyUrl) {
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

    const instance = new WebGPURenderer(canvas, context, adapter, device, negotiation);
    await instance.init(skyUrl);
    return instance;
  }

  constructor(canvas, context, adapter, device, negotiation) {
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
    this.uniformData = new Float32Array(UNIFORM_FLOATS);
    this.width = 1;
    this.height = 1;
    this.traceTexture = null;
    this.traceView = null;
    this.postBindGroup = null;
    this.lost = false;
    this.outputFallbackReason = "";
    this.resizeWasClamped = false;
    device.addEventListener?.("uncapturederror", (event) => {
      console.error("Uncaptured WebGPU validation error", event.error);
    });
  }

  get hdrMode() {
    if (this.outputHDR) {
      return matchMedia("(dynamic-range: high)").matches
        ? "HDR · P3 · FP16"
        : "P3 扩展 · 屏幕 SDR";
    }
    return this.displayP3 ? "Display‑P3 · SDR" : "sRGB · SDR";
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
      features,
      canvasFormat: this.format,
      canvasColorSpace: this.displayP3 ? "display-p3" : "srgb",
      canvasToneMapping: this.outputHDR ? "extended" : "standard",
      screenDynamicRange: matchMedia("(dynamic-range: high)").matches ? "high" : "standard",
      skyTexture: this.skyTexture
        ? `${this.skyTextureWidth}×${this.skyTextureHeight}`
        : "unavailable",
      skyUrl: this.skyUrl || "",
      outputFallbackReason: this.outputFallbackReason,
    });
    console.info("Black-hole renderer capabilities", this.capabilities);
  }

  async init(skyUrl) {
    const { device } = this;
    this.configureOutput();

    this.uniformBuffer = device.createBuffer({
      label: "Schwarzschild frame uniforms",
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
    } = await loadBestSkyTexture(device, skyUrl, blockForUltra);
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
      label: "Schwarzschild null-geodesic tracer",
      code: traceFragmentWGSL,
    });
    const postModule = device.createShaderModule({
      label: "HDR telescope post-process",
      code: postFragmentWGSL,
    });

    const compilation = await Promise.all([
      traceModule.getCompilationInfo(),
      postModule.getCompilationInfo(),
    ]);
    const errors = compilation.flatMap((info) => info.messages.filter((message) => message.type === "error"));
    if (errors.length) {
      throw new Error(errors.map((error) => error.message).join("\n"));
    }

    this.tracePipeline = await device.createRenderPipelineAsync({
      label: "Relativistic trace pipeline",
      layout: "auto",
      vertex: { module: vertexModule, entryPoint: "vsMain" },
      fragment: {
        module: traceModule,
        entryPoint: "fsMain",
        targets: [{ format: HDR_FORMAT }],
      },
      primitive: { topology: "triangle-list" },
    });

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

    this.traceBindGroup = this.createTraceBindGroup(this.skyTexture);

    device.lost.then((info) => {
      this.lost = true;
      this.onLost?.(info);
    });
    this.reportCapabilities();

    const source = typeof skyUrl === "string" ? {} : skyUrl;
    if (
      skyMode !== "high"
      && !blockForUltra
      && source.ultra
      && source.ultra !== selectedSkyUrl
      && this.maxRenderDimension >= ULTRA_SKY_DIMENSION
    ) {
      scheduleBackgroundTask(() => {
        void this.upgradeSkyTexture(source.ultra);
      });
    }
  }

  createTraceBindGroup(texture) {
    return this.device.createBindGroup({
      label: "Trace resources",
      layout: this.tracePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: texture.createView() },
        { binding: 2, resource: this.skySampler },
      ],
    });
  }

  async upgradeSkyTexture(url) {
    try {
      const next = await loadBestSkyTexture(this.device, url, false);
      if (this.lost) {
        next.texture.destroy();
        return;
      }
      const previousTexture = this.skyTexture;
      this.skyTexture = next.texture;
      this.skyTextureWidth = next.width;
      this.skyTextureHeight = next.height;
      this.skyUrl = next.url;
      this.skyRadianceScale = /gaia-edr3/i.test(next.url) ? 0.16 : 0.55;
      this.skyDetail = `${next.width}×${next.height} 原始全景 · 解析恒星层`;
      this.traceBindGroup = this.createTraceBindGroup(next.texture);
      previousTexture?.destroy();
      this.reportCapabilities();
      this.onSkyChanged?.();
    } catch (error) {
      console.info("The 16K sky upgrade was unavailable; keeping the responsive fallback.", error);
    }
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
    this.postBindGroup = this.device.createBindGroup({
      label: "Post-process resources",
      layout: this.postPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.traceView },
        { binding: 2, resource: this.postSampler },
      ],
    });
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
    this.device.queue.writeBuffer(this.uniformBuffer, 0, data);
  }

  render(frame) {
    if (this.lost || !this.traceView || !this.postBindGroup) {
      return;
    }

    this.writeUniforms(frame);
    const encoder = this.device.createCommandEncoder({ label: "Black-hole frame" });
    const tracePass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.traceView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    tracePass.setPipeline(this.tracePipeline);
    tracePass.setBindGroup(0, this.traceBindGroup);
    tracePass.draw(3);
    tracePass.end();

    const postPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    postPass.setPipeline(this.postPipeline);
    postPass.setBindGroup(0, this.postBindGroup);
    postPass.draw(3);
    postPass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  dispose() {
    this.traceTexture?.destroy();
    this.skyTexture?.destroy();
    this.uniformBuffer?.destroy();
  }
}
