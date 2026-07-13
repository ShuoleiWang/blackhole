import {
  fullscreenVertexWGSL,
  traceFragmentWGSL,
  postFragmentWGSL,
} from "./shaders.js";

const HDR_FORMAT = "rgba16float";
const UNIFORM_FLOATS = 40;

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

async function loadBestSkyBitmap(urls, maxTextureDimension) {
  const source = typeof urls === "string" ? { high: urls, fallback: urls } : urls;
  const candidates = maxTextureDimension >= 16000
    ? [source.ultra, source.high, source.fallback]
    : maxTextureDimension >= 6000
      ? [source.high, source.fallback]
      : [source.fallback];
  let lastError;
  for (const url of [...new Set(candidates.filter(Boolean))]) {
    try {
      return { bitmap: await loadBitmap(url), url };
    } catch (error) {
      lastError = error;
      console.info(`Sky candidate ${url} unavailable; trying the next size.`, error);
    }
  }
  throw lastError || new Error("No celestial panorama could be loaded");
}

function adapterLabel(adapter) {
  const info = adapter.info || {};
  const pieces = [info.vendor, info.architecture, info.device, info.description]
    .filter(Boolean)
    .map((item) => String(item).trim())
    .filter((item, index, all) => all.indexOf(item) === index);

  if (pieces.length) {
    return pieces.join(" · ");
  }

  return /Mac|iPhone|iPad/.test(navigator.platform) ? "Apple GPU · Metal" : "High-performance GPU";
}

export class WebGPURenderer {
  static async create(canvas, skyUrl) {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available");
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      throw new Error("No high-performance WebGPU adapter was returned");
    }

    const supportedTextureLimit = adapter.limits.maxTextureDimension2D;
    const requiredLimits = supportedTextureLimit >= 16000
      ? { maxTextureDimension2D: 16000 }
      : {};
    // WebGPU devices otherwise expose only the conservative default limit
    // even when Metal reports native 16K textures.  Request the real limit so
    // the scientific Gaia sky is not silently downgraded to the 6K fallback.
    const device = await adapter.requestDevice({ requiredLimits });
    const context = canvas.getContext("webgpu");
    if (!context) {
      throw new Error("Unable to create a WebGPU canvas context");
    }

    const instance = new WebGPURenderer(canvas, context, adapter, device);
    await instance.init(skyUrl);
    return instance;
  }

  constructor(canvas, context, adapter, device) {
    this.canvas = canvas;
    this.context = context;
    this.adapter = adapter;
    this.device = device;
    this.preferredFormat = navigator.gpu.getPreferredCanvasFormat();
    this.format = this.preferredFormat;
    this.backend = "WebGPU · Metal";
    this.gpu = adapterLabel(adapter);
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

    if (!hdrDisabled) {
      try {
        this.format = HDR_FORMAT;
        this.context.configure({
          ...common,
          format: this.format,
          colorSpace: "display-p3",
          toneMapping: { mode: "extended" },
        });
        const applied = this.context.getConfiguration?.();
        if (applied && (
          applied.format !== HDR_FORMAT
          || applied.colorSpace !== "display-p3"
          || applied.toneMapping?.mode !== "extended"
        )) {
          throw new Error("Browser did not retain the requested extended HDR canvas configuration");
        }
        this.outputHDR = true;
        this.displayP3 = true;
        // Relative to SDR diffuse white.  The WebGPU canvas compositor maps
        // values above 1.0 into the active macOS display's HDR headroom.
        this.hdrPeak = 4;
        this.outputDescription = "16 位浮点 Display‑P3 扩展 HDR（高光最高 4× SDR 白）";
        return;
      } catch (error) {
        console.info("Extended WebGPU HDR unavailable; trying wide-gamut SDR.", error);
        this.context.unconfigure?.();
      }
    }

    try {
      this.format = this.preferredFormat;
      this.context.configure({
        ...common,
        format: this.format,
        colorSpace: "display-p3",
        toneMapping: { mode: "standard" },
      });
      const applied = this.context.getConfiguration?.();
      if (applied && applied.colorSpace !== "display-p3") {
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
    this.context.configure({
      ...common,
      format: this.format,
      colorSpace: "srgb",
    });
    this.outputDescription = "sRGB 标准动态范围";
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

    const { bitmap, url: selectedSkyUrl } = await loadBestSkyBitmap(
      skyUrl,
      device.limits.maxTextureDimension2D,
    );
    this.skyRadianceScale = /gaia-edr3/i.test(selectedSkyUrl) ? 0.16 : 0.55;
    this.skyDetail = `${bitmap.width}×${bitmap.height} 原始全景 · 解析恒星层`;
    this.skyTexture = device.createTexture({
      label: "Authored deep-field celestial sphere",
      size: [bitmap.width, bitmap.height, 1],
      format: "rgba8unorm-srgb",
      mipLevelCount: 1,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture: this.skyTexture },
      [bitmap.width, bitmap.height],
    );
    bitmap.close?.();
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

    this.traceBindGroup = device.createBindGroup({
      label: "Trace resources",
      layout: this.tracePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.skyTexture.createView() },
        { binding: 2, resource: this.skySampler },
      ],
    });

    device.lost.then((info) => {
      this.lost = true;
      this.onLost?.(info);
    });
  }

  resize(width, height) {
    const nextWidth = Math.max(1, Math.floor(width));
    const nextHeight = Math.max(1, Math.floor(height));
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
