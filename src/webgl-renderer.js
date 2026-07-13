import * as THREE from "../vendor/three.module.js";
import {
  fullscreenVertexGLSL,
  traceFragmentGLSL,
  postFragmentGLSL,
} from "./shaders.js";

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

function loadTexture(url) {
  return new Promise((resolve, reject) => {
    new THREE.TextureLoader().load(url, resolve, undefined, reject);
  });
}

function gpuLabel(gl) {
  const extension = gl.getExtension("WEBGL_debug_renderer_info");
  if (!extension) {
    return isApplePlatform() ? "Apple GPU" : "Hardware GPU";
  }
  return gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) || "Hardware GPU";
}

function clearErrors(gl) {
  for (let index = 0; index < 16 && gl.getError() !== gl.NO_ERROR; index += 1) {
    // Drain stale errors before a capability probe or an eager texture upload.
  }
}

function probeHalfFloatRenderTarget(gl) {
  const floatExtension = gl.getExtension("EXT_color_buffer_float");
  const halfFloatExtension = gl.getExtension("EXT_color_buffer_half_float");
  if (!floatExtension && !halfFloatExtension) {
    return { supported: false, extension: "none" };
  }

  const texture = gl.createTexture();
  const framebuffer = gl.createFramebuffer();
  if (!texture || !framebuffer) {
    gl.deleteTexture(texture);
    gl.deleteFramebuffer(framebuffer);
    return { supported: false, extension: "allocation-failed" };
  }

  const previousTexture = gl.getParameter(gl.TEXTURE_BINDING_2D);
  const previousFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);
  clearErrors(gl);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, 2, 2, 0, gl.RGBA, gl.HALF_FLOAT, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    texture,
    0,
  );
  const complete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  const error = gl.getError();
  gl.bindFramebuffer(gl.FRAMEBUFFER, previousFramebuffer);
  gl.bindTexture(gl.TEXTURE_2D, previousTexture);
  gl.deleteFramebuffer(framebuffer);
  gl.deleteTexture(texture);

  return {
    supported: complete && error === gl.NO_ERROR,
    extension: floatExtension ? "EXT_color_buffer_float" : "EXT_color_buffer_half_float",
  };
}

function inspectCapabilities(gl) {
  const viewport = gl.getParameter(gl.MAX_VIEWPORT_DIMS);
  const maxTextureSize = finiteLimit(gl.getParameter(gl.MAX_TEXTURE_SIZE), 4096);
  const maxRenderbufferSize = finiteLimit(gl.getParameter(gl.MAX_RENDERBUFFER_SIZE), 4096);
  const maxViewportWidth = finiteLimit(viewport?.[0], maxRenderbufferSize);
  const maxViewportHeight = finiteLimit(viewport?.[1], maxRenderbufferSize);
  const halfFloat = probeHalfFloatRenderTarget(gl);
  const highFloat = gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);
  return {
    maxTextureSize,
    maxRenderbufferSize,
    maxViewportWidth,
    maxViewportHeight,
    maxRenderDimension: Math.min(
      maxTextureSize,
      maxRenderbufferSize,
      maxViewportWidth,
      maxViewportHeight,
    ),
    halfFloatRenderTarget: halfFloat.supported,
    halfFloatExtension: halfFloat.extension,
    fragmentHighpBits: highFloat?.precision || 0,
  };
}

function configureSkyTexture(texture) {
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  // The shader uses astronomical equirectangular coordinates with v=0 at
  // the north/top of the source image, matching WebGPU's copy convention.
  texture.flipY = false;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
}

export class WebGLRenderer {
  static async create(canvas, skyUrl) {
    const context = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: "high-performance",
    });
    if (!context) {
      throw new Error("WebGL2 is not available");
    }

    const capabilities = inspectCapabilities(context);
    if (!capabilities.fragmentHighpBits) {
      throw new Error("WebGL2 fragment high precision is unavailable");
    }
    const instance = new WebGLRenderer(canvas, context, capabilities);
    await instance.init(skyUrl);
    return instance;
  }

  constructor(canvas, context, capabilities) {
    this.canvas = canvas;
    this.context = context;
    this.backend = isApplePlatform()
      ? "WebGL2 · Metal fallback"
      : "WebGL2 · GPU";
    this.gpu = gpuLabel(context);
    this.glCapabilities = capabilities;
    this.maxRenderDimension = capabilities.maxRenderDimension;
    this.hdrMode = "sRGB · SDR";
    this.outputHDR = false;
    this.displayP3 = false;
    this.outputDescription = "WebGL2 回退路径使用 sRGB 标准动态范围";
    this.skyDetail = "银河背景待载入";
    this.skyRadianceScale = 0.55;
    this.width = 1;
    this.height = 1;
    this.resizeWasClamped = false;
  }

  async init(skyUrl) {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      context: this.context,
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: "high-performance",
    });
    this.renderer.autoClear = true;
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.setPixelRatio(1);
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;

    const skyMode = new URLSearchParams(location.search).get("sky");
    const blockForUltra = skyMode === "ultra";
    let lastSkyError;
    for (const url of skyCandidates(
      skyUrl,
      this.glCapabilities.maxTextureSize,
      blockForUltra,
    )) {
      let texture;
      try {
        texture = await loadTexture(url);
        const image = texture.image;
        const width = image.naturalWidth || image.width;
        const height = image.naturalHeight || image.height;
        if (
          width > this.glCapabilities.maxTextureSize
          || height > this.glCapabilities.maxTextureSize
        ) {
          throw new Error(
            `${width}×${height} exceeds the WebGL ${this.glCapabilities.maxTextureSize}px texture limit`,
          );
        }
        configureSkyTexture(texture);
        this.uploadSkyTexture(texture, url);
        this.skyTexture = texture;
        this.skyUrl = url;
        break;
      } catch (error) {
        lastSkyError = error;
        texture?.dispose();
        console.info(`Sky candidate ${url} unavailable; trying the next size.`, error);
      }
    }
    if (!this.skyTexture) {
      throw lastSkyError || new Error("No celestial panorama could be loaded");
    }
    this.skyRadianceScale = /gaia-edr3/i.test(this.skyUrl) ? 0.16 : 0.55;
    const image = this.skyTexture.image;
    this.skyDetail = `${image.naturalWidth || image.width}×${image.naturalHeight || image.height} 原始全景 · 解析恒星层`;

    this.traceUniforms = {
      tSky: { value: this.skyTexture },
      uResolution: { value: new THREE.Vector2(1, 1) },
      uTime: { value: 0 },
      uMassSolar: { value: 6.5e9 },
      uAccretion: { value: 0.02 },
      uExposure: { value: 1 },
      uMode: { value: 1 },
      uSteps: { value: 256 },
      uCameraPos: { value: new THREE.Vector3() },
      uCameraRadius: { value: 40 },
      uForward: { value: new THREE.Vector3() },
      uFov: { value: Math.PI / 4 },
      uRight: { value: new THREE.Vector3() },
      uSkyRotation: { value: 0 },
      uUp: { value: new THREE.Vector3() },
      uDiskOuterRadius: { value: 28 },
      uRenderScale: { value: 1 },
      uBloom: { value: 1 },
      uMotion: { value: 1 },
      uFrame: { value: 0 },
      uObserverVelocity: { value: new THREE.Vector3() },
      uObserverBeta: { value: 0 },
      uSkyRadianceScale: { value: this.skyRadianceScale },
    };

    this.postUniforms = {
      tScene: { value: null },
      uResolution: this.traceUniforms.uResolution,
      uTime: this.traceUniforms.uTime,
      uExposure: this.traceUniforms.uExposure,
      uMode: this.traceUniforms.uMode,
      uBloom: this.traceUniforms.uBloom,
      uFrame: this.traceUniforms.uFrame,
    };

    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.traceScene = new THREE.Scene();
    this.postScene = new THREE.Scene();

    this.traceMaterial = new THREE.ShaderMaterial({
      name: "Schwarzschild null-geodesic tracer",
      vertexShader: fullscreenVertexGLSL,
      fragmentShader: traceFragmentGLSL,
      uniforms: this.traceUniforms,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });
    this.postMaterial = new THREE.ShaderMaterial({
      name: "Deep-field telescope post-process",
      vertexShader: fullscreenVertexGLSL,
      fragmentShader: postFragmentGLSL,
      uniforms: this.postUniforms,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });

    const geometry = new THREE.PlaneGeometry(2, 2);
    this.traceScene.add(new THREE.Mesh(geometry, this.traceMaterial));
    this.postScene.add(new THREE.Mesh(geometry, this.postMaterial));

    this.targetType = this.glCapabilities.halfFloatRenderTarget
      ? THREE.HalfFloatType
      : THREE.UnsignedByteType;
    const intermediate = this.glCapabilities.halfFloatRenderTarget ? "RGBA16F" : "RGBA8";
    this.outputDescription = `WebGL2 回退路径使用 sRGB 标准动态范围 · ${intermediate} 中间缓冲`;
    this.capabilities = Object.freeze({
      api: "webgl2",
      backend: this.backend,
      adapter: this.gpu,
      maxTextureSize: this.glCapabilities.maxTextureSize,
      maxRenderbufferSize: this.glCapabilities.maxRenderbufferSize,
      maxViewport: `${this.glCapabilities.maxViewportWidth}×${this.glCapabilities.maxViewportHeight}`,
      fragmentHighpBits: this.glCapabilities.fragmentHighpBits,
      intermediateFormat: intermediate,
      halfFloatExtension: this.glCapabilities.halfFloatExtension,
      skyTexture: this.skyDetail.split(" ")[0],
      skyUrl: this.skyUrl,
    });
    console.info("Black-hole renderer capabilities", this.capabilities);

    const source = typeof skyUrl === "string" ? {} : skyUrl;
    if (
      skyMode !== "high"
      && !blockForUltra
      && source.ultra
      && source.ultra !== this.skyUrl
      && this.glCapabilities.maxTextureSize >= ULTRA_SKY_DIMENSION
    ) {
      scheduleBackgroundTask(() => {
        void this.upgradeSkyTexture(source.ultra);
      });
    }
  }

  uploadSkyTexture(texture, url) {
    clearErrors(this.context);
    this.renderer.initTexture(texture);
    const error = this.context.getError();
    if (error !== this.context.NO_ERROR) {
      throw new Error(`WebGL rejected sky texture ${url} (error 0x${error.toString(16)})`);
    }
  }

  async upgradeSkyTexture(url) {
    let texture;
    try {
      texture = await loadTexture(url);
      const image = texture.image;
      const width = image.naturalWidth || image.width;
      const height = image.naturalHeight || image.height;
      if (
        width > this.glCapabilities.maxTextureSize
        || height > this.glCapabilities.maxTextureSize
      ) {
        throw new Error(
          `${width}×${height} exceeds the WebGL ${this.glCapabilities.maxTextureSize}px texture limit`,
        );
      }
      configureSkyTexture(texture);
      this.uploadSkyTexture(texture, url);

      const previousTexture = this.skyTexture;
      this.skyTexture = texture;
      this.skyUrl = url;
      this.skyRadianceScale = /gaia-edr3/i.test(url) ? 0.16 : 0.55;
      this.skyDetail = `${width}×${height} 原始全景 · 解析恒星层`;
      this.traceUniforms.tSky.value = texture;
      this.traceUniforms.uSkyRadianceScale.value = this.skyRadianceScale;
      this.capabilities = Object.freeze({
        ...this.capabilities,
        skyTexture: `${width}×${height}`,
        skyUrl: url,
      });
      previousTexture?.dispose();
      console.info("Black-hole renderer capabilities", this.capabilities);
      this.onSkyChanged?.();
    } catch (error) {
      texture?.dispose();
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
        `Render target ${requestedWidth}×${requestedHeight} exceeds the WebGL2 limit; `
        + `using ${nextWidth}×${nextHeight}.`,
      );
    }
    if (nextWidth === this.width && nextHeight === this.height && this.traceTarget) {
      return;
    }

    this.width = nextWidth;
    this.height = nextHeight;
    this.renderer.setSize(nextWidth, nextHeight, false);
    this.traceUniforms.uResolution.value.set(nextWidth, nextHeight);

    this.traceTarget?.dispose();
    this.traceTarget = new THREE.WebGLRenderTarget(nextWidth, nextHeight, {
      type: this.targetType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
      colorSpace: THREE.LinearSRGBColorSpace,
    });
    this.traceTarget.texture.generateMipmaps = false;
    this.postUniforms.tScene.value = this.traceTarget.texture;
  }

  writeUniforms(frame) {
    const uniforms = this.traceUniforms;
    uniforms.uTime.value = frame.time;
    uniforms.uMassSolar.value = frame.massSolar;
    uniforms.uAccretion.value = frame.accretion;
    uniforms.uExposure.value = frame.exposure;
    uniforms.uMode.value = frame.mode;
    uniforms.uSteps.value = frame.steps;
    uniforms.uCameraPos.value.fromArray(frame.cameraPos);
    uniforms.uCameraRadius.value = frame.cameraRadius;
    uniforms.uForward.value.fromArray(frame.forward);
    uniforms.uFov.value = frame.fov;
    uniforms.uRight.value.fromArray(frame.right);
    uniforms.uSkyRotation.value = frame.skyRotation;
    uniforms.uUp.value.fromArray(frame.up);
    uniforms.uDiskOuterRadius.value = frame.diskOuterRadius;
    uniforms.uRenderScale.value = frame.renderScale;
    uniforms.uBloom.value = frame.bloom;
    uniforms.uMotion.value = frame.motion;
    uniforms.uFrame.value = frame.frame;
    uniforms.uObserverVelocity.value.fromArray(frame.observerVelocity);
    uniforms.uObserverBeta.value = frame.observerBeta;
  }

  render(frame) {
    if (!this.traceTarget) {
      return;
    }
    this.writeUniforms(frame);
    this.renderer.setRenderTarget(this.traceTarget);
    this.renderer.render(this.traceScene, this.camera);
    this.renderer.setRenderTarget(null);
    this.renderer.render(this.postScene, this.camera);
  }

  dispose() {
    this.traceTarget?.dispose();
    this.traceMaterial?.dispose();
    this.postMaterial?.dispose();
    this.skyTexture?.dispose();
    this.renderer?.dispose();
  }
}
