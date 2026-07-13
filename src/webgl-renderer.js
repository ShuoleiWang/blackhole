import * as THREE from "../vendor/three.module.js";
import {
  fullscreenVertexGLSL,
  traceFragmentGLSL,
  postFragmentGLSL,
} from "./shaders.js";

function loadTexture(url) {
  return new Promise((resolve, reject) => {
    new THREE.TextureLoader().load(url, resolve, undefined, reject);
  });
}

function gpuLabel(gl) {
  const extension = gl.getExtension("WEBGL_debug_renderer_info");
  if (!extension) {
    return /Mac|iPhone|iPad/.test(navigator.platform) ? "Apple GPU" : "Hardware GPU";
  }
  return gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) || "Hardware GPU";
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

    const instance = new WebGLRenderer(canvas, context);
    await instance.init(skyUrl);
    return instance;
  }

  constructor(canvas, context) {
    this.canvas = canvas;
    this.context = context;
    this.backend = /Mac|iPhone|iPad/.test(navigator.platform)
      ? "WebGL2 · Metal fallback"
      : "WebGL2 · GPU";
    this.gpu = gpuLabel(context);
    this.hdrMode = "sRGB · SDR";
    this.outputHDR = false;
    this.displayP3 = false;
    this.outputDescription = "WebGL2 回退路径使用 sRGB 标准动态范围";
    this.skyDetail = "银河背景待载入";
    this.skyRadianceScale = 0.55;
    this.width = 1;
    this.height = 1;
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

    const source = typeof skyUrl === "string" ? { high: skyUrl, fallback: skyUrl } : skyUrl;
    const maxTextureDimension = this.context.getParameter(this.context.MAX_TEXTURE_SIZE);
    const candidates = maxTextureDimension >= 16000
      ? [source.ultra, source.high, source.fallback]
      : maxTextureDimension >= 6000
        ? [source.high, source.fallback]
        : [source.fallback];
    let lastSkyError;
    for (const url of [...new Set(candidates.filter(Boolean))]) {
      try {
        this.skyTexture = await loadTexture(url);
        this.skyUrl = url;
        break;
      } catch (error) {
        lastSkyError = error;
      }
    }
    if (!this.skyTexture) {
      throw lastSkyError || new Error("No celestial panorama could be loaded");
    }
    this.skyRadianceScale = /gaia-edr3/i.test(this.skyUrl) ? 0.16 : 0.55;
    const image = this.skyTexture.image;
    this.skyDetail = `${image.naturalWidth || image.width}×${image.naturalHeight || image.height} 原始全景 · 解析恒星层`;
    this.skyTexture.wrapS = THREE.RepeatWrapping;
    this.skyTexture.wrapT = THREE.ClampToEdgeWrapping;
    // The shader uses astronomical equirectangular coordinates with v=0 at
    // the north/top of the source image, matching WebGPU's copy convention.
    this.skyTexture.flipY = false;
    this.skyTexture.minFilter = THREE.LinearFilter;
    this.skyTexture.magFilter = THREE.LinearFilter;
    this.skyTexture.colorSpace = THREE.SRGBColorSpace;
    this.skyTexture.generateMipmaps = false;
    this.skyTexture.needsUpdate = true;

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

    const canRenderFloat = this.renderer.extensions.has("EXT_color_buffer_float");
    this.targetType = canRenderFloat ? THREE.HalfFloatType : THREE.UnsignedByteType;
  }

  resize(width, height) {
    const nextWidth = Math.max(1, Math.floor(width));
    const nextHeight = Math.max(1, Math.floor(height));
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
