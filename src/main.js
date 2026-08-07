import {
  WebGPURenderer,
  webGLRecoveryUrl,
  webGPUFallbackDescription,
} from "./webgpu-renderer.js";
import { WebGLRenderer } from "./webgl-renderer.js";
import {
  applyStrongFieldFrameParameters,
  createStrongFieldQualityScheduler,
} from "./strong-field-quality.js";
import {
  applyDocumentI18n,
  createI18n,
  isSupportedLocale,
  languageUrl,
  persistLocale,
} from "./i18n.js";

const query = new URLSearchParams(window.location.search);
const i18n = createI18n(query);
if (isSupportedLocale(query.get("lang"))) {
  persistLocale(i18n.locale);
}
applyDocumentI18n(document, i18n);
const requestedSkyMode = query.get("sky") === "ultra" ? "ultra" : "high";
if (query.get("presentation") === "1") {
  document.documentElement.classList.add("is-presentation");
}

let activeScene = null;

const SKY_URLS = {
  ultra: "./assets/gaia-edr3-16k.png",
  high: "./assets/milky-way-360-6k.jpg",
};
const SCHWARZSCHILD_KM_PER_SOLAR_MASS = 2.953339382;
const GRAVITATIONAL_KM_PER_SOLAR_MASS = SCHWARZSCHILD_KM_PER_SOLAR_MASS / 2;
const AU_KM = 149_597_870.7;
const DEG = 180 / Math.PI;

let canvas = document.querySelector("#universe");
const app = document.querySelector(".app");

const ui = Object.fromEntries(
  [
    "backendStatus",
    "gpuStatus",
    "hdrStatus",
    "fpsValue",
    "renderScaleValue",
    "mass",
    "massValue",
    "accretion",
    "accretionValue",
    "exposure",
    "exposureValue",
    "timeScale",
    "timeScaleValue",
    "quality",
    "qualityValue",
    "observerValue",
    "rsValue",
    "shadowValue",
    "modeScience",
    "modeHubble",
    "modeLookback",
    "modeFrequency",
    "modeNull",
    "modeError",
    "toggleMotion",
    "resetView",
    "togglePanel",
    "panel",
    "interactionHint",
  ].map((id) => [id, document.querySelector(`#${id}`)]),
);

for (const [id, element] of Object.entries(ui)) {
  if (!element) {
    throw new Error(`Missing required interface element #${id}`);
  }
}

// Keep startup tolerant of a briefly stale cached index.html. A new module can
// otherwise arrive before the matching markup and prevent the renderer from
// initializing at all.
const skySourceSelect = document.getElementById("skySource");
if (skySourceSelect) {
  skySourceSelect.value = requestedSkyMode;
}
const languageSelect = document.getElementById("languageSelect");
if (languageSelect) {
  languageSelect.value = i18n.locale;
}

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

function cameraDistanceLimits() {
  return activeScene?.cameraDistanceLimits ?? { min: 34, max: 90 };
}

function clampCameraDistance(value) {
  const { min, max } = cameraDistanceLimits();
  return clamp(value, min, max);
}

function normalize(vector) {
  const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function scale(vector, factor) {
  return [vector[0] * factor, vector[1] * factor, vector[2] * factor];
}

function valueOrPower(input, powerThreshold = 20) {
  const value = Number(input.value);
  return Math.abs(value) <= powerThreshold ? 10 ** value : value;
}

const state = {
  running: true,
  phase: 0.55,
  orbitTilt: 0.42,
  distance: 50,
  time: 0,
  massSolar: valueOrPower(ui.mass),
  accretion: valueOrPower(ui.accretion),
  exposure: Number(ui.exposure.value),
  timeScale: Number(ui.timeScale.value),
  quality: Number(ui.quality.value),
  // Start in the physically neutral display transform.  The warmer, more
  // saturated Hubble treatment remains available as an explicit style choice.
  mode: 0,
  dynamicScale: 1,
  renderScale: 1,
  frame: 0,
  fps: 0,
  fpsFrames: 0,
  fpsElapsed: 0,
  lastAdaptation: 0,
  userHoldUntil: 0,
  pointers: new Map(),
  pinchDistance: 0,
  dragging: false,
  hintHidden: false,
  resizePending: true,
  lastWidth: 0,
  lastHeight: 0,
  lastScale: 0,
  needsRender: true,
};

let renderer;
let rendererFallbackReason = "";
let rendererRecoveryStarted = false;
let runtimeRenderFailed = false;
let strongFieldQualityScheduler = null;
let strongFieldPreviousFrameRendered = false;
let strongFieldInteractionUntil = 0;
const RENDERER_ROOT_CLASSES = Object.freeze([
  "renderer-webgpu",
  "renderer-webgl2",
]);

function setRendererRootClass(api = null) {
  document.documentElement.classList.remove(...RENDERER_ROOT_CLASSES);
  if (api === "webgpu" || api === "webgl2") {
    document.documentElement.classList.add(`renderer-${api}`);
  }
}

function readOnlySceneRenderer(rendererInstance) {
  const readableProperties = new Set([
    "backend",
    "gpu",
    "hdrMode",
    "outputDescription",
    "skyDetail",
    "capabilities",
    "outputHDR",
    "displayP3",
    "hdrPeak",
    "format",
    "progressiveAccumulation",
    "readyForFrame",
    "lastQueueCompletionAtMs",
    "lastCompletedFrameTimeMs",
  ]);
  const mutationError = () => {
    throw new TypeError(
      "onRendererReady() receives a read-only renderer diagnostic view",
    );
  };
  return new Proxy(rendererInstance, {
    get(target, property) {
      if (!readableProperties.has(property)) {
        throw new TypeError(
          `onRendererReady() cannot access renderer.${String(property)}`,
        );
      }
      return Reflect.get(target, property, target);
    },
    set: mutationError,
    defineProperty: mutationError,
    deleteProperty: mutationError,
    setPrototypeOf: mutationError,
  });
}

async function notifySceneRendererReady() {
  const hook = activeScene?.onRendererReady;
  if (hook == null) {
    return;
  }
  if (typeof hook !== "function") {
    throw new TypeError("Scene onRendererReady must be a function");
  }
  await hook.call(
    activeScene,
    renderer.capabilities,
    readOnlySceneRenderer(renderer),
  );
}

function usesStrongFieldQuality(scene) {
  return (
    scene?.qualityPolicy?.id === "m3-pro-strong-field-v1"
    || scene?.rendererOptions?.shaderBundle?.id === "binary-strong-field-v1"
  );
}

function invalidateStrongFieldQuality(reason, interactionMs = 0) {
  strongFieldQualityScheduler?.invalidate(reason);
  if (interactionMs > 0) {
    strongFieldInteractionUntil = Math.max(
      strongFieldInteractionUntil,
      performance.now() + interactionMs,
    );
  }
}

function revisionNumber(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error(`Strong-field ${name} must be finite`);
  }
  return Object.is(number, -0) ? "-0" : String(number);
}

function revisionSignature(fields) {
  const parts = [];
  for (const [name, value] of fields) {
    if (value == null) {
      parts.push(`${name}=null`);
    } else if (Array.isArray(value) || ArrayBuffer.isView(value)) {
      parts.push(
        `${name}=[${Array.from(
          value,
          (entry, index) => revisionNumber(entry, `${name}[${index}]`),
        ).join(",")}]`,
      );
    } else {
      parts.push(`${name}=${revisionNumber(value, name)}`);
    }
  }
  return parts.join("|");
}

function strongFieldRevisionTokens(frame) {
  const sceneTokens = activeScene?.renderRevision?.(frame, state);
  if (
    sceneTokens != null
    && (
      typeof sceneTokens !== "object"
      || Array.isArray(sceneTokens)
    )
  ) {
    throw new Error("Strong-field renderRevision() must return an object");
  }
  const camera = revisionSignature([
    ["cameraPos", frame.cameraPos],
    ["cameraRadius", frame.cameraRadius],
    ["forward", frame.forward],
    ["fov", frame.fov],
    ["right", frame.right],
    ["up", frame.up],
    ["observerVelocity", frame.observerVelocity],
    ["observerBeta", frame.observerBeta],
  ]);
  const physics = sceneTokens?.physics ?? revisionSignature([
    ["time", frame.time],
    ["sceneStrongFieldUniforms", frame.sceneStrongFieldUniforms],
  ]);
  const transport = sceneTokens?.transport ?? revisionSignature([
    ["mode", frame.mode],
    ["massSolar", frame.massSolar],
    ["accretion", frame.accretion],
    ["exposure", frame.exposure],
    ["skyRotation", frame.skyRotation],
    ["diskOuterRadius", frame.diskOuterRadius],
    ["bloom", frame.bloom],
    ["sceneStrongIntegrator", frame.sceneStrongIntegrator],
    ["sceneStrongDomain", frame.sceneStrongDomain],
    ["sceneStrongDiagnostics", frame.sceneStrongDiagnostics],
  ]);
  return { camera, physics, transport };
}

function sceneHref(sceneId) {
  const parameters = new URLSearchParams(query);
  if (sceneId === "binary-approx") {
    parameters.delete("scene");
    parameters.delete("reference");
    parameters.delete("diagnostic");
  } else {
    parameters.set("scene", sceneId);
    if (sceneId !== "transfer-map-reference") {
      parameters.delete("reference");
      parameters.delete("diagnostic");
    }
  }
  const encoded = parameters.toString();
  return `./${encoded ? `?${encoded}` : ""}`;
}

function configureSceneLinks(sceneId) {
  const links = [
    ["schwarzschild", document.querySelector("#sceneSchwarzschild")],
    ["binary-approx", document.querySelector("#sceneBinary")],
    ["transfer-map-reference", document.querySelector("#sceneTransferMap")],
  ];
  for (const [id, link] of links) {
    if (!link) {
      throw new Error(`Missing scene navigation link for ${id}`);
    }
    link.href = sceneHref(id);
    const active = sceneId === id;
    link.classList.toggle("is-active", active);
    if (active) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  }
}

async function loadRequestedScene() {
  const requestedScene = query.get("scene");
  if (requestedScene === "transfer-map-reference") {
    const {
      createTransferMapReferenceScene,
    } = await import("./scenes/transfer-map-reference-scene.js");
    const scene = await createTransferMapReferenceScene({
      document,
      ui,
      state,
      i18n,
      controls: {
        requestRender() {
          state.needsRender = true;
        },
      },
    });
    validateCustomScene(scene);
    return scene;
  }
  if (requestedScene === "schwarzschild") {
    return null;
  }
  const { createBinaryApproxScene } = await import("./scenes/binary-approx-scene.js");
  const scene = await createBinaryApproxScene({
    document,
    ui,
    state,
    i18n,
    formatMass,
    formatGravitationalRadius,
    controls: {
      setRunning: setMotion,
      requestRender() {
        state.needsRender = true;
      },
    },
  });
  validateCustomScene(scene);
  return scene;
}

function validateCustomScene(scene) {
  const bundle = scene?.rendererOptions?.shaderBundle;
  if (!bundle?.id || !bundle.wgsl?.trace || !bundle.glsl?.trace) {
    throw new Error(
      "A custom scene must provide both WGSL and GLSL trace shaders so GPU fallback stays consistent",
    );
  }
  if (
    bundle.wgsl.vertex
    || bundle.wgsl.post
    || bundle.glsl.vertex
    || bundle.glsl.post
  ) {
    throw new Error(
      "Version 1 scene bundles may replace only the trace shader; vertex and HDR post stages stay shared",
    );
  }
  if (
    bundle.resources
    && (
      typeof bundle.resources.createWebGPU !== "function"
      || typeof bundle.resources.createWebGL !== "function"
    )
  ) {
    throw new Error(
      "A resource-backed scene must support both WebGPU and WebGL2 fallback",
    );
  }
  if (scene.cameraDistanceLimits != null) {
    const { min, max } = scene.cameraDistanceLimits;
    if (
      !Number.isFinite(min)
      || !Number.isFinite(max)
      || min <= 2
      || max <= min
    ) {
      throw new Error(
        "Scene cameraDistanceLimits must be finite with 2 < min < max",
      );
    }
  }
}

function replaceCanvasForFallback() {
  const replacement = canvas.cloneNode(false);
  canvas.replaceWith(replacement);
  canvas = replacement;
}

function rendererErrorMessage(error) {
  return error instanceof Error ? error.message : String(error || "unknown");
}

function requestWebGLRendererRecovery(reason, error, source = renderer) {
  if (
    rendererRecoveryStarted
    || source?.capabilities?.api !== "webgpu"
  ) {
    return false;
  }
  rendererRecoveryStarted = true;
  const detail = rendererErrorMessage(error);
  if (reason === "device-lost") {
    strongFieldQualityScheduler?.signalDeviceLost(detail);
  }
  console.error(
    `WebGPU runtime failure (${reason}); reloading with the explicit WebGL2 fallback.`,
    error,
  );
  const recoveryUrl = webGLRecoveryUrl(window.location.href, reason);
  source.dispose?.();
  window.location.replace(recoveryUrl);
  return true;
}

async function createRenderer() {
  const requestedBackend = query.get("renderer");
  const rendererOptions = {
    ...activeScene?.rendererOptions,
    locale: i18n.locale,
  };
  if (requestedBackend === "webgl") {
    rendererFallbackReason = webGPUFallbackDescription(
      query.get("fallback"),
      i18n.locale,
    );
    return WebGLRenderer.create(canvas, SKY_URLS, rendererOptions);
  }
  try {
    const webgpu = await WebGPURenderer.create(canvas, SKY_URLS, rendererOptions);
    webgpu.onLost = (info) => {
      requestWebGLRendererRecovery(
        "device-lost",
        info.message || info.reason || "unknown",
        webgpu,
      );
    };
    webgpu.onError = (error) => {
      requestWebGLRendererRecovery("render-error", error, webgpu);
    };
    // A device can fail in the short interval between init() installing the
    // platform handlers and create() returning to this caller.
    if (webgpu.lost) {
      webgpu.onLost(
        webgpu.deviceLossInfo || { reason: "unknown", message: "device lost during startup" },
      );
    } else if (webgpu.pendingRuntimeError) {
      webgpu.onError(webgpu.pendingRuntimeError);
    }
    return webgpu;
  } catch (error) {
    rendererFallbackReason = error instanceof Error ? error.message : String(error);
    console.info("WebGPU unavailable; using WebGL2 hardware fallback.", error);
    replaceCanvasForFallback();
    strongFieldQualityScheduler?.setBackend("webgl2", rendererFallbackReason);
    return WebGLRenderer.create(canvas, SKY_URLS, rendererOptions);
  }
}

function formatMass(mass) {
  const exponent = Math.floor(Math.log10(mass));
  const mantissa = mass / 10 ** exponent;
  const superscripts = String(exponent).replace(/-/g, "⁻").replace(/0/g, "⁰").replace(/1/g, "¹")
    .replace(/2/g, "²").replace(/3/g, "³").replace(/4/g, "⁴").replace(/5/g, "⁵")
    .replace(/6/g, "⁶").replace(/7/g, "⁷").replace(/8/g, "⁸").replace(/9/g, "⁹");
  return `${mantissa.toFixed(2)} × 10${superscripts} M☉`;
}

function formatLength(km) {
  const au = km / AU_KM;
  if (au >= 0.1) {
    return `${i18n.formatNumber(au, { maximumFractionDigits: au < 10 ? 2 : 1 })} AU`;
  }
  if (km >= 1e6) {
    return `${(km / 1e6).toFixed(2)} × 10⁶ km`;
  }
  return `${i18n.formatNumber(Math.round(km))} km`;
}

function formatRadius(massSolar) {
  return formatLength(SCHWARZSCHILD_KM_PER_SOLAR_MASS * massSolar);
}

function formatGravitationalRadius(massSolar) {
  return formatLength(GRAVITATIONAL_KM_PER_SOLAR_MASS * massSolar);
}

function updateReadouts() {
  invalidateStrongFieldQuality("control-change");
  state.massSolar = valueOrPower(ui.mass);
  state.accretion = valueOrPower(ui.accretion);
  state.exposure = Number(ui.exposure.value);
  state.timeScale = Number(ui.timeScale.value);
  state.quality = Number(ui.quality.value);

  if (activeScene?.updateReadouts?.()) {
    state.needsRender = true;
    return;
  }

  const lapse = Math.sqrt(1 - 2 / state.distance);
  const shadowHalfAngle = Math.asin(clamp((3 * Math.sqrt(3) * lapse) / state.distance, 0, 1));
  const orbitalBeta = 1 / Math.sqrt(state.distance - 2);

  ui.massValue.textContent = formatMass(state.massSolar);
  const eddingtonPercent = state.accretion * 100;
  const eddingtonDigits = eddingtonPercent < 0.1 ? 3 : eddingtonPercent < 1 ? 2 : 1;
  ui.accretionValue.textContent = `${eddingtonPercent.toFixed(eddingtonDigits)}% Edd`;
  ui.exposureValue.textContent = `${state.exposure.toFixed(2)}×`;
  ui.timeScaleValue.textContent = `${state.timeScale.toFixed(0)} M/s`;
  ui.qualityValue.textContent = `${state.quality.toFixed(2)}×`;
  ui.observerValue.innerHTML = `${state.distance.toFixed(1)} r<sub>g</sub> · β ${orbitalBeta.toFixed(3)}c`;
  ui.rsValue.textContent = formatRadius(state.massSolar);
  ui.shadowValue.textContent = `${(2 * shadowHalfAngle * DEG).toFixed(2)}°`;
  state.needsRender = true;
}

function setMode(mode) {
  invalidateStrongFieldQuality("display-mode-change");
  state.mode = mode;
  [
    ui.modeScience,
    ui.modeHubble,
    ui.modeLookback,
    ui.modeFrequency,
    ui.modeNull,
    ui.modeError,
  ].forEach((button, buttonMode) => {
    const active = mode === buttonMode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  activeScene?.onModeChanged?.(mode);
  state.needsRender = true;
}

function setMotion(running) {
  if (activeScene?.motionEnabled === false) {
    running = false;
  }
  state.running = running;
  invalidateStrongFieldQuality("timeline-state-change");
  const labels = activeScene?.motionLabels ?? {
    pause: i18n.t("motion.pauseOrbit"),
    resume: i18n.t("motion.resumeOrbit"),
  };
  const actionLabel = running ? labels.pause : labels.resume;
  ui.toggleMotion.dataset.state = running ? "running" : "paused";
  ui.toggleMotion.disabled = activeScene?.motionEnabled === false;
  ui.toggleMotion.setAttribute("aria-pressed", String(!running));
  ui.toggleMotion.setAttribute("aria-label", actionLabel);
  ui.toggleMotion.setAttribute("title", actionLabel);
  const mark = ui.toggleMotion.querySelector("span");
  if (mark) {
    mark.textContent = running ? "Ⅱ" : "▶";
  }
  activeScene?.onMotionChanged?.(running);
  state.needsRender = true;
}

function resetView() {
  if (activeScene?.cameraLocked) {
    return;
  }
  if (activeScene?.resetState) {
    activeScene.resetState();
  } else {
    state.phase = 0.55;
    state.orbitTilt = 0.42;
    state.distance = 50;
  }
  state.dynamicScale = 1;
  invalidateStrongFieldQuality("camera-reset", 180);
  state.userHoldUntil = performance.now() + 1200;
  state.resizePending = true;
  state.needsRender = true;
  updateReadouts();
}

function hideInteractionHint() {
  if (state.hintHidden) {
    return;
  }
  state.hintHidden = true;
  ui.interactionHint.classList.add("is-hidden");
}

function pointerSeparation() {
  const points = [...state.pointers.values()];
  if (points.length < 2) {
    return 0;
  }
  return Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y);
}

function beginUserHold(duration = 2600) {
  state.userHoldUntil = performance.now() + duration;
  invalidateStrongFieldQuality("camera-input", 120);
  hideInteractionHint();
}

function bindInteractions() {
  canvas.addEventListener("pointerdown", (event) => {
    if (activeScene?.cameraLocked) {
      return;
    }
    canvas.setPointerCapture(event.pointerId);
    state.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    state.dragging = true;
    app.classList.add("is-dragging");
    state.pinchDistance = pointerSeparation();
    beginUserHold();
  });

  canvas.addEventListener("pointermove", (event) => {
    const previous = state.pointers.get(event.pointerId);
    if (!previous) {
      return;
    }

    const dx = event.clientX - previous.x;
    const dy = event.clientY - previous.y;
    state.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

    if (state.pointers.size === 1) {
      state.phase -= dx * 0.0052;
      state.orbitTilt = clamp(state.orbitTilt + dy * 0.0042, -1.46, 1.46);
    } else if (state.pointers.size >= 2) {
      const separation = pointerSeparation();
      if (state.pinchDistance > 0 && separation > 0) {
        state.distance = clampCameraDistance(
          state.distance * (state.pinchDistance / separation),
        );
      }
      state.pinchDistance = separation;
    }

    beginUserHold();
    updateReadouts();
  });

  const endPointer = (event) => {
    state.pointers.delete(event.pointerId);
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    state.pinchDistance = pointerSeparation();
    state.dragging = state.pointers.size > 0;
    app.classList.toggle("is-dragging", state.dragging);
  };

  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);

  canvas.addEventListener(
    "wheel",
    (event) => {
      if (activeScene?.cameraLocked) {
        return;
      }
      event.preventDefault();
      state.distance = clampCameraDistance(
        state.distance * Math.exp(event.deltaY * 0.0008),
      );
      beginUserHold(1800);
      updateReadouts();
    },
    { passive: false },
  );

  canvas.addEventListener("dblclick", () => {
    if (!activeScene?.cameraLocked) {
      resetView();
    }
  });
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());

  window.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLButtonElement) {
      return;
    }
    if (activeScene?.cameraLocked) {
      return;
    }
    let handled = true;
    if (event.key === "ArrowLeft") state.phase += 0.06;
    else if (event.key === "ArrowRight") state.phase -= 0.06;
    else if (event.key === "ArrowUp") state.orbitTilt = clamp(state.orbitTilt - 0.05, -1.46, 1.46);
    else if (event.key === "ArrowDown") state.orbitTilt = clamp(state.orbitTilt + 0.05, -1.46, 1.46);
    else if (event.key === "0") state.orbitTilt = 0;
    else if (event.key === "+" || event.key === "=") {
      state.distance = clampCameraDistance(state.distance - 1.5);
    } else if (event.key === "-" || event.key === "_") {
      state.distance = clampCameraDistance(state.distance + 1.5);
    }
    else if (event.key === " ") setMotion(!state.running);
    else handled = false;
    if (handled) {
      event.preventDefault();
      beginUserHold();
      updateReadouts();
    }
  });
}

function cameraFrame() {
  const cosPhase = Math.cos(state.phase);
  const sinPhase = Math.sin(state.phase);
  const cosTilt = Math.cos(state.orbitTilt);
  const sinTilt = Math.sin(state.orbitTilt);
  // A Schwarzschild circular geodesic lies in a plane through the origin.  The
  // user tilts that orbital plane; setting the tilt to zero makes the entire
  // orbit exactly coplanar with the accretion disk.
  const orbitBasis = [cosTilt, sinTilt, 0];
  const tangent = normalize([
    -sinPhase * orbitBasis[0],
    -sinPhase * orbitBasis[1],
    cosPhase,
  ]);
  const positionUnit = normalize([
    cosPhase * orbitBasis[0],
    cosPhase * orbitBasis[1],
    sinPhase,
  ]);
  const cameraPos = scale(positionUnit, state.distance);
  const forward = scale(positionUnit, -1);
  const right = tangent;
  const up = normalize(cross(forward, right));

  const baseCamera = {
    cameraPos,
    forward,
    right,
    up,
    observerVelocity: tangent,
    observerBeta: 1 / Math.sqrt(state.distance - 2),
  };
  return activeScene?.cameraFrame?.(baseCamera, state) ?? baseCamera;
}

function effectiveRenderScale() {
  const deviceScale = Math.min(window.devicePixelRatio || 1, 2);
  let scale = deviceScale * state.quality * state.dynamicScale;
  // Let the 1.25x quality setting supersample a Retina canvas on GPUs with
  // spare headroom.  The pixel budget and feedback governor below still keep
  // the default path at native display density on Apple Silicon Macs.
  scale = clamp(scale, 0.65, 2.5);

  const pixels = window.innerWidth * window.innerHeight * scale * scale;
  // Keep the ray-traced sky at or above native CSS resolution on ordinary
  // displays.  Retina panels still receive a multi-megapixel render while the
  // dynamic scaler protects interactivity on slower GPUs.
  const pixelBudget = matchMedia("(max-width: 760px)").matches ? 3_000_000 : 7_500_000;
  if (pixels > pixelBudget) {
    scale *= Math.sqrt(pixelBudget / pixels);
  }
  return clamp(scale, 0.65, 2.5);
}

function resizeRenderer(force = false) {
  const scaleValue = effectiveRenderScale();
  const width = Math.max(1, Math.floor(window.innerWidth * scaleValue));
  const height = Math.max(1, Math.floor(window.innerHeight * scaleValue));
  if (!force && width === state.lastWidth && height === state.lastHeight) {
    return;
  }
  state.lastWidth = width;
  state.lastHeight = height;
  state.lastScale = scaleValue;
  state.renderScale = scaleValue;
  renderer.resize(width, height);
  ui.renderScaleValue.textContent = `${scaleValue.toFixed(2)}× · ${width}×${height}`;
  state.resizePending = false;
  state.needsRender = true;
}

function adaptQuality(now) {
  if (now - state.lastAdaptation < 1600 || state.fps <= 0) {
    return;
  }
  state.lastAdaptation = now;
  const compactViewport = matchMedia("(max-width: 760px)").matches;
  const lowTarget = compactViewport ? 26 : 30;
  const highTarget = compactViewport ? 46 : 52;
  const minimumDynamicScale = compactViewport ? 0.34 : 0.38;
  const previous = state.dynamicScale;
  if (state.fps < lowTarget) {
    // Resolution cost is approximately quadratic.  A proportional reduction
    // converges much faster than fixed 0.08 steps when a lower-core-count GPU
    // starts far below target, while the floor still leaves a usable image.
    const correction = clamp(Math.sqrt(state.fps / lowTarget) * 0.96, 0.72, 0.92);
    state.dynamicScale = Math.max(minimumDynamicScale, state.dynamicScale * correction);
  } else if (state.fps > highTarget && state.dynamicScale < 1) {
    state.dynamicScale = Math.min(1, state.dynamicScale + 0.04);
  }
  if (Math.abs(previous - state.dynamicScale) > 0.001) {
    state.resizePending = true;
  }
}

function stepBudget() {
  let steps = matchMedia("(max-width: 760px)").matches ? 236 : 288;
  if (state.dynamicScale < 0.82) steps -= 32;
  if (state.dynamicScale < 0.64) steps -= 40;
  if (state.dynamicScale < 0.48) steps -= 32;
  if (state.fps > 0 && state.fps < 24) steps -= 32;
  return clamp(steps, 184, 288);
}

function frameParameters() {
  const camera = cameraFrame();
  const portraitFov = window.innerWidth / window.innerHeight < 0.8 ? 68 : 44;
  const baseFrame = {
    time: state.time,
    massSolar: state.massSolar,
    accretion: state.accretion,
    exposure: state.exposure,
    mode: state.mode,
    steps: stepBudget(),
    cameraPos: camera.cameraPos,
    cameraRadius: state.distance,
    forward: camera.forward,
    fov: portraitFov / DEG,
    right: camera.right,
    // Align the Gaia/ESO Galactic Centre with the initial line of sight.  The
    // celestial sphere remains fixed while the observer moves around the hole.
    skyRotation: -2.576,
    up: camera.up,
    diskOuterRadius: 18,
    renderScale: state.renderScale,
    // In the strong-field scene mode 1 is the categorical ray-outcome view,
    // not the legacy Hubble display look.  Diagnostic masks must never pass
    // through a photographic bloom transform.
    bloom: state.mode === 1 && !usesStrongFieldQuality(activeScene) ? 0.06 : 0,
    motion: state.running ? 1 : 0,
    frame: state.frame,
    observerVelocity: camera.observerVelocity,
    observerBeta: camera.observerBeta,
  };
  return activeScene?.extendFrame?.(baseFrame, state) ?? baseFrame;
}

function rendererCanSubmitFrame() {
  if (typeof renderer?.canSubmitFrame === "function") {
    return renderer.canSubmitFrame();
  }
  if (typeof renderer?.readyForFrame === "boolean") {
    return renderer.readyForFrame;
  }
  return true;
}

function consumeRendererFrameTimeMs(frameElapsed) {
  const completed = renderer?.consumeCompletedFrameTimeMs?.();
  if (Number.isFinite(completed) && completed > 0) {
    return completed;
  }
  const webgpu = (
    renderer?.capabilities?.api === "webgpu"
    || renderer instanceof WebGPURenderer
  );
  return !webgpu && strongFieldPreviousFrameRendered
    ? frameElapsed * 1_000
    : null;
}

function scheduledStrongFieldFrame(now, frameTimeMs) {
  const baseFrame = frameParameters();
  const revisions = strongFieldRevisionTokens(baseFrame);
  if (state.needsRender) {
    strongFieldQualityScheduler.invalidate("scene-render-request");
  }
  const decision = strongFieldQualityScheduler.nextFrame({
    nowMs: now,
    frameTimeMs,
    viewportWidth: window.innerWidth,
    viewportHeight: window.innerHeight,
    devicePixelRatio: window.devicePixelRatio || 1,
    requestedQuality: state.quality,
    cameraRevision: revisions.camera,
    physicsRevision: revisions.physics,
    transportRevision: revisions.transport,
    interactionActive: (
      state.dragging
      || now < strongFieldInteractionUntil
    ),
    timelineRunning: state.running,
    backend: renderer.capabilities?.api || (
      renderer instanceof WebGPURenderer ? "webgpu" : "webgl2"
    ),
    visible: !document.hidden,
  });
  const { width, height, renderScale } = decision.resolution;
  if (
    width !== state.lastWidth
    || height !== state.lastHeight
    || Math.abs(renderScale - state.lastScale) > 1e-9
  ) {
    renderer.resize(width, height);
    state.lastWidth = width;
    state.lastHeight = height;
    state.lastScale = renderScale;
    state.renderScale = renderScale;
  }
  ui.renderScaleValue.textContent = (
    `${decision.qualityTierId} · ${renderScale.toFixed(2)}× · ${width}×${height}`
  );
  app.dataset.strongFieldTier = decision.qualityTierId;
  app.dataset.strongFieldPerformanceTier = String(decision.performanceTier);
  app.dataset.strongFieldPhase = decision.phase;
  app.dataset.strongFieldGpuMs = Number.isFinite(decision.timing.frameTimeEmaMs)
    ? decision.timing.frameTimeEmaMs.toFixed(2)
    : "";
  app.dataset.strongFieldFpsEma = Number.isFinite(decision.timing.fpsEma)
    ? decision.timing.fpsEma.toFixed(2)
    : "";
  app.dataset.strongFieldHistoryEpoch = String(decision.historyEpoch);
  // Keep the completed running-average count observable after the scheduler
  // enters `steady`; the per-frame accumulation index intentionally returns to
  // zero when no further trace submission is required.
  app.dataset.strongFieldAccumulation = String(
    strongFieldQualityScheduler.snapshot().accumulationSamples,
  );
  state.resizePending = false;
  const qualityFrame = applyStrongFieldFrameParameters(
    baseFrame,
    decision,
  );
  const sceneQualityFrame = activeScene?.applyStrongFieldQuality?.(
    qualityFrame,
    decision,
  ) ?? qualityFrame;
  if (
    !sceneQualityFrame
    || typeof sceneQualityFrame !== "object"
    || Array.isArray(sceneQualityFrame)
  ) {
    throw new Error("applyStrongFieldQuality() must return a frame object");
  }
  return {
    decision,
    frame: {
      ...sceneQualityFrame,
      // The strong-field shader consumes this as its sub-pixel sample index.
      frame: decision.accumulationIndex,
    },
  };
}

function updateFps(dt) {
  state.fpsFrames += 1;
  state.fpsElapsed += dt;
  if (state.fpsElapsed >= 0.75) {
    state.fps = state.fpsFrames / state.fpsElapsed;
    ui.fpsValue.textContent = Math.round(state.fps).toString();
    state.fpsFrames = 0;
    state.fpsElapsed = 0;
  }
}

function bindUi() {
  const panelContext = activeScene?.panelLabel ?? i18n.t("panel.observationSettings");
  ui.togglePanel.setAttribute(
    "aria-label",
    i18n.t("panel.expand", { context: panelContext }),
  );
  [ui.mass, ui.accretion, ui.exposure, ui.timeScale].forEach((input) => {
    input.addEventListener("input", updateReadouts);
  });
  ui.quality.addEventListener("input", () => {
    updateReadouts();
    state.dynamicScale = 1;
    state.resizePending = true;
  });
  skySourceSelect?.addEventListener("change", () => {
    const selectedSkyMode = skySourceSelect.value === "ultra" ? "ultra" : "high";
    if (selectedSkyMode !== requestedSkyMode) {
      const nextUrl = new URL(window.location.href);
      nextUrl.searchParams.set("sky", selectedSkyMode);
      window.location.assign(nextUrl.href);
    }
  });
  languageSelect?.addEventListener("change", () => {
    const locale = persistLocale(languageSelect.value);
    // Scenes and renderers snapshot their locale. A bounded reload applies the
    // new catalog atomically while languageUrl preserves every route parameter.
    window.location.assign(languageUrl(window.location.href, locale));
  });
  [
    ui.modeScience,
    ui.modeHubble,
    ui.modeLookback,
    ui.modeFrequency,
    ui.modeNull,
    ui.modeError,
  ].forEach((button, mode) => {
    button.addEventListener("click", () => setMode(mode));
  });
  ui.toggleMotion.addEventListener("click", () => setMotion(!state.running));
  ui.resetView.addEventListener("click", resetView);
  ui.togglePanel.addEventListener("click", () => {
    const expanded = !ui.panel.classList.contains("is-open");
    ui.panel.classList.toggle("is-open", expanded);
    ui.togglePanel.setAttribute("aria-expanded", String(expanded));
    ui.togglePanel.setAttribute(
      "aria-label",
      i18n.t(expanded ? "panel.collapse" : "panel.expand", {
        context: panelContext,
      }),
    );
  });
  window.addEventListener("resize", () => {
    invalidateStrongFieldQuality("viewport-resize");
    state.resizePending = true;
    state.needsRender = true;
  });
  document.addEventListener("visibilitychange", () => {
    invalidateStrongFieldQuality(
      document.hidden ? "visibility-hidden" : "visibility-resume",
    );
    lastFrameTime = performance.now();
  });
}

function showFatalError(message) {
  ui.backendStatus.textContent = i18n.t("fatal.initialization");
  const error = app.querySelector(".fatal-error") || document.createElement("div");
  error.className = "fatal-error";
  error.setAttribute("role", "alert");
  const title = document.createElement("strong");
  title.textContent = i18n.t("fatal.rendererTitle");
  const detail = document.createElement("span");
  detail.textContent = String(message);
  error.replaceChildren(title, detail);
  if (!error.isConnected) {
    app.append(error);
  }
}

let lastFrameTime = performance.now();

function renderAnimationFrame(now) {
  const frameElapsed = Math.max((now - lastFrameTime) / 1000, 0);
  // Clamp only the physical simulation delta after a long stall.  FPS must use
  // the real wall-clock duration or the quality governor overestimates slow
  // frames and never reaches its lower compatibility tiers.
  const dt = Math.min(frameElapsed, 0.1);
  lastFrameTime = now;

  if (!document.hidden) {
    if (activeScene?.advance) {
      if (state.running) {
        activeScene.advance(dt, now);
      }
    } else {
      if (state.running && !state.dragging && now >= state.userHoldUntil) {
        const omega = 1 / state.distance ** 1.5;
        state.phase += omega * dt * state.timeScale;
      }
      if (state.running) {
        state.time += dt * state.timeScale;
      }
    }

    if (strongFieldQualityScheduler) {
      // A strong-field frame can take much longer than a display refresh.
      // Do not consume scheduler revisions or accumulation indices while the
      // preceding Metal submission is still running.  The next free slot is
      // built from the latest camera and physical time, so stale viewpoints
      // can never form a queue.
      if (rendererCanSubmitFrame()) {
        const frameTimeMs = consumeRendererFrameTimeMs(frameElapsed);
        const scheduled = scheduledStrongFieldFrame(now, frameTimeMs);
        strongFieldPreviousFrameRendered = scheduled.decision.shouldRender;
        if (scheduled.decision.shouldRender) {
          const submitted = renderer.render(scheduled.frame) !== false;
          strongFieldPreviousFrameRendered = submitted;
          if (submitted) {
            state.frame = (state.frame + 1) % 16_777_216;
            state.needsRender = false;
            updateFps((frameTimeMs ?? frameElapsed * 1_000) / 1_000);
          }
        }
      }
    } else {
      if (rendererCanSubmitFrame()) {
        if (state.resizePending) {
          resizeRenderer();
        }
        if (state.running || state.dragging || state.needsRender) {
          const frameTimeMs = renderer.consumeCompletedFrameTimeMs?.();
          const submitted = renderer.render(frameParameters()) !== false;
          if (submitted) {
            state.frame = (state.frame + 1) % 16_777_216;
            state.needsRender = false;
            updateFps(
              Number.isFinite(frameTimeMs) && frameTimeMs > 0
                ? frameTimeMs / 1_000
                : frameElapsed,
            );
            adaptQuality(now);
          }
        }
      }
    }
  }

}

function animate(now) {
  try {
    if (!runtimeRenderFailed) {
      renderAnimationFrame(now);
    }
  } catch (error) {
    console.error("Renderer frame failed", error);
    if (!requestWebGLRendererRecovery("render-error", error)) {
      if (!rendererRecoveryStarted) {
        runtimeRenderFailed = true;
        renderer?.dispose?.();
        showFatalError(rendererErrorMessage(error));
      }
    }
  } finally {
    // A synchronous canvas/device exception must never terminate the RAF
    // driver. During WebGPU recovery this loop stays inert until navigation
    // commits; on a non-recoverable backend it keeps the UI responsive.
    requestAnimationFrame(animate);
  }
}

async function start() {
  setRendererRootClass();
  try {
    activeScene = await loadRequestedScene();
    strongFieldQualityScheduler = usesStrongFieldQuality(activeScene)
      ? createStrongFieldQualityScheduler()
      : null;
    strongFieldPreviousFrameRendered = false;
    configureSceneLinks(activeScene?.id || "schwarzschild");
    activeScene?.initialize?.();
    bindUi();
    updateReadouts();
    setMode(state.mode);
    setMotion(activeScene?.startsRunning ?? true);

    renderer = await createRenderer();
    setRendererRootClass(renderer.capabilities?.api);
    await notifySceneRendererReady();
    strongFieldQualityScheduler?.setBackend(
      renderer.capabilities?.api || "webgl2",
      rendererFallbackReason || "renderer-created",
    );
    ui.backendStatus.textContent = renderer.backend;
    ui.gpuStatus.textContent = renderer.gpu;
    const updateOutputStatus = () => {
      ui.hdrStatus.textContent = renderer.hdrMode;
      ui.hdrStatus.title = `${renderer.outputDescription} · ${renderer.skyDetail}`;
    };
    updateOutputStatus();
    const dynamicRange = matchMedia("(dynamic-range: high)");
    dynamicRange.addEventListener?.("change", () => {
      updateOutputStatus();
      invalidateStrongFieldQuality("output-dynamic-range-change");
      state.needsRender = true;
    });
    if (rendererFallbackReason) {
      ui.backendStatus.title = i18n.t("fallback.reason", {
        reason: rendererFallbackReason,
      });
    }
    bindInteractions();
    if (strongFieldQualityScheduler) {
      state.resizePending = true;
      state.needsRender = true;
    } else {
      resizeRenderer(true);
    }
    lastFrameTime = performance.now();
    requestAnimationFrame(animate);
  } catch (error) {
    console.error(error);
    setRendererRootClass();
    renderer?.dispose?.();
    if (!error?.sceneUiHandled) {
      try {
        activeScene?.dispose?.();
      } catch (disposeError) {
        console.info("Scene cleanup after renderer startup failure was incomplete.", disposeError);
      }
    }
    if (error?.sceneUiHandled) {
      ui.backendStatus.textContent = i18n.t("fatal.dataValidation");
      return;
    }
    showFatalError(error instanceof Error ? error.message : error);
  }
}

start();
