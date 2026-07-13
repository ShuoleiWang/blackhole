import { WebGPURenderer } from "./webgpu-renderer.js";
import { WebGLRenderer } from "./webgl-renderer.js";

const SKY_URLS = {
  ultra: "./assets/gaia-edr3-16k.png",
  high: "./assets/milky-way-360-6k.jpg",
  fallback: "./assets/milky-way-360.webp",
};
const RG_SECONDS_PER_SOLAR_MASS = 4.925490947e-6;
const SCHWARZSCHILD_KM_PER_SOLAR_MASS = 2.953339382;
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

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

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

function replaceCanvasForFallback() {
  const replacement = canvas.cloneNode(false);
  canvas.replaceWith(replacement);
  canvas = replacement;
}

async function createRenderer() {
  const requestedBackend = new URLSearchParams(location.search).get("renderer");
  if (requestedBackend === "webgl") {
    return WebGLRenderer.create(canvas, SKY_URLS);
  }
  try {
    const webgpu = await WebGPURenderer.create(canvas, SKY_URLS);
    webgpu.onLost = (info) => {
      showFatalError(`GPU 设备连接已丢失：${info.message || info.reason || "unknown"}`);
    };
    return webgpu;
  } catch (error) {
    rendererFallbackReason = error instanceof Error ? error.message : String(error);
    console.info("WebGPU unavailable; using WebGL2 hardware fallback.", error);
    replaceCanvasForFallback();
    return WebGLRenderer.create(canvas, SKY_URLS);
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

function formatRadius(massSolar) {
  const km = SCHWARZSCHILD_KM_PER_SOLAR_MASS * massSolar;
  const au = km / AU_KM;
  if (au >= 0.1) {
    return `${au.toLocaleString("zh-CN", { maximumFractionDigits: au < 10 ? 2 : 1 })} AU`;
  }
  if (km >= 1e6) {
    return `${(km / 1e6).toFixed(2)} × 10⁶ km`;
  }
  return `${Math.round(km).toLocaleString("zh-CN")} km`;
}

function updateReadouts() {
  state.massSolar = valueOrPower(ui.mass);
  state.accretion = valueOrPower(ui.accretion);
  state.exposure = Number(ui.exposure.value);
  state.timeScale = Number(ui.timeScale.value);
  state.quality = Number(ui.quality.value);

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
  state.mode = mode;
  const science = mode === 0;
  ui.modeScience.classList.toggle("is-active", science);
  ui.modeScience.setAttribute("aria-pressed", String(science));
  ui.modeHubble.classList.toggle("is-active", !science);
  ui.modeHubble.setAttribute("aria-pressed", String(!science));
  state.needsRender = true;
}

function setMotion(running) {
  state.running = running;
  ui.toggleMotion.dataset.state = running ? "running" : "paused";
  ui.toggleMotion.setAttribute("aria-pressed", String(!running));
  ui.toggleMotion.setAttribute("aria-label", running ? "暂停物理轨道" : "继续物理轨道");
  ui.toggleMotion.setAttribute("title", running ? "暂停物理轨道" : "继续物理轨道");
  const mark = ui.toggleMotion.querySelector("span");
  if (mark) {
    mark.textContent = running ? "Ⅱ" : "▶";
  }
  state.needsRender = true;
}

function resetView() {
  state.phase = 0.55;
  state.orbitTilt = 0.42;
  state.distance = 50;
  state.dynamicScale = 1;
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
  hideInteractionHint();
}

function bindInteractions() {
  canvas.addEventListener("pointerdown", (event) => {
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
        state.distance = clamp(state.distance * (state.pinchDistance / separation), 34, 90);
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
      event.preventDefault();
      state.distance = clamp(state.distance * Math.exp(event.deltaY * 0.0008), 34, 90);
      beginUserHold(1800);
      updateReadouts();
    },
    { passive: false },
  );

  canvas.addEventListener("dblclick", resetView);
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());

  window.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLButtonElement) {
      return;
    }
    let handled = true;
    if (event.key === "ArrowLeft") state.phase += 0.06;
    else if (event.key === "ArrowRight") state.phase -= 0.06;
    else if (event.key === "ArrowUp") state.orbitTilt = clamp(state.orbitTilt - 0.05, -1.46, 1.46);
    else if (event.key === "ArrowDown") state.orbitTilt = clamp(state.orbitTilt + 0.05, -1.46, 1.46);
    else if (event.key === "0") state.orbitTilt = 0;
    else if (event.key === "+" || event.key === "=") state.distance = clamp(state.distance - 1.5, 34, 90);
    else if (event.key === "-" || event.key === "_") state.distance = clamp(state.distance + 1.5, 34, 90);
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

  return {
    cameraPos,
    forward,
    right,
    up,
    observerVelocity: tangent,
    observerBeta: 1 / Math.sqrt(state.distance - 2),
  };
}

function effectiveRenderScale() {
  const deviceScale = Math.min(window.devicePixelRatio || 1, 2);
  let scale = deviceScale * state.quality * state.dynamicScale;
  // Let the 1.25x quality setting supersample a Retina canvas on GPUs with
  // spare headroom.  The pixel budget and feedback governor below still keep
  // the default path at native display density on M3/M4 Macs.
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
  return {
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
    bloom: state.mode === 1 ? 0.06 : 0,
    motion: state.running ? 1 : 0,
    frame: state.frame,
    observerVelocity: camera.observerVelocity,
    observerBeta: camera.observerBeta,
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
  [ui.mass, ui.accretion, ui.exposure, ui.timeScale].forEach((input) => {
    input.addEventListener("input", updateReadouts);
  });
  ui.quality.addEventListener("input", () => {
    updateReadouts();
    state.dynamicScale = 1;
    state.resizePending = true;
  });
  ui.modeScience.addEventListener("click", () => setMode(0));
  ui.modeHubble.addEventListener("click", () => setMode(1));
  ui.toggleMotion.addEventListener("click", () => setMotion(!state.running));
  ui.resetView.addEventListener("click", resetView);
  ui.togglePanel.addEventListener("click", () => {
    const expanded = !ui.panel.classList.contains("is-open");
    ui.panel.classList.toggle("is-open", expanded);
    ui.togglePanel.setAttribute("aria-expanded", String(expanded));
    ui.togglePanel.setAttribute("aria-label", expanded ? "收起观测参数" : "展开观测参数");
  });
  window.addEventListener("resize", () => {
    state.resizePending = true;
    state.needsRender = true;
  });
  document.addEventListener("visibilitychange", () => {
    lastFrameTime = performance.now();
  });
}

function showFatalError(message) {
  ui.backendStatus.textContent = "初始化失败";
  const error = document.createElement("div");
  error.className = "fatal-error";
  error.innerHTML = `<strong>无法启动 GPU 渲染器</strong><span>${String(message)}</span>`;
  app.append(error);
}

let lastFrameTime = performance.now();

function animate(now) {
  const frameElapsed = Math.max((now - lastFrameTime) / 1000, 0);
  // Clamp only the physical simulation delta after a long stall.  FPS must use
  // the real wall-clock duration or the quality governor overestimates slow
  // frames and never reaches its lower compatibility tiers.
  const dt = Math.min(frameElapsed, 0.1);
  lastFrameTime = now;

  if (!document.hidden) {
    if (state.running && !state.dragging && now >= state.userHoldUntil) {
      const omega = 1 / state.distance ** 1.5;
      state.phase += omega * dt * state.timeScale;
    }
    if (state.running) {
      state.time += dt * state.timeScale;
    }

    if (state.resizePending) {
      resizeRenderer();
    }
    if (state.running || state.dragging || state.needsRender) {
      renderer.render(frameParameters());
      state.frame = (state.frame + 1) % 16_777_216;
      state.needsRender = false;
      updateFps(frameElapsed);
      adaptQuality(now);
    }
  }

  requestAnimationFrame(animate);
}

async function start() {
  bindUi();
  updateReadouts();
  setMode(state.mode);
  setMotion(true);

  try {
    renderer = await createRenderer();
    ui.backendStatus.textContent = renderer.backend;
    ui.gpuStatus.textContent = renderer.gpu;
    const updateOutputStatus = () => {
      ui.hdrStatus.textContent = renderer.hdrMode;
      ui.hdrStatus.title = `${renderer.outputDescription} · ${renderer.skyDetail}`;
    };
    updateOutputStatus();
    renderer.onSkyChanged = () => {
      updateOutputStatus();
      state.needsRender = true;
    };
    const dynamicRange = matchMedia("(dynamic-range: high)");
    dynamicRange.addEventListener?.("change", () => {
      updateOutputStatus();
      state.needsRender = true;
    });
    if (rendererFallbackReason) {
      ui.backendStatus.title = `WebGPU 回退原因：${rendererFallbackReason}`;
    }
    bindInteractions();
    resizeRenderer(true);
    app.classList.add("is-ready");
    lastFrameTime = performance.now();
    requestAnimationFrame(animate);
  } catch (error) {
    console.error(error);
    showFatalError(error instanceof Error ? error.message : error);
  }
}

start();
