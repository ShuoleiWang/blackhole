import { createStrongFieldOrbitRuntime } from "../strong-field-orbit.js";
import { strongFieldBinaryShaderBundle } from "../strong-field-shaders.js";
import { loadBinaryDynamics } from "./binary-dynamics-adapter.js";
import { createPlaybackClock } from "./binary-playback-clock.js";

const MANIFEST_URL = new URL(
  "../../assets/scenes/binary-sxs-bbh-0001-v2.json",
  import.meta.url,
);
const DEG = 180 / Math.PI;
const WAVEFORM_WIDTH = 280;
const MAX_STRONG_FIELD_STEPS = 320;
const STRONG_FIELD_ACCUMULATION_MODE = "linear-hdr-running-average-v1";
const STRONG_FIELD_TIER_POLICY = Object.freeze({
  emergency: Object.freeze({
    integrator: Object.freeze([0.065, 4.40, 2.7, 0.34]),
    escapeRadiusM: 56,
    maximumLookbackM: 164,
    maximumCriticalBonus: 268,
    stepCurveExponent: 0.50,
    capturePaddingM: 0.30,
  }),
  survival: Object.freeze({
    integrator: Object.freeze([0.050, 3.50, 3.0, 0.25]),
    escapeRadiusM: 60,
    maximumLookbackM: 180,
    maximumCriticalBonus: 256,
    stepCurveExponent: 0.65,
    capturePaddingM: 0.24,
  }),
  interactive: Object.freeze({
    integrator: Object.freeze([0.035, 3.00, 3.3, 0.18]),
    escapeRadiusM: 64,
    maximumLookbackM: 200,
    maximumCriticalBonus: 224,
    stepCurveExponent: 0.80,
    capturePaddingM: 0.16,
  }),
  balanced: Object.freeze({
    integrator: Object.freeze([0.018, 1.10, 3.6, 0.10]),
    escapeRadiusM: 80,
    maximumLookbackM: 220,
    maximumCriticalBonus: 160,
    stepCurveExponent: 1.50,
    capturePaddingM: 0.08,
  }),
  fine: Object.freeze({
    integrator: Object.freeze([0.010, 0.58, 4.0, 0.05]),
    escapeRadiusM: 96,
    maximumLookbackM: 240,
    maximumCriticalBonus: 64,
    stepCurveExponent: 1.90,
    capturePaddingM: 0.04,
  }),
});
const SCRUB_KEYS = new Set([
  "ArrowLeft",
  "ArrowRight",
  "ArrowUp",
  "ArrowDown",
  "PageUp",
  "PageDown",
  "Home",
  "End",
]);

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
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

function waveformPath(evaluate, firstTime, finalTime) {
  const height = 46;
  const padding = 3;
  const duration = finalTime - firstTime;
  const plotTimes = Array.from({ length: 1600 }, (_, index) => (
    firstTime + index * duration / 1599
  ));
  if (firstTime < 0 && finalTime > 0) {
    plotTimes.push(0);
  }
  plotTimes.sort((first, second) => first - second);
  const samples = plotTimes.map(evaluate);
  const amplitude = Math.max(
    ...samples.map((sample) => Math.abs(sample.waveform.h22Real)),
    1e-6,
  );
  return samples.map((sample, index) => {
    const x = WAVEFORM_WIDTH * (sample.tM - firstTime) / duration;
    const y = padding + (height - 2 * padding)
      * (0.5 - 0.46 * sample.waveform.h22Real / amplitude);
    return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(" ");
}

function regimeLabel(sample) {
  if (sample.regime === "nr-inspiral") {
    return "SXS NR 螺旋靠近";
  }
  if (sample.regime === "nr-horizon-gap") {
    return "A/B 轨迹结束 · 事件间隙";
  }
  if (sample.regime === "nr-merger") {
    return "共同视界形成";
  }
  return "余留体 ringdown";
}

function requiredElement(documentRef, id) {
  const element = documentRef.getElementById(id);
  if (!element) {
    throw new Error(`Binary scene requires interface element #${id}`);
  }
  return element;
}

function formatProtocolTime(timeM) {
  const normalized = Math.abs(timeM) < 0.0005 ? 0 : timeM;
  const digits = Math.abs(normalized) >= 1000 ? 0 : 2;
  return `t = ${normalized.toFixed(digits).replace("-", "−")} M`;
}

function snapshotElement(element, { content = true } = {}) {
  return {
    attributes: Array.from(
      element.attributes,
      (attribute) => [attribute.name, attribute.value],
    ),
    innerHTML: content ? element.innerHTML : null,
    value: "value" in element ? element.value : null,
    hidden: "hidden" in element ? element.hidden : null,
  };
}

function restoreElement(element, snapshot) {
  if (snapshot.innerHTML !== null) {
    element.innerHTML = snapshot.innerHTML;
  }
  for (const attribute of Array.from(element.attributes)) {
    element.removeAttribute(attribute.name);
  }
  for (const [name, value] of snapshot.attributes) {
    element.setAttribute(name, value);
  }
  if (snapshot.value !== null) {
    element.value = snapshot.value;
  }
  if (snapshot.hidden !== null) {
    element.hidden = snapshot.hidden;
  }
}

export async function createBinaryApproxScene({
  document: documentRef,
  ui,
  state,
  formatMass,
  formatGravitationalRadius,
  controls,
}) {
  if (
    typeof controls?.setRunning !== "function"
    || typeof controls?.requestRender !== "function"
  ) {
    throw new Error("Binary scene requires host playback controls");
  }

  const track = await loadBinaryDynamics(MANIFEST_URL);
  const manifest = track.manifest;
  const evaluate = track.sampleAt;
  const strongFieldRuntime = createStrongFieldOrbitRuntime({ track });
  const firstTime = track.firstTimeM;
  const finalTime = track.finalTimeM;
  const sceneQuery = new URLSearchParams(
    documentRef.defaultView?.location?.search || "",
  );
  const requestedInitialTime = sceneQuery.get("binaryTime");
  const parsedInitialTime = Number(requestedInitialTime);
  const initialTime = (
    requestedInitialTime !== null
    && requestedInitialTime !== ""
    && Number.isFinite(parsedInitialTime)
  )
    ? clamp(parsedInitialTime, firstTime, finalTime)
    : firstTime;
  const startsRunning = sceneQuery.get("paused") !== "1";
  const durationM = finalTime - firstTime;
  const defaults = manifest.rendererDefaults;
  const initialViewLatitude = (
    90 - defaults.initialViewingInclinationDeg
  ) / DEG;
  const binaryMasses = [
    manifest.physicalSystem.bodies[0].massFraction,
    manifest.physicalSystem.bodies[1].massFraction,
    manifest.physicalSystem.remnant.massFraction,
    0,
  ];
  const defaultTimeScale = (
    durationM / manifest.playback.cycleDurationSecondsAtNominalRate
  );
  const playbackClock = createPlaybackClock({
    firstTimeM: firstTime,
    finalTimeM: finalTime,
    endHoldSeconds: manifest.playback.endHoldSeconds,
    loop: manifest.playback.loop,
    slowMotion: manifest.playback.slowMotion,
  });

  const elements = {
    eyebrow: requiredElement(documentRef, "sceneEyebrow"),
    title: requiredElement(documentRef, "panelTitle"),
    observerLabel: requiredElement(documentRef, "observerLabel"),
    observerValue: ui.observerValue,
    radiusLabel: requiredElement(documentRef, "radiusLabel"),
    radiusValue: ui.rsValue,
    shadowLabel: requiredElement(documentRef, "shadowLabel"),
    shadowValue: ui.shadowValue,
    massLabel: requiredElement(documentRef, "massLabel"),
    massValue: ui.massValue,
    accretionControl: requiredElement(documentRef, "accretionControl"),
    accretionValue: ui.accretionValue,
    exposureValue: ui.exposureValue,
    timeScaleValue: ui.timeScaleValue,
    qualityValue: ui.qualityValue,
    modeScience: requiredElement(documentRef, "modeScience"),
    modeOutcome: requiredElement(documentRef, "modeHubble"),
    modeFrequency: requiredElement(documentRef, "modeFrequency"),
    modeLookback: requiredElement(documentRef, "modeLookback"),
    modeNull: requiredElement(documentRef, "modeNull"),
    modeCost: requiredElement(documentRef, "modeError"),
    advancedDiagnostics: requiredElement(
      documentRef,
      "transferAdvancedDiagnostics",
    ),
    physicsNote: requiredElement(documentRef, "physicsNote"),
    sceneStatus: requiredElement(documentRef, "sceneStatus"),
    binaryTimeline: requiredElement(documentRef, "binaryTimeline"),
    binaryRegime: requiredElement(documentRef, "binaryRegime"),
    waveformLabel: requiredElement(documentRef, "binaryWaveformLabel"),
    waveformPath: requiredElement(documentRef, "binaryWaveformPath"),
    timeCursor: requiredElement(documentRef, "binaryTimeCursor"),
    playPause: requiredElement(documentRef, "binaryPlayPause"),
    scrubber: requiredElement(documentRef, "binaryTimeScrubber"),
    timeValue: requiredElement(documentRef, "binaryTimeValue"),
    slowMotion: requiredElement(documentRef, "binarySlowMotion"),
    playbackRate: requiredElement(documentRef, "binaryPlaybackRate"),
    desktopHint: requiredElement(documentRef, "desktopHint"),
  };
  const original = {
    documentTitle: documentRef.title,
    rootHadClass: documentRef.documentElement.classList.contains(
      "scene-binary-approx",
    ),
    elements: new Map(
      Object.values(elements).map((element) => [
        element,
        snapshotElement(element, {
          content: (
            element !== elements.binaryTimeline
            && element !== elements.accretionControl
          ),
        }),
      ]),
    ),
    accretionDisabled: ui.accretion.disabled,
    exposureValue: ui.exposure.value,
    timeScale: {
      min: ui.timeScale.min,
      max: ui.timeScale.max,
      step: ui.timeScale.step,
      value: ui.timeScale.value,
    },
    state: {
      time: state.time,
      distance: state.distance,
      phase: state.phase,
      orbitTilt: state.orbitTilt,
      exposure: state.exposure,
      timeScale: state.timeScale,
    },
  };

  let abortController = null;
  let initialized = false;
  let scrubbing = false;
  let resumeAfterScrub = false;
  let slowMotionEnabled = manifest.playback.slowMotion.enabledByDefault;
  let playbackHolding = false;
  let rateAwaitingAdvance = false;
  let actualRateMPerSecond = 0;
  let transportRevision = 0;
  let transportSignature = null;

  const diagnosticElements = Object.freeze([
    elements.modeOutcome,
    elements.modeFrequency,
    elements.modeLookback,
    elements.modeNull,
    elements.modeCost,
    elements.advancedDiagnostics,
  ]);

  function bumpTransportRevision() {
    transportRevision = (transportRevision + 1) % Number.MAX_SAFE_INTEGER;
  }

  function revisionVector(value) {
    if (value == null) {
      return "default";
    }
    return Array.from(value, Number).join(",");
  }

  function updateTransportRevision(frame) {
    // These controls affect a traced pixel without changing the physical
    // provider frame.  Keep the token numeric for the quality scheduler while
    // comparing the full exact signature locally.
    const signature = [
      frame.mode,
      frame.exposure,
      frame.steps,
      frame.skyRotation,
      frame.accretion,
      frame.diskOuterRadius,
      revisionVector(frame.sceneStrongIntegrator),
      revisionVector(frame.sceneStrongDomain),
      revisionVector(frame.sceneStrongDiagnostics),
    ].join("|");
    if (signature !== transportSignature) {
      transportSignature = signature;
      bumpTransportRevision();
    }
    return transportRevision;
  }

  function updateMotionButton(running) {
    const action = running
      ? "暂停双黑洞时间线"
      : "继续双黑洞时间线";
    elements.playPause.setAttribute("aria-label", action);
    elements.playPause.setAttribute("title", action);
    elements.playPause.setAttribute("aria-pressed", String(!running));
    const mark = elements.playPause.querySelector("span");
    if (mark) {
      mark.textContent = running ? "Ⅱ" : "▶";
    }
  }

  function configureDiagnosticControls(strongWebGPU) {
    elements.modeScience.textContent = strongWebGPU
      ? "天空成像"
      : "弱场预览";
    elements.modeOutcome.textContent = "光线结果";
    elements.modeOutcome.setAttribute(
      "title",
      "蓝色为捕获，绿色为逃逸，洋红为未收敛；分类来自当前 WebGPU 光线积分",
    );
    elements.modeFrequency.textContent = "频移因子 g";
    elements.modeFrequency.setAttribute(
      "title",
      "观测频率与无穷远频率之比 g；仅对已逃逸光线有物理意义",
    );
    elements.modeLookback.textContent = "坐标回溯时间";
    elements.modeLookback.setAttribute(
      "title",
      "沿 fast-light 切片积分的坐标时间，不是可观测的相对到达时延",
    );
    elements.modeNull.textContent = "零性 / H 残差";
    elements.modeNull.setAttribute(
      "title",
      "沿光线记录的最大归一化零 Hamiltonian 残差",
    );
    elements.modeCost.textContent = "积分步数成本";
    elements.modeCost.setAttribute(
      "title",
      "已执行积分步数相对 320 步编译上限的比例；这是计算成本，不是物理量",
    );
    for (const element of diagnosticElements) {
      element.hidden = !strongWebGPU;
    }
  }

  function setRendererStatus(strongWebGPU, capabilities, rendererView) {
    elements.sceneStatus.classList.remove(
      "is-strong-field",
      "is-fallback",
    );
    if (strongWebGPU) {
      elements.sceneStatus.classList.add("is-strong-field");
      elements.sceneStatus.textContent = [
        rendererView?.backend || capabilities.backend || "WebGPU",
        "实时 3+1 Hamiltonian 强场光追",
        "boosted superposed Kerr–Schild",
        "fast-light 近似 · 非完整 NR",
        "诊断：光线结果 / 回溯时间 / g / 零性残差 / 积分成本",
      ].join(" · ");
      return;
    }
    elements.sceneStatus.classList.add("is-fallback");
    elements.sceneStatus.textContent = [
      rendererView?.backend || capabilities.backend || "WebGL2",
      "兼容性回退",
      "旧 two-centre weak-field 预览",
      "不具备 WebGPU 强场物理等价性",
      "强场 outcome / lookback / g / null residual / cost 诊断已隐藏",
    ].join(" · ");
  }

  function updateTransport(sample) {
    elements.scrubber.value = sample.tM.toFixed(6);
    const timeText = formatProtocolTime(sample.tM);
    elements.timeValue.textContent = timeText;
    elements.scrubber.setAttribute(
      "aria-valuetext",
      `${timeText}，${regimeLabel(sample)}`,
    );
    const factor = playbackClock.factorAt(
      sample.tM,
      slowMotionEnabled,
    );
    const stationary = !state.running || scrubbing || playbackHolding;
    elements.playbackRate.textContent = stationary
      ? playbackHolding && state.running
        ? "末尾停留 · 0 M/s"
        : "已暂停 · 0 M/s"
      : factor < 0.999
        ? `实际 ${actualRateMPerSecond.toFixed(1)} M/s · ${factor.toFixed(2)}×`
        : `实际 ${actualRateMPerSecond.toFixed(0)} M/s`;
    elements.slowMotion.setAttribute(
      "aria-pressed",
      String(slowMotionEnabled),
    );
    elements.slowMotion.textContent = slowMotionEnabled
      ? "合并慢放 开"
      : "合并慢放 关";
  }

  function updateDynamicReadouts(sample) {
    const phaseDegrees = (
      (sample.orbitalPhaseRad * DEG) % 360 + 360
    ) % 360;
    if (sample.individualHorizonsValid) {
      ui.observerValue.innerHTML = [
        `a<sub>coord,SXS</sub> ${sample.separationM.toFixed(2)} M`,
        `φ<sub>coord,SXS</sub> ${phaseDegrees.toFixed(0)}°`,
      ].join(" · ");
    } else if (sample.regime === "nr-horizon-gap") {
      ui.observerValue.textContent = (
        "A/B 轨迹已结束 · 共同视界事件尚未发生"
      );
    } else if (sample.renderTopologyBlend < 0.995) {
      ui.observerValue.textContent = "共同视界形成 · A/B 轨迹已结束";
    } else {
      ui.observerValue.textContent = "单一 SXS 余留体";
    }
    ui.shadowValue.textContent = regimeLabel(sample);
    elements.binaryRegime.textContent = [
      regimeLabel(sample),
      `|rh₂₂| ${sample.waveform.amplitude.toFixed(3)}`,
    ].join(" · ");
    const cursor = (
      WAVEFORM_WIDTH * clamp(sample.timelineFraction, 0, 1)
    ).toFixed(2);
    elements.timeCursor.setAttribute("x1", cursor);
    elements.timeCursor.setAttribute("x2", cursor);
    updateTransport(sample);
  }

  function seekFromScrubber() {
    state.time = playbackClock.seek(Number(elements.scrubber.value));
    playbackHolding = false;
    rateAwaitingAdvance = true;
    actualRateMPerSecond = 0;
    bumpTransportRevision();
    updateDynamicReadouts(evaluate(state.time));
    controls.requestRender();
  }

  function beginScrub() {
    if (scrubbing) {
      return;
    }
    scrubbing = true;
    resumeAfterScrub = state.running;
    rateAwaitingAdvance = true;
    actualRateMPerSecond = 0;
    bumpTransportRevision();
    controls.setRunning(false);
  }

  function endScrub() {
    if (!scrubbing) {
      return;
    }
    scrubbing = false;
    bumpTransportRevision();
    const shouldResume = resumeAfterScrub;
    resumeAfterScrub = false;
    if (shouldResume) {
      controls.setRunning(true);
    }
  }

  function bindPlaybackControls() {
    abortController?.abort();
    abortController = new AbortController();
    const options = { signal: abortController.signal };
    elements.playPause.addEventListener("click", () => {
      controls.setRunning(!state.running);
    }, options);
    elements.slowMotion.addEventListener("click", () => {
      slowMotionEnabled = !slowMotionEnabled;
      bumpTransportRevision();
      if (
        state.running
        && !scrubbing
        && !playbackHolding
        && !rateAwaitingAdvance
      ) {
        actualRateMPerSecond = (
          state.timeScale
          * playbackClock.factorAt(state.time, slowMotionEnabled)
        );
      }
      updateDynamicReadouts(evaluate(state.time));
      controls.requestRender();
    }, options);
    elements.scrubber.addEventListener("pointerdown", beginScrub, options);
    elements.scrubber.addEventListener("input", () => {
      if (!scrubbing) {
        beginScrub();
      }
      seekFromScrubber();
    }, options);
    elements.scrubber.addEventListener("change", endScrub, options);
    elements.scrubber.addEventListener("blur", endScrub, options);
    elements.scrubber.addEventListener("keydown", (event) => {
      if (SCRUB_KEYS.has(event.key)) {
        beginScrub();
      }
    }, options);
    elements.scrubber.addEventListener("keyup", (event) => {
      if (SCRUB_KEYS.has(event.key)) {
        endScrub();
      }
    }, options);
    documentRef.defaultView?.addEventListener("pointerup", endScrub, options);
    documentRef.defaultView?.addEventListener(
      "pointercancel",
      endScrub,
      options,
    );
  }

  return Object.freeze({
    id: "binary-approx",
    qualityPolicy: Object.freeze({
      id: "m3-pro-strong-field-v1",
    }),
    cameraDistanceLimits: Object.freeze({
      min: 34,
      max: 70,
    }),
    motionLabels: Object.freeze({
      pause: "暂停双黑洞时间线",
      resume: "继续双黑洞时间线",
    }),
    startsRunning,
    rendererOptions: Object.freeze({
      shaderBundle: strongFieldBinaryShaderBundle,
    }),
    manifest,

    onRendererReady(capabilities, rendererView) {
      if (
        !capabilities
        || typeof capabilities !== "object"
        || !["webgpu", "webgl2"].includes(capabilities.api)
      ) {
        throw new Error(
          "Binary scene requires explicit WebGPU or WebGL2 capabilities",
        );
      }
      if (
        rendererView?.capabilities
        && rendererView.capabilities !== capabilities
      ) {
        throw new Error("Binary renderer capability views must be identical");
      }
      const strongWebGPU = capabilities.api === "webgpu";
      if (
        strongWebGPU
        && capabilities.progressiveAccumulation
          !== STRONG_FIELD_ACCUMULATION_MODE
      ) {
        throw new Error(
          "Binary WebGPU strong-field path requires linear-HDR accumulation",
        );
      }
      configureDiagnosticControls(strongWebGPU);
      setRendererStatus(
        strongWebGPU,
        capabilities,
        rendererView,
      );
    },

    initialize() {
      if (initialized) {
        return;
      }
      initialized = true;
      scrubbing = false;
      resumeAfterScrub = false;
      slowMotionEnabled = manifest.playback.slowMotion.enabledByDefault;
      playbackHolding = false;
      rateAwaitingAdvance = false;
      transportSignature = null;
      bumpTransportRevision();
      documentRef.documentElement.classList.add("scene-binary-approx");
      documentRef.title = "实时双黑洞 · 深空观测台";
      state.time = playbackClock.seek(initialTime);
      state.distance = defaults.observerRadiusM;
      state.phase = 0.58;
      state.orbitTilt = initialViewLatitude;
      ui.exposure.value = defaults.exposure.toFixed(2);
      state.exposure = Number(ui.exposure.value);
      ui.timeScale.min = "0";
      ui.timeScale.max = (Math.ceil(defaultTimeScale * 2 / 10) * 10).toString();
      ui.timeScale.step = "5";
      ui.timeScale.value = (Math.round(defaultTimeScale / 5) * 5).toString();
      state.timeScale = Number(ui.timeScale.value);
      actualRateMPerSecond = (
        state.timeScale * playbackClock.factorAt(
          state.time,
          slowMotionEnabled,
        )
      );

      elements.eyebrow.textContent = "实时强场光追 · SXS 锚定";
      elements.title.textContent = "实时双黑洞";
      elements.observerLabel.textContent = "SXS 坐标证据（不驱动光追）";
      elements.radiusLabel.textContent = "1 M（GM/c²）";
      elements.shadowLabel.textContent = "数据区段";
      elements.massLabel.textContent = "系统总质量";
      elements.sceneStatus.hidden = false;
      elements.sceneStatus.textContent = [
        "WebGPU 强场生产路径",
        "boosted superposed Kerr–Schild",
        "SXS h₂₂ / 合并事件锚定",
        "fast-light 近似 · 非完整 NR",
        "WebGL2 回退为旧弱场",
      ].join(" · ");
      elements.binaryTimeline.hidden = false;
      elements.waveformLabel.innerHTML = (
        "SXS Extrapolated N=2 · (ℓ,m)=(2,2) · "
        + "Re[r h<sub>22</sub> / M]"
      );
      elements.waveformPath.setAttribute(
        "d",
        waveformPath(evaluate, firstTime, finalTime),
      );
      elements.scrubber.min = firstTime.toFixed(6);
      elements.scrubber.max = finalTime.toFixed(6);
      elements.scrubber.step = "0.25";
      elements.desktopHint.textContent = (
        "拖动观察 · 滚轮缩放 · 拖动时间轴 · 空格暂停"
      );
      elements.physicsNote.innerHTML = [
        "波形、共同视界时刻和余留体参数锚定到 ",
        '<a href="https://doi.org/10.5281/zenodo.3273935" target="_blank" rel="noreferrer">SXS:BBH:0001 Lev5</a>',
        "。右侧显示的视界质心分离/相位是<strong>依赖规范的坐标证据，绝不进入 WebGPU 黑洞位置</strong>；",
        "实时轨道由 h₂₂ 频率与 PN/EOB-like 准圆关系生成，光线在 boosted superposed Kerr–Schild 3+1 度规中积分。",
        "这是<strong>强场 fast-light 近似，不是约束求解后的完整 NR 时空，也不是 slow-light</strong>；",
        "WebGL2 会明确退回旧 weak-field 预览。合并慢放只改变播放墙钟速度。",
      ].join("");
      elements.accretionControl.setAttribute("aria-hidden", "true");
      ui.accretion.disabled = true;
      bindPlaybackControls();
      updateMotionButton(state.running);
      updateDynamicReadouts(evaluate(state.time));
    },

    onMotionChanged(running) {
      bumpTransportRevision();
      updateMotionButton(running);
      updateTransport(evaluate(state.time));
    },

    resetState() {
      state.distance = defaults.observerRadiusM;
      state.phase = 0.58;
      state.orbitTilt = initialViewLatitude;
      bumpTransportRevision();
      updateDynamicReadouts(evaluate(state.time));
    },

    updateReadouts() {
      if (
        state.running
        && !scrubbing
        && !playbackHolding
        && !rateAwaitingAdvance
      ) {
        actualRateMPerSecond = (
          state.timeScale
          * playbackClock.factorAt(state.time, slowMotionEnabled)
        );
      }
      ui.massValue.textContent = formatMass(state.massSolar);
      ui.accretionValue.textContent = "真空";
      ui.exposureValue.textContent = `${state.exposure.toFixed(2)}×`;
      ui.timeScaleValue.textContent = `${state.timeScale.toFixed(0)} M/s`;
      ui.qualityValue.textContent = `${state.quality.toFixed(2)}×`;
      ui.rsValue.textContent = formatGravitationalRadius(state.massSolar);
      updateDynamicReadouts(evaluate(state.time));
      return true;
    },

    cameraFrame() {
      // The shader orbit lies in the x-z plane, with angular momentum along +y.
      const cosPhase = Math.cos(state.phase);
      const sinPhase = Math.sin(state.phase);
      const cosLatitude = Math.cos(state.orbitTilt);
      const sinLatitude = Math.sin(state.orbitTilt);
      const positionUnit = normalize([
        cosLatitude * cosPhase,
        sinLatitude,
        cosLatitude * sinPhase,
      ]);
      const forward = scale(positionUnit, -1);
      const right = normalize([-sinPhase, 0, cosPhase]);
      return {
        cameraPos: scale(positionUnit, state.distance),
        forward,
        right,
        up: normalize(cross(forward, right)),
        observerVelocity: [0, 0, 0],
        observerBeta: 0,
      };
    },

    advance(deltaSeconds) {
      const result = playbackClock.advance(
        state.time,
        deltaSeconds,
        state.timeScale,
        slowMotionEnabled,
      );
      state.time = result.timeM;
      actualRateMPerSecond = result.effectiveRateMPerSecond;
      playbackHolding = result.holding;
      rateAwaitingAdvance = false;
    },

    extendFrame(baseFrame) {
      const sample = evaluate(state.time);
      const strongFieldFrame = strongFieldRuntime.frameAt(sample.tM);
      updateDynamicReadouts(sample);
      return {
        ...baseFrame,
        accretion: 0,
        fov: Math.max(baseFrame.fov, defaults.fieldOfViewDeg / DEG),
        diskOuterRadius: 0,
        observerVelocity: [0, 0, 0],
        observerBeta: 0,
        sceneStrongFieldUniforms: strongFieldFrame.uniforms,
        // Deliberate WebGL2-only compatibility payload.  The WebGPU strong
        // tracer consumes sceneStrongFieldUniforms above and never reads these
        // gauge-dependent SXS centroid proxies.
        sceneBinaryState: [
          sample.separationM,
          sample.orbitalPhaseRad,
          sample.renderTopologyBlend,
          0,
        ],
        sceneBinaryMasses: binaryMasses,
      };
    },

    applyStrongFieldQuality(frame, decision) {
      const tierId = (
        decision?.qualityTierId
        ?? frame?.strongFieldQuality?.tierId
        ?? "balanced"
      );
      const tier = (
        STRONG_FIELD_TIER_POLICY[tierId]
        ?? STRONG_FIELD_TIER_POLICY.balanced
      );
      const baseBudget = clamp(
        Math.trunc(Number(frame?.steps) || 0),
        0,
        MAX_STRONG_FIELD_STEPS,
      );
      const criticalBonus = Math.max(
        0,
        Math.min(
          tier.maximumCriticalBonus,
          MAX_STRONG_FIELD_STEPS - baseBudget,
        ),
      );
      const cameraRadius = Number(frame?.cameraRadius);
      if (!Number.isFinite(cameraRadius) || cameraRadius <= 0) {
        throw new Error(
          "Binary strong-field quality requires a positive camera radius",
        );
      }
      const escapeRadius = Math.max(
        tier.escapeRadiusM,
        cameraRadius + 8,
      );
      const maximumLookback = Math.max(
        tier.maximumLookbackM,
        2 * escapeRadius + 32,
      );
      return {
        ...frame,
        // Coarser settled/far-field steps let interaction rays leave the
        // tier-specific finite domain within their smaller budget. The escape
        // sphere always remains at least 8M outside the current observer.
        // Horizon/photon-region
        // accuracy tightens monotonically through the fine tier.
        sceneStrongIntegrator: tier.integrator,
        sceneStrongDomain: Object.freeze([
          escapeRadius,
          maximumLookback,
          tier.capturePaddingM,
          criticalBonus,
        ]),
        sceneStrongDiagnostics: Object.freeze([
          4,
          180,
          0.22,
          tier.stepCurveExponent,
        ]),
      };
    },

    renderRevision(frame) {
      const physics = Number(frame?.time ?? state.time);
      if (!Number.isFinite(physics)) {
        throw new Error("Binary strong-field physics revision must be finite");
      }
      return Object.freeze({
        physics,
        transport: updateTransportRevision(frame ?? {}),
      });
    },

    dispose() {
      abortController?.abort();
      abortController = null;
      initialized = false;
      scrubbing = false;
      resumeAfterScrub = false;
      slowMotionEnabled = manifest.playback.slowMotion.enabledByDefault;
      playbackHolding = false;
      rateAwaitingAdvance = false;
      actualRateMPerSecond = 0;
      transportSignature = null;
      bumpTransportRevision();
      elements.sceneStatus.classList.remove(
        "is-strong-field",
        "is-fallback",
      );
      if (!original.rootHadClass) {
        documentRef.documentElement.classList.remove("scene-binary-approx");
      }
      documentRef.title = original.documentTitle;
      for (const [element, snapshot] of original.elements) {
        restoreElement(element, snapshot);
      }
      ui.accretion.disabled = original.accretionDisabled;
      ui.exposure.value = original.exposureValue;
      Object.assign(ui.timeScale, original.timeScale);
      Object.assign(state, original.state);
      controls.requestRender();
    },
  });
}
