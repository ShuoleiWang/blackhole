import { createStrongFieldOrbitRuntime } from "../strong-field-orbit.js";
import { strongFieldBinaryShaderBundle } from "../strong-field-shaders.js";
import { createI18n } from "../i18n.js";
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
    integrator: Object.freeze([0.065, 3.50, 2.7, 0.34]),
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
    integrator: Object.freeze([0.010, 0.85, 4.0, 0.05]),
    escapeRadiusM: 80,
    maximumLookbackM: 220,
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

function regimeLabel(sample, i18n) {
  if (sample.regime === "nr-inspiral") {
    return i18n.t("binary.regime.inspiral");
  }
  if (sample.regime === "nr-horizon-gap") {
    return i18n.t("binary.regime.gap");
  }
  if (sample.regime === "nr-merger") {
    return i18n.t("binary.regime.merger");
  }
  return i18n.t("binary.regime.ringdown");
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
  i18n: providedI18n,
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
  const i18n = providedI18n ?? createI18n(sceneQuery);
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

  const advancedDiagnosticElements = Object.freeze([
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
      ? i18n.t("binary.pauseTimeline")
      : i18n.t("binary.resumeTimeline");
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
      ? i18n.t("binary.mode.sky")
      : i18n.t("binary.mode.weakField");
    // Outcome classification and frequency shift remain part of the traced
    // ray record and the scientific reference workbench, but they are not
    // useful as primary display modes in the interactive binary scene.
    elements.modeOutcome.hidden = true;
    elements.modeFrequency.hidden = true;
    elements.modeOutcome.textContent = i18n.t("binary.mode.outcome");
    elements.modeOutcome.setAttribute(
      "title",
      i18n.t("binary.mode.outcomeTitle"),
    );
    elements.modeFrequency.textContent = i18n.t("binary.mode.frequency");
    elements.modeFrequency.setAttribute(
      "title",
      i18n.t("binary.mode.frequencyTitle"),
    );
    elements.modeLookback.textContent = i18n.t("binary.mode.lookback");
    elements.modeLookback.setAttribute(
      "title",
      i18n.t("binary.mode.lookbackTitle"),
    );
    elements.modeNull.textContent = i18n.t("binary.mode.null");
    elements.modeNull.setAttribute(
      "title",
      i18n.t("binary.mode.nullTitle"),
    );
    elements.modeCost.textContent = i18n.t("binary.mode.cost");
    elements.modeCost.setAttribute(
      "title",
      i18n.t("binary.mode.costTitle"),
    );
    for (const element of advancedDiagnosticElements) {
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
        i18n.t("binary.status.strongTrace"),
        "boosted superposed Kerr–Schild",
        i18n.t("binary.status.fastLight"),
        i18n.t("binary.status.advanced"),
      ].join(" · ");
      return;
    }
    elements.sceneStatus.classList.add("is-fallback");
    elements.sceneStatus.textContent = [
      rendererView?.backend || capabilities.backend || "WebGL2",
      i18n.t("binary.status.compatibility"),
      i18n.t("binary.status.legacy"),
      i18n.t("binary.status.noParity"),
      i18n.t("binary.status.hiddenDiagnostics"),
    ].join(" · ");
  }

  function updateTransport(sample) {
    elements.scrubber.value = sample.tM.toFixed(6);
    const timeText = formatProtocolTime(sample.tM);
    elements.timeValue.textContent = timeText;
    elements.scrubber.setAttribute(
      "aria-valuetext",
      i18n.t("binary.ariaTime", {
        time: timeText,
        regime: regimeLabel(sample, i18n),
      }),
    );
    const factor = playbackClock.factorAt(
      sample.tM,
      slowMotionEnabled,
    );
    const stationary = !state.running || scrubbing || playbackHolding;
    elements.playbackRate.textContent = stationary
      ? playbackHolding && state.running
        ? i18n.t("binary.playback.endHold")
        : i18n.t("binary.playback.paused")
      : factor < 0.999
        ? i18n.t("binary.playback.actualWithFactor", {
          rate: actualRateMPerSecond.toFixed(1),
          factor: factor.toFixed(2),
        })
        : i18n.t("binary.playback.actual", {
          rate: actualRateMPerSecond.toFixed(0),
        });
    elements.slowMotion.setAttribute(
      "aria-pressed",
      String(slowMotionEnabled),
    );
    elements.slowMotion.textContent = slowMotionEnabled
      ? i18n.t("binary.playback.slowOn")
      : i18n.t("binary.playback.slowOff");
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
        i18n.t("binary.readout.gap")
      );
    } else if (sample.renderTopologyBlend < 0.995) {
      ui.observerValue.textContent = i18n.t("binary.readout.horizon");
    } else {
      ui.observerValue.textContent = i18n.t("binary.readout.remnant");
    }
    ui.shadowValue.textContent = regimeLabel(sample, i18n);
    elements.binaryRegime.textContent = [
      regimeLabel(sample, i18n),
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
      pause: i18n.t("binary.pauseTimeline"),
      resume: i18n.t("binary.resumeTimeline"),
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
      documentRef.title = i18n.t("binary.documentTitle");
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

      elements.eyebrow.textContent = i18n.t("binary.eyebrow");
      elements.title.textContent = i18n.t("binary.title");
      elements.observerLabel.textContent = i18n.t("binary.observerLabel");
      elements.radiusLabel.textContent = i18n.t("binary.radiusLabel");
      elements.shadowLabel.textContent = i18n.t("binary.segmentLabel");
      elements.massLabel.textContent = i18n.t("binary.massLabel");
      elements.sceneStatus.hidden = false;
      elements.sceneStatus.textContent = [
        i18n.t("binary.initialStatus.strong"),
        "boosted superposed Kerr–Schild",
        i18n.t("binary.initialStatus.anchor"),
        i18n.t("binary.status.fastLight"),
        i18n.t("binary.initialStatus.fallback"),
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
        i18n.t("binary.desktopHint")
      );
      elements.physicsNote.innerHTML = i18n.t("binary.physicsHtml", {
        sourceLink: '<a href="https://doi.org/10.5281/zenodo.3273935" target="_blank" rel="noreferrer">SXS:BBH:0001 Lev5</a>',
      });
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
      ui.accretionValue.textContent = i18n.t("binary.vacuum");
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
          // Keep numerical failures inspectable in the photographic view
          // without leaking a conspicuous magenta diagnostic overlay into
          // the default image. Raw outcomes remain available to regression
          // probes and the scientific reference workbench.
          0.055,
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
