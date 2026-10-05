// Shared host for the transient scenes. It borrows the shared observatory
// panel the way the binary scene does: the timeline section shows the bolometric light curve instead of a
// strain preview, the scrubber runs on the presentation timeline (linear and
// logarithmic segments), and every element is restored on dispose.
//
// A scene supplies:
//   id, rootClass, i18nPrefix, timeline (createPresentationTimeline),
//   evaluate(timeS) -> sample with at least
//     { timeS, scaleKm, lightCurveLog10L, regimeKey, readouts: { observer,
//       radius, segment }, exposureKey }
//   lightCurve: { timesS, log10LuminosityErgS } for the plot,
//   cameraDefaults: { distance, phase, latitude }, cameraDistanceLimits,
//   extendFrame(baseFrame, sample, host) -> frame,
//   modes: [{ label, title }] x3 (i18n keys; photographic + two false colour),
//   rendererOptions, qualityPolicy, applyStrongFieldQuality, onRendererReady.

import { SOLAR_LUMINOSITY_ERG_S } from "../transients/physical-constants.js";

const PLOT_WIDTH = 280;
const PLOT_HEIGHT = 46;
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

function requiredElement(documentRef, id) {
  const element = documentRef.getElementById(id);
  if (!element) {
    throw new Error(`Transient scene requires interface element #${id}`);
  }
  return element;
}

function snapshotElement(element, { content = true } = {}) {
  return {
    attributes: Array.from(element.attributes, (attribute) => [attribute.name, attribute.value]),
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

const SUPERSCRIPT = {
  "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
};

export function scientific(value, digits = 2) {
  if (!Number.isFinite(value) || value === 0) {
    return "0";
  }
  const exponent = Math.floor(Math.log10(Math.abs(value)));
  const mantissa = value / 10 ** exponent;
  if (exponent >= -2 && exponent <= 4) {
    return value.toLocaleString("en-US", { maximumSignificantDigits: digits + 1 });
  }
  const power = String(exponent).split("").map((character) => SUPERSCRIPT[character]).join("");
  return `${mantissa.toFixed(digits)} × 10${power}`;
}

export function formatDuration(seconds, i18n) {
  const sign = seconds < 0 ? "−" : "";
  const magnitude = Math.abs(seconds);
  if (magnitude < 1e-3) {
    return `t = ${sign}${(magnitude * 1e6).toFixed(0)} μs`;
  }
  if (magnitude < 1) {
    return `t = ${sign}${(magnitude * 1e3).toFixed(magnitude < 0.01 ? 2 : 1)} ms`;
  }
  if (magnitude < 120) {
    return `t = ${sign}${magnitude.toFixed(magnitude < 10 ? 2 : 1)} s`;
  }
  if (magnitude < 7200) {
    return `t = ${sign}${(magnitude / 60).toFixed(1)} ${i18n.t("transient.unit.minutes")}`;
  }
  if (magnitude < 172800) {
    return `t = ${sign}${(magnitude / 3600).toFixed(1)} ${i18n.t("transient.unit.hours")}`;
  }
  return `t = ${sign}${(magnitude / 86400).toFixed(1)} ${i18n.t("transient.unit.days")}`;
}

export function formatLuminosity(luminosityErgS) {
  return `${scientific(luminosityErgS)} erg/s · ${scientific(luminosityErgS / SOLAR_LUMINOSITY_ERG_S)} L☉`;
}

export function createTransientSceneHost({
  document: documentRef,
  ui,
  state,
  i18n,
  controls,
  scene,
}) {
  if (typeof controls?.setRunning !== "function" || typeof controls?.requestRender !== "function") {
    throw new Error("Transient scene requires host playback controls");
  }
  const timeline = scene.timeline;
  const prefix = scene.i18nPrefix;
  const elements = {
    eyebrow: requiredElement(documentRef, "sceneEyebrow"),
    title: requiredElement(documentRef, "panelTitle"),
    observerLabel: requiredElement(documentRef, "observerLabel"),
    radiusLabel: requiredElement(documentRef, "radiusLabel"),
    shadowLabel: requiredElement(documentRef, "shadowLabel"),
    massControl: ui.mass.closest("label") || ui.mass,
    accretionControl: requiredElement(documentRef, "accretionControl"),
    exposureInput: ui.exposure,
    exposureValue: ui.exposureValue,
    timeScaleInput: ui.timeScale,
    timeScaleValue: ui.timeScaleValue,
    qualityValue: ui.qualityValue,
    observerValue: ui.observerValue,
    radiusValue: ui.rsValue,
    shadowValue: ui.shadowValue,
    modeScience: requiredElement(documentRef, "modeScience"),
    modeHubble: requiredElement(documentRef, "modeHubble"),
    modeFrequency: requiredElement(documentRef, "modeFrequency"),
    advancedDiagnostics: requiredElement(documentRef, "transferAdvancedDiagnostics"),
    physicsNote: requiredElement(documentRef, "physicsNote"),
    sceneStatus: requiredElement(documentRef, "sceneStatus"),
    timelineSection: requiredElement(documentRef, "binaryTimeline"),
    regime: requiredElement(documentRef, "binaryRegime"),
    plotLabel: requiredElement(documentRef, "binaryWaveformLabel"),
    plotPath: requiredElement(documentRef, "binaryWaveformPath"),
    timeCursor: requiredElement(documentRef, "binaryTimeCursor"),
    playPause: requiredElement(documentRef, "binaryPlayPause"),
    scrubber: requiredElement(documentRef, "binaryTimeScrubber"),
    timeValue: requiredElement(documentRef, "binaryTimeValue"),
    followToggle: requiredElement(documentRef, "binarySlowMotion"),
    playbackRate: requiredElement(documentRef, "binaryPlaybackRate"),
    desktopHint: requiredElement(documentRef, "desktopHint"),
  };
  const original = {
    documentTitle: documentRef.title,
    rootHadClass: documentRef.documentElement.classList.contains(scene.rootClass),
    rootHadShared: documentRef.documentElement.classList.contains("scene-transient"),
    elements: new Map(Object.values(elements).map((element) => [
      element,
      snapshotElement(element, { content: element !== elements.timelineSection }),
    ])),
    exposure: { min: ui.exposure.min, max: ui.exposure.max, step: ui.exposure.step, value: ui.exposure.value },
    timeScale: { min: ui.timeScale.min, max: ui.timeScale.max, step: ui.timeScale.step, value: ui.timeScale.value },
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
  let holding = false;
  let holdElapsed = 0;
  let followExpansion = true;
  let actualRate = 0;
  let revision = 0;
  let lastSample = null;
  // Camera scale (km per scene unit): follows the scene's characteristic
  // size when enabled, otherwise frozen at its last value.
  let frozenScaleKm = null;

  function bump() {
    revision = (revision + 1) % Number.MAX_SAFE_INTEGER;
  }

  function setText(element, value) {
    if (element.textContent !== value) element.textContent = value;
  }
  function setHtml(element, value) {
    if (element.innerHTML !== value) element.innerHTML = value;
  }
  function setAttribute(element, name, value) {
    const text = String(value);
    if (element.getAttribute?.(name) !== text) element.setAttribute(name, text);
  }

  function lightCurvePath() {
    const { timesS, log10LuminosityErgS } = scene.lightCurve;
    let low = Infinity;
    let high = -Infinity;
    for (const value of log10LuminosityErgS) {
      low = Math.min(low, value);
      high = Math.max(high, value);
    }
    const span = Math.max(high - low, 1e-6);
    const padding = 3;
    return Array.from(timesS, (time, index) => {
      const x = PLOT_WIDTH * timeline.progressAtTime(time);
      const y = padding + (PLOT_HEIGHT - 2 * padding) * (1 - (log10LuminosityErgS[index] - low) / span);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    }).join(" ");
  }

  function updateMotionButton(running) {
    const action = running ? i18n.t("transient.pauseTimeline") : i18n.t("transient.resumeTimeline");
    setAttribute(elements.playPause, "aria-label", action);
    setAttribute(elements.playPause, "title", action);
    const mark = elements.playPause.querySelector("span");
    if (mark) mark.textContent = running ? "Ⅱ" : "▶";
  }

  function updateTransport(sample) {
    const progress = timeline.progressAtTime(sample.timeS);
    const value = progress.toFixed(6);
    if (elements.scrubber.value !== value) elements.scrubber.value = value;
    const timeText = formatDuration(sample.timeS, i18n);
    setText(elements.timeValue, timeText);
    setAttribute(elements.scrubber, "aria-valuetext", `${timeText} · ${i18n.t(sample.regimeKey)}`);
    const stationary = !state.running || scrubbing || holding;
    setText(elements.playbackRate, stationary
      ? (holding && state.running ? i18n.t("transient.playback.endHold") : i18n.t("transient.playback.paused"))
      : i18n.t("transient.playback.rate", { rate: scientific(actualRate, 1) }));
    setAttribute(elements.followToggle, "aria-pressed", String(followExpansion));
    setText(elements.followToggle, followExpansion
      ? i18n.t("transient.follow.on")
      : i18n.t("transient.follow.off"));
    const cursor = (PLOT_WIDTH * progress).toFixed(2);
    setAttribute(elements.timeCursor, "x1", cursor);
    setAttribute(elements.timeCursor, "x2", cursor);
  }

  function updateReadouts(sample) {
    setHtml(elements.observerValue, sample.readouts.observer);
    setHtml(elements.radiusValue, sample.readouts.radius);
    setHtml(elements.shadowValue, sample.readouts.segment);
    setText(elements.regime, i18n.t(sample.regimeKey));
    updateTransport(sample);
  }

  function evaluate() {
    lastSample = scene.evaluate(state.time);
    return lastSample;
  }

  function seekFromScrubber() {
    state.time = timeline.seek(timeline.timeAtProgress(Number(elements.scrubber.value)));
    holding = false;
    holdElapsed = 0;
    actualRate = 0;
    bump();
    updateReadouts(evaluate());
    controls.requestRender();
  }

  function beginScrub() {
    if (scrubbing) return;
    scrubbing = true;
    resumeAfterScrub = state.running;
    actualRate = 0;
    bump();
    controls.setRunning(false);
  }

  function endScrub() {
    if (!scrubbing) return;
    scrubbing = false;
    bump();
    const resume = resumeAfterScrub;
    resumeAfterScrub = false;
    if (resume) controls.setRunning(true);
  }

  function bindControls() {
    abortController?.abort();
    abortController = new AbortController();
    const options = { signal: abortController.signal };
    elements.playPause.addEventListener("click", () => controls.setRunning(!state.running), options);
    elements.followToggle.addEventListener("click", () => {
      followExpansion = !followExpansion;
      frozenScaleKm = followExpansion ? null : (lastSample?.scaleKm ?? frozenScaleKm);
      bump();
      updateReadouts(evaluate());
      controls.requestRender();
    }, options);
    elements.scrubber.addEventListener("pointerdown", beginScrub, options);
    elements.scrubber.addEventListener("input", () => {
      if (!scrubbing) beginScrub();
      seekFromScrubber();
    }, options);
    elements.scrubber.addEventListener("change", endScrub, options);
    elements.scrubber.addEventListener("blur", endScrub, options);
    elements.scrubber.addEventListener("keydown", (event) => {
      if (SCRUB_KEYS.has(event.key)) beginScrub();
    }, options);
    elements.scrubber.addEventListener("keyup", (event) => {
      if (SCRUB_KEYS.has(event.key)) endScrub();
    }, options);
    documentRef.defaultView?.addEventListener("pointerup", endScrub, options);
    documentRef.defaultView?.addEventListener("pointercancel", endScrub, options);
  }

  // scene.modes: three { label, title } i18n key pairs for the photographic
  // mode and two false-colour views.
  function configureModes() {
    const buttons = [elements.modeScience, elements.modeHubble, elements.modeFrequency];
    scene.modes.forEach((mode, index) => {
      const button = buttons[index];
      button.textContent = i18n.t(mode.label);
      button.setAttribute("title", i18n.t(mode.title));
      button.hidden = false;
      button.classList.remove("diagnostic-only");
    });
    elements.advancedDiagnostics.hidden = true;
  }

  function scaleKm(sample) {
    if (followExpansion || frozenScaleKm == null) {
      return sample.scaleKm;
    }
    return frozenScaleKm;
  }

  return Object.freeze({
    elements,
    get followExpansion() {
      return followExpansion;
    },
    get revision() {
      return revision;
    },
    get lastSample() {
      return lastSample;
    },
    scaleKm,
    bump,
    evaluate,
    updateReadouts,
    updateMotionButton,

    initialize(initialTime, startsRunning) {
      if (initialized) return;
      initialized = true;
      scrubbing = false;
      resumeAfterScrub = false;
      holding = false;
      holdElapsed = 0;
      followExpansion = true;
      frozenScaleKm = null;
      bump();
      documentRef.documentElement.classList.add("scene-transient", scene.rootClass);
      documentRef.title = i18n.t(`${prefix}.documentTitle`);
      state.time = timeline.seek(initialTime);
      state.distance = scene.cameraDefaults.distance;
      state.phase = scene.cameraDefaults.phase;
      state.orbitTilt = scene.cameraDefaults.latitude;
      // Exposure compensation in photographic stops around the physical
      // auto-exposure; playback speed as a multiplier of the presentation.
      ui.exposure.min = "-4";
      ui.exposure.max = "4";
      ui.exposure.step = "0.1";
      ui.exposure.value = "0";
      state.exposure = 0;
      ui.timeScale.min = "0";
      ui.timeScale.max = "4";
      ui.timeScale.step = "0.25";
      ui.timeScale.value = "1";
      state.timeScale = 1;
      elements.massControl.hidden = true;
      elements.accretionControl.hidden = true;
      elements.eyebrow.textContent = i18n.t(`${prefix}.eyebrow`);
      elements.title.textContent = i18n.t(`${prefix}.title`);
      elements.observerLabel.textContent = i18n.t(`${prefix}.observerLabel`);
      elements.radiusLabel.textContent = i18n.t(`${prefix}.radiusLabel`);
      elements.shadowLabel.textContent = i18n.t(`${prefix}.segmentLabel`);
      elements.sceneStatus.hidden = false;
      elements.sceneStatus.textContent = i18n.t(`${prefix}.initialStatus`);
      elements.timelineSection.hidden = false;
      elements.timelineSection.setAttribute("aria-label", i18n.t(`${prefix}.timelineLabel`));
      elements.plotLabel.innerHTML = i18n.t(`${prefix}.plotLabelHtml`);
      elements.plotPath.setAttribute("d", lightCurvePath());
      elements.scrubber.min = "0";
      elements.scrubber.max = "1";
      elements.scrubber.step = "0.0005";
      elements.desktopHint.textContent = i18n.t(`${prefix}.desktopHint`);
      elements.physicsNote.innerHTML = i18n.t(`${prefix}.physicsHtml`);
      elements.followToggle.setAttribute("title", i18n.t("transient.follow.title"));
      configureModes();
      bindControls();
      updateMotionButton(startsRunning);
      updateReadouts(evaluate());
    },

    readoutsForControls() {
      const ev = Number(state.exposure);
      setText(ui.exposureValue, `${ev >= 0 ? "+" : "−"}${Math.abs(ev).toFixed(1)} EV`);
      setText(ui.timeScaleValue, `${Number(state.timeScale).toFixed(2)}×`);
      setText(ui.qualityValue, `${Number(state.quality).toFixed(2)}×`);
      updateReadouts(evaluate());
    },

    timeAdvancing() {
      return !scrubbing && !holding;
    },

    advance(deltaSeconds) {
      const result = timeline.advance(state.time, deltaSeconds, Math.max(state.timeScale, 0), holdElapsed);
      state.time = result.time;
      holding = result.holding;
      holdElapsed = result.holdElapsed;
      actualRate = result.rate;
      if (result.wrapped) {
        followExpansion = true;
        frozenScaleKm = null;
        bump();
      }
    },

    onMotionChanged(running) {
      bump();
      updateMotionButton(running);
      if (lastSample) updateTransport(lastSample);
    },

    resetCamera() {
      state.distance = scene.cameraDefaults.distance;
      state.phase = scene.cameraDefaults.phase;
      state.orbitTilt = scene.cameraDefaults.latitude;
      bump();
    },

    dispose() {
      abortController?.abort();
      abortController = null;
      initialized = false;
      scrubbing = false;
      holding = false;
      bump();
      elements.sceneStatus.classList.remove("is-strong-field", "is-fallback");
      if (!original.rootHadClass) documentRef.documentElement.classList.remove(scene.rootClass);
      if (!original.rootHadShared) documentRef.documentElement.classList.remove("scene-transient");
      documentRef.title = original.documentTitle;
      for (const [element, snapshot] of original.elements) {
        restoreElement(element, snapshot);
      }
      Object.assign(ui.exposure, original.exposure);
      Object.assign(ui.timeScale, original.timeScale);
      Object.assign(state, original.state);
      controls.requestRender();
    },
  });
}
