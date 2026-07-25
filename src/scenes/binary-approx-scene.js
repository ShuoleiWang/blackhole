import { binaryApproxShaderBundle } from "../binary-shaders.js";

const MANIFEST_URL = new URL(
  "../../assets/scenes/binary-pn-equal-mass-v1.json",
  import.meta.url,
);
const DEG = 180 / Math.PI;

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

function interpolate(a, b, weight) {
  return a + (b - a) * weight;
}

async function loadManifest() {
  const response = await fetch(MANIFEST_URL);
  if (!response.ok) {
    throw new Error(`Binary scene manifest request failed (${response.status})`);
  }
  const manifest = await response.json();
  if (
    manifest.schema !== "blackhole.binary-scene/v1"
    || manifest.accuracy?.fullNumericalRelativity !== false
    || manifest.accuracy?.nearZoneSpacetimeIncluded !== false
    || !Array.isArray(manifest.timeline?.samples)
    || manifest.timeline.samples.length < 2
    || !Array.isArray(manifest.system?.bodies)
    || manifest.system.bodies.length !== 2
    || !Number.isInteger(manifest.rendererDefaults?.raySteps)
    || manifest.rendererDefaults.raySteps !== 512
  ) {
    throw new Error("Binary scene manifest does not satisfy the preview safety contract");
  }
  const masses = manifest.system.bodies.map((body) => Number(body.massFraction));
  const remnantMass = Number(manifest.system.previewRemnant?.massFraction);
  if (
    masses.some((mass) => !Number.isFinite(mass) || mass <= 0)
    || !Number.isFinite(remnantMass)
    || remnantMass <= 0
  ) {
    throw new Error("Binary scene manifest contains invalid mass parameters");
  }
  return manifest;
}

function timelineEvaluator(manifest) {
  const samples = manifest.timeline.samples;
  const eta = manifest.system.symmetricMassRatioEta;
  const inspiral = manifest.model.inspiral;
  const merger = manifest.model.mergerTransition;
  const pnCoefficient = (256 / 5) * eta;
  const firstTime = samples[0].tM;
  const lastTime = samples[samples.length - 1].tM;

  return (timeM) => {
    if (timeM <= firstTime) {
      return { ...samples[0], timelineFraction: 0 };
    }
    const last = samples[samples.length - 1];
    if (timeM >= lastTime) {
      return { ...last, timelineFraction: 1 };
    }

    if (timeM <= merger.startTimeM) {
      // Evaluate the declared leading-order PN model analytically between the
      // sparse manifest checkpoints.  Linear phase/waveform interpolation
      // would alias several gravitational-wave cycles in the early intervals.
      const separationM = (
        inspiral.matchingSeparationM ** 4 - pnCoefficient * timeM
      ) ** 0.25;
      const orbitalPhaseRad = (
        inspiral.startSeparationM ** 2.5 - separationM ** 2.5
      ) / (32 * eta);
      return {
        tM: timeM,
        regime: "pn-inspiral",
        separationM,
        orbitalPhaseRad,
        mergerBlend: 0,
        waveform: {
          hPlusScaled: Math.cos(2 * orbitalPhaseRad) / separationM,
          hCrossScaled: Math.sin(2 * orbitalPhaseRad) / separationM,
        },
        timelineFraction: (timeM - firstTime) / (lastTime - firstTime),
      };
    }

    let upperIndex = 1;
    while (upperIndex < samples.length && samples[upperIndex].tM < timeM) {
      upperIndex += 1;
    }
    const lower = samples[upperIndex - 1];
    const upper = samples[upperIndex];
    const local = clamp(
      (timeM - lower.tM) / Math.max(upper.tM - lower.tM, 1e-9),
      0,
      1,
    );
    const mergerBlend = interpolate(lower.mergerBlend, upper.mergerBlend, local);
    return {
      tM: timeM,
      separationM: interpolate(lower.separationM, upper.separationM, local),
      orbitalPhaseRad: interpolate(
        lower.orbitalPhaseRad,
        upper.orbitalPhaseRad,
        local,
      ),
      mergerBlend,
      waveform: {
        hPlusScaled: interpolate(
          lower.waveform.hPlusScaled,
          upper.waveform.hPlusScaled,
          local,
        ),
        hCrossScaled: interpolate(
          lower.waveform.hCrossScaled,
          upper.waveform.hCrossScaled,
          local,
        ),
      },
      regime: mergerBlend >= 1
        ? "remnant"
        : "phenomenological-merger",
      timelineFraction: (timeM - firstTime) / (lastTime - firstTime),
    };
  };
}

function waveformPath(evaluate, firstTime, finalTime) {
  const width = 280;
  const height = 46;
  const padding = 3;
  const duration = finalTime - firstTime;
  const denseSamples = Array.from({ length: 260 }, (_, index) => {
    const fraction = index / 259;
    return evaluate(firstTime + fraction * duration);
  });
  const amplitude = Math.max(
    ...denseSamples.map((sample) => Math.abs(sample.waveform.hPlusScaled)),
    1e-6,
  );
  return denseSamples.map((sample, index) => {
    const x = width * (sample.tM - firstTime) / duration;
    const y = padding + (height - 2 * padding)
      * (0.5 - 0.46 * sample.waveform.hPlusScaled / amplitude);
    return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(" ");
}

function regimeLabel(sample) {
  if (sample.regime === "pn-inspiral") {
    return "PN 螺旋靠近";
  }
  if (sample.regime === "phenomenological-merger") {
    return "现象学过渡 · 非 NR";
  }
  return "合并后近似";
}

function requiredElement(documentRef, id) {
  const element = documentRef.getElementById(id);
  if (!element) {
    throw new Error(`Binary scene requires interface element #${id}`);
  }
  return element;
}

export async function createBinaryApproxScene({
  document: documentRef,
  ui,
  state,
  formatMass,
  formatGravitationalRadius,
}) {
  const manifest = await loadManifest();
  const samples = manifest.timeline.samples;
  const evaluate = timelineEvaluator(manifest);
  const firstTime = samples[0].tM;
  const finalTime = samples[samples.length - 1].tM;
  const durationM = finalTime - firstTime;
  const defaults = manifest.rendererDefaults;
  const initialViewLatitude = (
    90 - defaults.initialViewingInclinationDeg
  ) / DEG;
  const binaryMasses = [
    manifest.system.bodies[0].massFraction,
    manifest.system.bodies[1].massFraction,
    manifest.system.previewRemnant.massFraction,
    0,
  ];
  const defaultTimeScale = durationM / defaults.cycleDurationSeconds;
  // Keep the remnant visible briefly before the explicitly discontinuous
  // preview loop returns to the wide binary.
  const remnantHoldM = defaultTimeScale * 2.5;

  const elements = {
    eyebrow: requiredElement(documentRef, "sceneEyebrow"),
    title: requiredElement(documentRef, "panelTitle"),
    observerLabel: requiredElement(documentRef, "observerLabel"),
    radiusLabel: requiredElement(documentRef, "radiusLabel"),
    shadowLabel: requiredElement(documentRef, "shadowLabel"),
    massLabel: requiredElement(documentRef, "massLabel"),
    accretionControl: requiredElement(documentRef, "accretionControl"),
    physicsNote: requiredElement(documentRef, "physicsNote"),
    sceneStatus: requiredElement(documentRef, "sceneStatus"),
    binaryTimeline: requiredElement(documentRef, "binaryTimeline"),
    binaryRegime: requiredElement(documentRef, "binaryRegime"),
    waveformPath: requiredElement(documentRef, "binaryWaveformPath"),
    timeCursor: requiredElement(documentRef, "binaryTimeCursor"),
    desktopHint: requiredElement(documentRef, "desktopHint"),
  };

  function updateDynamicReadouts(sample) {
    const phaseDegrees = ((sample.orbitalPhaseRad * DEG) % 360 + 360) % 360;
    if (sample.mergerBlend < 0.995) {
      ui.observerValue.innerHTML = `a ${sample.separationM.toFixed(2)} M · φ ${phaseDegrees.toFixed(0)}°`;
    } else {
      ui.observerValue.textContent = "单一合并后天体";
    }
    ui.shadowValue.textContent = regimeLabel(sample);
    elements.binaryRegime.textContent = `${regimeLabel(sample)} · h₊ ${sample.waveform.hPlusScaled.toFixed(3)}`;
    const cursor = (280 * clamp(sample.timelineFraction, 0, 1)).toFixed(2);
    elements.timeCursor.setAttribute("x1", cursor);
    elements.timeCursor.setAttribute("x2", cursor);
  }

  return Object.freeze({
    id: "binary-approx",
    rendererOptions: Object.freeze({
      shaderBundle: binaryApproxShaderBundle,
    }),
    manifest,

    initialize() {
      documentRef.documentElement.classList.add("scene-binary-approx");
      documentRef.title = "双黑洞合并近似预览 · 深空观测台";
      state.time = firstTime;
      state.distance = defaults.observerRadiusM;
      state.phase = 0.58;
      state.orbitTilt = initialViewLatitude;
      ui.exposure.value = defaults.exposure.toFixed(2);
      ui.timeScale.max = "96";
      ui.timeScale.step = "1";
      ui.timeScale.value = Math.round(defaultTimeScale).toString();

      elements.eyebrow.textContent = "PN 轨道 · 多中心弱场光线";
      elements.title.textContent = "双黑洞合并预览";
      elements.observerLabel.textContent = "轨道状态";
      elements.radiusLabel.textContent = "1 M（GM/c²）";
      elements.shadowLabel.textContent = "模型区段";
      elements.massLabel.textContent = "系统总质量";
      elements.sceneStatus.hidden = false;
      elements.sceneStatus.textContent = "实验场景 · PN / NR-informed preview · 不是完整数值相对论光追";
      elements.binaryTimeline.hidden = false;
      elements.waveformPath.setAttribute(
        "d",
        waveformPath(evaluate, firstTime, finalTime),
      );
      elements.desktopHint.textContent = "拖动观察 · 滚轮缩放 · 空格暂停时间线";
      elements.physicsNote.innerHTML = [
        "真空等质量、无自旋双黑洞。螺旋靠近采用最低阶 PN；合并与捕获面为现象学过渡；光线采用 fast-light 多中心弱场偏折，",
        "<strong>强场细节、共同视界与光子环不具定量精度</strong>。未添加缺乏气体初始条件的发光吸积盘。",
        '模型边界与完整 NR 路线见 <a href="./docs/binary-model.md" target="_blank" rel="noreferrer">binary-model.md</a>。',
      ].join("");
      elements.accretionControl.setAttribute("aria-hidden", "true");
      ui.accretion.disabled = true;
      updateDynamicReadouts(evaluate(state.time));
    },

    resetState() {
      state.distance = defaults.observerRadiusM;
      state.phase = 0.58;
      state.orbitTilt = initialViewLatitude;
      updateDynamicReadouts(evaluate(state.time));
    },

    updateReadouts() {
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
      // The binary orbit lies in the x-z plane, so its angular-momentum axis
      // is +y.  state.orbitTilt is camera latitude above that plane; the
      // conventional viewing inclination is therefore i = 90° - latitude.
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
        // No circular-Schwarzschild observer tetrad is meaningful in this
        // approximate time-dependent scene.  Keep the preview camera static in
        // its local frame instead of applying the single-hole aberration model.
        observerVelocity: [0, 0, 0],
        observerBeta: 0,
      };
    },

    advance(deltaSeconds) {
      state.time += deltaSeconds * state.timeScale;
      if (state.time > finalTime + remnantHoldM) {
        state.time = firstTime;
      }
    },

    extendFrame(baseFrame) {
      const sample = evaluate(state.time);
      updateDynamicReadouts(sample);
      return {
        ...baseFrame,
        accretion: 0,
        fov: Math.max(baseFrame.fov, defaults.fieldOfViewDeg / DEG),
        diskOuterRadius: 0,
        // Keep the ray integrator converged across quality tiers.  Adaptive
        // performance control may lower resolution, but not the physical step
        // budget, because unresolved rays deliberately render black.
        steps: defaults.raySteps,
        observerVelocity: [0, 0, 0],
        observerBeta: 0,
        sceneBinaryState: [
          sample.separationM,
          sample.orbitalPhaseRad,
          sample.mergerBlend,
          0,
        ],
        sceneBinaryMasses: binaryMasses,
      };
    },
  });
}
