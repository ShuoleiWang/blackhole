import {
  SUPERNOVA_IA_SPARSE_STRIDE,
  SUPERNOVA_IA_UNIFORM_FLOATS,
  SUPERNOVA_IA_UNIFORM_LAYOUT,
  supernovaIaShaderBundle,
} from "../supernova-ia-shaders.js";
import { createSupernovaIaModel } from "../transients/supernova-ia-model.js";
import { blackbodyLinearSrgbPerBolometric } from "../transients/blackbody-color.js";
import {
  ASTRONOMICAL_UNIT_KM,
  DAY_S,
  SOLAR_LUMINOSITY_ERG_S,
  SPEED_OF_LIGHT_KM_S,
  STEFAN_BOLTZMANN_CGS,
} from "../transients/physical-constants.js";
import { createTransientSceneHost, scientific } from "./transient-scene-host.js";

const DEG = 180 / Math.PI;
// Key of the auto exposure: display value of the reference surface luminance
// before the shared ACES/HDR transform. Self-luminous surfaces are exposed
// near display white (a middle-grey key makes a 20,000 K star look like a
// grey, diffusely lit ball); hotter parts then overexpose, and on HDR
// displays reach into the extended range.
export const SUPERNOVA_IA_EXPOSURE_KEY = 1.0;
const EXPOSURE_KEY = SUPERNOVA_IA_EXPOSURE_KEY;
// Sky-referenced exposure: the calibrated panorama median (~1.7e-4 cd/m^2).
const SKY_REFERENCE_LUMINANCE = 1.7e-4;
// Playback adaptation (wall seconds): like a camera or the eye, the exposure
// follows a brightening within ~0.35 s, so a flash first overexposes, and
// recovers more slowly after it.
const ADAPTATION_BRIGHTENING_S = 0.35;
const ADAPTATION_DARKENING_S = 1.2;
// [maximum ejecta steps, optical-depth step, surface bisections]; the motion
// tiers also trace sparsely (one node per 4x4 pixels, fail-closed radiance
// interpolation), paused refinement traces every pixel.
const TIER_STEPS = Object.freeze({
  emergency: Object.freeze({ steps: Object.freeze([48, 0.8, 8]), sparse: true }),
  survival: Object.freeze({ steps: Object.freeze([56, 0.7, 9]), sparse: true }),
  interactive: Object.freeze({ steps: Object.freeze([72, 0.55, 10]), sparse: true }),
  balanced: Object.freeze({ steps: Object.freeze([128, 0.3, 14]), sparse: false }),
  fine: Object.freeze({ steps: Object.freeze([192, 0.18, 18]), sparse: false }),
});

function normalize(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function formatDistanceKm(km) {
  if (km >= 0.2 * ASTRONOMICAL_UNIT_KM) {
    return `${(km / ASTRONOMICAL_UNIT_KM).toPrecision(3)} AU`;
  }
  return `${scientific(km, 2)} km`;
}

export async function createSupernovaIaScene({
  document: documentRef,
  ui,
  state,
  i18n,
  controls,
}) {
  const model = createSupernovaIaModel();
  const query = new URLSearchParams(documentRef.defaultView?.location?.search || "");
  const requestedTime = Number(query.get("snTime"));
  const initialTime = query.get("snTime") !== null && Number.isFinite(requestedTime)
    ? requestedTime
    : model.timeline.firstTime;
  const startsRunning = query.get("paused") !== "1";

  let sampleCache = null;
  let exposureLog2 = null;
  let rendererIsWebGPU = true;

  function sampleAt(time) {
    if (!sampleCache || sampleCache.timeS !== time) {
      sampleCache = model.evaluate(time);
    }
    return sampleCache;
  }

  function referenceTemperature(sample) {
    return Math.max(sample.effectiveTemperatureK, 1500);
  }

  // Camera auto exposure keyed to the model's reference surface luminance (the
  // quiescent accretor, the area-median exploding surface, or the mean
  // photospheric disk), or to the sky.
  function targetExposureLog2(sample) {
    const ev = Number(state.exposure) || 0;
    if (state.mode === 1) {
      return ev + Math.log2(EXPOSURE_KEY / SKY_REFERENCE_LUMINANCE);
    }
    return ev + Math.log2(EXPOSURE_KEY / Math.max(sample.referenceLuminance, 1e-30));
  }

  const host = createTransientSceneHost({
    document: documentRef,
    ui,
    state,
    i18n,
    controls,
    scene: {
      id: "supernova-ia",
      rootClass: "scene-supernova-ia",
      i18nPrefix: "supernovaIa",
      timeline: model.timeline,
      lightCurve: model.lightCurve,
      cameraDefaults: { distance: 3.4, phase: 0.6, latitude: 0.42 },
      modes: [
        { label: "supernovaIa.mode.science", title: "supernovaIa.mode.scienceTitle" },
        { label: "supernovaIa.mode.sky", title: "supernovaIa.mode.skyTitle" },
        { label: "supernovaIa.mode.composition", title: "supernovaIa.mode.compositionTitle" },
      ],
      evaluate(time) {
        const sample = sampleAt(time);
        const scaleKm = sample.scaleKm;
        const cameraKm = state.distance * scaleKm;
        const sinceExplosion = time - model.explosion.explosionTimeS;
        let radius;
        if (sample.homologous) {
          radius = sample.photosphereVelocityKmS > 0
            ? i18n.t("supernovaIa.readout.photosphere", {
              velocity: Math.round(sample.photosphereVelocityKmS).toLocaleString("en-US"),
              temperature: Math.round(sample.effectiveTemperatureK).toLocaleString("en-US"),
            })
            : i18n.t("supernovaIa.readout.nebular", {
              temperature: Math.round(sample.colourTemperatureK).toLocaleString("en-US"),
            });
        } else if (sample.exploded) {
          radius = i18n.t("supernovaIa.readout.breakout", {
            temperature: scientific(sample.surface.maximumTemperatureK, 1),
            radius: scientific(sample.surface.maximumRadiusKm, 2),
          });
        } else {
          radius = i18n.t("supernovaIa.readout.binary", {
            separation: Math.round(model.binary.separationKm).toLocaleString("en-US"),
            period: model.binary.periodS.toFixed(1),
          });
        }
        const magnitude = 4.74 - 2.5 * Math.log10(sample.luminosity / SOLAR_LUMINOSITY_ERG_S);
        return {
          timeS: time,
          scaleKm,
          regimeKey: sample.regimeKey,
          readouts: {
            observer: i18n.t("supernovaIa.readout.camera", {
              distance: formatDistanceKm(cameraKm),
              lightTime: scientific(cameraKm / SPEED_OF_LIGHT_KM_S, 2),
              age: sample.exploded ? (sinceExplosion >= DAY_S
                ? `${(sinceExplosion / DAY_S).toFixed(1)} d`
                : `${scientific(Math.max(sinceExplosion, 0), 2)} s`) : "—",
            }),
            radius,
            segment: i18n.t("supernovaIa.readout.luminosity", {
              luminosity: scientific(sample.luminosity, 2),
              solar: scientific(sample.luminosity / SOLAR_LUMINOSITY_ERG_S, 2),
              magnitude: magnitude.toFixed(1).replace("-", "−"),
            }),
          },
        };
      },
    },
  });

  function currentExposure(sample) {
    const target = targetExposureLog2(sample);
    const adapting = state.running && host.timeAdvancing();
    if (exposureLog2 === null || !adapting) {
      exposureLog2 = target;
    }
    return exposureLog2;
  }

  function fallbackPacket(sample, scaleKm, exposure) {
    const colour = (temperature, limb) => {
      const rgb = blackbodyLinearSrgbPerBolometric(temperature);
      const luminance = 0.683 * STEFAN_BOLTZMANN_CGS * temperature ** 4 / Math.PI;
      return [...rgb.map((c) => Math.max(c, 0) * luminance * 2 ** exposure), limb];
    };
    const toUnits = (v) => v.map((c) => c / scaleKm);
    if (sample.homologous) {
      const centre = sample.primaryCentre;
      const radius = sample.visibleRadiusKm / scaleKm;
      return {
        starA: [...toUnits(centre), radius],
        starB: [0, 0, 0, 0],
        colourA: colour(sample.effectiveTemperatureK, 0.55),
        colourB: [0, 0, 0, 0],
        sky: [2 ** exposure * 1.2e-3, 0, 0, 0],
      };
    }
    return {
      starA: [...toUnits(sample.primaryCentre), (sample.exploded ? sample.surface.maximumRadiusKm : model.binary.primaryRadiusKm) / scaleKm],
      starB: [...toUnits(sample.donorCentre), model.binary.donorRadiusKm / scaleKm],
      colourA: colour(referenceTemperature(sample), model.config.limbDarkening),
      colourB: colour(sample.donorTemperatureK, model.config.limbDarkening),
      sky: [2 ** exposure * 1.2e-3, 0, 0, 0],
    };
  }

  return Object.freeze({
    id: "supernova-ia",
    qualityPolicy: Object.freeze({ id: "m3-pro-strong-field-v1" }),
    cameraDistanceLimits: Object.freeze({ min: 2.2, max: 40 }),
    motionLabels: Object.freeze({
      pause: i18n.t("transient.pauseTimeline"),
      resume: i18n.t("transient.resumeTimeline"),
    }),
    startsRunning,
    rendererOptions: Object.freeze({ shaderBundle: supernovaIaShaderBundle }),
    model,

    onRendererReady(capabilities) {
      rendererIsWebGPU = capabilities?.api === "webgpu";
      const status = host.elements.sceneStatus;
      status.classList.remove("is-strong-field", "is-fallback");
      status.classList.add(rendererIsWebGPU ? "is-strong-field" : "is-fallback");
      status.textContent = rendererIsWebGPU
        ? i18n.t("supernovaIa.status.webgpu")
        : i18n.t("supernovaIa.status.fallback");
      host.updateReadouts(host.evaluate());
    },

    initialize() {
      host.initialize(initialTime, startsRunning);
    },

    onMotionChanged(running) {
      host.onMotionChanged(running);
    },

    onModeChanged() {
      host.bump();
    },

    resetState() {
      host.resetCamera();
    },

    updateReadouts() {
      host.readoutsForControls();
      return true;
    },

    cameraFrame() {
      const sample = sampleAt(state.time);
      const scaleKm = host.scaleKm(host.evaluate());
      const focus = sample.focusKm.map((c) => c / scaleKm);
      const cosPhase = Math.cos(state.phase);
      const sinPhase = Math.sin(state.phase);
      const cosLatitude = Math.cos(state.orbitTilt);
      const sinLatitude = Math.sin(state.orbitTilt);
      const direction = normalize([cosLatitude * cosPhase, sinLatitude, cosLatitude * sinPhase]);
      const forward = direction.map((c) => -c);
      const tangent = normalize([-sinPhase, 0, cosPhase]);
      return {
        cameraPos: focus.map((c, i) => c + direction[i] * state.distance),
        forward,
        right: tangent.map((c) => -c),
        up: normalize(cross(forward, tangent)),
        observerVelocity: [0, 0, 0],
        observerBeta: 0,
      };
    },

    timeAdvancing() {
      return host.timeAdvancing();
    },

    advance(deltaSeconds) {
      host.advance(deltaSeconds);
      const sample = sampleAt(state.time);
      const target = targetExposureLog2(sample);
      if (exposureLog2 === null) {
        exposureLog2 = target;
      } else {
        const time = target < exposureLog2 ? ADAPTATION_BRIGHTENING_S : ADAPTATION_DARKENING_S;
        exposureLog2 += (target - exposureLog2) * (1 - Math.exp(-deltaSeconds / time));
      }
    },

    extendFrame(baseFrame) {
      const sample = sampleAt(state.time);
      const scaleKm = host.scaleKm(host.evaluate());
      host.updateReadouts(host.lastSample);
      const exposure = currentExposure(sample);
      const packet = model.packUniforms(sample, {
        scaleKm,
        exposureLog2: exposure,
        quality: [0, 1, 0, 1],
        steps: TIER_STEPS.interactive.steps,
        mode: baseFrame.mode,
      }, SUPERNOVA_IA_UNIFORM_LAYOUT, SUPERNOVA_IA_UNIFORM_FLOATS);
      return {
        ...baseFrame,
        // Sky-referenced exposure is only a different exposure choice made
        // here; the shared post pass reserves mode 1 for the single-hole
        // scene's Hubble grade, so the renderer sees the photographic mode.
        mode: baseFrame.mode === 1 ? 0 : baseFrame.mode,
        // Exposure is applied in the trace pass (physical luminance spans
        // >25 decades); the post pass receives unity.
        exposure: 1,
        diagnosticDisplay: baseFrame.mode === 2 || baseFrame.mode === 3,
        bloom: 0,
        diskOuterRadius: 0,
        cameraRadius: state.distance,
        sceneTransientUniforms: packet,
        sceneTransientFallback: rendererIsWebGPU ? null : fallbackPacket(sample, scaleKm, exposure),
      };
    },

    applyStrongFieldQuality(frame, decision) {
      const tier = TIER_STEPS[decision?.qualityTierId] ?? TIER_STEPS.balanced;
      const sparse = tier.sparse && (frame.mode === 0 || frame.mode === 1);
      return {
        ...frame,
        sceneTransientSteps: tier.steps,
        // The renderer traces the coarse node field first when slot 3 equals
        // the bundle's declared stride.
        sceneStrongDiagnostics: [0, 0, 0, sparse ? SUPERNOVA_IA_SPARSE_STRIDE : 0],
      };
    },

    renderRevision(frame) {
      return Object.freeze({
        physics: `${frame.time}|${host.revision}`,
        transport: [
          frame.mode,
          Number(state.exposure),
          exposureLog2 ?? 0,
          host.followExpansion ? 1 : 0,
          frame.sceneTransientSteps?.join(",") ?? "",
        ].join("|"),
      });
    },

    dispose() {
      host.dispose();
    },
  });
}
