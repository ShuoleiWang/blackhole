import assert from "node:assert/strict";
import test from "node:test";

import { fakeDocument } from "./fixtures/transient-fake-dom.mjs";
import { createI18n } from "../src/i18n.js";
import { createSupernovaIaScene } from "../src/scenes/supernova-ia-scene.js";
import {
  SUPERNOVA_IA_SPARSE_STRIDE,
  SUPERNOVA_IA_UNIFORM_FLOATS,
  SUPERNOVA_IA_UNIFORM_LAYOUT,
  supernovaIaShaderBundle,
  supernovaIaTraceFragmentWGSL,
} from "../src/supernova-ia-shaders.js";
import {
  createSupernovaIaModel,
  nickelCobaltHeating,
  supernovaIaBV,
} from "../src/transients/supernova-ia-model.js";
import {
  DAY_S,
  GRAVITATIONAL_CONSTANT_CGS,
  SOLAR_MASS_G,
  SPEED_OF_LIGHT_CM_S,
  STEFAN_BOLTZMANN_CGS,
} from "../src/transients/physical-constants.js";

const model = createSupernovaIaModel();
const explosionTime = model.explosion.explosionTimeS;
const luminosityAtDay = (days) => model.bolometricLuminosity(explosionTime + days * DAY_S);

function peakOfLightCurve() {
  let best = { days: 0, luminosity: 0 };
  for (let days = 5; days <= 30; days += 0.01) {
    const luminosity = luminosityAtDay(days);
    if (luminosity > best.luminosity) best = { days, luminosity };
  }
  return best;
}

test("the double-detonation binary fills the donor's Roche lobe and feeds a direct-impact stream", () => {
  const { binary } = model;
  // Chandrasekhar-EOS radii of the cold 1.0 and 0.6 Msun C/O white dwarfs.
  assert.ok(Math.abs(binary.primaryRadiusKm - 5716) / 5716 < 0.03, `R1 ${binary.primaryRadiusKm}`);
  assert.ok(Math.abs(binary.donorRadiusKm - 8841) / 8841 < 0.03, `R2 ${binary.donorRadiusKm}`);
  // Eggleton lobe = donor radius fixes a; Kepler fixes P.
  assert.ok(Math.abs(binary.separationKm - 26338) / 26338 < 0.01, `a ${binary.separationKm}`);
  const gm = 1.32712440018e11 * 1.6;
  const keplerPeriod = 2 * Math.PI * Math.sqrt(binary.separationKm ** 3 / gm);
  assert.ok(Math.abs(binary.periodS - keplerPeriod) / keplerPeriod < 1e-3);
  assert.ok(binary.periodS > 50 && binary.periodS < 70, `P ${binary.periodS}`);
  // q = 0.6 is below the direct-impact limit: the stream strikes the
  // accretor instead of circularising into a disk.
  assert.equal(binary.streamReachesAccretor, true);
  assert.ok(binary.impactLeadDeg > 0 && binary.impactLeadDeg < 90, `impact ${binary.impactLeadDeg}`);
  assert.ok(binary.lagrangeX > -binary.mu && binary.lagrangeX < 1 - binary.mu);
});

test("the released donor is struck when the explosion surface reaches it", () => {
  const offset = SUPERNOVA_IA_UNIFORM_LAYOUT.donorShock - 36;
  const strikeTime = (time) => {
    const sample = model.evaluate(time);
    return model.packUniforms(sample, {
      scaleKm: sample.scaleKm, exposureLog2: 0, quality: [0, 1, 0, 1], steps: [72, 0.55, 10], mode: 0,
    }, SUPERNOVA_IA_UNIFORM_LAYOUT, SUPERNOVA_IA_UNIFORM_FLOATS)[offset];
  };
  const strike = strikeTime(3);
  // The helium ash (5,000 km/s since ~0.3 s) plus the core ash (up to the
  // stretched ejecta edge after the facing breakout at ~2.5 s) bridge
  // a - R1 - R2 = 11,800 km.
  assert.ok(strike > 2.6 && strike < 3.2, `strike ${strike}`);
  const before = model.evaluate(strike - 0.02);
  const separation = Math.hypot(...before.donorCentre.map((c, i) => c - before.primaryCentre[i]));
  assert.ok(separation - model.binary.donorRadiusKm > before.surface.maximumRadiusKm * 0.8);
  assert.equal(model.evaluate(strike - 0.05).donorTemperatureK, model.config.donorTemperatureK);
  assert.equal(model.evaluate(strike + 0.05).donorTemperatureK, model.config.donorPostShockTemperatureK);
});

test("the direct-impact spot radiates at the local Eddington flux", () => {
  const { binary, config } = model;
  const radiusCm = binary.primaryRadiusKm * 1e5;
  const gravity = GRAVITATIONAL_CONSTANT_CGS * config.primaryMassSolar * SOLAR_MASS_G / radiusCm ** 2;
  const eddingtonFlux = gravity * SPEED_OF_LIGHT_CM_S / config.impactOpacity;
  assert.ok(Math.abs(binary.impactTemperatureK / (eddingtonFlux / STEFAN_BOLTZMANN_CGS) ** 0.25 - 1) < 1e-9);
  assert.ok(binary.impactTemperatureK > 9e5 && binary.impactTemperatureK < 1.1e6, `T ${binary.impactTemperatureK}`);
  const spotArea = 2 * Math.PI * radiusCm ** 2 * (1 - Math.cos(config.impactAngularRadiusDeg * Math.PI / 180));
  assert.ok(Math.abs(binary.impactLuminosityErgS / (eddingtonFlux * spotArea) - 1) < 1e-9);
  assert.ok(binary.impactLuminosityErgS > 1e36 && binary.impactLuminosityErgS < 1e37);
  // The pre-explosion luminosity is what the tracer draws: the two
  // photospheres and the spot; irradiation is reprocessed light, not added.
  const photospheres = 4 * Math.PI * STEFAN_BOLTZMANN_CGS * 1e10 * (
    binary.primaryRadiusKm ** 2 * config.primaryTemperatureK ** 4
    + binary.donorRadiusKm ** 2 * config.donorTemperatureK ** 4
  );
  assert.ok(Math.abs(model.bolometricLuminosity(-30) / (photospheres + binary.impactLuminosityErgS) - 1) < 1e-9);
});

test("the camera exposes for the area-median surface of the exploding star", () => {
  const quiescent = model.evaluate(-30).referenceLuminance;
  // At 0.3 s the helium flash covers a ~5% cap: the median surface is still
  // the quiescent photosphere, so the flash overexposes instead of blacking
  // out the rest of the star.
  assert.ok(Math.abs(model.evaluate(0.3).referenceLuminance / quiescent - 1) < 1e-9);
  // Once the helium front has passed the equator (1.5 s) and after the core
  // breakout (2.5 s) most of the surface is ~1e6 K and the reference follows.
  for (const time of [1.5, 2.5]) {
    assert.ok(model.evaluate(time).referenceLuminance > 30 * quiescent, `t ${time}`);
  }
});

test("the fireball hands off continuously to the homologous ejecta", () => {
  const handoff = explosionTime + model.config.homologousStartS;
  const before = model.evaluate(handoff - 1e-3);
  const after = model.evaluate(handoff + 1e-3);
  assert.equal(before.homologous, false);
  assert.equal(after.homologous, true);
  for (const key of ["luminosity", "visibleRadiusKm", "scaleKm"]) {
    assert.ok(Math.abs(after[key] / before[key] - 1) < 0.01, `${key} ${before[key]} -> ${after[key]}`);
  }
  assert.ok(Math.abs(Math.log2(after.referenceLuminance / before.referenceLuminance)) < 0.1);
});

test("the fireball carries the ejecta velocity stretch towards the impact pole", () => {
  const late = model.evaluate(explosionTime + 90);
  const a = 0.5 * (model.config.impactPoleStretch - model.config.antipodeStretch);
  const b = 0.5 * (model.config.impactPoleStretch + model.config.antipodeStretch);
  // Pole radius vmax (1 + a + b) t against the equal-area radius.
  const expected = (1 + a + b) / Math.sqrt((1 + b) ** 2 + a * a / 3);
  const ratio = late.surface.maximumRadiusKm / late.surface.meanRadiusKm;
  assert.ok(Math.abs(ratio / expected - 1) < 0.02, `${ratio} vs ${expected}`);
});

test("the radioactive light curve matches a normal SN Ia (SN 2011fe-like)", () => {
  const nickel = model.explosion.nickelMassSolar;
  assert.ok(nickel > 0.45 && nickel < 0.6, `M_Ni ${nickel}`);
  const peak = peakOfLightCurve();
  assert.ok(peak.days > 15.5 && peak.days < 19.5, `t_peak ${peak.days}`);
  assert.ok(peak.luminosity > 8e42 && peak.luminosity < 1.4e43, `L_peak ${peak.luminosity}`);
  // Arnett's rule: peak luminosity ~ instantaneous decay power.
  const alpha = peak.luminosity / (nickel * SOLAR_MASS_G * nickelCobaltHeating(peak.days * DAY_S));
  assert.ok(alpha > 0.85 && alpha < 1.25, `Arnett alpha ${alpha}`);
  // Bolometric decline over 15 d after peak of normal SNe Ia: 0.7-1.1 mag.
  const decline = 2.5 * Math.log10(peak.luminosity / luminosityAtDay(peak.days + 15));
  assert.ok(decline > 0.7 && decline < 1.1, `dm15(bol) ${decline}`);
  // Katz integral: t*E(t) balances the time-weighted heating and losses.
  const katz = model.diffusion.katz;
  const error = (katz.finalTimeEnergy - katz.initialTimeEnergy - katz.heatingIntegral + katz.lossIntegral)
    / katz.heatingIntegral;
  assert.ok(Math.abs(error) < 5e-3, `Katz ${error}`);
});

test("breakout flashes radiate the breakout-shell budget, far below the radioactive peak", () => {
  // Integrate L over the helium sweep and the core breakout above the
  // quiescent binary: each is ~1e40-1e41 erg (Piro, Chang & Weinberg 2010),
  // and no flash outshines the 56Ni-powered maximum.
  const quiescent = model.bolometricLuminosity(-1);
  const step = 1e-4;
  let helium = 0;
  let core = 0;
  let peak = 0;
  for (let time = 0; time < 5; time += step) {
    const excess = model.bolometricLuminosity(time) - quiescent;
    if (time < model.config.coreIgnitionTimeS + 0.2) helium += excess * step;
    else core += excess * step;
    peak = Math.max(peak, excess + quiescent);
  }
  assert.ok(helium > 1e40 && helium < 1e41, `helium ${helium}`);
  assert.ok(core > 1e40 && core < 1e41, `core ${core}`);
  assert.ok(peak < 0.1 * peakOfLightCurve().luminosity, `flash ${peak}`);
});

test("late-time gamma-ray trapping follows the observed t0 of normal SNe Ia", () => {
  // f_gamma = 1 - exp(-(t0/t)^2) with t0 = 30-45 d (Wygoda et al. 2019);
  // positrons (3.2% of the Co power) deposit locally.
  const nickelGrams = model.explosion.nickelMassSolar * SOLAR_MASS_G;
  for (const days of [100, 130, 150]) {
    const decay = nickelGrams * nickelCobaltHeating(days * DAY_S);
    const deposited = luminosityAtDay(days) / decay;
    const gammaFraction = (deposited - model.config.positronFraction) / (1 - model.config.positronFraction);
    const trappingDays = days * Math.sqrt(-Math.log(1 - gammaFraction));
    assert.ok(trappingDays > 30 && trappingDays < 45, `t0 ${trappingDays} d at ${days} d`);
  }
});

test("the photosphere recedes through the ejecta at normal SN Ia velocities and colours", () => {
  const atPeak = model.evaluate(explosionTime + 18 * DAY_S);
  assert.ok(atPeak.photosphereVelocityKmS > 8000 && atPeak.photosphereVelocityKmS < 12000);
  assert.ok(atPeak.colourTemperatureK > 8000 && atPeak.colourTemperatureK < 12000);
  const later = model.evaluate(explosionTime + 45 * DAY_S);
  assert.ok(later.photosphereVelocityKmS < atPeak.photosphereVelocityKmS);
  assert.ok(later.colourTemperatureK < 5500);
  // Intrinsic colour: B-V ~ 0 at B maximum, ~1.1 a month later.
  assert.ok(Math.abs(supernovaIaBV(18)) < 0.1);
  assert.ok(supernovaIaBV(48) > 0.9 && supernovaIaBV(48) < 1.2);
});

function wgslParamSlots(wgsl) {
  const start = wgsl.indexOf("struct Params {");
  const body = wgsl.slice(start, wgsl.indexOf("};", start));
  const slots = [];
  for (const line of body.split("\n")) {
    const vector = line.match(/^\s*(\w+): vec4<f32>,/);
    const array = line.match(/^\s*(\w+): array<vec4<f32>, (\d+)>,/);
    if (vector) {
      slots.push(vector[1]);
    } else if (array) {
      for (let index = 0; index < Number(array[2]); index += 1) slots.push(`${array[1]}[${index}]`);
    }
  }
  return slots;
}

test("the packed uniform layout matches the WGSL Params struct slot for slot", () => {
  const slots = wgslParamSlots(supernovaIaTraceFragmentWGSL);
  assert.equal(slots.length * 4, SUPERNOVA_IA_UNIFORM_FLOATS);
  const arrays = { streamPoint: "streamPoints", ejectaProfile: "ejectaProfile", ejectaRadiative: "ejectaRadiative" };
  for (const [name, offset] of Object.entries(SUPERNOVA_IA_UNIFORM_LAYOUT)) {
    assert.equal(offset % 4, 0);
    const indexed = name.match(/^(streamPoint|ejectaProfile|ejectaRadiative)(\d+)$/);
    const expected = indexed ? `${arrays[indexed[1]]}[${indexed[2]}]` : name;
    assert.equal(slots[offset / 4], expected, `slot ${offset / 4}`);
  }
  assert.equal(supernovaIaShaderBundle.uniforms.requiredFloatCount, SUPERNOVA_IA_UNIFORM_FLOATS);
});

test("uniform packets stay finite over the whole presentation timeline", () => {
  const { timeline } = model;
  let previousScale = 0;
  for (let k = 0; k <= 400; k += 1) {
    const time = timeline.timeAtProgress(k / 400);
    const sample = model.evaluate(time);
    assert.ok(sample.scaleKm > 0 && Number.isFinite(sample.scaleKm));
    const packet = model.packUniforms(sample, {
      scaleKm: sample.scaleKm,
      exposureLog2: Math.log2(0.55 / Math.max(sample.referenceLuminance, 1e-30)),
      quality: [0, 1, 0, 1],
      steps: [72, 0.55, 10],
      mode: 0,
    }, SUPERNOVA_IA_UNIFORM_LAYOUT, SUPERNOVA_IA_UNIFORM_FLOATS);
    assert.equal(packet.length, SUPERNOVA_IA_UNIFORM_FLOATS - 36);
    const bad = packet.findIndex((value) => !Number.isFinite(value));
    assert.equal(bad, -1, `non-finite slot ${bad} at t=${time}`);
    if (sample.homologous) {
      // The view grows with the ejecta once expansion is homologous.
      assert.ok(sample.scaleKm >= previousScale * 0.999, `scale shrank at t=${time}`);
      previousScale = sample.scaleKm;
    }
  }
});

test("the tracer declares a sparse radiance pass that shares the trace resources", () => {
  const coarse = supernovaIaShaderBundle.wgsl.coarse;
  assert.deepEqual({ ...coarse }, {
    entryPoint: "fsCoarse",
    format: "rgba32float",
    stride: SUPERNOVA_IA_SPARSE_STRIDE,
    binding: 3,
    sharesTraceResources: true,
  });
  const wgsl = supernovaIaTraceFragmentWGSL;
  assert.match(wgsl, /@group\(0\) @binding\(3\) var coarseField: texture_2d<f32>;/);
  assert.match(wgsl, /@group\(0\) @binding\(4\) var blackbodyTable: texture_2d<f32>;/);
  assert.match(wgsl, new RegExp(`const SPARSE_STRIDE: f32 = ${SUPERNOVA_IA_SPARSE_STRIDE}\\.0;`));
  const sparse = wgsl.slice(wgsl.indexOf("fn sparseRadiance"), wgsl.indexOf("@fragment fn fsMain"));
  // Twelve-node stencil: the bilinear cell plus the second-difference ring.
  assert.equal((sparse.match(/coarseNode\(/g) || []).length, 12);
  // Photographic modes only; diagnostic false colour is always traced.
  assert.match(
    wgsl,
    /params\.steps\.w == SPARSE_STRIDE && params\.quality\.w > 0\.5 && \(mode == 0 \|\| mode == 1\)/,
  );
});

test("WebGPU extras carry the quality vector, tier steps and sparse opt-in", () => {
  const sample = model.evaluate(explosionTime + 17 * DAY_S);
  const packet = model.packUniforms(sample, {
    scaleKm: sample.scaleKm,
    exposureLog2: 3,
    quality: [0, 1, 0, 1],
    steps: [192, 0.18, 18],
    mode: 0,
  }, SUPERNOVA_IA_UNIFORM_LAYOUT, SUPERNOVA_IA_UNIFORM_FLOATS);
  const tail = new Float32Array(SUPERNOVA_IA_UNIFORM_FLOATS - 36);
  supernovaIaShaderBundle.uniforms.writeWebGPUExtras(tail, {
    sceneTransientUniforms: packet,
    sceneTransientSteps: [72, 0.55, 10],
    sceneStrongDiagnostics: [0, 0, 0, SUPERNOVA_IA_SPARSE_STRIDE],
    strongFieldQuality: { accumulationIndex: 3, accumulationWeight: 0.25, historyEpoch: 7, historyReset: false },
  });
  const at = (name) => Array.from(tail.slice(SUPERNOVA_IA_UNIFORM_LAYOUT[name] - 36, SUPERNOVA_IA_UNIFORM_LAYOUT[name] - 32));
  assert.deepEqual(at("quality"), [3, 0.25, 7, 0]);
  assert.deepEqual(at("steps").map((v) => +v.toFixed(4)), [72, 0.55, 10, SUPERNOVA_IA_SPARSE_STRIDE]);
  assert.throws(
    () => supernovaIaShaderBundle.uniforms.writeWebGPUExtras(tail, { sceneTransientUniforms: new Float32Array(3) }),
    /sceneTransientUniforms/,
  );
});

async function sceneFixture(search = "") {
  const host = fakeDocument(search);
  const state = { time: 0, running: true, distance: 10, phase: 0, orbitTilt: 0, exposure: 1, timeScale: 1, quality: 1, mode: 0 };
  const renders = [];
  const scene = await createSupernovaIaScene({
    document: host.document,
    ui: host.ui,
    state,
    i18n: createI18n("?lang=en", { getItem: () => null, setItem() {} }),
    controls: { setRunning: (running) => { state.running = running; }, requestRender: () => renders.push(1) },
  });
  return { host, state, scene };
}

function baseFrame(state, mode = 0) {
  return {
    time: state.time, massSolar: 1, accretion: 0, exposure: 1, mode, steps: 160,
    ...scene_camera(state), fov: 0.77, skyRotation: -2.576, diskOuterRadius: 0, bloom: 0,
    observerVelocity: [0, 0, 0], observerBeta: 0,
  };
}

function scene_camera(state) {
  return { cameraPos: [0, 0, state.distance], cameraRadius: state.distance, forward: [0, 0, -1], right: [1, 0, 0], up: [0, 1, 0] };
}

test("the Type Ia scene maps motion tiers to sparse tracing and paused tiers to full tracing", async () => {
  const { scene, state } = await sceneFixture();
  scene.initialize();
  assert.equal(scene.qualityPolicy.id, "m3-pro-strong-field-v1");
  for (const [tier, steps, sparse] of [
    ["emergency", [48, 0.8, 8], SUPERNOVA_IA_SPARSE_STRIDE],
    ["survival", [56, 0.7, 9], SUPERNOVA_IA_SPARSE_STRIDE],
    ["interactive", [72, 0.55, 10], SUPERNOVA_IA_SPARSE_STRIDE],
    ["balanced", [128, 0.3, 14], 0],
    ["fine", [192, 0.18, 18], 0],
  ]) {
    for (const mode of [0, 1, 2, 3]) {
      const frame = scene.applyStrongFieldQuality(baseFrame(state, mode), { qualityTierId: tier });
      assert.deepEqual([...frame.sceneTransientSteps], steps);
      // False-colour diagnostics are never interpolated.
      assert.equal(frame.sceneStrongDiagnostics[3], mode <= 1 ? sparse : 0, `${tier} mode ${mode}`);
    }
  }
  scene.dispose();
});

test("the Type Ia scene drives the shared panel, packs frames and restores the host", async () => {
  const { scene, state, host } = await sceneFixture("?snTime=1468800");
  scene.initialize();
  assert.equal(host.document.title, "Type Ia supernova · Deep-space observatory");
  assert.equal(host.elements.get("binaryTimeline").hidden, false);
  assert.equal(host.ui.mass.label.hidden, true);
  assert.equal(host.ui.exposure.min, "-4");
  assert.equal(host.ui.exposure.max, "4");
  assert.equal(state.time, 1468800);
  assert.ok(host.elements.get("binaryWaveformPath").getAttribute("d").startsWith("M"));

  const frame = scene.extendFrame({ ...baseFrame(state), ...scene.cameraFrame() });
  assert.equal(frame.exposure, 1);
  // Sky-referenced exposure must not trigger the post pass's Hubble grade.
  state.mode = 1;
  assert.equal(scene.extendFrame({ ...baseFrame(state, 1), ...scene.cameraFrame() }).mode, 0);
  state.mode = 0;
  assert.equal(frame.sceneTransientUniforms.length, SUPERNOVA_IA_UNIFORM_FLOATS - 36);
  assert.equal(frame.sceneTransientFallback, null);
  assert.ok(Array.from(frame.sceneTransientUniforms).every(Number.isFinite));

  scene.onRendererReady({ api: "webgl2" });
  const fallback = scene.extendFrame({ ...baseFrame(state), ...scene.cameraFrame() }).sceneTransientFallback;
  for (const key of ["starA", "starB", "colourA", "colourB", "sky"]) {
    assert.equal(fallback[key].length, 4, key);
    assert.ok(fallback[key].every(Number.isFinite), key);
  }
  assert.ok(fallback.starA[3] > 0);

  // Motion advances the presentation clock and adapts the exposure.
  const before = state.time;
  scene.advance(0.5);
  assert.ok(state.time > before);

  scene.dispose();
  assert.equal(host.document.title, "original");
  assert.equal(host.elements.get("binaryTimeline").hidden, true);
  assert.equal(host.ui.exposure.min, "0.2");
  assert.equal(host.ui.exposure.max, "3");
  assert.equal(host.elements.get("modeScience").title, "original science title");
  assert.equal(host.document.documentElement.classList.contains("scene-transient"), false);
  for (const id of ["binaryPlayPause", "binaryTimeScrubber", "binarySlowMotion"]) {
    assert.equal(host.elements.get(id).listenerCount(), 0, `${id} listeners`);
  }
});
