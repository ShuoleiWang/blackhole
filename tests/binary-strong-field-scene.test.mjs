import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  TIDAL_TRUNCATION_FACTOR,
  eggletonRocheLobeFraction,
} from "../src/scenes/binary-accretion-model.js";
import { createBinaryApproxScene } from "../src/scenes/binary-approx-scene.js";
import { createBinaryDualDiskScene } from "../src/scenes/binary-dual-disk-scene.js";

globalThis.crypto ??= webcrypto;

class FakeEventTarget {
  constructor() {
    this.listeners = new Map();
  }

  addEventListener(type, listener, options = {}) {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
    options?.signal?.addEventListener?.("abort", () => {
      listeners.delete(listener);
    }, { once: true });
  }

  dispatchEvent(event) {
    for (const listener of this.listeners.get(event.type) ?? []) {
      listener.call(this, event);
    }
    return true;
  }
}

class FakeClassList {
  constructor() {
    this.values = new Set();
  }

  contains(name) {
    return this.values.has(name);
  }

  add(...names) {
    for (const name of names) {
      this.values.add(name);
    }
  }

  remove(...names) {
    for (const name of names) {
      this.values.delete(name);
    }
  }
}

class FakeElement extends FakeEventTarget {
  constructor(id) {
    super();
    this.id = id;
    this.attributeValues = new Map();
    this.classList = new FakeClassList();
    this._innerHTML = "";
    this._textContent = "";
    this.value = "";
    this.disabled = false;
    this.hidden = false;
    this.mark = null;
  }

  get innerHTML() {
    return this._innerHTML;
  }

  set innerHTML(value) {
    this._innerHTML = String(value);
    this._textContent = this._innerHTML.replace(/<[^>]*>/g, "");
  }

  get textContent() {
    return this._textContent;
  }

  set textContent(value) {
    this._textContent = String(value);
    this._innerHTML = this._textContent;
  }

  get attributes() {
    return [...this.attributeValues].map(([name, value]) => ({ name, value }));
  }

  setAttribute(name, value) {
    this.attributeValues.set(name, String(value));
  }

  removeAttribute(name) {
    this.attributeValues.delete(name);
  }

  querySelector(selector) {
    return selector === "span" ? this.mark : null;
  }
}

const ELEMENT_IDS = [
  "sceneEyebrow",
  "panelTitle",
  "observerLabel",
  "radiusLabel",
  "shadowLabel",
  "massLabel",
  "massValue",
  "accretionControl",
  "accretionLabel",
  "physicsNote",
  "sceneStatus",
  "modeScience",
  "modeHubble",
  "modeFrequency",
  "modeLookback",
  "modeNull",
  "modeError",
  "transferAdvancedDiagnostics",
  "binaryTimeline",
  "binaryRegime",
  "binaryWaveformLabel",
  "binaryWaveformPath",
  "binaryTimeCursor",
  "binaryPlayPause",
  "binaryTimeScrubber",
  "binaryTimeValue",
  "binarySlowMotion",
  "binaryPlaybackRate",
  "desktopHint",
  "observerValue",
  "rsValue",
  "shadowValue",
  "accretionValue",
  "exposureValue",
  "timeScaleValue",
  "qualityValue",
  "accretion",
  "exposure",
  "timeScale",
];

function fakeHost(search = "") {
  const elements = new Map(
    ELEMENT_IDS.map((id) => [id, new FakeElement(id)]),
  );
  elements.get("binaryTimeline").hidden = true;
  elements.get("binaryPlayPause").mark = new FakeElement("play-mark");
  const documentElement = new FakeElement("html");
  const defaultView = new FakeEventTarget();
  defaultView.location = { search };
  const document = {
    title: "original",
    documentElement,
    defaultView,
    getElementById(id) {
      return elements.get(id) ?? null;
    },
  };
  const ui = {
    observerValue: elements.get("observerValue"),
    rsValue: elements.get("rsValue"),
    shadowValue: elements.get("shadowValue"),
    massValue: elements.get("massValue"),
    accretionValue: elements.get("accretionValue"),
    exposureValue: elements.get("exposureValue"),
    timeScaleValue: elements.get("timeScaleValue"),
    qualityValue: elements.get("qualityValue"),
    accretion: elements.get("accretion"),
    exposure: elements.get("exposure"),
    timeScale: elements.get("timeScale"),
  };
  ui.accretion.value = "-4.20";
  ui.exposure.value = "1";
  ui.timeScale.value = "100";
  return { document, elements, ui };
}

async function localFetch(url) {
  const bytes = await readFile(new URL(url));
  return {
    ok: true,
    status: 200,
    async json() {
      return JSON.parse(bytes.toString("utf8"));
    },
    async arrayBuffer() {
      return bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      );
    },
  };
}

function baseFrame(time) {
  return {
    time,
    massSolar: 60,
    accretion: 0,
    exposure: 1,
    mode: 0,
    steps: 256,
    cameraPos: [0, 8, 50],
    cameraRadius: 50,
    forward: [0, 0, -1],
    fov: 0.9,
    right: [1, 0, 0],
    skyRotation: -2.576,
    up: [0, 1, 0],
    diskOuterRadius: 0,
    bloom: 0,
    observerVelocity: [0, 0, 0],
    observerBeta: 0,
  };
}

test("binary scene wires the strong-field runtime without losing legacy fallback", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = localFetch;
  try {
    const host = fakeHost();
    const state = {
      time: 0,
      running: true,
      distance: 50,
      phase: 0,
      orbitTilt: 0,
      exposure: 1,
      timeScale: 100,
      massSolar: 60,
      quality: 1,
    };
    const controls = {
      setRunning(running) {
        state.running = running;
      },
      requestRender() {},
    };
    const scene = await createBinaryApproxScene({
      document: host.document,
      ui: host.ui,
      state,
      controls,
      formatMass: (value) => `${value} Msol`,
      formatGravitationalRadius: (value) => `${value} rg`,
    });
    scene.initialize();
    assert.equal(
      host.elements.get("binarySlowMotion").listeners.get("click").size,
      1,
    );

    assert.equal(scene.rendererOptions.shaderBundle.id, "binary-strong-field-v1");
    assert.equal(scene.qualityPolicy.id, "m3-pro-strong-field-v1");
    assert.equal(scene.startsRunning, true);
    assert.deepEqual(scene.cameraDistanceLimits, { min: 34, max: 70 });
    const webgpuCapabilities = Object.freeze({
      api: "webgpu",
      backend: "WebGPU · Metal",
      progressiveAccumulation: "linear-hdr-running-average-v1",
    });
    scene.onRendererReady(webgpuCapabilities, {
      backend: "WebGPU · Metal",
      capabilities: webgpuCapabilities,
    });
    assert.match(
      host.elements.get("sceneStatus").textContent,
      /WebGPU · Metal/,
    );
    assert.match(
      host.elements.get("sceneStatus").textContent,
      /Real-time 3\+1 Hamiltonian strong-field ray tracing/,
    );
    assert.match(
      host.elements.get("sceneStatus").textContent,
      /Advanced diagnostics.*lookback time.*null residual.*integration cost/,
    );
    assert.equal(host.elements.get("modeHubble").textContent, "Ray outcome");
    assert.equal(host.elements.get("modeFrequency").textContent, "Frequency shift g");
    assert.equal(host.elements.get("modeHubble").hidden, true);
    assert.equal(host.elements.get("modeFrequency").hidden, true);
    assert.equal(host.elements.get("modeLookback").hidden, false);
    assert.equal(host.elements.get("modeNull").hidden, false);
    assert.equal(host.elements.get("modeError").textContent, "Integration-step cost");
    assert.equal(host.elements.get("modeError").hidden, false);
    assert.match(
      host.elements.get("physicsNote").innerHTML,
      /never drive WebGPU black-hole positions/,
    );

    const frame = scene.extendFrame(baseFrame(state.time));
    // Only the photographic mode is tone mapped; modes 1-5 are false colour.
    assert.equal(frame.diagnosticDisplay, false);
    for (const mode of [1, 2, 3, 4, 5]) {
      assert.equal(
        scene.extendFrame({ ...baseFrame(state.time), mode }).diagnosticDisplay,
        true,
      );
    }

    // Image orientation: the camera triad must be right-handed
    // (right x up = -forward), otherwise real sky maps render mirrored.
    for (const [phase, orbitTilt] of [[0, 0], [0.58, 0.575], [2.1, -0.4]]) {
      Object.assign(state, { phase, orbitTilt });
      const camera = scene.cameraFrame();
      const [r, u, f] = [camera.right, camera.up, camera.forward];
      const handed = [
        r[1] * u[2] - r[2] * u[1],
        r[2] * u[0] - r[0] * u[2],
        r[0] * u[1] - r[1] * u[0],
      ];
      handed.forEach((component, axis) => {
        assert.ok(Math.abs(component + f[axis]) < 1e-12, "camera is not right-handed");
      });
    }
    Object.assign(state, { phase: 0, orbitTilt: 0 });
    assert.ok(frame.sceneStrongFieldUniforms instanceof Float32Array);
    assert.equal(frame.sceneStrongFieldUniforms.length, 44);
    assert.ok([...frame.sceneStrongFieldUniforms].every(Number.isFinite));
    assert.equal(frame.sceneBinaryState.length, 4);
    assert.equal(frame.sceneBinaryMasses.length, 4);
    assert.equal(frame.steps, 256);

    // RK4 tiers: integrator = [minimum step, maximum step, step fraction,
    // relative energy-drift gate]; domain = [escape radius, maximum lookback,
    // horizon backstop, maximum RK4 steps]. The test camera sits at r = 50 M,
    // so the escape sphere is at least 2 r + 16 = 116 M.
    const expectedTiers = {
      emergency: {
        integrator: [0.02, 64, 0.60, 0.05],
        domain: [116, 296, 0.02, 48],
      },
      survival: {
        integrator: [0.02, 64, 0.60, 0.05],
        domain: [116, 296, 0.02, 56],
      },
      interactive: {
        integrator: [0.01, 64, 0.60, 0.02],
        domain: [116, 296, 0.02, 80],
      },
      balanced: {
        integrator: [0.01, 64, 0.30, 0.01],
        domain: [160, 400, 0.01, 144],
      },
      fine: {
        integrator: [0.005, 64, 0.22, 0.005],
        domain: [200, 480, 0.005, 192],
      },
    };
    for (const [tierId, expected] of Object.entries(expectedTiers)) {
      const qualityFrame = scene.applyStrongFieldQuality(
        {
          ...frame,
          strongFieldQuality: { tierId },
        },
        { qualityTierId: tierId },
      );
      assert.deepEqual(
        qualityFrame.sceneStrongIntegrator,
        expected.integrator,
      );
      assert.deepEqual(qualityFrame.sceneStrongDomain, expected.domain);
      assert.deepEqual(
        qualityFrame.sceneStrongDiagnostics,
        [4, 180, 0.055, 0],
      );
      assert.ok(qualityFrame.sceneStrongDomain[3] <= 192);
    }
    const farCameraFrame = scene.applyStrongFieldQuality(
      {
        ...frame,
        cameraRadius: 100,
        steps: 64,
        strongFieldQuality: { tierId: "survival" },
      },
      { qualityTierId: "survival" },
    );
    assert.equal(farCameraFrame.sceneStrongDomain[0], 216);
    assert.equal(farCameraFrame.sceneStrongDomain[1], 496);

    const webglCapabilities = Object.freeze({
      api: "webgl2",
      backend: "WebGL2 · Compatibility",
    });
    scene.onRendererReady(webglCapabilities, {
      backend: "WebGL2 · Compatibility",
      capabilities: webglCapabilities,
    });
    assert.match(
      host.elements.get("sceneStatus").textContent,
      /Legacy two-centre weak-field preview/,
    );
    assert.match(
      host.elements.get("sceneStatus").textContent,
      /strong-field physical parity/,
    );
    for (const id of [
      "modeHubble",
      "modeFrequency",
      "modeLookback",
      "modeNull",
      "modeError",
      "transferAdvancedDiagnostics",
    ]) {
      assert.equal(host.elements.get(id).hidden, true);
    }

    const firstRevision = scene.renderRevision(frame);
    const repeatedRevision = scene.renderRevision(frame);
    assert.equal(firstRevision.physics, frame.time);
    assert.equal(firstRevision.transport, repeatedRevision.transport);
    const changedRevision = scene.renderRevision({ ...frame, mode: 4 });
    assert.ok(changedRevision.transport > repeatedRevision.transport);
    const legacyStepRevision = scene.renderRevision({
      ...frame,
      mode: 4,
      steps: frame.steps - 32,
    });
    assert.equal(
      legacyStepRevision.transport,
      changedRevision.transport,
      "legacy pre-scheduler step budgets must not invalidate strong-field history",
    );

    scene.dispose();
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("binary scene localizes dynamic strong-field status in Chinese", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = localFetch;
  try {
    const host = fakeHost("?lang=zh-CN");
    const state = {
      time: 0,
      running: true,
      distance: 50,
      phase: 0,
      orbitTilt: 0,
      exposure: 1,
      timeScale: 100,
      massSolar: 60,
      quality: 1,
    };
    const scene = await createBinaryApproxScene({
      document: host.document,
      ui: host.ui,
      state,
      controls: {
        setRunning(running) {
          state.running = running;
        },
        requestRender() {},
      },
      formatMass: (value) => `${value} Msol`,
      formatGravitationalRadius: (value) => `${value} rg`,
    });
    scene.initialize();
    const capabilities = Object.freeze({
      api: "webgpu",
      backend: "WebGPU · Metal",
      progressiveAccumulation: "linear-hdr-running-average-v1",
    });
    scene.onRendererReady(capabilities, {
      backend: capabilities.backend,
      capabilities,
    });
    assert.match(host.elements.get("sceneStatus").textContent, /实时 3\+1 Hamiltonian 强场光追/);
    assert.equal(host.elements.get("modeError").textContent, "积分步数成本");
    assert.match(host.elements.get("physicsNote").innerHTML, /绝不进入 WebGPU 黑洞位置/);
    scene.dispose();
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("binary scene supports a reproducible paused protocol-time permalink", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = localFetch;
  try {
    const host = fakeHost("?binaryTime=-16.8&paused=1");
    const state = {
      time: 0,
      running: true,
      distance: 50,
      phase: 0,
      orbitTilt: 0,
      exposure: 1,
      timeScale: 100,
      massSolar: 60,
      quality: 1,
    };
    const scene = await createBinaryApproxScene({
      document: host.document,
      ui: host.ui,
      state,
      controls: {
        setRunning(running) {
          state.running = running;
        },
        requestRender() {},
      },
      formatMass: (value) => `${value} Msol`,
      formatGravitationalRadius: (value) => `${value} rg`,
    });
    assert.equal(scene.startsRunning, false);
    scene.initialize();
    assert.ok(Math.abs(state.time + 16.8) < 1e-9);
    assert.equal(host.elements.get("binaryTimeValue").textContent, "t = −16.80 M");
    scene.dispose();
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("dual-disk scene derives two C2-truncated mini-disks from provider coordinates", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = localFetch;
  try {
    const host = fakeHost("?scene=binary-dual-disk&paused=1");
    const state = {
      time: 0,
      running: false,
      distance: 50,
      phase: 0,
      orbitTilt: 0,
      accretion: 10 ** -4.2,
      exposure: 1,
      timeScale: 100,
      massSolar: 6.5e9,
      quality: 1,
    };
    const scene = await createBinaryDualDiskScene({
      document: host.document,
      ui: host.ui,
      state,
      controls: {
        setRunning(running) {
          state.running = running;
        },
        requestRender() {},
      },
      formatMass: (value) => `${value} Msol`,
      formatGravitationalRadius: (value) => `${value} rg`,
    });
    scene.initialize();

    assert.equal(scene.id, "binary-dual-disk");
    assert.equal(
      scene.rendererOptions.shaderBundle.id,
      "binary-dual-disk-strong-field-v1",
    );
    assert.equal(host.document.title, "Dual-disk binary · Deep-space observatory");
    assert.equal(host.document.documentElement.classList.contains("scene-binary"), true);
    assert.equal(host.document.documentElement.classList.contains("scene-binary-dual-disk"), true);
    assert.equal(host.elements.get("accretionControl").attributeValues.has("aria-hidden"), false);
    assert.equal(host.elements.get("accretionLabel").textContent, "Per-disk emission proxy");
    assert.equal(host.ui.accretion.value, "-1.70");
    assert.ok(Math.abs(state.accretion - 10 ** -1.7) < 1e-12);

    const capabilities = Object.freeze({
      api: "webgpu",
      backend: "WebGPU · Metal",
      progressiveAccumulation: "linear-hdr-running-average-v1",
    });
    scene.onRendererReady(capabilities, {
      backend: capabilities.backend,
      capabilities,
    });
    assert.equal(host.ui.accretion.disabled, false);
    assert.match(host.elements.get("sceneStatus").textContent, /Idealized thin mini-disks/);
    assert.match(
      host.elements.get("sceneStatus").textContent,
      /No GRMHD or self-consistent radiative transfer/,
    );

    const initialFrame = scene.extendFrame({
      ...baseFrame(state.time),
      massSolar: state.massSolar,
      accretion: state.accretion,
    });
    const disk = initialFrame.sceneStrongAccretionUniforms;
    assert.ok(disk instanceof Float32Array);
    assert.equal(disk.length, 20);
    assert.equal(disk[0], 1);
    assert.ok(Math.abs(disk[1] - 0.8) < 1e-6);
    assert.equal(disk[2], 1);
    assert.equal(disk[3], 6);
    assert.deepEqual([...disk.slice(4, 7)], [0, 1, 0]);
    assert.deepEqual([...disk.slice(12, 15)], [0, 1, 0]);
    assert.ok(Math.abs(disk[7] - 3) < 1e-5);
    assert.ok(Math.abs(disk[15] - 3) < 1e-5);
    assert.ok(disk[8] > disk[7]);
    assert.ok(disk[16] > disk[15]);
    assert.ok(disk[8] <= 10 && disk[16] <= 10);
    assert.ok(disk[10] > 0.99 && disk[18] > 0.99);
    const providerSeparation = initialFrame.sceneBinaryState[0];
    const sxsSeparation = Number(
      host.elements.get("observerValue").innerHTML.match(
        /SXS<\/sub> ([0-9.]+) M/,
      )?.[1],
    );
    assert.ok(Number.isFinite(sxsSeparation));
    assert.ok(
      Math.abs(providerSeparation - sxsSeparation) > 0.1,
      "disk geometry must not reuse the displayed gauge-centroid separation",
    );
    const expectedOuterRadius = Math.min(
      10,
      TIDAL_TRUNCATION_FACTOR
        * eggletonRocheLobeFraction(1)
        * providerSeparation,
    );
    assert.ok(Math.abs(disk[8] - expectedOuterRadius) < 1e-5);
    assert.ok(Math.abs(disk[16] - expectedOuterRadius) < 1e-5);
    assert.match(host.elements.get("rsValue").textContent, /A .* M · B .* M/);
    assert.equal(
      host.elements.get("shadowValue").textContent,
      "Two idealized mini-disks active",
    );

    const initialRevision = scene.renderRevision(initialFrame);
    const changedDisk = Float32Array.from(disk);
    changedDisk[10] = 0.5;
    const changedRevision = scene.renderRevision({
      ...initialFrame,
      sceneStrongAccretionUniforms: changedDisk,
    });
    assert.ok(changedRevision.transport > initialRevision.transport);
    const restoredDiskRevision = scene.renderRevision(initialFrame);
    const changedMassRevision = scene.renderRevision({
      ...initialFrame,
      massSolar: initialFrame.massSolar * 2,
    });
    assert.ok(changedMassRevision.transport > restoredDiskRevision.transport);

    state.time = -5;
    const postMergerFrame = scene.extendFrame({
      ...baseFrame(state.time),
      massSolar: state.massSolar,
      accretion: state.accretion,
    });
    assert.equal(postMergerFrame.sceneStrongAccretionUniforms[10], 0);
    assert.equal(postMergerFrame.sceneStrongAccretionUniforms[18], 0);
    assert.equal(
      host.elements.get("shadowValue").textContent,
      "Post-merger emission unmodeled",
    );

    const webglCapabilities = Object.freeze({
      api: "webgl2",
      backend: "WebGL2 · Compatibility",
    });
    scene.onRendererReady(webglCapabilities, {
      backend: webglCapabilities.backend,
      capabilities: webglCapabilities,
    });
    assert.equal(host.ui.accretion.disabled, true);
    assert.match(host.elements.get("sceneStatus").textContent, /no disk parity/i);
    assert.equal(
      host.elements.get("shadowValue").textContent,
      "Dual-disk emission unavailable in WebGL2 preview",
    );
    assert.equal(
      host.elements.get("modeScience").attributeValues.has("title"),
      false,
    );

    scene.dispose();
    assert.equal(
      host.elements.get("binarySlowMotion").listeners.get("click").size,
      0,
    );
    assert.equal(host.ui.accretion.value, "-4.20");
    assert.equal(host.ui.accretion.disabled, false);
    assert.equal(host.ui.accretion.attributeValues.has("aria-describedby"), false);
    assert.equal(host.ui.accretion.attributeValues.has("aria-valuetext"), false);
    assert.equal(host.elements.get("accretionLabel").textContent, "");
    assert.equal(host.elements.get("modeScience").attributeValues.has("title"), false);
    assert.ok(Math.abs(state.accretion - 10 ** -4.2) < 1e-12);
    assert.equal(host.document.documentElement.classList.contains("scene-binary"), false);
    assert.equal(host.document.documentElement.classList.contains("scene-binary-dual-disk"), false);
    scene.initialize();
    assert.equal(
      host.elements.get("binarySlowMotion").listeners.get("click").size,
      1,
    );
    const slowMotionBeforeClick = host.elements.get("binarySlowMotion").textContent;
    host.elements.get("binarySlowMotion").dispatchEvent({ type: "click" });
    assert.notEqual(
      host.elements.get("binarySlowMotion").textContent,
      slowMotionBeforeClick,
    );
    scene.dispose();
    assert.equal(
      host.elements.get("binarySlowMotion").listeners.get("click").size,
      0,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("binary stylesheet removes outcome and frequency from the primary mode row", async () => {
  const stylesheet = await readFile(
    new URL("../src/styles.css", import.meta.url),
    "utf8",
  );
  assert.match(
    stylesheet,
    /\.scene-binary-approx \.mode-switch\s*\{[^}]*grid-template-columns:\s*1fr;/s,
  );
  assert.match(
    stylesheet,
    /\.scene-binary-approx #modeHubble,[\s\S]*\.scene-binary-approx #modeFrequency\s*\{[^}]*display:\s*none;/,
  );
  assert.match(
    stylesheet,
    /\.scene-binary-approx\.renderer-webgpu \.diagnostic-only\[hidden\]\s*\{[^}]*display:\s*none;/s,
  );
  assert.match(
    stylesheet,
    /\.scene-binary-dual-disk \.mode-switch\s*\{[^}]*grid-template-columns:\s*1fr;/s,
  );
  assert.doesNotMatch(
    stylesheet,
    /\.scene-binary-dual-disk #accretionControl\s*\{[^}]*display:\s*none;/s,
  );
});
