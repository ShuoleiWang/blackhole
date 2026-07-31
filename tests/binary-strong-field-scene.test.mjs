import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { createBinaryApproxScene } from "../src/scenes/binary-approx-scene.js";

globalThis.crypto ??= webcrypto;

class FakeEventTarget {
  constructor() {
    this.listeners = new Map();
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
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
    this.innerHTML = "";
    this.textContent = "";
    this.value = "";
    this.disabled = false;
    this.hidden = false;
    this.mark = null;
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
      /实时 3\+1 Hamiltonian 强场光追/,
    );
    assert.match(
      host.elements.get("sceneStatus").textContent,
      /光线结果.*回溯时间.*g.*零性残差.*积分成本/,
    );
    assert.equal(host.elements.get("modeHubble").textContent, "光线结果");
    assert.equal(host.elements.get("modeFrequency").textContent, "频移因子 g");
    assert.equal(host.elements.get("modeLookback").hidden, false);
    assert.equal(host.elements.get("modeNull").hidden, false);
    assert.equal(host.elements.get("modeError").textContent, "积分步数成本");
    assert.equal(host.elements.get("modeError").hidden, false);
    assert.match(
      host.elements.get("physicsNote").innerHTML,
      /绝不进入 WebGPU 黑洞位置/,
    );

    const frame = scene.extendFrame(baseFrame(state.time));
    assert.ok(frame.sceneStrongFieldUniforms instanceof Float32Array);
    assert.equal(frame.sceneStrongFieldUniforms.length, 44);
    assert.ok([...frame.sceneStrongFieldUniforms].every(Number.isFinite));
    assert.equal(frame.sceneBinaryState.length, 4);
    assert.equal(frame.sceneBinaryMasses.length, 4);
    assert.equal(frame.steps, 256);

    const expectedTiers = {
      emergency: {
        integrator: [0.065, 3.5, 2.7, 0.34],
        steps: 52,
        domain: [58, 164, 0.30, 268],
        stepCurveExponent: 0.50,
      },
      survival: {
        integrator: [0.050, 3.5, 3.0, 0.25],
        steps: 60,
        domain: [60, 180, 0.24, 256],
        stepCurveExponent: 0.65,
      },
      interactive: {
        integrator: [0.035, 3.0, 3.3, 0.18],
        steps: 96,
        domain: [64, 200, 0.16, 224],
        stepCurveExponent: 0.80,
      },
      balanced: {
        integrator: [0.018, 1.10, 3.6, 0.10],
        steps: 160,
        domain: [80, 220, 0.08, 160],
        stepCurveExponent: 1.50,
      },
      fine: {
        integrator: [0.010, 0.85, 4.0, 0.05],
        steps: 288,
        domain: [80, 220, 0.04, 32],
        stepCurveExponent: 1.90,
      },
    };
    for (const [tierId, expected] of Object.entries(expectedTiers)) {
      const qualityFrame = scene.applyStrongFieldQuality(
        {
          ...frame,
          steps: expected.steps,
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
        [4, 180, 0.055, expected.stepCurveExponent],
      );
      assert.ok(
        Math.min(qualityFrame.steps, 320)
          + qualityFrame.sceneStrongDomain[3]
          <= 320,
      );
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
    assert.equal(farCameraFrame.sceneStrongDomain[0], 108);
    assert.equal(farCameraFrame.sceneStrongDomain[1], 248);

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
      /旧 two-centre weak-field 预览/,
    );
    assert.match(
      host.elements.get("sceneStatus").textContent,
      /强场物理等价性/,
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
