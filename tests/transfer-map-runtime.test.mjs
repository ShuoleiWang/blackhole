import assert from "node:assert/strict";
import { createHash, webcrypto } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  decodeTransferMapChunks,
  loadTransferMap,
  readTransferMapRecord,
  TransferMapContractError,
  validateTransferMapManifest,
} from "../src/transfer-map-loader.js";
import {
  createTransferMapShaderBundle,
  transferMapTraceFragmentGLSL,
  transferMapTraceFragmentWGSL,
} from "../src/transfer-map-shaders.js";
import {
  canvasPointToTransferPixel,
  createTransferMapReferenceScene,
  diagnosticHref,
  diagnosticModeFromSearch,
  diagnosticRangeForMode,
  readoutsFromManifest,
  REFERENCE_MANIFEST_SHA256,
  referenceHref,
  resolveTrustedReference,
  TRANSFER_MAP_DIAGNOSTIC_MODES,
  TransferMapSceneLoadError,
  transferPixelToCanvasPoint,
  TRUSTED_REFERENCE_REGISTRY,
} from "../src/scenes/transfer-map-reference-scene.js";

const root = new URL("../", import.meta.url);
const datasetRoot = new URL(
  "assets/transfer-maps/schwarzschild-reference-v1/",
  root,
);
const manifestUrl = new URL("manifest.json", datasetRoot);
const manifestBytes = new Uint8Array(await readFile(manifestUrl));
const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
const kerrManifestUrl = new URL(
  "assets/transfer-maps/kerr-remnant-reference-v1/manifest.json",
  root,
);
const kerrDatasetRoot = new URL("./", kerrManifestUrl);
const kerrManifestBytes = new Uint8Array(await readFile(kerrManifestUrl));
const kerrManifest = JSON.parse(
  new TextDecoder().decode(kerrManifestBytes),
);
const chunkPayloads = new Map(
  await Promise.all(manifest.chunks.map(async (chunk) => [
    chunk.uri,
    new Uint8Array(await readFile(new URL(chunk.uri, datasetRoot))),
  ])),
);
let cachedDecodedDataset;

function decodedDataset() {
  cachedDecodedDataset ||= decodeTransferMapChunks(manifest, chunkPayloads);
  return cachedDecodedDataset;
}

class FakeClassList {
  constructor(element) {
    this.element = element;
    this.values = new Set();
  }

  contains(name) {
    return this.values.has(name);
  }

  add(...names) {
    names.forEach((name) => this.values.add(name));
    this.sync();
  }

  remove(...names) {
    names.forEach((name) => this.values.delete(name));
    this.sync();
  }

  toggle(name, force) {
    const enabled = force === undefined ? !this.values.has(name) : Boolean(force);
    if (enabled) {
      this.values.add(name);
    } else {
      this.values.delete(name);
    }
    this.sync();
    return enabled;
  }

  fromAttribute(value) {
    this.values = new Set(String(value).split(/\s+/).filter(Boolean));
  }

  sync() {
    if (this.values.size) {
      this.element.attributeValues.set("class", [...this.values].join(" "));
    } else {
      this.element.attributeValues.delete("class");
    }
  }
}

class FakeEventTarget {
  constructor() {
    this.listeners = new Map();
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) || new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type, listener) {
    this.listeners.get(type)?.delete(listener);
  }

  listenerCount(type) {
    return this.listeners.get(type)?.size || 0;
  }

  dispatch(type, event = {}) {
    for (const listener of this.listeners.get(type) || []) {
      listener({ type, ...event });
    }
  }
}

class FakeElement extends FakeEventTarget {
  constructor(tagName = "div", id = "") {
    super();
    this.tagName = tagName.toUpperCase();
    this.attributeValues = new Map();
    this.classList = new FakeClassList(this);
    this.children = [];
    this.style = {};
    this.textContent = "";
    this.innerHTML = "";
    this.value = "";
    this.disabled = false;
    this.rect = { left: 0, top: 0, width: 1600, height: 900 };
    if (id) {
      this.setAttribute("id", id);
    }
  }

  get attributes() {
    return [...this.attributeValues].map(([name, value]) => ({ name, value }));
  }

  setAttribute(name, value) {
    this.attributeValues.set(name, String(value));
    if (name === "class") {
      this.classList.fromAttribute(value);
    }
  }

  removeAttribute(name) {
    this.attributeValues.delete(name);
    if (name === "class") {
      this.classList.fromAttribute("");
    }
  }

  get hidden() {
    return this.attributeValues.has("hidden");
  }

  set hidden(value) {
    if (value) {
      this.setAttribute("hidden", "");
    } else {
      this.removeAttribute("hidden");
    }
  }

  get href() {
    return this.attributeValues.get("href") || "";
  }

  set href(value) {
    this.setAttribute("href", value);
  }

  append(...children) {
    this.children.push(...children);
  }

  replaceChildren(...children) {
    this.children = [...children];
    this.textContent = "";
  }

  getBoundingClientRect() {
    return this.rect;
  }

  focus() {
    this.focused = true;
  }
}

const SCENE_ELEMENT_IDS = [
  "universe",
  "panel",
  "togglePanel",
  "sceneEyebrow",
  "panelTitle",
  "observerLabel",
  "radiusLabel",
  "shadowLabel",
  "massLabel",
  "physicsNote",
  "sceneStatus",
  "binaryTimeline",
  "desktopHint",
  "modeSwitch",
  "modeScience",
  "modeHubble",
  "modeLookback",
  "modeFrequency",
  "modeNull",
  "modeError",
  "transferReferenceSwitch",
  "referenceSchwarzschild",
  "referenceKerr",
  "transferMapInspector",
  "transferInspectorCoordinates",
  "transferInspectorDirection",
  "transferInspectorFrequency",
  "transferInspectorLookback",
  "transferInspectorNull",
  "transferInspectorError",
  "transferInspectorOutcome",
  "transferInspectorValidity",
  "transferInspectorRaw",
  "transferPixelMarker",
  "toggleMotion",
  "resetView",
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
];

function fakeSceneHost(href = "https://blackhole.test/?scene=transfer-map-reference") {
  const elements = new Map(
    SCENE_ELEMENT_IDS.map((id) => [id, new FakeElement("div", id)]),
  );
  elements.get("universe").tagName = "CANVAS";
  elements.get("transferReferenceSwitch").hidden = true;
  elements.get("transferMapInspector").hidden = true;
  elements.get("transferPixelMarker").hidden = true;
  elements.get("binaryTimeline").hidden = true;
  const touchHint = new FakeElement("span");
  const windowRef = new FakeEventTarget();
  const location = new URL(href);
  windowRef.location = location;
  windowRef.history = {
    state: null,
    replaceState(state, _title, nextHref) {
      this.state = state;
      const next = new URL(nextHref);
      location.href = next.href;
    },
  };

  class FakeDocument extends FakeEventTarget {
    constructor() {
      super();
      this.title = "Original title";
      this.documentElement = new FakeElement("html");
      this.defaultView = windowRef;
    }

    getElementById(id) {
      return elements.get(id) || null;
    }

    querySelector(selector) {
      return selector === ".touch-hint" ? touchHint : null;
    }

    createElement(tagName) {
      return new FakeElement(tagName);
    }
  }

  const document = new FakeDocument();
  const ui = Object.fromEntries([
    "modeScience",
    "modeHubble",
    "toggleMotion",
    "resetView",
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
  ].map((id) => [id, elements.get(id)]));
  return { document, elements, ui, windowRef, location };
}

function sceneState() {
  return {
    running: true,
    distance: 50,
    phase: 0.55,
    orbitTilt: 0.42,
    mode: 0,
    exposure: 1,
    quality: 1,
  };
}

function testRegistry(datasetId = manifest.id) {
  return Object.freeze({
    schwarzschild: Object.freeze({
      key: "schwarzschild",
      datasetId,
      title: "Test reference",
      manifestUrl: "https://blackhole.test/reference/manifest.json",
      expectedManifestSha256: "a".repeat(64),
    }),
  });
}

function arrayBuffer(bytes) {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function localFetch({ corruptChunk = false } = {}) {
  const webRoot = new URL("https://blackhole.test/reference/");
  return async (input) => {
    const url = new URL(input);
    const relative = decodeURIComponent(url.pathname.slice(webRoot.pathname.length));
    let bytes;
    if (relative === "manifest.json") {
      bytes = manifestBytes;
    } else if (relative === "manifest.sha256") {
      bytes = new TextEncoder().encode(
        `${REFERENCE_MANIFEST_SHA256}  manifest.json\n`,
      );
    } else {
      bytes = chunkPayloads.get(relative);
    }
    if (!bytes) {
      return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };
    }
    const responseBytes = new Uint8Array(bytes);
    if (corruptChunk && relative === manifest.chunks[0].uri) {
      responseBytes[0] ^= 0x01;
    }
    return {
      ok: true,
      status: 200,
      arrayBuffer: async () => arrayBuffer(responseBytes),
    };
  };
}

test("bundled reference manifest is the pinned runtime trust root", async () => {
  const digest = createHash("sha256").update(manifestBytes).digest("hex");
  assert.equal(digest, REFERENCE_MANIFEST_SHA256);
  const kerrDigest = createHash("sha256")
    .update(kerrManifestBytes)
    .digest("hex");
  assert.equal(
    kerrDigest,
    TRUSTED_REFERENCE_REGISTRY["kerr-remnant"].expectedManifestSha256,
  );
  assert.equal(kerrManifest.id, "kerr-remnant-reference-v1");
  const metadata = validateTransferMapManifest(manifest);
  assert.equal(metadata.width, 1024);
  assert.equal(metadata.height, 576);
  assert.equal(metadata.recordCount, 589824);
  assert.equal(
    manifest.sampling.interpolation.continuous,
    "none-nearest-texel-center",
  );
  assert.equal(
    manifest.sampling.interpolation.escapeDirection,
    "nearest-no-blend",
  );
  assert.equal(manifest.accuracy.nrConvergence.status, "not-applicable");
  assert.equal(manifest.accuracy.constraintNorms.status, "not-applicable");
});

test("trusted reference registry accepts only pinned keys and dataset identities", () => {
  const selected = resolveTrustedReference(new URLSearchParams(""));
  assert.equal(selected.key, "schwarzschild");
  assert.equal(selected.datasetId, manifest.id);
  assert.equal(selected.expectedManifestSha256, REFERENCE_MANIFEST_SHA256);
  assert.match(selected.manifestUrl, /schwarzschild-reference-v1\/manifest\.json$/);
  assert.equal(TRUSTED_REFERENCE_REGISTRY["kerr-remnant"].datasetId, (
    "kerr-remnant-reference-v1"
  ));
  assert.match(
    TRUSTED_REFERENCE_REGISTRY["kerr-remnant"].manifestUrl,
    /kerr-remnant-reference-v1\/manifest\.json$/,
  );

  assert.throws(
    () => resolveTrustedReference(new URLSearchParams("reference=https://evil.test/map")),
    /unknown trusted reference key/,
  );
  if (TRUSTED_REFERENCE_REGISTRY["kerr-remnant"].expectedManifestSha256 === null) {
    assert.throws(
      () => resolveTrustedReference(
        new URLSearchParams("reference=kerr-remnant"),
      ),
      /unavailable until a pinned manifest SHA-256/,
    );
  } else {
    assert.match(
      TRUSTED_REFERENCE_REGISTRY["kerr-remnant"].expectedManifestSha256,
      /^[0-9a-f]{64}$/,
    );
  }

  const injected = {
    "kerr-remnant": {
      key: "kerr-remnant",
      datasetId: "kerr-remnant-reference-v1",
      title: "Injected test",
      manifestUrl: "https://blackhole.test/kerr/manifest.json",
      expectedManifestSha256: "b".repeat(64),
    },
  };
  assert.equal(
    resolveTrustedReference(
      new URLSearchParams("reference=kerr-remnant"),
      injected,
    ).expectedManifestSha256,
    "b".repeat(64),
  );
});

test("diagnostic modes have stable query identities and preserve reference selection", () => {
  assert.deepEqual(
    TRANSFER_MAP_DIAGNOSTIC_MODES.map((mode) => mode.id),
    [
      "sky",
      "outcome",
      "lookback",
      "frequency-shift",
      "null-residual",
      "projection-error",
    ],
  );
  for (const [mode, definition] of TRANSFER_MAP_DIAGNOSTIC_MODES.entries()) {
    const href = diagnosticHref(
      "https://blackhole.test/?scene=transfer-map-reference&reference=kerr-remnant",
      mode,
    );
    const parsed = new URL(href);
    assert.equal(parsed.searchParams.get("reference"), "kerr-remnant");
    assert.equal(
      parsed.searchParams.get("diagnostic"),
      definition.id === "sky" ? null : definition.id,
    );
    assert.equal(diagnosticModeFromSearch(parsed.searchParams), mode);
  }
  assert.equal(
    diagnosticModeFromSearch(new URLSearchParams("diagnostic=not-a-mode")),
    0,
  );
  assert.match(
    referenceHref(
      "https://blackhole.test/?scene=transfer-map-reference&diagnostic=null-residual",
      "kerr-remnant",
    ),
    /reference=kerr-remnant/,
  );
});

test("runtime loader authenticates, assembles, and preserves the v1 ray state", async () => {
  const progress = [];
  const dataset = await loadTransferMap(
    "https://blackhole.test/reference/manifest.json",
    {
      expectedManifestSha256: REFERENCE_MANIFEST_SHA256,
      fetchImpl: localFetch(),
      cryptoImpl: webcrypto,
      onProgress(update) {
        progress.push(update.phase);
      },
    },
  );
  assert.equal(dataset.counts.escaped + dataset.counts.captured, 1024 * 576);
  assert.equal(dataset.records.byteLength, 1024 * 576 * 32);
  assert.equal(dataset.primary.length, 1024 * 576 * 4);
  assert.equal(dataset.metrics.length, 1024 * 576 * 4);
  assert.ok(progress.includes("manifest"));
  assert.ok(progress.includes("sidecar"));
  assert.ok(progress.includes("chunks"));
  assert.equal(progress.at(-1), "decoded");

  const escapedState = dataset.metrics[3];
  assert.equal(escapedState, 0 + 256 * 255 + 65536 * 0x1f);
  const centrePixel = (288 * 1024 + 512) * 4;
  const capturedState = dataset.metrics[centrePixel + 3];
  assert.equal(capturedState, 1 + 256 * 0 + 65536 * 0x1c);
});

test("bundled Kerr reference passes the browser decoder without unusable rays", async () => {
  const kerrChunkPayloads = new Map(
    await Promise.all(kerrManifest.chunks.map(async (chunk) => [
      chunk.uri,
      new Uint8Array(await readFile(new URL(chunk.uri, kerrDatasetRoot))),
    ])),
  );
  const dataset = decodeTransferMapChunks(kerrManifest, kerrChunkPayloads);
  assert.equal(dataset.width, 1024);
  assert.equal(dataset.height, 576);
  assert.equal(dataset.counts.escaped, 558_684);
  assert.equal(dataset.counts.captured, 31_140);
  assert.equal(dataset.counts.unresolved, 0);
  assert.ok(dataset.diagnosticRanges.frequencyShift.minimum > 1);
  assert.ok(dataset.diagnosticRanges.projectionError.maximum < 0.004);
  const centre = readTransferMapRecord(dataset, 512, 288);
  assert.equal(centre.rayOutcomeName, "captured");
  assert.equal(centre.captureTargetId, "remnant");
});

test("decoded diagnostics and the pixel inspector preserve authenticated records", () => {
  const dataset = decodedDataset();
  for (const name of [
    "frequencyShift",
    "lookback",
    "nullResidual",
    "projectionError",
  ]) {
    const range = dataset.diagnosticRanges[name];
    assert.ok(Number.isFinite(range.minimum));
    assert.ok(Number.isFinite(range.maximum));
    assert.ok(range.maximum >= range.minimum);
    assert.ok(range.minimumPositive >= 0);
  }
  assert.ok(dataset.diagnosticRanges.lookback.maximum > (
    dataset.diagnosticRanges.lookback.minimum
  ));
  assert.ok(dataset.diagnosticRanges.nullResidual.maximum <= 5e-12);
  assert.ok(dataset.diagnosticRanges.projectionError.maximum <= 0.25);

  const centre = readTransferMapRecord(dataset, 512, 288);
  assert.equal(centre.byteOffset, (288 * 1024 + 512) * 32);
  assert.equal(centre.rayOutcomeName, "captured");
  assert.equal(centre.captureTargetId, "BH");
  assert.equal(centre.rawBytes.byteLength, 32);
  assert.equal(centre.rawHex.split(" ").length, 32);
  assert.equal(centre.validityMask, 0x1c);

  const escaped = readTransferMapRecord(dataset, 0, 0);
  assert.equal(escaped.rayOutcomeName, "escaped");
  assert.equal(escaped.captureTargetId, null);
  assert.equal(escaped.validityMask, 0x1f);
  assert.ok(Math.abs(Math.hypot(...escaped.escapeDirection) - 1) < 1e-6);
  assert.throws(
    () => readTransferMapRecord(dataset, dataset.width, 0),
    /lies outside the dataset/,
  );

  for (let mode = 0; mode < TRANSFER_MAP_DIAGNOSTIC_MODES.length; mode += 1) {
    const range = diagnosticRangeForMode(dataset, mode);
    assert.equal(range.length, 4);
    assert.ok(Number.isFinite(range[0]));
    assert.ok(Number.isFinite(range[1]));
    assert.ok(range[1] > range[0]);
  }
});

test("runtime loader rejects a chunk that differs from its pinned hash", async () => {
  await assert.rejects(
    loadTransferMap(
      "https://blackhole.test/reference/manifest.json",
      {
        expectedManifestSha256: REFERENCE_MANIFEST_SHA256,
        fetchImpl: localFetch({ corruptChunk: true }),
        cryptoImpl: webcrypto,
      },
    ),
    /hash mismatch/,
  );
});

test("runtime loader refuses to fetch without a pinned manifest trust root", async () => {
  await assert.rejects(
    loadTransferMap(
      "https://blackhole.test/reference/manifest.json",
      {
        fetchImpl: localFetch(),
        cryptoImpl: webcrypto,
      },
    ),
    /pinned lowercase manifest SHA-256 trust root is required/,
  );
});

test("decoder fails closed on outcome masks, null residuals, and render policy", () => {
  const invalidMaskChunks = new Map(
    [...chunkPayloads].map(([uri, bytes]) => [uri, new Uint8Array(bytes)]),
  );
  new DataView(invalidMaskChunks.get(manifest.chunks[0].uri).buffer)
    .setUint16(30, 0, true);
  assert.throws(
    () => decodeTransferMapChunks(manifest, invalidMaskChunks),
    TransferMapContractError,
  );

  const invalidResidualChunks = new Map(
    [...chunkPayloads].map(([uri, bytes]) => [uri, new Uint8Array(bytes)]),
  );
  new DataView(invalidResidualChunks.get(manifest.chunks[0].uri).buffer)
    .setFloat32(20, 1e-4, true);
  assert.throws(
    () => decodeTransferMapChunks(manifest, invalidResidualChunks),
    /exceeds the declared/,
  );

  const unrenderable = structuredClone(manifest);
  unrenderable.renderable = false;
  assert.throws(
    () => validateTransferMapManifest(unrenderable),
    /not approved for rendering/,
  );

  const unresolved = structuredClone(manifest);
  unresolved.accuracy.unresolvedFraction = 1e-4;
  unresolved.accuracy.outcomeFractions.unusable = 1e-4;
  assert.throws(
    () => validateTransferMapManifest(unresolved),
    /refuses unresolved rays/,
  );

  const blendedDirections = structuredClone(manifest);
  blendedDirections.sampling.interpolation.continuous = "validity-strict-linear";
  blendedDirections.sampling.interpolation.escapeDirection = (
    "validity-strict-linear-then-normalize"
  );
  assert.throws(
    () => validateTransferMapManifest(blendedDirections),
    /unsupported sampling or interpolation convention/,
  );

  const nonIcrsDirections = structuredClone(manifest);
  nonIcrsDirections.coordinates.sky.escapeDirectionFrame = "world";
  assert.throws(
    () => validateTransferMapManifest(nonIcrsDirections),
    /canonical ICRS sky directions/,
  );
});

test("manifest readouts distinguish Kerr rKS from Cartesian Euclidean radius", () => {
  const kerrManifest = structuredClone(manifest);
  const spin = 0.6;
  const kerrRadius = 40;
  const cartesianX = Math.sqrt(kerrRadius ** 2 + spin ** 2);
  kerrManifest.coordinates.nrChart.coordinates = (
    "Cartesian Kerr-Schild (t_KS,x,y,z)"
  );
  kerrManifest.coordinates.nrChart.gauge = (
    "ingoing Cartesian Kerr-Schild coordinates"
  );
  kerrManifest.physicalSystem.dimensionlessSpins = [
    { componentId: "BH", vector: [0, 0, spin] },
  ];
  kerrManifest.observer.samples[0].eventNr = [0, cartesianX, 0, 0];
  const readouts = readoutsFromManifest(kerrManifest);
  assert.equal(readouts.observerCoordinateLabel, "rKS");
  assert.ok(Math.abs(readouts.observerCoordinateRadius - kerrRadius) < 1e-12);
  assert.ok(readouts.affineCameraDistance > readouts.observerCoordinateRadius);
  assert.equal(readouts.spinMagnitude, spin);

  const schwarzschild = readoutsFromManifest(manifest);
  assert.equal(schwarzschild.observerCoordinateLabel, "r areal");
  assert.equal(schwarzschild.observerCoordinateRadius, 40);
});

test("canvas and transfer-map pixel coordinates round-trip through letterboxing", () => {
  const exactAspect = { left: 10, top: 20, width: 1600, height: 900 };
  assert.deepEqual(
    canvasPointToTransferPixel(810, 470, exactAspect, 1024, 576),
    { x: 512, y: 288 },
  );

  const wide = { left: 0, top: 0, width: 2000, height: 900 };
  assert.equal(
    canvasPointToTransferPixel(100, 450, wide, 1024, 576),
    null,
  );
  assert.deepEqual(
    canvasPointToTransferPixel(1000, 450, wide, 1024, 576),
    { x: 512, y: 288 },
  );
  for (const pixel of [
    { x: 0, y: 0 },
    { x: 512, y: 288 },
    { x: 1023, y: 575 },
  ]) {
    const point = transferPixelToCanvasPoint(
      pixel.x,
      pixel.y,
      wide,
      1024,
      576,
    );
    assert.deepEqual(
      canvasPointToTransferPixel(
        point.left,
        point.top,
        wide,
        1024,
        576,
      ),
      pixel,
    );
  }
});

test("shader bundle uploads canonical records on WebGPU and float planes on WebGL2", () => {
  const dataset = decodedDataset();
  const bundle = createTransferMapShaderBundle(dataset);
  const writes = [];
  let bufferDestroyed = 0;
  globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2 };
  const buffer = {
    destroy() {
      bufferDestroyed += 1;
    },
  };
  const device = {
    createBuffer(descriptor) {
      assert.equal(descriptor.size, dataset.records.byteLength);
      assert.equal(descriptor.usage, 3);
      return buffer;
    },
    queue: {
      writeBuffer(target, offset, bytes) {
        writes.push([target, offset, bytes.byteLength]);
      },
    },
  };
  const webgpu = bundle.resources.createWebGPU(device);
  assert.deepEqual(writes, [[buffer, 0, dataset.records.byteLength]]);
  assert.equal(webgpu.entries[0].binding, 3);
  assert.equal(webgpu.entries[0].resource.buffer, buffer);
  webgpu.dispose();
  assert.equal(bufferDestroyed, 1);

  let textureDisposals = 0;
  class DataTexture {
    constructor(data, width, height, format, type) {
      Object.assign(this, { data, width, height, format, type });
    }

    dispose() {
      textureDisposals += 1;
    }
  }
  const THREE = {
    DataTexture,
    Vector4: class Vector4 {},
    RGBAFormat: "rgba",
    FloatType: "float",
    NearestFilter: "nearest",
    ClampToEdgeWrapping: "clamp",
    NoColorSpace: "none",
  };
  const webgl = bundle.resources.createWebGL(THREE);
  assert.equal(webgl.uniforms.uSceneTransferPrimary.value.width, 1024);
  assert.equal(webgl.uniforms.uSceneTransferMetrics.value.height, 576);
  assert.ok(bundle.uniforms.createWebGLExtras(THREE).uSceneTransferRange);
  webgl.dispose();
  assert.equal(textureDisposals, 2);

  assert.match(transferMapTraceFragmentWGSL, /var<storage, read>/);
  assert.match(transferMapTraceFragmentWGSL, /sample\.state & 255u/);
  assert.match(transferMapTraceFragmentWGSL, /mode == 5u/);
  assert.match(transferMapTraceFragmentWGSL, /hasValidity\(sample, 16u\)/);
  assert.match(transferMapTraceFragmentWGSL, /sceneTransferRange/);
  assert.match(
    transferMapTraceFragmentGLSL,
    /mod\(transferValue\.metrics\.w, 256\.0\)/,
  );
  assert.match(transferMapTraceFragmentGLSL, /mode - 5\.0/);
  assert.match(
    transferMapTraceFragmentGLSL,
    /hasValidity\(transferValue, 16\.0\)/,
  );
  assert.match(transferMapTraceFragmentGLSL, /uSceneTransferRange/);
  assert.doesNotMatch(transferMapTraceFragmentGLSL, /\bsample\b/);
  assert.doesNotMatch(transferMapTraceFragmentWGSL, /mix\(s00\.primary/);
  assert.doesNotMatch(transferMapTraceFragmentGLSL, /mix\(s00\.primary/);
  assert.match(transferMapTraceFragmentWGSL, /atan2\(d\.y, d\.x\)/);
  assert.match(transferMapTraceFragmentWGSL, /asin\(clamp\(d\.z/);
  assert.match(transferMapTraceFragmentGLSL, /atan\(d\.y, d\.x\)/);
  assert.match(transferMapTraceFragmentGLSL, /asin\(clamp\(d\.z/);
});

test("reference scene inspector, diagnostic URL, and listeners survive dispose/re-init", async () => {
  const host = fakeSceneHost(
    "https://blackhole.test/?scene=transfer-map-reference&diagnostic=null-residual",
  );
  const state = sceneState();
  let renderRequests = 0;
  const scene = await createTransferMapReferenceScene({
    document: host.document,
    ui: host.ui,
    state,
    controls: {
      requestRender() {
        renderRequests += 1;
      },
    },
    searchParams: new URLSearchParams(host.location.search),
    location: host.location,
    history: host.windowRef.history,
    referenceRegistry: testRegistry(),
    async loadTransferMapImpl(_url, options) {
      assert.equal(options.expectedManifestSha256, "a".repeat(64));
      options.onProgress({ phase: "manifest", completed: 1, total: 1 });
      return decodedDataset();
    },
  });

  scene.initialize();
  scene.initialize();
  assert.equal(state.mode, 4);
  assert.equal(host.document.listenerCount("click"), 1);
  assert.equal(host.document.listenerCount("keydown"), 1);
  assert.equal(host.windowRef.listenerCount("resize"), 1);
  assert.equal(
    host.document.documentElement.classList.contains(
      "scene-transfer-map-reference",
    ),
    true,
  );
  scene.updateReadouts();
  assert.match(host.ui.observerValue.textContent, /r areal = 40\.00 M/);
  assert.doesNotMatch(host.ui.observerValue.textContent, /β/);

  const canvas = host.elements.get("universe");
  host.document.dispatch("click", {
    target: canvas,
    button: 0,
    clientX: 800,
    clientY: 450,
  });
  assert.equal(host.elements.get("transferMapInspector").hidden, false);
  assert.match(
    host.elements.get("transferInspectorCoordinates").textContent,
    /x 512 · y 288/,
  );
  assert.equal(
    host.elements.get("transferInspectorRaw").textContent.split(" ").length,
    32,
  );
  assert.equal(host.elements.get("transferPixelMarker").hidden, false);

  let keyboardPrevented = false;
  host.document.dispatch("keydown", {
    target: canvas,
    key: "ArrowRight",
    shiftKey: false,
    preventDefault() {
      keyboardPrevented = true;
    },
  });
  assert.equal(keyboardPrevented, true);
  assert.match(
    host.elements.get("transferInspectorCoordinates").textContent,
    /x 513 · y 288/,
  );

  state.mode = 5;
  scene.onModeChanged(5);
  assert.equal(host.location.searchParams.get("diagnostic"), "projection-error");
  assert.match(
    host.elements.get("referenceKerr").href,
    /diagnostic=projection-error/,
  );
  const frame = scene.extendFrame({});
  assert.equal(frame.sceneTransferState[0], 5);
  assert.ok(frame.sceneTransferRange[1] > frame.sceneTransferRange[0]);

  scene.dispose();
  assert.equal(host.document.listenerCount("click"), 0);
  assert.equal(host.document.listenerCount("keydown"), 0);
  assert.equal(host.windowRef.listenerCount("resize"), 0);
  assert.equal(
    host.document.documentElement.classList.contains(
      "scene-transfer-map-reference",
    ),
    false,
  );

  scene.initialize();
  assert.equal(state.mode, 5);
  assert.equal(host.document.listenerCount("click"), 1);
  assert.equal(host.document.listenerCount("keydown"), 1);
  assert.equal(host.windowRef.listenerCount("resize"), 1);
  scene.dispose();
  assert.equal(host.document.listenerCount("click"), 0);
  assert.ok(renderRequests >= 4);
});

test("reference scene failure UI stays fail-closed with retry and return actions", async () => {
  const host = fakeSceneHost(
    "https://blackhole.test/?scene=transfer-map-reference&diagnostic=lookback",
  );
  await assert.rejects(
    createTransferMapReferenceScene({
      document: host.document,
      ui: host.ui,
      state: sceneState(),
      controls: { requestRender() {} },
      searchParams: new URLSearchParams(host.location.search),
      location: host.location,
      history: host.windowRef.history,
      referenceRegistry: testRegistry(),
      async loadTransferMapImpl() {
        throw new Error("authenticated chunk mismatch");
      },
    }),
    (error) => {
      assert.ok(error instanceof TransferMapSceneLoadError);
      assert.equal(error.sceneUiHandled, true);
      return true;
    },
  );

  const status = host.elements.get("sceneStatus");
  assert.equal(status.getAttribute?.("role") ?? (
    status.attributeValues.get("role")
  ), "alert");
  assert.equal(status.children.length, 2);
  assert.match(status.children[0].textContent, /authenticated chunk mismatch/);
  const actions = status.children[1];
  assert.equal(actions.children.length, 2);
  assert.match(actions.children[0].href, /scene=transfer-map-reference/);
  const returnUrl = new URL(actions.children[1].href);
  assert.equal(returnUrl.searchParams.get("scene"), null);
  assert.equal(returnUrl.searchParams.get("diagnostic"), null);
  assert.equal(
    host.elements.get("panel").classList.contains("is-open"),
    true,
  );
  assert.equal(
    host.elements.get("togglePanel").attributeValues.get("aria-expanded"),
    "true",
  );
  assert.equal(
    host.elements.get("togglePanel").attributeValues.get("aria-label"),
    "收起观测参数",
  );
  assert.equal(host.document.listenerCount("click"), 0);
  assert.equal(host.windowRef.listenerCount("resize"), 0);
});

test("reference scene rejects an authenticated dataset with the wrong registry id", async () => {
  const host = fakeSceneHost();
  await assert.rejects(
    createTransferMapReferenceScene({
      document: host.document,
      ui: host.ui,
      state: sceneState(),
      controls: { requestRender() {} },
      searchParams: new URLSearchParams(host.location.search),
      location: host.location,
      history: host.windowRef.history,
      referenceRegistry: testRegistry("different-reference-v1"),
      async loadTransferMapImpl() {
        return decodedDataset();
      },
    }),
    /trusted registry expected different-reference-v1/,
  );
  assert.equal(
    host.elements.get("sceneStatus").attributeValues.get("role"),
    "alert",
  );
});
