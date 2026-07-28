import assert from "node:assert/strict";
import { createHash, webcrypto } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  decodeTransferMapChunks,
  loadTransferMap,
  TransferMapContractError,
  validateTransferMapManifest,
} from "../src/transfer-map-loader.js";
import {
  createTransferMapShaderBundle,
  transferMapTraceFragmentGLSL,
  transferMapTraceFragmentWGSL,
} from "../src/transfer-map-shaders.js";
import {
  REFERENCE_MANIFEST_SHA256,
} from "../src/scenes/transfer-map-reference-scene.js";

const root = new URL("../", import.meta.url);
const datasetRoot = new URL(
  "assets/transfer-maps/schwarzschild-reference-v1/",
  root,
);
const manifestUrl = new URL("manifest.json", datasetRoot);
const manifestBytes = new Uint8Array(await readFile(manifestUrl));
const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
const chunkPayloads = new Map(
  await Promise.all(manifest.chunks.map(async (chunk) => [
    chunk.uri,
    new Uint8Array(await readFile(new URL(chunk.uri, datasetRoot))),
  ])),
);

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

test("shader bundle uploads canonical records on WebGPU and float planes on WebGL2", () => {
  const dataset = decodeTransferMapChunks(manifest, chunkPayloads);
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
    RGBAFormat: "rgba",
    FloatType: "float",
    NearestFilter: "nearest",
    ClampToEdgeWrapping: "clamp",
    NoColorSpace: "none",
  };
  const webgl = bundle.resources.createWebGL(THREE);
  assert.equal(webgl.uniforms.uSceneTransferPrimary.value.width, 1024);
  assert.equal(webgl.uniforms.uSceneTransferMetrics.value.height, 576);
  webgl.dispose();
  assert.equal(textureDisposals, 2);

  assert.match(transferMapTraceFragmentWGSL, /var<storage, read>/);
  assert.match(transferMapTraceFragmentWGSL, /sample\.state & 255u/);
  assert.match(
    transferMapTraceFragmentGLSL,
    /mod\(transferValue\.metrics\.w, 256\.0\)/,
  );
  assert.doesNotMatch(transferMapTraceFragmentGLSL, /\bsample\b/);
  assert.doesNotMatch(transferMapTraceFragmentWGSL, /mix\(s00\.primary/);
  assert.doesNotMatch(transferMapTraceFragmentGLSL, /mix\(s00\.primary/);
  assert.match(transferMapTraceFragmentWGSL, /atan2\(d\.y, d\.x\)/);
  assert.match(transferMapTraceFragmentWGSL, /asin\(clamp\(d\.z/);
  assert.match(transferMapTraceFragmentGLSL, /atan\(d\.y, d\.x\)/);
  assert.match(transferMapTraceFragmentGLSL, /asin\(clamp\(d\.z/);
});
