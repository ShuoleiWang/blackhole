const RECORD_BYTES = 32;
const MAX_DATASET_BYTES = 128 * 1024 * 1024;
const VALID_DIRECTION = 1 << 0;
const VALID_FREQUENCY_SHIFT = 1 << 1;
const VALID_LOOKBACK = 1 << 2;
const VALID_NULL_RESIDUAL = 1 << 3;
const VALID_PROJECTION_ERROR = 1 << 4;
const VALID_ALL = (
  VALID_DIRECTION
  | VALID_FREQUENCY_SHIFT
  | VALID_LOOKBACK
  | VALID_NULL_RESIDUAL
  | VALID_PROJECTION_ERROR
);

const CANONICAL_FIELDS = Object.freeze([
  Object.freeze({
    name: "escapeDirection",
    offsetBytes: 0,
    componentType: "float32",
    components: 3,
  }),
  Object.freeze({
    name: "frequencyShiftG",
    offsetBytes: 12,
    componentType: "float32",
    components: 1,
  }),
  Object.freeze({
    name: "coordinateLookbackTimeM",
    offsetBytes: 16,
    componentType: "float32",
    components: 1,
  }),
  Object.freeze({
    name: "nullResidual",
    offsetBytes: 20,
    componentType: "float32",
    components: 1,
  }),
  Object.freeze({
    name: "projectionErrorPx",
    offsetBytes: 24,
    componentType: "float32",
    components: 1,
  }),
  Object.freeze({
    name: "rayOutcome",
    offsetBytes: 28,
    componentType: "uint8",
    components: 1,
  }),
  Object.freeze({
    name: "captureTarget",
    offsetBytes: 29,
    componentType: "uint8",
    components: 1,
  }),
  Object.freeze({
    name: "validityMask",
    offsetBytes: 30,
    componentType: "uint16",
    components: 1,
  }),
]);

const CANONICAL_OUTCOMES = Object.freeze({
  escaped: 0,
  captured: 1,
  unresolved: 2,
  "outside-domain": 3,
  "integrator-failure": 4,
  missing: 255,
});

const EXPECTED_MASKS = Object.freeze({
  [CANONICAL_OUTCOMES.escaped]: VALID_ALL,
  [CANONICAL_OUTCOMES.captured]: (
    VALID_LOOKBACK | VALID_NULL_RESIDUAL | VALID_PROJECTION_ERROR
  ),
  [CANONICAL_OUTCOMES.unresolved]: (
    VALID_NULL_RESIDUAL | VALID_PROJECTION_ERROR
  ),
  [CANONICAL_OUTCOMES["outside-domain"]]: (
    VALID_LOOKBACK | VALID_NULL_RESIDUAL | VALID_PROJECTION_ERROR
  ),
  [CANONICAL_OUTCOMES["integrator-failure"]]: VALID_NULL_RESIDUAL,
  [CANONICAL_OUTCOMES.missing]: 0,
});

export class TransferMapContractError extends Error {
  constructor(path, message) {
    super(`Transfer-map contract violation at ${path}: ${message}`);
    this.name = "TransferMapContractError";
    this.path = path;
  }
}

function fail(path, message) {
  throw new TransferMapContractError(path, message);
}

function plainObject(value, path) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(path, "expected an object");
  }
  return value;
}

function finiteNumber(value, path) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    fail(path, "expected a finite number");
  }
  return value;
}

function positiveInteger(value, path) {
  if (!Number.isInteger(value) || value <= 0) {
    fail(path, "expected a positive integer");
  }
  return value;
}

function exactObject(actual, expected, path) {
  plainObject(actual, path);
  const actualKeys = Object.keys(actual).sort();
  const expectedKeys = Object.keys(expected).sort();
  if (
    actualKeys.length !== expectedKeys.length
    || actualKeys.some((key, index) => key !== expectedKeys[index])
  ) {
    fail(path, `expected exactly ${expectedKeys.join(", ")}`);
  }
  for (const key of expectedKeys) {
    if (actual[key] !== expected[key]) {
      fail(`${path}.${key}`, `expected ${JSON.stringify(expected[key])}`);
    }
  }
}

function positiveZero(view, byteOffset) {
  return view.getUint32(byteOffset, true) === 0;
}

function expectInvalidFloatField(view, byteOffset, components, path) {
  for (let component = 0; component < components; component += 1) {
    if (!positiveZero(view, byteOffset + component * 4)) {
      fail(path, "invalid fields must contain canonical positive float32 zero");
    }
  }
}

function assertSafeAssetUrl(manifestUrl, uri, path) {
  if (typeof uri !== "string" || !uri.length || uri.includes("\\")) {
    fail(path, "expected a non-empty POSIX relative URI");
  }
  const base = new URL(".", manifestUrl);
  const resolved = new URL(uri, base);
  if (
    resolved.origin !== base.origin
    || !resolved.pathname.startsWith(base.pathname)
    || resolved.search
    || resolved.hash
  ) {
    fail(path, "asset URI must stay inside the manifest directory");
  }
  return resolved;
}

function bytesOf(value) {
  if (value instanceof Uint8Array) {
    return value;
  }
  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value);
  }
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  throw new TypeError("Expected an ArrayBuffer or typed-array view");
}

export async function sha256Hex(value, cryptoImpl = globalThis.crypto) {
  if (!cryptoImpl?.subtle) {
    throw new Error("Web Crypto SHA-256 is unavailable");
  }
  const bytes = bytesOf(value);
  const digest = await cryptoImpl.subtle.digest(
    "SHA-256",
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  return Array.from(new Uint8Array(digest), (byte) => (
    byte.toString(16).padStart(2, "0")
  )).join("");
}

export function validateTransferMapManifest(manifest) {
  plainObject(manifest, "$");
  if (manifest.schema !== "blackhole.nr-transfer-map/v1") {
    fail("$.schema", "only blackhole.nr-transfer-map/v1 is supported");
  }
  if (manifest.datasetKind !== "stationary-reference-transfer-map") {
    fail("$.datasetKind", "this scene accepts only a stationary reference transfer map");
  }
  if (manifest.renderable !== true) {
    fail("$.renderable", "the dataset is not approved for rendering");
  }

  const scientificStatus = plainObject(manifest.scientificStatus, "$.scientificStatus");
  for (const flag of [
    "sourceIsNumericalRelativity",
    "derivedFromNearZoneSpacetime",
    "derivedWithSlowLightGeodesics",
  ]) {
    if (scientificStatus[flag] !== false) {
      fail(
        `$.scientificStatus.${flag}`,
        "stationary reference maps must not claim NR or slow-light provenance",
      );
    }
  }
  if (manifest.physicalSystem?.kind !== "stationary-black-hole") {
    fail("$.physicalSystem.kind", "expected a stationary black hole");
  }
  if (manifest.physicalSystem?.vacuum !== true) {
    fail("$.physicalSystem.vacuum", "the reference map must describe a vacuum spacetime");
  }
  if (manifest.camera?.motion !== "fixed") {
    fail("$.camera.motion", "v1 transfer-map playback requires a fixed camera");
  }

  const width = positiveInteger(manifest.projection?.widthPixels, "$.projection.widthPixels");
  const height = positiveInteger(manifest.projection?.heightPixels, "$.projection.heightPixels");
  if (width < 256 || height < 144) {
    fail("$.projection", "a renderable reference map must be at least 256x144");
  }
  finiteNumber(
    manifest.projection?.verticalFieldOfViewRad,
    "$.projection.verticalFieldOfViewRad",
  );
  if (
    manifest.projection.model !== "rectilinear-pinhole"
    || manifest.projection.imageOrigin !== "top-left"
    || manifest.projection.pixelSampleLocation !== "center"
  ) {
    fail("$.projection", "unsupported projection convention");
  }

  const times = manifest.sampling?.observationTimesM;
  if (!Array.isArray(times) || times.length !== 1 || times[0] !== 0) {
    fail("$.sampling.observationTimesM", "stationary v1 playback requires one t=0 sample");
  }
  if (
    manifest.sampling.dimensionOrder?.join("/") !== "time/y/x"
    || manifest.sampling.pixelOrder !== "row-major"
    || manifest.sampling.interpolation?.continuous !== "none-nearest-texel-center"
    || manifest.sampling.interpolation?.escapeDirection !== "nearest-no-blend"
    || manifest.sampling.interpolation?.categorical !== "nearest-no-blend"
    || manifest.sampling.interpolation?.invalidRecords !== "never-sample-sky"
  ) {
    fail("$.sampling", "unsupported sampling or interpolation convention");
  }
  if (
    manifest.coordinates?.sky?.referenceFrame !== "ICRS"
    || manifest.coordinates?.sky?.escapeDirectionFrame !== "ICRS"
    || manifest.coordinates?.sky?.projection !== "equirectangular"
    || manifest.coordinates?.sky?.longitudeMapping
      !== "u=fract(longitude/(2*pi)+0.5)"
    || manifest.coordinates?.sky?.latitudeMapping !== "v=0.5-latitude/pi"
  ) {
    fail("$.coordinates.sky", "the runtime requires canonical ICRS sky directions");
  }

  const layout = plainObject(manifest.recordLayout, "$.recordLayout");
  if (
    layout.storage !== "raw-struct-array"
    || layout.byteOrder !== "little-endian"
    || layout.structFormat !== "<7fBBH"
    || layout.recordBytes !== RECORD_BYTES
    || layout.captureTargetNone !== 255
  ) {
    fail("$.recordLayout", "the 32-byte little-endian v1 ABI is required");
  }
  if (!Array.isArray(layout.fields) || layout.fields.length !== CANONICAL_FIELDS.length) {
    fail("$.recordLayout.fields", "field count does not match the v1 ABI");
  }
  layout.fields.forEach((field, index) => {
    exactObject(field, CANONICAL_FIELDS[index], `$.recordLayout.fields[${index}]`);
  });
  exactObject(layout.rayOutcomes, CANONICAL_OUTCOMES, "$.recordLayout.rayOutcomes");
  exactObject(layout.validityBits, {
    escapeDirection: 0,
    frequencyShiftG: 1,
    coordinateLookbackTimeM: 2,
    nullResidual: 3,
    projectionErrorPx: 4,
  }, "$.recordLayout.validityBits");

  const accuracy = plainObject(manifest.accuracy, "$.accuracy");
  if (accuracy.status !== "measured") {
    fail("$.accuracy.status", "renderable scientific maps require measured accuracy");
  }
  for (const name of [
    "geodesicNullResidual",
    "interpolationError",
  ]) {
    const measurement = plainObject(accuracy[name], `$.accuracy.${name}`);
    if (measurement.status !== "measured") {
      fail(`$.accuracy.${name}.status`, "required accuracy metric was not measured");
    }
    finiteNumber(measurement.value, `$.accuracy.${name}.value`);
  }
  for (const name of ["nrConvergence", "constraintNorms"]) {
    const section = plainObject(accuracy[name], `$.accuracy.${name}`);
    if (
      section.status !== "not-applicable"
      || section.method !== null
      || section.value !== null
    ) {
      fail(
        `$.accuracy.${name}`,
        "NR-only metric must be not-applicable with null method/value",
      );
    }
  }
  const fractions = plainObject(accuracy.outcomeFractions, "$.accuracy.outcomeFractions");
  for (const name of [...Object.keys(CANONICAL_OUTCOMES), "unusable"]) {
    const value = finiteNumber(fractions[name], `$.accuracy.outcomeFractions.${name}`);
    if (value < 0 || value > 1) {
      fail(`$.accuracy.outcomeFractions.${name}`, "fraction must be between zero and one");
    }
  }
  if (fractions.unusable !== 0 || accuracy.unresolvedFraction !== 0) {
    fail("$.accuracy.outcomeFractions.unusable", "reference playback refuses unresolved rays");
  }

  if (!Array.isArray(manifest.captureTargets) || manifest.captureTargets.length !== 1) {
    fail("$.captureTargets", "stationary reference requires one capture target");
  }
  const captureTarget = manifest.captureTargets[0];
  if (
    captureTarget.code !== 0
    || captureTarget.id !== manifest.physicalSystem.componentIds?.[0]
  ) {
    fail("$.captureTargets[0]", "capture target does not match the stationary black hole");
  }

  if (!Array.isArray(manifest.chunks) || manifest.chunks.length === 0) {
    fail("$.chunks", "at least one chunk is required");
  }
  const coverage = new Uint8Array(width * height);
  let totalBytes = 0;
  let previousKey = null;
  for (const [index, chunk] of manifest.chunks.entries()) {
    const path = `$.chunks[${index}]`;
    plainObject(chunk, path);
    if (chunk.sampleIndex !== 0 || chunk.recordBytes !== RECORD_BYTES) {
      fail(path, "chunk sample or record size does not match the stationary v1 ABI");
    }
    const tile = plainObject(chunk.tile, `${path}.tile`);
    const x = Number.isInteger(tile.x) ? tile.x : -1;
    const y = Number.isInteger(tile.y) ? tile.y : -1;
    const tileWidth = positiveInteger(tile.width, `${path}.tile.width`);
    const tileHeight = positiveInteger(tile.height, `${path}.tile.height`);
    if (x < 0 || y < 0 || x + tileWidth > width || y + tileHeight > height) {
      fail(`${path}.tile`, "tile lies outside the projection");
    }
    const key = `${String(y).padStart(10, "0")}/${String(x).padStart(10, "0")}`;
    if (previousKey !== null && key < previousKey) {
      fail("$.chunks", "chunks are not in canonical row-major tile order");
    }
    previousKey = key;
    const records = tileWidth * tileHeight;
    if (
      chunk.recordCount !== records
      || chunk.byteLength !== records * RECORD_BYTES
    ) {
      fail(path, "chunk length does not match tile area and record stride");
    }
    if (!/^[0-9a-f]{64}$/.test(chunk.sha256)) {
      fail(`${path}.sha256`, "expected a lowercase SHA-256 digest");
    }
    totalBytes += chunk.byteLength;
    for (let row = y; row < y + tileHeight; row += 1) {
      for (let column = x; column < x + tileWidth; column += 1) {
        const pixel = row * width + column;
        if (coverage[pixel]) {
          fail("$.chunks", `overlapping tile coverage at pixel ${column},${row}`);
        }
        coverage[pixel] = 1;
      }
    }
  }
  if (totalBytes > MAX_DATASET_BYTES) {
    fail("$.chunks", `dataset exceeds the ${MAX_DATASET_BYTES}-byte runtime budget`);
  }
  if (coverage.some((value) => value !== 1)) {
    fail("$.chunks", "tiles do not cover the complete projection");
  }

  return Object.freeze({
    width,
    height,
    totalBytes,
    recordCount: width * height,
  });
}

function decodeRecord(view, offset, path, captureCodes, nullResidualLimit) {
  const values = Array.from(
    { length: 7 },
    (_, index) => view.getFloat32(offset + index * 4, true),
  );
  if (values.some((value) => !Number.isFinite(value))) {
    fail(path, "record contains NaN or Infinity");
  }
  const [
    directionX,
    directionY,
    directionZ,
    frequencyShift,
    lookback,
    nullResidual,
    projectionError,
  ] = values;
  const outcome = view.getUint8(offset + 28);
  const target = view.getUint8(offset + 29);
  const validityMask = view.getUint16(offset + 30, true);
  const expectedMask = EXPECTED_MASKS[outcome];
  if (expectedMask === undefined) {
    fail(`${path}.rayOutcome`, `unknown ray outcome ${outcome}`);
  }
  if (validityMask !== expectedMask) {
    fail(
      `${path}.validityMask`,
      `outcome ${outcome} requires 0x${expectedMask.toString(16)}, got 0x${validityMask.toString(16)}`,
    );
  }
  if (outcome === CANONICAL_OUTCOMES.captured) {
    if (!captureCodes.has(target)) {
      fail(`${path}.captureTarget`, `unknown capture target ${target}`);
    }
  } else if (target !== 255) {
    fail(`${path}.captureTarget`, "non-captured rays require the no-target sentinel");
  }

  const validityFields = [
    [VALID_DIRECTION, 0, 3, `${path}.escapeDirection`],
    [VALID_FREQUENCY_SHIFT, 12, 1, `${path}.frequencyShiftG`],
    [VALID_LOOKBACK, 16, 1, `${path}.coordinateLookbackTimeM`],
    [VALID_NULL_RESIDUAL, 20, 1, `${path}.nullResidual`],
    [VALID_PROJECTION_ERROR, 24, 1, `${path}.projectionErrorPx`],
  ];
  for (const [bit, fieldOffset, components, fieldPath] of validityFields) {
    if (!(validityMask & bit)) {
      expectInvalidFloatField(view, offset + fieldOffset, components, fieldPath);
    }
  }

  if (validityMask & VALID_DIRECTION) {
    const norm = Math.hypot(directionX, directionY, directionZ);
    if (Math.abs(norm - 1) > 1e-6) {
      fail(`${path}.escapeDirection`, `unit-vector error ${Math.abs(norm - 1)}`);
    }
  }
  if ((validityMask & VALID_FREQUENCY_SHIFT) && frequencyShift <= 0) {
    fail(`${path}.frequencyShiftG`, "frequency shift must be positive");
  }
  if ((validityMask & VALID_LOOKBACK) && lookback < 0) {
    fail(`${path}.coordinateLookbackTimeM`, "lookback time must be non-negative");
  }
  if ((validityMask & VALID_NULL_RESIDUAL) && nullResidual < 0) {
    fail(`${path}.nullResidual`, "null residual must be non-negative");
  }
  if (
    (validityMask & VALID_NULL_RESIDUAL)
    && nullResidual > nullResidualLimit
  ) {
    fail(
      `${path}.nullResidual`,
      `residual ${nullResidual} exceeds the declared ${nullResidualLimit} gate`,
    );
  }
  if ((validityMask & VALID_PROJECTION_ERROR) && projectionError < 0) {
    fail(`${path}.projectionErrorPx`, "projection error must be non-negative");
  }
  if ((validityMask & VALID_PROJECTION_ERROR) && projectionError > 0.25) {
    fail(`${path}.projectionErrorPx`, "resolved ray exceeds the 0.25px quality gate");
  }

  return {
    directionX,
    directionY,
    directionZ,
    frequencyShift,
    lookback,
    nullResidual,
    projectionError,
    outcome,
    target,
    validityMask,
  };
}

function createDiagnosticRange() {
  return {
    minimum: Infinity,
    maximum: -Infinity,
    minimumPositive: Infinity,
  };
}

function includeDiagnosticValue(range, value) {
  range.minimum = Math.min(range.minimum, value);
  range.maximum = Math.max(range.maximum, value);
  if (value > 0) {
    range.minimumPositive = Math.min(range.minimumPositive, value);
  }
}

function freezeDiagnosticRange(range) {
  const hasValues = Number.isFinite(range.minimum);
  return Object.freeze({
    minimum: hasValues ? range.minimum : 0,
    maximum: hasValues ? range.maximum : 0,
    minimumPositive: Number.isFinite(range.minimumPositive)
      ? range.minimumPositive
      : 0,
  });
}

export function decodeTransferMapChunks(manifest, chunkPayloads) {
  const metadata = validateTransferMapManifest(manifest);
  const primary = new Float32Array(metadata.recordCount * 4);
  const metrics = new Float32Array(metadata.recordCount * 4);
  const records = new Uint8Array(metadata.recordCount * RECORD_BYTES);
  const counts = Object.fromEntries(
    Object.keys(CANONICAL_OUTCOMES).map((name) => [name, 0]),
  );
  const outcomeNames = Object.fromEntries(
    Object.entries(CANONICAL_OUTCOMES).map(([name, code]) => [code, name]),
  );
  const diagnosticRanges = {
    frequencyShift: createDiagnosticRange(),
    lookback: createDiagnosticRange(),
    nullResidual: createDiagnosticRange(),
    projectionError: createDiagnosticRange(),
  };
  const captureCodes = new Set(manifest.captureTargets.map((target) => target.code));
  const nullResidualLimit = finiteNumber(
    manifest.rayIntegration?.tolerances?.nullConstraint,
    "$.rayIntegration.tolerances.nullConstraint",
  );
  if (nullResidualLimit <= 0 || nullResidualLimit > 1e-8) {
    fail(
      "$.rayIntegration.tolerances.nullConstraint",
      "reference playback requires a positive null-residual gate no larger than 1e-8",
    );
  }

  for (const [chunkIndex, chunk] of manifest.chunks.entries()) {
    const payload = chunkPayloads instanceof Map
      ? chunkPayloads.get(chunk.uri)
      : chunkPayloads[chunkIndex];
    if (!payload) {
      fail(`$.chunks[${chunkIndex}]`, "chunk payload is missing");
    }
    const bytes = bytesOf(payload);
    if (bytes.byteLength !== chunk.byteLength) {
      fail(
        `$.chunks[${chunkIndex}].byteLength`,
        `expected ${chunk.byteLength}, received ${bytes.byteLength}`,
      );
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const tile = chunk.tile;
    for (let localIndex = 0; localIndex < chunk.recordCount; localIndex += 1) {
      const localX = localIndex % tile.width;
      const localY = Math.floor(localIndex / tile.width);
      const x = tile.x + localX;
      const y = tile.y + localY;
      const pixel = y * metadata.width + x;
      const sourceOffset = localIndex * RECORD_BYTES;
      const record = decodeRecord(
        view,
        sourceOffset,
        `$.chunks[${chunkIndex}].records[${localIndex}]`,
        captureCodes,
        nullResidualLimit,
      );
      primary.set([
        record.directionX,
        record.directionY,
        record.directionZ,
        record.frequencyShift,
      ], pixel * 4);
      metrics.set([
        record.lookback,
        record.nullResidual,
        record.projectionError,
        (
          record.outcome
          + 256 * record.target
          + 65_536 * record.validityMask
        ),
      ], pixel * 4);
      records.set(
        bytes.subarray(sourceOffset, sourceOffset + RECORD_BYTES),
        pixel * RECORD_BYTES,
      );
      if (record.validityMask & VALID_FREQUENCY_SHIFT) {
        includeDiagnosticValue(
          diagnosticRanges.frequencyShift,
          record.frequencyShift,
        );
      }
      if (record.validityMask & VALID_LOOKBACK) {
        includeDiagnosticValue(diagnosticRanges.lookback, record.lookback);
      }
      if (record.validityMask & VALID_NULL_RESIDUAL) {
        includeDiagnosticValue(
          diagnosticRanges.nullResidual,
          record.nullResidual,
        );
      }
      if (record.validityMask & VALID_PROJECTION_ERROR) {
        includeDiagnosticValue(
          diagnosticRanges.projectionError,
          record.projectionError,
        );
      }
      counts[outcomeNames[record.outcome]] += 1;
    }
  }

  const total = metadata.recordCount;
  for (const name of Object.keys(CANONICAL_OUTCOMES)) {
    const actual = counts[name] / total;
    const declared = manifest.accuracy.outcomeFractions[name];
    if (Math.abs(actual - declared) > 1e-12) {
      fail(
        `$.accuracy.outcomeFractions.${name}`,
        `declares ${declared}, decoded ${actual}`,
      );
    }
  }
  const unusable = (
    counts.unresolved
    + counts["outside-domain"]
    + counts["integrator-failure"]
    + counts.missing
  ) / total;
  if (Math.abs(unusable - manifest.accuracy.outcomeFractions.unusable) > 1e-12) {
    fail("$.accuracy.outcomeFractions.unusable", "decoded unusable fraction differs");
  }

  return Object.freeze({
    manifest,
    width: metadata.width,
    height: metadata.height,
    primary,
    metrics,
    records,
    counts: Object.freeze(counts),
    diagnosticRanges: Object.freeze(
      Object.fromEntries(
        Object.entries(diagnosticRanges).map(([name, range]) => [
          name,
          freezeDiagnosticRange(range),
        ]),
      ),
    ),
    totalBytes: metadata.totalBytes,
  });
}

export function readTransferMapRecord(dataset, x, y) {
  if (
    !(dataset?.records instanceof Uint8Array)
    || !Number.isInteger(dataset.width)
    || !Number.isInteger(dataset.height)
    || dataset.records.byteLength !== dataset.width * dataset.height * RECORD_BYTES
  ) {
    throw new TypeError(
      "Transfer-map inspection requires a validated canonical record array",
    );
  }
  if (
    !Number.isInteger(x)
    || !Number.isInteger(y)
    || x < 0
    || y < 0
    || x >= dataset.width
    || y >= dataset.height
  ) {
    throw new RangeError(`Transfer-map pixel ${x},${y} lies outside the dataset`);
  }

  const byteOffset = (y * dataset.width + x) * RECORD_BYTES;
  const bytes = dataset.records.subarray(byteOffset, byteOffset + RECORD_BYTES);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const outcome = view.getUint8(28);
  const captureTarget = view.getUint8(29);
  const validityMask = view.getUint16(30, true);
  const outcomeName = Object.entries(CANONICAL_OUTCOMES)
    .find(([, code]) => code === outcome)?.[0] ?? `unknown-${outcome}`;
  const captureTargetId = captureTarget === 255
    ? null
    : dataset.manifest?.captureTargets
      ?.find((target) => target.code === captureTarget)?.id ?? null;

  return Object.freeze({
    x,
    y,
    byteOffset,
    escapeDirection: Object.freeze([
      view.getFloat32(0, true),
      view.getFloat32(4, true),
      view.getFloat32(8, true),
    ]),
    frequencyShiftG: view.getFloat32(12, true),
    coordinateLookbackTimeM: view.getFloat32(16, true),
    nullResidual: view.getFloat32(20, true),
    projectionErrorPx: view.getFloat32(24, true),
    rayOutcome: outcome,
    rayOutcomeName: outcomeName,
    captureTarget,
    captureTargetId,
    validityMask,
    rawBytes: new Uint8Array(bytes),
    rawHex: Array.from(
      bytes,
      (byte) => byte.toString(16).padStart(2, "0"),
    ).join(" "),
  });
}

async function fetchBytes(fetchImpl, url, label) {
  const response = await fetchImpl(url);
  if (!response.ok) {
    throw new Error(`${label} request failed (${response.status})`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

export async function loadTransferMap(
  manifestUrl,
  {
    expectedManifestSha256,
    fetchImpl = globalThis.fetch,
    cryptoImpl = globalThis.crypto,
    onProgress = () => {},
  } = {},
) {
  if (typeof fetchImpl !== "function") {
    throw new Error("Fetch is unavailable");
  }
  if (
    typeof expectedManifestSha256 !== "string"
    || !/^[0-9a-f]{64}$/.test(expectedManifestSha256)
  ) {
    fail(
      "$.integrity",
      "a pinned lowercase manifest SHA-256 trust root is required",
    );
  }
  const resolvedManifestUrl = new URL(manifestUrl, globalThis.location?.href);
  onProgress({ phase: "manifest", completed: 0, total: 1 });
  const manifestBytes = await fetchBytes(fetchImpl, resolvedManifestUrl, "Manifest");
  const manifestDigest = await sha256Hex(manifestBytes, cryptoImpl);
  if (manifestDigest !== expectedManifestSha256) {
    fail("$.integrity", `pinned manifest SHA-256 mismatch (${manifestDigest})`);
  }
  onProgress({ phase: "manifest", completed: 1, total: 1 });

  const decoder = new TextDecoder("utf-8", { fatal: true });
  let manifest;
  try {
    manifest = JSON.parse(decoder.decode(manifestBytes));
  } catch (error) {
    fail("$", `manifest is not strict UTF-8 JSON: ${error.message}`);
  }
  validateTransferMapManifest(manifest);

  const sidecarUrl = assertSafeAssetUrl(
    resolvedManifestUrl,
    manifest.integrity?.manifestSidecar,
    "$.integrity.manifestSidecar",
  );
  onProgress({ phase: "sidecar", completed: 0, total: 1 });
  const sidecar = decoder.decode(
    await fetchBytes(fetchImpl, sidecarUrl, "Manifest sidecar"),
  );
  const expectedSidecar = `${manifestDigest}  manifest.json\n`;
  if (sidecar !== expectedSidecar) {
    fail("$.integrity.manifestSidecar", "sidecar digest does not match manifest bytes");
  }
  onProgress({ phase: "sidecar", completed: 1, total: 1 });

  let completed = 0;
  const chunks = await Promise.all(manifest.chunks.map(async (chunk, index) => {
    const url = assertSafeAssetUrl(
      resolvedManifestUrl,
      chunk.uri,
      `$.chunks[${index}].uri`,
    );
    const bytes = await fetchBytes(fetchImpl, url, `Chunk ${index + 1}`);
    if (bytes.byteLength !== chunk.byteLength) {
      fail(
        `$.chunks[${index}].byteLength`,
        `expected ${chunk.byteLength}, received ${bytes.byteLength}`,
      );
    }
    const digest = await sha256Hex(bytes, cryptoImpl);
    if (digest !== chunk.sha256) {
      fail(`$.chunks[${index}].sha256`, `hash mismatch (${digest})`);
    }
    completed += 1;
    onProgress({ phase: "chunks", completed, total: manifest.chunks.length });
    return bytes;
  }));
  const decoded = decodeTransferMapChunks(manifest, chunks);
  onProgress({ phase: "decoded", completed: decoded.width * decoded.height, total: decoded.width * decoded.height });
  return Object.freeze({
    ...decoded,
    manifestSha256: manifestDigest,
  });
}

export const TRANSFER_MAP_ABI = Object.freeze({
  recordBytes: RECORD_BYTES,
  outcomes: CANONICAL_OUTCOMES,
  validity: Object.freeze({
    direction: VALID_DIRECTION,
    frequencyShift: VALID_FREQUENCY_SHIFT,
    lookback: VALID_LOOKBACK,
    nullResidual: VALID_NULL_RESIDUAL,
    projectionError: VALID_PROJECTION_ERROR,
    all: VALID_ALL,
  }),
});
