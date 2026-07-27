const MANIFEST_SCHEMA = "blackhole.binary-scene/v2";
const SAMPLE_SCHEMA = "blackhole.binary-dynamics-samples/v1";
const EXPECTED_COLUMNS = Object.freeze([
  "tProtocolM",
  "separationM",
  "orbitalPhaseUnwrappedRad",
  "h22Real",
  "h22Imag",
  "renderTopologyBlend",
  "individualHorizonsValid",
]);
const EXPECTED_MANIFEST_KEYS = Object.freeze([
  "schema",
  "id",
  "title",
  "scientificStatus",
  "source",
  "units",
  "physicalSystem",
  "timeReference",
  "events",
  "dynamics",
  "rendererAdapter",
  "playback",
  "rendererDefaults",
  "generation",
]);
const EXPECTED_EVENT_KEYS = Object.freeze([
  "relaxation",
  "individualHorizonsLast",
  "commonApparentHorizonFirst",
  "waveformPeak",
  "playbackEnd",
]);
const EXPECTED_ERROR_CHANNELS = Object.freeze([
  "separationM",
  "orbitalPhaseUnwrappedRad",
  "h22Real",
  "h22Imag",
  "renderTopologyBlend",
]);
const EXPECTED_STATE_ABI = Object.freeze([
  "separationM",
  "orbitalPhaseUnwrappedRad",
  "renderTopologyBlend",
  "reserved",
]);
const EXPECTED_MASS_ABI = Object.freeze([
  "bodyAMassFraction",
  "bodyBMassFraction",
  "remnantMassFraction",
  "reserved",
]);
const PINNED_SOURCE_ARTIFACTS = Object.freeze({
  metadata: Object.freeze({
    byteLength: 4170,
    md5: "099d4c93d9466fe4b7ecad6c94499cf3",
    path: "SXS:BBH:0001/Lev5/metadata.json",
    sha256: "329d0643f9d33361eafaeae7ef1818dcda3311b33477ecef4f002ead17f42668",
    url: "https://zenodo.org/api/records/3273935/files/SXS:BBH:0001/Lev5/metadata.json/content",
  }),
  horizons: Object.freeze({
    byteLength: 3501232,
    md5: "484ea88842209e64983793159bcc7d7c",
    path: "SXS:BBH:0001/Lev5/Horizons.h5",
    sha256: "cf97de4a60a4cd5c6a56f219ea9fa81f1849647f134250e95ae79e40be4dd957",
    url: "https://zenodo.org/api/records/3273935/files/SXS:BBH:0001/Lev5/Horizons.h5/content",
  }),
  waveform: Object.freeze({
    byteLength: 142641207,
    md5: "c271e0b905c74f434f00c9b14f67850c",
    path: "SXS:BBH:0001/Lev5/rhOverM_Asymptotic_GeometricUnits_CoM.h5",
    sha256: "d760add0693e458781f8db9958b4669971e816d7c026cdbe5f09b7d8fd6bd21f",
    url: "https://zenodo.org/api/records/3273935/files/SXS:BBH:0001/Lev5/rhOverM_Asymptotic_GeometricUnits_CoM.h5/content",
  }),
});
const MAX_SHADER_STEPS = 512;
const CONTRACT_EPSILON = 1e-9;

function require(condition, message) {
  if (!condition) {
    throw new Error(`Binary dynamics contract violation: ${message}`);
  }
}

function finite(value, label) {
  require(
    typeof value === "number" && Number.isFinite(value),
    `${label} must be a finite JSON number`,
  );
  return value;
}

function sameArray(actual, expected) {
  return Array.isArray(actual)
    && actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
}

function exactKeys(value, expected, label) {
  require(
    value !== null && typeof value === "object" && !Array.isArray(value),
    `${label} must be an object`,
  );
  const actual = Object.keys(value);
  const missing = expected.filter((key) => !actual.includes(key));
  const unknown = actual.filter((key) => !expected.includes(key));
  require(
    missing.length === 0 && unknown.length === 0,
    `${label} keys differ (missing: ${missing.join(",") || "none"}; `
      + `unknown: ${unknown.join(",") || "none"})`,
  );
  return value;
}

function nonEmptyString(value, label) {
  require(
    typeof value === "string" && value.trim().length > 0,
    `${label} must be a non-empty string`,
  );
  return value;
}

function finiteVector3(value, label) {
  require(
    Array.isArray(value) && value.length === 3,
    `${label} must be a three-vector`,
  );
  return value.map((component, index) => (
    finite(component, `${label}[${index}]`)
  ));
}

function validateErrorBounds(dynamics) {
  const declared = exactKeys(
    dynamics.declaredMaxInterpolationError,
    EXPECTED_ERROR_CHANNELS,
    "dynamics.declaredMaxInterpolationError",
  );
  const measured = exactKeys(
    dynamics.measuredMaxInterpolationError,
    EXPECTED_ERROR_CHANNELS,
    "dynamics.measuredMaxInterpolationError",
  );
  for (const channel of EXPECTED_ERROR_CHANNELS) {
    const declaredValue = finite(
      declared[channel],
      `dynamics.declaredMaxInterpolationError.${channel}`,
    );
    const measuredValue = finite(
      measured[channel],
      `dynamics.measuredMaxInterpolationError.${channel}`,
    );
    require(declaredValue > 0, `${channel} error bound must be positive`);
    require(
      measuredValue >= 0 && measuredValue <= declaredValue,
      `${channel} measured error exceeds its declared bound`,
    );
  }
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function interpolate(first, second, weight) {
  return first + (second - first) * weight;
}

async function sha256Hex(bytes) {
  require(
    globalThis.crypto?.subtle,
    "Web Crypto SHA-256 support is required",
  );
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function validateManifest(manifest) {
  exactKeys(manifest, EXPECTED_MANIFEST_KEYS, "manifest");
  require(manifest.schema === MANIFEST_SCHEMA, "unexpected manifest schema");
  require(
    manifest.id === "binary-sxs-bbh-0001-v2",
    "unexpected manifest id",
  );
  nonEmptyString(manifest.title, "title");

  const scientificStatus = exactKeys(
    manifest.scientificStatus,
    [
      "classification",
      "description",
      "dynamicsSourceIsNumericalRelativity",
      "nearZoneSpacetimeUsedForLightPropagation",
      "prohibitedClaim",
      "slowLightGeodesicsIncluded",
    ],
    "scientificStatus",
  );
  require(
    scientificStatus.classification
      === "NR-driven dynamics with weak-field fast-light rendering",
    "scientific classification is invalid",
  );
  nonEmptyString(scientificStatus.description, "scientificStatus.description");
  nonEmptyString(
    scientificStatus.prohibitedClaim,
    "scientificStatus.prohibitedClaim",
  );
  require(
    scientificStatus.dynamicsSourceIsNumericalRelativity === true,
    "dynamics must declare its NR source",
  );
  require(
    scientificStatus.nearZoneSpacetimeUsedForLightPropagation === false,
    "this scene must not claim to consume the NR near-zone spacetime",
  );
  require(
    scientificStatus.slowLightGeodesicsIncluded === false,
    "this scene must not claim slow-light NR geodesics",
  );

  const source = exactKeys(
    manifest.source,
    [
      "artifacts",
      "catalog",
      "level",
      "license",
      "recordDoi",
      "recordId",
      "simulation",
      "waveformDataset",
    ],
    "source",
  );
  require(
    source.catalog === "SXS"
      && source.simulation === "SXS:BBH:0001"
      && source.level === "Lev5"
      && source.recordDoi === "https://doi.org/10.5281/zenodo.3273935"
      && source.recordId === 3273935
      && source.waveformDataset === "Extrapolated_N2.dir/Y_l2_m2.dat",
    "the pinned source must be SXS:BBH:0001 Lev5",
  );
  const sourceLicense = exactKeys(
    source.license,
    ["attributionRequired", "spdx", "status"],
    "source.license",
  );
  require(
    sourceLicense.attributionRequired === true
      && sourceLicense.spdx === null
      && sourceLicense.status === "not-declared-in-pinned-zenodo-record",
    "source license declaration is invalid",
  );
  const sourceArtifacts = exactKeys(
    source.artifacts,
    ["metadata", "horizons", "waveform"],
    "source.artifacts",
  );
  for (const role of ["metadata", "horizons", "waveform"]) {
    const expected = PINNED_SOURCE_ARTIFACTS[role];
    const artifact = exactKeys(
      sourceArtifacts[role],
      ["byteLength", "md5", "path", "sha256", "url"],
      `source.artifacts.${role}`,
    );
    require(
      artifact.byteLength === expected.byteLength
        && artifact.md5 === expected.md5
        && artifact.path === expected.path
        && artifact.sha256 === expected.sha256
        && artifact.url === expected.url,
      `source.artifacts.${role} differs from the pinned source bytes`,
    );
  }

  const units = exactKeys(
    manifest.units,
    [
      "G",
      "c",
      "distanceUnit",
      "massUnit",
      "referenceMassDefinition",
      "referenceMassTotalInCodeUnits",
      "system",
      "timeUnit",
    ],
    "units",
  );
  require(
    units.system === "geometric"
      && units.G === 1
      && units.c === 1
      && units.distanceUnit === "M"
      && units.massUnit === "M"
      && units.timeUnit === "M",
    "geometric units declaration is invalid",
  );
  nonEmptyString(units.referenceMassDefinition, "units.referenceMassDefinition");
  require(
    finite(
      units.referenceMassTotalInCodeUnits,
      "units.referenceMassTotalInCodeUnits",
    ) > 0,
    "reference mass must be positive",
  );

  const physicalSystem = exactKeys(
    manifest.physicalSystem,
    [
      "bodies",
      "massRatioQ",
      "referenceEccentricity",
      "referenceTimeCodeUnits",
      "remnant",
    ],
    "physicalSystem",
  );
  require(
    Array.isArray(physicalSystem.bodies)
      && physicalSystem.bodies.length === 2,
    "physicalSystem.bodies must contain exactly A and B",
  );
  const bodyMasses = physicalSystem.bodies.map((body, index) => {
    const validated = exactKeys(
      body,
      [
        "dimensionlessSpin",
        "id",
        "massFraction",
        "orbitPositionScale",
      ],
      `physicalSystem.bodies[${index}]`,
    );
    require(
      validated.id === (index === 0 ? "A" : "B"),
      `physicalSystem.bodies[${index}].id is invalid`,
    );
    const mass = finite(
      validated.massFraction,
      `physicalSystem.bodies[${index}].massFraction`,
    );
    require(
      mass > 0 && mass < 1,
      `physicalSystem.bodies[${index}].massFraction is invalid`,
    );
    const spin = finiteVector3(
      validated.dimensionlessSpin,
      `physicalSystem.bodies[${index}].dimensionlessSpin`,
    );
    require(
      Math.hypot(...spin) < 1,
      `physicalSystem.bodies[${index}] violates the Kerr bound`,
    );
    finite(
      validated.orbitPositionScale,
      `physicalSystem.bodies[${index}].orbitPositionScale`,
    );
    return mass;
  });
  require(
    Math.abs(bodyMasses[0] + bodyMasses[1] - 1) <= CONTRACT_EPSILON,
    "body mass fractions must sum to one",
  );
  require(
    Math.abs(
      physicalSystem.bodies[0].orbitPositionScale + bodyMasses[1]
    ) <= CONTRACT_EPSILON
      && Math.abs(
        physicalSystem.bodies[1].orbitPositionScale - bodyMasses[0]
      ) <= CONTRACT_EPSILON,
    "body orbit-position scales disagree with the centre-of-mass convention",
  );
  const massRatio = finite(
    physicalSystem.massRatioQ,
    "physicalSystem.massRatioQ",
  );
  const eccentricity = finite(
    physicalSystem.referenceEccentricity,
    "physicalSystem.referenceEccentricity",
  );
  finite(
    physicalSystem.referenceTimeCodeUnits,
    "physicalSystem.referenceTimeCodeUnits",
  );
  require(massRatio > 0 && massRatio <= 1, "mass ratio is invalid");
  require(
    eccentricity >= 0 && eccentricity < 1,
    "reference eccentricity is invalid",
  );

  const remnant = exactKeys(
    physicalSystem.remnant,
    [
      "dimensionlessSpin",
      "finalHorizonDiagnostic",
      "massFraction",
      "metadataSource",
    ],
    "physicalSystem.remnant",
  );
  const remnantMass = finite(
    remnant.massFraction,
    "physicalSystem.remnant.massFraction",
  );
  const remnantSpin = finiteVector3(
    remnant.dimensionlessSpin,
    "physicalSystem.remnant.dimensionlessSpin",
  );
  require(
    remnantMass > 0 && remnantMass < 1,
    "remnant mass fraction is invalid",
  );
  require(
    Math.hypot(...remnantSpin) < 1,
    "remnant spin violates the Kerr bound",
  );
  nonEmptyString(remnant.metadataSource, "physicalSystem.remnant.metadataSource");
  const finalDiagnostic = exactKeys(
    remnant.finalHorizonDiagnostic,
    ["dimensionlessSpin", "massFraction", "sourceTimeCodeUnits"],
    "physicalSystem.remnant.finalHorizonDiagnostic",
  );
  const finalDiagnosticMass = finite(
    finalDiagnostic.massFraction,
    "physicalSystem.remnant.finalHorizonDiagnostic.massFraction",
  );
  const finalDiagnosticSpin = finiteVector3(
    finalDiagnostic.dimensionlessSpin,
    "physicalSystem.remnant.finalHorizonDiagnostic.dimensionlessSpin",
  );
  finite(
    finalDiagnostic.sourceTimeCodeUnits,
    "physicalSystem.remnant.finalHorizonDiagnostic.sourceTimeCodeUnits",
  );
  require(
    finalDiagnosticMass > 0 && finalDiagnosticMass < 1,
    "final horizon diagnostic mass is invalid",
  );
  require(
    Math.hypot(...finalDiagnosticSpin) < 1,
    "final horizon diagnostic spin violates the Kerr bound",
  );

  const timeReference = exactKeys(
    manifest.timeReference,
    [
      "alignmentCaveat",
      "horizonMapping",
      "protocolZeroEvent",
      "waveformMapping",
      "waveformPeakAmplitude",
      "waveformPeakSourceTimeM",
    ],
    "timeReference",
  );
  require(
    timeReference.protocolZeroEvent
      === "maximum amplitude of Extrapolated_N2 h(2,2)",
    "protocol time zero is invalid",
  );
  nonEmptyString(timeReference.alignmentCaveat, "timeReference.alignmentCaveat");
  nonEmptyString(timeReference.horizonMapping, "timeReference.horizonMapping");
  nonEmptyString(timeReference.waveformMapping, "timeReference.waveformMapping");
  require(
    finite(
      timeReference.waveformPeakAmplitude,
      "timeReference.waveformPeakAmplitude",
    ) > 0,
    "waveform peak amplitude must be positive",
  );
  finite(
    timeReference.waveformPeakSourceTimeM,
    "timeReference.waveformPeakSourceTimeM",
  );

  const events = exactKeys(manifest.events, EXPECTED_EVENT_KEYS, "events");
  const eventTimes = {};
  for (const eventName of EXPECTED_EVENT_KEYS) {
    const event = exactKeys(
      events[eventName],
      ["source", "tProtocolM"],
      `events.${eventName}`,
    );
    nonEmptyString(event.source, `events.${eventName}.source`);
    eventTimes[eventName] = finite(
      event.tProtocolM,
      `events.${eventName}.tProtocolM`,
    );
  }

  const dynamics = exactKeys(
    manifest.dynamics,
    [
      "asset",
      "declaredMaxInterpolationError",
      "finalTimeM",
      "firstTimeM",
      "interpolation",
      "measuredMaxInterpolationError",
      "postHorizonPolicy",
      "renderTransition",
      "sourceChannels",
    ],
    "dynamics",
  );
  const asset = exactKeys(
    dynamics.asset,
    [
      "byteLength",
      "columns",
      "encoding",
      "sampleCount",
      "schema",
      "sha256",
      "uri",
    ],
    "dynamics.asset",
  );
  require(
    sameArray(asset.columns, EXPECTED_COLUMNS),
    "manifest sample columns changed",
  );
  require(
    asset.schema === SAMPLE_SCHEMA,
    "unexpected sample asset schema",
  );
  require(
    asset.encoding === "utf-8 minified JSON",
    "unexpected sample encoding",
  );
  require(
    typeof asset.uri === "string"
      && /^[A-Za-z0-9._-]+\.json$/.test(asset.uri),
    "sample URI must be a local JSON filename",
  );
  require(
    typeof asset.sha256 === "string"
      && /^[0-9a-f]{64}$/.test(asset.sha256),
    "sample SHA-256 is invalid",
  );
  require(
    Number.isInteger(asset.byteLength) && asset.byteLength > 0,
    "sample byte length is invalid",
  );
  require(
    Number.isInteger(asset.sampleCount) && asset.sampleCount >= 8,
    "sample count is invalid",
  );
  nonEmptyString(dynamics.interpolation, "dynamics.interpolation");
  nonEmptyString(dynamics.postHorizonPolicy, "dynamics.postHorizonPolicy");
  const sourceChannels = exactKeys(
    dynamics.sourceChannels,
    ["h22", "orbitalPhaseUnwrappedRad", "separationM"],
    "dynamics.sourceChannels",
  );
  for (const channel of Object.keys(sourceChannels)) {
    nonEmptyString(
      sourceChannels[channel],
      `dynamics.sourceChannels.${channel}`,
    );
  }
  const renderTransition = exactKeys(
    dynamics.renderTransition,
    ["completeEvent", "kind", "quantity", "startEvent"],
    "dynamics.renderTransition",
  );
  require(
    renderTransition.startEvent === "commonApparentHorizonFirst"
      && renderTransition.completeEvent === "waveformPeak"
      && renderTransition.kind === "smoothstep",
    "render transition anchors or interpolation are invalid",
  );
  nonEmptyString(
    renderTransition.quantity,
    "dynamics.renderTransition.quantity",
  );
  validateErrorBounds(dynamics);
  const firstTimeM = finite(dynamics.firstTimeM, "dynamics.firstTimeM");
  const finalTimeM = finite(dynamics.finalTimeM, "dynamics.finalTimeM");
  const individualHorizonsLastM = eventTimes.individualHorizonsLast;
  const commonHorizonM = eventTimes.commonApparentHorizonFirst;
  const waveformPeakM = eventTimes.waveformPeak;
  require(
    firstTimeM < individualHorizonsLastM
      && individualHorizonsLastM < commonHorizonM
      && commonHorizonM < waveformPeakM
      && waveformPeakM < finalTimeM,
    "source events are not ordered",
  );
  require(waveformPeakM === 0, "waveform peak must define protocol time zero");
  require(
    Math.abs(eventTimes.relaxation - firstTimeM) <= 1e-6,
    "relaxation event disagrees with the dynamics start",
  );
  require(
    Math.abs(eventTimes.playbackEnd - finalTimeM) <= CONTRACT_EPSILON,
    "playback-end event disagrees with the dynamics end",
  );

  const rendererAdapter = exactKeys(
    manifest.rendererAdapter,
    [
      "lightPropagation",
      "massAbi",
      "nearZoneMetricConsumed",
      "positionsFrozenPerRay",
      "shaderBundleId",
      "stateAbi",
    ],
    "rendererAdapter",
  );
  require(
    rendererAdapter.shaderBundleId === "binary-approx-v1",
    "renderer adapter must preserve the existing shader bundle",
  );
  require(
    rendererAdapter.lightPropagation === "weak-field-fast-light"
      && rendererAdapter.nearZoneMetricConsumed === false
      && rendererAdapter.positionsFrozenPerRay === true,
    "light-propagation boundary is ambiguous",
  );
  require(
    sameArray(rendererAdapter.stateAbi, EXPECTED_STATE_ABI),
    "renderer state ABI changed",
  );
  require(
    sameArray(rendererAdapter.massAbi, EXPECTED_MASS_ABI),
    "renderer mass ABI changed",
  );

  const playback = exactKeys(
    manifest.playback,
    [
      "cycleDurationSecondsAtNominalRate",
      "endHoldSeconds",
      "loop",
      "scrubCoordinate",
      "slowMotion",
    ],
    "playback",
  );
  const cycleDuration = finite(
    playback.cycleDurationSecondsAtNominalRate,
    "playback.cycleDurationSecondsAtNominalRate",
  );
  const endHoldSeconds = finite(
    playback.endHoldSeconds,
    "playback.endHoldSeconds",
  );
  require(cycleDuration > 0, "nominal cycle duration must be positive");
  require(
    endHoldSeconds >= 0 && endHoldSeconds < cycleDuration,
    "end hold must be non-negative and shorter than the nominal cycle",
  );
  require(playback.loop === true, "this pinned playback must loop");
  require(
    playback.scrubCoordinate === "linear protocol time",
    "scrub coordinate must be linear protocol time",
  );
  const slowMotion = exactKeys(
    playback.slowMotion,
    [
      "enabledByDefault",
      "endTimeM",
      "rateMultiplier",
      "startTimeM",
      "status",
    ],
    "playback.slowMotion",
  );
  require(
    typeof slowMotion.enabledByDefault === "boolean",
    "slowMotion.enabledByDefault must be boolean",
  );
  const slowStartM = finite(
    slowMotion.startTimeM,
    "playback.slowMotion.startTimeM",
  );
  const slowEndM = finite(
    slowMotion.endTimeM,
    "playback.slowMotion.endTimeM",
  );
  const slowMultiplier = finite(
    slowMotion.rateMultiplier,
    "playback.slowMotion.rateMultiplier",
  );
  require(
    firstTimeM <= slowStartM
      && slowStartM < slowEndM
      && slowEndM <= finalTimeM,
    "slow-motion window must lie inside the playback range",
  );
  require(
    slowStartM < commonHorizonM && waveformPeakM < slowEndM,
    "slow-motion window must contain the common horizon and waveform peak",
  );
  require(
    slowMultiplier > 0 && slowMultiplier < 1,
    "slow-motion multiplier must be in (0, 1)",
  );
  nonEmptyString(slowMotion.status, "playback.slowMotion.status");
  require(
    slowMotion.status.includes("presentation-only")
      && slowMotion.status.includes("not gravitational time dilation"),
    "slow motion must be labelled as presentation-only",
  );

  const rendererDefaults = exactKeys(
    manifest.rendererDefaults,
    [
      "exposure",
      "fieldOfViewDeg",
      "initialViewingInclinationDeg",
      "observerRadiusM",
      "raySteps",
    ],
    "rendererDefaults",
  );
  const observerRadiusM = finite(
    rendererDefaults.observerRadiusM,
    "rendererDefaults.observerRadiusM",
  );
  const inclinationDeg = finite(
    rendererDefaults.initialViewingInclinationDeg,
    "rendererDefaults.initialViewingInclinationDeg",
  );
  const fieldOfViewDeg = finite(
    rendererDefaults.fieldOfViewDeg,
    "rendererDefaults.fieldOfViewDeg",
  );
  const exposure = finite(
    rendererDefaults.exposure,
    "rendererDefaults.exposure",
  );
  require(observerRadiusM > 2, "observer radius must stay outside 2 M");
  require(
    inclinationDeg >= 0 && inclinationDeg <= 180,
    "initial viewing inclination is invalid",
  );
  require(
    fieldOfViewDeg > 0 && fieldOfViewDeg < 180,
    "field of view is invalid",
  );
  require(exposure > 0 && exposure <= 16, "exposure is invalid");
  require(
    Number.isInteger(rendererDefaults.raySteps)
      && rendererDefaults.raySteps > 64
      && rendererDefaults.raySteps <= MAX_SHADER_STEPS,
    `raySteps must be an integer in [65, ${MAX_SHADER_STEPS}]`,
  );

  const generation = exactKeys(
    manifest.generation,
    ["command", "deterministic", "generator", "generatorSha256"],
    "generation",
  );
  nonEmptyString(generation.command, "generation.command");
  nonEmptyString(generation.generator, "generation.generator");
  require(
    generation.deterministic === true,
    "asset generation must be deterministic",
  );
  require(
    typeof generation.generatorSha256 === "string"
      && /^[0-9a-f]{64}$/.test(generation.generatorSha256),
    "generator SHA-256 is invalid",
  );
}

function validatePayload(manifest, payload) {
  exactKeys(payload, ["schema", "columns", "samples"], "sample payload");
  require(payload.schema === SAMPLE_SCHEMA, "unexpected payload schema");
  require(
    sameArray(payload?.columns, EXPECTED_COLUMNS),
    "payload columns changed",
  );
  require(Array.isArray(payload?.samples), "payload samples are missing");
  require(
    payload.samples.length === manifest.dynamics.asset.sampleCount,
    "payload sample count disagrees with manifest",
  );

  let previousTime = -Infinity;
  let previousPhase = -Infinity;
  let previousBlend = -Infinity;
  let invalidSeen = false;
  let lastValidRow = null;
  let firstInvalidRow = null;
  for (let index = 0; index < payload.samples.length; index += 1) {
    const row = payload.samples[index];
    require(
      Array.isArray(row) && row.length === EXPECTED_COLUMNS.length,
      `sample ${index} has the wrong width`,
    );
    const values = row.slice(0, 6).map((value, column) => (
      finite(value, `sample ${index} column ${column}`)
    ));
    const validity = row[6];
    require(validity === 0 || validity === 1, `sample ${index} validity is illegal`);
    require(values[0] > previousTime, `sample ${index} time is not increasing`);
    require(values[1] > 0, `sample ${index} separation must stay positive`);
    require(
      values[2] >= previousPhase,
      `sample ${index} unwrapped phase runs backward`,
    );
    require(
      values[5] >= previousBlend && values[5] >= 0 && values[5] <= 1,
      `sample ${index} render blend is invalid`,
    );
    if (validity === 0) {
      if (!invalidSeen) {
        firstInvalidRow = row;
        require(
          lastValidRow !== null,
          "the first sample cannot have expired A/B horizons",
        );
        require(
          Math.abs(values[1] - lastValidRow[1]) <= CONTRACT_EPSILON
            && Math.abs(values[2] - lastValidRow[2]) <= CONTRACT_EPSILON,
          `sample ${index} does not hold the last valid A/B trajectory`,
        );
      } else {
        require(
          Math.abs(values[1] - firstInvalidRow[1]) <= CONTRACT_EPSILON
            && Math.abs(values[2] - firstInvalidRow[2]) <= CONTRACT_EPSILON,
          `sample ${index} invents post-horizon A/B motion`,
        );
      }
      invalidSeen = true;
    } else {
      require(!invalidSeen, `sample ${index} restores expired A/B horizons`);
      lastValidRow = row;
    }
    previousTime = values[0];
    previousPhase = values[2];
    previousBlend = values[5];
  }
  const first = payload.samples[0];
  const last = payload.samples[payload.samples.length - 1];
  require(
    invalidSeen && first[6] === 1 && last[6] === 0,
    "horizon validity transition is incomplete",
  );
  require(
    Math.abs(
      lastValidRow[0]
        - manifest.events.individualHorizonsLast.tProtocolM
    ) <= 1e-6,
    "last valid A/B sample disagrees with the declared source event",
  );
  require(
    Math.abs(
      firstInvalidRow[0]
        - manifest.events.commonApparentHorizonFirst.tProtocolM
    ) <= 1e-6,
    "first invalid A/B sample disagrees with the common-horizon event",
  );
  require(
    Math.abs(first[0] - manifest.dynamics.firstTimeM) <= 1e-9,
    "payload first time disagrees with manifest",
  );
  require(
    Math.abs(last[0] - manifest.dynamics.finalTimeM) <= 1e-9,
    "payload final time disagrees with manifest",
  );
  require(first[5] === 0 && last[5] === 1, "render transition is incomplete");
}

export function createDynamicsTrack(manifest, payload) {
  validateManifest(manifest);
  validatePayload(manifest, payload);
  const rows = payload.samples;
  const firstTimeM = rows[0][0];
  const finalTimeM = rows[rows.length - 1][0];
  const individualHorizonsLastM = (
    manifest.events.individualHorizonsLast.tProtocolM
  );
  const commonHorizonM = (
    manifest.events.commonApparentHorizonFirst.tProtocolM
  );
  const waveformPeakM = manifest.events.waveformPeak.tProtocolM;

  function sampleAt(requestedTimeM) {
    const timeM = clamp(
      finite(requestedTimeM, "requested time"),
      firstTimeM,
      finalTimeM,
    );
    let lowerIndex = 0;
    let upperIndex = rows.length - 1;
    while (upperIndex - lowerIndex > 1) {
      const middle = Math.floor((lowerIndex + upperIndex) / 2);
      if (rows[middle][0] <= timeM) {
        lowerIndex = middle;
      } else {
        upperIndex = middle;
      }
    }
    const lower = rows[lowerIndex];
    const upper = rows[
      Math.abs(lower[0] - timeM) <= 1e-12 ? lowerIndex : upperIndex
    ];
    const duration = upper[0] - lower[0];
    const weight = duration > 0 ? (timeM - lower[0]) / duration : 0;
    const separationM = interpolate(lower[1], upper[1], weight);
    const orbitalPhaseRad = interpolate(lower[2], upper[2], weight);
    const h22Real = interpolate(lower[3], upper[3], weight);
    const h22Imag = interpolate(lower[4], upper[4], weight);
    const renderTopologyBlend = interpolate(lower[5], upper[5], weight);
    const individualHorizonsValid = lower[6] === 1 && upper[6] === 1;
    const regime = timeM <= individualHorizonsLastM
      ? "nr-inspiral"
      : timeM < commonHorizonM
        ? "nr-horizon-gap"
        : timeM < waveformPeakM
          ? "nr-merger"
          : "nr-ringdown";
    return Object.freeze({
      tM: timeM,
      separationM,
      orbitalPhaseRad,
      renderTopologyBlend,
      individualHorizonsValid,
      regime,
      waveform: Object.freeze({
        h22Real,
        h22Imag,
        amplitude: Math.hypot(h22Real, h22Imag),
      }),
      timelineFraction: (timeM - firstTimeM) / (finalTimeM - firstTimeM),
    });
  }

  return Object.freeze({
    manifest,
    firstTimeM,
    finalTimeM,
    sampleCount: rows.length,
    sampleAt,
  });
}

export async function loadBinaryDynamics(manifestUrl, fetchImpl = fetch) {
  const manifestResponse = await fetchImpl(manifestUrl);
  require(
    manifestResponse.ok,
    `manifest request failed (${manifestResponse.status})`,
  );
  const manifest = await manifestResponse.json();
  validateManifest(manifest);

  const sampleUrl = new URL(manifest.dynamics.asset.uri, manifestUrl);
  const sampleResponse = await fetchImpl(sampleUrl);
  require(
    sampleResponse.ok,
    `sample request failed (${sampleResponse.status})`,
  );
  const bytes = await sampleResponse.arrayBuffer();
  require(
    bytes.byteLength === manifest.dynamics.asset.byteLength,
    "sample byte length disagrees with manifest",
  );
  const actualSha256 = await sha256Hex(bytes);
  require(
    actualSha256 === manifest.dynamics.asset.sha256,
    "sample SHA-256 disagrees with manifest",
  );
  let payload;
  try {
    payload = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch (error) {
    throw new Error(
      `Binary dynamics contract violation: sample JSON is invalid: ${error}`,
    );
  }
  return createDynamicsTrack(manifest, payload);
}
