import {
  loadTransferMap,
  readTransferMapRecord,
  TransferMapContractError,
  TRANSFER_MAP_ABI,
} from "../transfer-map-loader.js";
import {
  createTransferMapShaderBundle,
} from "../transfer-map-shaders.js";

const SCHWARZSCHILD_MANIFEST_URL = new URL(
  "../../assets/transfer-maps/schwarzschild-reference-v1/manifest.json",
  import.meta.url,
);
const KERR_REMNANT_MANIFEST_URL = new URL(
  "../../assets/transfer-maps/kerr-remnant-reference-v1/manifest.json",
  import.meta.url,
);

// These hashes are the browser trust roots. A query parameter may select a
// registry key, but it can never provide a URL or digest.
export const REFERENCE_MANIFEST_SHA256 = (
  "b70898ebe36f9f72147481c68fbd1ac31053cc6718da5258c60f84f4f0723e20"
);

export const TRUSTED_REFERENCE_REGISTRY = Object.freeze({
  schwarzschild: Object.freeze({
    key: "schwarzschild",
    datasetId: "schwarzschild-reference-v1",
    title: "Schwarzschild 离线校准",
    manifestUrl: SCHWARZSCHILD_MANIFEST_URL.href,
    expectedManifestSha256: REFERENCE_MANIFEST_SHA256,
  }),
  "kerr-remnant": Object.freeze({
    key: "kerr-remnant",
    datasetId: "kerr-remnant-reference-v1",
    title: "Kerr 余留体离线校准",
    manifestUrl: KERR_REMNANT_MANIFEST_URL.href,
    expectedManifestSha256: (
      "5b0022ab963c0cc35d3d8acab17190bd1294bc72da2b49003d785f964ac81d99"
    ),
  }),
});

export const TRANSFER_MAP_DIAGNOSTIC_MODES = Object.freeze([
  Object.freeze({ id: "sky", label: "合成图", range: null }),
  Object.freeze({ id: "outcome", label: "光线结果", range: null }),
  Object.freeze({ id: "lookback", label: "坐标回溯时间", range: "lookback" }),
  Object.freeze({ id: "frequency-shift", label: "频移因子 g", range: "frequencyShift" }),
  Object.freeze({ id: "null-residual", label: "零性残差", range: "nullResidual", logarithmic: true }),
  Object.freeze({ id: "projection-error", label: "投影误差", range: "projectionError", logarithmic: true }),
]);

const ADVANCED_DIAGNOSTIC_MODES = new Set([2, 4, 5]);

export class TransferMapSceneLoadError extends Error {
  constructor(message, options = {}) {
    super(message, options);
    this.name = "TransferMapSceneLoadError";
    this.sceneUiHandled = true;
  }
}

function requiredElement(documentRef, id) {
  const element = documentRef.getElementById(id);
  if (!element) {
    throw new Error(`Transfer-map scene requires interface element #${id}`);
  }
  return element;
}

function snapshotElement(element, { content = true } = {}) {
  return {
    attributes: Array.from(
      element.attributes,
      (attribute) => [attribute.name, attribute.value],
    ),
    innerHTML: content ? element.innerHTML : null,
    value: "value" in element ? element.value : null,
    disabled: "disabled" in element ? element.disabled : null,
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
  if (snapshot.disabled !== null) {
    element.disabled = snapshot.disabled;
  }
}

function locationHref(locationRef) {
  return locationRef?.href || "http://localhost/";
}

export function referenceHref(href, referenceKey) {
  const url = new URL(href);
  url.searchParams.set("scene", "transfer-map-reference");
  if (referenceKey === "schwarzschild") {
    url.searchParams.delete("reference");
  } else {
    url.searchParams.set("reference", referenceKey);
  }
  return url.href;
}

function defaultSceneHref(href) {
  const url = new URL(href);
  url.searchParams.delete("scene");
  url.searchParams.delete("reference");
  url.searchParams.delete("diagnostic");
  return url.href;
}

export function diagnosticModeFromSearch(searchParams) {
  const requested = searchParams?.get?.("diagnostic") || "sky";
  const mode = TRANSFER_MAP_DIAGNOSTIC_MODES.findIndex(
    (definition) => definition.id === requested,
  );
  return mode >= 0 ? mode : 0;
}

export function diagnosticHref(href, mode) {
  const definition = TRANSFER_MAP_DIAGNOSTIC_MODES[mode];
  if (!definition) {
    throw new RangeError(`Unknown transfer-map diagnostic mode ${mode}`);
  }
  const url = new URL(href);
  if (definition.id === "sky") {
    url.searchParams.delete("diagnostic");
  } else {
    url.searchParams.set("diagnostic", definition.id);
  }
  return url.href;
}

export function resolveTrustedReference(
  searchParams,
  registry = TRUSTED_REFERENCE_REGISTRY,
) {
  const requested = searchParams?.get?.("reference") || "schwarzschild";
  if (!Object.hasOwn(registry, requested)) {
    throw new TransferMapContractError(
      "$.reference",
      `unknown trusted reference key ${JSON.stringify(requested)}`,
    );
  }
  const entry = registry[requested];
  if (
    entry?.key !== requested
    || typeof entry.datasetId !== "string"
    || !entry.datasetId.length
    || typeof entry.manifestUrl !== "string"
  ) {
    throw new TransferMapContractError(
      `$.referenceRegistry.${requested}`,
      "registry entry is malformed",
    );
  }
  if (
    typeof entry.expectedManifestSha256 !== "string"
    || !/^[0-9a-f]{64}$/.test(entry.expectedManifestSha256)
  ) {
    throw new TransferMapContractError(
      `$.referenceRegistry.${requested}.expectedManifestSha256`,
      "reference is unavailable until a pinned manifest SHA-256 is registered",
    );
  }
  return entry;
}

function progressLabel(progress, title) {
  switch (progress.phase) {
    case "manifest":
      return progress.completed
        ? `${title} manifest SHA-256 已验证`
        : `正在获取 ${title} 的固定版本 manifest…`;
    case "sidecar":
      return progress.completed
        ? "Manifest sidecar 已交叉验证"
        : "正在核对 manifest sidecar…";
    case "chunks":
      return `正在验证 transfer-map chunks ${progress.completed}/${progress.total}…`;
    case "decoded":
      return `已严格解码 ${progress.total.toLocaleString("zh-CN")} 条光线记录`;
    default:
      return "正在验证 stationary transfer map…";
  }
}

function cameraFromManifest(manifest) {
  const matrix = manifest.camera.cameraToWorld;
  return Object.freeze({
    cameraPos: Object.freeze([matrix[3], matrix[7], matrix[11]]),
    right: Object.freeze([matrix[0], matrix[4], matrix[8]]),
    up: Object.freeze([matrix[1], matrix[5], matrix[9]]),
    forward: Object.freeze([matrix[2], matrix[6], matrix[10]]),
    observerVelocity: Object.freeze([0, 0, 0]),
    observerBeta: 0,
  });
}

function finiteVectorMagnitude(vector) {
  return Array.isArray(vector) && vector.length === 3
    ? Math.hypot(...vector)
    : 0;
}

function observerCoordinate(manifest, spinMagnitude) {
  const event = manifest.observer.samples[0].eventNr;
  const x = event[1];
  const y = event[2];
  const z = event[3];
  const euclideanNorm = Math.hypot(x, y, z);
  const chartDescription = [
    manifest.coordinates.nrChart.coordinates,
    manifest.coordinates.nrChart.gauge,
  ].join(" ");
  if (/Kerr-Schild/i.test(chartDescription)) {
    // In Cartesian Kerr-Schild coordinates, Euclidean rho is not the Kerr
    // oblate-spheroidal radial coordinate. Invert the defining quartic:
    // r^4 - (rho^2-a^2)r^2 - a^2 z^2 = 0.
    const aSquared = spinMagnitude ** 2;
    const rhoSquared = x * x + y * y + z * z;
    const difference = rhoSquared - aSquared;
    const rSquared = 0.5 * (
      difference
      + Math.sqrt(difference * difference + 4 * aSquared * z * z)
    );
    return Object.freeze({
      affineCameraDistance: euclideanNorm,
      coordinateRadius: Math.sqrt(Math.max(rSquared, 0)),
      label: "rKS",
    });
  }
  if (/Schwarzschild/i.test(chartDescription) && /areal/i.test(chartDescription)) {
    return Object.freeze({
      affineCameraDistance: euclideanNorm,
      coordinateRadius: euclideanNorm,
      label: "r areal",
    });
  }
  return Object.freeze({
    affineCameraDistance: euclideanNorm,
    coordinateRadius: euclideanNorm,
    label: "|x| chart",
  });
}

export function readoutsFromManifest(manifest) {
  const spinMagnitude = Math.max(
    0,
    ...manifest.physicalSystem.dimensionlessSpins.map(
      (spin) => finiteVectorMagnitude(spin.vector),
    ),
  );
  const coordinate = observerCoordinate(manifest, spinMagnitude);
  const captured = manifest.accuracy.outcomeFractions.captured;
  const escaped = manifest.accuracy.outcomeFractions.escaped;
  const fovDegrees = manifest.projection.verticalFieldOfViewRad * 180 / Math.PI;
  const mass = manifest.units.massNormalization;
  return Object.freeze({
    affineCameraDistance: coordinate.affineCameraDistance,
    observerCoordinateRadius: coordinate.coordinateRadius,
    observerCoordinateLabel: coordinate.label,
    spinMagnitude,
    captured,
    escaped,
    fovDegrees,
    massLabel: `${mass.symbol} = ${Number(mass.value).toLocaleString("zh-CN")}`,
    outcomeLabel: (
      `${(captured * 100).toFixed(2)}% 捕获 · `
      + `${(escaped * 100).toFixed(2)}% 逃逸`
    ),
  });
}

export function diagnosticRangeForMode(dataset, mode) {
  const definition = TRANSFER_MAP_DIAGNOSTIC_MODES[mode];
  const range = definition?.range
    ? dataset.diagnosticRanges[definition.range]
    : null;
  if (!range) {
    return Object.freeze([0, 1, 0, 0]);
  }
  let minimum = definition.logarithmic
    ? range.minimumPositive
    : range.minimum;
  let maximum = range.maximum;
  if (!(maximum > minimum)) {
    if (maximum > 0) {
      minimum = maximum * 0.5;
    } else {
      minimum = 0;
      maximum = 1;
    }
  }
  return Object.freeze([minimum, maximum, 0, 0]);
}

export function canvasPointToTransferPixel(
  clientX,
  clientY,
  rect,
  mapWidth,
  mapHeight,
) {
  if (
    !(rect?.width > 0)
    || !(rect?.height > 0)
    || !Number.isInteger(mapWidth)
    || !Number.isInteger(mapHeight)
    || mapWidth <= 0
    || mapHeight <= 0
  ) {
    return null;
  }
  const canvasUv = [
    (clientX - rect.left) / rect.width,
    (clientY - rect.top) / rect.height,
  ];
  if (
    canvasUv[0] < 0
    || canvasUv[0] > 1
    || canvasUv[1] < 0
    || canvasUv[1] > 1
  ) {
    return null;
  }

  const canvasAspect = rect.width / rect.height;
  const mapAspect = mapWidth / mapHeight;
  const mapUv = [...canvasUv];
  if (canvasAspect > mapAspect) {
    const widthFraction = mapAspect / canvasAspect;
    mapUv[0] = (canvasUv[0] - 0.5) / widthFraction + 0.5;
  } else {
    const heightFraction = canvasAspect / mapAspect;
    mapUv[1] = (canvasUv[1] - 0.5) / heightFraction + 0.5;
  }
  if (
    mapUv[0] < 0
    || mapUv[0] > 1
    || mapUv[1] < 0
    || mapUv[1] > 1
  ) {
    return null;
  }
  return Object.freeze({
    x: Math.min(mapWidth - 1, Math.floor(mapUv[0] * mapWidth)),
    y: Math.min(mapHeight - 1, Math.floor(mapUv[1] * mapHeight)),
  });
}

export function transferPixelToCanvasPoint(x, y, rect, mapWidth, mapHeight) {
  const mapUv = [(x + 0.5) / mapWidth, (y + 0.5) / mapHeight];
  const canvasAspect = rect.width / rect.height;
  const mapAspect = mapWidth / mapHeight;
  const canvasUv = [...mapUv];
  if (canvasAspect > mapAspect) {
    const widthFraction = mapAspect / canvasAspect;
    canvasUv[0] = 0.5 + (mapUv[0] - 0.5) * widthFraction;
  } else {
    const heightFraction = canvasAspect / mapAspect;
    canvasUv[1] = 0.5 + (mapUv[1] - 0.5) * heightFraction;
  }
  return Object.freeze({
    left: rect.left + canvasUv[0] * rect.width,
    top: rect.top + canvasUv[1] * rect.height,
  });
}

function createRecoveryLink(documentRef, className, label, href) {
  const link = documentRef.createElement("a");
  link.className = className;
  link.textContent = label;
  link.href = href;
  return link;
}

function showLoadFailure(documentRef, elements, error, href) {
  elements.sceneStatus.classList.add("is-error");
  elements.sceneStatus.setAttribute("role", "alert");
  const message = documentRef.createElement("span");
  message.textContent = `Transfer map 验证失败：${error.message}`;
  const actions = documentRef.createElement("span");
  actions.className = "scene-recovery-actions";
  actions.append(
    createRecoveryLink(documentRef, "scene-recovery-link", "重试验证", href),
    createRecoveryLink(
      documentRef,
      "scene-recovery-link",
      "返回实时双黑洞",
      defaultSceneHref(href),
    ),
  );
  elements.sceneStatus.replaceChildren(message, actions);

  // Asset authentication happens before the renderer and the shared mobile
  // controls are initialized. Keep recovery actions reachable on compact
  // viewports even when startup fails at that earlier boundary.
  elements.panel.classList.add("is-open");
  elements.panelToggle.setAttribute("aria-expanded", "true");
  elements.panelToggle.setAttribute("aria-label", "收起显示设置");
}

function formatMetric(value, digits = 3) {
  if (value === 0) {
    return "0";
  }
  if (Math.abs(value) < 1e-3 || Math.abs(value) >= 1e4) {
    return value.toExponential(3);
  }
  return value.toFixed(digits);
}

function validityNames(manifest, mask) {
  return Object.entries(manifest.recordLayout.validityBits)
    .filter(([, bit]) => mask & (1 << bit))
    .map(([name]) => name);
}

export async function createTransferMapReferenceScene({
  document: documentRef,
  ui,
  state,
  controls,
  searchParams = new URLSearchParams(documentRef.defaultView?.location?.search || ""),
  location = documentRef.defaultView?.location,
  history = documentRef.defaultView?.history,
  referenceRegistry = TRUSTED_REFERENCE_REGISTRY,
  loadTransferMapImpl = loadTransferMap,
}) {
  if (typeof controls?.requestRender !== "function") {
    throw new Error("Transfer-map scene requires host render controls");
  }

  const elements = {
    canvas: requiredElement(documentRef, "universe"),
    panel: requiredElement(documentRef, "panel"),
    panelToggle: requiredElement(documentRef, "togglePanel"),
    eyebrow: requiredElement(documentRef, "sceneEyebrow"),
    title: requiredElement(documentRef, "panelTitle"),
    observerLabel: requiredElement(documentRef, "observerLabel"),
    radiusLabel: requiredElement(documentRef, "radiusLabel"),
    shadowLabel: requiredElement(documentRef, "shadowLabel"),
    massLabel: requiredElement(documentRef, "massLabel"),
    physicsNote: requiredElement(documentRef, "physicsNote"),
    parameterTitle: requiredElement(documentRef, "parameterTitle"),
    parameterContext: requiredElement(documentRef, "parameterContext"),
    sceneStatus: requiredElement(documentRef, "sceneStatus"),
    binaryTimeline: requiredElement(documentRef, "binaryTimeline"),
    desktopHint: requiredElement(documentRef, "desktopHint"),
    touchHint: documentRef.querySelector(".touch-hint"),
    modeSwitch: requiredElement(documentRef, "modeSwitch"),
    scienceMode: ui.modeScience,
    alternateMode: ui.modeHubble,
    lookbackMode: requiredElement(documentRef, "modeLookback"),
    frequencyMode: requiredElement(documentRef, "modeFrequency"),
    nullMode: requiredElement(documentRef, "modeNull"),
    errorMode: requiredElement(documentRef, "modeError"),
    advancedDiagnostics: requiredElement(
      documentRef,
      "transferAdvancedDiagnostics",
    ),
    referenceSwitch: requiredElement(documentRef, "transferReferenceSwitch"),
    referenceSchwarzschild: requiredElement(documentRef, "referenceSchwarzschild"),
    referenceKerr: requiredElement(documentRef, "referenceKerr"),
    inspector: requiredElement(documentRef, "transferMapInspector"),
    inspectorClose: requiredElement(documentRef, "transferInspectorClose"),
    inspectorCoordinates: requiredElement(documentRef, "transferInspectorCoordinates"),
    inspectorDirection: requiredElement(documentRef, "transferInspectorDirection"),
    inspectorFrequency: requiredElement(documentRef, "transferInspectorFrequency"),
    inspectorLookback: requiredElement(documentRef, "transferInspectorLookback"),
    inspectorNull: requiredElement(documentRef, "transferInspectorNull"),
    inspectorError: requiredElement(documentRef, "transferInspectorError"),
    inspectorOutcome: requiredElement(documentRef, "transferInspectorOutcome"),
    inspectorValidity: requiredElement(documentRef, "transferInspectorValidity"),
    inspectorRaw: requiredElement(documentRef, "transferInspectorRaw"),
    marker: requiredElement(documentRef, "transferPixelMarker"),
    motion: ui.toggleMotion,
    reset: ui.resetView,
  };
  const shallowSnapshotElements = new Set([
    elements.panel,
    elements.panelToggle,
    elements.binaryTimeline,
    elements.modeSwitch,
    elements.advancedDiagnostics,
    elements.referenceSwitch,
    elements.inspector,
  ]);
  const original = {
    documentTitle: documentRef.title,
    rootHadClass: documentRef.documentElement.classList.contains(
      "scene-transfer-map-reference",
    ),
    elements: new Map(
      Object.values(elements)
        .filter(Boolean)
        .map((element) => [
          element,
          snapshotElement(element, {
            content: !shallowSnapshotElements.has(element),
          }),
        ]),
    ),
    state: {
      running: state.running,
      distance: state.distance,
      phase: state.phase,
      orbitTilt: state.orbitTilt,
      mode: state.mode,
    },
  };

  const href = locationHref(location);
  const currentDiagnosticMode = () => diagnosticModeFromSearch(
    location?.search !== undefined
      ? new URLSearchParams(location.search)
      : searchParams,
  );
  function configureReferenceLinks(activeKey, baseHref = locationHref(location)) {
    elements.referenceSwitch.hidden = false;
    for (const [key, link] of [
      ["schwarzschild", elements.referenceSchwarzschild],
      ["kerr-remnant", elements.referenceKerr],
    ]) {
      link.href = referenceHref(baseHref, key);
      const active = key === activeKey;
      link.classList.toggle("is-active", active);
      if (active) {
        link.setAttribute("aria-current", "true");
      } else {
        link.removeAttribute("aria-current");
      }
    }
  }
  configureReferenceLinks(searchParams.get("reference") || "schwarzschild", href);
  elements.sceneStatus.hidden = false;
  elements.sceneStatus.setAttribute("aria-live", "polite");

  let reference;
  let dataset;
  try {
    reference = resolveTrustedReference(searchParams, referenceRegistry);
    configureReferenceLinks(reference.key, href);
    elements.eyebrow.textContent = "固定信任根 · 正在验证";
    elements.title.textContent = reference.title;
    dataset = await loadTransferMapImpl(reference.manifestUrl, {
      expectedManifestSha256: reference.expectedManifestSha256,
      onProgress(progress) {
        elements.sceneStatus.textContent = progressLabel(
          progress,
          reference.title,
        );
      },
    });
    if (dataset.manifest.id !== reference.datasetId) {
      throw new TransferMapContractError(
        "$.id",
        `trusted registry expected ${reference.datasetId}, received ${dataset.manifest.id}`,
      );
    }
  } catch (error) {
    const cause = error instanceof Error ? error : new Error(String(error));
    showLoadFailure(documentRef, elements, cause, href);
    throw new TransferMapSceneLoadError(cause.message, { cause });
  }

  const manifest = dataset.manifest;
  const fixedCamera = cameraFromManifest(manifest);
  const readouts = readoutsFromManifest(manifest);
  const verticalFov = manifest.projection.verticalFieldOfViewRad;
  const shaderBundle = createTransferMapShaderBundle(dataset);
  const modeButtons = [
    elements.scienceMode,
    elements.alternateMode,
    elements.lookbackMode,
    elements.frequencyMode,
    elements.nullMode,
    elements.errorMode,
  ];
  let initialized = false;
  let selectedPixel = null;

  function currentCanvas() {
    return documentRef.getElementById("universe");
  }

  function updateMarker() {
    if (!selectedPixel) {
      elements.marker.hidden = true;
      return;
    }
    const canvas = currentCanvas();
    const point = transferPixelToCanvasPoint(
      selectedPixel.x,
      selectedPixel.y,
      canvas.getBoundingClientRect(),
      dataset.width,
      dataset.height,
    );
    elements.marker.style.left = `${point.left}px`;
    elements.marker.style.top = `${point.top}px`;
    elements.marker.hidden = false;
  }

  function inspectPixel(x, y) {
    const record = readTransferMapRecord(dataset, x, y);
    const validity = validityNames(manifest, record.validityMask);
    const directionValid = (
      record.validityMask & TRANSFER_MAP_ABI.validity.direction
    ) !== 0;
    const target = record.captureTargetId
      ? ` → ${record.captureTargetId}`
      : "";
    selectedPixel = { x, y };
    elements.inspectorCoordinates.textContent = `像素 x ${x} · y ${y}`;
    elements.inspectorDirection.textContent = directionValid
      ? record.escapeDirection.map((value) => value.toFixed(6)).join(", ")
      : "—";
    elements.inspectorFrequency.textContent = (
      record.validityMask & TRANSFER_MAP_ABI.validity.frequencyShift
    ) ? formatMetric(record.frequencyShiftG, 6) : "—";
    elements.inspectorLookback.textContent = (
      record.validityMask & TRANSFER_MAP_ABI.validity.lookback
    ) ? `${formatMetric(record.coordinateLookbackTimeM)} M` : "—";
    elements.inspectorNull.textContent = (
      record.validityMask & TRANSFER_MAP_ABI.validity.nullResidual
    ) ? formatMetric(record.nullResidual) : "—";
    elements.inspectorError.textContent = (
      record.validityMask & TRANSFER_MAP_ABI.validity.projectionError
    ) ? `${formatMetric(record.projectionErrorPx)} px` : "—";
    elements.inspectorOutcome.textContent = `${record.rayOutcomeName}${target}`;
    elements.inspectorValidity.textContent = (
      `0x${record.validityMask.toString(16).padStart(4, "0")} · `
      + (validity.join(", ") || "none")
    );
    elements.inspectorRaw.textContent = record.rawHex;
    elements.inspector.hidden = false;
    updateMarker();
  }

  function closeInspector({ focusCanvas = false } = {}) {
    selectedPixel = null;
    elements.inspector.hidden = true;
    elements.marker.hidden = true;
    if (focusCanvas) {
      currentCanvas().focus({ preventScroll: true });
    }
  }

  function handleInspectorClose() {
    closeInspector({ focusCanvas: true });
  }

  function inspectAtClientPoint(event) {
    const canvas = currentCanvas();
    if (event.target !== canvas || event.button !== 0) {
      return;
    }
    const pixel = canvasPointToTransferPixel(
      event.clientX,
      event.clientY,
      canvas.getBoundingClientRect(),
      dataset.width,
      dataset.height,
    );
    canvas.focus({ preventScroll: true });
    if (pixel) {
      inspectPixel(pixel.x, pixel.y);
    }
  }

  function inspectWithKeyboard(event) {
    if (event.target !== currentCanvas()) {
      return;
    }
    const step = event.shiftKey ? 10 : 1;
    let next = selectedPixel || {
      x: Math.floor(dataset.width / 2),
      y: Math.floor(dataset.height / 2),
    };
    let handled = true;
    if (event.key === "ArrowLeft") {
      next = { ...next, x: Math.max(0, next.x - step) };
    } else if (event.key === "ArrowRight") {
      next = { ...next, x: Math.min(dataset.width - 1, next.x + step) };
    } else if (event.key === "ArrowUp") {
      next = { ...next, y: Math.max(0, next.y - step) };
    } else if (event.key === "ArrowDown") {
      next = { ...next, y: Math.min(dataset.height - 1, next.y + step) };
    } else if (event.key === "Enter" || event.key === " ") {
      // Selecting the centre on first activation makes the keyboard path
      // useful without requiring a prior pointer event.
    } else if (event.key === "Escape") {
      closeInspector();
      event.preventDefault();
      return;
    } else {
      handled = false;
    }
    if (handled) {
      event.preventDefault();
      inspectPixel(next.x, next.y);
    }
  }

  function handleResize() {
    updateMarker();
  }

  return Object.freeze({
    id: "transfer-map-reference",
    startsRunning: false,
    motionEnabled: false,
    cameraLocked: true,
    panelLabel: "显示设置",
    manifest,
    dataset,
    reference,
    rendererOptions: Object.freeze({ shaderBundle }),

    initialize() {
      if (initialized) {
        return;
      }
      initialized = true;
      documentRef.documentElement.classList.add("scene-transfer-map-reference");
      documentRef.title = `${reference.title} · 深空观测台`;
      state.running = false;
      state.distance = readouts.affineCameraDistance;
      state.phase = 0;
      state.orbitTilt = 0;
      state.mode = currentDiagnosticMode();
      if (ADVANCED_DIAGNOSTIC_MODES.has(state.mode)) {
        elements.advancedDiagnostics.setAttribute("open", "");
      } else {
        elements.advancedDiagnostics.removeAttribute("open");
      }
      configureReferenceLinks(reference.key);
      closeInspector();

      const canvas = currentCanvas();
      canvas.setAttribute("tabindex", "0");
      canvas.setAttribute(
        "aria-label",
        `${reference.title}固定相机离线校准画面；点击像素或使用方向键检查光线记录`,
      );
      canvas.setAttribute("aria-describedby", "transferMapInspectorHelp");
      elements.eyebrow.textContent = "研究工具 · 固定相机离线校准";
      elements.title.textContent = reference.title;
      elements.observerLabel.textContent = "固定观测相机";
      elements.radiusLabel.textContent = "无量纲自旋";
      elements.shadowLabel.textContent = "光线结果";
      elements.massLabel.textContent = "质量标度";
      elements.parameterTitle.textContent = "显示设置";
      elements.parameterContext.textContent = "固定数据";
      const panelExpanded = elements.panel.classList.contains("is-open");
      elements.panelToggle.setAttribute("aria-expanded", String(panelExpanded));
      elements.panelToggle.setAttribute(
        "aria-label",
        panelExpanded ? "收起显示设置" : "展开显示设置",
      );
      elements.sceneStatus.hidden = false;
      elements.sceneStatus.classList.remove("is-error");
      elements.sceneStatus.setAttribute("role", "status");
      elements.sceneStatus.textContent = [
        "解析真空参考",
        "固定相机",
        `${dataset.width}×${dataset.height}`,
        `${manifest.chunks.length}/${manifest.chunks.length} 数据块 SHA-256 已验证`,
      ].join(" · ");
      elements.binaryTimeline.hidden = true;
      elements.modeSwitch.setAttribute(
        "aria-label",
        "固定相机离线校准诊断视图",
      );
      modeButtons.forEach((button, mode) => {
        button.textContent = TRANSFER_MAP_DIAGNOSTIC_MODES[mode].label;
      });
      elements.desktopHint.textContent = "点击查看光线记录 · 方向键移动 · Shift 加速";
      if (elements.touchHint) {
        elements.touchHint.textContent = "轻点画面查看像素光线记录";
      }
      const referenceDescription = reference.key === "kerr-remnant"
        ? "解析 Kerr 余留体真空时空"
        : "解析 Schwarzschild 真空时空";
      elements.physicsNote.textContent = [
        `${referenceDescription}的固定相机离线校准画面，只用于研究与验证 transfer-map 数据链；`,
        "不是双黑洞合并画面、NR 光追或高保真成品，不包含吸积发射。",
        ` 坐标：${manifest.coordinates.nrChart.coordinates}。`,
        ` 积分器：${manifest.rayIntegration.integrator.name}。`,
      ].join("");

      for (const input of [ui.mass, ui.accretion, ui.timeScale]) {
        input.disabled = true;
      }
      elements.motion.disabled = true;
      elements.motion.setAttribute("aria-hidden", "true");
      elements.reset.disabled = true;
      elements.reset.setAttribute("aria-hidden", "true");
      documentRef.addEventListener("click", inspectAtClientPoint);
      documentRef.addEventListener("keydown", inspectWithKeyboard);
      elements.inspectorClose.addEventListener("click", handleInspectorClose);
      documentRef.defaultView?.addEventListener("resize", handleResize);
      controls.requestRender();
    },

    updateReadouts() {
      ui.massValue.textContent = (
        `${readouts.massLabel} · |χ| = ${readouts.spinMagnitude.toFixed(6)}`
      );
      ui.accretionValue.textContent = "真空 · 无发射模型";
      ui.exposureValue.textContent = `${state.exposure.toFixed(2)}×`;
      ui.timeScaleValue.textContent = "静态";
      ui.qualityValue.textContent = `${state.quality.toFixed(2)}×`;
      ui.observerValue.textContent = (
        `${readouts.observerCoordinateLabel} = `
        + `${readouts.observerCoordinateRadius.toFixed(2)} M · `
        + `固定正交标架 · FOV ${readouts.fovDegrees.toFixed(1)}°`
      );
      ui.rsValue.textContent = `|χ| ${readouts.spinMagnitude.toFixed(6)}`;
      ui.shadowValue.textContent = readouts.outcomeLabel;
      return true;
    },

    cameraFrame() {
      return fixedCamera;
    },

    onModeChanged(mode) {
      if (!TRANSFER_MAP_DIAGNOSTIC_MODES[mode]) {
        return;
      }
      if (ADVANCED_DIAGNOSTIC_MODES.has(mode)) {
        elements.advancedDiagnostics.setAttribute("open", "");
      }
      const nextHref = diagnosticHref(locationHref(location), mode);
      if (typeof history?.replaceState === "function") {
        history.replaceState(history.state, "", nextHref);
      }
      configureReferenceLinks(reference.key, nextHref);
    },

    extendFrame(baseFrame) {
      return {
        ...baseFrame,
        time: 0,
        // Keep the shared post pass scientifically neutral. Diagnostic mode is
        // carried separately so false-colour values are not Hubble-graded.
        mode: 0,
        bloom: 0,
        motion: 0,
        cameraPos: fixedCamera.cameraPos,
        cameraRadius: readouts.affineCameraDistance,
        forward: fixedCamera.forward,
        right: fixedCamera.right,
        up: fixedCamera.up,
        fov: verticalFov,
        skyRotation: 0,
        observerVelocity: fixedCamera.observerVelocity,
        observerBeta: 0,
        sceneTransferState: [
          state.mode,
          dataset.width,
          dataset.height,
          0,
        ],
        sceneTransferRange: diagnosticRangeForMode(dataset, state.mode),
      };
    },

    dispose() {
      if (initialized) {
        documentRef.removeEventListener("click", inspectAtClientPoint);
        documentRef.removeEventListener("keydown", inspectWithKeyboard);
        elements.inspectorClose.removeEventListener("click", handleInspectorClose);
        documentRef.defaultView?.removeEventListener("resize", handleResize);
      }
      initialized = false;
      selectedPixel = null;
      if (!original.rootHadClass) {
        documentRef.documentElement.classList.remove(
          "scene-transfer-map-reference",
        );
      }
      documentRef.title = original.documentTitle;
      const liveCanvas = currentCanvas();
      const originalCanvasSnapshot = original.elements.get(elements.canvas);
      if (liveCanvas !== elements.canvas && originalCanvasSnapshot) {
        restoreElement(liveCanvas, originalCanvasSnapshot);
      }
      for (const [element, snapshot] of original.elements) {
        restoreElement(element, snapshot);
      }
      Object.assign(state, original.state);
      controls.requestRender();
    },
  });
}
