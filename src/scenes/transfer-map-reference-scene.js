import {
  loadTransferMap,
} from "../transfer-map-loader.js";
import {
  createTransferMapShaderBundle,
} from "../transfer-map-shaders.js";

const MANIFEST_URL = new URL(
  "../../assets/transfer-maps/schwarzschild-reference-v1/manifest.json",
  import.meta.url,
);

// This is the trust root for the bundled scientific product. The sidecar and
// every chunk are checked only after these exact manifest bytes are accepted.
export const REFERENCE_MANIFEST_SHA256 = (
  "8ab373b5af8f51dc7c702f7be418caf0a7bac5d8f59dd108afb0f0ee99d0e792"
);

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

function progressLabel(progress) {
  switch (progress.phase) {
    case "manifest":
      return progress.completed
        ? "Manifest SHA-256 已验证"
        : "正在获取固定版本 manifest…";
    case "sidecar":
      return progress.completed
        ? "Manifest sidecar 已交叉验证"
        : "正在核对 manifest sidecar…";
    case "chunks":
      return `正在验证 transfer-map chunks ${progress.completed}/${progress.total}…`;
    case "decoded":
      return `已严格解码 ${progress.total.toLocaleString("zh-CN")} 条光线记录`;
    default:
      return "正在验证 Schwarzschild transfer map…";
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

export async function createTransferMapReferenceScene({
  document: documentRef,
  ui,
  state,
  controls,
}) {
  if (typeof controls?.requestRender !== "function") {
    throw new Error("Transfer-map scene requires host render controls");
  }

  const elements = {
    canvas: requiredElement(documentRef, "universe"),
    eyebrow: requiredElement(documentRef, "sceneEyebrow"),
    title: requiredElement(documentRef, "panelTitle"),
    observerLabel: requiredElement(documentRef, "observerLabel"),
    radiusLabel: requiredElement(documentRef, "radiusLabel"),
    shadowLabel: requiredElement(documentRef, "shadowLabel"),
    massLabel: requiredElement(documentRef, "massLabel"),
    physicsNote: requiredElement(documentRef, "physicsNote"),
    sceneStatus: requiredElement(documentRef, "sceneStatus"),
    binaryTimeline: requiredElement(documentRef, "binaryTimeline"),
    desktopHint: requiredElement(documentRef, "desktopHint"),
    touchHint: documentRef.querySelector(".touch-hint"),
    scienceMode: ui.modeScience,
    alternateMode: ui.modeHubble,
    motion: ui.toggleMotion,
    reset: ui.resetView,
  };
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
            content: element !== elements.binaryTimeline,
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

  elements.sceneStatus.hidden = false;
  elements.sceneStatus.setAttribute("aria-live", "polite");
  let dataset;
  try {
    dataset = await loadTransferMap(MANIFEST_URL, {
      expectedManifestSha256: REFERENCE_MANIFEST_SHA256,
      onProgress(progress) {
        elements.sceneStatus.textContent = progressLabel(progress);
      },
    });
  } catch (error) {
    elements.sceneStatus.setAttribute("role", "alert");
    elements.sceneStatus.textContent = `Transfer map 验证失败：${error.message}`;
    throw error;
  }

  const manifest = dataset.manifest;
  const fixedCamera = cameraFromManifest(manifest);
  const observerRadius = Math.hypot(...manifest.observer.samples[0].eventNr.slice(1));
  const verticalFov = manifest.projection.verticalFieldOfViewRad;
  const lapse = Math.sqrt(1 - 2 / observerRadius);
  const shadowDiameterDeg = (
    2 * Math.asin(3 * Math.sqrt(3) * lapse / observerRadius) * 180 / Math.PI
  );
  const capturePercent = (
    100 * manifest.accuracy.outcomeFractions.captured
  );
  const shaderBundle = createTransferMapShaderBundle(dataset);
  let initialized = false;

  return Object.freeze({
    id: "transfer-map-reference",
    startsRunning: false,
    motionEnabled: false,
    cameraLocked: true,
    manifest,
    dataset,
    rendererOptions: Object.freeze({ shaderBundle }),

    initialize() {
      if (initialized) {
        return;
      }
      initialized = true;
      documentRef.documentElement.classList.add("scene-transfer-map-reference");
      documentRef.title = "Schwarzschild 传递图参考 · 深空观测台";
      state.running = false;
      state.distance = observerRadius;
      state.phase = 0;
      state.orbitTilt = 0;
      state.mode = 0;

      elements.canvas.setAttribute(
        "aria-label",
        "固定相机 Schwarzschild 真空传递图参考画面",
      );
      elements.eyebrow.textContent = "离线零测地线 · SHA-256 验证";
      elements.title.textContent = "Schwarzschild 传递图参考";
      elements.observerLabel.textContent = "固定观测者";
      elements.radiusLabel.textContent = "传递图 ABI";
      elements.shadowLabel.textContent = "解析阴影";
      elements.massLabel.textContent = "质量归一化";
      elements.sceneStatus.hidden = false;
      elements.sceneStatus.setAttribute("role", "status");
      elements.sceneStatus.textContent = [
        "stationary Schwarzschild reference",
        `${dataset.width}×${dataset.height}`,
        `${manifest.chunks.length} chunks SHA-256 verified`,
        "非 NR",
      ].join(" · ");
      elements.binaryTimeline.hidden = true;
      elements.scienceMode.textContent = "天空合成";
      elements.alternateMode.textContent = "结果分类";
      elements.desktopHint.textContent = "固定相机与投影 · 可调曝光和显示画质";
      if (elements.touchHint) {
        elements.touchHint.textContent = "固定相机 · 可调曝光和显示画质";
      }
      elements.physicsNote.innerHTML = [
        "本场景逐像素消费项目生成的<strong>真空 Schwarzschild ",
        "stationary transfer map</strong>：逃逸方向来自精确球对称零测地线，",
        "捕获像素不采样天空。它用于验证数据协议与 GPU 消费链，",
        "<strong>不是双黑洞 NR 光追，也不包含吸积盘或 GRMHD 辐射转移</strong>。",
        "当前银河照片仅作显示合成；其绝对 ICRS 天球配准尚未独立验证。",
      ].join("");

      for (const input of [ui.mass, ui.accretion, ui.timeScale]) {
        input.disabled = true;
      }
      elements.motion.disabled = true;
      elements.motion.setAttribute("aria-hidden", "true");
      elements.reset.disabled = true;
      elements.reset.setAttribute("aria-hidden", "true");
      controls.requestRender();
    },

    updateReadouts() {
      ui.massValue.textContent = "M = 1";
      ui.accretionValue.textContent = "真空 · 无发射模型";
      ui.exposureValue.textContent = `${state.exposure.toFixed(2)}×`;
      ui.timeScaleValue.textContent = "静态";
      ui.qualityValue.textContent = `${state.quality.toFixed(2)}×`;
      ui.observerValue.innerHTML = `R = ${observerRadius.toFixed(0)} M · β = 0 · FOV ${(verticalFov * 180 / Math.PI).toFixed(0)}°`;
      ui.rsValue.textContent = `${dataset.width}×${dataset.height} · 32 B/ray`;
      ui.shadowValue.textContent = `${shadowDiameterDeg.toFixed(3)}° · ${capturePercent.toFixed(2)}% captured`;
      return true;
    },

    cameraFrame() {
      return fixedCamera;
    },

    extendFrame(baseFrame) {
      return {
        ...baseFrame,
        time: 0,
        mode: 0,
        bloom: 0,
        motion: 0,
        cameraPos: fixedCamera.cameraPos,
        cameraRadius: observerRadius,
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
      };
    },

    dispose() {
      initialized = false;
      if (!original.rootHadClass) {
        documentRef.documentElement.classList.remove(
          "scene-transfer-map-reference",
        );
      }
      documentRef.title = original.documentTitle;
      for (const [element, snapshot] of original.elements) {
        restoreElement(element, snapshot);
      }
      Object.assign(state, original.state);
      controls.requestRender();
    },
  });
}
