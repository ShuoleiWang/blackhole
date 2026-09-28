import { webcrypto } from "node:crypto";
import { readFile } from "node:fs/promises";

import { WebGPURenderer } from "../../src/webgpu-renderer.js";

globalThis.crypto ??= webcrypto;

// main.js starts immediately and owns browser globals/RAF for its lifetime, so
// this harness runs in a disposable child process. The renderer constructor is
// the only stubbed boundary; main.js, its dynamic scene import, and scene data
// loading all execute from the production modules.

class FakeEventTarget {
  constructor() {
    this.listeners = new Map();
  }

  addEventListener(type, listener, options = {}) {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
    if (options && typeof options === "object") {
      options.signal?.addEventListener?.("abort", () => {
        listeners.delete(listener);
      }, { once: true });
    }
  }

  removeEventListener(type, listener) {
    this.listeners.get(type)?.delete(listener);
  }

  dispatchEvent(event) {
    for (const listener of this.listeners.get(event.type) ?? []) {
      listener.call(this, event);
    }
    return true;
  }
}

class FakeClassList {
  constructor(initial = []) {
    this.values = new Set(initial);
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

  contains(name) {
    return this.values.has(name);
  }

  toggle(name, force = undefined) {
    const enabled = force === undefined ? !this.values.has(name) : Boolean(force);
    if (enabled) {
      this.values.add(name);
    } else {
      this.values.delete(name);
    }
    return enabled;
  }

  snapshot() {
    return [...this.values].sort();
  }
}

class FakeElement extends FakeEventTarget {
  constructor(id = "", tagName = "div") {
    super();
    this.id = id;
    this.tagName = tagName;
    this.attributeValues = new Map();
    this.classList = new FakeClassList();
    this.dataset = {};
    this._innerHTML = "";
    this._textContent = "";
    this.value = "";
    this.min = "";
    this.max = "";
    this.step = "";
    this.disabled = false;
    this.hidden = false;
    this.href = "";
    this.title = "";
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

  getAttribute(name) {
    return this.attributeValues.get(name) ?? null;
  }

  hasAttribute(name) {
    return this.attributeValues.has(name);
  }

  removeAttribute(name) {
    this.attributeValues.delete(name);
  }

  querySelector(selector) {
    if (selector === "span") {
      return this.mark;
    }
    return null;
  }

  replaceChildren(...children) {
    this.children = children;
  }

  append(...children) {
    this.children = [...(this.children ?? []), ...children];
  }

  closest(selector) {
    const tags = selector.split(",").map((part) => part.trim().split("[")[0]);
    return tags.includes(this.tagName) ? this : null;
  }

  setPointerCapture() {}

  releasePointerCapture() {}

  hasPointerCapture() {
    return false;
  }
}

const indexHtml = await readFile(
  new URL("../../index.html", import.meta.url),
  "utf8",
);
const ids = [...indexHtml.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
const elements = new Map(ids.map((id) => [id, new FakeElement(id)]));
const element = (id) => {
  const value = elements.get(id);
  if (!value) {
    throw new Error(`Entry harness is missing #${id}`);
  }
  return value;
};

Object.assign(element("mass"), { value: "9.81", min: "6", max: "10", step: "0.01" });
Object.assign(element("accretion"), {
  value: "-4.20",
  min: "-6",
  max: "-1",
  step: "0.01",
});
Object.assign(element("exposure"), {
  value: "1.00",
  min: "0.55",
  max: "1.55",
  step: "0.01",
});
Object.assign(element("timeScale"), { value: "24", min: "0", max: "64", step: "1" });
Object.assign(element("quality"), {
  value: "1.00",
  min: "0.50",
  max: "1.25",
  step: "0.05",
});
element("binaryTimeline").hidden = true;
element("transferReferenceSwitch").hidden = true;
element("transferMapInspector").hidden = true;
element("sceneStatus").hidden = true;
element("toggleMotion").mark = new FakeElement("toggle-motion-mark");
element("languageSelect").tagName = "select";
element("skySource").tagName = "select";
element("binaryPlayPause").mark = new FakeElement("binary-play-mark");

const root = new FakeElement("html");
root.lang = "en";
const app = new FakeElement("app");
const windowTarget = new FakeEventTarget();
const documentTarget = new FakeEventTarget();
const locationUrl = new URL(
  "https://blackhole.test/?scene=binary-dual-disk&binaryTime=-500&paused=1&lang=zh-CN&sky=ultra",
);
windowTarget.location = {
  get href() {
    return locationUrl.href;
  },
  get search() {
    return locationUrl.search;
  },
  assign(value) {
    locationUrl.href = value;
  },
  replace(value) {
    locationUrl.href = value;
  },
};
Object.assign(windowTarget, {
  innerWidth: 1440,
  innerHeight: 900,
  devicePixelRatio: 2,
});

const documentRef = {
  title: "Real-time Relativistic Black-hole Ray Tracing",
  documentElement: root,
  defaultView: windowTarget,
  hidden: false,
  addEventListener: documentTarget.addEventListener.bind(documentTarget),
  removeEventListener: documentTarget.removeEventListener.bind(documentTarget),
  dispatchEvent: documentTarget.dispatchEvent.bind(documentTarget),
  querySelector(selector) {
    if (selector === ".app") {
      return app;
    }
    if (selector.startsWith("#")) {
      return elements.get(selector.slice(1)) ?? null;
    }
    return null;
  },
  querySelectorAll() {
    return [];
  },
  getElementById(id) {
    return elements.get(id) ?? null;
  },
  createElement() {
    return new FakeElement();
  },
};

const storage = new Map();
globalThis.localStorage = {
  getItem(key) {
    return storage.get(key) ?? null;
  },
  setItem(key, value) {
    storage.set(key, String(value));
  },
};
globalThis.window = windowTarget;
globalThis.document = documentRef;
globalThis.location = windowTarget.location;
globalThis.HTMLInputElement = FakeElement;
globalThis.HTMLButtonElement = FakeElement;
globalThis.matchMedia = () => ({ matches: false, addEventListener() {} });
windowTarget.matchMedia = globalThis.matchMedia;

const animationFrames = [];
globalThis.requestAnimationFrame = (callback) => {
  animationFrames.push(callback);
  return animationFrames.length;
};
globalThis.cancelAnimationFrame = () => {};
windowTarget.requestAnimationFrame = globalThis.requestAnimationFrame;

globalThis.fetch = async (input) => {
  const bytes = await readFile(new URL(input));
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
};

const rendererCalls = [];
let renderCount = 0;
WebGPURenderer.create = async (canvas, skyUrls, options) => {
  rendererCalls.push({
    canvasId: canvas.id,
    skyUrls: { ...skyUrls },
    locale: options?.locale,
    shaderBundleId: options?.shaderBundle?.id,
  });
  const capabilities = Object.freeze({
    api: "webgpu",
    backend: "Stub WebGPU · Metal",
    progressiveAccumulation: "linear-hdr-running-average-v1",
    skyTexture: "16000×8000",
  });
  return {
    backend: capabilities.backend,
    gpu: "Stub M3 Pro",
    hdrMode: "Display-P3 · test",
    outputDescription: "Stub Display-P3 output",
    skyDetail: "16000×8000 native panorama",
    maxRenderDimension: 16384,
    capabilities,
    outputHDR: false,
    lost: false,
    pendingRuntimeError: null,
    resize() {},
    render() {
      renderCount += 1;
    },
    canSubmitFrame() {
      return true;
    },
    dispose() {},
  };
};

const startupErrors = [];
const startupInfo = [];
console.info = (...values) => {
  startupInfo.push(values.map(String).join(" "));
};
console.error = (...values) => {
  startupErrors.push(values.map(String).join(" "));
};

await import(new URL("../../src/main.js?dual-disk-entry-harness", import.meta.url));
for (let attempt = 0; attempt < 200; attempt += 1) {
  if (element("backendStatus").textContent === "Stub WebGPU · Metal") {
    break;
  }
  await new Promise((resolve) => setImmediate(resolve));
}

if (element("backendStatus").textContent !== "Stub WebGPU · Metal") {
  throw new Error(
    `main.js did not finish startup: ${[...startupErrors, ...startupInfo].join(" | ")}`,
  );
}

const rafCountAtStartup = animationFrames.length;
// Robustness probes on the live frame loop and keyboard handler.
let clock = performance.now() + 1_000;
const runAnimationFrame = () => {
  clock += 16;
  animationFrames[animationFrames.length - 1](clock);
};
const locationBeforeProbes = locationUrl.href;
const errorsBeforeProbes = startupErrors.length;
windowTarget.innerWidth = 0;
const rendersBeforeEmpty = renderCount;
runAnimationFrame();
runAnimationFrame();
const emptyViewport = {
  renders: renderCount - rendersBeforeEmpty,
  locationUnchanged: locationUrl.href === locationBeforeProbes,
  errors: startupErrors.slice(errorsBeforeProbes),
};
windowTarget.innerWidth = 1440;
const rendersBeforeRestore = renderCount;
for (let frame = 0; frame < 3; frame += 1) {
  runAnimationFrame();
}
emptyViewport.rendersAfterRestore = renderCount - rendersBeforeRestore;

const keyPrevented = (key, target, modifiers = {}) => {
  let prevented = false;
  windowTarget.dispatchEvent({
    type: "keydown",
    key,
    target,
    repeat: false,
    metaKey: false,
    ctrlKey: false,
    altKey: false,
    defaultPrevented: false,
    ...modifiers,
    preventDefault() {
      prevented = true;
    },
  });
  return prevented;
};
const body = new FakeElement("", "body");
const keyboard = {
  arrowOnSelect: keyPrevented("ArrowDown", element("languageSelect")),
  spaceOnSelect: keyPrevented(" ", element("skySource")),
  spaceOnSummary: keyPrevented(" ", new FakeElement("", "summary")),
  browserZoom: keyPrevented("=", body, { metaKey: true }),
  historyBack: keyPrevented("ArrowLeft", body, { altKey: true }),
  spaceRepeatToggles: (() => {
    const before = element("toggleMotion").dataset.state;
    keyPrevented(" ", body, { repeat: true });
    return element("toggleMotion").dataset.state !== before;
  })(),
  arrowOnBody: keyPrevented("ArrowLeft", body),
};

const attribute = (id, name) => element(id).getAttribute(name);
const report = {
  document: {
    title: documentRef.title,
    lang: root.lang,
    rootClasses: root.classList.snapshot(),
  },
  rendererCalls,
  navigation: {
    dualClasses: element("sceneBinaryDualDisk").classList.snapshot(),
    dualCurrent: attribute("sceneBinaryDualDisk", "aria-current"),
    dualHref: element("sceneBinaryDualDisk").href,
    otherCurrent: ["sceneBinary", "sceneSchwarzschild", "sceneTransferMap"]
      .map((id) => attribute(id, "aria-current")),
  },
  controls: {
    language: element("languageSelect").value,
    sky: element("skySource").value,
    accretionValue: element("accretion").value,
    accretionDisabled: element("accretion").disabled,
    accretionAriaHidden: attribute("accretionControl", "aria-hidden"),
    accretionLabel: element("accretionLabel").textContent,
  },
  readouts: {
    time: element("binaryTimeValue").textContent,
    playback: element("binaryPlaybackRate").textContent,
    emission: element("shadowValue").textContent,
    radius: element("rsValue").textContent,
    sceneStatus: element("sceneStatus").textContent,
  },
  rafCount: rafCountAtStartup,
  startupErrors: startupErrors.slice(0, errorsBeforeProbes),
  emptyViewport,
  keyboard,
};

process.stdout.write(`${JSON.stringify(report)}\n`);
