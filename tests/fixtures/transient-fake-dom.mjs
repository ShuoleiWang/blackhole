// Fake observatory panel for the transient scene tests: the elements the
// shared transient host borrows, with reflected attributes, listeners that
// honour AbortSignal, and a document whose location carries the query.

export class FakeClassList {
  constructor() {
    this.values = new Set();
  }

  contains(name) {
    return this.values.has(name);
  }

  add(...names) {
    names.forEach((name) => this.values.add(name));
  }

  remove(...names) {
    names.forEach((name) => this.values.delete(name));
  }
}

export class FakeElement {
  constructor(id) {
    this.id = id;
    this.attributeValues = new Map();
    this.classList = new FakeClassList();
    this.listeners = new Map();
    this.html = "";
    this.value = "";
    this.hidden = false;
    this.mark = null;
    this.label = null;
  }

  get innerHTML() {
    return this.html;
  }

  set innerHTML(value) {
    this.html = String(value);
  }

  get textContent() {
    return this.html.replace(/<[^>]*>/g, "");
  }

  set textContent(value) {
    this.html = String(value);
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

  removeAttribute(name) {
    this.attributeValues.delete(name);
  }

  // Reflected attributes, as on HTMLInputElement / HTMLElement.
  get min() { return this.getAttribute("min") ?? ""; }
  set min(value) { this.setAttribute("min", value); }
  get max() { return this.getAttribute("max") ?? ""; }
  set max(value) { this.setAttribute("max", value); }
  get step() { return this.getAttribute("step") ?? ""; }
  set step(value) { this.setAttribute("step", value); }
  get title() { return this.getAttribute("title") ?? ""; }
  set title(value) { this.setAttribute("title", value); }

  addEventListener(type, listener, options = {}) {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
    options?.signal?.addEventListener?.("abort", () => listeners.delete(listener), { once: true });
  }

  listenerCount() {
    return [...this.listeners.values()].reduce((sum, set) => sum + set.size, 0);
  }

  querySelector(selector) {
    return selector === "span" ? this.mark : null;
  }

  closest(selector) {
    return selector === "label" ? this.label : null;
  }
}

const HOST_IDS = [
  "sceneEyebrow", "panelTitle", "observerLabel", "radiusLabel", "shadowLabel", "accretionControl",
  "modeScience", "modeHubble", "modeFrequency", "transferAdvancedDiagnostics", "physicsNote",
  "sceneStatus", "binaryTimeline", "binaryRegime", "binaryWaveformLabel", "binaryWaveformPath",
  "binaryTimeCursor", "binaryPlayPause", "binaryTimeScrubber", "binaryTimeValue", "binarySlowMotion",
  "binaryPlaybackRate", "desktopHint",
];
const UI_IDS = [
  "observerValue", "rsValue", "shadowValue", "exposureValue", "timeScaleValue", "qualityValue",
  "exposure", "timeScale", "mass",
];

export function fakeDocument(search = "") {
  const elements = new Map([...HOST_IDS, ...UI_IDS].map((id) => [id, new FakeElement(id)]));
  elements.get("binaryTimeline").hidden = true;
  elements.get("binaryPlayPause").mark = new FakeElement("play-mark");
  elements.get("mass").label = new FakeElement("mass-label");
  elements.get("modeScience").setAttribute("title", "original science title");
  elements.get("exposure").min = "0.2";
  elements.get("exposure").max = "3";
  elements.get("exposure").value = "1";
  const defaultView = { location: { search }, addEventListener() {} };
  const document = {
    title: "original",
    documentElement: new FakeElement("html"),
    defaultView,
    getElementById: (id) => elements.get(id) ?? null,
  };
  const ui = Object.fromEntries(UI_IDS.map((id) => [id, elements.get(id)]));
  return { document, elements, ui };
}
