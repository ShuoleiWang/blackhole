import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  applyDocumentI18n,
  catalogKeys,
  createI18n,
  isSupportedLocale,
  languageUrl,
  localeFrom,
  persistLocale,
} from "../src/i18n.js";

function memoryStorage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem(key) {
      return values.get(key) ?? null;
    },
    setItem(key, value) {
      values.set(key, String(value));
    },
    value(key) {
      return values.get(key);
    },
  };
}

function fakeElement(dataset = {}) {
  return {
    dataset,
    textContent: "",
    innerHTML: "",
    attributes: new Map(),
    setAttribute(name, value) {
      this.attributes.set(name, String(value));
    },
  };
}

test("English is the strict default while explicit and stored Chinese remain available", () => {
  const storedChinese = memoryStorage({ "blackhole.language": "zh-CN" });
  assert.equal(localeFrom("", memoryStorage()), "en");
  assert.equal(localeFrom("", storedChinese), "zh-CN");
  assert.equal(localeFrom("?lang=en", storedChinese), "en");
  assert.equal(localeFrom("?lang=zh-CN", memoryStorage()), "zh-CN");
  assert.equal(localeFrom("?lang=unsupported", storedChinese), "en");
  assert.equal(createI18n("", memoryStorage()).locale, "en");
  assert.equal(isSupportedLocale("en"), true);
  assert.equal(isSupportedLocale("zh-CN"), true);
  assert.equal(isSupportedLocale("unsupported"), false);
  assert.equal(localeFrom("", { getItem() { throw new Error("blocked"); } }), "en");
  assert.equal(
    persistLocale("zh-CN", { setItem() { throw new Error("blocked"); } }),
    "zh-CN",
  );
});

test("a blocked global storage getter cannot prevent English startup", () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    get() {
      throw new Error("storage blocked");
    },
  });
  try {
    assert.equal(createI18n("").locale, "en");
    assert.equal(persistLocale("zh-CN"), "zh-CN");
  } finally {
    if (original) {
      Object.defineProperty(globalThis, "localStorage", original);
    } else {
      delete globalThis.localStorage;
    }
  }
});

test("language preference and URL switching preserve the full observation route", () => {
  const storage = memoryStorage();
  assert.equal(persistLocale("zh-CN", storage), "zh-CN");
  assert.equal(storage.value("blackhole.language"), "zh-CN");
  const chinese = new URL(languageUrl(
    "https://blackhole.test/?scene=transfer-map-reference&reference=kerr-remnant&diagnostic=null-residual&sky=ultra&renderer=webgl&hdr=0&binaryTime=12&paused=1#pixel",
    "zh-CN",
  ));
  assert.equal(chinese.searchParams.get("lang"), "zh-CN");
  for (const key of [
    "scene", "reference", "diagnostic", "sky", "renderer", "hdr", "binaryTime", "paused",
  ]) {
    assert.ok(chinese.searchParams.has(key));
  }
  assert.equal(chinese.hash, "#pixel");
  assert.equal(new URL(languageUrl(chinese.href, "en")).searchParams.has("lang"), false);
});

test("English and Chinese catalogs have exact key parity and preserve native sky contracts", () => {
  assert.deepEqual(catalogKeys("en"), catalogKeys("zh-CN"));
  const english = createI18n("en", null);
  const chinese = createI18n("zh-CN", null);
  assert.equal(english.t("scene.binary"), "Vacuum binary");
  assert.equal(chinese.t("scene.binary"), "真空双黑洞");
  assert.equal(english.t("scene.binaryDualDisk"), "Dual-disk binary");
  assert.equal(chinese.t("scene.binaryDualDisk"), "双吸积盘双黑洞");
  assert.match(
    english.t("dualDisk.physicsHtml", { sourceLink: "SXS" }),
    /C² transition.*strictly zero after the common horizon.*not full NR/s,
  );
  assert.match(
    chinese.t("dualDisk.physicsHtml", { sourceLink: "SXS" }),
    /C² 过渡关闭.*共同视界后严格为零.*非完整 NR/s,
  );
  for (const translator of [english, chinese]) {
    const sky = [
      translator.t("sky.eso"),
      translator.t("sky.gaia"),
      translator.t("sky.hint"),
    ].join(" ");
    assert.match(sky, /6000×3000/);
    assert.match(sky, /16000×8000/);
    assert.match(sky, /236 MB/);
    assert.match(sky, /488 MiB/);
  }
  assert.throws(() => english.t("missing.key"), /Unknown i18n key/);
  assert.throws(() => english.t("fallback.reason"), /Missing i18n value reason/);
});

test("catalog placeholders and every static HTML localization key stay in sync", async () => {
  const english = createI18n("en", null);
  const chinese = createI18n("zh-CN", null);
  const placeholderNames = (translator, key) => {
    const used = new Set();
    const values = new Proxy({}, {
      getOwnPropertyDescriptor(_target, property) {
        used.add(String(property));
        return { configurable: true, enumerable: true, value: property };
      },
      get(_target, property) {
        return `{${String(property)}}`;
      },
    });
    translator.t(key, values);
    return [...used].sort();
  };
  const knownKeys = catalogKeys("en");
  for (const key of knownKeys) {
    assert.deepEqual(
      placeholderNames(english, key),
      placeholderNames(chinese, key),
      key,
    );
  }

  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const staticKeys = [...html.matchAll(/data-i18n(?:-[a-z-]+)?="([^"]+)"/g)]
    .map((match) => match[1]);
  for (const key of staticKeys) {
    assert.ok(knownKeys.includes(key), `Unknown static i18n key ${key}`);
  }
});

test("document translation updates text, HTML, metadata, accessibility, and selector state", () => {
  const text = fakeElement({ i18n: "scene.binary" });
  const html = fakeElement({ i18nHtml: "physics.schwarzschildHtml" });
  const aria = fakeElement({ i18nAriaLabel: "canvas.ariaLabel" });
  const title = fakeElement({ i18nTitle: "view.reset" });
  const meta = fakeElement({ i18nContent: "meta.description" });
  const languageSelect = fakeElement();
  languageSelect.value = "en";
  const selectors = new Map([
    ["[data-i18n]", [text]],
    ["[data-i18n-html]", [html]],
    ["[data-i18n-aria-label]", [aria]],
    ["[data-i18n-title]", [title]],
    ["[data-i18n-content]", [meta]],
  ]);
  const documentRef = {
    documentElement: { lang: "en" },
    querySelectorAll(selector) {
      return selectors.get(selector) ?? [];
    },
    getElementById(id) {
      return id === "languageSelect" ? languageSelect : null;
    },
  };

  applyDocumentI18n(documentRef, createI18n("zh-CN", null));
  assert.equal(documentRef.documentElement.lang, "zh-CN");
  assert.equal(text.textContent, "真空双黑洞");
  assert.match(html.innerHTML, /Schwarzschild 度规/);
  assert.match(aria.attributes.get("aria-label"), /深空观测画面/);
  assert.equal(title.attributes.get("title"), "重置观测视角");
  assert.match(meta.attributes.get("content"), /Apple M3 Pro/);
  assert.equal(languageSelect.value, "zh-CN");
});
