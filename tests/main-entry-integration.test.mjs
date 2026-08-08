import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import test from "node:test";

test("main entry dynamically loads and configures the dual-disk scene", () => {
  const harness = fileURLToPath(
    new URL("./fixtures/main-dual-disk-entry-harness.mjs", import.meta.url),
  );
  const result = spawnSync(process.execPath, [harness], {
    cwd: fileURLToPath(new URL("..", import.meta.url)),
    encoding: "utf8",
    timeout: 20_000,
  });
  assert.equal(
    result.status,
    0,
    `entry harness failed\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`,
  );
  const report = JSON.parse(result.stdout.trim());

  assert.deepEqual(report.rendererCalls, [{
    canvasId: "universe",
    skyUrls: {
      ultra: "./assets/gaia-edr3-16k.png",
      high: "./assets/milky-way-360-6k.jpg",
    },
    locale: "zh-CN",
    shaderBundleId: "binary-dual-disk-strong-field-v1",
  }]);
  assert.equal(report.document.title, "双吸积盘双黑洞 · 深空观测台");
  assert.equal(report.document.lang, "zh-CN");
  for (const className of [
    "scene-binary",
    "scene-binary-dual-disk",
    "renderer-webgpu",
  ]) {
    assert.ok(report.document.rootClasses.includes(className), className);
  }

  assert.ok(report.navigation.dualClasses.includes("is-active"));
  assert.equal(report.navigation.dualCurrent, "page");
  assert.deepEqual(report.navigation.otherCurrent, [null, null, null]);
  const dualHref = new URL(report.navigation.dualHref, "https://blackhole.test/");
  assert.equal(dualHref.searchParams.get("scene"), "binary-dual-disk");
  assert.equal(dualHref.searchParams.get("binaryTime"), "-500");
  assert.equal(dualHref.searchParams.get("paused"), "1");
  assert.equal(dualHref.searchParams.get("lang"), "zh-CN");
  assert.equal(dualHref.searchParams.get("sky"), "ultra");

  assert.deepEqual(report.controls, {
    language: "zh-CN",
    sky: "ultra",
    accretionValue: "-1.70",
    accretionDisabled: false,
    accretionAriaHidden: null,
    accretionLabel: "单盘发射强度参数",
  });
  assert.equal(report.readouts.time, "t = −500.00 M");
  assert.equal(report.readouts.playback, "已暂停 · 0 M/s");
  assert.equal(report.readouts.emission, "无稳定微型盘环带 · 发射关闭");
  assert.match(report.readouts.radius, /^A \d+\.\d{2} M · B \d+\.\d{2} M$/);
  assert.match(report.readouts.sceneStatus, /Stub WebGPU · Metal/);
  assert.match(report.readouts.sceneStatus, /理想化薄微型盘/);
  assert.match(report.readouts.sceneStatus, /无 GRMHD 或自洽辐射转移 · 非完整 NR/);
  assert.equal(report.rafCount, 1);
  assert.deepEqual(report.startupErrors, []);
});
