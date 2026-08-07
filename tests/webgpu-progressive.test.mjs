import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  progressiveAccumulationFragmentWGSL,
  progressiveFrameState,
  progressiveHistorySignature,
  webGLRecoveryUrl,
  webGPUFallbackDescription,
  WebGPUFrameSubmissionGate,
  WebGPURenderer,
} from "../src/webgpu-renderer.js";

function frame(overrides = {}) {
  return {
    time: -10,
    massSolar: 1,
    accretion: 0,
    exposure: 1,
    mode: 0,
    steps: 160,
    cameraPos: [0, 0, 40],
    cameraRadius: 40,
    forward: [0, 0, -1],
    fov: 0.7,
    right: [1, 0, 0],
    skyRotation: 0,
    up: [0, 1, 0],
    diskOuterRadius: 18,
    renderScale: 1,
    bloom: 0,
    motion: 0,
    frame: 99,
    observerVelocity: [0, 0, 0],
    observerBeta: 0,
    sceneStrongFieldUniforms: new Float32Array(44),
    sceneStrongIntegrator: [0.02, 0.75, 3.6, 0.08],
    sceneStrongDomain: [96, 240, 0.035, 32],
    sceneStrongDiagnostics: [4, 180, 0.22, 1],
    strongFieldQuality: {
      accumulationIndex: 0,
      accumulationWeight: 1,
      historyEpoch: 1,
      historyReset: true,
    },
    ...overrides,
  };
}

function renderHarness(overrides = {}) {
  const passes = [];
  const queueWrites = [];
  const submitted = [];
  const encoder = {
    beginRenderPass(descriptor) {
      const record = {
        descriptor,
        pipeline: null,
        bindGroup: null,
        drawCount: 0,
        ended: false,
      };
      passes.push(record);
      return {
        setPipeline(pipeline) {
          record.pipeline = pipeline;
        },
        setBindGroup(index, bindGroup) {
          assert.equal(index, 0);
          record.bindGroup = bindGroup;
        },
        draw(count) {
          record.drawCount = count;
        },
        end() {
          record.ended = true;
        },
      };
    },
    finish() {
      return "command-buffer";
    },
  };
  const queue = {
    writeBuffer(...args) {
      queueWrites.push(args);
    },
  };
  const renderer = Object.assign(Object.create(WebGPURenderer.prototype), {
    progressiveAccumulation: {
      mode: "linear-hdr-running-average-v1",
    },
    progressiveHistoryValid: true,
    progressiveHistorySignature: null,
    progressiveHistoryEpoch: 1,
    traceView: { id: "trace-view" },
    postBindGroup: { id: "trace-post-bind-group" },
    accumulationBuffer: { id: "accumulation-buffer" },
    accumulationData: new Float32Array(4),
    accumulationViews: [
      { id: "history-view-0" },
      { id: "history-view-1" },
    ],
    accumulationBindGroups: [
      { id: "history-read-bind-group-0" },
      { id: "history-read-bind-group-1" },
    ],
    progressivePostBindGroups: [
      { id: "history-post-bind-group-0" },
      { id: "history-post-bind-group-1" },
    ],
    accumulationReadIndex: 0,
    tracePipeline: { id: "trace-pipeline" },
    accumulationPipeline: { id: "accumulation-pipeline" },
    postPipeline: { id: "post-pipeline" },
    context: {
      getCurrentTexture() {
        return {
          createView() {
            return { id: "canvas-view" };
          },
        };
      },
    },
    device: {
      queue,
      createCommandEncoder() {
        return encoder;
      },
    },
    submissionGate: {
      submit(receivedQueue, commandBuffers) {
        assert.equal(receivedQueue, queue);
        submitted.push(commandBuffers);
        return true;
      },
    },
    canSubmitFrame() {
      return true;
    },
    writeUniforms(nextFrame) {
      this.writtenFrame = nextFrame;
    },
    ...overrides,
  });
  return { renderer, passes, queueWrites, submitted };
}

test("progressive accumulation shader averages linear HDR before post", () => {
  assert.match(
    progressiveAccumulationFragmentWGSL,
    /textureLoad\(rawTrace/,
  );
  assert.match(
    progressiveAccumulationFragmentWGSL,
    /textureLoad\(previousFrame/,
  );
  assert.match(
    progressiveAccumulationFragmentWGSL,
    /return mix\(previous, raw, weight\)/,
  );
  assert.match(progressiveAccumulationFragmentWGSL, /if \(reset\)/);
  assert.doesNotMatch(progressiveAccumulationFragmentWGSL, /tone|exposure/i);
});

test("progressive state requires an exact running-average weight", () => {
  assert.deepEqual(progressiveFrameState(frame()), {
    accumulationIndex: 0,
    accumulationWeight: 1,
    historyEpoch: 1,
    historyReset: true,
  });
  assert.deepEqual(progressiveFrameState(frame({
    strongFieldQuality: {
      accumulationIndex: 3,
      accumulationWeight: 0.25,
      historyEpoch: 2,
      historyReset: false,
    },
  })), {
    accumulationIndex: 3,
    accumulationWeight: 0.25,
    historyEpoch: 2,
    historyReset: false,
  });
  assert.throws(
    () => progressiveFrameState(frame({
      strongFieldQuality: {
        accumulationIndex: 3,
        accumulationWeight: 0.5,
        historyEpoch: 2,
        historyReset: false,
      },
    })),
    /1\/\(sampleIndex\+1\)/,
  );
  assert.throws(
    () => progressiveFrameState(frame({
      strongFieldQuality: {
        accumulationIndex: 1,
        accumulationWeight: 0.5,
        historyEpoch: 2,
        historyReset: true,
      },
    })),
    /restart at sample zero/,
  );
});

test("history signature changes for every ray-domain input but not sample index", () => {
  const base = frame();
  const signature = progressiveHistorySignature(base);
  for (const changed of [
    frame({ time: -9 }),
    frame({ cameraPos: [0.1, 0, 40] }),
    frame({ fov: 0.71 }),
    frame({ steps: 96 }),
    frame({ sceneStrongIntegrator: [0.03, 1.1, 3.4, 0.1] }),
    frame({
      sceneStrongFieldUniforms: Float32Array.from(
        { length: 44 },
        (_, index) => index === 12 ? 1 : 0,
      ),
    }),
  ]) {
    assert.notEqual(progressiveHistorySignature(changed), signature);
  }
  assert.equal(
    progressiveHistorySignature(frame({
      frame: 7,
      strongFieldQuality: {
        accumulationIndex: 7,
        accumulationWeight: 0.125,
        historyEpoch: 1,
        historyReset: false,
      },
    })),
    signature,
  );
});

test("renderer independently forces sample zero after a stale camera history", () => {
  const renderer = Object.create(WebGPURenderer.prototype);
  renderer.progressiveAccumulation = {
    mode: "linear-hdr-running-average-v1",
  };
  renderer.progressiveHistoryValid = true;
  renderer.progressiveHistorySignature = progressiveHistorySignature(frame());
  renderer.progressiveHistoryEpoch = 1;

  let prepared = renderer.prepareProgressiveFrame(frame({
    strongFieldQuality: {
      accumulationIndex: 1,
      accumulationWeight: 0.5,
      historyEpoch: 1,
      historyReset: false,
    },
  }));
  assert.equal(prepared.state.historyReset, false);
  assert.equal(prepared.frame.frame, 1);

  prepared = renderer.prepareProgressiveFrame(frame({
    cameraPos: [1, 0, 40],
    strongFieldQuality: {
      accumulationIndex: 2,
      accumulationWeight: 1 / 3,
      historyEpoch: 1,
      historyReset: false,
    },
  }));
  assert.equal(prepared.state.historyReset, true);
  assert.equal(prepared.state.accumulationIndex, 0);
  assert.equal(prepared.state.accumulationWeight, 1);
  assert.equal(prepared.frame.frame, 0);
  assert.equal(prepared.frame.strongFieldQuality.historyReset, true);
});

test("motion always prevents cross-time history reuse", () => {
  const renderer = Object.create(WebGPURenderer.prototype);
  renderer.progressiveAccumulation = {
    mode: "linear-hdr-running-average-v1",
  };
  const moving = frame({
    motion: 1,
    strongFieldQuality: {
      accumulationIndex: 4,
      accumulationWeight: 0.2,
      historyEpoch: 5,
      historyReset: false,
    },
  });
  renderer.progressiveHistoryValid = true;
  renderer.progressiveHistorySignature = progressiveHistorySignature(moving);
  renderer.progressiveHistoryEpoch = 5;
  const prepared = renderer.prepareProgressiveFrame(moving);
  assert.equal(prepared.state.historyReset, true);
  assert.equal(prepared.state.accumulationIndex, 0);
  assert.equal(prepared.bypassAccumulation, true);
});

test("interactive frames independently reset and bypass progressive history", () => {
  const renderer = Object.create(WebGPURenderer.prototype);
  renderer.progressiveAccumulation = {
    mode: "linear-hdr-running-average-v1",
  };
  const interactive = frame({
    motion: 0,
    strongFieldQuality: {
      convergencePhase: "interactive",
      accumulationIndex: 4,
      accumulationWeight: 0.2,
      historyEpoch: 5,
      historyReset: false,
    },
  });
  renderer.progressiveHistoryValid = true;
  renderer.progressiveHistorySignature = progressiveHistorySignature(interactive);
  renderer.progressiveHistoryEpoch = 5;

  const prepared = renderer.prepareProgressiveFrame(interactive);
  assert.equal(prepared.state.historyReset, true);
  assert.equal(prepared.state.accumulationIndex, 0);
  assert.equal(prepared.bypassAccumulation, true);
});

test("dynamic frames encode trace and post only, then invalidate history", () => {
  const { renderer, passes, queueWrites, submitted } = renderHarness();
  const moving = frame({
    motion: 1,
    strongFieldQuality: {
      convergencePhase: "realtime",
      accumulationIndex: 7,
      accumulationWeight: 0.125,
      historyEpoch: 4,
      historyReset: false,
    },
  });
  renderer.progressiveHistorySignature = progressiveHistorySignature(moving);
  renderer.progressiveHistoryEpoch = 4;

  assert.equal(renderer.render(moving), true);
  assert.equal(passes.length, 2);
  assert.deepEqual(
    passes.map((pass) => pass.pipeline.id),
    ["trace-pipeline", "post-pipeline"],
  );
  assert.equal(passes[1].bindGroup, renderer.postBindGroup);
  assert.equal(queueWrites.length, 0);
  assert.equal(renderer.accumulationReadIndex, 0);
  assert.equal(renderer.progressiveHistoryValid, false);
  assert.equal(renderer.progressiveHistorySignature, null);
  assert.equal(renderer.progressiveHistoryEpoch, null);
  assert.equal(renderer.writtenFrame.frame, 0);
  assert.deepEqual(submitted, [["command-buffer"]]);
  for (const pass of passes) {
    assert.equal(pass.drawCount, 3);
    assert.equal(pass.ended, true);
  }
});

test("the first static sample still seeds FP16 history before post", () => {
  const { renderer, passes, queueWrites } = renderHarness({
    progressiveHistoryValid: false,
    progressiveHistoryEpoch: null,
  });
  const stationary = frame({
    motion: 0,
    strongFieldQuality: {
      convergencePhase: "accumulating",
      accumulationIndex: 0,
      accumulationWeight: 1,
      historyEpoch: 6,
      historyReset: true,
    },
  });

  assert.equal(renderer.render(stationary), true);
  assert.equal(passes.length, 3);
  assert.deepEqual(
    passes.map((pass) => pass.pipeline.id),
    ["trace-pipeline", "accumulation-pipeline", "post-pipeline"],
  );
  assert.equal(passes[1].bindGroup, renderer.accumulationBindGroups[0]);
  assert.equal(passes[2].bindGroup, renderer.progressivePostBindGroups[1]);
  assert.equal(queueWrites.length, 1);
  assert.equal(renderer.accumulationReadIndex, 1);
  assert.equal(renderer.progressiveHistoryValid, true);
  assert.equal(
    renderer.progressiveHistorySignature,
    progressiveHistorySignature(stationary),
  );
  assert.equal(renderer.progressiveHistoryEpoch, 6);
});

test("WebGPU submission gate permits exactly one in-flight frame", async () => {
  let now = 100;
  let resolveWork;
  const workDone = new Promise((resolve) => {
    resolveWork = resolve;
  });
  let submissions = 0;
  const queue = {
    submit(commandBuffers) {
      submissions += 1;
      assert.deepEqual(commandBuffers, ["frame-0"]);
    },
    onSubmittedWorkDone() {
      return workDone;
    },
  };
  const gate = new WebGPUFrameSubmissionGate(() => now);

  assert.equal(gate.readyForFrame, true);
  assert.equal(gate.submit(queue, ["frame-0"]), true);
  assert.equal(gate.readyForFrame, false);
  assert.equal(gate.canSubmitFrame(), false);
  assert.equal(gate.submit(queue, ["frame-1"]), false);
  assert.equal(submissions, 1);
  assert.equal(gate.consumeCompletedFrameTimeMs(), null);

  now = 143;
  resolveWork();
  await workDone;
  await Promise.resolve();

  assert.equal(gate.readyForFrame, true);
  assert.equal(gate.lastQueueCompletionAtMs, 143);
  assert.equal(gate.lastCompletedFrameTimeMs, 43);
  assert.equal(gate.consumeCompletedFrameTimeMs(), 43);
  assert.equal(gate.consumeCompletedFrameTimeMs(), null);
});

test("external queue work cannot inflate the next rendered-frame timing", async () => {
  let now = 100;
  let resolveFrame;
  const frameDone = new Promise((resolve) => {
    resolveFrame = resolve;
  });
  const queue = {
    submit() {},
    onSubmittedWorkDone() {
      return frameDone;
    },
  };
  const gate = new WebGPUFrameSubmissionGate(() => now);

  const finishUpload = gate.beginExternalQueueWork();
  assert.equal(gate.readyForFrame, false);
  assert.equal(gate.submit(queue, ["blocked-by-upload"]), false);

  // Hundreds of milliseconds spent decoding/copying a large sky texture are
  // outside the render sample. The first frame starts only after that queue
  // boundary has completed.
  now = 620;
  finishUpload();
  finishUpload();
  assert.equal(gate.readyForFrame, true);
  assert.equal(gate.submit(queue, ["first-clean-frame"]), true);

  now = 641;
  resolveFrame();
  await frameDone;
  await Promise.resolve();

  assert.equal(gate.lastCompletedFrameTimeMs, 21);
  assert.equal(gate.consumeCompletedFrameTimeMs(), 21);
});

test("a later resource upload does not overwrite an in-flight frame sample", async () => {
  let now = 40;
  let resolveFrame;
  const frameDone = new Promise((resolve) => {
    resolveFrame = resolve;
  });
  const queue = {
    submit() {},
    onSubmittedWorkDone() {
      return frameDone;
    },
  };
  const gate = new WebGPUFrameSubmissionGate(() => now);

  assert.equal(gate.submit(queue, ["frame"]), true);
  now = 45;
  const finishUpload = gate.beginExternalQueueWork();
  now = 72;
  resolveFrame();
  await frameDone;
  await Promise.resolve();

  assert.equal(gate.lastCompletedFrameTimeMs, 32);
  assert.equal(gate.readyForFrame, false);
  now = 900;
  finishUpload();
  assert.equal(gate.readyForFrame, true);
  assert.equal(gate.consumeCompletedFrameTimeMs(), 32);
});

test("failed WebGPU work closes the gate without inventing a timing sample", async () => {
  let rejectWork;
  const workDone = new Promise((resolve, reject) => {
    rejectWork = reject;
  });
  const expected = new Error("device lost");
  const gate = new WebGPUFrameSubmissionGate(() => 10);
  const failures = [];
  gate.onFailure = (error) => failures.push(error);
  const queue = {
    submit() {},
    onSubmittedWorkDone() {
      return workDone;
    },
  };

  assert.equal(gate.submit(queue, [{}]), true);
  rejectWork(expected);
  await assert.rejects(workDone, expected);
  await Promise.resolve();

  assert.equal(gate.readyForFrame, false);
  assert.equal(gate.lastCompletionError, expected);
  assert.deepEqual(failures, [expected]);
  assert.equal(gate.consumeCompletedFrameTimeMs(), null);
});

test("closing a submission gate ignores a late queue completion", async () => {
  let resolveWork;
  const workDone = new Promise((resolve) => {
    resolveWork = resolve;
  });
  let now = 20;
  const gate = new WebGPUFrameSubmissionGate(() => now);
  const queue = {
    submit() {},
    onSubmittedWorkDone() {
      return workDone;
    },
  };

  assert.equal(gate.submit(queue, [{}]), true);
  gate.close();
  now = 80;
  resolveWork();
  await workDone;
  await Promise.resolve();

  assert.equal(gate.readyForFrame, false);
  assert.equal(gate.lastQueueCompletionAtMs, null);
  assert.equal(gate.consumeCompletedFrameTimeMs(), null);
});

test("runtime recovery URL explicitly selects WebGL2 and records one bounded reason", () => {
  const recovered = new URL(webGLRecoveryUrl(
    "http://localhost:4173/?scene=binary-approx&renderer=webgpu#view",
    "device-lost",
  ));
  assert.equal(recovered.searchParams.get("scene"), "binary-approx");
  assert.equal(recovered.searchParams.get("renderer"), "webgl");
  assert.equal(recovered.searchParams.get("fallback"), "webgpu-device-lost");
  assert.equal(recovered.hash, "#view");
  assert.equal(
    webGPUFallbackDescription(recovered.searchParams.get("fallback")),
    "WebGPU device connection lost",
  );
  assert.equal(
    webGPUFallbackDescription(recovered.searchParams.get("fallback"), "zh-CN"),
    "WebGPU 设备连接丢失",
  );
  assert.equal(webGPUFallbackDescription("unknown"), "");
  assert.throws(
    () => webGLRecoveryUrl("http://localhost/", "arbitrary-message"),
    /Unsupported WebGPU recovery reason/,
  );
});

test("sky uploads enter a non-frame queue scope around the GPU copy", async () => {
  const source = await readFile(
    new URL("../src/webgpu-renderer.js", import.meta.url),
    "utf8",
  );
  assert.match(
    source,
    /finishExternalQueueWork = beginExternalQueueWork\?\.\(\)[\s\S]*copyExternalImageToTexture[\s\S]*await device\.queue\.onSubmittedWorkDone\(\)[\s\S]*finally \{[\s\S]*finishExternalQueueWork\?\.\(\)/,
  );
  assert.match(
    source,
    /loadSkyTexture\([\s\S]*\(\) => this\.beginResourceQueueWork\(\)/,
  );
});

test("the 16K sky is loaded only for an explicit ultra request", async () => {
  const [source, webGLSource] = await Promise.all([
    readFile(new URL("../src/webgpu-renderer.js", import.meta.url), "utf8"),
    readFile(new URL("../src/webgl-renderer.js", import.meta.url), "utf8"),
  ]);
  assert.match(
    source,
    /const blockForUltra = skyMode === "ultra"/,
  );
  assert.match(
    source,
    /loadSkyTexture\([\s\S]*skyUrl,[\s\S]*blockForUltra/,
  );
  assert.doesNotMatch(source, /scheduleBackgroundTask/);
  assert.doesNotMatch(
    source,
    /upgradeSkyTexture|adoptSkyTextureUpgrade|skyUpgradeController|lifecycleGeneration/,
  );
  assert.doesNotMatch(webGLSource, /scheduleBackgroundTask|upgradeSkyTexture/);
});

test("ESO 6K and Gaia 16K upload at their decoded original dimensions without fallback", async () => {
  const [webGPU, webGL, main] = await Promise.all([
    readFile(new URL("../src/webgpu-renderer.js", import.meta.url), "utf8"),
    readFile(new URL("../src/webgl-renderer.js", import.meta.url), "utf8"),
    readFile(new URL("../src/main.js", import.meta.url), "utf8"),
  ]);

  assert.match(main, /ultra: "\.\/assets\/gaia-edr3-16k\.png"/);
  assert.match(main, /high: "\.\/assets\/milky-way-360-6k\.jpg"/);
  for (const source of [webGPU, webGL]) {
    assert.match(
      source,
      /token: "gaia-edr3-16k", width: 16000, height: 8000/,
    );
    assert.match(
      source,
      /token: "milky-way-360-6k", width: 6000, height: 3000/,
    );
    assert.match(source, /const selected = requireUltra \? source\.ultra : source\.high/);
    const selector = source.match(/function selectedSkyUrl\([\s\S]*?\n}/)?.[0] || "";
    assert.doesNotMatch(selector, /fallback/);
    assert.match(
      source,
      /assertOriginalSkyDimensions\(url, width, height\)/,
    );
    assert.doesNotMatch(source, /resizeWidth|resizeHeight/);
  }

  assert.match(
    webGPU,
    /createImageBitmap\(blob, \{ colorSpaceConversion: "none" \}\)/,
  );
  assert.match(
    webGPU,
    /assertOriginalSkyDimensions\(url, bitmap\.width, bitmap\.height\)[\s\S]*size: \[bitmap\.width, bitmap\.height, 1\][\s\S]*\[bitmap\.width, bitmap\.height\]/,
  );
  assert.match(webGL, /new THREE\.TextureLoader\(\)\.load\(url, resolve, undefined, reject\)/);
  assert.match(
    webGL,
    /assertOriginalSkyDimensions\(url, width, height\)[\s\S]*this\.renderer\.initTexture\(texture\)/,
  );
});

test("the visible sky selector preserves the URL and exposes the 16K memory cost", async () => {
  const [html, main] = await Promise.all([
    readFile(new URL("../index.html", import.meta.url), "utf8"),
    readFile(new URL("../src/main.js", import.meta.url), "utf8"),
  ]);
  assert.match(html, /<html lang="en">/);
  assert.match(html, /for="languageSelect"[\s\S]*<select id="languageSelect"/);
  assert.match(html, /<label class="control-name" for="skySource" data-i18n="sky\.label">/);
  assert.match(html, /<select id="skySource"[^>]*aria-describedby="skySourceHint"/);
  assert.match(html, /<option value="high" data-i18n="sky\.eso">[\s\S]*ESO native 6000×3000/);
  assert.match(html, /<option value="ultra" data-i18n="sky\.gaia">[\s\S]*Gaia native 16000×8000/);
  assert.match(html, /native dimensions[\s\S]*no downsampling or silent fallback[\s\S]*236 MB[\s\S]*488 MiB/);
  assert.match(
    main,
    /const requestedSkyMode = query\.get\("sky"\) === "ultra" \? "ultra" : "high"/,
  );
  assert.match(
    main,
    /const skySourceSelect = document\.getElementById\("skySource"\)[\s\S]*skySourceSelect\.value = requestedSkyMode/,
  );
  assert.match(
    main,
    /skySourceSelect\?\.addEventListener\("change"[\s\S]*new URL\(window\.location\.href\)[\s\S]*searchParams\.set\("sky", selectedSkyMode\)[\s\S]*window\.location\.assign\(nextUrl\.href\)/,
  );
});

test("dispose releases renderer-owned resources and is idempotent", () => {
  const calls = [];
  const lifecycle = { renderer: null };
  const renderer = Object.assign(Object.create(WebGPURenderer.prototype), {
    disposed: false,
    deviceLossLifecycle: lifecycle,
    device: {
      removeEventListener: () => calls.push("remove-listener"),
      destroy: () => calls.push("destroy-device"),
    },
    handleUncapturedError() {},
    submissionGate: { close: () => calls.push("close-gate") },
    sceneResourceState: { dispose: () => calls.push("dispose-scene") },
    destroyProgressiveTargets: () => calls.push("destroy-history"),
    traceTexture: { destroy: () => calls.push("destroy-trace") },
    skyTexture: { destroy: () => calls.push("destroy-sky") },
    accumulationBuffer: { destroy: () => calls.push("destroy-accumulation") },
    uniformBuffer: { destroy: () => calls.push("destroy-uniforms") },
    context: { unconfigure: () => calls.push("unconfigure") },
    onLost() {},
    onError() {},
  });
  lifecycle.renderer = renderer;

  renderer.dispose();
  renderer.dispose();

  assert.equal(lifecycle.renderer, null);
  assert.equal(renderer.onLost, null);
  assert.equal(renderer.onError, null);
  assert.deepEqual(calls, [
    "remove-listener",
    "close-gate",
    "dispose-scene",
    "destroy-history",
    "destroy-trace",
    "destroy-sky",
    "destroy-accumulation",
    "destroy-uniforms",
    "unconfigure",
    "destroy-device",
  ]);
});

test("main runtime routes WebGPU failures through recovery and always reschedules RAF", async () => {
  const source = await readFile(
    new URL("../src/main.js", import.meta.url),
    "utf8",
  );
  assert.match(source, /webgpu\.onLost = \(info\) => \{/);
  assert.match(source, /webgpu\.onError = \(error\) => \{/);
  assert.match(
    source,
    /requestWebGLRendererRecovery\("render-error", error, webgpu\)/,
  );
  assert.match(source, /if \(webgpu\.lost\) \{/);
  assert.match(
    source,
    /finally \{[\s\S]*requestAnimationFrame\(animate\);[\s\S]*\}/,
  );
  assert.doesNotMatch(
    source,
    /webgpu\.onLost[\s\S]{0,260}showFatalError/,
  );
});
