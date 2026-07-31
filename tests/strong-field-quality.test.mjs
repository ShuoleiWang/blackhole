import assert from "node:assert/strict";
import test from "node:test";

import {
  applyStrongFieldFrameParameters,
  createStrongFieldQualityScheduler,
  M3_PRO_STRONG_FIELD_PROFILE,
} from "../src/strong-field-quality.js";

function frame(overrides = {}) {
  return {
    nowMs: 0,
    viewportWidth: 1920,
    viewportHeight: 1080,
    devicePixelRatio: 1,
    requestedQuality: 1,
    cameraRevision: 0,
    physicsRevision: 0,
    transportRevision: 0,
    interactionActive: false,
    timelineRunning: false,
    ...overrides,
  };
}

test("M3 Pro profile exposes the 24 FPS hard gate and 30 FPS target", () => {
  assert.equal(M3_PRO_STRONG_FIELD_PROFILE.hardMinimumFps, 24);
  assert.equal(M3_PRO_STRONG_FIELD_PROFILE.targetFps, 30);
  assert.ok(
    M3_PRO_STRONG_FIELD_PROFILE.upgradeFps
      > M3_PRO_STRONG_FIELD_PROFILE.targetFps,
  );
  assert.deepEqual(
    M3_PRO_STRONG_FIELD_PROFILE.tiers.map((tier) => tier.id),
    ["emergency", "survival", "interactive", "balanced", "fine"],
  );
  assert.deepEqual(
    M3_PRO_STRONG_FIELD_PROFILE.tiers.map((tier) => tier.stepBudget),
    [52, 52, 72, 160, 288],
  );
  assert.deepEqual(
    M3_PRO_STRONG_FIELD_PROFILE.tiers.map((tier) => tier.maxPixels),
    [280_000, 460_000, 12_000_000, 12_000_000, 12_000_000],
  );
  assert.equal(
    M3_PRO_STRONG_FIELD_PROFILE.tiers[2].resolutionScale,
    1,
  );
  assert.equal(M3_PRO_STRONG_FIELD_PROFILE.initialTier, 3);
  assert.equal(M3_PRO_STRONG_FIELD_PROFILE.interactionTier, 2);
  assert.equal(M3_PRO_STRONG_FIELD_PROFILE.realtimeTier, 2);
  assert.equal(M3_PRO_STRONG_FIELD_PROFILE.realtimeHardFloorTier, 2);
  assert.ok(
    M3_PRO_STRONG_FIELD_PROFILE.upgradeFrames
      < M3_PRO_STRONG_FIELD_PROFILE.maxAccumulationSamples,
  );
  assert.ok(
    M3_PRO_STRONG_FIELD_PROFILE.tiers.every(
      (tier) => tier.stepBudget <= 320,
    ),
  );
});

test("first frame invalidates history and dragging stays in immediate mode", () => {
  const scheduler = createStrongFieldQualityScheduler();
  let decision = scheduler.nextFrame(frame({ interactionActive: true }));
  assert.equal(decision.phase, "interactive");
  assert.equal(decision.qualityTierId, "interactive");
  assert.equal(decision.historyReset, true);
  assert.equal(decision.accumulationAllowed, false);
  assert.equal(decision.shouldRender, true);

  decision = scheduler.nextFrame(frame({
    nowMs: 16,
    interactionActive: true,
  }));
  assert.equal(decision.historyReset, true);
  assert.ok(decision.invalidationReasons.includes("interaction"));
  assert.equal(decision.accumulationIndex, 0);
});

test("a static view progresses through settling, refinement, accumulation, and steady", () => {
  const scheduler = createStrongFieldQualityScheduler({
    initialTier: 4,
    realtimeTier: 3,
    settleDelayMs: 100,
    refineDelayMs: 300,
    maxAccumulationSamples: 3,
  });
  let decision = scheduler.nextFrame(frame());
  assert.equal(decision.convergencePhase, "settling");
  assert.equal(decision.qualityTierId, "balanced");
  assert.equal(decision.historyReset, true);

  decision = scheduler.nextFrame(frame({ nowMs: 110 }));
  assert.equal(decision.convergencePhase, "refining");
  assert.equal(decision.qualityTierId, "balanced");
  assert.equal(decision.historyReset, true);
  assert.equal(decision.accumulationAllowed, true);
  assert.ok(decision.invalidationReasons.includes("accumulation-start"));

  decision = scheduler.nextFrame(frame({ nowMs: 310 }));
  assert.equal(decision.convergencePhase, "accumulating");
  assert.equal(decision.qualityTierId, "fine");
  assert.equal(decision.historyReset, true);
  assert.equal(decision.accumulationIndex, 0);

  decision = scheduler.nextFrame(frame({ nowMs: 326 }));
  assert.equal(decision.accumulationIndex, 1);
  assert.equal(decision.accumulationWeight, 0.5);
  assert.equal(decision.shouldRender, true);

  decision = scheduler.nextFrame(frame({ nowMs: 342 }));
  assert.equal(decision.accumulationIndex, 2);
  assert.equal(decision.accumulationWeight, 1 / 3);
  assert.equal(decision.shouldRender, true);

  decision = scheduler.nextFrame(frame({ nowMs: 358 }));
  assert.equal(decision.phase, "steady");
  assert.equal(decision.shouldRender, false);
});

test("camera and physical changes always restart from a full trace", () => {
  const scheduler = createStrongFieldQualityScheduler({
    settleDelayMs: 10,
    refineDelayMs: 20,
  });
  scheduler.nextFrame(frame());
  scheduler.nextFrame(frame({ nowMs: 25 }));
  let decision = scheduler.nextFrame(frame({ nowMs: 41 }));
  assert.ok(decision.accumulationIndex > 0);

  decision = scheduler.nextFrame(frame({
    nowMs: 57,
    cameraRevision: 1,
  }));
  assert.equal(decision.historyReset, true);
  assert.equal(decision.accumulationAllowed, false);
  assert.equal(decision.accumulationIndex, 0);
  assert.ok(decision.invalidationReasons.includes("camera-change"));

  decision = scheduler.nextFrame(frame({
    nowMs: 73,
    cameraRevision: 1,
    physicsRevision: 1,
  }));
  assert.equal(decision.historyReset, true);
  assert.ok(decision.invalidationReasons.includes("physics-change"));
});

test("a running timeline retraces every frame even with a coarse physics revision", () => {
  const scheduler = createStrongFieldQualityScheduler();
  scheduler.nextFrame(frame());
  for (let index = 1; index <= 4; index += 1) {
    const decision = scheduler.nextFrame(frame({
      nowMs: index * 16,
      timelineRunning: true,
    }));
    assert.equal(decision.convergencePhase, "realtime");
    assert.equal(decision.qualityTierId, "interactive");
    assert.equal(decision.historyReset, true);
    assert.equal(decision.accumulationAllowed, false);
    assert.equal(decision.accumulationIndex, 0);
    assert.ok(decision.invalidationReasons.includes("timeline-running"));
  }
});

test("catastrophic real-time timing preserves the dynamic floor and paused refinement", () => {
  const scheduler = createStrongFieldQualityScheduler({
    emaAlpha: 1,
    settleDelayMs: 100,
    refineDelayMs: 300,
  });

  let decision = scheduler.nextFrame(frame({ timelineRunning: true }));
  decision = scheduler.nextFrame(frame({
    nowMs: 70,
    frameTimeMs: 70,
    physicsRevision: 1,
    timelineRunning: true,
  }));
  assert.equal(decision.performanceTier, 2);
  decision = scheduler.nextFrame(frame({
    nowMs: 140,
    frameTimeMs: 70,
    physicsRevision: 2,
    timelineRunning: true,
  }));
  assert.equal(decision.performanceTier, 2);
  assert.equal(decision.qualityTierId, "interactive");

  decision = scheduler.nextFrame(frame({
    nowMs: 210,
    physicsRevision: 2,
    transportRevision: 1,
    timelineRunning: false,
  }));
  assert.equal(decision.performanceTier, 2);
  assert.equal(decision.qualityTierId, "balanced");
  assert.equal(decision.historyReset, true);

  decision = scheduler.nextFrame(frame({
    nowMs: 320,
    physicsRevision: 2,
    transportRevision: 1,
    timelineRunning: false,
  }));
  assert.equal(decision.convergencePhase, "refining");
  assert.equal(decision.qualityTierId, "balanced");
  assert.equal(decision.historyReset, true);
  assert.ok(decision.invalidationReasons.includes("accumulation-start"));

  decision = scheduler.nextFrame(frame({
    nowMs: 520,
    physicsRevision: 2,
    transportRevision: 1,
    timelineRunning: false,
  }));
  assert.equal(decision.performanceTier, 2);
  assert.equal(decision.qualityTierId, "fine");
  assert.equal(decision.historyReset, true);
});

test("starting paused accumulation resets sample zero before the first jitter", () => {
  const scheduler = createStrongFieldQualityScheduler({
    settleDelayMs: 100,
    refineDelayMs: 1_000,
  });
  let decision = scheduler.nextFrame(frame());
  assert.equal(decision.accumulationAllowed, false);

  decision = scheduler.nextFrame(frame({ nowMs: 110 }));
  assert.equal(decision.accumulationAllowed, true);
  assert.equal(decision.historyReset, true);
  assert.equal(decision.accumulationIndex, 0);
  assert.equal(decision.accumulationWeight, 1);

  decision = scheduler.nextFrame(frame({ nowMs: 126 }));
  assert.equal(decision.historyReset, false);
  assert.equal(decision.accumulationIndex, 1);
  assert.equal(decision.accumulationWeight, 0.5);

  for (let index = 2; index < 8; index += 1) {
    decision = scheduler.nextFrame(frame({ nowMs: 126 + index * 16 }));
    assert.equal(
      decision.shouldRender
        && decision.accumulationAllowed
        && !decision.historyReset
        && decision.accumulationIndex === 0,
      false,
    );
  }
});

test("a running timeline raises an underspecified initial tier to the hard floor", () => {
  const scheduler = createStrongFieldQualityScheduler({
    initialTier: 0,
    realtimeTier: 2,
    emaAlpha: 1,
    upgradeFrames: 2,
    qualityCooldownMs: 0,
  });
  let decision = scheduler.nextFrame(frame({ timelineRunning: true }));
  assert.equal(decision.performanceTier, 2);
  assert.equal(decision.qualityTierId, "interactive");
  assert.ok(
    decision.invalidationReasons.includes("realtime-resolution-floor"),
  );

  for (let index = 1; index <= 5; index += 1) {
    decision = scheduler.nextFrame(frame({
      nowMs: index * 20,
      frameTimeMs: 20,
      physicsRevision: index,
      timelineRunning: true,
    }));
  }
  assert.equal(decision.performanceTier, 2);
  assert.equal(decision.qualityTierId, "interactive");
});

test("an explicit input invalidates history on exactly the next scheduled frame", () => {
  const scheduler = createStrongFieldQualityScheduler({
    settleDelayMs: 10,
    refineDelayMs: 20,
  });
  scheduler.nextFrame(frame());
  scheduler.nextFrame(frame({ nowMs: 25 }));
  scheduler.nextFrame(frame({ nowMs: 41 }));
  scheduler.invalidate("wheel");
  assert.equal(scheduler.snapshot().pendingInvalidation, true);

  let decision = scheduler.nextFrame(frame({ nowMs: 57 }));
  assert.equal(decision.historyReset, true);
  assert.ok(decision.invalidationReasons.includes("wheel"));
  assert.equal(decision.stableElapsedMs, 0);

  decision = scheduler.nextFrame(frame({ nowMs: 58 }));
  assert.equal(decision.historyReset, false);
  assert.equal(decision.shouldRender, false);
});

test("the 24 FPS hard gate drops quality immediately and resets history", () => {
  const scheduler = createStrongFieldQualityScheduler({ emaAlpha: 1 });
  scheduler.nextFrame(frame({ timelineRunning: true }));
  const decision = scheduler.nextFrame(frame({
    nowMs: 50,
    frameTimeMs: 50,
    timelineRunning: true,
  }));
  assert.equal(decision.timing.fpsEma, 20);
  assert.equal(decision.performanceTier, 2);
  assert.equal(decision.historyReset, true);
  assert.ok(decision.invalidationReasons.includes("hard-fps-gate"));
});

test("a sustained severe miss cannot cross the native-resolution floor", () => {
  const scheduler = createStrongFieldQualityScheduler({ emaAlpha: 1 });
  scheduler.nextFrame(frame({ timelineRunning: true }));
  let decision = scheduler.nextFrame(frame({
    nowMs: 50,
    frameTimeMs: 70,
    timelineRunning: true,
  }));
  assert.equal(decision.performanceTier, 2);
  decision = scheduler.nextFrame(frame({
    nowMs: 120,
    frameTimeMs: 70,
    timelineRunning: true,
  }));
  assert.equal(decision.qualityTierId, "interactive");
  assert.equal(decision.performanceTier, 2);
  assert.ok(
    decision.resolution.pixelCount
      > M3_PRO_STRONG_FIELD_PROFILE.tiers[1].maxPixels,
  );
});

test("the hard gate cannot select emergency or survival", () => {
  const scheduler = createStrongFieldQualityScheduler({
    emaAlpha: 0.25,
    realtimeTier: 2,
  });
  scheduler.nextFrame(frame({ timelineRunning: true }));

  let decision = scheduler.nextFrame(frame({
    nowMs: 70,
    frameTimeMs: 70,
    timelineRunning: true,
  }));
  assert.equal(decision.performanceTier, 2);
  assert.equal(decision.qualityTierId, "interactive");
  assert.equal(decision.timing.sampleCount, 1);

  decision = scheduler.nextFrame(frame({
    nowMs: 83,
    frameTimeMs: 13,
    timelineRunning: true,
  }));
  assert.equal(decision.performanceTier, 2);
  assert.equal(decision.qualityTierId, "interactive");
  assert.ok(decision.timing.frameTimeEmaMs > 13);

  decision = scheduler.nextFrame(frame({
    nowMs: 96,
    frameTimeMs: 13,
    timelineRunning: true,
  }));
  assert.ok(decision.timing.frameTimeEmaMs > 13);
  assert.equal(
    decision.invalidationReasons.includes("hard-fps-gate"),
    false,
  );
});

test("the 30 FPS target may lower bookkeeping tier only to the hard floor", () => {
  const scheduler = createStrongFieldQualityScheduler({
    initialTier: 3,
    realtimeTier: 2,
    emaAlpha: 1,
    targetMissFrames: 3,
    upgradeFrames: 3,
    qualityCooldownMs: 100,
    settleDelayMs: 10,
    refineDelayMs: 20,
  });
  scheduler.nextFrame(frame({ timelineRunning: true }));
  let decision;
  for (let index = 1; index <= 3; index += 1) {
    decision = scheduler.nextFrame(frame({
      nowMs: index * 50,
      frameTimeMs: 35,
      timelineRunning: true,
    }));
  }
  assert.equal(decision.performanceTier, 2);
  assert.ok(decision.invalidationReasons.includes("target-fps-miss"));

  // 30-35 FPS is deliberately inside the hysteresis band and cannot upgrade.
  for (let index = 4; index <= 9; index += 1) {
    decision = scheduler.nextFrame(frame({
      nowMs: index * 50,
      frameTimeMs: 30,
      timelineRunning: true,
    }));
  }
  assert.equal(decision.performanceTier, 2);

  for (let index = 10; index <= 12; index += 1) {
    decision = scheduler.nextFrame(frame({
      nowMs: index * 50,
      frameTimeMs: 20,
      timelineRunning: true,
    }));
  }
  assert.equal(decision.performanceTier, 2);
  assert.equal(decision.qualityTierId, "interactive");
});

test("the real-time operating tier uses the 24-30 FPS envelope at native resolution", () => {
  const scheduler = createStrongFieldQualityScheduler({
    initialTier: 1,
    emaAlpha: 1,
    targetMissFrames: 2,
    qualityCooldownMs: 0,
  });
  scheduler.nextFrame(frame({ timelineRunning: true }));
  let decision;
  for (let index = 1; index <= 8; index += 1) {
    decision = scheduler.nextFrame(frame({
      nowMs: index * 35,
      frameTimeMs: 35,
      timelineRunning: true,
    }));
  }
  assert.equal(decision.timing.fpsEma, 1_000 / 35);
  assert.equal(decision.performanceTier, 2);
  assert.equal(decision.qualityTierId, "interactive");
  assert.equal(
    decision.invalidationReasons.includes("target-fps-miss"),
    false,
  );
});

test("timeline motion and dragging retain the full panel Retina raster under extreme misses", () => {
  for (const dynamicState of [
    { timelineRunning: true },
    { interactionActive: true },
  ]) {
    const scheduler = createStrongFieldQualityScheduler({ emaAlpha: 1 });
    let decision = scheduler.nextFrame(frame({
      viewportWidth: 1836,
      viewportHeight: 1376,
      devicePixelRatio: 2,
      ...dynamicState,
    }));

    for (let index = 1; index <= 12; index += 1) {
      decision = scheduler.nextFrame(frame({
        nowMs: index * 250,
        viewportWidth: 1836,
        viewportHeight: 1376,
        devicePixelRatio: 2,
        frameTimeMs: 250,
        physicsRevision: index,
        ...dynamicState,
      }));
      assert.equal(decision.qualityTierId, "interactive");
      assert.equal(decision.performanceTier, 2);
      assert.equal(decision.stepBudget, 72);
      assert.equal(decision.resolution.renderScale, 2);
      assert.equal(decision.resolution.width, 3672);
      assert.equal(decision.resolution.height, 2752);
    }
  }
});

test("deadline-adjacent jitter does not make the quality tier flap", () => {
  const scheduler = createStrongFieldQualityScheduler({
    emaAlpha: 0.25,
    targetMissFrames: 8,
    upgradeFrames: 8,
    qualityCooldownMs: 500,
    settleDelayMs: 10,
    refineDelayMs: 20,
  });
  scheduler.nextFrame(frame({ timelineRunning: true }));
  const observed = [];
  for (let index = 1; index <= 80; index += 1) {
    const decision = scheduler.nextFrame(frame({
      nowMs: index * 16,
      frameTimeMs: index % 2 ? 31 : 35,
      timelineRunning: true,
    }));
    observed.push(decision.performanceTier);
  }
  const changes = observed.filter(
    (value, index) => index > 0 && value !== observed[index - 1],
  );
  assert.ok(changes.length <= 1);
  assert.ok(new Set(observed.slice(-20)).size <= 1);
});

test("device loss suspends rendering and restoration starts a fresh epoch", () => {
  const scheduler = createStrongFieldQualityScheduler();
  scheduler.nextFrame(frame());
  scheduler.signalDeviceLost("Metal device removed");
  let decision = scheduler.nextFrame(frame({ nowMs: 16 }));
  assert.equal(decision.phase, "device-lost");
  assert.equal(decision.shouldRender, false);
  assert.equal(decision.backend.deviceReason, "Metal device removed");

  scheduler.signalDeviceRestored();
  decision = scheduler.nextFrame(frame({ nowMs: 32 }));
  assert.equal(decision.backend.deviceState, "ready");
  assert.equal(decision.historyReset, true);
  assert.equal(decision.shouldRender, true);
  assert.ok(decision.invalidationReasons.includes("device-restored"));
});

test("WebGL fallback is signalled, capped, and never accumulates history", () => {
  const scheduler = createStrongFieldQualityScheduler({
    settleDelayMs: 10,
    refineDelayMs: 20,
  });
  scheduler.nextFrame(frame());
  scheduler.setBackend("webgl2", "WebGPU unavailable");
  let decision = scheduler.nextFrame(frame({ nowMs: 25 }));
  assert.equal(decision.phase, "fallback");
  assert.equal(decision.backend.fallback, true);
  assert.ok(
    decision.qualityTier <= M3_PRO_STRONG_FIELD_PROFILE.fallbackMaxTier,
  );
  assert.equal(decision.accumulationAllowed, false);
  assert.equal(decision.historyReset, true);

  decision = scheduler.nextFrame(frame({ nowMs: 50 }));
  assert.equal(decision.phase, "fallback");
  assert.equal(decision.accumulationAllowed, false);
  assert.equal(decision.shouldRender, false);
});

test("hidden frames are ignored and resume invalidates accumulated history", () => {
  const scheduler = createStrongFieldQualityScheduler();
  scheduler.nextFrame(frame());
  let decision = scheduler.nextFrame(frame({
    nowMs: 1_000,
    visible: false,
    frameTimeMs: 1_000,
  }));
  assert.equal(decision.phase, "suspended");
  assert.equal(decision.shouldRender, false);
  assert.equal(decision.timing.sampleCount, 0);

  decision = scheduler.nextFrame(frame({ nowMs: 1_016, visible: true }));
  assert.equal(decision.historyReset, true);
  assert.ok(decision.invalidationReasons.includes("visibility-resume"));
});

test("render dimensions honor the M3 Pro pixel cap", () => {
  const scheduler = createStrongFieldQualityScheduler({
    initialTier: 4,
    settleDelayMs: 1,
    refineDelayMs: 2,
  });
  const decision = scheduler.nextFrame(frame({
    viewportWidth: 5120,
    viewportHeight: 2880,
    devicePixelRatio: 2,
    requestedQuality: 1.25,
    timelineRunning: true,
  }));
  assert.ok(
    decision.resolution.pixelCount
      <= M3_PRO_STRONG_FIELD_PROFILE.tiers[2].maxPixels,
  );
  assert.ok(decision.resolution.renderScale < 1);

  const fine = scheduler.nextFrame(frame({
    nowMs: 3,
    viewportWidth: 5120,
    viewportHeight: 2880,
    devicePixelRatio: 2,
    requestedQuality: 1.25,
  }));
  assert.equal(fine.qualityTierId, "fine");
  assert.ok(
    fine.resolution.pixelCount
      <= M3_PRO_STRONG_FIELD_PROFILE.tiers[4].maxPixels,
  );
  assert.equal(fine.resolution.pixelCount, decision.resolution.pixelCount);
  assert.ok(fine.stepBudget > decision.stepBudget);
});

test("unified frame parameters replace only scheduler-owned values", () => {
  const scheduler = createStrongFieldQualityScheduler();
  const decision = scheduler.nextFrame(frame());
  const base = {
    time: 12,
    steps: 999,
    renderScale: 9,
    cameraPos: [1, 2, 3],
  };
  const merged = applyStrongFieldFrameParameters(base, decision);
  assert.equal(merged.time, 12);
  assert.deepEqual(merged.cameraPos, [1, 2, 3]);
  assert.equal(merged.steps, decision.stepBudget);
  assert.equal(merged.renderScale, decision.resolution.renderScale);
  assert.equal(
    merged.strongFieldQuality.historyEpoch,
    decision.historyEpoch,
  );
  assert.equal(Object.isFrozen(merged), true);
});

test("invalid revisions, time, and malformed profiles fail closed", () => {
  const scheduler = createStrongFieldQualityScheduler();
  assert.throws(
    () => scheduler.nextFrame(frame({ cameraRevision: undefined })),
    /cameraRevision/,
  );
  assert.throws(
    () => scheduler.nextFrame(frame({ nowMs: -1 })),
    /nowMs/,
  );
  scheduler.nextFrame(frame({ nowMs: 1 }));
  assert.throws(
    () => scheduler.nextFrame(frame({ nowMs: 0 })),
    /monotonic/,
  );
  assert.throws(
    () => createStrongFieldQualityScheduler({
      hardMinimumFps: 30,
      targetFps: 30,
    }),
    /hardMinimumFps/,
  );
  assert.throws(
    () => createStrongFieldQualityScheduler({
      tiers: [
        {
          id: "high",
          resolutionScale: 1,
          stepBudget: 320,
          maxPixels: 5_000_000,
        },
        {
          id: "low",
          resolutionScale: 0.5,
          stepBudget: 128,
          maxPixels: 1_000_000,
        },
      ],
    }),
    /ordered/,
  );
});
