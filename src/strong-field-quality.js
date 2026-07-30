/*
 * M3 Pro quality scheduler for a future real-time strong-field renderer.
 *
 * This module is deliberately independent of the DOM and WebGPU.  The host
 * supplies monotonic camera/physics revisions and timing samples; the returned
 * decision is the only quality state a renderer needs for the next frame.
 *
 * Safety invariant: temporal accumulation is permitted only when the camera,
 * physical state, transport parameters, backend, viewport, and quality tier
 * are all unchanged and the timeline is paused.
 */

const BASE_TIERS = Object.freeze([
  Object.freeze({
    id: "emergency",
    resolutionScale: 0.38,
    stepBudget: 52,
    maxPixels: 280_000,
  }),
  Object.freeze({
    id: "survival",
    resolutionScale: 0.50,
    stepBudget: 64,
    maxPixels: 520_000,
  }),
  Object.freeze({
    id: "interactive",
    resolutionScale: 0.65,
    stepBudget: 96,
    maxPixels: 921_600,
  }),
  Object.freeze({
    id: "balanced",
    resolutionScale: 0.82,
    stepBudget: 160,
    maxPixels: 2_000_000,
  }),
  Object.freeze({
    id: "fine",
    resolutionScale: 1.00,
    stepBudget: 288,
    maxPixels: 5_000_000,
  }),
]);

export const M3_PRO_STRONG_FIELD_PROFILE = Object.freeze({
  targetFps: 30,
  hardMinimumFps: 24,
  upgradeFps: 36,
  emaAlpha: 0.12,
  targetMissFrames: 6,
  upgradeFrames: 8,
  qualityCooldownMs: 600,
  settleDelayMs: 180,
  refineDelayMs: 720,
  maxAccumulationSamples: 32,
  maxDevicePixelRatio: 2,
  maxRenderPixels: 5_000_000,
  // Do not make the first Metal submission an unprofiled fine-quality frame.
  // The scheduler starts with a balanced performance ceiling, traces moving
  // frames at the interactive tier, and unlocks fine quality only after real
  // completed-queue timings demonstrate sustained headroom.
  initialTier: 3,
  interactionTier: 2,
  realtimeTier: 2,
  fallbackMaxTier: 2,
  tiers: BASE_TIERS,
});

const VALID_BACKENDS = new Set(["webgpu", "webgl2", "none"]);

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function positiveFinite(value, name) {
  if (!Number.isFinite(value) || value <= 0) {
    throw new TypeError(`${name} must be a positive finite number`);
  }
  return value;
}

function nonNegativeFinite(value, name) {
  if (!Number.isFinite(value) || value < 0) {
    throw new TypeError(`${name} must be a non-negative finite number`);
  }
  return value;
}

function positiveInteger(value, name) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new TypeError(`${name} must be a positive integer`);
  }
  return value;
}

function tierIndex(value, tiers, name) {
  if (!Number.isInteger(value) || value < 0 || value >= tiers.length) {
    throw new RangeError(`${name} must identify a configured quality tier`);
  }
  return value;
}

function validateRevision(value, name) {
  const type = typeof value;
  if (
    value === undefined
    || (type === "number" && !Number.isFinite(value))
    || !["string", "number", "bigint", "boolean"].includes(type)
  ) {
    throw new TypeError(
      `${name} must be an explicit finite primitive revision token`,
    );
  }
  return value;
}

function makeProfile(overrides = {}) {
  if (!overrides || typeof overrides !== "object" || Array.isArray(overrides)) {
    throw new TypeError("quality scheduler options must be an object");
  }
  const tierSource = overrides.tiers ?? M3_PRO_STRONG_FIELD_PROFILE.tiers;
  if (!Array.isArray(tierSource) || tierSource.length < 2) {
    throw new TypeError("tiers must contain at least two quality levels");
  }
  const tiers = [];
  tierSource.forEach((tier, index) => {
    if (!tier || typeof tier !== "object" || Array.isArray(tier)) {
      throw new TypeError(`tiers[${index}] must be an object`);
    }
    const normalized = {
      id: String(tier.id || ""),
      resolutionScale: positiveFinite(
        tier.resolutionScale,
        `tiers[${index}].resolutionScale`,
      ),
      stepBudget: positiveInteger(
        tier.stepBudget,
        `tiers[${index}].stepBudget`,
      ),
      maxPixels: positiveInteger(
        tier.maxPixels,
        `tiers[${index}].maxPixels`,
      ),
    };
    if (!normalized.id) {
      throw new TypeError(`tiers[${index}].id must be non-empty`);
    }
    if (
      index > 0
      && (
        normalized.resolutionScale < tiers[index - 1].resolutionScale
        || normalized.stepBudget < tiers[index - 1].stepBudget
        || normalized.maxPixels < tiers[index - 1].maxPixels
      )
    ) {
      throw new RangeError("quality tiers must be ordered from cheapest to finest");
    }
    tiers.push(Object.freeze(normalized));
  });

  const profile = {
    ...M3_PRO_STRONG_FIELD_PROFILE,
    ...overrides,
    tiers: Object.freeze(tiers),
  };
  positiveFinite(profile.targetFps, "targetFps");
  positiveFinite(profile.hardMinimumFps, "hardMinimumFps");
  positiveFinite(profile.upgradeFps, "upgradeFps");
  if (profile.hardMinimumFps >= profile.targetFps) {
    throw new RangeError("hardMinimumFps must be lower than targetFps");
  }
  if (profile.upgradeFps <= profile.targetFps) {
    throw new RangeError("upgradeFps must be higher than targetFps");
  }
  if (
    !Number.isFinite(profile.emaAlpha)
    || profile.emaAlpha <= 0
    || profile.emaAlpha > 1
  ) {
    throw new RangeError("emaAlpha must be in (0, 1]");
  }
  positiveInteger(profile.targetMissFrames, "targetMissFrames");
  positiveInteger(profile.upgradeFrames, "upgradeFrames");
  nonNegativeFinite(profile.qualityCooldownMs, "qualityCooldownMs");
  nonNegativeFinite(profile.settleDelayMs, "settleDelayMs");
  nonNegativeFinite(profile.refineDelayMs, "refineDelayMs");
  if (profile.refineDelayMs <= profile.settleDelayMs) {
    throw new RangeError("refineDelayMs must be greater than settleDelayMs");
  }
  positiveInteger(profile.maxAccumulationSamples, "maxAccumulationSamples");
  positiveFinite(profile.maxDevicePixelRatio, "maxDevicePixelRatio");
  positiveInteger(profile.maxRenderPixels, "maxRenderPixels");
  for (const name of [
    "initialTier",
    "interactionTier",
    "realtimeTier",
    "fallbackMaxTier",
  ]) {
    profile[name] = tierIndex(profile[name], tiers, name);
  }
  return Object.freeze(profile);
}

function renderDimensions(profile, tier, input) {
  const cssWidth = positiveFinite(input.viewportWidth, "viewportWidth");
  const cssHeight = positiveFinite(input.viewportHeight, "viewportHeight");
  const devicePixelRatio = positiveFinite(
    input.devicePixelRatio ?? 1,
    "devicePixelRatio",
  );
  const requestedQuality = positiveFinite(
    input.requestedQuality ?? 1,
    "requestedQuality",
  );
  let renderScale = (
    Math.min(devicePixelRatio, profile.maxDevicePixelRatio)
    * requestedQuality
    * tier.resolutionScale
  );
  const requestedPixels = cssWidth * cssHeight * renderScale * renderScale;
  const pixelLimit = Math.min(profile.maxRenderPixels, tier.maxPixels);
  if (requestedPixels > pixelLimit) {
    renderScale *= Math.sqrt(pixelLimit / requestedPixels);
  }
  const width = Math.max(1, Math.floor(cssWidth * renderScale));
  const height = Math.max(1, Math.floor(cssHeight * renderScale));
  return Object.freeze({
    width,
    height,
    renderScale,
    pixelCount: width * height,
  });
}

function freezeDecision(decision) {
  Object.freeze(decision.resolution);
  Object.freeze(decision.timing);
  Object.freeze(decision.backend);
  Object.freeze(decision.invalidationReasons);
  Object.freeze(decision.frameParameters);
  return Object.freeze(decision);
}

class StrongFieldQualityScheduler {
  constructor(options) {
    this.profile = makeProfile(options);
    this.performanceTier = this.profile.initialTier;
    this.selectedTier = null;
    this.backendName = "webgpu";
    this.backendReason = null;
    this.deviceState = "ready";
    this.deviceReason = null;
    this.visible = true;
    this.pendingReasons = new Set(["initial"]);
    this.pendingInput = false;
    this.historyEpoch = 0;
    this.accumulationSamples = 0;
    this.frameTimeEmaMs = null;
    this.fpsEma = null;
    this.timingSamples = 0;
    this.targetMissStreak = 0;
    this.upgradeStreak = 0;
    this.lastTierChangeMs = -Infinity;
    this.lastNowMs = null;
    this.stableSinceMs = null;
    this.lastResolutionKey = null;
    this.hasRevisions = false;
    this.cameraRevision = null;
    this.physicsRevision = null;
    this.transportRevision = null;
    this.lastDecision = null;
  }

  invalidate(reason = "external") {
    const label = String(reason || "external");
    this.pendingReasons.add(label);
    this.pendingInput = true;
  }

  signalDeviceLost(reason = "unknown") {
    this.deviceState = "lost";
    this.deviceReason = String(reason);
    this.accumulationSamples = 0;
    this.pendingReasons.add("device-lost");
  }

  signalDeviceRestored({ backend = "webgpu", reason = "device-restored" } = {}) {
    this.deviceState = "ready";
    this.deviceReason = null;
    this.setBackend(backend, reason);
    this.resetTiming();
    this.performanceTier = Math.min(
      this.profile.initialTier,
      backend === "webgl2"
        ? this.profile.fallbackMaxTier
        : this.profile.tiers.length - 1,
    );
    this.selectedTier = null;
    this.pendingReasons.add(String(reason));
    this.pendingInput = true;
  }

  setBackend(backend, reason = "backend-change") {
    if (!VALID_BACKENDS.has(backend)) {
      throw new RangeError(`unsupported renderer backend ${JSON.stringify(backend)}`);
    }
    if (backend === this.backendName) {
      return;
    }
    this.backendName = backend;
    this.backendReason = String(reason);
    this.selectedTier = null;
    this.accumulationSamples = 0;
    this.pendingReasons.add("backend-change");
    this.pendingInput = true;
    if (backend === "webgl2") {
      this.performanceTier = Math.min(
        this.performanceTier,
        this.profile.fallbackMaxTier,
      );
    }
  }

  resetTiming() {
    this.frameTimeEmaMs = null;
    this.fpsEma = null;
    this.timingSamples = 0;
    this.targetMissStreak = 0;
    this.upgradeStreak = 0;
  }

  updateTiming(frameTimeMs, nowMs, allowUpgrade) {
    if (frameTimeMs === undefined || frameTimeMs === null) {
      return;
    }
    positiveFinite(frameTimeMs, "frameTimeMs");
    // A background-tab pause is not a useful GPU sample.  The host should
    // normally omit it; the clamp is a final guard against poisoning the EMA.
    const sample = Math.min(frameTimeMs, 250);
    this.frameTimeEmaMs = this.frameTimeEmaMs === null
      ? sample
      : (
        this.profile.emaAlpha * sample
        + (1 - this.profile.emaAlpha) * this.frameTimeEmaMs
      );
    this.fpsEma = 1_000 / this.frameTimeEmaMs;
    this.timingSamples += 1;

    if (this.fpsEma < this.profile.hardMinimumFps) {
      const severe = this.fpsEma < this.profile.hardMinimumFps * 0.70;
      const decrement = severe ? 2 : 1;
      this.changePerformanceTier(
        Math.max(0, this.performanceTier - decrement),
        nowMs,
        "hard-fps-gate",
      );
      this.targetMissStreak = 0;
      this.upgradeStreak = 0;
      return;
    }

    if (this.fpsEma < this.profile.targetFps) {
      this.targetMissStreak += 1;
    } else {
      this.targetMissStreak = 0;
    }
    if (
      this.targetMissStreak >= this.profile.targetMissFrames
      && nowMs - this.lastTierChangeMs >= this.profile.qualityCooldownMs
    ) {
      this.changePerformanceTier(
        Math.max(0, this.performanceTier - 1),
        nowMs,
        "target-fps-miss",
      );
      this.targetMissStreak = 0;
      this.upgradeStreak = 0;
      return;
    }

    if (allowUpgrade && this.fpsEma >= this.profile.upgradeFps) {
      this.upgradeStreak += 1;
    } else {
      this.upgradeStreak = 0;
    }
    if (
      this.upgradeStreak >= this.profile.upgradeFrames
      && nowMs - this.lastTierChangeMs >= this.profile.qualityCooldownMs
    ) {
      this.changePerformanceTier(
        Math.min(
          this.profile.tiers.length - 1,
          this.performanceTier + 1,
        ),
        nowMs,
        "sustained-headroom",
      );
      this.targetMissStreak = 0;
      this.upgradeStreak = 0;
    }
  }

  changePerformanceTier(nextTier, nowMs, reason) {
    if (nextTier === this.performanceTier) {
      return;
    }
    this.performanceTier = nextTier;
    this.lastTierChangeMs = nowMs;
    this.pendingReasons.add(reason);
  }

  checkRevisions(input, nowMs) {
    const camera = validateRevision(input.cameraRevision, "cameraRevision");
    const physics = validateRevision(input.physicsRevision, "physicsRevision");
    const transport = validateRevision(
      input.transportRevision ?? 0,
      "transportRevision",
    );
    const hadRevisions = this.hasRevisions;
    if (!hadRevisions) {
      this.hasRevisions = true;
    } else {
      if (!Object.is(camera, this.cameraRevision)) {
        this.pendingReasons.add("camera-change");
      }
      if (!Object.is(physics, this.physicsRevision)) {
        this.pendingReasons.add("physics-change");
      }
      if (!Object.is(transport, this.transportRevision)) {
        this.pendingReasons.add("transport-change");
      }
    }
    const changed = hadRevisions && (
      !Object.is(camera, this.cameraRevision)
      || !Object.is(physics, this.physicsRevision)
      || !Object.is(transport, this.transportRevision)
    );
    this.cameraRevision = camera;
    this.physicsRevision = physics;
    this.transportRevision = transport;
    if (changed && this.stableSinceMs !== null) {
      this.stableSinceMs = nowMs;
    }
    return changed;
  }

  nextFrame(input) {
    if (!input || typeof input !== "object" || Array.isArray(input)) {
      throw new TypeError("nextFrame input must be an object");
    }
    const nowMs = nonNegativeFinite(input.nowMs, "nowMs");
    if (this.lastNowMs !== null && nowMs < this.lastNowMs) {
      throw new RangeError("nowMs must be monotonic");
    }
    this.lastNowMs = nowMs;

    if (input.backend !== undefined) {
      this.setBackend(input.backend, input.backendReason);
    }
    const nextVisible = input.visible !== false;
    if (nextVisible !== this.visible) {
      this.visible = nextVisible;
      if (nextVisible) {
        this.pendingReasons.add("visibility-resume");
        this.pendingInput = true;
      } else {
        this.accumulationSamples = 0;
      }
    }

    const revisionsChanged = this.checkRevisions(input, nowMs);
    const interactionActive = input.interactionActive === true;
    const timelineRunning = input.timelineRunning === true;
    if (this.stableSinceMs === null) {
      this.stableSinceMs = nowMs;
    }
    if (this.pendingInput || interactionActive || revisionsChanged) {
      this.stableSinceMs = nowMs;
      this.pendingInput = false;
    }
    if (interactionActive) {
      this.pendingReasons.add("interaction");
    }
    if (timelineRunning) {
      // A running timeline is dynamic even if a caller accidentally reuses a
      // coarse physics revision.  Never blend distinct physical times.
      this.pendingReasons.add("timeline-running");
    }

    const fullyStatic = (
      !interactionActive
      && !timelineRunning
      && !revisionsChanged
    );
    const stableElapsedMs = Math.max(0, nowMs - this.stableSinceMs);
    const allowUpgrade = (
      fullyStatic
      && stableElapsedMs >= this.profile.settleDelayMs
      && this.deviceState === "ready"
      && this.backendName === "webgpu"
      && this.visible
    );
    if (this.visible && this.deviceState === "ready") {
      this.updateTiming(input.frameTimeMs, nowMs, allowUpgrade);
    }

    let convergencePhase;
    let desiredTier;
    if (interactionActive) {
      convergencePhase = "interactive";
      desiredTier = this.profile.interactionTier;
    } else if (timelineRunning || revisionsChanged) {
      convergencePhase = "realtime";
      desiredTier = this.profile.realtimeTier;
    } else if (stableElapsedMs < this.profile.settleDelayMs) {
      convergencePhase = "settling";
      desiredTier = this.profile.interactionTier;
    } else if (stableElapsedMs < this.profile.refineDelayMs) {
      convergencePhase = "refining";
      desiredTier = this.profile.realtimeTier;
    } else {
      convergencePhase = "accumulating";
      desiredTier = this.profile.tiers.length - 1;
    }

    const backendCeiling = this.backendName === "webgl2"
      ? this.profile.fallbackMaxTier
      : this.profile.tiers.length - 1;
    const selectedTier = Math.min(
      desiredTier,
      this.performanceTier,
      backendCeiling,
    );
    if (selectedTier !== this.selectedTier) {
      this.selectedTier = selectedTier;
      this.pendingReasons.add("quality-tier-change");
    }
    const tier = this.profile.tiers[selectedTier];
    const resolution = renderDimensions(this.profile, tier, input);
    const resolutionKey = [
      resolution.width,
      resolution.height,
      tier.stepBudget,
    ].join("/");
    if (
      this.lastResolutionKey !== null
      && resolutionKey !== this.lastResolutionKey
    ) {
      this.pendingReasons.add("render-domain-change");
    }
    this.lastResolutionKey = resolutionKey;

    const backendAvailable = this.backendName !== "none";
    const canRender = (
      this.visible
      && this.deviceState === "ready"
      && backendAvailable
    );
    const dynamic = interactionActive || timelineRunning || revisionsChanged;
    const invalidationReasons = [...this.pendingReasons].sort();
    const historyReset = canRender && (
      dynamic
      || invalidationReasons.length > 0
    );
    if (historyReset) {
      this.historyEpoch += 1;
      this.accumulationSamples = 0;
    }

    const accumulationAllowed = (
      canRender
      && fullyStatic
      && stableElapsedMs >= this.profile.settleDelayMs
      && this.backendName === "webgpu"
    );
    let accumulationIndex = 0;
    let accumulationWeight = 1;
    let shouldRender = false;
    if (canRender && (historyReset || dynamic)) {
      shouldRender = true;
      if (accumulationAllowed) {
        this.accumulationSamples = 1;
      }
    } else if (
      accumulationAllowed
      && this.accumulationSamples < this.profile.maxAccumulationSamples
    ) {
      accumulationIndex = this.accumulationSamples;
      accumulationWeight = 1 / (accumulationIndex + 1);
      this.accumulationSamples += 1;
      shouldRender = true;
    }

    if (
      convergencePhase === "accumulating"
      && accumulationAllowed
      && !shouldRender
    ) {
      convergencePhase = "steady";
    }

    let phase = convergencePhase;
    if (!this.visible) {
      phase = "suspended";
    } else if (this.deviceState === "lost") {
      phase = "device-lost";
    } else if (!backendAvailable) {
      phase = "backend-unavailable";
    } else if (this.backendName === "webgl2") {
      phase = "fallback";
    }

    const flags = (
      (historyReset ? 1 : 0)
      | (accumulationAllowed ? 2 : 0)
      | (timelineRunning ? 4 : 0)
      | (interactionActive ? 8 : 0)
      | (this.backendName === "webgl2" ? 16 : 0)
    );
    const frameParameters = {
      renderScale: resolution.renderScale,
      steps: tier.stepBudget,
      strongFieldQuality: Object.freeze({
        tier: selectedTier,
        tierId: tier.id,
        phase,
        convergencePhase,
        historyEpoch: this.historyEpoch,
        historyReset,
        accumulationIndex,
        accumulationWeight,
        flags,
      }),
    };
    const decision = freezeDecision({
      shouldRender,
      phase,
      convergencePhase,
      qualityTier: selectedTier,
      qualityTierId: tier.id,
      performanceTier: this.performanceTier,
      stepBudget: tier.stepBudget,
      historyReset,
      historyEpoch: this.historyEpoch,
      accumulationAllowed,
      accumulationIndex,
      accumulationWeight,
      stableElapsedMs,
      invalidationReasons,
      resolution: { ...resolution },
      timing: {
        frameTimeEmaMs: this.frameTimeEmaMs,
        fpsEma: this.fpsEma,
        targetFps: this.profile.targetFps,
        hardMinimumFps: this.profile.hardMinimumFps,
        deadlineMs: 1_000 / this.profile.targetFps,
        hardDeadlineMs: 1_000 / this.profile.hardMinimumFps,
        sampleCount: this.timingSamples,
      },
      backend: {
        name: this.backendName,
        reason: this.backendReason,
        deviceState: this.deviceState,
        deviceReason: this.deviceReason,
        fallback: this.backendName === "webgl2",
      },
      frameParameters,
    });
    this.pendingReasons.clear();
    this.lastDecision = decision;
    return decision;
  }

  snapshot() {
    return Object.freeze({
      performanceTier: this.performanceTier,
      selectedTier: this.selectedTier,
      historyEpoch: this.historyEpoch,
      accumulationSamples: this.accumulationSamples,
      pendingInvalidation: this.pendingReasons.size > 0,
      frameTimeEmaMs: this.frameTimeEmaMs,
      fpsEma: this.fpsEma,
      backend: this.backendName,
      deviceState: this.deviceState,
      lastDecision: this.lastDecision,
    });
  }
}

export function createStrongFieldQualityScheduler(options = {}) {
  return new StrongFieldQualityScheduler(options);
}

export function applyStrongFieldFrameParameters(baseFrame, decision) {
  if (!baseFrame || typeof baseFrame !== "object" || Array.isArray(baseFrame)) {
    throw new TypeError("baseFrame must be an object");
  }
  if (
    !decision?.frameParameters
    || !decision.frameParameters.strongFieldQuality
  ) {
    throw new TypeError("decision must come from the strong-field scheduler");
  }
  return Object.freeze({
    ...baseFrame,
    ...decision.frameParameters,
  });
}
