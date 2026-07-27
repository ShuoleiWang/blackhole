import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  createDynamicsTrack,
} from "../src/scenes/binary-dynamics-adapter.js";
import {
  createPlaybackClock,
  playbackRateFactor,
} from "../src/scenes/binary-playback-clock.js";

const root = new URL("../", import.meta.url);
const manifest = JSON.parse(
  await readFile(
    new URL("assets/scenes/binary-sxs-bbh-0001-v2.json", root),
    "utf8",
  ),
);
const payload = JSON.parse(
  await readFile(
    new URL("assets/scenes/binary-sxs-bbh-0001-v2.samples.json", root),
    "utf8",
  ),
);

function clock() {
  return createPlaybackClock({
    firstTimeM: manifest.dynamics.firstTimeM,
    finalTimeM: manifest.dynamics.finalTimeM,
    endHoldSeconds: manifest.playback.endHoldSeconds,
    loop: manifest.playback.loop,
    slowMotion: manifest.playback.slowMotion,
  });
}

function mutatedManifest(mutator) {
  const copy = structuredClone(manifest);
  mutator(copy);
  return copy;
}

test("SXS dynamics track preserves its source anchors", () => {
  const track = createDynamicsTrack(manifest, payload);
  assert.equal(track.sampleCount, manifest.dynamics.asset.sampleCount);
  assert.equal(track.sampleAt(track.firstTimeM).tM, track.firstTimeM);
  assert.equal(track.sampleAt(track.finalTimeM).tM, track.finalTimeM);

  const common = track.sampleAt(
    manifest.events.commonApparentHorizonFirst.tProtocolM,
  );
  assert.equal(common.regime, "nr-merger");
  assert.ok(common.renderTopologyBlend <= 1e-9);

  const peak = track.sampleAt(0);
  assert.equal(peak.regime, "nr-ringdown");
  assert.ok(Math.abs(
    peak.waveform.amplitude
      - manifest.timeReference.waveformPeakAmplitude,
  ) < 2e-9);
  assert.ok(peak.renderTopologyBlend >= 1 - 1e-9);
});

test("runtime manifest validation rejects unsafe renderer and source mutations", () => {
  const invalidManifests = [
    mutatedManifest((value) => {
      value.source.artifacts.waveform.sha256 = "0".repeat(64);
    }),
    mutatedManifest((value) => {
      value.physicalSystem.bodies[0].massFraction = -4;
      value.physicalSystem.bodies[1].massFraction = 9;
    }),
    mutatedManifest((value) => {
      value.playback.cycleDurationSecondsAtNominalRate = 0;
    }),
    mutatedManifest((value) => {
      value.rendererDefaults.observerRadiusM = -1;
    }),
    mutatedManifest((value) => {
      value.rendererDefaults.raySteps = -5;
    }),
    mutatedManifest((value) => {
      value.rendererAdapter.stateAbi[0] = "untrustedState";
    }),
  ];

  for (const invalidManifest of invalidManifests) {
    assert.throws(
      () => createDynamicsTrack(invalidManifest, payload),
      /Binary dynamics contract violation/,
    );
  }
});

test("slow motion changes only the presentation rate", () => {
  const slow = manifest.playback.slowMotion;
  assert.equal(playbackRateFactor(slow.startTimeM - 1, slow, true), 1);
  assert.equal(
    playbackRateFactor(slow.startTimeM, slow, true),
    slow.rateMultiplier,
  );
  assert.equal(
    playbackRateFactor(slow.endTimeM - 1e-6, slow, true),
    slow.rateMultiplier,
  );
  assert.equal(playbackRateFactor(slow.endTimeM, slow, true), 1);
  assert.equal(playbackRateFactor(0, slow, false), 1);
});

function simulateAtFps(fps) {
  const playback = clock();
  let timeM = playback.seek(-170);
  for (let frame = 0; frame < fps; frame += 1) {
    timeM = playback.advance(timeM, 1 / fps, 100, true).timeM;
  }
  return timeM;
}

test("segmented playback is frame-rate independent across slow-zone entry", () => {
  const at30 = simulateAtFps(30);
  const at60 = simulateAtFps(60);
  const at120 = simulateAtFps(120);
  assert.ok(Math.abs(at30 - at60) < 1e-8);
  assert.ok(Math.abs(at60 - at120) < 1e-8);
  assert.ok(Math.abs(at120 - (-149.2)) < 1e-8);
});

test("slow-zone boundaries have no floating-point dead zone", () => {
  const playback = createPlaybackClock({
    firstTimeM: -4,
    finalTimeM: 4,
    endHoldSeconds: 0,
    loop: false,
    slowMotion: {
      startTimeM: -2,
      endTimeM: 2,
      rateMultiplier: 0.1,
    },
  });

  let result = playback.advance(
    playback.seek(-2 - 5e-10),
    1,
    1,
    true,
  );
  assert.ok(Math.abs(result.timeM - (-1.90000000005)) < 1e-12);

  result = playback.advance(
    playback.seek(2 - 5e-10),
    1,
    1,
    true,
  );
  assert.ok(Math.abs(result.timeM - 2.999999995) < 1e-12);
});

test("seek clamps and resets end hold before deterministic looping", () => {
  const playback = clock();
  const final = manifest.dynamics.finalTimeM;
  assert.equal(playback.seek(final + 100), final);
  let result = playback.advance(
    final,
    manifest.playback.endHoldSeconds - 0.1,
    100,
    false,
  );
  assert.equal(result.timeM, final);
  assert.equal(result.holding, true);
  assert.equal(result.effectiveRateMPerSecond, 0);
  result = playback.advance(result.timeM, 0.2, 100, false);
  assert.equal(result.looped, true);
  assert.ok(result.timeM > manifest.dynamics.firstTimeM);
});

test("track interpolation stays finite at representative source events", () => {
  const track = createDynamicsTrack(manifest, payload);
  for (const event of Object.values(manifest.events)) {
    const sample = track.sampleAt(event.tProtocolM);
    for (const value of [
      sample.tM,
      sample.separationM,
      sample.orbitalPhaseRad,
      sample.renderTopologyBlend,
      sample.waveform.h22Real,
      sample.waveform.h22Imag,
      sample.waveform.amplitude,
    ]) {
      assert.ok(Number.isFinite(value));
    }
  }
});
