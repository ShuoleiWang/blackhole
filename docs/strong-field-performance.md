# Strong-field frame scheduling on M3 Pro

`src/strong-field-quality.js` is the policy boundary between the binary
strong-field scene and the renderers. It is a pure state machine: it
does not own DOM state, request animation frames, GPU resources, or physical
state. This keeps quality decisions testable without weakening the physical
invalidation rules.

## Non-negotiable history rule

Temporal accumulation is valid only while all of the following remain fixed:

- camera revision;
- physical-time revision;
- transport/shader revision;
- viewport and requested quality;
- selected resolution/step tier; and
- renderer backend.

A running timeline disables accumulation even if its caller supplies a coarse
or accidentally unchanged physical revision. Dragging also forces a complete
trace every frame. Any revision change, explicit input invalidation, backend
change, device restoration, visibility resume, or render-domain change resets
history before the next submitted frame.

The host must use monotonic primitive revision tokens. Object identity or
rounded visual values are not sufficient for camera or physical time.

## M3 Pro policy

The default policy targets 30 FPS and treats 24 FPS as a hard floor:

| Tier | Resolution multiplier | Pixel ceiling | Base steps | Purpose |
| --- | ---: | ---: | ---: | --- |
| `emergency` | 0.38 | 0.28 MP | 52 | Last-resort HDR/deadline floor below 24 FPS |
| `survival` | 0.50 | 0.52 MP | 64 | Deadline floor after a severe miss |
| `interactive` | 0.65 | 0.92 MP | 96 | Pointer drag and moving timeline |
| `balanced` | 0.82 | 2.0 MP | 160 | First settled refinement |
| `fine` | 1.00 | 5.0 MP | 288 | Paused, strictest settled convergence |

The scene's corresponding numerical policy is explicit rather than hidden in
the shader:

| Tier | Minimum / maximum step | Residual gate | Capture padding | Critical-zone bonus | Escape / lookback floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| `emergency` | 0.065 / 4.40 M | 0.34 | 0.30 M | 268 | 56 / 164 M |
| `survival` | 0.050 / 3.50 M | 0.25 | 0.24 M | 256 | 60 / 180 M |
| `interactive` | 0.035 / 3.00 M | 0.18 | 0.16 M | 224 | 64 / 200 M |
| `balanced` | 0.018 / 1.10 M | 0.10 | 0.08 M | 160 | 80 / 220 M |
| `fine` | 0.010 / 0.58 M | 0.05 | 0.04 M | 64 | 96 / 240 M |

The scheduler applies both the tier ceiling and the global five-million-pixel
ceiling, and limits device pixel ratio to 2×. Interactive Retina rendering
therefore cannot accidentally expand to several million pixels. All base step
budgets stay below the shader's 320-step compile-time maximum; a scene-owned
critical-zone bonus must also clamp the combined budget to that maximum.
The scene uses `max(tier floor, camera radius + 8 M)` for its finite escape
sphere and limits the interactive camera to 34–70 M. This keeps even the
52-step emergency tier able to traverse the central domain instead of turning a
large fraction of ordinary sky rays into false unresolved pixels. Rays that
enter a horizon/photon critical zone receive the tier-specific bonus above,
but the base-plus-bonus total never exceeds 320.

Capture padding is measured outward from an isolated Kerr radius; it is a
declared excision surface, not a computed binary apparent/event horizon. The
reduced coordinate-time tracer also has a failure-only analytic guard inside
the relevant isolated-Kerr photon shell (`0.95 M` for the non-spinning
individual holes, tapering to `0.25 M` for the pinned remnant). It is consulted
only when energy projection fails; outside it the ray remains `unresolved`.
The larger padding, longer steps, and looser residual gates in `emergency`,
`survival`, and `interactive` are latency tradeoffs. Settled `fine` is deliberately stricter,
but no tier upgrades the approximate metric to NR.

These are M3 Pro policy values, not general physics-accuracy claims;
shader-specific acceptance must still prove each declared convergence
boundary.

Completed GPU submission time uses an exponential moving average. Falling
below 24 FPS lowers the performance ceiling immediately. Repeated misses below
the 30 FPS target lower it after a streak. An upgrade requires sustained
headroom above 36 FPS, a cooldown, a paused timeline, a settled camera, WebGPU,
and a visible document. The gap between downgrade and upgrade thresholds
prevents tier oscillation.

In one local M3 Pro run at a 1280×720 CSS viewport, the scheduler selected
`survival` at an internal 961×540 resolution. The completed WebGPU queue-time
EMA was approximately 31.8–33.7 ms, corresponding to roughly 29–31 FPS. This
is one observation, not a guarantee or a cross-scene benchmark.

Startup is deliberately deadline-safe: the initial performance ceiling is
`balanced`, while a moving timeline and pointer interaction both request the
`interactive` tier. The renderer therefore does not put an unprofiled
five-million-pixel fine frame in front of the user's first input. A paused view
can climb from balanced to fine only after eight completed GPU submissions
above the headroom threshold. Eight is below the 32-sample accumulation cap, so
the renderer cannot become `steady` before it has had an opportunity to prove
that the next tier is affordable. A tier change starts a fresh accumulation
epoch.

### Submission backpressure

`requestAnimationFrame` measures display wakeups, not completion of an
asynchronous WebGPU queue. The WebGPU renderer therefore permits exactly one
frame in flight. `queue.onSubmittedWorkDone()` releases the next slot and
records the elapsed wall time from submission through queue completion.

While that slot is busy, the host does not call `scheduler.nextFrame()`. This
is essential: a skipped display wakeup must not consume an accumulation index
or clear a pending camera invalidation. Once the slot is free, the host builds
one frame from the latest camera and physical time. Thus an input can wait at
most for the currently executing frame; old camera views never accumulate in a
GPU queue. WebGL2 has no asynchronous completion signal and retains RAF timing.

## Progressive sequence

After input stops, the scheduler progresses through:

1. `settling`: immediate tier while waiting for input to remain quiet;
2. `refining`: balanced tier with provisional static accumulation;
3. `accumulating`: fine tier, restarting history at the new render domain; and
4. `steady`: stop submitting identical frames after the accumulation cap.

An input or physical change at any point restarts this sequence. WebGL2
fallback is explicitly reported, capped at the interactive tier, and does not
accumulate. A hidden document returns `shouldRender=false` and visibility
resume starts a new history epoch. WebGPU device loss aborts the current
submission, marks the scheduler lost, and reloads through the explicit WebGL2
recovery URL instead of leaving the animation loop dead.

## Host integration

The strong-field scene should own one scheduler instance. On every animation
tick it supplies the previous rendered frame time, viewport, backend, and
revision tokens:

```js
const decision = scheduler.nextFrame({
  nowMs,
  frameTimeMs,
  viewportWidth: innerWidth,
  viewportHeight: innerHeight,
  devicePixelRatio,
  requestedQuality: state.quality,
  cameraRevision,
  physicsRevision,
  transportRevision,
  interactionActive: state.dragging,
  timelineRunning: state.running,
  backend: renderer.capabilities.api,
  visible: !document.hidden,
});
```

DOM input handlers call `scheduler.invalidate("input-kind")`; the next
decision then resets history. WebGPU's `device.lost` callback calls
`signalDeviceLost` before the host navigates to the labelled WebGL2 recovery
path.

`decision.resolution` drives `renderer.resize`. The scene merges
`decision.frameParameters` into its ordinary frame through
`applyStrongFieldFrameParameters`. The strong-field uniform writer consumes
`strongFieldQuality.historyReset`, `accumulationIndex`,
`accumulationWeight`, and `historyEpoch`. WebGPU traces a fresh linear-HDR
sample, blends it into a ping-pong `rgba16float` running average, and only then
applies the shared display/tone-mapping pass. The existing single-hole,
WebGL2 binary preview, and stationary transfer-map bundles remain untouched.

The host submits a frame only when `decision.shouldRender` is true. It must
continue scheduling lightweight animation ticks while refinement or external
input can wake the renderer, even after the decision reaches `steady`.
