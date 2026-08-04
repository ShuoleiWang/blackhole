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

## M3 Pro quality-locked policy

The production policy now treats spatial resolution as an invariant and frame
rate as the variable. A slow frame may reduce completed-frame throughput, but
it cannot select a smaller raster:

| Tier | Resolution multiplier | Pixel ceiling | Base steps | Purpose |
| --- | ---: | ---: | ---: | --- |
| `emergency` | 0.38 | 0.28 MP | 52 | Retained only for explicit fallback/custom profiles |
| `survival` | 0.50 | 0.46 MP | 52 | Retained only for explicit fallback/custom profiles |
| `interactive` | 1.00 | 12.0 MP | 72 | Production motion and dragging floor |
| `balanced` | 1.00 | 12.0 MP | 160 | First settled refinement |
| `fine` | 1.00 | 12.0 MP | 288 | Paused, strictest settled convergence |

The scene's corresponding numerical policy is explicit rather than hidden in
the shader:

| Tier | Minimum / maximum step | Residual gate | Capture padding | Critical-zone bonus | Escape / lookback floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| `emergency` | 0.065 / 3.50 M | 0.34 | 0.30 M | 268 | 56 / 164 M |
| `survival` | 0.050 / 3.50 M | 0.25 | 0.24 M | 256 | 60 / 180 M |
| `interactive` | 0.035 / 3.00 M | 0.18 | 0.16 M | 224 | 64 / 200 M |
| `balanced` | 0.018 / 1.10 M | 0.10 | 0.08 M | 160 | 80 / 220 M |
| `fine` | 0.010 / 0.85 M | 0.05 | 0.04 M | 64 | 80 / 220 M |

The scheduler applies both the tier ceiling and the global twelve-million-pixel
ceiling, and limits device pixel ratio to 2×. A 1280×720 CSS viewport therefore
renders at 2560×1440 on a 2× Retina display; the reported 1836×1376 case renders
at 3672×2752 (about 10.1 MP), without the former 1.43× clamp. All base step
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

Completed ray-trace submission time still uses an exponential moving average,
but it is telemetry rather than authority to downsample. Resource uploads enter
a separate queue scope and cannot be counted as rendered frames. Both the hard
and sustained-miss paths clamp at the native-resolution `interactive` tier;
even a 250 ms frame cannot select `survival` or `emergency`. Timing is reset at
each tier boundary, and the first completion after a raster/tier switch is
excluded because it includes allocation and resize work rather than steady
tracing cost.

Startup begins at `balanced`; a moving M3 Pro timeline and active dragging use
the 12 MP `interactive` raster with 72 base steps. Once paused, the static
controller starts at `balanced`, then enters `fine` at the same spatial raster.
Accumulation starts with an explicit
unjittered sample zero; the first jittered sample is index one with weight one
half. A tier change starts a fresh accumulation epoch.

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

### M3 Pro production trace-path optimization

The WebGPU path applies two optimizations without lowering any numerical or
spatial quality budget:

- moving or continuously playing frames go directly from the linear-HDR trace
  target to the existing post pass; they do not copy a sample that is forbidden
  from becoming temporal history into an otherwise unused accumulation target;
- three Metal pipelines specialize the frame-uniform spacetime phase as binary,
  transition, or remnant, allowing the compiler to remove inactive
  Kerr-Schild providers instead of retaining the worst-case provider graph.

Two more aggressive algebraic candidates -- replacing production `pow` and
fusing inverse-metric derivative contractions -- were rejected after the GPU
readback exposed critical-ray drift. They are deliberately not part of the
production diff even though their scalar identities hold in exact arithmetic.

Static sample zero and all later paused samples still use the original
`rgba16float` history path. Spatial resolution, sky texture dimensions,
integration tiers, escape/capture rules, HDR/P3 output, and the WebGL2 fallback
are not changed by these optimizations.

The repository also contains a production-WGSL GPU readback harness. It appends
a compute entry point to the exact fragment-tracer module and records outcome,
termination reason, escape direction, frequency shift, lookback, maximum null
Hamiltonian residual, iteration count, and minimum horizon distance. This is
the acceptance boundary for future algebra, pipeline, or native-backend work;
a shader-string test or visually similar screenshot is not sufficient.

#### Controlled local result (2026-08-04)

One local Apple M3 Pro A/B used the same 1280x720 CSS viewport at DPR 2
(`2560x1440` internal raster), ESO 6K sky, SDR output, binary protocol time
`-965.30 M`, a running timeline frozen at `0 M/s`, and the unchanged 72-step
interactive tier. Against clean commit `7845101`, completed WebGPU submission
EMA changed from `250.00 ms` (`4.00 FPS`) to `97.62 ms` (`10.24 FPS`): 61.0%
less queue-completion time and 2.56x throughput. This telemetry is
submit-to-queue-completion time, not a claim about end-to-end display-present
latency.

An independent M3 Pro readback compared 2,405 deterministic rays in each of
the binary, transition, remnant, and forced budget-exhaustion cases (9,620
total) against the clean baseline. It covered all three raw outcomes. All
7,084 escaped rays kept their classification; maximum escape-direction drift
was `1.06e-5`, maximum frequency-shift drift was `2.38e-7`, and no escaped ray
became shadow or unresolved. Transition output was identical in every recorded
channel, and remnant outcome/termination output was identical.

The strict comparator intentionally reports two separatrix classifications
rather than hiding them: one of 78 production binary captured probes became
`unresolved` after exhausting the 296-step critical budget only `0.00255 M`
outside the declared capture padding; one deliberately under-budget probe
moved from `unresolved` to captured. Captured/non-sky rays also showed expected
floating-point path-length and iteration drift after Metal specialization.
Both boundary changes remain fail-closed with respect to sky -- no unresolved
ray is painted as escaped sky -- but they are not claimed as bitwise
equivalence. The paused fine tier reached all 32 accumulation samples; two
full-page captures two seconds apart were byte identical after `steady`.

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
