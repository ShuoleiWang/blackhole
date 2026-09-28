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
the shader. Rays are integrated with classical RK4 whose step is a fraction of
`max(r - w r+, r/2)` for the nearest term. The scheduler's "base steps" above
no longer reach the RK4 tracer (its budgets are in the table below); they only
distinguish render domains for history invalidation.

| Tier | Step fraction | Minimum / maximum step | Energy-drift gate | Maximum RK4 steps | Horizon backstop | Escape / lookback floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `emergency` | 0.60 | 0.02 / 64 M | 0.05 | 48 | 0.02 M | 96 / 260 M |
| `survival` | 0.60 | 0.02 / 64 M | 0.05 | 56 | 0.02 M | 96 / 260 M |
| `interactive` | 0.60 | 0.01 / 64 M | 0.02 | 80 | 0.02 M | 96 / 260 M |
| `balanced` | 0.30 | 0.01 / 64 M | 0.01 | 144 | 0.01 M | 160 / 400 M |
| `fine` | 0.22 | 0.005 / 64 M | 0.005 | 192 | 0.005 M | 200 / 480 M |

The scheduler applies both the tier ceiling and the global twelve-million-pixel
ceiling, and limits device pixel ratio to 2×. A 1280×720 CSS viewport therefore
renders at 2560×1440 on a 2× Retina display; the reported 1836×1376 case renders
at 3672×2752 (about 10.1 MP), without the former 1.43× clamp. The escape
sphere is `max(tier floor, 2 r_camera + 16 M)` and the lookback budget
`max(tier floor, 2 r_escape + 64 M)`; far-field RK4 steps grow geometrically,
so the larger sphere costs only a few steps while keeping the coordinate
direction within `~M b^3 / r^4` of the asymptotic sky direction.

Rays are captured at the innermost photon orbit of the nearest term (radii
scaled by metric weight during the merger blend), which a ray from outside
cannot leave; the horizon padding is only a backstop. A metric failure or an
exhausted budget inside the unscaled photon orbit of any term present in the
metric is also a capture; elsewhere the ray remains `unresolved`. The larger
step fraction of the motion tiers is a latency tradeoff; fractions above 0.6
start flipping capture outcomes, so no tier uses one. Settled `fine` is
deliberately stricter, but no tier upgrades the approximate metric to NR.

#### RK4 accuracy and cost on M3 Pro (2026-09-29)

Sky-direction errors are measured with the production WGSL through the
compute probe of `src/strong-field-gpu-probe.js`, driven by a local harness
that is not part of the repository. The reference is the same tracer at step
fraction 0.04 with a 1,900-step budget; frames are SXS:BBH:0001 at
`t = -9210, -1000, -60, -8, +10 M` on a 192×108 full-frame grid (camera 42 M,
57° inclination, 52° vertical field of view). "Near" rays are those whose
reference path passes within 8 M of a horizon. Frame times are for the
production renderer tracing a full native-Retina 3456×2234 frame at
`t = -1000 M` (offscreen canvas, 6K sky, SDR post).

| Tier | All rays: median / p99 error | Near rays: median / p90 error | Outcome flips per 20,736 rays | Native 3456×2234 frame |
| --- | ---: | ---: | ---: | ---: |
| `interactive` | 0.3–0.4′ / 9–21′ | 2.7–8.0′ / 6.6–22′ | 0–3 | 238 ms |
| `balanced` | 0.2–0.3′ / 0.6–0.7′ | 0.14–0.28′ / 0.22–0.61′ | 0–2 | 485 ms |
| `fine` | 0.2–0.25′ / 0.6–0.7′ | 0.06–0.10′ / 0.09–0.19′ | 0–2 | 673 ms |

A native-Retina pixel at this field of view is about 1.4′, so moving frames
displace lensed features near the holes by several pixels (most near merger,
when the holes are closest), and a paused view visibly settles when it refines
to `balanced`/`fine`. The error falls roughly as the fifth power of the step
fraction (0.5 would cut it by 60% for 20% more frame time, 0.4 by 87% for
47%). Scaling the fraction with local curvature or capping the far-field step
was measured to be less efficient than a uniform fraction.

With the native-resolution lock, the `interactive` tier therefore runs at about
4 FPS on a full-screen Retina panel; the frame time scales linearly with pixel
count (about 31 ns per pixel), so a 1728×1117 raster takes 65 ms. The previous
symplectic-Euler `interactive` tier took 59 ms at that raster while
misplacing directions by 0.55–0.61° (median) and 13–45° (p99), with a
Schwarzschild shadow radius 17% too large.

Against exact Schwarzschild orbit integrals (same harness), the photon-orbit
capture reproduces the shadow edge `b_c = 3 sqrt(3) M` within the 0.1% sampling
bracket in every tier, and at pixel resolution no ray across the Kerr remnant's
shadow edge ends `unresolved`.

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
the 12 MP `interactive` raster and RK4 tier. Once paused, the static
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

#### Controlled local result (2026-08-04, previous symplectic-Euler integrator)

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

#### Controlled dual-disk result (2026-08-08, previous symplectic-Euler integrator)

The independent dual-disk scene was exercised on the local Apple M3 Pro at the
native `2560x1440` internal raster. The browser reported `WebGPU · Metal`,
`apple · metal-3`, and `HDR · P3 · FP16`; both the locked ESO `6000x3000` and
Gaia `16000x8000` textures were observed without resolution substitution. The
binary, transition, and remnant specializations rendered without console or
shader-compilation errors. The paused fine dual-disk view reported about
`3 FPS`; this is an observed UI sample rather than a throughput promise.

The reusable browser probe at
`tests/strong-field-gpu-probe-browser.html?requireAdapter=apple-metal` ran the exact 116-float production
WGSL over a deterministic `64x36` corpus (`2,304` rays) at the same nominal
`2560x1440` raster. The combined disks produced 79 radiative/absorptive hits,
the A-only and B-only runs produced 47 and 32 respectively, and none reported a
disk-transfer failure. A repeated combined run was identical in every decoded
channel. With both weights zero, all 2,304 records had zero disk radiance,
unit transmittance, and zero transfer failure. The ray outcomes were 2,270
escaped and 34 fail-closed unresolved.

The final probe pipeline compilation took `463.1 ms`; its queue-completion
sample was `6.1 ms`. The warm repeated corpus compiled from cache in `1.1 ms`
and completed in `4.7 ms`. These timings cover a small compute/readback corpus,
not a complete displayed frame or present latency.

The visible-realism pass replaced the unit-luminance three-wavelength proxy
with a 15-sample CIE visible response and applied the C² edge/active covering
fraction once. On the same deterministic M3 Pro corpus, maximum scene-linear
disk radiance fell from `16.5708466` to `1.1574888`; maximum luminance was
`0.6080606`, all 79 transfer hits remained present, all 79 retained measurable
chromatic response, and zero records had all RGB channels at or above display
white. There were still zero transfer failures, the repeated readback remained
identical, and FP16 peak headroom exceeded `56,591×`. This is a radiometric
regression gate before the shared HDR/SDR post transform, not a claim of GRMHD
spectral accuracy.

The `requireAdapter=apple-metal` query is an explicit vendor/backend gate: it
fails before allocation unless exposed adapter metadata contains both Apple and
Metal. The compute corpus still ends before the shared post stage; the separate
full application smoke above is what exercised the HDR/P3 `rgba16float` canvas
and native sky uploads. Neither check is silently substituted for the other.

With the RK4 tracer, static-observer camera and Novikov-Thorne mini-disks
(2026-09-29, production `fine` tier) the same 64×36 probe records 94 disk
hits, zero transfer failures, and 2,265 escaped, 39 captured and zero
unresolved rays; maximum scene-linear disk radiance is 1.0096. The probe page
now also compiles every production WGSL module (Schwarzschild trace, post,
progressive accumulation, both strong-field traces and the transfer-map
compositor) and fails on any compilation error.

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
  timelineRunning: timelineAdvancing(), // running, time scale > 0, not held
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
