# Relativistic Black Hole Renderer

**English** | [简体中文](./README.zh-CN.md)

An interactive, real-time black-hole renderer built with **WebGPU and WebGL2**.

The default scene numerically integrates past-directed null geodesics in a Schwarzschild spacetime. The same ray path determines capture by the event horizon, intersections with an idealized accretion disk, relativistic frequency shifts, and gravitational lensing of an all-sky Milky Way background.

An isolated, opt-in `?scene=binary-approx` experiment adds an equal-mass,
effectively non-spinning binary-black-hole preview. **Phase 2 now drives the
motion and waveform readout from pinned `SXS:BBH:0001` Lev5 numerical-
relativity data**: the A/B apparent-horizon coordinate centroids provide
separation and orbital phase, and the CoM-corrected
`Extrapolated_N2.dir/Y_l2_m2.dat` mode provides the complex `h22` strip.

That upgrade applies to the **dynamics only**. The image still comes from the
unchanged, frame-frozen, multi-centre **weak-field fast-light shader**. No SXS
near-zone metric or ray-transfer product is consumed, and the result is
**not NR ray tracing**, not a solved binary-spacetime image, and not a
quantitatively accurate merger shadow. The project is intended for real-time
visualization and education. Its real-time scenes are not Kerr or full-NR
light-propagation solvers, and the project is not a GRMHD or high-precision
radiative-transfer solver. The optional stationary Kerr reference described
below is deliberately isolated from that real-time binary renderer.

A second opt-in path, `?scene=transfer-map-reference`, exercises the
transfer-map pipeline with project-generated stationary analytic
**Schwarzschild and Kerr** references. The Kerr product uses the pinned
`SXS:BBH:0001` remnant spin only; its metric and pixels are analytic,
project-generated data, not SXS near-zone data. Both fixed 1024×576 cameras
contain no accretion disk and are **not numerical relativity**. They validate
offline ray generation, authenticated playback, GPU consumption, diagnostics,
and sky composition without changing either existing scene.

![A Schwarzschild black hole, accretion disk, and gravitationally lensed Milky Way](./docs/images/blackhole-galaxy-hero.webp)

<sub>A 5120×2576 in-app screenshot of the WebGPU/Metal renderer running on Apple Silicon, with the controls and live backend, output, and performance readouts visible. Milky Way source: ESO/S. Brunier; geodesically transformed, composited, and transcoded by this project from an original used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [`assets/SOURCES.md`](./assets/SOURCES.md) for full provenance.</sub>

## Key features

- **Per-pixel null-geodesic integration** — Uses Störmer–Verlet integration of `u'' = -u + 3u²` instead of a screen-space distortion effect.
- **Unified ray-path composition** — A single traced ray handles capture, multiple disk-plane intersections, and the final sky escape direction, producing critical-curve arcs and higher-order images.
- **Relativistic disk appearance** — Includes frequency shifts from Schwarzschild circular motion, the bolometric intensity transfer factor `g⁴`, approximate blackbody chromaticity, surface optical depth, and limb darkening.
- **Real-time procedural disk structure** — Turbulence-inspired, finite-lifetime noise is advected at the local Keplerian angular velocity. This is a visual approximation, not an MHD simulation.
- **SXS-driven binary dynamics** — `?scene=binary-approx` lazy-loads a
  2,732-sample, approximately 198 KiB track derived from
  `SXS:BBH:0001/Lev5`: real A/B horizon-centroid coordinate separation and
  phase, CoM-corrected extrapolated `h22`, source events, and exact remnant
  metadata.
- **Interactive binary transport** — The waveform timeline can be scrubbed,
  paused from either transport control, and replayed with an optional
  presentation-only `0.12×` slow-motion window around merger. Slow motion
  changes wall-clock playback only, never the source time or physics data.
- **Explicit rendering boundary** — The binary scene retains the existing
  WGSL/GLSL weak-field fast-light lens shader. NR-derived trajectories and a
  real NR waveform do not make the pixels NR ray tracing.
- **Schwarzschild/Kerr transfer-map workbench** —
  `?scene=transfer-map-reference` authenticates one of two bundled 1024×576
  stationary maps before either backend consumes it. The Kerr reference
  numerically integrates separated null geodesics of the exact analytic Kerr
  metric, with a finite-distance BL-ZAMO, a constant-Kerr-r oblate capture
  surface, and continuation to infinity.
- **Inspectable scientific diagnostics** — Stable URL modes show sky,
  outcomes, lookback time, frequency shift, null residual, or projection
  error. Clicking a texel exposes its decoded canonical 32-byte record.
- **Non-breaking scene architecture** — Optional scene descriptors and shader bundles extend the shared WebGPU/WebGL2 backends, while the default URL retains the original Schwarzschild shader, observer model, disk, and controls.
- **WebGPU first, WebGL2 fallback** — Chooses the rendering path from the GPU limits, texture dimensions, and framebuffer capabilities exposed at runtime, without chip-model-specific branches.
- **Progressive sky assets** — Ships with ESO 6K and 4K fallbacks and can optionally load the 16000×8000 ESA/Gaia all-sky map.
- **Capability-negotiated HDR** — Requests Display-P3, FP16, and extended-range output where available, then falls back to P3 or sRGB SDR. WebGL2 is used when WebGPU initialization is unavailable or fails.

## Quick start

There is no build step and no JavaScript package installation. Python is used only to serve the static files.

```bash
git clone https://github.com/ShuoleiWang/blackhole.git
cd blackhole
python3 -m http.server 4173
```

Open <http://localhost:4173>. WebGPU requires a secure context such as `localhost` or HTTPS; the application automatically attempts the WebGL2 fallback when WebGPU is unavailable.

The current application interface is in Simplified Chinese; this does not affect the rendering controls or URL parameters documented below.

Use the in-app scene selector, or open
<http://localhost:4173/?scene=binary-approx>, to enter the experimental binary
preview. Open
<http://localhost:4173/?scene=transfer-map-reference> for the fixed-camera
Schwarzschild transfer-map reference, or append `&reference=kerr-remnant` for
the stationary Kerr remnant reference. Returning to the default URL restores
the interactive Schwarzschild scene; all paths remain isolated.

The bundled 6K Milky Way background works immediately. To install the optional, approximately 236 MiB Gaia 16K map:

```bash
./scripts/fetch_gaia_sky.sh
```

The script downloads the original asset from ESA and verifies a pinned SHA-256 digest before installation. The large source file is intentionally excluded from Git.

## Controls

| Input | Action |
| --- | --- |
| Mouse drag / one-finger drag | Change orbital phase and the observer's orbital plane |
| Wheel / pinch | Change observer radius |
| Double-click the canvas | Reset the view |
| Arrow keys | Fine-tune orbital phase and plane |
| `0` | Place the observer orbit in the disk plane for a strict edge-on view |
| `+` / `-` | Decrease / increase observer radius |
| Space | Pause / resume simulation time |

The neutral science color mode and the stylized Hubble palette alter only the display mapping and lightweight PSF. They do not change geodesics, disk occlusion, or frequency shifts.

In the binary preview, drag and zoom still control the camera. The transport
button and Space pause or resume the same timeline; the range control scrubs
protocol time, and **Merger slow motion** toggles a presentation-only `0.12×`
rate from `t = -160 M` through `t = 70 M`. The waveform strip is the real,
CoM-corrected SXS `Extrapolated_N2` `h22` mode, with peak amplitude at protocol
`t = 0`. The accretion control is disabled because the source is a vacuum
binary. None of these controls changes the weak-field fast-light rendering
model.

The transfer-map workbench has a fixed camera and projection, so drag, zoom,
reset, motion, mass, accretion, and time controls are disabled. It can switch
between the Schwarzschild and Kerr references and display sky composition,
ray outcomes, lookback time, frequency factor, null residual, or projection
error. Click the canvas to inspect one canonical 32-byte ray record; arrow keys
move the selected texel, Shift accelerates movement, and Escape closes the
inspector. Exposure and display quality remain presentation controls.

## URL parameters

| Parameter | Purpose |
| --- | --- |
| `?scene=binary-approx` | Opt into SXS-driven binary dynamics rendered by the isolated weak-field fast-light preview; the default remains Schwarzschild |
| `?scene=transfer-map-reference` | Open the fixed-camera stationary analytic Schwarzschild transfer-map reference; not NR and no accretion disk |
| `?scene=transfer-map-reference&reference=kerr-remnant` | Open the stationary analytic Kerr remnant-spin reference; not NR and no accretion disk |
| `&diagnostic=sky\|outcome\|lookback\|frequency-shift\|null-residual\|projection-error` | Select a stable transfer-map workbench view |
| `?renderer=webgl` | Force the WebGL2 fallback path |
| `?hdr=0` | Disable extended HDR and use stable SDR output |
| `?sky=high` | Force the bundled ESO 6K Milky Way background |
| `?sky=ultra` | Block at startup while attempting to load the local Gaia 16K map |
| `?presentation=1` | Hide controls and status readouts for presentation or capture |

Parameters can be combined:

```text
http://localhost:4173/?scene=transfer-map-reference&reference=kerr-remnant&diagnostic=outcome&renderer=webgl&hdr=0
```

## Rendering pipeline

The default Schwarzschild path:

1. Generate camera rays in the local comoving frame of a circular-orbit observer.
2. Apply a Lorentz transformation into the local static Schwarzschild frame.
3. Integrate each null geodesic in the fragment shader and classify capture, escape, and disk-plane crossings.
4. Accumulate disk emission and transmittance from near to far, then sample the all-sky background in the escaped direction.
5. On WebGPU, ray trace into an FP16 intermediate target and select extended-range or SDR canvas output from the capabilities the browser preserves. WebGL2 provides an sRGB/SDR fallback.

The opt-in binary path lazy-loads and integrity-checks a versioned Phase 2
manifest plus its compact sample asset. The runtime linearly interpolates SXS
A/B apparent-horizon centroid separation and unwrapped coordinate phase, the
CoM-corrected extrapolated complex `h22`, and a separately labelled render-only
topology blend. It then supplies separation, phase, and blend to the existing
binary trace shader on both backends.

That shader applies a fast-light, frame-frozen two-centre weak-field deflection
and blends to one spherical visual remnant before reusing the shared sky,
post-processing, and HDR stages. It does **not** load the SXS near-zone
spacetime, integrate null geodesics through an NR metric, render remnant spin,
or compute either capture surface as an apparent/event horizon. The camera
also does not reuse the single-hole circular-observer Lorentz boost.

The reference path completes a separate fail-closed chain: select a reference
from a hard-coded trust registry; authenticate the exact manifest bytes,
sidecar, and chunks; validate the v1 schema, 32-byte records, coordinates,
outcomes, and accuracy; upload 589,824 records through the selected WebGPU or
WebGL2 resource path; select the nearest stored texel without blending ray
directions; and sample the panorama only for `escaped` outcomes before the
shared HDR/SDR stage.

Both products are single observations at `r = 40M`, with a 40-degree vertical
field of view and a fixed 1024×576 projection. The Schwarzschild map contains
557,772 escaped and 32,052 captured rays. The Kerr map uses
`a/M = 0.686461676493`, a finite-distance BL-ZAMO, and ingoing Cartesian
Kerr-Schild manifest coordinates; it contains 558,684 escaped and 31,140
captured rays. Both have zero unusable records. They are analytic stationary
references, not NR spacetimes, binary-merger images, accretion disks, or
GRMHD/radiative-transfer results.

Primary implementation files:

- [`src/main.js`](./src/main.js) — Scene selection, camera orbit, physical parameters, interaction, and adaptive quality
- [`src/shaders.js`](./src/shaders.js) — Default WGSL/GLSL Schwarzschild geodesics, disk emission, sky sampling, and post-processing
- [`src/scenes/binary-approx-scene.js`](./src/scenes/binary-approx-scene.js) — Opt-in scene lifecycle, SXS-driven timeline, transport UI, and frame parameters
- [`src/scenes/binary-dynamics-adapter.js`](./src/scenes/binary-dynamics-adapter.js) — Fail-closed browser loader, integrity checks, and deterministic dynamics interpolation
- [`src/scenes/binary-playback-clock.js`](./src/scenes/binary-playback-clock.js) — Scrubbing, frame-rate-independent playback, end hold, looping, and presentation-only slow motion
- [`src/binary-shaders.js`](./src/binary-shaders.js) — Matching WebGPU/WebGL2 weak-field binary trace shaders and scene-uniform adapter
- [`src/scenes/transfer-map-reference-scene.js`](./src/scenes/transfer-map-reference-scene.js) — Fixed-camera reference lifecycle and fail-closed loading UI
- [`src/transfer-map-loader.js`](./src/transfer-map-loader.js) — Browser-side manifest, sidecar, chunk, ABI, outcome, and accuracy validation
- [`src/transfer-map-shaders.js`](./src/transfer-map-shaders.js) — Matching nearest-texel WebGPU/WebGL2 consumers
- [`assets/transfer-maps/schwarzschild-reference-v1/manifest.json`](./assets/transfer-maps/schwarzschild-reference-v1/manifest.json) — Renderable 1024×576 analytic reference and nine hashed chunks
- [`scripts/generate_schwarzschild_transfer_map.py`](./scripts/generate_schwarzschild_transfer_map.py) — Deterministic offline generator
- [`scripts/verify_schwarzschild_transfer_map.py`](./scripts/verify_schwarzschild_transfer_map.py) — Independent stationary-physics verifier
- [`assets/transfer-maps/kerr-remnant-reference-v1/manifest.json`](./assets/transfer-maps/kerr-remnant-reference-v1/manifest.json) — Renderable 1024×576 analytic Kerr reference and hashed chunks
- [`scripts/generate_kerr_transfer_map.py`](./scripts/generate_kerr_transfer_map.py) — Deterministic Kerr null-geodesic generator with full-ray tolerance refinement
- [`scripts/verify_kerr_transfer_map.py`](./scripts/verify_kerr_transfer_map.py) — Independent finite-ZAMO shadow, Kerr-Schild identity, and fixed-step ray verifier
- [`docs/kerr-reference.md`](./docs/kerr-reference.md) — Kerr configuration, equations, validation boundary, and reproduction guide
- [`assets/scenes/binary-sxs-bbh-0001-v2.json`](./assets/scenes/binary-sxs-bbh-0001-v2.json) — Phase 2 source, scientific-status, event, integrity, error, renderer-boundary, and playback manifest
- [`assets/scenes/binary-sxs-bbh-0001-v2.samples.json`](./assets/scenes/binary-sxs-bbh-0001-v2.samples.json) — 2,732-sample compact SXS dynamics and waveform track
- [`scripts/generate_binary_sxs_dynamics.py`](./scripts/generate_binary_sxs_dynamics.py) — Offline, deterministic generator from three pinned official SXS files
- [`scripts/verify_binary_dynamics.py`](./scripts/verify_binary_dynamics.py) — Fail-closed Phase 2 source, asset, event, interpolation, remnant, renderer-boundary, and playback checks
- [`tests/binary-playback.test.mjs`](./tests/binary-playback.test.mjs) — Node tests for source anchors, interpolation, scrubbing, slow motion, end hold, and frame-rate independence
- [`assets/scenes/binary-pn-equal-mass-v1.json`](./assets/scenes/binary-pn-equal-mass-v1.json) — Legacy v1 PN/phenomenological asset, retained only for regression
- [`scripts/verify_binary_preview.py`](./scripts/verify_binary_preview.py) — Legacy PN contract and unchanged weak-field shader convergence regression
- [`docs/binary-model.md`](./docs/binary-model.md) — Binary scientific boundary, protocol status, and offline NR transfer-map architecture
- [`docs/nr-transfer-map-v1.md`](./docs/nr-transfer-map-v1.md) — Normative
  terminology, field semantics, safety rules, and status of the transfer-map v1
  protocol
- [`schemas/nr-transfer-map-v1.schema.json`](./schemas/nr-transfer-map-v1.schema.json) — Machine-readable transfer-map manifest schema
- [`assets/transfer-maps/contract-fixture-v1/manifest.json`](./assets/transfer-maps/contract-fixture-v1/manifest.json) — Small project-generated conformance fixture; it contains no NR-derived payload
- [`scripts/generate_nr_contract_fixture.py`](./scripts/generate_nr_contract_fixture.py) — Deterministically regenerate the conformance fixture
- [`scripts/verify_nr_contract.py`](./scripts/verify_nr_contract.py) — Fail-closed manifest, sidecar, coordinate-frame, and per-ray record validator
- [`tests/test_nr_contract.py`](./tests/test_nr_contract.py) — Positive and adversarial protocol regression tests
- [`src/webgpu-renderer.js`](./src/webgpu-renderer.js) — Two-stage WebGPU renderer and HDR/P3 configuration negotiation
- [`src/webgl-renderer.js`](./src/webgl-renderer.js) — WebGL2 fallback and half-float framebuffer probing

## Model scope and limitations

| Scene / component | Implemented | Current boundary |
| --- | --- | --- |
| Default single black hole | Non-rotating Schwarzschild spacetime and numerical GPU null-geodesic integration | No Kerr spin or frame dragging; the narrowest critical-curve features remain sampling-limited |
| Default accretion disk | Idealized zero-thickness surface from `r = 6M` to `18M`, frequency shifts, approximate emission, and turbulence-inspired structure | No finite scale height, GRMHD, complete spectrum, polarization, or self-consistent radiative transfer |
| Binary orbital dynamics | SXS:BBH:0001 Lev5 A/B apparent-horizon inertial-coordinate centroid separation and unwrapped phase from relaxation through the last paired A/B sample | Real NR diagnostics, but coordinate- and gauge-dependent; after individual horizons end, the renderer holds their last separation/phase rather than inventing trajectories |
| Binary waveform | CoM-corrected `Extrapolated_N2` complex `h22`, aligned so its maximum amplitude is protocol `t = 0` | A far-zone waveform is not a near-zone metric and cannot determine camera-ray propagation |
| Binary merger/remnant data | Common apparent horizon at `t = -6.072285 M`; exact metadata remnant mass `0.951609417715 M` and spin vector `(-7.29520687012e-10, 7.40468371215e-10, 0.686461676493)` | The topology blend from the common-horizon event to waveform peak is a presentation proxy; the shader does not render horizon geometry, recoil, Kerr spin, or frame dragging |
| Binary lensing | Fast-light bending from two frame-frozen weak-field monopoles, followed by one spherical remnant proxy | Not strong-field geodesic integration; remnant spin/frame dragging are not rendered, and fine photon rings, caustics, delays, and horizon topology are not quantitatively reliable |
| Binary emission | Vacuum sky lensing with no accretion disk | Adding luminous plasma would require physical gas initial data, GRMHD, and radiative transfer |
| Stationary Schwarzschild reference | Fixed 1024×576 analytic vacuum map, authenticated chunks, nearest-texel WebGPU/WebGL2 playback | Fixed camera; no disk, NR source, time interpolation, or binary slow-light rays |
| Stationary Kerr remnant reference | Numerically integrated vacuum geodesics of the exact analytic Kerr metric at `a/M = 0.686461676493`, finite BL-ZAMO camera, oblate Kerr-r capture surface, authenticated playback and diagnostics | Uses only the SXS remnant spin parameter; no SXS near-zone metric, binary time dependence, emission, or NR-derived pixels |
| NR transfer-map protocol | Versioned schema, deterministic synthetic fixture, fail-closed validators, reference consumer, and regression tests | The runtime is proven with analytic data only; no NR-derived transfer map is bundled |
| Shared renderer | WebGPU primary path with WebGL2 fallback | HDR, P3, FP16, and 16K textures depend on runtime capabilities; HDR does not improve model accuracy |

See [`docs/physics-notes.md`](./docs/physics-notes.md) (currently in Simplified
Chinese) for the real-time Schwarzschild model,
[`docs/kerr-reference.md`](./docs/kerr-reference.md) for the stationary Kerr
product, and [`docs/binary-model.md`](./docs/binary-model.md) for the binary
preview boundary and offline NR-to-transfer-map architecture.

## M3 Pro compatibility and HDR

The current hardware target is **M3 Pro**. It has been manually exercised with
WebGPU/Metal, WebGL2/ANGLE-on-Metal fallback, Display-P3 FP16 output, SDR
fallback, and the 16K background upgrade. Texture limits, canvas formats,
half-float framebuffer completeness, and display range are negotiated at
runtime; this document makes no separate M4 compatibility claim.

The upper-right status bar reports the active backend, available adapter label, output mode, FPS, and internal render resolution. Adaptive quality adjusts ordinary-ray step counts and resolution within the user-selected ceiling, while rays near the critical impact parameter retain a larger integration budget.

## Validation

```bash
python3 scripts/verify_physics.py
python3 scripts/verify_binary_dynamics.py
node --test tests/binary-playback.test.mjs
python3 scripts/verify_binary_preview.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_nr_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_nr_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_nr_contract.py assets/transfer-maps/schwarzschild-reference-v1/manifest.json
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_schwarzschild_transfer_map.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_schwarzschild_transfer_map.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_nr_contract.py assets/transfer-maps/kerr-remnant-reference-v1/manifest.json
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_kerr_transfer_map.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_kerr_transfer_map.py
node --test tests/transfer-map-runtime.test.mjs
```

Regenerate the bundled reference deterministically with:

```bash
python3 scripts/generate_schwarzschild_transfer_map.py
python3 scripts/generate_kerr_transfer_map.py
```

The Schwarzschild numerical regression checks cover:

- Critical impact parameter `b_c = 3√3 M`
- Agreement between weak-field deflection and `4M/b`
- Shadow angular diameter for a finite-distance observer
- The null-geodesic integration invariant
- Capture and escape behavior under the 184- and 288-step real-time budgets

The Phase 2 binary validator pins the three official source files by URL, size,
MD5, and SHA-256; checks the 2,732-sample sidecar hash and schema; confirms the
SXS event ordering, `h22` peak at `t = 0`, common horizon at
`t = -6.072285 M`, exact metadata remnant values, validity-strict
post-horizon hold, and all declared interpolation bounds. The largest measured
orbital-phase interpolation residual is `6.442e-4 rad`. The Node tests exercise
event anchors, finite interpolation, scrub clamping, presentation-only slow
motion, deterministic looping/end hold, and frame-rate independence.

The legacy binary regression still checks the old PN manifest and a
representative 90×45 CPU ray grid against the unchanged shader equations. Its
fixed 512-step production budget must leave no unresolved rays in representative
wide-binary, transition, and remnant views. Together these checks validate
source integrity, dynamics playback, and declared real-time convergence gates;
they do **not** validate NR light propagation, a solved binary spacetime,
strong-field lensing, or a quantitatively accurate merger image.

The NR contract checks strict JSON/schema conformance, immutable sidecar
hashes and sizes, portable artifact-location rules, contiguous ordered chunks,
physical-system and source/protocol-time declarations, mutually inverse
spatial affine frames, proper ICRS rotations, observer-tetrad orthonormality,
ray-integration and boundary semantics, and legal finite per-ray outcomes. It
also cross-checks decoded outcome fractions before a dataset can be marked
renderable. Unknown or missing fields, duplicate keys, non-finite numbers,
path escapes, and ambiguous invalid-ray states are rejected. Passing these
checks means **protocol-conformant**, not **NR-backed** or
**physically validated**.

The Schwarzschild verifier independently recovers a `14.548010°` finite-distance
shadow diameter and boundary frequency factor `g = 1.024951860`. It reports a
maximum sampled analytic null residual of `7.678e-14`, maximum independent
direction error `1.062e-8 rad`, and maximum stored per-ray projection estimate
`1.415e-2 px`. These are stationary-reference checks: the NR convergence and
constraint-norm fields are correctly `not-applicable`, not zero-valued NR
measurements.

The Kerr verifier independently reconstructs the Cartesian Kerr-Schild metric
and BL-ZAMO tetrad, evaluates the finite-distance spherical-photon critical
curve, checks the complete capture mask, and traces representative full rays
with a fixed-step RK4 implementation. The wider Kerr validation suite adds
generator-level Schwarzschild-limit and spin-reversal mirror regressions, while
the verifier checks first-integral separation, infinity-tail, null, and
per-record projection gates. The bundled map has zero analytic capture-mask
mismatches, maximum stored null residual `3.068e-9`, p95/maximum projection
estimates `1.929e-4 / 3.752e-3 px`, and maximum independent direction error
`8.679e-9 rad`. See
[`docs/kerr-reference.md`](./docs/kerr-reference.md) for the exact model and
measured acceptance criteria.

Together, these scripts validate selected numerical properties and architecture
contracts. They are not complete visual, radiative-model, or cross-GPU
validation. The repository does not currently include GPU image-regression CI.

## Sky assets and attribution

- **ESA/Gaia/DPAC · A. Moitinho** — Optional 16000×8000 Gaia EDR3 data-derived all-sky map, licensed under CC BY-SA 3.0 IGO.
- **ESO/S. Brunier** — Bundled 6000×3000 photographic Milky Way panorama, licensed under CC BY 4.0.
- `assets/deep-field.webp` — Script-generated deep-space fallback asset; it is not the default sky.

See [`assets/SOURCES.md`](./assets/SOURCES.md) for download locations, transformations, hashes, and complete license information. Third-party assets are not relicensed by any future license selected for this project's code.

## License

No license has currently been declared for the project code. Third-party sky
assets, SXS-derived data, and vendored dependencies remain subject to their
source terms. The pinned Zenodo record used for the Phase 2 SXS files does not
declare a license; this repository therefore records that status without
inventing an SPDX identifier or inferring a license from another page. Until a
project license is selected, do not assume that the repository is available
under MIT, Apache-2.0, or another software license.
