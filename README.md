# Schwarzschild Black Hole Renderer

**English** | [简体中文](./README.zh-CN.md)

An interactive, real-time Schwarzschild black hole renderer built with **WebGPU and WebGL2**.

The renderer numerically integrates past-directed null geodesics in a GPU fragment shader. The same ray path determines capture by the event horizon, intersections with an idealized accretion disk, relativistic frequency shifts, and gravitational lensing of an all-sky Milky Way background. This project is intended for real-time visualization and education; it is not a Kerr, GRMHD, or high-precision radiative-transfer solver.

![A Schwarzschild black hole, accretion disk, and gravitationally lensed Milky Way](./docs/images/blackhole-galaxy-hero.webp)

<sub>A 5120×2576 in-app screenshot of the WebGPU/Metal renderer running on Apple Silicon, with the controls and live backend, output, and performance readouts visible. Milky Way source: ESO/S. Brunier; geodesically transformed, composited, and transcoded by this project from an original used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [`assets/SOURCES.md`](./assets/SOURCES.md) for full provenance.</sub>

## Key features

- **Per-pixel null-geodesic integration** — Uses Störmer–Verlet integration of `u'' = -u + 3u²` instead of a screen-space distortion effect.
- **Unified ray-path composition** — A single traced ray handles capture, multiple disk-plane intersections, and the final sky escape direction, producing critical-curve arcs and higher-order images.
- **Relativistic disk appearance** — Includes frequency shifts from Schwarzschild circular motion, the bolometric intensity transfer factor `g⁴`, approximate blackbody chromaticity, surface optical depth, and limb darkening.
- **Real-time procedural disk structure** — Turbulence-inspired, finite-lifetime noise is advected at the local Keplerian angular velocity. This is a visual approximation, not an MHD simulation.
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

## URL parameters

| Parameter | Purpose |
| --- | --- |
| `?renderer=webgl` | Force the WebGL2 fallback path |
| `?hdr=0` | Disable extended HDR and use stable SDR output |
| `?sky=high` | Force the bundled ESO 6K Milky Way background |
| `?sky=ultra` | Block at startup while attempting to load the local Gaia 16K map |
| `?presentation=1` | Hide controls and status readouts for presentation or capture |

Parameters can be combined:

```text
http://localhost:4173/?presentation=1&sky=high&hdr=0
```

## Rendering pipeline

1. Generate camera rays in the local comoving frame of a circular-orbit observer.
2. Apply a Lorentz transformation into the local static Schwarzschild frame.
3. Integrate each null geodesic in the fragment shader and classify capture, escape, and disk-plane crossings.
4. Accumulate disk emission and transmittance from near to far, then sample the all-sky background in the escaped direction.
5. On WebGPU, ray trace into an FP16 intermediate target and select extended-range or SDR canvas output from the capabilities the browser preserves. WebGL2 provides an sRGB/SDR fallback.

Primary implementation files:

- [`src/shaders.js`](./src/shaders.js) — WGSL/GLSL geodesics, disk emission, sky sampling, and post-processing
- [`src/webgpu-renderer.js`](./src/webgpu-renderer.js) — Two-stage WebGPU renderer and HDR/P3 configuration negotiation
- [`src/webgl-renderer.js`](./src/webgl-renderer.js) — WebGL2 fallback and half-float framebuffer probing
- [`src/main.js`](./src/main.js) — Camera orbit, physical parameters, interaction, and adaptive quality

## Model scope and limitations

| Implemented | Current boundary |
| --- | --- |
| Non-rotating Schwarzschild spacetime | No Kerr spin or frame dragging |
| Numerical GPU null-geodesic integration | The narrowest critical-curve features remain limited by step count and pixel sampling |
| Idealized, geometrically zero-thickness disk from `r = 6M` to `18M` | No finite scale height or three-dimensional volume emission |
| Gravitational/Doppler shifts and real-time disk-emission approximations | Not a complete spectrum, polarization model, or self-consistent radiative-transfer solution |
| Turbulence-inspired procedural disk structure | Does not solve magnetohydrodynamics or reproduce measured MRI data |
| WebGPU primary path with WebGL2 fallback | HDR, P3, FP16, and 16K textures depend on runtime capabilities |

See [`docs/physics-notes.md`](./docs/physics-notes.md) (currently in Simplified Chinese) for notes on geometric units, critical orbits, edge-on images, and relativistic brightness asymmetry.

## Compatibility and HDR

The renderer contains no M3-, M4-, or vendor-specific rendering branch. It negotiates texture limits, canvas formats, half-float framebuffer completeness, and display dynamic range at runtime, allowing the same code to select the appropriate WebGPU/Metal or WebGL2/Metal path across Apple Silicon systems.

- **M3 Pro** — Manually tested with WebGPU/Metal, WebGL2/Metal, the Display-P3 FP16 path, SDR fallback, and background upgrade to the 16K map.
- **M4** — Uses the same capability-negotiation path and requires no M4-specific feature. The repository does not yet record an M4 hardware smoke test.
- **Other platforms** — WebGPU, HDR, large textures, and color-space support depend on the browser, operating system, driver, display, and the screen containing the window.

The upper-right status bar reports the active backend, available adapter label, output mode, FPS, and internal render resolution. Adaptive quality adjusts ordinary-ray step counts and resolution within the user-selected ceiling, while rays near the critical impact parameter retain a larger integration budget.

## Validation

```bash
python3 scripts/verify_physics.py
```

The numerical regression checks cover:

- Critical impact parameter `b_c = 3√3 M`
- Agreement between weak-field deflection and `4M/b`
- Shadow angular diameter for a finite-distance observer
- The null-geodesic integration invariant
- Capture and escape behavior under the 184- and 288-step real-time budgets

These checks validate a selected set of Schwarzschild numerical properties. They are not complete visual, radiative-model, or cross-GPU validation. The repository does not currently include GPU image-regression CI.

## Sky assets and attribution

- **ESA/Gaia/DPAC · A. Moitinho** — Optional 16000×8000 Gaia EDR3 data-derived all-sky map, licensed under CC BY-SA 3.0 IGO.
- **ESO/S. Brunier** — Bundled 6000×3000 photographic Milky Way panorama, licensed under CC BY 4.0.
- `assets/deep-field.webp` — Script-generated deep-space fallback asset; it is not the default sky.

See [`assets/SOURCES.md`](./assets/SOURCES.md) for download locations, transformations, hashes, and complete license information. Third-party assets are not relicensed by any future license selected for this project's code.

## License

No license has currently been declared for the project code. Third-party sky assets and vendored dependencies remain subject to their original licenses. Until a project license is selected, do not assume that the repository is available under MIT, Apache-2.0, or another software license.
