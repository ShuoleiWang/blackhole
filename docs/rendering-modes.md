# Rendering modes and scientific boundaries

This document separates the repository's implemented products from its two
long-term rendering routes. The distinction is architectural and scientific:
an interactive GPU renderer, a fixed-camera transfer-map reference, and a
high-fidelity offline renderer solve different problems and support different
claims.

## Product layers

| Layer | Status | What it does | Permitted claim |
| --- | --- | --- | --- |
| Interactive binary scene at the root URL | Implemented | Reconstructs the camera and integrates a 3+1 null Hamiltonian through a frame-frozen boosted-superposed Kerr-Schild approximation on WebGPU | Real-time approximate strong-field fast-light rendering; not NR |
| Interactive Schwarzschild scene at `?scene=schwarzschild` | Implemented | Reconstructs the camera and numerically traces its Schwarzschild rays on the GPU for every rendered frame | Real-time single-hole Schwarzschild visualization with an idealized disk |
| Stationary Schwarzschild/Kerr workbench | Implemented | Plays authenticated fixed-camera analytic vacuum maps and exposes record-level diagnostics | Stationary analytic calibration, delivery validation, and regression oracle |
| WebGL2 binary compatibility path | Implemented | Runs the previous two-centre weak-field shader when WebGPU is unavailable or forced off | Explicit weak-field preview; no physical-parity claim with WebGPU |
| High-fidelity offline renderer | Planned, not implemented | Would trace ray bundles through a pinned four-dimensional NR spacetime and optionally perform physical radiative transfer | No current repository output qualifies |

The stationary references are not intermediate frames from the root binary scene.
They are deliberately separate analytic products with fixed cameras.

## Route A: real-time interactive rendering

The interactive route keeps camera control and ray generation on the GPU:

```text
mouse / touch / timeline input
        ↓
camera event and local tetrad
        ↓
fresh per-pixel rays for the rendered frame
        ↓
GPU geodesic or lens integration
        ↓
sky / declared emission model
        ↓
HDR or SDR display transform
```

The root binary scene and the explicit `?scene=schwarzschild` scene recompute
their GPU ray paths for each rendered frame; they do not look up a fixed-camera
transfer map. The legacy `?scene=binary-approx` URL remains an alias for the
root binary scene. Their physics are different:

- the Schwarzschild scene integrates a reduced null-geodesic equation;
- the WebGPU binary scene freezes a declared boosted-superposed Kerr-Schild
  spacetime for each ray, evaluates its lapse, shift, spatial metric, and
  analytic spatial derivatives, and integrates a reduced 3+1 null Hamiltonian;
- the WebGL2 binary fallback remains the old two-centre weak-field deflection
  and is labelled as a distinct model.

The WebGPU path uses a local ADM-orthonormal camera tetrad, explicit
captured/escaped/unresolved outcomes, a C² transition to the analytic Kerr
remnant, and a weak-field analytic continuation from its finite escape sphere.
It stores the future-directed covector of the photon arriving at the camera and
integrates the Hamiltonian flow with negative coordinate-time steps, so the
traced boundary-value path is past-directed. It retains its approximate metric,
isolated-Kerr excision surfaces, null residual, and unresolved rays as visible
scientific boundaries.

The M3 Pro scheduler also exposes a numerical hierarchy rather than pretending
all frames have the same convergence. Deadline-oriented `emergency`, `survival`,
and `interactive` tiers use larger declared capture padding, longer far-zone
steps, and smaller budgets. A paused view may progress through `balanced` to
`fine`, the strictest settled tier. These are latency/convergence policies
inside the same approximate fast-light model, not different physical
spacetimes. See [`strong-field-performance.md`](./strong-field-performance.md)
for the exact values and measured boundary.

### Fast light and slow light

In **fast-light** rendering, a ray samples one frame-frozen spacetime state
along its complete path. This is suitable for responsive exploration and is the
model used by the current binary scene. It omits changes in the binary while
the photon propagates, so it cannot reproduce time-of-flight effects or
time-dependent strong-field lensing.

In **slow-light** rendering, every integration point samples the spacetime at
the coordinate time reached by that ray. Slow light therefore requires
time-dependent metric data, gauge-aware spacetime interpolation, moving horizon
worldtubes, and a sufficiently long source-time interval. Substituting a
heuristic retarded body position into a frozen metric is not equivalent.

The implemented interactive strong-field fast-light renderer is a substantial
improvement over the retained WebGL2 weak-field shader, but it is still not NR
ray tracing. A superposed analytic metric remains an approximation even if its
geodesics are integrated accurately.

## Route B: high-fidelity offline rendering

The offline route prioritizes physical provenance and convergence over frame
latency:

```text
pinned 4D NR spacetime + horizon worldtubes
        ↓
gauge / frame / time adapter + source convergence metadata
        ↓
float64 slow-light ray bundles + geodesic deviation / Jacobi fields
        ↓
independent ray-tolerance and stationary Schwarzschild/Kerr gates
        ↓
optional GRMHD fluid + electron / emissivity prescription
        ↓
spectral and polarized GR radiative transfer
        ↓
adaptive subpixel sampling and convergence
        ↓
multilayer OpenEXR scientific master + immutable audit manifest
        ↓
derived HDR image/video and optional browser transfer-map proxy
```

A vacuum NR lensing product may stop after the ray-bundle stage and sample a
declared distant source. A luminous accretion or merger product needs additional
matter data and a declared emission, absorption, optical-depth, spectral, and
polarization model. GRMHD and GR radiative transfer are optional pipeline
layers, but their absence must remain visible in the scientific status.

The scientific master and browser delivery product should be separate. Tone
mapping, bloom, colour gamut, video encoding, and HDR presentation must not
alter or replace the underlying physics acceptance record.

## Role of transfer-map v1

`blackhole.nr-transfer-map/v1` is retained as an immutable,
camera-specific **vacuum escape-transfer** ABI. Its canonical 32-byte record
stores:

- an escaped ICRS direction;
- a frequency factor;
- a coordinate lookback time;
- a terminal outcome and capture target;
- null-residual and projection-error diagnostics;
- an explicit validity mask.

This is sufficient for authenticated vacuum sky mapping and for stationary or
future NR slow-light endpoint products. It is not a general ray-bundle or
radiative-transfer format: it has no adaptive subpixel sample list, geodesic
deviation or Jacobi matrix, finite-volume emission history, optical depth,
spectral bins, Stokes parameters, or Faraday coefficients.

The v1 schema, generators, independent verifiers, fixtures, bundled
Schwarzschild/Kerr maps, hashes, and browser consumer remain part of the
project's regression foundation. Future high-fidelity products must use a new,
separately versioned ray-bundle and/or radiative-frame contract rather than
changing the meaning or binary layout of v1.

## Scientific claim ladder

| Claim | Minimum evidence |
| --- | --- |
| **Contract-conformant** | The declared schema, manifest, artifacts, chunks, coordinates, outcomes, and records pass fail-closed validation |
| **Stationary-analytic-validated** | Independent Schwarzschild/Kerr limits, capture masks, conserved quantities, and integration refinements pass declared gates |
| **NR-backed vacuum lensing** | The actual pixel rays use a pinned four-dimensional NR near-zone spacetime, documented horizon data, and slow-light integration |
| **GRMHD/GRRT-backed radiance** | Pixel radiance additionally uses pinned matter fields and declared emissivity, absorption, spectral, and polarization transport |
| **Physically validated offline render** | Spacetime resolution, constraints, metric interpolation, ray integration, subpixel/spectral sampling, and relevant radiative-transfer comparisons all converge within published tolerances |

“Photorealistic,” HDR, Display-P3, and high resolution are presentation
properties. They do not establish any scientific claim in this table.

## What is implemented now

The repository currently provides:

- interactive per-frame WebGPU strong-field approximate fast-light tracing for
  the root binary scene, with SXS waveform/event/remnant anchors and analytic
  renderer coordinates that exclude gauge-dependent SXS centroids;
- documented past-directed ray conventions, isolated-Kerr excision semantics,
  independent analytic-limit oracles, and adaptive M3 Pro convergence tiers;
- an explicit WebGL2 weak-field binary compatibility path;
- interactive GPU ray recomputation for the explicit Schwarzschild scene;
- deterministic stationary Schwarzschild and Kerr vacuum generators;
- independent stationary physics verifiers;
- authenticated v1 map playback, diagnostics, and record inspection.

It does not currently provide four-dimensional NR metric data, NR slow-light
pixel rays, ray bundles or Jacobi fields, GRMHD matter data,
spectral/polarized GR radiative transfer, or multilayer OpenEXR scientific
masters.

The implemented metric and ray conventions are specified in
[`strong-field-equations.md`](./strong-field-equations.md); scheduling and
one-frame backpressure in
[`strong-field-performance.md`](./strong-field-performance.md); and the
independent CPU acceptance boundary in
[`strong-field-ray-oracles.md`](./strong-field-ray-oracles.md).
