# Binary black-hole model

This document defines the scientific scope of the optional binary black-hole
scene. It is intentionally stricter than a visual feature list: a convincing
image is not, by itself, evidence that the underlying spacetime is a solution
of Einstein's field equations.

## Status at a glance

| Component | Version 1 | Scientific status |
| --- | --- | --- |
| Binary configuration | Equal mass, non-spinning, quasi-circular | A standard reference configuration |
| Inspiral motion | Leading-order post-Newtonian (PN) quadrupole evolution | Controlled only in the weak-field, adiabatic regime |
| Merger and post-merger display | Smooth phenomenological transition and non-oscillatory amplitude decay | Render-only interpolation, not an Einstein-equation or quasinormal-mode solution |
| Light propagation | Real-time, multi-centre weak-field bending | Useful for an interactive preview; invalid as precision strong-field ray tracing |
| Emission | Lensed all-sky background in vacuum | No accretion disks, plasma, GRMHD, or radiative transfer |
| Display | WebGPU with WebGL2 fallback and the existing HDR pipeline | Display fidelity does not increase physical accuracy |

The scene must therefore be described as a **PN / NR-informed preview**, never
as an exact simulation of a merger. The machine-readable source for the v1
timeline is
[`assets/scenes/binary-pn-equal-mass-v1.json`](../assets/scenes/binary-pn-equal-mass-v1.json).
Its metadata repeats this limitation so that downstream code cannot
accidentally lose the distinction.

The rounded remnant mass and spin in that manifest come from the official
`SXS:BBH:0001` Lev5 metadata. The inspiral/merger timeline and all rendered ray
paths remain project-generated approximations; using two catalog metadata
values does not turn them into NR data.

## Version 1 dynamics

The preview uses geometric units,

```text
G = c = M = 1,
```

where `M = m1 + m2` is the initial total mass. The bundled configuration has
`m1 = m2 = M/2`, symmetric mass ratio `η = m1 m2 / M² = 1/4`, and zero
individual spins.

During the inspiral, the orbital separation `a` and angular frequency `Ω`
follow the leading-order circular quadrupole model:

```text
da/dt = -(64/5) η / a³
M Ω   = a^(-3/2)
```

Equivalently, relative to the matching sample,

```text
a(t)^4 = a_match^4 - (256/5) η t.
```

The orbital phase is the integral of `Ω`. The runtime evaluates `a(t)`, `φ(t)`,
and the leading-order `h+`/`h×` preview analytically between the manifest's PN
checkpoints; only the explicitly phenomenological `t > 0` section is linearly
interpolated. These equations capture the qualitative chirp and secular
inspiral, but omit higher-order PN terms, spin couplings, eccentricity, tidal
effects, gauge dynamics, and the nonlinear strong-field interaction. Their
error grows rapidly near merger.

The v1 timeline switches from PN inspiral to a finite-duration visual blend and
then to a representative equal-mass remnant. That blend is deliberately
labelled `phenomenological-merger` in the data. It does not compute a common
apparent horizon, gravitational recoil, nonlinear wave generation, or
quasinormal-mode amplitudes from first principles. Likewise, the small waveform
readout is a presentation aid: its inspiral is leading-order and its merger
continuation is phenomenological.

## Version 1 light propagation

The interactive renderer cannot integrate null geodesics through a solved,
time-dependent binary metric because no such four-dimensional metric is
bundled. Instead, it approximates the weak-field gravitational potential as a
sum of two frame-frozen monopoles whose positions change between display
frames,

```text
Φ(x, t) = -Σ_i m_i / |x - x_i(t)|,
```

and bends a ray direction `n` along path length `s` using the transverse
potential gradient,

```text
dn/ds ≈ -2 [∇Φ - n(n · ∇Φ)].
```

The two positions are frozen at the current timeline sample for the duration
of each ray integration (the `fast-light` approximation). The shader does not
evaluate retarded positions or let one ray traverse a changing binary metric;
it also omits velocity-dependent and gravitomagnetic terms.

For one isolated mass and a large impact parameter this has the expected
leading deflection scale `4M/b`. It is not a valid strong-field metric near
either horizon, and the sum of two potentials is not a binary solution of
Einstein's equations. Capture surfaces and the transition from two dark
objects to one remnant are consequently visualization boundaries, not a
computed event-horizon world tube.

The metadata-backed remnant spin is retained in the data contract but is not
rendered. Version 1 uses a spherical monopole and a spherical `r = 2M` capture
proxy, with no Kerr geometry, frame dragging, or spin-dependent shadow shape.

The practical implications are important:

- The broad distortion of the Milky Way and the qualitative tightening of the
  binary can be interpreted as a physically motivated preview.
- Fine photon rings, higher-order images, caustic magnifications, arrival-time
  delays, and the shape or topology of a common horizon are not quantitatively
  reliable in v1.
- A frame must not be used to infer masses, spins, orbital parameters, lensing
  cross sections, or gravitational-wave observables.
- HDR preserves highlight range and colour on supported displays; it does not
  repair missing spacetime physics.

The scene models a vacuum binary and intentionally does not place a bright
accretion disk around each object. A physically defensible luminous merger
would additionally require an initial gas configuration, general-relativistic
magnetohydrodynamics (GRMHD), electron-temperature and emissivity prescriptions,
and time-dependent polarized radiative transfer.

## Why this is not full numerical relativity

A binary merger has no global, stationary Schwarzschild or Kerr metric.
Superposing two single-hole metrics, forces, shadows, or screen-space
distortions does not solve

```text
G_μν = 8π T_μν.
```

In vacuum, a production calculation instead evolves constraint-satisfying
initial data on a numerical grid using a formulation such as BSSN or generalized
harmonic evolution. Ray tracing then needs the resulting **time-dependent
near-zone geometry**, not only the black-hole trajectories or the gravitational
waveform measured far away.

Public waveform and horizon time series are valuable for validating masses,
spins, phase, remnant properties, and radiation. By themselves, however, they
do not determine the metric between the camera and the source. An arbitrary
camera ray requires access to a suitable four-metric (or equivalent 3+1 fields)
throughout the spacetime region and time interval crossed by that ray.

Even a full numerical-relativity implementation would be a convergent numerical
approximation rather than mathematically exact. A defensible scientific claim
would report resolution studies, constraint violation, null-geodesic error,
interpolation error, and comparisons against known limits.

## Isolation from the Schwarzschild renderer

The binary scene is an opt-in experiment. The default URL continues to select
the existing single, non-rotating Schwarzschild black hole and its original
geodesic shader, observer model, disk model, controls, and validation suite.
The binary module and its shader bundle are loaded only when the binary scene is
requested explicitly, for example:

```text
http://localhost:4173/?scene=binary-approx
```

Renderer extensions are optional: without a scene-provided shader and extra
uniform writer, WebGPU and WebGL2 use their existing sources and uniform layout.
This preserves the previous final single-black-hole behavior and avoids making
the experimental model an implicit dependency of the default path. Scene
switching in v1 uses a page reload rather than mutating GPU pipelines in place.

The two validation suites also have different meanings:

- `python3 scripts/verify_physics.py` checks selected Schwarzschild invariants
  and capture/escape behavior.
- `python3 scripts/verify_binary_preview.py` checks the binary manifest,
  equal-mass symmetry, PN sample equations, monotonic inspiral, transition
  bounds, and a representative CPU ray-grid convergence gate matching the
  shader equations. It does **not** certify the merger as a
  numerical-relativity result.

## Target architecture for an NR-backed scene

The browser should remain a deterministic playback and presentation layer.
Expensive spacetime evolution and geodesic integration belong in an offline
pipeline:

```text
constraint-satisfying initial data
        ↓
3+1 numerical-relativity evolution
        ↓
time-dependent metric + horizon diagnostics
        ↓
offline slow-light null-geodesic integration
        ↓
camera-specific transfer-map chunks
        ↓
WebGPU/WebGL interpolation, composition, and HDR display
```

“Slow light” is essential here: every ray samples the metric at the coordinate
time it reaches each integration point. Freezing one numerical-relativity slice
for an entire ray can be useful diagnostically, but it does not represent a
rapidly changing merger.

### Proposed transfer-map contract

A future data-backed scene should consume versioned, immutable transfer maps
rather than embedding a particular NR solver into the web application. Each
dataset should contain:

1. **Provenance** — simulation identifier and version, code/release, DOI,
   extraction process, license, source hashes, generation command, and transfer
   map hashes.
2. **Physical metadata** — mass ratio, initial spins and eccentricity, total
   mass normalization, reference time/phase, remnant properties, and the
   waveform convention used for synchronization.
3. **Coordinate contract** — NR gauge and coordinates, time slicing, world-to-
   camera transform, observer worldline and orthonormal tetrad, image-plane
   projection, units, handedness, and sky-map orientation.
4. **Sampling contract** — time samples, camera samples, image dimensions,
   interpolation rules, chunking, precision, compression, and validity masks.
5. **Per-ray observables** — escape direction, capture/body identifier,
   frequency-shift factor `g = ν_obs/ν_emit`, coordinate or arrival-time delay,
   and optional magnification/Jacobian information.
6. **Optional radiation fields** — accumulated optical depth and Stokes
   parameters, but only when generated from a documented GRMHD and
   general-relativistic radiative-transfer model.
7. **Error metadata** — NR resolution, constraint norms, geodesic null residual,
   interpolation residual, unresolved/terminated ray flags, and comparisons
   against stationary Schwarzschild/Kerr limits.

The v1 `blackhole.binary-scene/v1` JSON is a lightweight scene timeline, not
this transfer-map format. Keeping the two concepts separate allows the current
preview to be replaced by NR-derived data without pretending that its compact
orbital samples contain a spacetime.

## Validation gates for a future scientific mode

An NR-backed release should not be labelled physically faithful until it passes
at least the following gates:

- Convergence of the spacetime solution across multiple grid resolutions.
- Hamiltonian/momentum constraint and horizon diagnostics within declared
  tolerances.
- Preservation of the ray null constraint and agreement between independent
  integration tolerances.
- Recovery of analytic or high-accuracy Schwarzschild/Kerr lensing in stationary
  limits.
- Temporal and spatial transfer-map interpolation error bounds.
- Cross-checks of orbital phase, waveform, radiated energy/angular momentum,
  final mass, and final spin against the source simulation.
- Image-regression checks that are separated from physics checks so a tone-map
  change cannot masquerade as a physical change.

## References and data sources

- SXS Collaboration, [SXS Gravitational Waveform
  Database](https://data.black-holes.org/waveforms/catalog.html) and
  [documentation](https://data.black-holes.org/waveforms/documentation.html),
  plus the [`SXS:BBH:0001` dataset and Lev5
  metadata](https://doi.org/10.5281/zenodo.3273935). The bundled v1 file uses
  the catalog configuration and rounded remnant mass/spin; its compact
  timeline is not extracted from SXS spacetime, waveform, or horizon data.
- M. Boyle et al., [The SXS Collaboration catalog of binary black hole
  simulations](https://doi.org/10.1088/1361-6382/ab34e2), *Classical and
  Quantum Gravity* 36, 195006 (2019).
- A. Bohn et al., [What does a binary black hole merger look
  like?](https://arxiv.org/abs/1410.7775), *Classical and Quantum Gravity* 32,
  065002 (2015). This work demonstrates ray tracing through a dynamical
  numerical-relativity spacetime.
- Einstein Toolkit, [GW150914 binary black-hole example and gallery
  entry](https://www.einsteintoolkit.org/gallery/bbh/index.html).
- Einstein Toolkit, [TwoPunctures initial-data
  documentation](https://einsteintoolkit.org/thornguide/EinsteinInitialData/TwoPunctures/documentation.html)
  and [ADMBase 3+1 field
  documentation](https://einsteintoolkit.org/thornguide/EinsteinBase/ADMBase/documentation.html).
- O. Porth et al., [The Event Horizon General Relativistic Magnetohydrodynamic
  Code Comparison Project](https://arxiv.org/abs/1904.04923), for the additional
  numerical scope required when a luminous plasma is introduced.
