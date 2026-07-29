# Binary black-hole model

This document defines the scientific scope of the real-time binary black-hole
scene at the root URL. The legacy `?scene=binary-approx` URL remains an alias;
the interactive single-hole renderer is explicit at `?scene=schwarzschild`.
This document is intentionally stricter than a visual feature list: a
convincing image is not, by itself, evidence that the underlying spacetime is a
solution of Einstein's field equations. The repository-wide product layers and
two development routes are defined in
[`rendering-modes.md`](./rendering-modes.md).

## Status at a glance

| Component | Phase 2 implementation | Scientific status |
| --- | --- | --- |
| Binary configuration | `SXS:BBH:0001` Lev5, equal mass, effectively non-spinning, quasi-circular | Pinned numerical-relativity source |
| Orbital dynamics | Separation and phase from the actual A/B apparent-horizon inertial-coordinate centroids | NR-derived diagnostics, but coordinate- and gauge-dependent |
| Waveform | CoM-corrected, extrapolated N=2 complex `(l,m) = (2,2)` strain mode | Real SXS far-zone waveform; not a near-zone metric |
| Merger events and remnant | Published common-horizon time plus exact metadata remnant mass and spin | Source-backed events/metadata; the visual topology blend remains a presentation proxy |
| Playback | Scrubbing, shared pause/resume, end hold/loop, and optional merger slow motion | Deterministic presentation mapping; slow motion is not gravitational time dilation |
| Light propagation | Existing real-time, frame-frozen multi-centre weak-field bending | **Not NR ray tracing** and invalid as precision strong-field imaging |
| Emission | Lensed all-sky background in vacuum | No accretion disks, plasma, GRMHD, or radiative transfer |
| Display | WebGPU with WebGL2 fallback and the existing HDR pipeline | Display fidelity does not increase physical accuracy |
| Transfer-map interface | Versioned schema, synthetic fixture, fail-closed validators, plus fixed-camera stationary Schwarzschild and Kerr reference consumers | The consumer is proven with analytic data only; no NR-derived transfer map or binary slow-light playback |

The precise classification is **NR-driven dynamics with weak-field fast-light
rendering**. “NR-driven” applies to the orbital track, waveform, source events,
and remnant metadata. It does not apply to light propagation. The scene does
not consume an SXS near-zone metric, a four-dimensional numerical spacetime, or
an NR-derived ray-transfer payload. The rendered image must not be described as
NR ray tracing, a solved binary-spacetime image, or a quantitatively accurate
merger shadow.

Mouse or touch input changes the camera, and every rendered frame rebuilds its
camera rays and recomputes the GPU lens result. This makes the scene
interactive; it does not strengthen the weak-field or fast-light physics
classification.

The runtime manifest is
[`binary-sxs-bbh-0001-v2.json`](../assets/scenes/binary-sxs-bbh-0001-v2.json);
its compact sidecar is
[`binary-sxs-bbh-0001-v2.samples.json`](../assets/scenes/binary-sxs-bbh-0001-v2.samples.json).
Both repeat the scientific boundary so a consumer cannot silently promote
NR-derived motion into an NR-derived image claim.

## Phase 2 SXS-driven dynamics

### Pinned source and quantities

The deterministic offline generator consumes exactly three official files from
the pinned [`SXS:BBH:0001` Zenodo record](https://doi.org/10.5281/zenodo.3273935):

- `Lev5/metadata.json`;
- `Lev5/Horizons.h5`; and
- `Lev5/rhOverM_Asymptotic_GeometricUnits_CoM.h5`.

Their official URLs, byte sizes, MD5 values, and SHA-256 values are recorded in
[`assets/SOURCES.md`](../assets/SOURCES.md). The pinned Zenodo record does not
declare a license, so this project records `spdx = null` and does not invent or
infer a license.

The current SXS catalog marks `SXS:BBH:0001` as deprecated and superseded by
the longer `SXS:BBH:1132` simulation. Phase 2 intentionally pins the archived
`0001/Lev5` bytes for a deterministic first integration; it never asks a client
library to auto-supersede the source behind the manifest. Migrating the fixture
to a pinned `SXS:BBH:1132` version and resolution is a data-source upgrade, not
evidence that the current weak-field renderer has become NR ray tracing.

The track uses geometric units,

```text
G = c = M = 1,
```

where `M` is the sum of the two Christodoulou masses at the metadata relaxation
time. For each common A/B horizon sample, the generator forms

```text
Δx(t) = x_B(t) - x_A(t)
a_coord(t) = |Δx(t)| / M
φ_coord(t) = unwrap(atan2(Δx_y, Δx_x)).
```

The first retained phase is shifted to zero. These values come from
`AhA.dir/CoordCenterInertial.dat` and
`AhB.dir/CoordCenterInertial.dat`; they are actual SXS apparent-horizon
coordinate-centroid diagnostics, not a PN fit. They are nevertheless
**gauge-dependent coordinates**, not invariant proper separation or invariant
orbital phase.

The complex waveform channels come from
`Extrapolated_N2.dir/Y_l2_m2.dat` in the CoM-corrected asymptotic waveform
file. The displayed strip is `Re[r h22 / M]`, while the runtime retains both
real and imaginary channels and computes `|h22|`.

### Time origin, source events, and remnant

Protocol `t = 0` is the maximum amplitude of the CoM-corrected
`Extrapolated_N2 h(2,2)` mode. The bundled range is
`-9210.155252 M <= t <= 120 M`. Key anchors are:

| Event | Protocol time |
| --- | ---: |
| Metadata relaxation time / first bundled horizon sample | `-9210.155251999806 M` |
| Last paired A/B centroid sample | `-6.158268002352997 M` |
| First common apparent horizon from metadata | `-6.072285420526896 M` |
| Maximum `|Extrapolated_N2 h22|` | `0 M` |
| Configured ringdown display endpoint | `120 M` |

The horizon coordinate time and extrapolated-waveform retarded time are
different source coordinates. The generator retains their published numeric
origins and subtracts the waveform peak from both. The apparent ordering above
is useful for deterministic playback, but it is not a gauge-invariant
light-travel-time measurement or a proof that events in the two source
coordinates are physically simultaneous.

The exact Lev5 metadata remnant values are:

```text
M_f / M = 0.951609417715
χ_f = (-7.29520687012e-10,
        7.40468371215e-10,
        0.686461676493)
```

The manifest also records the last common-horizon diagnostic and checks that it
agrees with the metadata mass and spin to better than `0.1%`. The render shader
uses the mass fraction for its spherical remnant proxy but does not render the
spin vector, Kerr geometry, frame dragging, recoil, or spin-dependent shadow
shape.

### Compact track and interpolation

The sidecar contains 2,732 adaptively selected, strictly time-ordered samples
in 202,606 bytes (approximately 198 KiB). It stores:

```text
t_protocol, a_coord, φ_coord, Re(h22), Im(h22),
renderTopologyBlend, individualHorizonsValid
```

Piecewise-linear reconstruction is measured against the dense source track.
The largest orbital-phase residual is `0.000644202687 rad`
(`6.442e-4 rad`), below the declared `0.003 rad` bound. The other measured
maximum residuals and their declared bounds are recorded in the manifest and
checked by `scripts/verify_binary_dynamics.py`.

The A/B centroids end before the common-horizon metadata event. Once
`individualHorizonsValid` becomes false, the sidecar holds the last valid
separation and phase rather than inventing post-horizon A/B trajectories. A
smoothstep from the common-horizon event to the waveform peak removes the
two-centre visual proxy. That `renderTopologyBlend` is explicitly a
**presentation quantity, not a horizon observable**.

## Scrubbing, pause, and merger slow motion

The waveform panel exposes a real protocol-time range input, a local
play/pause button synchronized with the header button and Space key, and a
merger slow-motion toggle. Scrubbing clamps to the bundled source interval,
updates the waveform cursor and physical readouts while paused, and resumes
only if playback was running before the drag.

By default, `-160 M <= t < 70 M` plays at `0.12×` the selected base rate. The
clock segments a frame at rate boundaries so crossing the slow zone is
frame-rate independent. Playback holds the `t = 120 M` endpoint for 2.5 wall
seconds and then loops.

The slow-motion factor changes only the mapping from wall-clock seconds to
protocol time. It does not alter sample values, retime the source data, or
represent gravitational time dilation. The UI displays the actual effective
rate to keep this distinction visible.

## Light propagation remains weak-field fast-light

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

The camera ray and two positions are reconstructed for every rendered frame.
The positions are then frozen at the current timeline sample for the duration
of each ray integration (the `fast-light` approximation). This is GPU
re-tracing from the current camera, not fixed-camera transfer-map playback.
The shader does not evaluate retarded positions or let one ray traverse a
changing binary metric; it also omits velocity-dependent and gravitomagnetic
terms.

For one isolated mass and a large impact parameter this has the expected
leading deflection scale `4M/b`. It is not a valid strong-field metric near
either horizon, and the sum of two potentials is not a binary solution of
Einstein's equations. Capture surfaces and the transition from two dark
objects to one remnant are consequently visualization boundaries, not a
computed event-horizon world tube.

The metadata-backed remnant spin is retained in the data contract but is not
rendered. Phase 2 continues to use the original spherical monopole and
spherical `r = 2M` capture proxy, with no Kerr geometry, frame dragging, or
spin-dependent shadow shape.

The practical implications are important:

- The broad distortion of the Milky Way and the qualitative tightening of the
  binary can be interpreted as a physically motivated preview.
- Fine photon rings, higher-order images, caustic magnifications, arrival-time
  delays, and the shape or topology of a common horizon are not quantitatively
  reliable.
- A frame must not be used to infer masses, spins, orbital parameters, lensing
  cross sections, or gravitational-wave observables.
- HDR preserves highlight range and colour on supported displays; it does not
  repair missing spacetime physics.

The scene models a vacuum binary and intentionally does not place a bright
accretion disk around each object. A physically defensible luminous merger
would additionally require an initial gas configuration, general-relativistic
magnetohydrodynamics (GRMHD), electron-temperature and emissivity prescriptions,
and time-dependent polarized radiative transfer.

In short: the motion and waveform are now NR-derived; the rendered rays are
still weak-field fast-light. A source-backed orbit does not turn the shader
into NR ray tracing.

## Legacy PN regression asset

[`binary-pn-equal-mass-v1.json`](../assets/scenes/binary-pn-equal-mass-v1.json)
contains the earlier leading-order PN inspiral and phenomenological
merger/remnant preview. It is no longer loaded by the runtime scene. It remains
in the repository so `scripts/verify_binary_preview.py` can independently
regress the old manifest equations, explicit non-NR safety flags, parameter
bounds, and the unchanged weak-field shader's 512-step CPU convergence gate.

Keeping the legacy asset separates two questions:

1. Did Phase 2 preserve the established renderer behavior?
2. Does the new SXS-derived dynamics track satisfy its own source and playback
   contract?

Neither regression turns the weak-field shader into NR light propagation.

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

## Scene routing and isolation

The root URL selects the real-time binary scene. The legacy URL remains
compatible:

```text
http://localhost:4173/?scene=binary-approx
```

The existing single, non-rotating Schwarzschild renderer and its geodesic
shader, observer model, disk model, controls, and validation suite remain
available at:

```text
http://localhost:4173/?scene=schwarzschild
```

The separate
[`?scene=transfer-map-reference`](../README.md#url-parameters) path consumes
fixed 1024×576 stationary analytic Schwarzschild or Kerr vacuum maps. They have
no accretion disk and no relationship to the SXS binary pixel path; their
purpose is to provide analytic calibration, authenticated transfer-map
delivery, and regression oracles. See
[`nr-transfer-map-v1.md`](./nr-transfer-map-v1.md) for its exact claims,
nearest-texel sampling rule, and validation metrics.

Each route owns its scene descriptor, shader assumptions, controls, and
validation. Scene switching uses a page reload rather than mutating GPU
pipelines in place, and scene lifecycles restore shared UI state when disposed.
Changing which route owns the root URL does not merge their physical models.

The validation suites have different meanings:

- `python3 scripts/verify_physics.py` checks selected Schwarzschild invariants
  and capture/escape behavior.
- `python3 scripts/verify_binary_dynamics.py` checks the pinned SXS source,
  generated sidecar, events, remnant metadata, interpolation bounds, playback
  declarations, and the explicit weak-field renderer boundary.
- `node --test tests/binary-playback.test.mjs` checks source anchors,
  interpolation, scrub/seek behavior, end hold/looping, slow motion, and
  frame-rate independence.
- `python3 scripts/verify_binary_preview.py` retains the legacy PN and
  weak-field shader convergence regression. It does **not** certify NR light
  propagation or merger imaging.

## Two rendering routes and current phase

The project now has two explicit, complementary routes.

### Real-time interactive route

```text
mouse / touch / timeline input
        ↓
camera event and local tetrad
        ↓
fresh rays for every rendered frame
        ↓
WebGPU/WebGL ray or lens integration
        ↓
sky composition and HDR/SDR display
```

The implemented binary scene follows this route with weak-field, frame-frozen
fast-light bending. The next intended physics layer is an isolated
strong-field WebGPU geodesic tracer backed by a declared approximate binary
metric. That layer is not implemented. Even when its numerical integration is
accurate, a superposed or matched analytic metric must still be described as an
approximate fast-light model rather than NR ray tracing.

### High-fidelity offline route

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

“Slow light” is essential for NR-backed merger pixels: every ray samples the
metric at the coordinate time it reaches each integration point. Freezing one
NR slice for an entire ray can be useful diagnostically, but it does not
represent a rapidly changing merger. A luminous result additionally needs
physical matter fields and declared radiative-transfer assumptions; vacuum
lensing alone cannot determine colour or brightness.

The scientific master and browser delivery products are distinct. The browser
may consume a compact vacuum transfer-map proxy, HDR still, or video without
becoming the NR or GRRT solver.

The project status is deliberately reported by layer:

| Layer | Status | Permitted claim |
| --- | --- | --- |
| Phase 1 transfer-map schema, fixture, validator, and tests | Implemented | `contract-conformant` ingestion boundary |
| Phase 2 SXS orbital dynamics, waveform, events, and remnant metadata | Implemented | `NR-driven dynamics` |
| Root interactive binary scene | Implemented | Per-frame GPU recomputation with weak-field fast-light pixels |
| Stationary Schwarzschild and Kerr reference maps and runtime consumer | Implemented | Analytic fixed-camera calibration, authenticated playback, diagnostics, and regression oracles; not NR |
| Interactive strong-field WebGPU binary metric and geodesics | Not implemented | No strong-field binary claim |
| Slow-light ray bundles through a time-dependent NR spacetime | Not implemented | No `NR-backed` pixel or image path |
| GRMHD plasma and spectral/polarized GR radiative transfer | Not implemented | No physically modelled luminous merger |
| Multilayer OpenEXR scientific master | Not implemented | No offline radiance master or associated convergence record |

The status words are not interchangeable:

- **Contract-conformant** means only that a dataset satisfies the versioned
  structural, integrity, coordinate-frame, and record-consistency rules.
- **NR-driven dynamics** means the displayed trajectories, waveform, events,
  and remnant parameters are derived from the pinned SXS simulation. It makes
  no claim about the pixel-generating light propagation.
- **Stationary-analytic-validated** means a fixed analytic Schwarzschild or
  Kerr product passes its independent geodesic, capture, refinement, and
  delivery gates. It is not evidence for a binary spacetime.
- **NR-ready** applies only to the ingestion boundary: it can reject or accept
  a future offline product without changing the protocol. It does not mean that
  such a product is bundled or that the browser solves Einstein's equations.
- **NR-backed vacuum lensing** may be used only when the actual rendered
  payload was derived from a pinned four-dimensional numerical-relativity
  spacetime, horizon data, and documented slow-light geodesic integration.
- **GRMHD/GRRT-backed radiance** additionally requires pinned matter fields and
  declared emission, absorption, optical-depth, spectral, and polarization
  transport.
- **Physically validated** additionally requires the convergence and
  comparison gates below. Passing a schema validator is not a physics result.

### Implemented transfer-map protocol boundary

The implemented delivery baseline is a versioned, immutable,
camera-specific vacuum escape-transfer map. It allows a browser to consume a
scientific endpoint product without embedding its generator. The v1 contract
uses the discriminator
`blackhole.nr-transfer-map/v1`; its normative semantics and safety rules are in
[`nr-transfer-map-v1.md`](./nr-transfer-map-v1.md), and its machine-readable
shape is in
[`schemas/nr-transfer-map-v1.schema.json`](../schemas/nr-transfer-map-v1.schema.json).
Every v1 top-level field is required and unknown fields are rejected:

| Field | Required meaning |
| --- | --- |
| `schema`, `id`, `datasetKind`, `renderable` | Exact protocol discriminator, stable identity, dataset class, and explicit playback gate |
| `scientificStatus` | Whether the source is NR and whether the payload was derived from near-zone spacetime with slow-light geodesics, plus prohibited claims |
| `physicalSystem` | System/vacuum class, component IDs, parameter epoch, mass ratio, component spins, eccentricity, reference phase, remnant, and non-applicability reason |
| `provenance` | Source simulation/catalog/version/DOI, evolution-code release and commit/reason, generator revision/command, license, artifact storage/base, hashes, and sizes |
| `units` | Geometric units and an explicitly defined mass normalization, including the source-time reference epoch when applicable |
| `timeReference` | The source-to-protocol time origin, future direction, zero event, and any waveform-time mapping |
| `coordinates` | NR gauge/chart/time slicing, spatial NR↔world affine maps, and proper world↔ICRS rotations and sky axes |
| `observer` | Source events, protocol/proper times, covariant/inverse metrics, four-velocity, and orthonormal tetrad samples |
| `camera` | A fixed spatial affine visualization frame and its inverse; it is not a four-dimensional coordinate transform |
| `projection` | Pixel-centre, image-origin, field-of-view, screen-coordinate, and past-directed local-ray conventions |
| `sampling` | Protocol observation times, dimension/pixel/tile order, and validity-strict interpolation |
| `rayIntegration` | Time-dependent/stationary/synthetic spacetime mode, interpolants, integrator, tolerances, precision, normalization, and terminal-state semantics |
| `escapeBoundary` | Outer surface, reference observer, past-directed frequency-shift convention, and continuation into the ICRS escape direction |
| `recordLayout` | Exact 32-byte ABI, observable definitions, outcomes, validity bits, and invalid-float policy |
| `captureTargets` | Capture-surface IDs/codes, surface semantics, protocol-time validity intervals, priority, and source artifact role |
| `accuracy` | NR/constraint/geodesic/interpolation status, unresolved and decoded outcome fractions, and fixture assertions |
| `integrity`, `chunks` | Manifest sidecar plus ordered tiled payload paths, sizes, counts, and hashes |

The v1 payload is an immutable vacuum escape-transfer ABI. It does not define
adaptive ray bundles, subpixel samples, geodesic deviation or Jacobi fields,
GRMHD radiation, optical depth, spectra, Stokes parameters, or Faraday
coefficients. High-fidelity offline work therefore requires separately
versioned ray-bundle and radiative-frame contracts; it must not change the
meaning or 32-byte binary layout of v1.

The legacy `blackhole.binary-scene/v1` timeline and current
`blackhole.binary-scene/v2` dynamics track are both distinct from the
`blackhole.nr-transfer-map/v1` format. Version 2 contains real NR-derived
horizon and waveform diagnostics, but its compact samples still do not contain
a near-zone spacetime or solved camera rays. Keeping these concepts separate
prevents “NR-driven dynamics” from being misreported as “NR ray tracing.”

The repository's
[`contract-fixture-v1`](../assets/transfer-maps/contract-fixture-v1/manifest.json)
is deliberately marked `datasetKind = synthetic-contract-fixture` and
`renderable = false`. It is small, deterministic, project-generated test data;
it contains no SXS/NR metric, waveform, horizon, or ray payload. The fixture
proves that the protocol tooling agrees with itself, not that the project has
completed an NR simulation.

The renderable
[`schwarzschild-reference-v1`](../assets/transfer-maps/schwarzschild-reference-v1/manifest.json)
is distinct from that fixture: it contains 589,824 analytic stationary rays in
nine chunks and is consumed at `?scene=transfer-map-reference`. It validates the
format-to-browser path but supplies no evidence for a time-dependent NR merger.

The separately selectable
[`kerr-remnant-reference-v1`](../assets/transfer-maps/kerr-remnant-reference-v1/manifest.json)
contains the same number of project-generated stationary vacuum rays for the
pinned remnant-spin magnitude. Its generator and independent verifier add a
finite BL-ZAMO, horizon-penetrating Kerr-Schild coordinates, an oblate
Kerr-radius capture surface, frame dragging, and an analytic critical-curve
oracle. It uses no SXS near-zone metric and is likewise not a merger frame.
Together, the Schwarzschild and Kerr products are stationary calibration and
delivery regressions for both future rendering routes.

The fail-closed validator rejects unknown schema versions and fields, missing
fields, duplicate JSON keys, non-finite values, booleans used as numbers,
unsafe or symbolic-link paths, sidecar hash/size mismatches, chunk
overlap/gaps/order errors, non-monotonic source/protocol sampling, invalid
spatial affine maps or ICRS rotations, invalid observer tetrads, inconsistent
integration/boundary/capture declarations, mismatched outcome fractions, and
illegal per-ray outcome/target/validity combinations. The stored
`coordinateLookbackTimeM` is a non-negative, gauge-dependent protocol-coordinate
difference, not a physical relative arrival-time delay. Invalid or unresolved
rays remain explicit data states; they must never be silently converted into
ordinary escaped sky samples.

Run the contract validation independently of the existing rendering checks:

```bash
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

The first command validates
`assets/transfer-maps/contract-fixture-v1/manifest.json` by default and accepts
an explicit manifest path for offline datasets. Neither command validates an NR
spacetime or geodesic physics.

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
- Temporal and spatial metric/interpolant error bounds and convergence of
  adaptive ray bundles, ray differentials, and Jacobi fields.
- Cross-checks of orbital phase, waveform, radiated energy/angular momentum,
  final mass, and final spin against the source simulation.
- For luminous output, convergence of spectral/subpixel sampling and explicit
  validation of the GRMHD, electron, emissivity, absorption, optical-depth, and
  polarization assumptions that actually determine radiance.
- A multilayer scientific master whose physical channels and audit manifest
  remain independent of tone mapping, gamut conversion, HDR mastering, and
  video encoding.
- Image-regression checks that are separated from physics checks so a tone-map
  change cannot masquerade as a physical change.

## References and data sources

- SXS Collaboration, [SXS Gravitational Waveform
  Database](https://data.black-holes.org/waveforms/catalog.html) and
  [documentation](https://data.black-holes.org/waveforms/documentation.html),
  plus the pinned [`SXS:BBH:0001` Lev5
  record](https://doi.org/10.5281/zenodo.3273935). Phase 2 derives coordinate
  separation/phase, complex `h22`, events, and remnant metadata from its
  official files. The record does not declare a license. Exact source URLs,
  sizes, MD5, and SHA-256 values are recorded in
  [`assets/SOURCES.md`](../assets/SOURCES.md).
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
