# Stationary Kerr remnant reference

This document defines the physics, data flow, validation boundary, and
reproduction procedure for the opt-in Kerr transfer-map reference:

```text
?scene=transfer-map-reference&reference=kerr-remnant
```

The product is a project-generated, stationary, analytic vacuum reference. It
uses the final dimensionless spin magnitude recorded for `SXS:BBH:0001`, but
it does **not** use an SXS near-zone metric and its pixels are **not numerical
relativity ray tracing**. It contains no accretion disk, plasma, emissivity,
absorption, polarization, GRMHD, or binary time evolution.

## Physical configuration

Geometric units are used throughout:

```text
G = c = M = 1
a / M = 0.686461676493
```

Here `M` is the mass parameter of the stationary Kerr spacetime. The spin axis
is world `+Z`. The magnitude is computed from the complete pinned
`SXS:BBH:0001/Lev5` remnant vector already authenticated by the Phase 2 binary
asset. The manifest declares the rigid spatial alignment that maps that vector,
including its small transverse components, to `+Z`; it does not silently
truncate those components. SXS supplies this parameter only, not the metric or
ray data.

The future event-horizon radius in Boyer-Lindquist coordinates is

```text
r+ = M + sqrt(M² - a²) = 1.727165982913406 M.
```

Capture is recorded at the declared stretched horizon
`r = r+ + 0.02M = 1.747165982913406M`. This is a constant-Kerr-radius oblate
worldtube, not a Euclidean sphere.

## Observer and camera

The observer is a zero-angular-momentum observer (ZAMO) at Boyer-Lindquist
`r = 40M` in the equatorial plane. Its four-velocity is

```text
u = alpha^-1 (partial_t + omega partial_phi),
```

where `alpha` and `omega` are evaluated from the exact Kerr metric. The local
camera basis is:

```text
right   = -e_phi
up      = -e_theta
forward = -e_r.
```

The camera looks radially inward with a 40-degree vertical field of view and a
1024×576 pixel-centre projection. The finite-distance ZAMO frame matters: an
asymptotic Bardeen screen formula cannot be substituted without changing the
shadow position and scale.

The manifest stores the observer event, metric, four-velocity, and tetrad in
ingoing Cartesian Kerr-Schild coordinates. Its affine camera matrices exist
only for deterministic playback coordinates; physical initial rays are formed
from the observer tetrad.

## Offline null-geodesic production

For every pixel, the generator constructs a past-directed local null vector

```text
k^(a) = (-1, normalize(screenX, screenY, 1))
```

and derives the conserved energy `E`, axial angular momentum `Lz`, and Carter
constant `Q`. It integrates the separated Kerr Hamilton-Jacobi equations in
Mino time using float64 adaptive Dormand-Prince 5(4). Ingoing Kerr-Schild time
and azimuth are evolved with the exact Boyer-Lindquist-to-Kerr-Schild
differential transformation.

The primary trace ends when a ray either:

- crosses the stretched horizon and is classified `captured`;
- crosses the constant-Kerr-radius `r = 1000M` boundary outward and is
  classified `escaped`; or
- fails a declared convergence gate and is classified `unresolved`.

For an escaped ray, coordinate lookback time and the frequency factor

```text
g = (u_observer · k_observer) / (u_boundary · k_boundary)
```

are measured at `r = 1000M` using a boundary Boyer-Lindquist ZAMO. The angular
trajectory is then numerically continued through the exact stationary Kerr
equations to `u = 1/r = 0`; the resulting asymptotic direction is rotated into
ICRS and stored. The finite boundary direction is not mislabeled as the
direction at infinity.

Each production ray is also traced with a second tolerance. Outcome
disagreement or a full-ray endpoint difference above 0.25 pixel fails closed
instead of being hidden by the adaptive integrator's local error estimate.
Runtime interpolation is disabled across the capture separatrix.

## Runtime and 32-byte ABI

The browser selects the Kerr product only through a fixed, in-source registry.
The exact manifest bytes, sidecar, every chunk, record layout, outcome
fractions, and numerical gates are authenticated before GPU allocation. A URL
parameter cannot supply an arbitrary manifest URL or digest.

Each little-endian record is exactly 32 bytes:

```text
3 × float32  escapeDirectionICRS
1 × float32  frequencyShiftG
1 × float32  coordinateLookbackTimeM
1 × float32  nullResidual
1 × float32  projectionErrorPx
1 × uint8    rayOutcome
1 × uint8    captureTarget
1 × uint16   validityMask
```

The workbench can display the lensed sky, categorical outcomes, lookback time,
frequency factor, null residual, or projection error. Clicking a texel exposes
the decoded values and the original 32 bytes. Diagnostic colours are display
tools; they do not alter the stored geodesics.

## Independent validation

Validation is deliberately split into three layers:

1. `verify_nr_contract.py` checks strict JSON/schema conformance, hashes,
   coordinate transforms, observer tetrad normalization, tiling, ABI records,
   and declared outcome fractions.
2. `verify_kerr_transfer_map.py` does not import the generator. It reconstructs
   the Cartesian Kerr-Schild metric and transformed BL-ZAMO tetrad, evaluates
   the finite-distance spherical-photon-orbit shadow, checks every capture
   texel, and integrates selected complete rays with a fixed-step RK4 method.
3. Browser tests exercise both WebGPU and WebGL2 playback, trusted reference
   selection, diagnostics, record inspection, and fail-closed recovery.

The independent physics verifier additionally checks:

- the analytic prograde/retrograde critical curve and vertical extremum;
- equatorial reflection symmetry;
- independent direction, frequency, and Kerr-Schild lookback values;
- separation of the `E`, `Lz`, and `Q` first integrals;
- null residual, full-ray tolerance refinement, and infinity-tail error.

Separate generator-level unit regressions compare `a -> 0` against the
independent Schwarzschild generator and compare complete small `+a` and `-a`
maps under the expected horizontal mirror. These are part of the validation
suite, not checks performed by the single-asset physics verifier.

Run the complete offline checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_nr_contract.py \
  assets/transfer-maps/kerr-remnant-reference-v1/manifest.json
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_kerr_transfer_map.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests/test_kerr_transfer_map.py
```

Regenerate the asset deterministically with:

```bash
python3 scripts/generate_kerr_transfer_map.py
```

## Bundled v1 measurements

The committed 1024×576 product contains 589,824 records in nine 2 MiB chunks:

```text
escaped   = 558684
captured  = 31140
unresolved = 0
unusable   = 0
```

The production run recorded:

```text
maximum null residual              = 3.068e-9
p95 stored projection estimate     = 1.929e-4 px
maximum stored projection estimate = 3.752e-3 px
generation time on the M3 Pro      = 1186.79 s
```

The independent verifier found zero analytic capture-mask mismatches and:

```text
maximum independent direction error = 8.679e-9 rad
maximum independent frequency error = 5.713e-8
maximum independent KS lookback error = 5.982e-5 M
fixed-step h/h2 direction difference = 3.401e-14 rad
maximum E/Lz/Q separation residual = 3.155e-12
boundary Richardson error = 1.168e-5 rad
Kerr-Schild metric / ZAMO / spin-binding error = 0
```

The exact manifest trust root is
`5b0022ab963c0cc35d3d8acab17190bd1294bc72da2b49003d785f964ac81d99`.
Generation time is an operational observation, not a scientific acceptance
threshold; the recorded residual and independent-comparison gates are the
acceptance evidence.

## Scientific boundary

This reference demonstrates stationary strong-field frame dragging, a
spin-displaced critical curve, capture classification on the declared oblate
stretched-horizon worldtube, and consistent sky mapping for one camera. It
does not make the binary scene's weak-field shader more accurate and it does
not reconstruct the time-dependent merger spacetime. A complete binary
slow-light implementation still requires a four-dimensional near-zone
numerical metric, gauge-aware interpolation, time-dependent horizons, and rays
evolved through changing source time.

## References

- B. Carter, “Global Structure of the Kerr Family of Gravitational Fields,”
  *Physical Review* 174, 1559–1571 (1968),
  [DOI](https://doi.org/10.1103/PhysRev.174.1559).
- J. M. Bardeen, W. H. Press, and S. A. Teukolsky, “Rotating Black Holes:
  Locally Nonrotating Frames, Energy Extraction, and Scalar Synchrotron
  Radiation,” *The Astrophysical Journal* 178, 347–370 (1972),
  [DOI](https://doi.org/10.1086/151796).
- O. James et al., “Gravitational lensing by spinning black holes in
  astrophysics, and in the movie Interstellar,” *Classical and Quantum
  Gravity* 32, 065001 (2015),
  [DOI](https://doi.org/10.1088/0264-9381/32/6/065001).
- The pinned [`SXS:BBH:0001` Lev5
  record](https://doi.org/10.5281/zenodo.3273935), used only for the remnant
  spin parameter and its provenance.
