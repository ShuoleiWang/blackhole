# NR transfer-map protocol v1

This document defines the repository's implemented data boundary for future
camera-specific, vacuum, slow-light transfer maps. The schema discriminator is
`blackhole.nr-transfer-map/v1`.

> **Status:** the schema, deterministic synthetic fixture, fail-closed
> validator, and regression tests are implemented. No numerical-relativity
> spacetime, NR-derived transfer map, runtime decoder, or NR playback scene is
> bundled. The default Schwarzschild renderer and the opt-in
> `?scene=binary-approx` preview are unchanged.

## Scientific claim levels

The following terms have intentionally different meanings:

| Term | Required evidence |
| --- | --- |
| **Contract-conformant** | The manifest, sidecars, chunks, coordinate declarations, and records pass the v1 validator. |
| **NR-ready ingestion boundary** | The protocol can represent and fail-closed validate a future offline NR-derived product. This describes the interface, not the repository's current data or renderer. |
| **NR-backed** | The actual displayed payload was derived from a pinned numerical-relativity near-zone spacetime by a documented slow-light null-geodesic pipeline. |
| **Physically validated** | In addition to being NR-backed, the spacetime, geodesics, transfer-map interpolation, and stationary limits pass declared convergence and error gates. |

Protocol conformance is necessary for future ingestion, but it is not evidence
that Einstein's equations were solved or that a ray, merger, or image is
physically correct. A browser that eventually plays an NR-derived transfer map
will still be a playback/composition layer, not an NR solver.

The existing `binary-approx` scene is more precisely described as a
**PN/phenomenological weak-field preview with rounded remnant reference values
from `SXS:BBH:0001`**. It does not use SXS waveform, horizon, spacetime, or
ray-transfer data.

## Repository entry points

- [`schemas/nr-transfer-map-v1.schema.json`](../schemas/nr-transfer-map-v1.schema.json)
  is the Draft 2020-12 machine-readable manifest schema.
- [`assets/transfer-maps/contract-fixture-v1/manifest.json`](../assets/transfer-maps/contract-fixture-v1/manifest.json)
  is a deterministic, non-renderable conformance fixture.
- [`scripts/generate_nr_contract_fixture.py`](../scripts/generate_nr_contract_fixture.py)
  recreates the fixture, its binary chunk, and manifest sidecar.
- [`scripts/verify_nr_contract.py`](../scripts/verify_nr_contract.py) performs
  strict JSON/schema, integrity, frame, sampling, and record checks.
- [`tests/test_nr_contract.py`](../tests/test_nr_contract.py) covers accepted
  input and adversarial rejection cases.
- [`binary-model.md`](./binary-model.md) explains how the contract fits into the
  longer NR → slow-light geodesic → browser playback architecture.

Generate and validate the project fixture with:

```bash
python3 scripts/generate_nr_contract_fixture.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_nr_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_nr_contract.py
```

The validator uses
`assets/transfer-maps/contract-fixture-v1/manifest.json` by default. A producer
can validate a separate dataset without copying it into that location:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_nr_contract.py /path/to/manifest.json
```

Success means `protocol-conformant` only.

## Dataset kinds and renderability

`datasetKind` has three closed values:

| Value | Meaning |
| --- | --- |
| `synthetic-contract-fixture` | Project-generated sentinel data for exercising the format and validator. It must be non-renderable, non-NR, synthetic, and physically unmeasured. |
| `stationary-reference-transfer-map` | A stationary black-hole reference product for analytic or high-accuracy comparisons. |
| `nr-slow-light-transfer-map` | A product derived from a catalogued NR near-zone spacetime using time-dependent slow-light ray integration. |

An `nr-slow-light-transfer-map` must identify a catalog simulation and DOI,
classify its physical system as `binary-black-hole`, declare
`vacuum=true` and `spacetimeMode=time-dependent`, include pinned
`near-zone-metric` and `horizon-data` source artifacts, supply positive
integration tolerances and a mass-normalization source epoch, and set all three
`scientificStatus` facts:

```text
sourceIsNumericalRelativity = true
derivedFromNearZoneSpacetime = true
derivedWithSlowLightGeodesics = true
```

Both non-fixture dataset kinds require measured accuracy; setting
`renderable=false` cannot be used to omit scientific metadata. Conversely,
`renderable=true` is a fail-closed playback gate, not a claim of mathematical
exactness. Declared unresolved/outcome fractions must agree with the binary
records, and at least one ray must be resolved as escaped or captured. A
consumer must still apply independent scientific quality thresholds.

The shipped fixture has:

```text
id = nr-contract-fixture-v1
datasetKind = synthetic-contract-fixture
renderable = false
```

Its one 4×2 chunk contains eight sentinel records that cover all six terminal
outcomes. It has no SXS/NR metric, waveform, horizon, or geodesic payload and
must never be presented as an image.

## Required manifest sections

Every top-level field is required and unknown fields are rejected.

| Section | Contract |
| --- | --- |
| `schema`, `id`, `datasetKind`, `renderable` | Exact discriminator, stable identity, dataset class, and playback gate. |
| `scientificStatus` | Source-NR, near-zone derivation, and slow-light derivation booleans plus description and prohibited claim. |
| `physicalSystem` | System/vacuum class, component IDs, parameter epoch, mass ratio, component spins, eccentricity, orbital phase, remnant, and non-applicability reason. |
| `provenance` | Source simulation and evolution code, generator revision/command, license, artifact storage/base, hashes, and sizes. |
| `units` | Geometric units and a defined, epoch-qualified mass normalization. |
| `timeReference` | Source/protocol time relation, future direction, zero event, and waveform-time mapping. |
| `coordinates` | NR chart/gauge/slicing, spatial NR↔world affine maps, and proper world↔ICRS rotations and axes. |
| `observer` | Source events, protocol/proper times, covariant and inverse metrics, four-velocity, and orthonormal tetrad at each sample. |
| `camera` | Fixed spatial affine visualization frame; physical initial rays use the observer tetrad. |
| `projection` | Pixel-centre, field-of-view, screen-coordinate, and past-directed local-ray formulas. |
| `sampling` | Protocol observation times, order, tiling, and validity-strict interpolation. |
| `rayIntegration` | Spacetime mode/interpolation, integrator, tolerances, precision, normalization, and termination semantics. |
| `escapeBoundary` | Outer surface, boundary observer, frequency-shift convention, and declared ICRS continuation. |
| `recordLayout` | Exact 32-byte ABI, observable definitions, outcomes, validity bits, and invalid-float policy. |
| `captureTargets` | Surface IDs/codes, validity intervals, classification priority, and source role. |
| `accuracy` | Convergence/error status, unresolved and outcome fractions, and fixture assertions. |
| `integrity`, `chunks` | Manifest sidecar and ordered tiled payload URIs, sizes, counts, and hashes. |

The JSON Schema fixes the serializable shape. The Python validator additionally
enforces cross-field and binary-payload invariants that JSON Schema alone
cannot express.

## Units and source/protocol time

The v1 protocol uses geometric units:

```text
G = c = M = 1
```

The mass normalization `M` must give a textual definition and a
`referenceEpochSourceM`. Supported quantities are:

- `synthetic unit mass`;
- `initial ADM total mass`;
- `reference-epoch total Christodoulou mass`;
- `stationary black-hole mass`.

For a time-varying binary, a reference-epoch Christodoulou mass must identify
the source-coordinate epoch at which the component horizon masses are summed.
Length and coordinate time are then reported in this declared `M`.

An NR binary must also populate `physicalSystem` at a declared protocol epoch:
`massRatioQ ≥ 1`, non-negative eccentricity, reference orbital phase, one
dimensionless spin vector per component, and remnant mass fraction/spin.
Component and remnant spin magnitudes cannot exceed one. The synthetic fixture
keeps these fields null/empty and supplies an explicit non-applicability reason.

Source simulation time and protocol playback time are separate:

```text
t_protocol = t_source - sourceTimeAtProtocolZeroM
```

Both increase toward the future. `observer.samples[*].eventNr[0]` is source
coordinate time; `observer.samples[*].protocolTimeM` and
`sampling.observationTimesM[*]` are the same protocol time. Each sample also
records the observer's `properTimeM`, which must increase strictly. The
validator checks these relations sample by sample. `timeReference.zeroEvent`
explains the chosen origin, while `waveformTimeMapping` either states the
waveform synchronization convention or explicitly marks it not applicable.

## Spatial frames, ICRS, and observer tetrads

The NR chart includes `(t,x,y,z)`, but `nrToWorld`, `worldToNr`,
`cameraToWorld`, and `worldToCamera` are **spatial affine** 4×4 matrices acting
only on `[x,y,z,1]`. They are not spacetime-coordinate transformations and
must not be applied to time coordinates, four-velocities, wave vectors, or
tetrads. Time is handled by `timeReference`; physical ray initialization stays
in the supplied NR-coordinate observer tetrad.

The validator checks inverse spatial maps and representative
world→camera→world point round trips at a strict `1e-10` gate; the synthetic
fixture declares a tighter assertion. Each spatial 3×3 block must be a proper
rotation, so scaling, shear, and reflection are rejected. The camera is fixed
in v1, and every observer sample must remain anchored to its spatial origin; it
exists for deterministic image coordinates, not as a substitute for a physical
observer.

World/sky orientation uses mutually inverse proper right-handed 3×3 rotations.
ICRS axes are fixed as:

```text
+X = right ascension 0 degrees,  declination 0 degrees
+Y = right ascension 90 degrees, declination 0 degrees
+Z = ICRS north celestial pole
```

For metric signature `-+++`, each observer sample supplies mutually inverse
covariant and contravariant metrics, a future-directed unit four-velocity `u`,
and a contravariant tetrad ordered:

```text
[time, right, up, forward]
```

The validator requires orthonormality and `e_(time)=u`, including tangent-vector
round trips through the tetrad and metric.

## Projection, past-directed rays, and frequency shift

Pixel centres and the vertical field of view determine:

```text
screenX = ((x+0.5)/width*2-1) * aspect * tan(verticalFov/2)
screenY = (1-(y+0.5)/height*2) * tan(verticalFov/2)
k^(a)   = (-1, normalize(screenX,screenY,1))
```

`k` is past-directed. Under `-+++`, the contract normalizes it so that
`u_observer·k_observer=1`, not with the future-directed-photon
`-u·p` convention. The stored frequency factor is:

```text
g = (u_observer·k_observer) / (u_boundary·k_boundary)
```

The denominator uses the future-directed unit reference observer declared at
`escapeBoundary`; it must be positive. `frequencyShiftG` is therefore positive
and dimensionless.

`rayIntegration` records whether the spacetime is time-dependent, stationary,
or synthetic, together with the spatial/temporal metric interpolation,
integrator, tolerances, internal precision, and exact meaning of every terminal
outcome. A real NR product must use slow light: the ray samples the
time-dependent geometry at the source-coordinate time it reaches each point.

An escaped ray terminates at the declared `escapeBoundary` surface. The stored
`escapeDirection` is the outgoing unit direction **after** the producer's
declared continuation beyond that boundary, expressed in ICRS. The contract
does not silently assume that a finite coordinate sphere is already infinity.

## Capture surfaces and terminal outcomes

A capture target is a declared classification surface, not automatically an
event horizon. Each entry names its `surfaceKind`, protocol-time validity
interval, classification priority, and optional source-artifact role. For a
future NR product that source role must be `horizon-data`, identifying the
apparent-horizon or world-tube data used to construct the surface. A captured
record is valid only when its target exists at that protocol time; unique
priority values make overlapping classification deterministic.

Terminal outcomes have exact meanings:

| Code | Outcome | Termination |
| ---: | --- | --- |
| 0 | `escaped` | Intersected `escapeBoundary`. |
| 1 | `captured` | Intersected a declared `captureTargets` surface. |
| 2 | `unresolved` | Exhausted the step or affine-parameter budget. |
| 3 | `outside-domain` | Left the declared spacetime domain somewhere other than `escapeBoundary`. |
| 4 | `integrator-failure` | Encountered a non-finite state or tolerance failure. |
| 255 | `missing` | The record was not generated. |

Only `escaped` may sample the sky. All other states remain explicit.

## Binary record ABI

Each pixel record is exactly 32 bytes with Python struct format `<7fBBH`:

| Offset | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 0 | `escapeDirection` | `3 × float32` | Unit ICRS direction after the declared escape-boundary continuation. |
| 12 | `frequencyShiftG` | `float32` | Positive past-directed frequency factor defined above. |
| 16 | `coordinateLookbackTimeM` | `float32` | `t_observer_protocol - t_terminal_protocol ≥ 0`; gauge-dependent coordinate lookback, not a physical relative arrival-time delay. |
| 20 | `nullResidual` | `float32` | Maximum `|g_mu_nu k^mu k^nu|` after observer-frequency normalization. |
| 24 | `projectionErrorPx` | `float32` | Estimated image-plane displacement under geodesic/interpolation refinement. |
| 28 | `rayOutcome` | `uint8` | Terminal outcome code. |
| 29 | `captureTarget` | `uint8` | Declared target code for captured rays; `255` otherwise. |
| 30 | `validityMask` | `uint16` | Authoritative per-field validity bits. |

The outcome state machine fixes the mask and target exactly:

| Outcome | Target | Mask | Valid float fields |
| --- | ---: | ---: | --- |
| `escaped` | `255` | `0x1f` | direction, `g`, lookback, null residual, projection error |
| `captured` | declared target | `0x1c` | lookback, null residual, projection error |
| `unresolved` | `255` | `0x18` | null residual, projection error |
| `outside-domain` | `255` | `0x1c` | lookback, null residual, projection error |
| `integrator-failure` | `255` | `0x08` | null residual |
| `missing` | `255` | `0x00` | none |

Validity bits 0–4 correspond respectively to `escapeDirection`,
`frequencyShiftG`, `coordinateLookbackTimeM`, `nullResidual`, and
`projectionErrorPx`. Unknown bits are rejected. Every float is finite; a float
whose bit is clear must be canonical positive zero. The mask, never a stored
sentinel value, is authoritative.

## Sampling, chunks, and outcome fractions

Chunks cover each declared time/image plane in monotonically ordered,
non-overlapping, gap-free row-major tiles. Each chunk's byte length is:

```text
recordCount × 32
```

Protocol observation times are strictly increasing and align one-to-one with
observer and chunk sample indices. Continuous interpolation is allowed only
when the relevant validity bit is present at every contributor. Escape
directions are then renormalized; categorical fields use nearest/no-blend.
Invalid, missing, unresolved, failed, or out-of-domain records never fall back
to the sky.

`accuracy.outcomeFractions` reports each decoded outcome fraction plus:

```text
unusable = unresolved + outside-domain + integrator-failure + missing
```

For the non-renderable synthetic fixture all fractions are `null`; its expected
counts are fixture assertions. A renderable product must provide finite
fractions that sum consistently and match the decoded payload, as well as a
matching `unresolvedFraction`.

## Provenance, artifact bases, and integrity

Each source artifact declares `storage`:

- `bundled` means the file is locally hashable;
- `external-reference` preserves a pinned HTTPS or DOI URI, byte size, and
  SHA-256 without pretending the file is bundled.

`artifactUriBase=repository-root` supports repository-native fixtures such as
the one shipped here. `artifactUriBase=manifest-directory` supports portable
dataset bundles whose local artifact paths travel with the manifest. Chunk URIs
are always relative to the manifest directory. Local paths must be normalized,
relative, traversal-free, and non-symlinked.

The exact manifest bytes are covered by `manifest.sha256`; bundled source
artifacts and chunks are checked against their declared byte sizes and SHA-256
hashes. The generator URI must name the `generator-source` artifact and its
`codeRevision` must bind that artifact's SHA-256.

## Fail-closed behavior

The validator rejects ambiguity, including:

- duplicate JSON keys, `NaN`, `Infinity`, booleans used as numbers, missing or
  unknown fields, and unknown schema versions;
- unsafe local paths, illegal storage/base combinations, symlinks, missing
  bundled artifacts, and size/hash/sidecar drift;
- non-monotonic or inconsistent source/protocol times;
- singular spatial affine maps, improper ICRS rotations, failed point round
  trips, non-timelike observer velocities, and invalid tetrads;
- incomplete/overlapping/out-of-order tiles or inconsistent record sizes;
- undeclared or inactive capture surfaces and ambiguous classification priority;
- non-finite records, non-unit directions, non-positive `g`, negative coordinate
  lookback, unknown bits/codes, illegal outcome/target/mask combinations, and
  decoded fractions that disagree with accuracy metadata.

These rules are stricter than what is needed merely to display pixels.
Corrupted or scientifically ambiguous input stops at the offline boundary
instead of producing a plausible but unexplained image.

## Requirements for a future NR-derived dataset

Before a dataset can be described as NR-backed, its producer must at minimum:

1. pin the source simulation, evolution-code name/release and commit (or an
   explicit reason the commit is unavailable), DOI, license, near-zone metric
   and horizon artifacts, generator revision, and exact command;
2. define mass ratio, component spins, eccentricity, phase, remnant, the
   source/protocol parameter epoch and mass reference epoch, NR gauge/slicing,
   observer tetrads, spatial frames, projection, and ICRS orientation;
3. document the time-dependent spacetime interpolants, slow-light integrator,
   tolerances, escape boundary/continuation, and capture surfaces;
4. publish immutable chunks with explicit invalid outcomes and decoded outcome
   fractions;
5. report NR convergence, constraint norms, ray null residuals, interpolation
   error, unresolved fraction, and stationary Schwarzschild/Kerr comparisons;
6. pass the protocol validator and independent scientific convergence gates.

A future runtime decoder must refuse `renderable=false`, preserve validity and
outcome states through interpolation, and keep image-regression tests separate
from physics tests. Until real NR data and that consumer exist, the protocol is
an implemented architecture boundary rather than an NR rendering feature.
