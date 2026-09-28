# Offline Kerr Renderer Checkpoint — 2026-08-10

Snapshot time: 2026-08-10 12:27 CST

This file is a resumable engineering/science checkpoint, not a publication
claim.  The worktree is intentionally uncommitted and unpushed.  Historical
products remain preserved, but source changes made after their publication make
them source-stale evidence rather than current products.

## Objective

Continue the local offline high-fidelity black-hole rendering architecture in
this order:

1. exact Kerr 4D null geodesics and finite-surface topology;
2. a Novikov–Thorne material/emission reference;
3. returning-radiation and slow-light transfer with honest convergence gates;
4. bounded, resumable execution and byte-exact product replay;
5. future NR/GRMHD adapters without overstating current physics.

Constraints remain: modular code, bounded resources, explicit evidence levels,
no commit, and no push.

## Current plan

1. **Freeze the transport/reduction cache split.**  Finish independent review
   of the latest budget-before-I/O and renderer plan-freeze fixes.
2. **Add a tracer-issued, process-local primitive capability.**  The forward
   kernel may consume a primitive freshly issued by the canonical tracer without
   running the public two-ray replay again.  Public tracer/verifier semantics
   remain unchanged.  Expected forward cost: four to two whole-ray traces per
   logical direction.
3. **Add an authenticated adapter for convergence v2.**  It must derive ordered
   cells, areas, fate columns, and maximum sample weights from a verified direct
   or cached kernel; caller-built summaries remain diagnostic-only.
4. **Run a cost-controlled real Kerr refinement.**  The first useful target is
   one annulus with rho=8, mu=16, psi=32.  It is the first examined current-rule
   grid whose maximum normalized sample weight is below 0.001.  Estimated cost
   before the two-ray optimization is roughly five hours at eight workers; after
   it, roughly 2.5 hours.  These are planning estimates, not guarantees.
5. **Use the v2 diagnostics to choose the next refinement.**  Do not blindly
   double all dimensions.  Refine the dimension that dominates proper-power,
   fate, phase, or support-flip error.
6. **Only after a real v2-qualified kernel exists**, rebuild the coupled thermal
   profile, spectral frame, CIE/linear/SDR derivatives, structural verification,
   live same-code replay, and attestation.
7. Defer substantial NR/GRMHD implementation until the Kerr/NT/returning baseline
   has a source-current, v2-qualified real product.  Preserve interfaces and
   scientific boundaries in the meantime.

## Implemented and independently exercised

### Kerr and finite-thickness transport

- exact Kerr/Kerr–Schild Hamiltonian null-ray integration;
- accepted-step event localization, capture/escape/worldtube termination, and
  multi-surface initial-contact topology including fast re-entry;
- finite-thickness upper/lower photospheres, actual off-equatorial emitter frame,
  signed emission/incidence cosines, frequency shift, and first-visible transfer;
- an independent fixed-step Boyer–Lindquist selected-ray oracle with an explicit
  same-physics/shared-scalar boundary;
- finite-thickness proper area from the induced spatial metric, not Euclidean
  annulus area.

### Novikov–Thorne and thermal coupling

- Page–Thorne/Novikov–Thorne local flux and colour-corrected Planck spectrum;
- exact binary64 overflow/subnormal boundary handling independent of ambient
  Decimal context;
- forward four-face returning-energy kernel, receiver-centred kernel, symmetry
  reduction, fixed-point thermal profile, thermal spectrum provider, and frozen
  process-local frame authority;
- explicit exclusions: no returning-radiation stress work F_S, no scattering or
  spectral redistribution, no solved atmosphere, no polarization, no GRMHD, and
  no independent geodesic oracle for same-code replay.

### Deterministic execution and products

- bounded direction jobs, resumable authenticated cache records, stable
  deterministic reduction order, source/runtime closure, and no-follow/stable
  artifact reads;
- separation of transport scientific identity from reduction configuration:
  convergence tolerances, symmetry tolerances, work budgets, and area quadrature
  may change while the same trace cache is reused; the resulting kernel and
  thermal scientific binding still contain the exact reduction configuration;
- finite spectral product, CIE XYZ, linear sRGB, RGB16 SDR quicklook, structural
  verification, same-process live frame replay, and a sibling live-replay
  attestation with honest false tiers for unreplayed fixed point/cache/rays;
- source-closure inclusion of package initializers and transitive local imports,
  exact module-origin gates, bounded I/O, and late-mutation fail-closed checks.

### Numerical hardening completed in the latest phase

- Gauss–Legendre binary64 cycle handling now covers all forward/receiver orders
  1..64 and finite-area internal orders 1..256 while leaving noncycling legacy
  nodes/weights bitwise unchanged;
- fate closure now binds exact per-direction ordinal and weight.hex coverage before
  binary64 representation correction;
- scale-aware convergence v2 compares proper-power coordinates
  `P[i,j] = A_receiver[i] K[i,j] / A_emitter[j]`, direct g2 columns, normalized
  column L1, significant cells, insignificant tail, sampled-zero support flips,
  fate TV/absolute/relative errors, and maximum normalized sample weight;
- convergence-v2 reports are factory-only, resource bounded, canonical, and
  exactly replayable.  They remain caller-summary diagnostics until the planned
  authenticated adapter is implemented.

## Latest verification state

- Gauss–Legendre kernel/receiver/area tests: 54/54 passed.
- Fate quadrature tests: 11/11 passed.
- Convergence-v2 tests: 6/6 passed after correcting the five historical grid
  sample-weight values.
- Jobs + cached-kernel + renderer tests before the latest review fixes: 83 tests
  passed, one sandbox multiprocessing test skipped.
- Cached-kernel + renderer tests after the latest review fixes: 67/67 passed.
- Latest P1 fixes still require one fresh independent read-only re-audit because
  that audit was interrupted when this checkpoint was requested.
- No new long-running Kerr grid was started after the source changes.

## Current real numerical evidence and its limit

The preserved one-annulus rho=4, mu=8, psi=8 run (historical product `-06`) is
useful regression evidence, but it is not the final scientific result.

- axisymmetric returning coefficient: approximately 0.03339715742234205;
- local thermal result: F0 approximately 6.63749e11 W/m2,
  Fin approximately 2.29332e10 W/m2, and Teff/T0 approximately 1.008528;
- legacy absolute/relative convergence policy accepted all four comparison grids;
- scale-aware v2 rejects all four: proper-power/g2 differences are approximately
  38.83% (half-rho), 39.96% (half-mu), 33.08% (half-psi), and 28.41% (phase);
- maximum normalized sample weights are approximately 0.012742 (full), 0.022125
  (half-rho), 0.021756 (half-mu), 0.025483 (half-psi), and 0.012742 (phase), all
  above the v2 limit of 0.001.

Therefore the current honest status is **policy-pass under the historical loose
gate, science-unconverged under v2**.

## Open work and blockers

### Immediate engineering work

- complete independent re-audit of the transport/reduction identity fixes;
- implement and attack-test the tracer-issued primitive capability;
- update forward work accounting and budgets from four to two whole rays per
  logical direction without changing public primitive replay semantics;
- add the authenticated convergence-v2 adapter;
- rerun the affected targeted and integrated suites on a frozen source tree.

### Scientific work

- obtain at least one source-current real kernel that passes v2 resolution and
  comparison gates;
- determine whether rho, mu, psi, or phase refinement dominates after the first
  higher-order run;
- rebuild and replay the thermal/spectral/display chain from that kernel;
- add genuine slow-light/time-dependent material state rather than treating the
  stationary finite-height disk as slow-light completion;
- define and test NR/GRMHD data adapters, coordinates/tetrads/interpolation and
  conservation contracts.  No NR/GRMHD evolution or dataset is implemented now.

## Completion estimate

Two completion scopes must be kept separate:

- **Offline stationary Kerr + NT + finite-thickness returning-radiation v1:**
  architecture/code is about 80–85% complete; source-current scientific
  acceptance is about 55–65% complete because strict real-grid convergence is
  still missing.  Likely remaining work is several focused implementation days
  plus one or more multi-hour numerical runs.
- **The full stated roadmap including convincing slow light and future NR/GRMHD
  integration:** about 50–60% complete overall.  The Kerr/NT foundation is strong,
  but time-dependent transfer and NR/GRMHD adapters remain substantial phases,
  not polish.

These percentages are planning estimates, not evidence claims.

## Drift assessment

The project has **not fundamentally moved away from the goal**.  The exact Kerr,
NT reference, finite-height transfer, convergence diagnostics, caching, and
provenance work all directly support a trustworthy offline renderer.

There is, however, a clear local drift risk: too much effort has recently gone
into contract/security/TOCTOU hardening after those layers were already strong,
while the main scientific gap is still a v2-qualified real kernel and genuine
slow-light evolution.  The corrective decision is:

- finish only the two presently blocking correctness/performance items
  (identity re-audit and two-ray issued primitive);
- add the minimal authenticated v2 adapter;
- freeze infrastructure and spend the next major budget on real refinement and
  physical interpretation;
- do not expand attestation, cache generality, or NR/GRMHD implementation until
  that Kerr acceptance milestone is met.

## Resume point

On the next session/continuation:

1. run the interrupted read-only audit against the latest cached verifier and
   renderer plan-freeze changes;
2. run jobs + cached-kernel + renderer full targeted regression;
3. implement the forward-only tracer-issued capability under independent attack
   review, preserving all kernel float leaves and sample-audit hashes;
4. implement the authenticated convergence-v2 adapter;
5. freeze the source tree, compute new cache/product keys, and only then launch
   the rho=8, mu=16, psi=32 real Kerr probe.

