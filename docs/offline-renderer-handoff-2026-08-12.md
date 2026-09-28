# Offline renderer handoff — 2026-08-12

This is the minimal source of truth for continuing the offline Kerr renderer in
a fresh Codex task.  Do not recover state from the old conversation.  Read this
file, the files named below, and the current checkout only.

## Git preservation boundary

- Repository: `/Users/shuolei/Documents/blackhole`
- Branch: `main`
- `HEAD == origin/main == 7b84c76ad3c605c5d8471e7e1b9bd437a5263081`
- Index is empty.
- Existing unpublished work is intentional: 5 tracked modified files and 117
  untracked files at handoff time.
- Tracked documentation diff fingerprint:
  `8089f47c3e729db88c34f2750897168f183e9f1b4fc1291b8c3ab2fba2b87a22`
- Do not commit, push, stage, reset, clean, stash, or rewrite unrelated files.
- Any later user-approved publication goes directly to remote `main`, followed
  by exact local `main` / `origin/main` fast-forward while preserving all other
  unpublished files.

## Scientific objective and honest status

Continue the local high-fidelity offline renderer in this order:

1. source-current stationary exact Kerr finite-thickness returning-radiation
   kernel;
2. strict authenticated finite-grid convergence-v2 checkpoint;
3. only after all four v2 comparisons pass, thermal/profile/frame/product;
4. genuine slow-light/time-dependent material state later;
5. NR/GRMHD adapters after the Kerr baseline is accepted.

Do not describe the current result as slow-light, NR/GRMHD, an independent
physics oracle, or a rigorous continuum error bound.  The historical
`rho=4, mu=8, psi=8` product is source-stale and v2-unconverged.

## Completed and independently audited

### Forward issued primitive: four rays to two

Files:

- `offline/kerr_returning_radiation_rays.py`
- `offline/kerr_returning_radiation_kernel.py`
- corresponding ray/kernel tests
- cached/renderer budget consumers

The production forward kernel now traces the canonical fine/coarse primitive
once and consumes a process-local, single-use issued capability.  Public
primitive verification still replays two rays.  Receiver remains four rays per
direction.  All matrix/fate/g2 leaves and five sample-audit hashes are bitwise
unchanged.

Independent result: P0=0, P1=0; 130 tests passed and one sandbox multiprocessing
test skipped.

Frozen source hashes at the accepted audit point:

- rays: `bbf5162b54d7887265cc4b3bc62105db63f02855ba4f6ff8c4531df812067ee9`
- kernel: `c3889e10d60a6a3c204087cbbe7542f87edfa2d343dec8ae60c00b46d1c6dc99`
- cached: `3ac2e1a4453cbc66a61af3d5825c10c26c795b68a72be5c25cc884a57f9abd90`

### Producer-authenticated convergence-v2 adapter

Files:

- `offline/kerr_returning_radiation_convergence_v2.py`
- the same kernel/cached modules above
- three corresponding tests

One canonical forward execution now yields producer-owned evidence for exactly
five passes: `full`, `half-rho`, `half-mu`, `half-psi`, and `phase-shifted`.
The authenticated adapter produces five summaries and four fixed comparisons.
Direct authentication performs a fresh canonical whole-kernel replay.  Cached
authentication replays authenticated records with zero evaluator jobs and zero
rays.  Caller-built summaries remain diagnostic-only.

Independent result: P0=0, P1=0; 59/59 tests passed twice.  The only noted P2s
were error normalization for an `object.__new__` empty shell and direct private
same-process use of an internal builder; neither is reachable through public
adapters under the declared threat boundary.

Frozen source hash:

- convergence-v2:
  `6cfeaca4f58ee198e4ead0a6aa88e7352ff6b72afbc4243c8122822fe38e1b6b`

## Checkpoint and renderer gate completed and frozen — 2026-08-13

The previous blocking work 1–8 is complete in the current unpublished
checkout.  The kernel-only path is now:

`production cached forward -> producer-authenticated convergence v2 -> atomic
closed two-file checkpoint -> external-digest secure verification`

Both qualified and non-qualified authenticated reports are published.  The
checkpoint path has no CIE, observer-screen, thermal, spectral, tile, frame, or
product dependency.  Publication is no-replace and post-publication success is
closed by two secure verifier passes plus source/runtime gates.

The renderer now requires both `--required-v2-checkpoint` and an externally
retained `--required-v2-checkpoint-sha256` before cache access.  It binds the
checkpoint's scientific job key, cache job key, complete reduction
configuration, and kernel descriptor to the exact planned product kernel.  It
then uses a cache-only integrator: missing, corrupt, replaced, symlinked, or
irregular cache state fails without evaluator construction, cache repair, or
ray tracing.  The exact all-reused/zero-executed audit is checked before
thermal work, and the checkpoint ID and digest are retained as a product input
artifact.

The cache-only callable identity is owner-frozen in the jobs module.  A clean
process test that replaces the public jobs callable before first importing the
cached bridge is rejected before definition construction or cache I/O, with
zero calls to the replacement.  The declared boundary still makes no claim of
protection from malicious same-process private-global rewriting.

Independent final read-only result: **P0=0, P1=0**.  The six direct suites for
jobs, cached bridge, convergence-v2, checkpoint, refinement CLI, and renderer
passed together: **139 tests passed, one existing sandbox multiprocessing test
skipped**.  Two final CLI owner-binding regressions were then added; the frozen
six-suite result is **141 tests passed, one existing sandbox multiprocessing
test skipped**.  `git diff --check` passed.

Frozen production SHA-256 values:

- rays:
  `bbf5162b54d7887265cc4b3bc62105db63f02855ba4f6ff8c4531df812067ee9`
- forward kernel:
  `c3889e10d60a6a3c204087cbbe7542f87edfa2d343dec8ae60c00b46d1c6dc99`
- direction jobs/cache substrate:
  `d6fb0068ef321776f9832cacc1e6120a72ea379317d6d9ad0a414bf2ccf3f4e5`
- cached kernel bridge:
  `62c3569dd447994c76a24b6eeb22f545ac2551f3c9f95a4367f0a9521281b05b`
- convergence-v2 adapter:
  `1968ab6896503edfb429c79a13c4000bf5f6f6d0e29ac17fda0679cefe1c0d3e`
- checkpoint publisher/verifier:
  `fbb08c28ddf2e7ebca7ac98aa7170cfbeb021a3036d0603476b12d7edeb5d27b`
- refinement CLI:
  `e637f79f4fffae164ca5cba8c35353ca162232572fe2bb8552561fafdeb7c8f0`
- returning-radiation renderer:
  `cb36aa8f87874c74c53cb6b4515fb2eeb1043f2ce0876b03abf86fa0a4988d1c`

No product is accepted yet: a source-current qualified real Kerr checkpoint
does not yet exist.

The first probe was started after freeze and deliberately interrupted when an
additional independent audit found a public CLI execute-alias binding gap.
That gap is now closed and independently re-audited at P0=0, P1=0.  The
interrupted run published no checkpoint and left 22 authenticated resumable
cache tasks.  The exact probe was then restarted and did reuse them.

## First real diagnostic probe after freeze — completed 2026-08-13

Target one annulus with:

- full grid: `rho=8, mu=16, psi=32`
- directions: `7 * rho * mu * psi = 28,672`
- issued-primitive whole rays: `57,344`
- directions per task: 64
- tasks: 448
- workers: 8
- max in flight: 16
- planning estimate: roughly 2.5 hours, not a guarantee

Static maximum normalized sample weights from the current quadrature are:

- full: `0.0008809120487773668`
- half-rho: `0.0016629912089007736`
- half-mu: `0.0016873538633150379`
- half-psi: `0.0017618240975547335`
- phase-shifted: `0.0008809120487773668`

The v2 cap is `0.001` and each report gates both full and comparison grids.
Therefore this probe was known in advance to be non-qualified in at least the
three half-grid reports.  It was run only to measure which dynamic
K/g2/fate/phase error dominates.  The policy was not relaxed, and no
thermal/frame/product artifact was run or published.

Authenticated result:

- checkpoint directory:
  `artifacts/offline-kerr-returning-radiation-refinement-rho8-mu16-psi32-20260813`
- checkpoint ID:
  `kerr-returning-radiation-refinement-64dd92b60d450b11ecc7380d`
- external manifest SHA-256:
  `cdd2311f327b6929ab469c9c824d9d4dbc2641d3a6159090d279479eedc83b68`
- internally bound checkpoint SHA-256:
  `64dd92b60d450b11ecc7380d22c147f5643c10eb626e5e4252ec766504ea2fce`
- classification: `source-current-v2-non-qualified-finite-grid-checkpoint`
- `qualified=false`; the public verifier accepted the closed two-file tree only
  when supplied the exact external manifest digest above
- restart audit: 22 tasks / 1,408 directions reused, 426 tasks / 27,264
  directions executed, 448 tasks / 28,672 directions authenticated, maximum
  in flight 16

Worst authenticated gate values are shown as multiples of their exact v2
tolerances (`1.0x` is the boundary):

| comparison | sample weight | K/g2 and column L1 | significant cell | fate TV | fate absolute | result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| half-rho | 1.662991x | 2.622961x | 1.328909x | 2.110698x | 1.930519x | fail |
| half-mu | 1.687354x | 4.795342x | 2.456572x | 0.675186x | 0.536833x | fail |
| half-psi | 1.761824x | 3.613019x | 1.839744x | 0.840766x | 0.705213x | fail |
| phase-shifted | 0.880912x | 0.740471x | 0.371611x | **1.102163x** | 0.932910x | fail |

All sampled support-flip and insignificant-tail gates passed.  The half-mu
comparison is the largest spatial K/g2/column error, but it is not a unique
overall refinement diagnosis: the resolution-qualified phase-shifted report
independently fails fate total variation (`0.0011021634921027875` against
`0.001`).  Its column gates all pass, and its largest fate-component absolute
difference also passes (`0.0009329098006939512` against `0.001`).  The half-rho
report independently has the largest half-grid fate error.  The evidence
therefore says **mixed rho/mu/phase failure**, not "mu alone".

The original selection rule was:

- rho dominates -> `16/16/32`
- mu dominates -> `8/32/32`
- psi dominates -> `8/16/64`
- phase or mixed dominance -> stop for human review; do not blindly double all
  dimensions

The result takes the final branch: **stop for human review**.  Do not start
`8/32/32`, do not blindly double all dimensions, and do not relax the v2
policy.  Each single-axis candidate would have 57,344 directions and 114,688
whole rays, while the present evidence does not authorize any one of them.

The local, zero-ray review further localized the phase failure.  On both faces
the phase shift changes the fate partition by the same amounts:

- same-face return: `+0.0009329098006939512`
- captured: `+0.00016925369140880636`
- escaped: `-0.000790636395157085`
- plunge sink: `-0.00031152709694562755`

The discrepancy is therefore a redistribution led by return versus escape,
not a support flip.  It is spread over several mu nodes; the largest
uncancelled per-mu fate-TV contributions occur at mu indices 7
(`0.00049709691667529224`), 9 (`0.00044137586208743641`), and 8
(`0.00028405839542467375`).  This supports the angular part of the mixed
rho/mu/phase diagnosis rather than reducing the result to one spatial axis.

The existing `full`, `phase-shifted`, and `half-mu` passes already supply three
corners of a focused `mu x phase` 2-by-2 diagnostic:

- A: `rho=8, mu=16, psi=32, phase=0.0` (`full`)
- B: `rho=8, mu=16, psi=32, phase=0.5` (`phase-shifted`)
- C: `rho=8, mu=8, psi=32, phase=0.0` (`half-mu`)
- D: `rho=8, mu=8, psi=32, phase=0.5` (missing)

The mu=8 Gauss nodes are not a subset of the mu=16 nodes, and the two phase
lattices do not overlap, so D cannot be inferred from the current records.  If
a new expensive diagnostic is explicitly approved, D is the smallest focused
experiment: 4,096 new directions and 8,192 issued-primitive whole rays for one
annulus and two faces.  Compare the phase effects `B-A` and `D-C`, the mu
effects `A-C` and `B-D`, and the interaction `(B-A)-(D-C)`.  Publish it only as
diagnostic evidence, never as a canonical v2 checkpoint or product gate.

There is also a useful zero-ray lattice check: the physical-direction union of
the existing `psi=32, phase=0.0` and `psi=32, phase=0.5` passes equals the
`psi=64, phase=0.5` lattice.  A diagnostic reconstruction must use the
canonical psi=64 weights, rather than blindly halving every psi=32 weight,
because exact float64 closure adjusts the last weights.  This union reuses the
same samples and is not an independent phase test or producer-authenticated v2
report.

If changing the evidence contract is not approved, the smallest job accepted
by the current fixed five-pass contract that contains D is a complete
`rho=8, mu=8, psi=32` job: 14,336 directions and 28,672 whole rays.  Its job
and pass identities prevent claiming cross-job cache reuse.  The narrow D-only
contract is therefore the preferred diagnostic design, but it requires an
explicit scientific decision before implementation.

The D-only experiment isolates the `mu x phase` interaction; it does not settle
the independent half-rho fate failure or authorize a canonical refinement
axis.  A separate rho-focused decision remains after D is interpreted.

## Second real diagnostic probe — completed 2026-08-14

The user approved continuing along the measured scientific gaps.  Rather than
introduce a new one-pass evidence contract, the existing frozen five-pass
pipeline was used at `rho=8, mu=8, psi=32`.  This cost 14,336 directions,
28,672 issued-primitive whole rays, and 224 tasks.  It contains the missing
`mu=8, phase=0.5` corner while preserving the already audited cache,
authentication, checkpoint, and product-gate boundaries.

Authenticated result:

- checkpoint directory:
  `artifacts/offline-kerr-returning-radiation-refinement-rho8-mu8-psi32-20260814`
- checkpoint ID:
  `kerr-returning-radiation-refinement-55daa3abd0cf22e59b81000f`
- external manifest SHA-256:
  `086e71403de23932548acfa431a5ca033960f83139f43251b94107ab8ff8aee2`
- classification: `source-current-v2-non-qualified-finite-grid-checkpoint`
- all 224 tasks / 14,336 directions were executed and authenticated; none were
  reused; maximum in flight was 16
- the public verifier accepted the exact closed two-file tree with the external
  digest above; the renderer correctly rejected it as non-qualified

An independent reproducibility gate compared the new `full 8/8/32, phase=0`
pass against the old probe's physically identical `half-mu` pass.  After
excluding the necessarily different pass/grid identity, all 4,096 cached
records and every summary value were bitwise identical.  Both sample audits
have SHA-256
`3be9c01124af19ad494c9191668ff065916baf9359349f907484b28920249041`.
The interaction result is therefore not a restart or cache artifact.

Let A/B be the old probe's `mu=16` phase-zero/phase-half summaries and C/D be
the corresponding `mu=8` summaries.  The signed interaction
`(B-A)-(D-C)` is:

- proper-power and g2 diagonal: approximately
  `-0.00063558944607` on both faces
- fate half-L1: `0.0005665693207193715`
- upper-source fate components in canonical order:
  `[-0.0000318594380265, 0, +0.000501197161106,
  +0.0000653721596136, -0.000534709882693]`

The g2 phase response is `+0.000184292570967` at `mu=16` but
`+0.000819882017041` at `mu=8`, a factor of `4.44880666`.  Refining mu from 8
to 16 suppresses about 77.52 percent of that coarse-grid phase dependence.
The fate interaction mainly repartitions captured versus plunge-sink rays.
This is strong finite-grid `mu x phase` coupling, not evidence for a
mu-only failure and not a canonical v2 result.

The two five-pass probes also close the existing `rho x mu` and `psi x mu`
rectangles at phase zero.  They show observable-dependent interactions: the
rho/mu interaction is small in proper power but large in fate classification,
while the psi/mu interaction also exceeds the strict fate scale.  Combined
with the independently failing half-rho and half-psi reports, this rules out a
blind single-axis canonical run.

## Current resume point

Preserve both completed checkpoint directories, their external digests, and
their caches as immutable evidence.  Do not relax the v2 policy and do not use
either checkpoint as a product gate.

The proposed `rho=8, mu=16, psi=16` five-pass rerun was started, then safely
interrupted after 26 authenticated resumable tasks when an exact zero-ray
lattice audit proved it would add almost no new scientific sampling.  It
published no checkpoint and left no staging directory.  Its cache remains at
`artifacts/offline-kerr-returning-radiation-kernel-cache-rho8-mu16-psi16-20260814`
under cache job key
`e68dd104552e2b324dfd6597316e85416fc50937bfe2eb9adca9e1afa5607bdc`.

The exact midpoint lattice identity is stronger than the earlier description:
the old `psi=32, phase=0.5` pass's even psi indices are exactly the
`psi=16, phase=0` lattice and its odd indices are exactly the
`psi=16, phase=0.5` lattice.  Replaying the old authenticated transports with
canonical psi=16 weights reproduced the old half-psi summary bitwise and
predicted the missing half-cell summary.  The fine phase summary closes as
one half of the two coarse summaries: K/g2 residuals are bitwise zero and the
only fate residual is `3.469446951953614e-18`.

The coarse psi=16 half-cell response in g2 is approximately
`0.00215372413495`, versus `0.000184292570967` at psi=32, a factor of
`11.6864`.  The signed parity contrast has g2/proper-power diagonal
approximately `-0.00196943156398` and fate half-L1
`0.00061978912452438`.  This securely localizes strong periodic-lattice
odd/even aliasing.  It is a cross-pass diagnostic reweighting of the same
physical rays, not a producer-authenticated v2 report or new continuum sample.

The genuinely missing `rho=4, mu=16, psi=32, phase=0.5` corner was then run
through the unchanged fixed-five-pass contract.  The 4,096 target directions
had zero exact physical overlap with either completed cache; the full job cost
14,336 directions / 28,672 issued-primitive whole rays and 224 tasks.

Authenticated result:

- checkpoint directory:
  `artifacts/offline-kerr-returning-radiation-refinement-rho4-mu16-psi32-20260814`
- checkpoint ID:
  `kerr-returning-radiation-refinement-668598aef4d7160a58a90e54`
- external manifest SHA-256:
  `e9d6d77464fdf674fcffd27d88f450051079fbb5168ee11c6615e2dd9eced7a9`
- classification: `source-current-v2-non-qualified-finite-grid-checkpoint`
- 224/224 tasks and 14,336/14,336 directions executed; zero reuse; maximum in
  flight 16
- independent read-only audit: P0=0, P1=0; public external-digest verifier,
  closed two-file tree, source/runtime closure, counters and all leaf digests
  passed; the renderer rejected it before product work

The new full summary reproduced the old half-rho summary bitwise except for
the required grid ID.  The rho/half-cell-lattice 2-by-2 then showed a strong
non-separable response.  Proper power/g2 phase sensitivity changes from
`+0.000184292570967` at rho=8 to `-0.002065610642489` at rho=4: it reverses
sign and grows by about 11.2 times.  The signed interaction is
`+0.002249903213456`, about 9.11 percent of the rho=8 full value.  Fate
interaction half-L1 is `0.001966016186736`.

The same two checkpoints close a rho-by-psi rectangle at phase zero.  The
psi16-to-psi32 g2 effect is 3.01467 times larger on rho=4 than on rho=8, and
the rho effect changes sign with psi resolution.  Its signed g2 interaction
is about `0.00179823111960` and fate half-L1 is
`0.001440162208408`.  Together with the existing mu-by-phase and
mu-dependent fate evidence, these results reject a single-axis canonical
refinement.  They remain finite-grid same-code diagnostics, not qualification
or a continuum bound.

The completed interaction synthesis selects the joint production canonical
grid `rho=16, mu=32, psi=64`; do not run the staged `16/16/32` job first.  The
staged job would retain the already measured `mu=16` versus `mu=8` comparison,
whose K/g2 error is about 4.8 percent at both measured rho levels, so it has
negligible qualification probability despite costing one quarter of the joint
run.  The joint choice is evidence-driven by the closed rho/mu/psi/phase
rectangles, not a blind all-axis doubling.

Exact joint preflight:

- output:
  `artifacts/offline-kerr-returning-radiation-refinement-rho16-mu32-psi64-20260814`
- cache:
  `artifacts/offline-kerr-returning-radiation-kernel-cache-rho16-mu32-psi64-20260814`
- scientific plan SHA-256:
  `a04fb6a9180a69e8cab8a6fb96bb615715408982ba5f25ad0ac8844cc8f6c0f7`
- scientific job key:
  `fc38da1601f48e01ebc69ca21b44429ab45cee37a24416ac540ecacb36b189b7`
- cache job key:
  `2ca1b762165af1c1897cd9fd11dc0127bfdb6dd5cf04e315a89be4e37bcd0c99`
- 229,376 directions / 458,752 issued whole rays / 3,584 tasks
- pass direction/task counts:
  `65,536/1,024`, `32,768/512`, `32,768/512`, `32,768/512`,
  `65,536/1,024`
- exact maximum normalized sample weights:
  `0.0001165370973966069`, `0.00022362570237667738`,
  `0.00022953294754395626`, `0.0002330741947932138`,
  `0.0001165370973966069`; all are below the strict `0.001` cap
- 141 core tests passed, with one existing sandbox multiprocessing skip; all
  frozen production hashes remained unchanged; 365 GiB was free
- measured central runtime estimate roughly 18--21 hours; use a 16--24 hour
  operational window, not a guarantee

Stop rules for the joint run:

1. contract, source/runtime, numeric, cache, or authentication failure stops
   fail-closed; resume only the exact same identity from its authenticated
   cache;
2. do not interpret ordered partial tasks as scientific early-stop evidence;
3. a complete checkpoint unlocks product only if the externally anchored
   public verifier accepts it and all four exact reports have both
   `resolutionQualified=true` and `converged=true`;
4. a non-qualified result remains evidence and never enters thermal/frame;
5. if exactly one report fails, every resolution gate passes, and every dynamic
   excess is at most 1.25 times its tolerance with monotone behavior, review
   one further axis refinement; otherwise stop and review the integration
   strategy--do not automatically escalate to `32/64/128`.

The first attempt at this joint grid used the historical default surface-event
profile.  It stopped correctly before publishing any task payload or receipt
with `independent fine/coarse complete crossing phase spaces disagree`.  The
failed cache under job key
`2ca1b762165af1c1897cd9fd11dc0127bfdb6dd5cf04e315a89be4e37bcd0c99`
contains only `job.json` and 16 released zero-byte lock files; it has no live
worker, payload, receipt, output, or staging directory.  Do not delete or
reinterpret those lock files.

The canonical first witness was `full`, upper face, annulus 0, rho index 0,
mu index 5, psi index 49, ordinal 369:

- source rho: `3.507634788167437`
- emission cosine: `0.07531619313371501`
- azimuth: `4.859651136021712`
- default fine/coarse crossing event difference: approximately
  `6.62360e-5 M`, above the unchanged `2e-5 M` gate
- fate/topology: both escaped with the same single transparent
  outside-outer-radius upper-face crossing

Ordinal 433 (`muIndex=6`, `psiIndex=49`) failed the same complete-crossing
gate.  Decomposition showed a coupled coarse-ray/coarse-surface localization
error, not permission to relax the gate or reclassify either direction.  A
fully nested tighter ray/surface pair passed both witnesses, but would be much
more expensive than necessary.

The selected recovery is instead one uniform, source-unchanged surface-event
profile.  Ray options remain exactly at the historical defaults.  All six
fine surface tolerances/limits are divided by eight and fine subdivisions are
raised from 4 to 8, so the automatically derived coarse surface policy is
exactly the historical fine surface policy.  The exact additional CLI values
are:

```text
--surface-absolute-tolerance 6.25e-11
--surface-relative-tolerance 6.25e-11
--surface-null-residual-limit 2.5e-8
--surface-metric-interpolation-error-limit 1.25e-8
--surface-value-tolerance 1.25e-10
--surface-affine-tolerance-over-mass 1.25e-11
--surface-subdivisions-per-segment 8
```

This is a new uniform scientific identity, not an exception-triggered retry
and not a continuation of the failed cache:

- scientific plan SHA-256:
  `a04fb6a9180a69e8cab8a6fb96bb615715408982ba5f25ad0ac8844cc8f6c0f7`
- scientific job key:
  `ec04fe63c9c1106207eec865e2bb20c6a2b339ab2cb6feea2ab04d3d35e5df94`
- cache job key:
  `dff18aa8380473404d1b6f28fcdac05129a766711cb5ead1cb9753487fc95e1d`
- output:
  `artifacts/offline-kerr-returning-radiation-refinement-rho16-mu32-psi64-surface8-20260814`
- cache:
  `artifacts/offline-kerr-returning-radiation-kernel-cache-rho16-mu32-psi64-surface8-20260814`

Before the full run, the exact public tracer was applied to all 1,024
directions in the failed attempt's first in-flight rho block.  The new uniform
profile passed 1,024/1,024 directions.  The maximum complete-crossing event
difference was `2.37212934898e-6 M` at ordinal 369, leaving about 8.4 times
margin to the unchanged `2e-5 M` gate.  This probe authorizes the new full
run; it does not prove that a later direction cannot still fail.  Any later
failure remains fail-closed and must be diagnosed rather than caught or
assigned to one of the five fate bins.

The new identity was launched on 2026-08-14 with 8 workers and maximum 16
tasks in flight.  Its first three task payload/receipt pairs were committed
successfully, proving that worker startup, atomic task publication, and the
sliding in-flight queue are active.  The exact resumable command is:

```bash
python3 scripts/run_offline_kerr_returning_radiation_refinement.py \
  /Users/shuolei/Documents/blackhole/artifacts/offline-kerr-returning-radiation-refinement-rho16-mu32-psi64-surface8-20260814 \
  --kernel-cache /Users/shuolei/Documents/blackhole/artifacts/offline-kerr-returning-radiation-kernel-cache-rho16-mu32-psi64-surface8-20260814 \
  --surface-absolute-tolerance 6.25e-11 \
  --surface-relative-tolerance 6.25e-11 \
  --surface-null-residual-limit 2.5e-8 \
  --surface-metric-interpolation-error-limit 1.25e-8 \
  --surface-value-tolerance 1.25e-10 \
  --surface-affine-tolerance-over-mass 1.25e-11 \
  --surface-subdivisions-per-segment 8 \
  --kernel-rho-order 16 \
  --kernel-mu-order 32 \
  --kernel-psi-count 64 \
  --kernel-jobs 8 \
  --kernel-max-in-flight 16
```

If interrupted, use this exact command and these exact paths.  Do not resume
the historical-default cache, mix identities, delete lock files, or change
any source/runtime file while this run is active.

If and only if this checkpoint is externally anchored, independently verified,
and qualified, the first product should be a new-path one-annulus 1-by-1
calibration frame, not a claimed high-quality final image.  It must bind the
same kernel cache, rho/mu/psi orders, directions-per-task, and the exact
surface8 options above; its cache-only kernel audit must report all 229,376
directions and all 3,584 tasks reused with zero executed.  Run the full
spectral structural verifier and live-replay attestation before selecting a
larger field of view.  A visually useful frame still requires separate FOV,
pixel-resolution, and adaptive-depth convergence, and a multi-annulus radial
solution still requires a new exact multi-annulus qualified checkpoint.

Even if this one-annulus checkpoint qualifies, it authorizes only the exact
same one-annulus cache/definition.  A multi-annulus final radial solution must
obtain its own exact qualified checkpoint; this result cannot be substituted.

Post-qualification work is now preflighted but must remain read-only while the
active cache is running:

- first execute a new-path one-annulus 1-by-1 calibration product with the
  exact qualified checkpoint and cache, then require structural verification,
  full live byte replay, and zero-executed/all-reused kernel audit;
- establish a real FOV with fixed-pitch scouts over `[-1,1]^2`,
  `[-1.5,1.5]^2`, and `[-2,2]^2` (extend to `[-2.5,2.5]^2` only if needed),
  then perform `32^2 -> 64^2 -> 128^2` resolution and depth 0/1/2/3
  convergence on new product identities;
- for radial convergence, use independently qualified nested log-radius
  `2 -> 4 -> 8` annulus checkpoints.  Eight annuli cost 1,835,008 directions
  and fit the current 2,000,000-direction hard cap; 16 annuli do not.  No
  checkpoint or cache is reusable across annulus counts;
- the current live replay cannot attest a native-resolution 471-bin frame:
  its 1 GiB total-tile and 10-million-geodesic caps, plus whole-product memory
  snapshots, limit it to at most about 139,519 spectral pixels.  After the
  current checkpoint is externally anchored, implement streaming, bounded,
  resumable segment attestation rather than merely increasing constants.

These steps define the path to a visually high-quality result for the exact
stationary finite-grid model.  They still do not constitute a continuum,
caustic-complete, slow-light, complete-KERRBB, polarization, or GRMHD result.

## Minimal commands for the fresh task

Start by verifying preservation and the completed checkpoint files:

```bash
cd /Users/shuolei/Documents/blackhole
git branch --show-current
git rev-parse HEAD origin/main
git diff --cached --name-only
python3 -m py_compile \
  offline/kerr_returning_radiation_refinement_checkpoint.py \
  scripts/run_offline_kerr_returning_radiation_refinement.py
```

Then inspect only this handoff, the completed manifest/sidecar, the two
checkpoint implementation files, the authenticated-v2 module, and their direct
tests.  Do not load the old conversation history.
