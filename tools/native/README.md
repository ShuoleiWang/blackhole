# Native Kerr qualification harness

`golden_cache.py` authenticates a frozen returning-radiation task cache without
importing `offline`, constructing a cache definition, or invoking a runner. It
is therefore safe to point at the copied, incomplete nested16 cache: missing
task pairs remain missing and no ray can be evaluated.

The default trust anchors bind the exact copied checkpoint evidence:

- cache job: `d8286face863a0755d70c20c1638c6a5dc4d9d9037c0227626fb805bb4d0f751`
- scientific job: `19c8c3ebd465a4f66d6e893935dc18134ecd47d8842b2e77809823d4148ec980`
- scientific plan: `a04fb6a9180a69e8cab8a6fb96bb615715408982ba5f25ad0ac8844cc8f6c0f7`
- job document: `d66a1c7f5de21af2492747fc5fa7cf2835c7977d9abce659454cdf4469898a1f`
- authenticated payload set: `010f6a21dcd774eba15afa1760d8a5df76415b9260c231d3cea715a58c8f8d91`
- authenticated direction stream: `0bf7548ea6f93cd1986a04adca2e42090db5ed4848c51b3f4289348db472bf4e`

The checked-in `nested16_golden_corpus.json` was deterministically extracted
from 596 authenticated task pairs (38,144 directions, continuous ordinals
0 through 38,143). Its corpus SHA-256 is
`15758851df05d4a6126eee00995ea99a5d76261bbb71266015b023b83f191606`.
The narrow records are ordinals 369, 433, 726, 732, 736, 31,263, and 37,331
and collectively cover escaped, captured, plunge-sink, return-upper, and
return-lower fates.

## Re-extract

Use an absolute path. The extractor rejects an output inside the cache.

```sh
python3 tools/native/golden_cache.py extract \
  /private/tmp/blackhole-native-opt.Ze0wcI/golden-cache/d8286face863a0755d70c20c1638c6a5dc4d9d9037c0227626fb805bb4d0f751 \
  --output /private/tmp/reextracted-nested16-corpus.json

cmp tools/native/nested16_golden_corpus.json \
  /private/tmp/reextracted-nested16-corpus.json
```

The reader verifies the external job-document digest, content-addressed job
and scientific identities, exact five-pass task plan/order, directory entry
types, receipt schema, payload length/SHA-256, payload schema, evaluator
transport schema, fate/receiver topology, and every record coordinate.
One-sided pairs, unknown files, symlinks, duplicate/non-canonical JSON, and
tampering all fail closed.

## Candidate and comparison

A native backend writes canonical JSON with this narrow shape:

```json
{
  "records": [
    {"coordinate": {}, "transport": {}}
  ],
  "schema": "blackhole.native-kerr-golden-candidate/v1",
  "version": 1
}
```

Each actual `coordinate` and evaluator `transport` object must match the cache
record schema. Records must contain exactly the golden ordinals in golden
order. Optional backend diagnostics may appear only under a separate top-level
`expanded` object; narrow qualification never depends on it.

```sh
python3 tools/native/golden_cache.py compare \
  "$(pwd)/tools/native/nested16_golden_corpus.json" \
  /absolute/path/to/native-candidate.json \
  --require-byte-exact \
  --output /private/tmp/native-comparison.json
```

The CLI externally authenticates the checked-in golden file against the corpus
SHA above before comparing it; the selected-record hash inside the document is
not treated as its own trust anchor.

Categorical values and topology are always exact. Floating-point thresholds
are explicit and default to zero (`--absolute-tolerance`,
`--relative-tolerance`, and `--maximum-ulp`). Exit status is 0 for qualified,
2 for an authenticated comparison with drift, and 1 for malformed or
unauthenticated input.

## Tests

```sh
python3 -m unittest -v tools.native.tests.test_golden_cache
```

The fixtures are intentionally partial caches. Tests cover receipt and payload
SHA tampering, schema tampering with a freshly forged receipt, record
reordering, non-canonical JSON, one-sided publication, symlink substitution,
external job-hash anchoring, categorical drift, numeric thresholds, and
byte-exact comparison.

This is deliberately a **seven-record narrow gate**. A qualified report states
that it does not prove parity over all 38,144 authenticated directions and does
not prove native execution provenance. Full-corpus differential execution plus
backend-binary/runtime provenance remain mandatory before production use.

## Expanded phase-space golden

`phase_space_golden.py` is the milestone-2 full-ray differential gate.  Its
checked-in `nested16_phase_space_golden.json` is externally frozen at SHA-256
`8ff7391df7a39992ba2733a61aec9c63460fa1ff33ad458f3e70dae8b137cc67`;
its internal manifest SHA-256 is
`47bafc7c1ee27e6200bdb4b2d5ca58f549e7c9ad83464195b3d23fc6d5e16ecd`.

The reference contains the seven authenticated-cache ordinals above plus the
critical noncached ordinal 75,785.  Every cached ordinal was freshly traced by
the unchanged Python backend and accepted only after its primitive descriptor
SHA matched the authenticated cache.  Ordinal 75,785 is explicitly marked as
noncached; no cache provenance is fabricated for it.

For both independent fine and coarse whole rays, each record retains:

- terminal event/covector, outcome, target, and affine length;
- every accepted segment's start, reconstructed midpoint, end, affine length,
  and midpoint null residual;
- accepted/rejected counters and maximum null/metric errors;
- the ordered two-face crossing topology, localized phase-space states,
  classifications, root diagnostics, N/2N probe evidence, and work counters;
- receiver event, model descriptor, radius, face, frequency ratio, incidence,
  g4, and incidence-weighted g4 where applicable; and
- every fine/coarse convergence gate and scalar difference.

The reference additionally binds the exact scientific document, production
source closure, CPython executable, native `math` extension, numeric-runtime
descriptor, and all per-record/record-set/configuration hashes.  Generation
rebuilds the frozen cache/scientific keys before tracing and rechecks the
source/runtime closure after tracing.  It opens the original cache only through
the milestone-1 read-only authenticator and never invokes a cache runner.

Re-generate to a new path (roughly several minutes on the M3 Pro):

```sh
python3 tools/native/phase_space_golden.py generate \
  /private/tmp/blackhole-native-opt.Ze0wcI/golden-cache/d8286face863a0755d70c20c1638c6a5dc4d9d9037c0227626fb805bb4d0f751 \
  --output /private/tmp/rebuilt-phase-space-golden.json

cmp tools/native/nested16_phase_space_golden.json \
  /private/tmp/rebuilt-phase-space-golden.json
```

Verify the checked-in bytes and every embedded invariant/hash:

```sh
python3 tools/native/phase_space_golden.py verify \
  "$(pwd)/tools/native/nested16_phase_space_golden.json"
```

A native backend candidate has the exact top-level schema
`blackhole.native-kerr-phase-space-candidate/v1`, binds the frozen golden SHA,
identifies its own backend provenance, and supplies the same ordered
`{coordinate, sample, phaseSpace}` records.  Build a reference-copy candidate
and exercise the comparator with:

```sh
python3 tools/native/phase_space_golden.py make-reference-candidate \
  "$(pwd)/tools/native/nested16_phase_space_golden.json" \
  --output /private/tmp/reference-phase-space-candidate.json

python3 tools/native/phase_space_golden.py compare \
  "$(pwd)/tools/native/nested16_phase_space_golden.json" \
  /private/tmp/reference-phase-space-candidate.json \
  --require-byte-exact
```

Topology, strings, booleans, integers, array lengths/order, object schemas, and
the sign of zero are always exact.  Floats are exact by default; a qualification
run must explicitly declare any absolute, relative, or ULP allowance.  The
expanded eight-record gate is still a differential qualification corpus, not
an independent geodesic/physics oracle and not proof over all 229,376 planned
directions.

Run both harness suites with:

```sh
python3 -m unittest -v \
  tools.native.tests.test_golden_cache \
  tools.native.tests.test_phase_space_golden
```

## Read-only mu x psi x phase interaction analysis

`mu_psi_phase_interaction.py` accepts an externally authenticated native
`rho16/mu32/psi64` parent checkpoint and a native `rho16/mu16/psi64` target
checkpoint. Both must pass the public checkpoint verifier against the same
explicit dylib. It extracts cells A--F, requires target `full` to reproduce the
parent `half-mu` physical summary, raw identity-free physical sample-audit SHA,
and pass-normalized retained evidence, emits the seven available v2 edge
diagnostics, and evaluates the identifiable `mu-by-phase` and `mu-by-psi`
mixed differences.

An optional parent cache job enables an additional read-only parity gate. The
tool authenticates every task/receipt without importing a runner, selects the
even and odd directions of the parent phase-shifted psi64 pass, doubles and
canonically re-closes the angular weights, and requires the B-even normalized
record stream to reproduce E exactly. B-odd is retained as a bound psi32,
phase-0.5 stream with no five-pass comparator.

```sh
python3 tools/native/mu_psi_phase_interaction.py \
  --parent-manifest /absolute/parent/manifest.json \
  --parent-sha256 PARENT_EXTERNAL_SHA256 \
  --target-manifest /absolute/target/manifest.json \
  --target-sha256 TARGET_EXTERNAL_SHA256 \
  --native-library /absolute/libblackhole_cpu.dylib \
  --parent-cache-job /absolute/parent-cache/CACHE_JOB_KEY \
  --parent-cache-payload-set-sha256 EXTERNAL_PAYLOAD_SET_SHA256 \
  --parent-cache-direction-stream-sha256 EXTERNAL_DIRECTION_STREAM_SHA256 \
  --output /absolute/new-interaction-report.json
```

Both cache snapshot digests must have been retained outside the cache. A job
specification plus freshly resealed payload/receipt pairs is not an
authentication anchor, so the cache replay option refuses to run without both
external digests.

The output is a no-overwrite atomic new file and permanently sets
`qualified=false`, `productionQualified=false`, `productEligible=false`, and
`automaticEscalation=false`. Missing phase-shifted psi32 cells are explicit;
the tool neither changes the production five-pass contract nor starts another
checkpoint.

The no-ray synthetic suite is:

```sh
python3 -m unittest -v tools.native.tests.test_mu_psi_phase_interaction
```
