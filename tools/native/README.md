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
