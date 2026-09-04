# Milestone 3 measured result

Measured on 2026-08-24 on an Apple M3 Pro with Metal Toolchain
`com.apple.dt.toolchain.Metal.32023.883`.  The reproducible full run was:

```sh
native/metal/run_milestone3.sh
```

It exited zero.  The generated metallib SHA-256 was
`65a4836be4b7ec7d8c061700395d7d366c58d9323f51179e479f09747636cac2`;
the frozen 4,096-record corpus SHA-256 remained
`bc62957cf2ae421e2d4e8b91fdbb19794b03258c4d42554efa29e10bdd0819a3`.
The measured CPU timing sidecar SHA-256 was
`9b61c96ec831fdc92d0e2fc4bb540a92bdc9832d93e16fdf47f823e228e952a3`
and the report SHA-256 was
`d2e8c494df6a10ae118475e49bdd58acde833723cd30b0eaddfe96dd8d682b34`.
Timing-bearing sidecar and report hashes are expected to change between
machines or benchmark runs; the runner authenticates their corpus closure.

This milestone remains deliberately **not production-qualified**.  It passes
the algebra, DOPRI-combination, and first planar step/probe gates, but does not
implement an adaptive loop, finite-thickness surface root localization, path
recording, crossing order, whole-ray fate, or renderer integration.

## Precision repair

Milestone 2 used `(hi + lo) * 2^exponent` throughout.  Its approximately
48-bit product/accumulator changed the sign of four ill-conditioned RHS
components and missed the DOPRI embedded-error gate.

Milestone 3 preserves those exact two-word wire inputs, then promotes products
and reductions to a shared-exponent three-float expansion.  Every exact
expansion component is folded through a three-level error-free accumulator;
only the residual below the third word is discarded.  Public arithmetic
boundaries are rounded to the normalized binary64 grid, with the third word
carrying guard/round information.  Safe Metal math and explicit product-error
FMA remain mandatory.

An early implementation incorrectly selected only the three largest expansion
components.  That lost the guard/round tail and left the DOPRI state at 37.36
bits.  The final implementation folds the entire expansion before rounding;
the same unchanged gate then rose to 41.10 bits.

## Frozen witness-first replay

The runner authenticates the full milestone-2 corpus digest, selects these
four records in fixed order, and dispatches them before the aggregate corpus:

```text
adversarial-470
adversarial-514
adversarial-1528
adversarial-1862
```

Their canonical frozen witness-set SHA-256 is
`bb8c785dab7aec34978c8ae5c5f7ecb7712c5503259bf7eefd71fa0c89d04105`.
All 32 output components had zero sign drift, zero zero/nonzero drift, and
53.00 worst condition-normalized bits.  Any witness failure aborts before the
full corpus.  A malformed nonfinite RHS input also returned the distinct
fail-closed status expected by the host.

## Full 4,096-record gates

| Gate | Components | Worst direct bits | Worst condition-normalized bits | Sign / zero drift |
| --- | ---: | ---: | ---: | ---: |
| Real analytic Kerr RHS | 16,384 | 37.10 | 51.08 | 0 / 0 |
| Adversarial RHS | 16,384 | 3.67 | 51.06 | 0 / 0 |
| DOPRI fifth-order state | 32,768 | 41.10 | 41.10 | 0 / 0 |
| DOPRI embedded error | 32,768 | 39.40 | 42.88 | 0 / 0 |
| Batched step state | 32,768 | 41.10 | 41.10 | 0 / 0 |
| Signed planar surface probe | 4,096 | 18.11 | 45.48 | 0 / 0 |

The fixed gates were 40 condition-normalized bits for RHS and 38 bits for the
DOPRI state, embedded error, batched state, and planar probe.  No threshold,
corpus record, category, or categorical mismatch policy was relaxed.

The embedded error is accumulated directly as
`h * sum((b5 - b4) * stage)`, so it does not subtract two separately rounded,
state-sized results.  The CPU reference uses the same exact quantized wire
coefficients and direct error formula.

## Throughput

For 32,768 records, the reproducible run measured:

- triple-word precomputed-metric RHS: `2.183 million records/s` wall;
- current Python binary64 RHS reference: `61,985 records/s`;
- measured RHS wall-rate ratio: `35.22x`; and
- precomputed-stage DOPRI fifth-state plus planar z probe:
  `5.317 million records/s` wall.

These are resident algebra/batching measurements.  They exclude Kerr metric
construction, seven stage RHS evaluations, adaptive rejection, finite-thickness
surface reintegration, root localization, topology divergence, path recording,
and cache I/O.  They must not be projected as an end-to-end whole-ray speedup.

## Next evidence boundary

Do not connect this kernel to the renderer yet.  The next milestone must batch
a complete one-resolution DOPRI step with its stage RHS evaluations, then add
the actual finite-thickness surface probe/reintegration policy.  Only after
expanded phase-space replay and complete fine/coarse whole-ray fate, target,
crossing-order, and path parity may a new scientific runtime identity be
considered.
