# Milestone 2 measured result

Measured on 2026-08-24 on an Apple M3 Pro with Metal Toolchain
`com.apple.dt.toolchain.Metal.32023.883`.  The generated metallib SHA-256 was
`bd667ec95236f754e8e44bdf4d18163ea5e5aa54e3ee70bd15b63cad010669c5`.
The 4,096-record corpus SHA-256 was
`bc62957cf2ae421e2d4e8b91fdbb19794b03258c4d42554efa29e10bdd0819a3`;
the CPU timing sidecar SHA-256 was
`b2aa71eae3480c2bed402f280ac9a8d48547ddb4f34550730fe291fbb47a768a`;
the machine-readable report SHA-256 was
`4ac5d0afaa6c4908cf6508093679ea706990ea937d2f0987c078afe10eba3d48`.
The timing sidecar embeds and the runner verifies the deterministic corpus
digest before comparing CPU and GPU rates.

This milestone is deliberately **not** production-qualified.  The final full
run exited non-zero because the adversarial RHS and DOPRI diagnostic exposed
precision limits that would be unsafe to waive.

## Subnormal low-word fix

Milestone 1 encoded a value as absolute `hi + lo`.  For sufficiently small
values the low word became FP32 subnormal and Metal flushed it.  Milestone 2
uses:

```text
(hi + lo) * 2^signedExponent
```

Every non-zero mantissa is normalized to `0.5 <= abs(hi) < 1`, so `lo` is a
mantissa residual rather than a word at the value's absolute exponent.  Across
4,096 inputs per operation spanning binary64-scale exponents, all five
operations had zero sign drift, zero zero/non-zero drift, and no non-finite
result:

| Operation | Worst effective bits |
| --- | ---: |
| add | 48.01 |
| subtract | 48.03 |
| multiply | 46.96 |
| divide | 47.57 |
| square root | 48.24 |

Explicit divide-by-zero, negative-square-root, and malformed-nonfinite probes
also returned distinct fail-closed status codes.  This resolves the milestone-1
subnormal representation defect for the tested finite binary64 exponent range.

## Precomputed-metric Hamiltonian RHS

The GPU kernel consumes the exact quantized inverse metric, four inverse-metric
derivatives, and covector.  CPU references came from the current checkout's
binary64 `offline.spacetime.matrix_vector` and `bilinear` implementations.

| Corpus | Records | Components | Worst result-relative bits | Worst condition-normalized bits | Sign drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real analytic Kerr | 2,048 | 16,384 | 32.56 | 46.21 | 0 |
| Adversarial exponent/cancellation | 2,048 | 16,384 | 0.00 | 46.23 | 4 |

The real-Kerr corpus had no component sign or zero/non-zero drift, but its worst
direct relative error was `1.5765e-10`.  Four deliberately ill-conditioned
adversarial components changed sign.  Their condition numbers were roughly
`3.4e14` to `4.4e15`, beyond the approximately 48-bit information carried by a
two-float result.  The exact witnesses are preserved in the JSON report:

```text
adversarial-470:component-6
adversarial-514:component-6
adversarial-1528:component-4
adversarial-1862:component-4
```

This is a categorical failure for a general scientific RHS backend.  It cannot
be reclassified on the basis that the measured real-Kerr sample happened to
have zero sign drift.

## DOPRI combination diagnostic

The prototype also batches the fifth-order state and embedded fourth-order
error combination from precomputed stages.  It had zero sign drift, but missed
the diagnostic precision gate:

- fifth-order state: 37.71 worst result-relative bits;
- embedded error: 33.86 worst error normalized to the fifth-order state;
- embedded-error direct relative precision: 22.21 bits.

It is therefore not suitable for adaptive-step acceptance decisions.  The
diagnostic remains non-gating for the narrower RHS milestone, but it is recorded
as failed rather than omitted.

## Throughput

For 32,768 resident precomputed records, median GPU time was `1.0128 ms`
(`32.35 million RHS records/s`); command-complete wall time was `1.1698 ms`
(`28.01 million records/s`).  The current Python binary64 reference processed
`60,946 records/s` on the same generated corpus, so the measured wall-rate ratio
was `459.60x`.

This is a narrowly scoped algebra benchmark.  It excludes Kerr metric
construction, seven adaptive DOPRI samples, surface reintegration, divergent
event topology, path recording, and cache I/O; it must not be projected as an
end-to-end ray speedup.

## Evidence-driven next step

Do not wire this two-word RHS into the renderer.  The next GPU prototype should
retain the shared exponent but use at least a three-word product/accumulator for
the 4x4 dot and quadratic forms, then rerun these four categorical witnesses and
the full corpus.  DOPRI promotion additionally requires a higher-precision
embedded-error path.  Whole-ray fate, target, crossing order, and surface-root
parity remain mandatory after those algebra gates pass.
