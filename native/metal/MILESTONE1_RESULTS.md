# Milestone 1 measured result

Measured on 2026-08-24 with an Apple M3 Pro using Metal Toolchain
`com.apple.dt.toolchain.Metal.32023.883` (`metalfe-32023.883`).  The shader was
compiled with:

```text
-std=metal3.2
-fmetal-math-mode=safe
-fmetal-math-fp32-functions=precise
-ffp-contract=on
```

The generated metallib SHA-256 was
`64cc1679ba76c6464471e69e741740e1f8cb8d425fe0bcaf75c0632de88247e4`.
The machine-readable report SHA-256 was
`2bd8362275ba573dba697e6fc6458a2d8f03c71569c0e9bdc48e3342062a953a`.

## Supported-domain accuracy

Each operation used 4,096 deterministic adversarial vectors.  CPU binary64
operated on the exact `Double(hi) + Double(lo)` values sent to the GPU.

| Operation | Exact binary64 results | Maximum binary64 ULP error | Worst effective bits |
| --- | ---: | ---: | ---: |
| add | 1,910 | 32 | 47.12 |
| subtract | 1,883 | 32 | 47.53 |
| multiply | 604 | 49 | 46.99 |
| divide | 466 | 35 | 47.50 |
| square root | 688 | 22 | 47.87 |

All five operations passed the arithmetic-milestone gate of at least 40
effective bits and at most 8,192 binary64 ULP.  This is deliberately only a
prototype gate; it is not a renderer qualification threshold.

## Exponent-boundary diagnostic

The separate 1,024-vector-per-operation diagnostic demonstrates that the
representation is not uniformly precise across the entire FP32 exponent range.
When a float-float low word enters Metal's subnormal region, the observed worst
effective precision fell to:

| Operation | Worst effective bits |
| --- | ---: |
| add | 0.00 |
| subtract | 0.00 |
| multiply | 30.91 |
| divide | 25.21 |
| square root | 24.57 |

This is a fail-visible result, not a waived gate.  Any later whole-ray prototype
must use explicit dynamic scaling or a representation that avoids subnormal low
words, then repeat the full differential corpus before it can be considered.

## Batched throughput

The benchmark used 1,048,576 elements and nine timed iterations after one
warm-up dispatch.  Times came from `MTLCommandBuffer.gpuStartTime/gpuEndTime`.

| Operation | Million elements per second |
| --- | ---: |
| add | 3,650.39 |
| subtract | 3,638.78 |
| multiply | 3,662.62 |
| divide | 4,643.14 |
| square root | 4,544.21 |

These are isolated, warm batched arithmetic rates.  They do not predict
end-to-end ray throughput because the prototype contains no adaptive stepping,
surface-event topology, divergent control flow, path recording, or cache I/O.

The generated JSON report explicitly records `productionQualified: false`.
