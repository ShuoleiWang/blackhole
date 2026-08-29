# Metal float-float arithmetic milestone

This directory is an isolated feasibility prototype for a possible Metal
whole-ray backend.  It is **not** a production renderer and is explicitly not
scientifically qualified.

The prototype implements normalized float-float (`hi + lo`) addition,
subtraction, multiplication, division, and square root in Metal.  The shader is
compiled with safe floating-point semantics and precise FP32 library functions;
unsafe fast math is not enabled.  A Swift host runner:

1. constructs deterministic cancellation, exponent-boundary, and randomized
   adversarial inputs;
2. quantizes each CPU `Double` input to a two-float expansion;
3. compares GPU results with binary64 operations over those exact quantized
   values; and
4. reports binary64 ULP error, effective precision, and batched GPU throughput.

The pass gate covers an exponent domain in which both words remain normal FP32
values (including deliberate cancellation).  A separate, non-gating exponent
boundary suite probes low words near FP32 subnormal range.  Metal may flush
those lanes, so the JSON report preserves those failures as a known limitation;
future ray work must introduce explicit scaling or another representation rather
than claiming the issue away.

Run:

```sh
native/metal/run_milestone1.sh
```

The default benchmark dispatches 1,048,576 elements per operation for nine
timed iterations.  For a shorter smoke run:

```sh
BENCHMARK_COUNT=65536 BENCHMARK_ITERATIONS=3 \
  native/metal/run_milestone1.sh
```

Generated binaries and the machine-readable report are kept under
`native/metal/.build/` and are ignored by Git.  The report sets
`productionQualified` to `false`; passing this arithmetic milestone does not
qualify ray integration, surface-event topology, convergence, or any renderer
scientific policy.
