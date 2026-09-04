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

## Milestone 2: scaled encoding and RHS feasibility

Milestone 2 is a separate fail-closed experiment:

```sh
native/metal/run_milestone2.sh
```

It replaces the absolute low word with a normalized mantissa pair plus a shared
signed binary exponent, exercises the tested binary64-scale exponent range, and
adds:

- a batched Hamiltonian RHS from a precomputed, CPU-authenticated metric sample;
- a deterministic 2,048-record real Kerr and 2,048-record adversarial corpus;
- a current-checkout Python binary64 reference and throughput baseline;
- a non-gating DOPRI5(4) result/error-combination diagnostic; and
- explicit fail-closed arithmetic status probes.

The default full run is expected to exit non-zero at the current checkpoint:
the subnormal defect is fixed and real-Kerr RHS signs match, but four
ill-conditioned adversarial RHS components change sign and the DOPRI diagnostic
misses its precision gate.  See `MILESTONE2_RESULTS.md`.  These failures are the
reason the prototype remains disconnected from the renderer.

For a short plumbing smoke test (not scientific evidence):

```sh
REAL_KERR_COUNT=16 ADVERSARIAL_COUNT=16 \
BENCHMARK_COUNT=1024 BENCHMARK_ITERATIONS=3 \
  native/metal/run_milestone2.sh
```

## Milestone 3: triple-word product/accumulator

Milestone 3 preserves the frozen milestone-2 two-word input corpus but promotes
products, reductions, and DOPRI combinations to a shared-exponent three-float
expansion.  It always replays the four milestone-2 RHS sign-drift witnesses
before the complete 2,048 real-Kerr + 2,048 adversarial corpus:

```sh
native/metal/run_milestone3.sh
```

The gate also covers the fifth-order DOPRI state, a directly accumulated
embedded error, fail-closed malformed input, and the first batched boundary:
one precomputed-stage fifth-order combine plus a signed planar z probe.  The
measured full run passes these gates; see `MILESTONE3_RESULTS.md`.

The result remains `productionQualified=false`.  The planar probe is not the
renderer finite-thickness surface function, root localization, or topology
policy, and no complete adaptive whole ray has been replayed.  The milestone-3
kernel therefore remains isolated from the renderer.
