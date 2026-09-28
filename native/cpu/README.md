# Strict binary64 CPU backend — milestone 3

This directory is an isolated, opt-in numerical prototype. It is not imported
by the production renderer and it does not change any checkpoint or cache
identity. Milestone 3 closes one complete fine-or-coarse Kerr returning-ray
integration in C17 while retaining the reference operation order and every
generic `MetricSample`, controller, termination, and two-surface topology
gate. Nothing in this directory is selected by the production renderer yet.

## Implemented numerical boundary

- A fixed ABI/runtime descriptor and a fail-closed `FE_TONEAREST` guard.
- The finite-input CPython 3.12 `math.fsum` partial-expansion algorithm.
- Row-major 4×4 matrix-vector and compensated bilinear products with Python's
  exact operation and summation order.
- Symmetry and covariant/inverse consistency gates matching `MetricSample`.
- The reference 64-sweep Jacobi certification of one negative and three
  positive metric eigenvalues.
- Byte-exact ingoing-Cartesian Kerr--Schild covariant/inverse tensors and all
  four analytic inverse derivatives. The Kerr discriminant uses the CPython
  3.12 compensated `math.hypot` path rather than the platform `libm hypot`,
  because those differ by one ULP for real production states.
- The normalized null residual used by the geodesic audit.
- Scalar and allocation-free batch Hamiltonian RHS evaluation from an already
  authenticated inverse metric and its four derivatives.
- Direct Kerr Hamiltonian RHS evaluation, including a complete generic audit
  of every constructed metric sample.
- The canonical seven-stage Dormand--Prince 5(4) step with byte-exact fifth
  order state and embedded error.
- An allocation-free same-start probe batch for accepted-step `N`/`2N`
  surface grids. Every offset is independently reintegrated; this API does not
  substitute dense-output interpolation or share numerical stage states.
- ABI-v3 caller-owned model, option, path-segment, crossing, and result
  layouts for one complete resolution of the fixed finite-thickness Kerr
  returning-ray product.
- The exact adaptive DOPRI accept/reject controller, oblate capture/escape
  localization, authenticated initial emitting-face contact, and initial
  worldtube classification.
- A shared two-face `N`/`2N` probe cache, member-specific root bisection,
  stable cross-surface grouping/order checks, transparent continuation,
  return/plunge classification, and complete topology/work diagnostics.
- Fail-closed caller capacity semantics: no result, segment, or crossing is
  published unless every declared output fits.

The API rejects non-finite inputs and a non-default rounding mode. It never
changes the caller's floating-point environment.

## Strict build contract

The Makefile uses Apple Clang C17 with optimization but explicitly disables
fast math, reassociation, reciprocal approximations, fused contraction, and
LTO. The binary audit checks the host Mach-O architecture and exact ABI-v3 export
surface, rejects an embedded LLVM/LTO segment, and rejects fused
multiply-add/subtract instructions in the final disassembly. Compile-time
assertions require radix-2, 53-bit, 8-byte binary64, including gradual
underflow; the runtime descriptor repeats those facts and additionally
requires little-endian storage and `FLT_EVAL_METHOD == 0`. Apple Clang does not expose the
optional `__STDC_IEC_559__` feature macro, so IEEE binary64 is established from
the radix, mantissa, exponent, maximum, and true-minimum constants instead.
`make test` also executes the C contract under AddressSanitizer and
UndefinedBehaviorSanitizer. Darwin's sanitizer runtime does not implement leak
detection, so this target covers address and undefined-behavior faults, not a
platform-unavailable leak detector.

```sh
make -C native/cpu test
make -C native/cpu benchmark
```

The differential suite uses real analytic Kerr samples from the current frozen
Python provider. It demands byte equality, not a tolerance, for randomized
`fsum`, matrix-vector, bilinear, normalized-null-residual, full Kerr
MetricSamples, direct Kerr RHS, scalar DOPRI output/error, and 64-probe batch
results. Metric cases cover both spin signs, extremal and zero spin, the
rationalized oblate-radius branch, axes, equatorial points, large coordinates,
zero/tiny/negative DOPRI steps, and randomized real states. It also checks
fail-closed singularity guards, exact completed-prefix reporting, corrupt
metrics, positive-definite signature rejection, alias safety, and
FP-environment failures. The milestone-3 whole-ray gate additionally replays
eight authenticated nested16 directions at both fine and coarse resolution:
16 rays, 6,242 complete path segments, 14 crossings, 191,098 independent
probe reintegrations, and 394,696 surface evaluations. Every terminal value,
state component, segment, crossing diagnostic/classification, probe/value
counter, and `N`/`2N` maximum must be byte exact. Metric-sample and RHS
counters are required to be positive diagnostics but are not golden-bound.
A separate test proves that
insufficient caller capacity leaves the result buffer unchanged.

On the development M3 Pro, one indicative milestone-2 benchmark run reported 36.8x for 512
scalar full-step calls and 40.6x for one 48-probe batch while preserving byte
parity. These are single-process microbenchmarks of exact Kerr construction,
all generic audits, Hamiltonian RHS, and DOPRI only. They explicitly exclude
the adaptive accept/reject controller, surface values and root topology,
terminal fate policy, path recording, multiprocessing, and cache I/O; they are
not an end-to-end ray or checkpoint speed claim. The milestone-3 authenticated
16-ray replay completed in about 3.35 seconds; ordinal 369 fine resolution
completed in about 0.72 seconds. Those are whole-ray core measurements, but
still exclude Python launch/receiver reconstruction, primitive/transport
serialization, cache receipts, multiprocessing, and checkpoint reduction.

## Deliberate milestone-3 limits

- Fine/coarse comparison, receiver reconstruction/projection, primitive
  descriptor construction, transport narrowing, task serialization, and
  reduction remain product-wrapper-owned Python policy.
- One ABI call owns one resolution; there is no divergent multi-direction or
  canonical 64-direction task batch yet.
- This prototype has not been wired into a renderer, qualified against the
  complete 38,144-direction copied cache, or assigned a new scientific/cache
  identity. The eight-ray corpus is a differential gate, not an independent
  physics oracle or complete phase-space proof.
- Non-finite `math.fsum` behavior is intentionally rejected rather than
  duplicating Python's infinity/NaN semantics, because scientific inputs are
  required to be finite.
- Promotion into a renderer backend requires new source/runtime identities and
  zero-categorical-drift differential qualification against the frozen corpus
  plus an end-to-end speed gate.

Third-party attribution for the finite-input summation and compensated norm
algorithms is recorded in `THIRD_PARTY_NOTICES.md`.
