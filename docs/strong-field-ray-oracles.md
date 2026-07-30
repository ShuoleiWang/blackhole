# Strong-field ray-oracle acceptance

`tests/strong-field-ray-oracle.test.mjs` is an independent CPU acceptance
layer for the real-time tracer. It evaluates the declared
`SpacetimeProvider`, constructs camera rays in the local ADM-orthonormal
tetrad, stores the future-directed covector of the photon arriving at the
camera, and advances

\[
H=\alpha\sqrt{\gamma^{ij}p_i p_j}-\beta^i p_i=-p_t
\]

with negative coordinate-time steps using a midpoint integrator and central
spatial differences. The resulting path is past-directed. This sign convention
is tested independently of the WGSL text because it controls the direction of
Kerr frame dragging and the escaped sky direction. The oracle does not reuse
the WGSL dual-number derivatives or judge correctness from a rendered image.

In ingoing Kerr-Schild coordinate time, a past-directed shadow ray can approach
the past horizon asymptotically. The CPU trace therefore declares a `0.02 M`
just-outside-horizon excision for its analytic boundary tests. That surface is
an isolated-Kerr numerical proxy, not an apparent/event horizon. It is tighter
than every production quality-tier padding and is intentionally separate from
the WebGPU failure-only capture guard.

The deterministic gates are:

| Limit | Acceptance |
| --- | --- |
| Minkowski | Momentum is constant and the integrated position is a straight line to numerical precision. |
| Single Schwarzschild | The finite-camera screen boundary is obtained from the local tetrad and the exact \(b_\mathrm{crit}=3\sqrt{3}M\). Rays at 0.96 and 1.04 times that screen slope must respectively capture and escape. |
| Wide binary | At impact \(b=30M\) and separation \(10M\), the measured far deflection must agree with the finite-distance form of \(4M/b\) within 24%. The tolerance covers the deliberately approximate superposed metric, finite endpoints, and second-order CPU integrator; it is not a fit to the rendered image. |
| Single Kerr | Exact equatorial Kerr photon-orbit impact parameters set the two horizontal edges. Reversing spin must mirror the edge locations and reverse a probe ray's captured/escaped outcome. |
| Failure semantics | A one-step budget and an invalid metric provider must both return `unresolved`; neither may be relabelled as shadow or sky. |

These tests establish analytic limits, qualitative frame-dragging direction,
past-directed boundary semantics, and fail-closed classification. They do
**not** certify that the superposed binary metric solves the Einstein
constraints, replace full-image Schwarzschild/Kerr comparisons, measure M3 Pro
frame rate, or validate slow-light/NR/GRMHD physics.
