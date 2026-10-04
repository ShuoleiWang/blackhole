# Real-time strong-field spacetime contract

This note defines the CPU reference implemented in
`src/strong-field-spacetime.js`. It is the physics oracle for the WebGPU
implementation, not the production per-pixel implementation itself.

## Scientific classification

The provider is a **strong-field approximate fast-light metric**. It is not a
constraint-satisfying numerical-relativity spacetime and is not slow-light.
The two body positions are frozen while a ray is integrated. The construction
is a frozen, unboosted variant of the superposed Kerr-Schild strategy developed
for inexpensive (time-dependent, boosted) binary backgrounds by:

- L. Combi and S. M. Ressler, *A binary black hole metric approximation from
  inspiral to merger*, [arXiv:2403.13308](https://arxiv.org/abs/2403.13308).
- L. Combi et al., *Superposed metric for spinning black hole binaries
  approaching merger*,
  [arXiv:2103.15707](https://arxiv.org/abs/2103.15707).
- M. F. Huq, M. W. Choptuik, and R. A. Matzner, *Locating Boosted Kerr and
  Schwarzschild Apparent Horizons*,
  [arXiv:gr-qc/0002076](https://arxiv.org/abs/gr-qc/0002076).

The authors' public reference implementation associated with the first paper is
archived at [Zenodo 10841021](https://doi.org/10.5281/zenodo.10841021).

## Single Kerr-Schild term

With signature `(-,+,+,+)` and geometric units, each rest-frame black hole is

```text
g_mu_nu = eta_mu_nu + 2 H l_mu l_nu
```

where `l_mu` is null with respect to both `eta_mu_nu` and `g_mu_nu`. For a
Cartesian position `x` and spin parameter vector `a = M chi`, the spheroidal
Kerr radius is the non-negative solution

```text
r^2 = 1/2 [rho^2 - a^2
           + sqrt((rho^2 - a^2)^2 + 4 (a dot x)^2)]

H = M r^3 / [r^4 + (a dot x)^2]

l_0 = 1
l = [r x + x cross a + (a dot x) a / r] / (r^2 + a^2).
```

For `a=0`, these reduce to ingoing Schwarzschild Kerr-Schild coordinates:
`r=|x|`, `H=M/r`, and `l=(1,x/r)`.

A Lorentz transformation maps the rest-frame scalar and covector to a frame
in which the hole moves with velocity `v`; that boosted term is an exact vacuum
solution only while its centre moves as `X0 + v t`. The fast-light provider
freezes body positions for the whole ray, so it uses **unboosted** terms at the
frame positions: a frozen boosted term is not a vacuum solution, and its
trailing-side null surface moves out to `r = r+ gamma^2 (1 + v)^2` (1.26 r+ at
`v = 0.116`, 1.74 r+ at `v = 0.27`), beyond the photon orbit once `v > 0.2`.
Unboosted frozen terms keep their horizons at `r = r+` and deflect light no
less accurately than frozen boosted ones. Body velocities remain in the frame
ABI for matter kinematics and for a future slow-light provider.

## Binary superposition and remnant

At each frame, the covariant metric is

```text
g = eta
  + (1-w) [A_A 2 H_A l_A l_A + A_B 2 H_B l_B l_B]
  + w 2 H_R l_R l_R .
```

`w=smootherstep(mergerBlend)` is `C2` at both endpoints. At `w=1` the binary
terms vanish exactly, leaving the analytic remnant Kerr metric. Remnant mass
and spin may be anchored to SXS metadata, but SXS apparent-horizon centroid
coordinates are not accepted as Kerr-Schild body positions.

An optional attenuation of term A near the companion B,

```text
A_A = 1 - exp[-(r_B / sigma)^p]      (p = 4, sigma = 0.35 separation),
```

with the reciprocal definition for B, remains in the provider but is
**disabled by default** (`A_A = A_B = 1`). Its window gradient acts as a
spurious diverging lens around each hole: rays passing one to two shadow
radii from a hole pick up an extra +8 to +30 degrees of deflection at a 10 M
separation. The plain superposition has a smoother, smaller error: the
companion's nearly constant potential couples nonlinearly into each hole's
term, so each shadow is too large by roughly `2 m_companion / d` (about 5% at
the initial 18.6 M separation and 10% at 10 M). Removing that bias would
require writing each term in the local flat frame of its companion's
potential; this is not implemented. Neither variant solves the Hamiltonian or
momentum constraints.

The ring singularity is protected by a small Kerr-radius floor and a finite
`H` ceiling. Every sample records whether regularization was touched.

The merger transition starts before the superposition breaks down. Two
significant terms with anti-aligned null vectors make the metric non-Lorentzian
where `c_A c_B (l_A . l_B)^2 >= 1` with `c = 2 H w`; at the midpoint of two
unboosted Schwarzschild terms `(l_A . l_B)^2 = 4`, so this happens once the
separation drops below `8 sqrt(m_A m_B)` (4 M for SXS:BBH:0001, where
`m_A = m_B = 0.5`). The orbit adapter therefore starts the `C2` metric blend
when the separation reaches `8.5 sqrt(m_A m_B)` (4.25 M, at `t = -18.09 M`;
the crossing is interpolated on the orbit grid rather than taken at the next
grid sample), or at the common horizon if that comes first. Body trajectories
are unchanged.

### Capture

A ray is captured at the innermost photon orbit. For an isolated Kerr hole no
null geodesic has a radial turning point inside the prograde equatorial
circular photon orbit

```text
r_ph = 2m [1 + cos(2/3 acos(-chi))]      (3m for chi = 0, m for chi = 1),
```

so a ray found there while moving inward cannot escape. Time reversal maps
Kerr to Kerr with the opposite spin, which has the same innermost orbit, so the
test also applies to the past-directed rays traced here. In the superposed
metric it is applied to the nearest term, using the Kerr radius of that term
and its gradient for the inward test. A term with metric weight `w` acts like a
hole of mass `w m` near its centre, so its horizon and photon radii are scaled
by `w`; during the merger blend the capture regions therefore grow and shrink
continuously rather than switching on.

A ray inside the unscaled photon orbit of any term present in the metric is
also captured once its coordinate speed `|dx/dt|` falls below 0.2, provided
the weighted fields summed along the terms' null directions,
`|sum_a 2 w_a H_a n_a|`, reach 2/3 there. Around an isolated hole this field is
`2H` (`2m/r` for Schwarzschild, 2/3 at `r = 3m`), and wherever `2H >= 2/3`
every photon whose past-directed ray still reaches the sky moves at an ingoing
Kerr-Schild coordinate speed of at least `1/sqrt(3)` (Schwarzschild, reached
tangentially at `r = 3m`), 0.57 for `chi = 0.69` or 0.55 for `chi = 0.95`
(numerical bound from the Carter-constant radial potential). A slower photon
is outgoing and peeling off a trapped surface: traced backward it only
approaches the horizon. Aligned terms add like one hole. Between two holes
their fields cancel, and there light moving along the axis is slow (about 0.19
at the start of the blend) yet escapes; the field condition excludes that
saddle. During the merger blend the hovering rays sit at the blended metric's
horizon, which the other terms push outside the weight-scaled capture radii
(to 0.9–1.0 M from an individual hole at `t = -10 ... -8 M`), and without the
rule they run until the step budget is spent. On 258 SXS:BBH:0001 probe frames
(every 0.5 M through the blend plus controls, three cameras, `interactive` and
`fine`) the rule changes no outcome and adds no disagreement with the
converged reference.

If a sample fails (non-Lorentzian metric, non-positive Hamiltonian branch or
energy drift beyond the tier gate) or the step budget runs out while the ray is
inside the unscaled photon orbit of any term present in the metric, the ray is
captured: a ray from outside cannot escape from there, and during the merger
blend that region lies inside the common horizon. "Present" means any non-zero
weight: early in the blend the remnant's ring singularity already breaks the
superposition near the centre of mass while its weight is below 1e-4. Every
metric failure found by the CPU oracle in the orbital plane, sampled densely
through the blend, lies inside this region. Elsewhere failing rays stay
`unresolved`. A
horizon padding remains only as a backstop. Validation against exact
Schwarzschild integrals and a converged Kerr reference (`chi = 0.686`) found no
ray misclassified by the photon-orbit test.

## 3+1 fields

The provider decomposes

```text
ds^2 = -alpha^2 dt^2
     + gamma_ij (dx^i + beta^i dt)(dx^j + beta^j dt)
```

and returns:

- lapse `alpha`;
- contravariant shift `beta^i`;
- spatial metric `gamma_ij`;
- inverse spatial metric `gamma^ij`;
- covariant and inverse four-metrics.

It rejects a non-positive spatial metric or non-real lapse. The safe evaluation
API converts such a domain failure to `unresolved`; it must never be sampled as
sky or silently painted as a captured black pixel.

## Coordinate-time null Hamiltonian

For spatial covector momentum `p_i`,

```text
q = sqrt(gamma^ij p_i p_j)
H(t,x,p) = alpha q - beta^i p_i = -p_t

dx^i/dt = alpha gamma^ij p_j / q - beta^i
dp_i/dt = -partial_i H.
```

The CPU oracle uses centered finite differences for `partial_i H` and
`partial_t H`. Production WGSL may use analytic or automatic derivatives, but
must agree with this reference within the declared integration tolerance.

The camera is the static observer `u = d/dt / alpha_s`, with
`alpha_s^2 = alpha^2 - beta_k beta^k`, whose rest space carries the metric
`h_ij = gamma_ij + beta_i beta_j / alpha_s^2`. For a view direction `n^i`
(unit with respect to `h`, pointing from the observer into the scene) the
future-directed photon that arrives at the camera is `p = E_s (u - n)`; with
`E_s = 1`,

```text
p^t = (1 - beta_k n^k / alpha_s) / alpha_s,
p_i = beta_i p^t - gamma_ij n^j.
```

Its conserved energy is `E = -p_t = alpha_s`, so every sky photon arrives with
frequency shift `g = 1 / alpha_s`; the frozen metric is stationary, so this is
uniform over the sky. The Eulerian (normal) observer of ingoing Kerr-Schild
slicing is not suitable: it falls inward at `v = 2M/r` relative to static
observers, which shrinks the shadow by `sqrt((1 - v)/(1 + v))` (5-6% at a
30-40 M camera distance), and it does not match the static camera of the
Schwarzschild scene.

The equations above give the future-directed Hamiltonian flow for that
covector. Ray tracing integrates them backward in coordinate time, i.e. the
past-directed flow `d(x, p)/d tau = -(dx/dt, dp/dt)` with lookback `tau = -t`,
using classical fourth-order Runge-Kutta. Each step is a tier-dependent
fraction of `max(r - w r+, r/2)` for the nearest term, so steps shrink near the
holes and grow geometrically in the far field. Beyond 30 M, rays moving
outward use twice the fraction (at most 2): they only move into weaker field,
and the measured sky directions are unchanged. Inbound rays keep the tier
fraction, since a large step set from the start-of-step distance would
overshoot into the strong field. The WGSL writes RK4 as a stage machine so the
metric provider is evaluated at one call site per iteration.
Because `H` is homogeneous of degree one in `p`, rescaling `p` by
`E / H(x, p)` at each step restores the conserved energy exactly without
changing the spatial path; the pre-projection drift is the reported residual.

The outward direction used at the escape sphere is consequently `-dx/dt`. The
escape sphere lies at least `2 r_camera + 16 M` out. Ingoing Kerr-Schild
coordinates follow the ingoing principal null congruence, so the coordinate
direction of an arriving photon there already matches its asymptotic direction
to about `M b^3 / r^4` (0.004 deg for `b = 20 M` at 96 M); adding the
harmonic-gauge tail `(2M/b)(1 - s/r)` would instead introduce a ~0.1 deg
error, so none is added.
This past-directed convention is essential for the sign of Kerr frame
dragging and for frequency shifts in boosted or rotating spacetimes; merely
launching the view vector as a future-directed ray gives the wrong boundary
problem.

### Production GPU evaluation (low-rank form)

The WGSL does not assemble lapse, shift and a 3x3 inverse metric with their
derivatives. The superposition is a low-rank update of Minkowski,
`g = eta + U C U^T` with columns `l_a = (1, n_a)` (`|n_a| = 1`) and
`C = diag(c_a)`, `c_a = 2 w_a H_a`, so the Woodbury identity gives the exact
inverse

```text
g^-1 = eta - L K L^T,   L_a = (-1, n_a),   K = C (I + N C)^-1,
N_ab = n_a . n_b - 1   (N_aa = 0),
```

a 2x2 system for the binary, 3x3 during the merger blend and a scalar for the
remnant. `det(I + N C) > 0` exactly when `g` is Lorentzian, and
`alpha = 1/sqrt(1 + k)`, `beta^i = sum_a (K 1)_a n_a^i / (1 + k)` with
`k = 1^T K 1`. For `p = (-E, p)` and `u_a = E + n_a . p` the null condition is
`-E^2 + |p|^2 - u^T K u = 0`, a quadratic whose future-directed root is
`H(x, p)`; its discriminant equals `(q / alpha)^2`. The flow follows as

```text
dx/dt = (p - sum_a y_a n_a) / sqrt(D),
dp/dt = grad_x (u^T K u) / (2 sqrt(D)),     y = K u,
d(u^T K u) = 2 y^T du + z^T dC z - y^T dN y,   z = (I + N C)^-1 u,
```

so only per-term scalar gradients are needed: `grad c_a`, and the Jacobian
products `J_a^T v` of `n_a`. For a Kerr term with spin vector `a`, implicit
differentiation of the Kerr radius gives

```text
grad r = r (r^2 x + (a.x) a) / W,              W = r^4 + (a.x)^2,
grad H = m r^2 [(3 (a.x)^2 - r^4) grad r - 2 r (a.x) a] / W^2,
J^T v  = [(w.v) grad r + r v + a x v + (a.v / r) a] / (r^2 + a^2),
w = x - (a.x) a / r^2 - 2 r n,
```

which reduces to `J^T v = (v - n (n.v)) / r` for a Schwarzschild term. A float64
mirror of these formulas matches the CPU oracle to machine precision
(`tests/strong-field-shaders.test.mjs`), and on the GPU the new evaluation
reproduces the previous 3+1 implementation with no changed outcome and
float32-level direction differences. Render pipelines are specialized by the
frame phase (binary, blend, remnant) and by whether the binary bodies spin,
so inactive terms and the unused Kerr branch are compiled out.

The residual diagnostic is the relative energy drift `|H/E - 1|` measured
before each projection. Outside the unscaled photon orbits (see Capture), a
ray whose drift, step budget, or metric domain fails ends as `unresolved`.

## Orbit adapter boundary

The provider consumes `blackhole.pn-eob-orbit-adapter/v1`. An adapter must
declare:

- its PN or EOB dynamics model;
- the coordinate frame mapped to the asymptotically inertial Kerr-Schild
  center-of-mass frame;
- source/provenance;
- explicitly that its body positions are not SXS horizon-centroid
  coordinates.

SXS remains appropriate for the waveform, phase-event alignment, common
apparent-horizon time, final mass, and final spin. Its gauge-dependent
centroids are not physical positions and are rejected by this contract.

### Runtime waveform-to-orbit adapter

`src/strong-field-orbit.js` implements the runtime adapter used by the
interactive scene. It never reads the bundled `sample.separationM` or
`sample.orbitalPhaseRad` centroid channels. Instead it:

1. samples and unwraps the complex SXS `h22` phase on a deterministic grid;
2. masks samples below a relative/absolute amplitude floor and bridges those
   intervals using the surrounding unwrapped phase trend;
3. obtains the positive orbital frequency
   `Omega = |d arg(h22)/dt| / 2`, robustly filters it, and clamps it to a
   declared finite interval;
4. uses the explicit quasi-circular relation
   `x=(M Omega)^(2/3)`, `r=M/x`;
5. places both holes about the analytic center of mass and differentiates the
   same radius/phase model for their boost velocities;
6. joins the inspiral state at the SXS common-horizon event to the
   waveform-peak state with quintic Hermite polynomials matching value, first
   derivative, and second derivative, while the metric blend weight rises
   from the superposition limit (see above) to one at the waveform peak;
7. supplies the SXS remnant mass/spin after mapping the source orbital axis to
   the renderer axis.

The frequency-radius relation is the leading PN relation at wide separation
and the exact circular Schwarzschild test-mass relation. Calling the adapter
“PN/EOB-like” describes its current coordinate-state contract; it is not
itself a complete calibrated EOB Hamiltonian. Between common-horizon formation
and waveform peak the two individual positions are a smooth metric-removal
trajectory, not observable black-hole worldlines. At the peak their metric
weight is exactly zero.

## GPU frame uniform ABI

`blackhole.strong-field-uniforms/v1` contains eleven aligned `vec4<f32>`
records (176 bytes):

1. time, transition, attenuation scale, regularization fraction;
2. body A position and mass;
3. body A velocity and active flag;
4. body A dimensionless spin;
5. body B position and mass;
6. body B velocity and active flag;
7. body B dimensionless spin;
8. remnant position and mass;
9. remnant velocity and active flag;
10. remnant dimensionless spin and raw merger blend;
11. attenuation/regularization controls.

This packet contains only the analytic spacetime state for one frame. Camera
rays are generated separately and must be regenerated for every changed camera
frame; no fixed-camera transfer map is part of this contract.
