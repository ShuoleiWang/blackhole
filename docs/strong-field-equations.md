# Real-time strong-field spacetime contract

This note defines the CPU reference implemented in
`src/strong-field-spacetime.js`. It is the physics oracle for the WebGPU
implementation, not the production per-pixel implementation itself.

## Scientific classification

The provider is a **strong-field approximate fast-light metric**. It is not a
constraint-satisfying numerical-relativity spacetime and is not slow-light.
The two body positions are frozen while a ray is integrated. The construction
follows the boosted, superposed Kerr-Schild strategy developed for inexpensive
binary backgrounds by:

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

An instantaneous Lorentz transformation maps the rest-frame scalar and
covector to the asymptotically inertial binary frame. Acceleration-dependent
coordinate terms are intentionally omitted: this is the instantaneous boosted
ansatz used by the fast-light approximation, not an exact accelerated-hole
solution.

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

The attenuation applied to term A near the companion B is

```text
A_A = 1 - exp[-(r_B / sigma)^p],
```

with the reciprocal definition for B. The default is `p=4` and
`sigma=0.35 separation`. This removes the companion's singular contribution
from each local hole neighborhood. It is an explicit numerical prescription;
it does not solve the Hamiltonian or momentum constraints.

The ring singularity is protected by a small Kerr-radius floor and a finite
`H` ceiling. A ray must be classified as captured before entering this region.
Every sample records whether regularization was touched. Capture distances are
excision proxies measured outward from the isolated Kerr radius `r_+`; they
are not computed binary apparent or event horizons.

The production quality tiers declare capture padding of `0.30 M`
(`emergency`), `0.24 M` (`survival`), `0.16 M` (`interactive`), `0.08 M`
(`balanced`), and `0.04 M` (`fine`). In ingoing Kerr-Schild coordinate time,
a past-directed shadow ray
can approach the past horizon asymptotically. If the reduced coordinate-time
energy projection fails, the WebGPU tracer therefore has one additional
failure-only capture guard: it may classify a ray as captured only inside
`0.95 M` of either non-spinning individual horizon, tapering to `0.25 M` for
the pinned spinning remnant. Both limits remain inside the corresponding
isolated-Kerr photon shell. Outside that declared bound the same failure stays
`unresolved`.

The larger low-tier surfaces and looser step budgets are latency policies, not
new horizons or accuracy claims. A paused view can progress to `fine`, the
strictest settled tier, which uses the smallest padding, smallest maximum
step, and tightest residual threshold. Even `fine` remains a trace through the
approximate frame-frozen metric.

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

For a camera view direction `n_view^i` pointing from the observer into the
scene, the future-directed photon that arrives at the camera has the opposite
Eulerian spatial direction. After normalization the initial covector is

```text
p_i = -gamma_ij n_view^j.
```

The equations above give the future-directed Hamiltonian flow for that
covector. Ray tracing advances them with a negative coordinate-time increment:

```text
x <- x - Delta_t dx/dt
p <- p - Delta_t dp/dt,     Delta_t > 0.
```

The outward direction used at the escape sphere is consequently `-dx/dt`.
This past-directed convention is essential for the sign of Kerr frame
dragging and for frequency shifts in boosted or rotating spacetimes; merely
launching the view vector as a future-directed ray gives the wrong boundary
problem.

The null diagnostic reconstructs `p_mu=(-H,p_i)` and reports both
`g^mu_nu p_mu p_nu` and a scale-normalized residual. Unless it is already
inside the declared failure-only capture guard above, a ray whose residual,
step budget, or metric domain fails must end as `unresolved`.

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
   derivative, and second derivative;
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
