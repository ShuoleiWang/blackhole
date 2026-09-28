# Offline scientific renderer

This document defines the implemented foundation and the remaining evidence
boundary for the project's high-fidelity offline route.  The browser renderer
is a separate interactive delivery product; offline modules operate on float64
geodesics and linear physical quantities before any HDR, gamut, bloom, or tone
mapping transform.

## Current status

The repository now implements a modular analytic-reference core:

- a full four-dimensional affine null-Hamiltonian integrator;
- metric providers for exact Minkowski, Schwarzschild, and arbitrary-spin Kerr
  in ingoing Cartesian Kerr-Schild coordinates, including exact spatial
  derivatives, finite BL-ZAMO cameras, Carter-constant audits, and oblate
  constant-Kerr-radius capture worldtubes;
- time-dependent `MetricProvider` sampling at the ray's actual `(t,x,y,z)`
  event, without assuming conserved `p_t`;
- a source-to-observer invariant Stokes formal solver with exact homogeneous
  slabs and locally converged coupled dichroic/Faraday paths;
- Hamiltonian-reintegrated, finite-stencil adaptive scalar-medium transfer
  with a single whole-ray error budget and hard work/depth limits;
- explicit reversal from observer-traced ray segments to physical radiative
  propagation order;
- certified Hamiltonian re-sampling inside recorded ray segments plus adaptive
  scalar medium transfer with whole-versus-halves convergence and one
  path-global affine-weighted error budget;
- a stationary Novikov-Thorne/Page-Thorne equatorial emitter with spin- and
  orientation-dependent ISCO, one-face SI flux, colour-corrected Planck
  spectra, and a flux-normalized KERRBB-D20 angular-emission law;
- a literature-bounded stationary finite-thickness calibration geometry with
  upper/lower signed photosphere faces, analytic gradients, Kerr-BL unit
  normals, a shared-probe multi-surface first-visible event layer, and a
  separately authenticated off-equatorial matter frame and local launch
  tetrad;
- deterministic first-visible finite-thickness scalar transfer plus
  independent fine/coarse whole-ray, topology, radius, frequency-shift,
  signed-angle, and spectrum gates, followed by authenticated exact-471-bin
  tile publication and byte-exact same-code full-product replay;
- forward and receiver-centred finite-grid, local-comoving bolometric
  returning-radiation kernels, with face-resolved proper-area transport,
  bidirectional coefficient diagnostics, and a replay-bound thermal profile
  whose absorbed/reradiated fixed point has a non-relaxable relative equation
  residual and publishes physical watts only for certified Kerr geometry;
- ordered equatorial surface events, first-opaque-surface scalar transfer,
  invariant `I_nu,observer = g^3 I_nu,emitter`, and fine/coarse ray, radius,
  frequency-shift, emission-angle, and spectral gates;
- accepted-step first-visible-surface termination with declared `N` versus
  `2N` probe-topology agreement, avoiding integration of hidden post-disk paths;
- deterministic finite-stencil adaptive spectral pixels with explicit
  topology, source coverage, unresolved solid angle, and convergence evidence;
- converged screen-space finite-difference ray-bundle diagnostics with
  Jacobian, parity, singular values, condition and finite magnification gates;
- content-addressed tile jobs with bounded concurrency, atomic receipts,
  corruption detection, and deterministic resume;
- a canonical, bounded direction-task/cache substrate connected to the
  forward and receiver returning-kernel reducers, with production-evaluator
  source/runtime binding, resumable receipts, layout-independent scientific
  identity, and exact direct-versus-cached reduction parity; cached records are
  same-code evidence rather than an independent ray or physics oracle;
- a streaming vacuum spectral compositor for authenticated v1 endpoint maps;
- a fixed binary64 scientific spectral-pixel ABI plus transactional,
  content-addressed frame publication, JSON Schema, and an independent
  structural verifier;
- same-code-family deterministic full-product replay gates and separate zero-
  and finite-thickness selected-ray Boyer-Lindquist fixed-RK4 oracles whose
  integration and event localization do not call the production DOPRI tracer;
- an authenticated official CIE 1931 2-degree, 360--830 nm, 1 nm
  spectral-to-XYZ layer and a separately versioned signed, unclamped
  scene-linear-sRGB derivative, followed by a separately versioned SDR RGB16
  PPM quicklook whose explicit display transform never rewrites the scientific
  linear product.

This is an **analytic calibration and production-architecture layer**.  It can
now produce stationary exact-Kerr zero- and finite-height analytic spectral
frames plus finite-grid returning-energy kernels and a replay-certified
absorbed thermal profile, but it is not an NR-backed image, a GRMHD simulation,
a solved disk atmosphere, a complete KERRBB model with `F_S`, a returning-
radiation-coupled published spectral frame, or an OpenEXR scientific master.

## Ownership and module boundaries

```text
MetricProvider(t,x)                 future NR metric-brick adapter
        ↓
offline.geodesic                    full x^mu,p_mu Hamiltonian
        ↓ observer→source RayPathSegment[]
        ├─ offline.kerr_disk_transfer / kerr_disk_frame
        │      exact Kerr + Page-Thorne surface + g^3 + angular law
        │               ↓
        │   adaptive_frame / spectral_frame / spectral_product
        │               ↓
        │   transactional binary64 tiles
        │        ├─ independent structural verifier
        │        ├─ deterministic same-code full replay
        │        └─ selected-ray independent-RK4 physics oracle
        │               ↓ exact 471-bin visible spectrum
        │          authenticated CIE XYZ
        │                    ↓
        │          signed unclamped linear sRGB
        ├─ offline.kerr_finite_thickness
        │      bounded upper/lower calibration faces
        │               ↓
        │   kerr_finite_thickness_surface / _emitter / _launch
        │      shared-probe events + matter frame + local null launch
        │               ↓
        │   kerr_finite_thickness_transfer / _frame
        │      deterministic first-visible replay + fine/coarse spectrum
        │               ↓
        │   authenticated exact-471-bin tiles + same-code full replay
        │               ↘ independent finite selected-ray BL RK4 oracle
        ├─ offline.returning_radiation
        │      receiver-first local bolometric K + monotone fixed point
        │      ├─ kerr_returning_radiation_rays: forward one-direction fate
        │      ├─ kerr_returning_radiation_receiver_rays: backward direction
        │      ├─ returning_radiation_fate_quadrature: emitter fate estimate
        │      ├─ kerr_returning_radiation_kernel: forward four-face K
        │      └─ kerr_returning_radiation_receiver_kernel: receiver-sky K
        │         (same transport family; F_S and independent oracle absent)
        │
offline.pipeline                    explicit order/frequency boundary
        ↓ source→observer scalar TransferSegment[]
AffineMediumProvider                future GRMHD/electron adapter
        ├─ offline.adaptive_medium_transfer
        │      certified finite-stencil scalar I_nu integration
        └─ offline.radiative_transfer
               invariant spectral/Stokes formal solution
        ↓
scientific artifact writers         linear data + diagnostics + hashes
        ↓
separate display transform          SDR/P3/HDR image or video

offline.job surrounds independent tiles at every expensive producer stage.
```

The interfaces deliberately prevent three common category errors:

1. A realtime frame-frozen metric is not silently reused as NR slow light.
2. Observer-to-source ray storage is not passed directly into an absorptive
   source-to-observer transfer solver.
3. Unresolved rays, unframed polarization, and display RGB transforms are not
   treated as radiative observables.

## Geodesic equations

The offline path integrates the complete Hamiltonian system in binary64:

```text
H = 1/2 g^{mu nu}(x) p_mu p_nu = 0
dx^mu/dlambda = g^{mu nu} p_nu
dp_mu/dlambda = -1/2 partial_mu(g^{alpha beta}) p_alpha p_beta
```

Every Runge-Kutta derivative evaluation calls the provider with the current
coordinate time and position.  Therefore a future time-dependent provider may
change `p_t`; the integrator contains no stationary-energy projection.

Accepted segments are checked at their endpoint and a fifth-order trajectory
midpoint with a momentum-scale-invariant null residual.  Terminal worldtubes
use a generic signed event interface and are localized by reintegrating the
accepted bracket.  `RadialTermination` is an analytic coordinate-sphere
calibration only; its chord-refinement gate prevents a large accepted step
from leaping through the capture sphere.  A future NR adapter must provide its
moving horizon worldtubes, interior-crossing refinement rule, and validity
intervals.

Exact Kerr rays instead use `KerrOblateTermination`, whose signed event is a
constant Kerr-radius ellipsoid. Recorded disk-plane events are Hamiltonian-
reintegrated inside accepted segments and retain root brackets plus
null/interpolation diagnostics. The finite probe hierarchy can reject an
observed topology disagreement, but it cannot prove that an arbitrarily close
even pair of roots never falls between all probes. Products therefore report
finite-probe topology agreement, not surface-complete certification.

Every metric sample is finite, symmetric, mutually inverse, and checked for
the `-+++` Lorentzian signature.  Provider interpolation error is a normalized
quantity with a hard ray-level limit; it is not merely recorded as metadata.

The local DP5(4) error is not called a global image error.  The
`trace_refined_null_geodesic` entry point performs a coarse/fine whole-ray
comparison with distinct tolerance and step-size hierarchies, and exposes
outcome, terminal-event, covector, and discretization agreement.  Product
generators must add convergence gates for escape
direction, frequency transfer, Jacobi fields, and radiance.

The stationary Kerr sampler adds fine/coarse checks for the visible source
prefix, disk radius, `g`, emission angle, spectral bins, and finite-worldtube
escape direction. Its zero-thickness disk rejects an exactly edge-on observer:
a coplanar ray is a degenerate contact problem and must eventually be handled
by a declared finite-thickness photosphere rather than floating-point sign
changes.

`offline.ray_bundle` adds a separate nine-ray screen-space diagnostic. It
compares central differences at `h` and `h/2`, requires every endpoint and
source chart to agree, and reports a scale-normalized 2x2 source Jacobian,
determinant/parity, singular values, condition number, and solid-angle
magnification. Near a singular determinant it returns no finite
magnification; mixed disk/capture/escape topology fails closed. This is a
finite-difference image-plane bundle, not Sachs geodesic-deviation transport:
it contains no along-ray Jacobi field, optical scalars, wavefront curvature,
or time-delay Hessian, and cannot prove caustic completeness.

## Spectral and polarized transfer convention

The transported state is

```text
S = (I,Q,U,V) / nu^3
J = j_nu / nu^2
K = nu K_nu
dS/dlambda = J - K S
```

`TransferSegment` values are ordered from source to observer.  Scalar
homogeneous absorption/emission uses the exact formal solution with `expm1`;
coupled dichroism and Faraday rotation/conversion use an A-stable implicit
midpoint solve with a declared `h ||K||_inf` limit, local full-step versus
two-half-step Richardson gates, and a hard work budget.  Local checking avoids
accepting a large Faraday phase error merely because two whole-segment results
differ by an integer number of turns.  Passive inputs, finite values, Stokes
physicality, pivots, convergence, and final output are fail-closed.

`offline.adaptive_medium_transfer` closes the variable scalar-medium gap left
by one midpoint sample per accepted geodesic segment. Each stencil state is
reintegrated from its recorded segment start by the same four-dimensional
Hamiltonian, metric provider, and DOPRI5(4) method; it is never linearly
interpolated. A whole midpoint slab is compared with two physical-affine half
slabs, and a detected parent mismatch forces an additional nested probe level
so a feature seen by a coarser stencil cannot silently disappear on the next
one. Local error allowances are weighted by interval length divided by the
whole ray length, so splitting one recorded segment into many does not multiply
the declared absolute or relative budget. The relative scale is the maximum
accepted invariant intensity along the physical propagation path and is
reported with the final global limit. Coefficient evaluations,
reintegrations, depth, and minimum affine step are all hard fail-closed limits.

That capability is still finite-stencil and scalar-only. It rejects Q/U/V,
dichroism, and Faraday terms; a discontinuity that does not converge before the
declared limits is rejected. Agreement of the sampled hierarchy cannot exclude
an arbitrarily thin structure missed by every stencil, and does not supply a
GRMHD evolution, an electron model, or a complete GRRT claim.

For a monochromatic endpoint map with
`g = nu_observer / nu_emitter`, a spectral environment is sampled at
`nu_emitter = nu_observer/g` and transformed with `I_nu,observer = g^3
I_nu,emitter`.  An additional `g^4` multiplier must not be applied to the same
spectral value; the fourth power appears only after the appropriate bolometric
frequency integration.

The standalone formal solver accepts polarized coefficients expressed in one
consistently parallel-transported screen basis.  The ray pipeline currently
rejects every Q/U/V emissivity, dichroism, Faraday term, and polarized boundary
because ray records do not yet carry that basis.  No repository producer yet
supplies a physical magnetic field, electron prescription, or transported
Kerr/NR screen basis; consequently no current output may be described as
physical polarization.

The adaptive scalar bridge can now resolve declared smooth finite-gradient
media beyond one coefficient sample per recorded geodesic segment. A future
matter producer must still declare its physical support, pass the finite-
stencil hierarchy, and publish its own fluid/electron provenance. The current
Kerr thin-disk product does not use that bridge: it remains a first-surface
analytic emitter, not a finite-gradient plasma calculation.

## Stationary exact-Kerr thin-disk reference

The analytic reference is deliberately split across `offline.kerr`,
`offline.novikov_thorne`, `offline.kerr_disk`,
`offline.disk_atmosphere`, `offline.kerr_disk_transfer`, and
`offline.kerr_disk_frame`:

1. the generic binary64 Hamiltonian integrator traces photons through the
   exact stationary Kerr metric;
2. independently recovered `E`, `Lz`, Carter `Q`, and Carter `K` audit each
   recorded ray without replacing that integrator;
3. the first equatorial crossing in `[r_ISCO, r_outer]` is an opaque surface,
   while the plunge-region gap and crossings outside the declared disk remain
   transparent;
4. Page-Thorne supplies one-face radial flux and a colour-corrected diluted
   Planck spectrum supplies local isotropic `I_nu`;
5. the default production sampler applies the flux-normalized KERRBB D20 law
   `I(mu) proportional to 1/2 + 3/4 mu`; the angle is evaluated from separated
   Kerr invariants so it stays conditioned near a prograde extremal ISCO;
6. Liouville transfer evaluates the local spectrum at `nu_obs/g` and applies
   exactly one `g^3` factor.

The emitter radius, event, four-velocity, circular-orbit constants, disk
crossing, isotropic spectrum, angular multiplier, and observed spectrum are
jointly cross-validated. Production frame identity accepts only closed
built-in angular and escaped-boundary laws, so an arbitrary callable cannot
reuse a built-in descriptor while silently changing radiance.

This reference still assumes a stationary, geometrically infinitesimal,
optically opaque, zero-torque disk with a frequency-independent analytic
angular law. It has no vertical hydrostatic structure, frequency-dependent
atmosphere, self-irradiation/returning radiation, plunging emission, magnetic
turbulence, corona, polarization, or GRMHD evolution.

## Finite-thickness calibration geometry

`offline.kerr_finite_thickness` implements only the stationary fiducial height
prescription used by [Zhou et al. (2020)](https://arxiv.org/abs/2004.12589),
following Taylor and Reynolds:

```text
H(rho) = 3 dot_m / (2 eta) [1 - sqrt(r_ISCO / rho)]
z_photosphere(rho) = 2 H(rho)
eta = 1 - E_ISCO
```

It supplies bounded upper and lower Boyer-Lindquist faces, outward-positive
signed functions, analytic gradients, Kerr-BL unit normals, the exact
zero-thickness limit, and explicit spin, accretion-ratio, thinness and
inclination gates. The input `dot_m` is the paper's dimensionless
`dot(M)/dot(M)_Edd` parameter. The source does not fix an SI definition for
`dot(M)_Edd` in this height equation, so the module deliberately performs no
kg/s conversion.

`offline.kerr_finite_thickness_surface` connects the two independent signed
faces to the generic accepted-step event layer. One Hamiltonian probe cache is
shared across all face values; separate `N` and `2N` grids must agree on the
observer-to-source visible prefix. Transparent continuations outside the
physical annulus do not create radial sidewalls, the first proven opaque face
terminates the ray, unresolved ordering or tangency fails closed, and hidden
degenerate roots behind that face are never allowed to invalidate the visible
prefix. A meridional Kerr inverse that does not request the undefined axial
azimuth keeps the signed fields valid on the spin axis.

`offline.kerr_finite_thickness_emitter` supplies the complementary local
matter frame. It uses the equatorial circular `Omega(rho)` prescribed by Zhou
et al., but normalizes `u^t` with the exact Kerr metric at the actual upper or
lower photosphere event. The surface normal is metric-dualized and checked as
unit spacelike and orthogonal to the future-timelike material velocity. A
stored past-directed photon therefore has a positive local frequency and a
**signed** face cosine; backside illumination is rejected or explicitly
classified, never hidden by an absolute value.

`offline.kerr_finite_thickness_area` supplies the proper finite-volume measure
used by the conservative returning-radiation kernel. For the stationary
face embedding `X(rho,phi)`, it forms the comoving spatial projector
`h_mu_nu = g_mu_nu + u_mu u_nu`, pulls it back to
`q_AB = h_mu_nu X^mu_,A X^nu_,B`, and integrates
`sqrt(det(q)) d rho d phi` with an independently compared `N`/`2N`
Gauss--Legendre rule. It therefore includes the exact Kerr metric and the
actual off-equatorial prescribed material velocity. A Euclidean
`2 pi rho sqrt(1+z'^2)` area or a constant-Boyer--Lindquist-time coordinate
area is not substituted. This module computes geometry only; it is not a ray
kernel or receiver-area Jacobian.

`offline.kerr_finite_thickness_launch` completes that matter frame with
authenticated meridional and azimuthal face tangents. It constructs a local
future null photon from `(nu, mu, psi)` and independently checks the reversed
past-directed state against the emitter projection. This is the kinematic
launch boundary needed by a later disk-to-disk returning-radiation tracer; it
does not itself trace a geodesic or generate a returning-radiation kernel.

`offline.kerr_finite_thickness_transfer` consumes the exact surface, thermal
proxy, Kerr worldtubes, ray options, surface options, and one recorded ray.
Before applying `I_nu,obs = g^3 f_D20(mu) I_nu,NT(rho,nu/g)`, it deterministically
replays the complete ray from the observer and requires exact first-visible
path/topology equality. This prevents a later photosphere hit, a forged
capture/escape boundary, or a deleted transparent crossing from becoming a
valid pixel. The transfer binds policy-limited tolerances, requires the escape
worldtube to enclose the maximum finite-height oblate radius, and rejects an
observer on or inside the opaque material.

`offline.kerr_finite_thickness_frame` adds independent fine/coarse traces and
compares the complete upper/lower topology, terminal boundary, visible face,
pseudo-cylindrical radius, frequency shift, signed emission cosine, and every
spectral bin. `scripts/render_offline_kerr_finite_thickness_frame.py` integrates
those samples into authenticated exact-471-bin adaptive tiles, and
`offline.kerr_finite_thickness_replay` reconstructs the closed sampler and
replays every public spectral record byte-exactly while binding the current
source closure, official CIE inputs, numeric backend, resource limits, and
TOCTOU checks. This is same-code-family consistency evidence, not an
independent physics oracle. The height calibration rate remains distinct from
the SI thermal accretion rate; no undocumented Eddington conversion is
introduced.

`offline.kerr_finite_thickness_selected_oracle` supplies a separate transport
calibration for selected rays. It evolves the canonical Boyer-Lindquist
Hamiltonian with fixed RK4 and partial-step event bisection, independently
implements both prescribed finite photospheres and their transparent radial
continuations, and recomputes the actual off-equatorial matter velocity,
surface normal, `g`, signed emission cosine, and D20 spectral transfer. Its
`h`/`h/2` checks cover real upper-face, lower-face, capture, and escape rays;
the lower example also retains two transparent ISCO-interior crossings before
the first opaque source. The oracle deliberately shares the public
Page-Thorne radial scalar and the canonical-BL RK4 framework already owned by
the thin-disk selected-ray oracle. It never calls the production Kerr-Schild
geodesic, multi-surface locator, finite-face emitter, transfer, or frame
sampler. These are strong selected-ray transport checks, not an independent
full-frame proof or an independent derivation of the thermal disk model.

This is a published stationary analytic finite-thickness spectral product.
The height profile is still a strictly Newtonian fiducial prescription and `z=2H`
is an assumed photosphere. There is no radial sidewall, vertical hydrostatic
solution, solved atmosphere, returning radiation, radial advection, GRMHD, or
time dependence. The matter velocity is a prescribed off-equatorial circular
reference, not an off-equatorial geodesic.

## Returning-radiation kernel foundation

`offline.returning_radiation` owns a separate receiver-first, axisymmetric
local-comoving bolometric kernel:

```text
F_in[i]  = sum_j K[i,j] F_out[j]
F_out    = F_0 + F_in = F_0 + K F_out
```

Every non-negative `K[receiver][emitter]` coefficient is receiver-defined to
already contain the emitting angular law, bolometric transport, and finite
source/receiver measure. It may be evaluated as the receiver-sky
`mu_i g^4 D20(mu_e)` integral or by the equivalent forward finite-volume
`(dA_e/A_i) w_emitted g^2` deposition. The solver must not apply either set of
factors again. It uses a bounded
Collatz--Wielandt subcriticality certificate followed by a componentwise
monotone Neumann fixed point and a direct equation-residual gate. Critical,
supercritical, non-finite, or insufficiently converged kernels fail closed.
Photon return/capture/escape probabilities have a separate type and cannot be
substituted for this local energy-flux kernel.

`offline.kerr_returning_radiation_rays` supplies the narrower transport
primitive below that kernel contract. From an authenticated finite-height
surface launch it independently traces fine and coarse whole rays, classifies
return to either face, capture, escape, or plunge, and for a return recomputes
the receiving material frame, frequency ratio, signed incidence cosine,
`g^4`, and `mu_receiver g^4`. Both the result and its numerical options are
exact-schema checked and publicly self-replayed; a known separatrix where fine
and coarse disagree fails closed. This is evidence for one local emission
direction only. It contains neither the emission-solid-angle quadrature nor
the receiver-area/Jacobian normalization needed to form `K`, so its scalar
weight must never be inserted directly as a kernel coefficient.

`offline.kerr_returning_radiation_receiver_rays` supplies the complementary
receiver-centred measure. It launches a unit-frequency future incoming photon
from one authenticated receiver face, traces the corresponding past-directed
covector, and accepts only the first front-face disk source. Its directional
integrand is `mu_i g^4 (1/2 + 3 mu_e/4)`. A generic authenticated initial-contact
record in `offline.geodesic` supplies the declared surface side only at affine
zero; every positive-affine probe evaluates the physical surface, so a rapid
return cannot be skipped. Fine/coarse whole rays and public self-replay remain
mandatory. Reversing a certified forward example recovers the source radius
and frequency ratio to about `1e-11`, but this shares the same exact-Kerr
transport family and is not an independent geodesic oracle. One directional
value still lacks the receiver-sky solid-angle integral and is not `K`.

`offline.returning_radiation_fate_quadrature` integrates those certified
directions over one emitter-local outward hemisphere with the KERRBB-D20
emitted-flux measure
`2 mu_e (1/2 + 3 mu_e/4) d mu_e d psi_e/(2 pi)`. The full `mu` and periodic
azimuth grids are compared separately with half-resolution rules, and the
full azimuth grid is also compared with a half-cell phase shift. The D20
measure retains its analytic unit normalization; per-grid renormalization is
not allowed to hide a low-order quadrature error. Published differences are
finite-grid diagnostics rather than rigorous asymptotic error bounds. The
output partitions emitter-local **energy flux** among return, capture,
escape, and plunge sinks; it is not a photon-number probability, receiver
`F_in`, or an area-normalized coefficient of `K`.

`offline.kerr_returning_radiation_kernel` performs the forward finite-volume
construction. It retains four explicit blocks `UU`, `UL`, `LU`, and `LL`, all
indexed as `K[receiver_annulus][emitter_annulus]`, and deposits every returned
direction as
`Delta K = (Delta A_e/A_i) w_emitted g^2`. Capture, escape, and plunge enter
only the separate unshifted fate diagnostics. For every emitter face and
annulus it checks
`sum_i A_i K[i,j]/A_j = <g^2 1_return>_j`; this is an auditable algebraic
energy closure over the same samples, not an independent physical oracle.
Full grids are compared separately with half radial, half emission-cosine,
half azimuth, and half-cell azimuth-phase grids. Fine and coarse whole rays
must also land on the same receiver face and in the same user annulus before
any coefficient is deposited. Adjacent annuli may be merged only by the
proper-area receiver average and emitter-column sum that preserves the action
on piecewise-constant source flux. Reduction to the radial fixed-point kernel
uses `0.5*(UU+UL+LU+LL)` only after upper/lower area and block-symmetry gates.
The full result and any coarsened projection are same-code replayed exactly.

`offline.kerr_returning_radiation_receiver_kernel` independently assembles the
same four face-resolved matrix blocks from the receiver-centred quadrature
`K=(1/pi) integral mu_i g^4 D20(mu_e) dOmega_i`. Each receiver annulus is
averaged over its actual comoving proper area; a sky direction contributes
zero unless the authenticated backward primitive finds a first front-face disk
source. Fine and coarse whole rays must identify the same source face and
source annulus before deposition. Full grids are again compared separately
with half receiver-radius, incidence-cosine, azimuth, and half-cell phase
grids. A typed comparator replays both products and reports every block and
proper-area column-energy difference. This is a strong bidirectional
finite-volume consistency test, but both constructions share the same
exact-Kerr metric, geodesic, event, and surface implementation: it is not an
independent geodesic oracle or a continuum error bound.

This is only the absorbed-and-thermally-reradiated `F_in` part of KERRBB
Appendix D equation D17. The returning-radiation stress/work contribution
`F_S` is not implemented. The generated forward and receiver-centred kernels
are finite-grid, same-code-family results; their coefficient-wise agreement is
not an independent reverse-kernel oracle. There is no spectral redistribution,
scattering,
polarization, solved atmosphere, or GRMHD in these modules; they must not be
described as complete KERRBB.

## Deterministic execution and performance

`offline.job.JobSpec` hashes all scientific and numerical identity inputs:
producer/version, canonical parameters, task geometry, input artifact sizes and
SHA-256 digests, producer source hashes, and record size.  Worker count,
in-flight depth, output location, progress, and wall time are excluded because
they may change throughput but never result bytes.

The default scientific-frame numeric-backend descriptor is deliberately
platform-specific. It binds the CPython version, build, compiler and cache
tag; pointer width and binary64 layout; operating-system release and machine;
and SHA-256 digests of both the Python executable and native `math` extension.
Consequently a Darwin/arm64 tile cannot be reused from a Linux/x86 cache merely
because both hosts report the same Python version. This is a conservative
reproducibility boundary, not a claim that different conforming libm
implementations must produce different physics.

Each task writes a payload and canonical receipt through `fsync` plus atomic
replace.  Resume trusts a task only when job key, task key, byte length, record
count, and payload SHA-256 all agree.  Orphans, partial files, symlinks,
malformed receipts, and corrupted payloads are recomputed.  Results are reduced
in canonical `TaskKey` order even when workers finish out of order, and the
number of submitted futures is bounded.  A persistent per-task file lock
serializes two independent runners targeting the same cache.

`InputArtifact` hashes define job identity; the generic scheduler cannot prove
that an arbitrary producer actually read those bytes.  Product orchestrators
must authenticate a private input snapshot and pass only that snapshot to the
producer.  The vacuum compositor implements this binding; future Kerr/NR/GRMHD
producers must do the same before they may claim deterministic provenance.

The scientific spectral-frame publisher additionally decodes every cached
pixel before publication, binds canonical receipts to the exact `JobSpec`,
writes tiles, sidecar, and manifest in a private sibling staging directory,
and promotes the complete directory with an OS-level no-replace rename. Its
independent verifier rechecks schema, topology, hashes, public pixel
sentinels, convergence/source fields, aggregate statistics, and product
identity. This is strong artifact conformance, but the report deliberately
keeps `physicsVerified=false` because it does not retrace the rays.

`verify_offline_kerr_nt_replay.py` then reconstructs the closed exact-Kerr/NT
sampler and every adaptive pixel from the authenticated manifest, requiring
the complete public spectral ABI to replay byte-for-byte. It rechecks the
current numeric backend and all product-bound source hashes. Its result is
labelled `physicsReplayVerified=true` and
`independentPhysicsOracle=false`: a shared implementation error can replay
perfectly.

`verify_offline_kerr_nt_selected_rays.py` supplies the complementary check. It
uses an independently implemented Boyer-Lindquist canonical Hamiltonian,
fixed-step RK4, partial-step event bisection, and `h` versus `h/2` refinement
for selected disk/capture/escape rays. It independently checks `E`, `Lz`,
Carter invariants, disk radius, `g`, emission angle, and spectral intensity;
only the Page-Thorne radial flux scalar is shared and reported as such. This
is a high-accuracy selected-ray calibration, not a full-frame physics proof.

This removes the architectural need to retain a full image of Python
`RaySolution` objects in memory.  Existing Schwarzschild/Kerr v1 generators
remain byte-stable for now; migrating them onto this runner is a separate
change that must preserve their bundled hashes and independent physics gates.

## Existing scientific products

`blackhole.nr-transfer-map/v1` remains an immutable 32-byte vacuum endpoint
proxy.  It stores direction, frequency shift, lookback time, outcome, capture
target, and numerical diagnostics.  It contains no ray histories, adaptive
subpixel bundle, Jacobi matrix, emissivity, opacity, spectrum, or Stokes data.
The new offline modules do not change that ABI.

The streaming vacuum compositor may consume the authenticated stationary
Schwarzschild/Kerr maps and produce observer-frequency linear radiance chunks
for its independently reproducible Planck environment.  Its versioned JSON
Schema and independent verifier bind the input manifest, copy each 32-byte
state record exactly, authenticate every file, re-evaluate every Planck plus
Liouville sample to one float32 ULP, and recompute the product identity.  That
output is stationary analytic vacuum lensing.  It is not GRRT merely because
it uses spectral bins, and it is not NR merely because v1 has an NR-ready
dataset kind.

`blackhole.scientific-spectral-frame/v1` stores adaptive observer-frame
`I_nu` pixels and their complete convergence/source evidence in a fixed
little-endian binary64 ABI. It is an authenticated stationary analytic
product when fed by either the exact-Kerr/NT or the closed finite-thickness
sampler. Its structural verifier does not turn a recognized producer identifier
into an independent physics proof; each sampler has a separately scoped
same-code full-product replay. Both zero- and finite-thickness references also
have separately scoped canonical-Boyer-Lindquist fixed-RK4 selected-ray
oracles; neither selected subset is an independent full-frame proof.

`offline.cie_color` consumes only the exact frequency grid corresponding to
the bundled official CIE 1931 2-degree table from 360 through 830 nm at 1 nm.
It authenticates the CSV and metadata checksums, DOI, license, row count,
sample row, column sums, and parsed binary64 payload before use. It converts
`I_nu` to `I_lambda`, integrates unnormalised XYZ with the 1 nm trapezoidal
rule, and produces unclamped linear sRGB. UV, X-ray, extra, missing, rounded,
or reordered bins fail closed. Exposure, negative-channel clipping,
luminance-domain Reinhard mapping, uniform gamut scaling, and sRGB encoding
are an explicitly derived display operation; XYZ remains the scientific
colour product. This is a standard-observer calculation, not a camera sensor
or an absolute human-appearance model.

`blackhole.scientific-cie-xyz-frame/v1` is the versioned transactional form
of that scientific XYZ derivation. It first requires the strict spectral-frame
structural verifier and the exact 471-bin grid, then converts one tile at a
time to an 88-byte little-endian binary64 record containing XYZ, propagated
estimated absolute XYZ error, the complete input-record SHA-256, and copied
source/convergence masks. Its manifest binds the input manifest and product,
official CIE hashes, full converter source closure, numeric backend, topology,
and every output tile. The verifier independently authenticates those artifact
contracts and replays every record to at most one binary64 ULP. It deliberately
uses the same canonical CIE integrator, so it is not an independent colour
algorithm oracle; the separate high-precision Decimal Planck-spectrum goldens
are the external numerical gate for that algorithm. No linear sRGB, exposure,
tone mapping, gamut mapping, or display encoding is stored in the primary XYZ
product.

`blackhole.scientific-linear-srgb-frame/v1` is a separate transactional
derivative of the verified XYZ product. Its 120-byte binary64 record stores
signed, unclamped D65 linear-sRGB coordinates, `abs(M)`-propagated estimated
absolute errors, both the XYZ-record and original spectral-record SHA-256
digests, and copied source/convergence masks. Finite negative channels remain
valid out-of-gamut coordinates. The converter and verifier bind the verified
XYZ product, original spectral product, matrix, numeric backend, source
closure, topology, and every tile, and replay each record to at most one ULP.
This layer applies no exposure, clipping, tone map, gamut map, HDR transfer,
sRGB transfer curve, or integer quantization; it is not a display image or an
independent colour-algorithm oracle.

`blackhole.sdr-display-quicklook/v1` is the separate presentation derivative.
It authenticates that linear/XYZ/spectral lineage, applies a fixed manual
exposure, clips negative coordinates only at the display boundary, scales all
non-negative RGB channels uniformly by the Rec.709-luminance Reinhard factor
and any required maximum-channel gamut factor, applies the IEC sRGB transfer
curve, and emits deterministic big-endian RGB16 PPM frames. Its verifier
independently reconstructs every code value and output byte. The product
explicitly reports `isHdr=false`, `inputPhysicsVerified=false`, and never
modifies the signed linear-sRGB records.

## Validation commands

Use the bundled workspace Python when the system interpreter is unavailable:

```bash
python3 -m unittest \
  tests/test_offline_geodesic.py \
  tests/test_offline_radiative_transfer.py \
  tests/test_offline_pipeline.py \
  tests/test_offline_adaptive_medium_transfer.py \
  tests/test_offline_cie_color.py \
  tests/test_offline_cie_product.py \
  tests/test_verify_offline_cie_xyz.py \
  tests/test_offline_linear_rgb_product.py \
  tests/test_verify_offline_linear_srgb.py \
  tests/test_offline_ray_bundle.py \
  tests/test_offline_job.py \
  tests/test_offline_kerr.py \
  tests/test_offline_novikov_thorne.py \
  tests/test_offline_kerr_disk.py \
  tests/test_offline_disk_atmosphere.py \
  tests/test_offline_kerr_disk_transfer.py \
  tests/test_offline_kerr_disk_early_stop.py \
  tests/test_offline_kerr_disk_frame.py \
  tests/test_offline_kerr_finite_thickness.py \
  tests/test_offline_multi_surface.py \
  tests/test_offline_kerr_finite_thickness_surface.py \
  tests/test_offline_kerr_finite_thickness_emitter.py \
  tests/test_offline_kerr_finite_thickness_area.py \
  tests/test_offline_kerr_finite_thickness_launch.py \
  tests/test_offline_kerr_finite_thickness_transfer.py \
  tests/test_offline_kerr_finite_thickness_frame.py \
  tests/test_offline_kerr_finite_thickness_replay_certificate.py \
  tests/test_offline_kerr_finite_thickness_selected_oracle.py \
  tests/test_offline_returning_radiation.py \
  tests/test_offline_kerr_returning_radiation_rays.py \
  tests/test_offline_kerr_returning_radiation_receiver_rays.py \
  tests/test_offline_returning_radiation_fate_quadrature.py \
  tests/test_offline_kerr_returning_radiation_kernel.py \
  tests/test_offline_kerr_returning_radiation_receiver_kernel.py \
  tests/test_offline_kerr_returning_radiation_kernel_jobs.py \
  tests/test_offline_kerr_returning_radiation_thermal_profile.py \
  tests/test_offline_kerr_selected_oracle.py \
  tests/test_offline_adaptive_frame.py \
  tests/test_offline_spectral_frame.py \
  tests/test_offline_spectral_product.py \
  tests/test_render_offline_kerr_nt_frame.py \
  tests/test_render_offline_kerr_finite_thickness_frame.py \
  tests/test_verify_offline_spectral_frame.py \
  tests/test_verify_offline_kerr_nt_replay.py \
  tests/test_verify_offline_kerr_finite_thickness_replay.py \
  tests/test_verify_offline_kerr_nt_selected_rays.py \
  tests/test_offline_vacuum.py \
  tests/test_verify_offline_vacuum.py

python3 scripts/verify_offline_vacuum.py \
  artifacts/offline-vacuum/manifest.json \
  --input-manifest \
  assets/transfer-maps/kerr-remnant-reference-v1/manifest.json

python3 scripts/verify_offline_spectral_frame.py \
  artifacts/offline-kerr-nt/manifest.json

python3 scripts/convert_offline_spectral_to_cie_xyz.py \
  artifacts/offline-kerr-nt/manifest.json \
  artifacts/offline-kerr-nt-cie-xyz

python3 scripts/verify_offline_cie_xyz.py \
  artifacts/offline-kerr-nt-cie-xyz/manifest.json \
  artifacts/offline-kerr-nt/manifest.json

python3 scripts/verify_offline_kerr_nt_replay.py \
  artifacts/offline-kerr-nt/manifest.json

python3 scripts/verify_offline_kerr_nt_selected_rays.py \
  artifacts/offline-kerr-nt/manifest.json

python3 scripts/render_offline_kerr_finite_thickness_frame.py \
  artifacts/offline-kerr-finite-thickness

python3 scripts/verify_offline_kerr_finite_thickness_replay.py \
  artifacts/offline-kerr-finite-thickness/manifest.json

python3 scripts/convert_offline_linear_srgb_to_sdr_display.py \
  artifacts/offline-kerr-finite-thickness-linear-srgb/manifest.json \
  artifacts/offline-kerr-finite-thickness-cie-xyz/manifest.json \
  artifacts/offline-kerr-finite-thickness/manifest.json \
  artifacts/offline-kerr-finite-thickness-sdr \
  --exposure 1e-8

python3 scripts/verify_offline_sdr_display.py \
  artifacts/offline-kerr-finite-thickness-sdr/manifest.json \
  artifacts/offline-kerr-finite-thickness-linear-srgb/manifest.json \
  artifacts/offline-kerr-finite-thickness-cie-xyz/manifest.json \
  artifacts/offline-kerr-finite-thickness/manifest.json
```

The gates cover metric inversion/derivatives, Minkowski straight rays,
Schwarzschild capture and escape around `3 sqrt(3) M`, exact Kerr derivative
and horizon identities, Carter constants, spin-dependent ISCO/Page-Thorne
benchmarks, localized terminal and disk events, whole-ray refinement, vacuum
invariance, LTE slab solutions,
source/observer ordering, Faraday phase and anti-alias convergence,
deterministic job keys, concurrent cache publication, crash resume, corruption
recovery, canonical output order, authenticated endpoint composition,
transactional spectral-frame publication, and adversarial output-contract
verification.

## Next production stages

The implementation order is intentionally evidence-driven:

1. Keep the implemented SDR quicklook downstream of the scientific products;
   add a multilayer linear OpenEXR master and separately identified P3/HDR
   derivatives without moving exposure, gamut, or tone mapping upstream.
2. Connect the bounded direction-task cache to the forward and receiver kernel
   reducers, use the independent selected-ray oracles while raising
   radial/angular resolution, then feed the certified absorbed thermal profile
   into a new spectral-frame product. Add the missing `F_S` physics only as a
   separately identified model; do not fold it invisibly into the zero-
   thickness Novikov-Thorne reference.
3. Add along-ray Sachs/Jacobi transport and a new, separately versioned
   ray-bundle contract; the implemented screen finite-difference diagnostic is
   only a calibration and adaptive-sampling gate.
4. Add pinned four-dimensional NR metric/horizon providers with no time
   extrapolation and measured metric/interpolation/constraint convergence.
5. Add coordinate-matched GRMHD fluid, electron, emissivity, absorption, and
   polarization providers; only this stage can support a GRMHD/GRRT-backed
   radiance claim.
6. Add a multilayer linear OpenEXR scientific master and immutable audit
   manifest.  SDR/P3/HDR deliverables remain derived products.
