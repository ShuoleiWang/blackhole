# Transient scenes: physics, rendering and performance

This note documents the explosive-transient scene: the Type Ia supernova
(`?scene=supernova-ia`). It lists every physical ingredient, where its numbers
come from, how the renderer turns it into pixels, what is deliberately left
out, and how fast it runs on the M3 Pro target.

The scene follows one rule: **every rendered quantity comes from the physical
model module; the shader only transports light**. The model solves the binary,
the explosion kinematics and a radiation-diffusion light curve on the CPU once
(~90 ms in Node), then packs a per-frame state into one uniform block. The
WGSL tracer integrates emission and absorption along each camera ray through
that state.

## Shared machinery (`src/transients/`)

### Units, photometry and exposure

- Radiance is photometric luminance in cd/m² before exposure. A blackbody of
  temperature `T` has bolometric radiance `σT⁴/π`; its luminous fraction and
  chroma come from the CIE 1931 2° colour-matching functions (Wyman, Sloan &
  Shirley 2013 fit) integrated over the Planck spectrum and converted to linear
  sRGB. 6500 K is display-neutral.
- Checks (`tests/transient-physics.test.mjs`): a 5778 K blackbody has a
  luminous efficacy of 92 lm/W, the maximum is 95.5 lm/W near 6600 K, and the
  tabulated chromaticities lie on the Planckian locus.
- A 512-entry texture (log T from 10^2.5 to 10^9.5 K) stores rgb/Y and
  log10 Y, so the shader never evaluates Planck integrals.
- Exposure is applied in the trace pass as log₂, because luminance spans more
  than 25 decades from the quiescent binary to the peak. The photographic
  mode maps a reference surface luminance to display white (key 1.0 before
  the shared ACES/HDR transform). A middle-grey key suits lit scenes, but it
  makes a 20,000 K star look like a grey, diffusely lit ball. The reference
  is the quiescent accretor, the area-median surface of the exploding star
  (so a flash covering a small cap overexposes instead of blacking out the
  rest), or the mean surface luminance of the photospheric disk. Hotter parts
  overexpose; on HDR displays they reach into the extended range.
- During playback the exposure follows a brightening within 0.35 s and a
  dimming within 1.2 s, like a camera or the eye, so a flash first
  overexposes and then adapts. Paused frames use the target directly.
  There is ±4 EV of compensation. The sky-referenced mode fixes the exposure
  to the calibrated panorama, whose median pixel is taken as V = 22
  mag/arcsec² (~1.7×10⁻⁴ cd/m²).

### Camera glare (`src/glare.js`)

A display cannot show a stellar surface ~10⁹ times brighter than daylight.
In a photograph, or to the eye, a dazzling source reads as dazzling because
it overexposes and the optics scatter its light around it. The transient
scenes add that scattered light.

- **Shape.** The CIE 146:2002 general disability-glare function of a young
  observer (25 yr, pigmentation 0.5):
  `L_veil/E_glare = 10/θ³ + (5/θ² + 0.1p/θ)(1 + (A/62.5)⁴) + 0.0025p` sr⁻¹,
  θ in degrees from 0.1°.
- **Strength.** A young eye scatters 7.8% of a point source's light beyond
  1°; a good camera lens has a veiling-glare index of ~2% (ISO 9358). The
  scenes render a camera, so they use the CIE shape at 0.3 of its strength.
- **Implementation.** A Gaussian pyramid. The scene is box-averaged over
  4×4 pixels, then 13-tap downsampled (Jimenez 2014) to levels 2–8. Each
  level's kernel was measured on the GPU at 1.07–1.22 × 2ᵏ pixels, with
  energy conserved to 0.3%. Each level carries the PSF energy of its octave
  annulus; the last level takes everything out to the frame diagonal. This
  reproduces the function to ~10% rms in log over 0.1–6°. The levels are
  recombined by tent upsampling into a quarter-resolution field, which the
  shared post pass adds before tone mapping. Scenes without glare bind a
  1×1 black field.
- **Scope.** Glare models the observer, not the source: no astrophysical
  quantity changes. Diagnostic false-colour frames never receive it, and
  the black-hole scenes do not use it.
- **Cost.** About 1.1 ms per native 3456×2234 frame on an M3 Pro (interleaved
  A/B median, 0.8–1.4 ms across phases).

### Grey flux-limited diffusion in homologous ejecta

`homologous-diffusion.js` solves the comoving radiation energy equation for
freely expanding ejecta (`r = vt`):

```text
∂(E t⁴)/∂t = t⁴ (−∇·F + ρ q),   F = −(c λ(R)/κρ) ∇E,   λ: Levermore–Pomraning
```

- The grid is in velocity space: 192 shells to v_max, with a Marshak outer
  boundary.
- Time steps are implicit and log-spaced from homology (t₀ = 100 s) to
  150 days.
- Opacity `κ(cell, T, t)` and heating `q(cell, t)` are supplied by the scene
  model.
- Outputs:
  - the emergent luminosity and the energy deposition;
  - the internal energy;
  - the τ = 2/3 photosphere velocity and its effective temperature;
  - per-shell radiation temperatures at 288 sample times.
- **Katz integral check.** It verifies energy conservation in the form
  `t E(t) − t₀ E(t₀) = ∫ t (Q − L) dt`. The canonical Type Ia solution closes
  it to 5×10⁻⁴.
- **Tests.** A uniform sphere with constant opacity reproduces Arnett's
  semi-analytic light curve, the solution converges with resolution, and pure
  adiabatic cooling follows `E ∝ 1/t`.

### Emission–absorption ray marching (`transient-wgsl.js`)

`marchEjecta` walks a camera ray through the homologous ejecta. For each
sample it does the following:

- **Retarded time.** The ejecta are evaluated at the emission time
  `t_e = t_obs − (distance to camera)/c`, including the bulk drift of the
  ejecta centre, so the far side is seen younger.
- **Relativistic Doppler.** The shift uses `D = 1/(γ(1 − β·n̂))`: the source
  function is `B(DT)`, and absorption transforms as `α/D`. At peak the
  approaching photosphere is ~3% hotter at the disk centre.
- **Adaptive optical-depth stepping.** The step is
  `Δs = clamp(Δτ(1 + ½τ)/κρ, s_min, chord/48)`, with 72–192 steps depending
  on the quality tier. The march stops once the transmittance falls below
  10⁻³.
- **Photosphere cap.** The source function is capped at the Eddington grey
  atmosphere, `T⁴ ≤ ¾T_eff⁴(τ + ⅔)`, evaluated at the Eddington–Barbier
  depth `min(τ_out, μ(τ_ray + 1))`. An unresolved photosphere then cannot
  overshoot, and the limb darkens as in a real atmosphere.
- **Nebular emission.** When the outward optical depth `τ_out` falls below
  25, the gas also emits its radioactive deposition, `j = ρq/(4π) e^{−τ_out}`,
  with the observed nebular colour temperature (one table fetch per ray).

### Presentation timeline

- Real time spans ten orders of magnitude, from 58 s orbits and a 2 s
  detonation to 150 days.
- `presentation-timeline.js` maps it onto 55 wall seconds using linear and
  logarithmic segments, with an end hold, looping, scrubbing, and a
  rate readout ("1 s = 3.2 d"). In the Type Ia scene the helium sweep plays
  at about a third of real time and the core breakout and donor strike
  (1.9–3.4 s) about 7× slowed, so the explosion itself can be followed.
- Playback speed and scrubbing change only wall-clock presentation, never
  the physics.

### Sparse radiance interpolation

Moving frames (the `emergency`, `survival` and `interactive` tiers) in the
photographic modes use sparse interpolation:

1. A coarse pass traces one node per 4×4 pixels, plus a one-node stencil
   ring. Each node stores its radiance and an object signature (which
   surface, ejecta or stream the ray met).
2. The full-resolution pass interpolates a pixel bilinearly only when all
   12 surrounding nodes agree on the signature and the bilinear error bound
   passes. The bound is 1/8 of the largest second difference; it must stay
   below 0.4% of the local display value, or 0.004 of display white in dark
   regions.
3. Every other pixel is traced in full, so edges, the donor, the stream and
   the shadow-cone rim are never interpolated.

The false-colour diagnostic modes and paused refinement (`balanced`, `fine`)
always trace every pixel. The coarse pass shares the trace pass's sky,
sampler and colour-table bindings (`sharesTraceResources`).

## Type Ia supernova (`?scene=supernova-ia`)

### Channel

The scene renders the **dynamically driven double-degenerate double
detonation ("D6")**. This is the best-supported channel for normal SNe Ia
(Ruiter & Seitenzahl 2025, A&ARv 33, 1). The main lines of evidence:

- nebular Ni/Fe ratios consistent with sub-Chandrasekhar masses in most normal
  SNe Ia (Flörs et al. 2020);
- the double Ca shell of SNR 0509−67.5 (Das et al. 2025);
- the hypervelocity D6 white dwarfs found by Gaia (Shen et al. 2018b;
  El-Badry et al. 2023);
- no detected non-degenerate companions, e.g. SN 2011fe with
  R_progenitor < 0.02 R☉ (Bloom et al. 2012).

The explosion is calibrated to Boos et al. 2021 (model d2e5_m100) and Shen et
al. 2021. The companion interaction follows Tanikawa, Nomoto & Leung 2018.
The light curve and colours are checked against the normal SN Ia SN 2011fe
(Pereira et al. 2013). A Chandrasekhar-mass delayed detonation remains
possible for a minority of events and is not rendered.

### Binary before the explosion

| Quantity | Value | Source |
| --- | --- | --- |
| Primary | 1.00 M☉ C/O white dwarf with a 0.016 M☉ He shell, R = 5716 km, ρ_c = 2.8×10⁷ g/cm³ | cold Chandrasekhar equation of state, solved by shooting (`white-dwarf-structure.js`) |
| Donor | 0.60 M☉ C/O white dwarf, R = 8841 km, T_pole = 12,000 K | same |
| Orbit | a = 26,338 km (the donor fills its Eggleton Roche lobe), P = 58.3 s, v₁ = 1065 km/s, v₂ = 1775 km/s | Eggleton 1983; Kepler |
| Gravitational-wave inspiral time at contact | 321 yr | Peters 1964 |
| Donor surface | the exact Roche equipotential through L1, found by distance-estimate marching of `|Φ − Φ_L1| / |∇Φ|`; von Zeipel gravity darkening T ∝ g^0.25; linear limb darkening u = 0.45 | Roche geometry |
| Accretion stream | ballistic particle from L1 in the co-rotating frame (Coriolis and centrifugal terms), 16 control points | Lubow & Shu 1975 |
| Impact | q = 0.6 is a direct-impact system: the stream strikes the accretor 29.7° ahead of the line of centres. Accretion in the dynamical phase is super-Eddington by orders of magnitude, so the emergent flux of the 14° impact spot saturates at the local Eddington flux g c/κ_es (κ_es = 0.2 cm²/g for He): 1.02×10⁶ K and 3.7×10³⁶ erg/s. The rest of the accretion power is advected into the shell | Guillochon et al. 2010 |
| Irradiation | the donor and the stream absorb and re-radiate half of the light of the spot (a Lambertian emitter) and the accretor that falls on them (Bond albedo 0.5), `T⁴ = T_int⁴ + ½F/σ`, with a penumbra for the finite sources (the accretor subtends ~16°, the spot ~4°) | reflection effect of close binaries |

The pre-explosion luminosity, 3.7×10³⁶ erg/s, is the two photospheres plus
the spot; irradiation is reprocessed light and is not added. The donor's
facing side near L1 is heated to 2.6–2.9×10⁵ K. The stream leaves L1 at the
donor's 12,000 K and is heated to 2.2×10⁵ K there, rising to ~9×10⁵ K near
the spot; gas in the impact region has the spot's temperature. The spot,
the stream and the donor's facing side therefore outshine the 20,000 K
accretor photosphere. The helium shell ignites at the spot.

### Detonations (0–5 s after helium ignition)

- **Helium detonation.** The front sweeps around the accretor and reaches the
  antipode after 2.0 s.
  - Behind the front the surface photosphere follows
    `T = 10⁷ K (Δt/0.01 s)^−½`.
  - The helium ash is kicked outward at 5000 km/s, with a 0.1 s acceleration
    time.
  - The whole sweep radiates ~5×10⁴⁰ erg. Its light, treated as a point
    source of the explosion surface's luminosity, irradiates the donor's
    facing hemisphere.
- **Core ignition.** The converging shock ignites the C/O core at 2.0 s, at
  0.4 R on the antipode side. The core detonation runs at 12,500 km/s.
- **Core breakout.** It starts at the antipode (2.27 s) and ends on the impact
  side (2.64 s).
  - Each surface element flashes at 6×10⁶ K, then fades as
    `(Δt/0.01 s)^−⅓ e^{−Δt/0.1 s}`.
  - Integrated over the surface, that is the ~10⁴⁰ erg radiation content of
    the breakout shell (Piro, Chang & Weinberg 2010 scalings).
  - The flash peaks at 1.7×10⁴¹ erg/s, 60× below the radioactive maximum.
- **Fireball.** The core detonation drives each surface element to the
  stretched ejecta edge `v_max s(θ)` (see below), so the fireball has the
  ejecta's asymmetry: 20% faster towards the helium-ignition pole and 15%
  slower at the antipode, which to first order displaces the sphere towards
  the ignition side.
  Its temperature follows `T_eff = 10⁶ K (t/1 s)^−0.46` until 10 s. It then
  blends in log t onto the same power law, anchored to the diffusion
  solution's photospheric temperature at the 100 s hand-off and lowered for
  the stretched surface's 6% larger area. Luminosity, radius and framing
  are continuous there to 0.3%.
- **Release.** The binary is held at its t = 0 configuration through the
  detonations (14° of orbital motion are neglected). It is released when the
  core detonation unbinds the accretor at 2.32 s; the donor keeps its Roche
  shape until the ejecta strike it.
- **Ejecta strike the donor** at 2.69 s. This is when the opaque explosion
  surface reaches the donor's near side, solved from the same helium-ash and
  core-ash kinematics that the tracer draws.
  - The struck hemisphere flashes at `3×10⁶ K (Δt/0.01 s)^−½` and settles at
    ~10⁵ K.
  - The shadowed hemisphere stays at 12,000 K.
  - The released donor flies off at its orbital speed of 1775 km/s as a
    hypervelocity survivor.
- **Light-travel time.** Every explosion surface point is drawn at its
  retarded time, and each front rises over the light-crossing time R/c.

### Homologous ejecta

- **Density profile.** Exponential, `ρ = M/(8πv_e³t³) e^{−v/v_e}`, with
  M = 1.0 M☉, E_k = 1.15×10⁵¹ erg, v_e = √(E_k/6M) = 3105 km/s and
  v_max = 32,000 km/s. The exponential tail beyond v_max (0.2% of the mass)
  is left out: folding it into the outermost cell would create an
  artificial dense, hot edge shell.
- **Composition by velocity** (Boos et al. 2021; Shen et al. 2018/2021):
  - below 8000 km/s: an ⁵⁶Ni core (87%) with stable Fe-group elements;
  - up to ~16,500 km/s: Si/S/Ca;
  - up to 24,000 km/s: O/C/Mg;
  - above that: helium-detonation ash (He with IME).

  The total is **0.515 M☉ of ⁵⁶Ni**. No ⁵⁶Ni is placed in the thin helium
  ash, so there is no early bump (Polin et al. 2019).
- **Asymmetries** (applied by the tracer at every sample):
  - **Companion shadow cone.** Half-angle 44°, with the interior density ×0.2
    and a rim ×3 over 5° (Tanikawa et al. 2018; Kasen 2010).
  - **Velocity stretch.** +20% toward the helium-ignition pole and −15% at the
    antipode (Boos et al. 2021).
  - **Bulk kick.** The ejecta inherit the primary's 1065 km/s orbital velocity
    (the D6 slingshot).

### Radiation

- **Heating.** Bateman ⁵⁶Ni → ⁵⁶Co → ⁵⁶Fe with τ_Ni = 8.764 d,
  τ_Co = 111.4 d, ε_Ni = 3.9×10¹⁰ and ε_Co = 6.8×10⁹ erg/g/s (Nadyozhin
  1994). 3.2% of the Co power goes into positrons, which deposit locally.
- **γ-ray deposition.** Each shell's escape probability comes from column
  masses along 8 Gauss–Legendre directions with κ_γ = 0.025 cm²/g, and scales
  as t⁻².
- **Optical opacity.** Grey, temperature-dependent line-expansion opacity
  (Pinto & Eastman 2000), calibrated once against the SN 2011fe light curve:
  - Fe-group: 0.18 cm²/g above 10⁴ K, 0.05 at 7500 K, 0.02 below 5000 K
    (recombination);
  - Si/S/Ca: 0.04 below 8500 K, rising to 0.4 above 11,000 K;
  - O: 0.03; He: 0.03;
  - electron scattering (0.2 cm²/g) once fully ionised.
- **Initial thermal energy.** 4×10⁴⁸ erg at homology. The outer layers
  radiate a cooling-envelope plateau of ~1.5×10⁴¹ erg/s over 100–1000 s,
  of the order of the Rabinak & Waxman (2011) white-dwarf estimate
  extrapolated to minutes, followed by the dark phase at ~5×10³⁹ erg/s.
- **Colour.** The chroma uses the colour temperature of the observed intrinsic
  B−V track of normal SNe Ia: first light +0.45, bluest −0.1 at −5 d, −0.03 at
  B maximum, +1.1 at +30 d, then the Lira law (Burns et al. 2014; Phillips et
  al. 1999; conversion by Ballesteros 2012). The bolometric surface brightness
  comes from the diffusion solution. Grey transport cannot produce UV line
  blanketing, so the observed colour stands in for the non-LTE spectrum.

### Light curve against the template

| Observable | Model | Normal SN Ia / SN 2011fe |
| --- | --- | --- |
| ⁵⁶Ni mass | 0.515 M☉ | 0.50 (d2e5_m100); 0.53 ± 0.11 (SN 2011fe) |
| Bolometric peak | 1.09×10⁴³ erg/s at 17.1 d (M_bol = −18.9) | 1.1–1.2×10⁴³ erg/s at ~17–18 d |
| Arnett ratio L_peak / L_decay(t_peak) | 0.98 | ≈1 (Arnett 1982) |
| Bolometric decline in 15 d | 0.89 mag | 0.8–1.0 mag |
| Photosphere at peak | 9460 km/s, colour temperature 10,500 K | Si II 10,400 km/s at maximum |
| γ-ray trapping time t₀ | 36–38 d (fit at 100–150 d) | 30–45 d (Wygoda et al. 2019) |
| L(100 d) | 4.6×10⁴¹ erg/s | ~5×10⁴¹ erg/s |

`tests/supernova-ia.test.mjs` enforces these ranges, the breakout energy
budget, the donor strike timing, the Roche and Kepler relations, the
Eddington-limited impact spot, the continuity of the fireball hand-off and
the median-surface exposure reference. `tests/glare.test.mjs` checks the CIE
glare function, the pyramid level energies and which scenes use glare.

### What an observer sees

| Phase | Appearance |
| --- | --- |
| Binary (−146 → −1 s) | A white-blue accretor with a blazing 10⁶ K impact spot; the spot lights the teardrop donor's facing side to ~3×10⁵ K and heats the stream into a bright ribbon, all three blooming through the camera glare; orbital period 58 s |
| Helium detonation (0–2 s) | A brilliant arc sweeps around the star and overexposes while most of the surface is still quiescent; its light flares the donor's facing limb |
| Core breakout (2.3–2.6 s, played ~7× slowed) | The flash runs from the antipode to the impact side and briefly overexposes the frame before the exposure adapts; the donor transits the fireball as a black silhouette before it is engulfed |
| Fireball (seconds → hours) | A blue-white sphere displaced towards the ignition pole, cooling toward ~10⁴ K, with the shadow cone and its brighter rim on the donor side |
| Rise and peak (0.5–18 d) | The photosphere recedes in velocity while it grows in radius; limb-darkened, Doppler-brightened toward the centre; white at peak |
| Decline (18–60 d) | Orange (B−V ≈ 1.1); the photosphere sinks into the Fe core |
| Nebular (≳ 80 d) | A transparent, warm-white Fe/Co core glowing by radioactive deposition inside dark, transparent outer ejecta |

### Deliberate omissions

- **Radiation transport.** Grey and one-dimensional in velocity. Angular
  structure enters only through the analytic shadow cone, stretch and kick;
  there is no multi-group, non-LTE or line transport, and no polarization.
- **Explosion hydrodynamics.** The detonation fronts are kinematic, not
  hydrodynamic. The pre-homology surface is an opaque photosphere with
  prescribed temperatures, so the ejecta do not wrap around the donor.
- **Early excess.** No early-excess mechanism (thick-shell ⁵⁶Ni, companion
  shocking) is included: the canonical thin-shell model has a smooth rise.
- **Spacetime.** Flat. The white dwarfs' surface redshift is z ≈ 3×10⁻⁴, far
  below any visible effect.
- **WebGL2 fallback.** It draws limb-darkened spheres at the model's
  temperatures, without volumetric transport.

### Performance (M3 Pro, native 3456×2234)

- While the timeline plays, the scheduler holds the native-resolution
  `interactive` tier: 72 ejecta steps with sparse tracing. The camera glare
  adds ~1.1 ms per frame on top of the timings below.
- Paused views refine through the `balanced` (128 steps) and `fine`
  (192 steps) tiers, which trace every pixel and accumulate jittered samples.

| Phase | Dense, per pixel | Sparse (motion tier) | Sparse vs dense |
| --- | --- | --- | --- |
| Binary | 6.2 ms | 3.1 ms | PSNR 82.5 dB, max 4 display codes |
| Helium detonation | 3.6 ms | 2.9 ms | 86.6 dB |
| Fireball (10 min) | 6.3 ms | 3.3 ms | 62.0 dB; 0.0003% of pixels differ by more than 8 codes |
| 5 d | 22.9 ms | 4.2 ms | 56.9 dB, max 5 codes |
| Peak (17 d) | 34.6 ms | 4.8 ms | 58.6 dB, max 4 codes |
| Decline (45 d) | 35.2 ms | 5.4 ms | 64.1 dB, max 7 codes |
| Nebular (100 d) | 30.2 ms | 4.9 ms | 61.5 dB, max 6 codes |

Timings are completed-GPU-queue times per frame. Against a converged
reference (256 steps, 16 jittered samples), the motion tier's remaining error
is single-sample edge aliasing: 98–99.9% of the squared error lies on edges.
Integration depth barely matters, since doubling the motion-tier step budget
improves the photospheric PSNR by only 0.5 dB.
