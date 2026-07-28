# Asset sources

## `docs/images/blackhole-galaxy-hero.webp` (README hero)

- Description: project-native 5120×2576 screenshot captured from the WebGPU/Metal path on Apple Silicon, with the science display mode, control panel, and live renderer status visible
- Capture date: 2026-07-14
- Source capture: `ScreenShot_2026-07-14_215821_601.png`
- Sky source included in the rendered image: **ESO/S. Brunier**, `eso0932a`
- License for the incorporated sky panorama: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), under the [ESO image and video usage policy](https://www.eso.org/public/outreach/copyright/)
- Modification: the panorama was distorted by this project's Schwarzschild ray tracer and composited with the rendered accretion disk and analytic stars; the resulting PNG screenshot was encoded as WebP without cropping or AI generation
- Encoding: PNG screenshot converted to WebP with `cwebp -q 92 -m 6 -sharp_yuv`
- Source PNG SHA-256: `a47c8fe25acbec6377f64b1559c43180ad170dc01616db1c0e56d0ae64f451e1`
- WebP SHA-256: `5ae6ad166809fca75af69cfc851620965af750f91ca73d7592334268fe11ddca`

The screenshot demonstrates this repository's renderer but incorporates and
modifies the ESO panorama below. Redistribution outside the context of this
README must be accompanied by the `ESO/S. Brunier` credit, source, and license.

## `scenes/binary-sxs-bbh-0001-v2*` (SXS-derived Phase 2 dynamics)

- Description: compact browser playback track derived from the official
  **SXS:BBH:0001 Lev5** numerical-relativity diagnostics
- Pinned Zenodo record: <https://doi.org/10.5281/zenodo.3273935>
- Catalog status: SXS now marks `SXS:BBH:0001` as deprecated and superseded by
  `SXS:BBH:1132`; this first playback fixture deliberately keeps the exact
  archived Lev5 bytes below instead of silently substituting another simulation
- Generator:
  [`scripts/generate_binary_sxs_dynamics.py`](../scripts/generate_binary_sxs_dynamics.py)
- Generated manifest:
  [`scenes/binary-sxs-bbh-0001-v2.json`](./scenes/binary-sxs-bbh-0001-v2.json)
- Generated samples:
  [`scenes/binary-sxs-bbh-0001-v2.samples.json`](./scenes/binary-sxs-bbh-0001-v2.samples.json)

The generator consumes exactly three official files. The original files are
not bundled in this repository:

| Role | Official file URL | Size (bytes) | MD5 | SHA-256 |
| --- | --- | ---: | --- | --- |
| Lev5 metadata | <https://zenodo.org/api/records/3273935/files/SXS:BBH:0001/Lev5/metadata.json/content> | 4,170 | `099d4c93d9466fe4b7ecad6c94499cf3` | `329d0643f9d33361eafaeae7ef1818dcda3311b33477ecef4f002ead17f42668` |
| A/B/common horizons | <https://zenodo.org/api/records/3273935/files/SXS:BBH:0001/Lev5/Horizons.h5/content> | 3,501,232 | `484ea88842209e64983793159bcc7d7c` | `cf97de4a60a4cd5c6a56f219ea9fa81f1849647f134250e95ae79e40be4dd957` |
| CoM-corrected asymptotic waveform | <https://zenodo.org/api/records/3273935/files/SXS:BBH:0001/Lev5/rhOverM_Asymptotic_GeometricUnits_CoM.h5/content> | 142,641,207 | `c271e0b905c74f434f00c9b14f67850c` | `d760add0693e458781f8db9958b4669971e816d7c026cdbe5f09b7d8fd6bd21f` |

The pinned Zenodo record does **not declare a license**. Accordingly, the Phase
2 manifest records `spdx = null` and
`status = not-declared-in-pinned-zenodo-record`. This repository preserves the
SXS/Zenodo attribution and records integrity metadata, but it does not invent
an SPDX identifier or infer a license from a catalog-wide statement, another
record, or another web page.

The track uses `AhA.dir/CoordCenterInertial.dat` and
`AhB.dir/CoordCenterInertial.dat` for the gauge-dependent coordinate separation
and phase. Its waveform channels are the complex
`Extrapolated_N2.dir/Y_l2_m2.dat` mode from the CoM-corrected waveform file.
The maximum `|h22|` defines protocol `t = 0`; the metadata common-horizon event
is `t = -6.072285420526896 M`. The exact metadata remnant values retained in
the manifest are mass fraction `0.951609417715` and dimensionless spin
`(-7.29520687012e-10, 7.40468371215e-10, 0.686461676493)`.

The generated sample sidecar has 2,732 rows, is 202,606 bytes (approximately
198 KiB), and has SHA-256
`3090229218ca12bf944fbf039e9d744cca367c18af1e6c7d67a8843e311c51f5`.
Its maximum measured orbital-phase interpolation residual is
`0.000644202687 rad` (`6.442e-4 rad`).

These source diagnostics drive the motion and waveform display only. The
sidecar contains no SXS near-zone metric, four-dimensional spacetime, null
geodesics, or ray-transfer data. The existing shader remains a frame-frozen
multi-centre **weak-field fast-light** approximation; the resulting image is
**not NR ray tracing**.

## `scenes/binary-pn-equal-mass-v1.json` (project-generated preview data)

- Description: compact leading-order PN inspiral samples followed by a clearly
  labelled phenomenological merger/remnant display transition; retained as a
  legacy regression asset and no longer used by the runtime binary scene
- Reference configuration: **SXS:BBH:0001**, equal mass, non-spinning,
  quasi-circular
- Pinned dataset: <https://doi.org/10.5281/zenodo.3273935>
- Metadata file: `SXS:BBH:0001/Lev5/metadata.json`
- Metadata MD5 published by Zenodo: `099d4c93d9466fe4b7ecad6c94499cf3`
- License status in the pinned Zenodo record: not declared; no SPDX identifier
  is asserted by this project

The rounded remnant mass and dimensionless spin are taken from the pinned SXS
metadata. The manifest's orbital samples, waveform strip, merger interpolation,
capture surfaces, and rendered rays are generated by this project and are not
extracted from SXS waveform, horizon, or four-dimensional spacetime data. See
[`docs/binary-model.md`](../docs/binary-model.md) for the scientific boundary.
The asset and `scripts/verify_binary_preview.py` remain useful for checking the
legacy PN contract and the unchanged weak-field shader independently of the
Phase 2 SXS-driven dynamics.

## `transfer-maps/contract-fixture-v1/` (project-generated protocol fixture)

- Description: deterministic 4×2 binary-record fixture for the
  `blackhole.nr-transfer-map/v1` data contract
- Dataset ID: `nr-contract-fixture-v1`
- Dataset kind: `synthetic-contract-fixture`
- Origin: project-generated
- Generator: `python3 scripts/generate_nr_contract_fixture.py`
- Schema:
  [`schemas/nr-transfer-map-v1.schema.json`](../schemas/nr-transfer-map-v1.schema.json)
- Manifest:
  [`assets/transfer-maps/contract-fixture-v1/manifest.json`](./transfer-maps/contract-fixture-v1/manifest.json)
- Integrity: the exact manifest bytes are covered by `manifest.sha256`; the
  manifest records the binary chunk and generator/schema source hashes and
  byte sizes
- Artifact location: bundled generator/schema paths use
  `artifactUriBase = repository-root`; portable external datasets may instead
  use the protocol's `manifest-directory` base
- License declaration in manifest: `NOASSERTION`
- Scientific status: `renderable = false`; not numerical-relativity data

The fixture contains hand-authored finite sentinel records for every terminal
ray outcome so that schema, coordinate-frame, hashing, chunking, and
invalid-data behavior can be tested. Its mass, source/protocol time origin,
escape sphere, reference observer, capture surfaces, and ICRS continuation are
all explicitly synthetic; accuracy outcome fractions remain `null`. It has no
source NR simulation and contains no SXS waveform, horizon, near-zone
spacetime, or NR-derived geodesic payload. Passing its validator demonstrates
protocol conformance only; it must not be rendered or described as a physical
black-hole simulation. See
[`docs/nr-transfer-map-v1.md`](../docs/nr-transfer-map-v1.md) for the normative
field and safety semantics.

## `transfer-maps/kerr-remnant-reference-v1/` (project-generated analytic reference)

- Description: deterministic 1024×576 stationary Kerr vacuum transfer map
- Dataset ID: `kerr-remnant-reference-v1`
- Metric: exact analytic Kerr solution, normalized to `M = 1`
- Dimensionless spin: `a/M = 0.686461676493`, aligned with world `+Z`
- Spin provenance:
  [`scenes/binary-sxs-bbh-0001-v2.json`](./scenes/binary-sxs-bbh-0001-v2.json),
  which in turn pins the official `SXS:BBH:0001/Lev5` metadata listed above
- Generator:
  [`scripts/generate_kerr_transfer_map.py`](../scripts/generate_kerr_transfer_map.py)
- Independent verifier:
  [`scripts/verify_kerr_transfer_map.py`](../scripts/verify_kerr_transfer_map.py)
- Scientific specification:
  [`docs/kerr-reference.md`](../docs/kerr-reference.md)
- License declaration in manifest: `NOASSERTION`
- Records: 589,824 (`escaped=558,684`, `captured=31,140`, `unusable=0`)
- Manifest SHA-256:
  `5b0022ab963c0cc35d3d8acab17190bd1294bc72da2b49003d785f964ac81d99`

The SXS-derived input to this product is exactly one remnant-spin parameter.
Its magnitude is computed from the full pinned three-vector, and the manifest
declares a rigid alignment of that vector to world `+Z` rather than dropping
the small transverse components.
The generator does not read an SXS near-zone metric, horizon geometry, or
ray-transfer product. Every pixel is generated locally from the analytic Kerr
metric with a finite-distance Boyer-Lindquist ZAMO camera. The resulting image
is therefore a stationary analytic reference, **not NR ray tracing** and not a
binary-merger reconstruction.

The manifest authenticates the generator, schema, remnant-spin source, and
every binary chunk by byte length and SHA-256. Reproduce the product with:

```bash
python3 scripts/generate_kerr_transfer_map.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_nr_contract.py \
  assets/transfer-maps/kerr-remnant-reference-v1/manifest.json
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_kerr_transfer_map.py
```

## `gaia-edr3-16k.png` (default on native 16K GPUs)

- Title: **The colour of the sky from Gaia's Early Data Release 3 – equirectangular projection**
- Description: full-sky brightness and colour map produced from more than 1.8 billion Gaia EDR3 sources
- Credit: **ESA/Gaia/DPAC; CC BY-SA 3.0 IGO; acknowledgement: A. Moitinho**
- Official image page: <https://sci.esa.int/web/gaia/-/the-colour-of-the-sky-from-gaia-s-early-data-release-3-equirectangular-projection>
- Official 16000x8000 PNG: <https://cdn.sci.esa.int/documents/33580/35361/Gaia_EDR3_flux_cartesian_16k.png/f116e989-fc70-0dac-e453-f1f2141420be?t=1606986368242&version=1.0>
- License: Creative Commons Attribution-ShareAlike 3.0 IGO
- Retrieved: 2026-07-13

`gaia-edr3-16k.png` is the unmodified official 16000x8000 RGB PNG. No resize,
recompression, sharpening, crop, or compositing was applied. It is selected
when the GPU exposes a 2D texture dimension of at least 16000 pixels. The
WebGPU path explicitly requests that native Metal limit instead of accepting
WebGPU's conservative default device limit.

The 236 MiB original is intentionally ignored by Git. Run
`./scripts/fetch_gaia_sky.sh` after cloning; the script downloads this exact
official asset and refuses to install it unless the SHA-256 below matches.

### Integrity

- Local 16K original PNG SHA-256: `10a372d392e9493f6333b7f782e6a973742b71a8da8adc926e0129807462b7e9`

## `milky-way-360-6k.jpg` (photographic fallback)

- Title: **The Milky Way panorama** (`eso0932a`)
- Description: 360-degree photographic panorama of the northern and southern celestial sphere
- Credit / author: **ESO/S. Brunier**
- Official image page: <https://www.eso.org/public/images/eso0932a/>
- Original 6000x3000 JPEG: <https://cdn.eso.org/images/large/eso0932a.jpg>
- ESO image and video usage policy: <https://www.eso.org/public/outreach/copyright/>
- License under that policy: Creative Commons Attribution 4.0 International (CC BY 4.0), with the full credit kept clear and visible
- Retrieved: 2026-07-13

`milky-way-360-6k.jpg` is the unmodified official 6000x3000 download. It keeps
the source sRGB ICC profile and is selected when a GPU supports 6000-pixel but
not 16000-pixel 2D textures, or when the Gaia asset cannot be decoded. No
resize, recompression, crop, or compositing was applied.

## `milky-way-360.webp` (compatibility fallback)

The fallback was resized without cropping to 4096x2048 using Pillow's Lanczos
resampler, then encoded as a high-quality WebP (`quality=95`, `method=6`). The
source ICC colour profile and EXIF metadata were preserved. It is selected only
when the GPU cannot accept the 6K texture or the original request fails.

The shader also adds a deterministic, direction-locked sub-pixel stellar layer.
Those stars are generated after each ray's Schwarzschild escape direction is
known, so they undergo the same gravitational lensing and observer frequency
transfer as the photographic sky. They are not a screen-space overlay.

The source images remain subject to their respective policies. Preserve the
Gaia credit above when the 16K map is displayed, and preserve
**ESO/S. Brunier** wherever the photographic fallback is displayed or
redistributed.

### Integrity

- Local 6K original JPEG SHA-256: `60400c92c54b7c1bd12299c69e83b16e5b6256e7dabacc478c021758ecd28179`
- Derived WebP SHA-256: `ebf6a28a7371fb86297eb9776816815ea0aacef3846563e0ff75a1427be3b223`
