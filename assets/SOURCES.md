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
