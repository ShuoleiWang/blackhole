# Native CPU milestone 4 measured results

Measured on 2026-08-24 on the Apple M3 Pro candidate at commit `5952be2`.
All measurements used the explicit ABI-v3 adapter and the authenticated,
read-only copied partial nested16 cache.  No cache runner was invoked and no
cache file was created or modified.

The strict dylib SHA-256 was
`03a8d632bdf6256eeac174161246a6b605af6b214cfc5dc2866da7ed99f47bdd`.
The adapter descriptor SHA-256 was
`ff0f10c489c9ee7672c274cb31ee0edd07c24c1757c83807b4c1bcb5cd55e33c`.
Every report authenticated 596 task pairs / 38,144 directions before and
after evaluation, with direction-stream SHA-256
`0bf7548ea6f93cd1986a04adca2e42090db5ed4848c51b3f4289348db472bf4e`
and payload-set SHA-256
`010f6a21dcd774eba15afa1760d8a5df76415b9260c231d3cea715a58c8f8d91`.

## Same-direction A/B

Canonical ordinal 369 was evaluated through the complete issued primitive,
including launch construction, independent fine/coarse whole rays, path and
surface evidence reconstruction, receiver/fate policy, primitive descriptor,
narrow transport, and canonical JSON serialization.

| Backend | Repeats | Median wall | Directions/s | Report SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Python binary64 | 1 | 42.1514105 s | 0.0237240 | `2b9d573b3d1fb421769e738044fbc94b469f8b6e1ad4d6a6699bf08fb5145973` |
| Native strict binary64 | 3 | 1.0516793 s | 0.9508602 | `d361657b93cbc9130d204a39128d1ff8687f6d926611049eb24991de055b8a82` |

The measured complete-evaluator wall speedup was **40.0801x**.  Both reports
produced the same authenticated 724-byte transport and all repeats were
byte-exact.  Native backend construction took 0.0635 seconds and is reusable.

## Broad single-process task sample

Three real 64-direction tasks were selected at canonical ordinal starts 704,
31,232, and 37,312.  The 192 directions cover 130 escaped, 25 captured,
14 plunge-sink, 12 return-upper, and 11 return-lower outcomes, including the
outer stress and lower-face regions.

- Wall: 65.9665115 seconds.
- Throughput: 2.9105677 directions/s.
- Mean task evaluator time: 21.9888372 seconds.
- Peak RSS: 77,135,872 bytes.
- Report SHA-256:
  `a71dc36e74b50fd87c960c58cf7b20e49f30529bd4743b7c1f55661432bb616e`.
- Every full transport document was byte-exact against the authenticated
  copied cache.

## Eight-process scaling sample

Eight separate processes evaluated the real tasks starting at ordinals 704,
768, 832, 896, 960, 1,024, 1,088, and 1,152 concurrently.  Every process
independently authenticated the copied cache and every one of the 512
transports was byte-exact.

- Slowest 64-direction evaluator: 34.8917786 seconds.
- Effective concurrent throughput: 14.6739439 directions/s.
- CPU scheduling efficiency relative to eight occupied processes: 95.20%.
- Sum of per-process peak RSS: 612,532,224 bytes.
- Per-process report SHA-256 values:
  - `ff5713765fa8c99ee8ed490512e86e691e5db9d4c5c56d37be1408cd9eb842e9`
  - `867d04b83c608db894177d33963ba956bd7c4914fe46178c3f1667d48cfcf7fb`
  - `949ed9d4d81ae77484f6ec104a197c79ab2af9c50615c8fc91c731f074a6e7be`
  - `bc5ee2828bee2652eefee3dff8bf4bb8e19e805883125e75a0d1c7351eb50ee6`
  - `2c9eeba2154fad6fab79771cb400ead7966d11dfc636baa4349af58f19d46e37`
  - `9bffdd3d2f71ef8b373253f61ec7890b202daf19c249cecc5a00efba72514307`
  - `bc6150c44104fc36c9faaee2a051bd2eda9ef17d1dc07321feaa4ab1c1727426`
  - `7f595e5c988e3f0df573d5cb416f903675cac52b3664196323dc1bc0cd1201c0`

At this measured throughput, the 191,232 directions in the previously
reported 2,988-task remainder correspond to 13,032 seconds, or **3.62 hours**
of evaluator work.  That remainder estimate applies only after the existing
596 tasks have been recomputed and authenticated under the new identity; they
cannot be copied from `d8286...` by assertion.  A fresh 3,584-task identity is
229,376 directions, or **4.34 evaluator hours** at the same measured rate.
The conservative full operational estimate is therefore **5--6 hours** until
the real task producer, receipt serialization, and worker startup are measured
under the new identity.  This replaces the old 104.75-hour estimate.  The
bounded production runner integration is validated below; the fresh full-cache
wall remains to be measured.

## Full authenticated present-prefix replay

Using replay-tool bytes shared by commits `3f8b404` and `380ee437`
(SHA-256
`fa20965105e1d01cf9fd33eaae665e777e565ad245094f61b3585ddba81f4d97`),
the eight-worker read-only replay evaluated ordinals 0 through 38,143 exactly
once.  All 38,144 authenticated transports were byte-exact.  The copied cache
was unchanged before and after the run.

The replay used native scientific key
`4c8c07a05b7bba34e2f387192224f7af01fc5319490cce18f8689ee6b142022c`,
native cache key
`8b607ec4613fd800c503753f0841f3ea307f29a95fb31a722a4bc94c4b3058a1`,
and source/runtime closure manifest
`a71b6014fef41bd2317fd87f70d83af58d1b960838e848b5193d2e5b721655ca`.

- Replay wall: 1,977.8643038 seconds.
- Effective throughput: 19.2854484 directions/s.
- Unique worker processes: 8.
- Sum worker CPU: 15,662.9456650 seconds.
- Maximum worker peak RSS: 79,085,568 bytes.
- Report basis SHA-256:
  `1e25fc8083299925e60ec1dacd2d5205191e33d8bba993bc903f133a83adb9b2`.
- Report artifact SHA-256:
  `c50913c989026a560370c688ae9d6897dcb0a7a95744c81e446d2c6c3d957e25`.

## Active-root production plumbing

The active-root deterministic rebuild reproduced dylib SHA-256
`03a8d632bdf6256eeac174161246a6b605af6b214cfc5dc2866da7ed99f47bdd`.
The C contract, Mach-O audit, ASan/UBSan whole-ray path, 23 arithmetic and
geodesic parity tests, and four complete whole-ray tests passed.  The 169-test
Phase A--D, replay, cache, checkpoint, and CLI suite passed; its two
multiprocessing cases were also executed outside the semaphore-restricted
sandbox and passed.

A bounded real checkpoint then executed 36 native tasks / 448 direction
records with zero reuse, published the native-v2 manifest, and passed an
external digest-anchored verification against the same dylib.  The absolute
library path was absent from the manifest.  The deliberately coarse 4x4x4
kernel was non-qualified as expected; this smoke validates production routing
and publication, not the full scientific grid.

- Checkpoint ID:
  `kerr-returning-radiation-refinement-7586b231d62c5a330881a7e2`.
- Manifest SHA-256:
  `26111ea2ec83d2f446b08671f95efd838267f8571d0ae9020fa92b733de851ce`.
- Smoke scientific key:
  `6ed5b2cf2daf420e005262fa6046a8ca69ec2528c0efbdab0f2db717891bc895`.
- Smoke cache key:
  `890bcf1ea68b3b04146308cdd1def09bd516bbeab6dbe4ec0c398f94f8dcac7a`.

## Qualification boundary

These results qualify deterministic same-code agreement for all 38,144
authenticated present-prefix directions under the new native identity.  The
copied cache remains incomplete: 596 of 3,584 tasks and 38,144 of 229,376
planned directions are present.  The replay does not authorize reuse of the
old `d8286...` cache key, qualify the remaining uncached directions or a
production checkpoint, or provide an independent geodesic or physics oracle.

The active-root build, integration gates, and bounded real task-producer and
checkpoint run are complete.  Production must start a fresh 3,584-task cache
under the new native identity, preserve the exact source/runtime closure and
dylib until external verification finishes, and retain the frozen scientific
policies without widening a threshold after observing the result.
