from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import offline.spectral_product as spectral_product_module
from offline.adaptive_frame import (
    AdaptivePixelOptions,
    RayConvergenceAudit,
    SpectralRaySample,
)
from offline.job import InputArtifact, JobRun, JobSpec, TaskKey, run_job
from offline.spectral_frame import SpectralPixelLayout, unpack_spectral_pixel
from offline.spectral_product import (
    AdaptiveSpectralTileProducer,
    SpectralFrameGrid,
    SpectralProductError,
    build_spectral_job_spec,
    default_numeric_backend_descriptor,
    publish_spectral_product,
)


SOURCE_HASH = hashlib.sha256(b"mock spectral tile producer source").hexdigest()
INPUT_HASH = hashlib.sha256(b"mock immutable input").hexdigest()


@dataclass(frozen=True, slots=True)
class MockSpectralSampler:
    implementation_id: str = "tests.mock-spectral-sampler/v1"
    intensity_scale: float = 1.0

    def descriptor(self) -> dict[str, object]:
        return {
            "frequencyFrame": "observer",
            "implementationId": self.implementation_id,
            "intensityScale": self.intensity_scale,
            "scientificStatus": "deterministic-test-double",
        }

    def sample(
        self,
        screen_x: float,
        screen_y: float,
        observer_frequencies_hz: tuple[float, ...],
    ) -> SpectralRaySample:
        del screen_y
        is_disk = screen_x < 0.0
        values = tuple(
            self.intensity_scale * (2.0 if is_disk else 1.0) * (index + 1)
            for index in range(len(observer_frequencies_hz))
        )
        return SpectralRaySample(
            specific_intensities_nu=values,
            absolute_errors_nu=(0.0,) * len(values),
            visible_source="disk" if is_disk else "escaped-boundary",
            topology_signature="disk" if is_disk else "escape",
            frequency_shift_g=1.1 if is_disk else None,
            escape_direction=None if is_disk else (0.0, 0.0, 1.0),
            convergence_audit=RayConvergenceAudit(
                maximum_null_residual=1.0e-12,
                maximum_metric_interpolation_error=2.0e-13,
                terminal_event_difference_m=3.0e-10,
                terminal_covector_relative_difference=4.0e-11,
                disk_radius_difference_m=5.0e-10 if is_disk else 0.0,
                relative_g_difference=6.0e-12 if is_disk else 0.0,
                surface_bracket_affine_width=7.0e-9 if is_disk else 0.0,
                accepted_steps=12,
                rejected_steps=1,
                ray_gate_passed=True,
                source_gate_passed=True,
                transfer_gate_passed=True,
            ),
            ray_converged=True,
        )


@dataclass(frozen=True, slots=True)
class ProductFixture:
    root: Path
    cache: Path
    output: Path
    layout: SpectralPixelLayout
    grid: SpectralFrameGrid
    options: AdaptivePixelOptions
    backend: dict[str, object]
    sampler: MockSpectralSampler
    spec: JobSpec
    producer: AdaptiveSpectralTileProducer
    run: object
    publication: object


def make_options(frequency_count: int = 2) -> AdaptivePixelOptions:
    return AdaptivePixelOptions(
        maximum_depth=0,
        maximum_ray_evaluations=64,
        radiance_absolute_tolerances=(1.0e-12,) * frequency_count,
        radiance_relative_tolerance=1.0e-10,
        radiance_guard_ceilings=(100.0,) * frequency_count,
        weighted_log_g_tolerance=1.0,
        weighted_direction_tolerance_rad=1.0,
    )


def create_product_fixture(root: Path) -> ProductFixture:
    cache = root / "cache"
    output = root / "product"
    layout = SpectralPixelLayout((4.0e14, 6.0e14))
    grid = SpectralFrameGrid(
        width_pixels=4,
        height_pixels=2,
        screen_x_min=-1.0,
        screen_x_max=1.0,
        screen_y_min=-0.5,
        screen_y_max=0.5,
    )
    options = make_options()
    backend = {
        "floatEvaluation": "test deterministic binary64",
        "implementationId": "tests.mock-numeric-backend/v1",
    }
    sampler = MockSpectralSampler()
    spec = build_spectral_job_spec(
        layout,
        grid,
        options,
        sampler.descriptor(),
        tile_width=2,
        tile_height=1,
        numeric_backend=backend,
        inputs=(InputArtifact("mock-input.bin", 20, INPUT_HASH),),
        producer_source_hashes=(SOURCE_HASH,),
    )
    producer = AdaptiveSpectralTileProducer(
        sampler,
        layout,
        grid,
        options,
        backend,
    )
    job_run = run_job(spec, producer, cache, jobs=1)
    publication = publish_spectral_product(
        output,
        job_spec=spec,
        job_run=job_run,
        layout=layout,
        grid=grid,
        options=options,
        sampler_descriptor=sampler.descriptor(),
        numeric_backend=backend,
    )
    return ProductFixture(
        root,
        cache,
        output,
        layout,
        grid,
        options,
        backend,
        sampler,
        spec,
        producer,
        job_run,
        publication,
    )


class OfflineSpectralProductTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temporary.name).resolve(strict=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_mock_sampler_job_packs_canonical_tiles_and_public_records(self) -> None:
        fixture = create_product_fixture(self.root)
        self.assertEqual(
            tuple(result.key for result in fixture.run.results),
            fixture.grid.tasks(2, 1),
        )
        self.assertEqual(fixture.publication.tile_count, 4)
        self.assertEqual(fixture.publication.record_count, 8)
        for result in fixture.run.results:
            self.assertEqual(result.record_count, result.key.width * result.key.height)
            payload = result.payload_path.read_bytes()
            for offset in range(0, len(payload), fixture.layout.record_bytes):
                record = unpack_spectral_pixel(
                    fixture.layout,
                    payload[offset : offset + fixture.layout.record_bytes],
                )
                self.assertGreater(record.sample_count, 0)

    def test_publication_is_deterministic_and_never_overwrites(self) -> None:
        fixture = create_product_fixture(self.root)
        second_output = self.root / "second-product"
        second = publish_spectral_product(
            second_output,
            job_spec=fixture.spec,
            job_run=fixture.run,
            layout=fixture.layout,
            grid=fixture.grid,
            options=fixture.options,
            sampler_descriptor=fixture.sampler.descriptor(),
            numeric_backend=fixture.backend,
        )
        self.assertEqual(second.product_id, fixture.publication.product_id)
        self.assertEqual(second.product_sha256, fixture.publication.product_sha256)
        self.assertEqual(
            second.manifest_path.read_bytes(),
            fixture.publication.manifest_path.read_bytes(),
        )
        with self.assertRaisesRegex(SpectralProductError, "refusing to overwrite"):
            publish_spectral_product(
                second_output,
                job_spec=fixture.spec,
                job_run=fixture.run,
                layout=fixture.layout,
                grid=fixture.grid,
                options=fixture.options,
                sampler_descriptor=fixture.sampler.descriptor(),
                numeric_backend=fixture.backend,
            )

    def test_job_identity_binds_science_backend_and_source_hashes(self) -> None:
        fixture = create_product_fixture(self.root)
        variants = (
            build_spectral_job_spec(
                fixture.layout,
                fixture.grid,
                replace(
                    fixture.options,
                    radiance_relative_tolerance=2.0e-10,
                ),
                fixture.sampler.descriptor(),
                tile_width=2,
                tile_height=1,
                numeric_backend=fixture.backend,
                producer_source_hashes=(SOURCE_HASH,),
            ),
            build_spectral_job_spec(
                fixture.layout,
                fixture.grid,
                fixture.options,
                MockSpectralSampler(intensity_scale=2.0).descriptor(),
                tile_width=2,
                tile_height=1,
                numeric_backend=fixture.backend,
                producer_source_hashes=(SOURCE_HASH,),
            ),
            build_spectral_job_spec(
                fixture.layout,
                fixture.grid,
                fixture.options,
                fixture.sampler.descriptor(),
                tile_width=2,
                tile_height=1,
                numeric_backend={
                    **fixture.backend,
                    "floatEvaluation": "different backend",
                },
                producer_source_hashes=(SOURCE_HASH,),
            ),
            build_spectral_job_spec(
                fixture.layout,
                fixture.grid,
                fixture.options,
                fixture.sampler.descriptor(),
                tile_width=2,
                tile_height=1,
                numeric_backend=fixture.backend,
                producer_source_hashes=("f" * 64,),
            ),
        )
        self.assertEqual(
            len({fixture.spec.job_key, *(variant.job_key for variant in variants)}),
            5,
        )

    def test_default_numeric_backend_binds_native_runtime_identity(self) -> None:
        backend = default_numeric_backend_descriptor()
        self.assertEqual(
            backend["implementationId"],
            "cpython-binary64-struct-libm/v2",
        )
        self.assertEqual(len(backend["mathExtension"]["sha256"]), 64)
        self.assertEqual(len(backend["pythonExecutable"]["sha256"]), 64)
        fixture = create_product_fixture(self.root)
        variants = []
        for mutation in (
            {"machine": f"{backend['machine']}-different"},
            {
                "mathExtension": {
                    **backend["mathExtension"],
                    "sha256": "f" * 64,
                }
            },
        ):
            variants.append(
                build_spectral_job_spec(
                    fixture.layout,
                    fixture.grid,
                    fixture.options,
                    fixture.sampler.descriptor(),
                    tile_width=2,
                    tile_height=1,
                    numeric_backend={**backend, **mutation},
                    producer_source_hashes=(SOURCE_HASH,),
                )
            )
        baseline = build_spectral_job_spec(
            fixture.layout,
            fixture.grid,
            fixture.options,
            fixture.sampler.descriptor(),
            tile_width=2,
            tile_height=1,
            numeric_backend=backend,
            producer_source_hashes=(SOURCE_HASH,),
        )
        self.assertEqual(
            len({baseline.job_key, *(variant.job_key for variant in variants)}),
            3,
        )

    def test_publication_rejects_overlap_with_matching_total_area(
        self,
    ) -> None:
        fixture = create_product_fixture(self.root)
        overlap_tasks = (
            TaskKey(0, 0, 0, 3, 2),
            TaskKey(0, 0, 2, 1, 2),
        )
        overlap_spec = JobSpec(
            producer=fixture.spec.producer,
            algorithm_version=fixture.spec.algorithm_version,
            tasks=overlap_tasks,
            parameters=fixture.spec.as_dict()["parameters"],
            inputs=fixture.spec.inputs,
            producer_source_hashes=fixture.spec.producer_source_hashes,
            record_bytes=fixture.layout.record_bytes,
        )
        overlap_run = run_job(
            overlap_spec,
            fixture.producer,
            self.root / "overlap-cache",
            jobs=1,
        )
        with self.assertRaisesRegex(SpectralProductError, "overlap|coverage gap"):
            publish_spectral_product(
                self.root / "overlap-product",
                job_spec=overlap_spec,
                job_run=overlap_run,
                layout=fixture.layout,
                grid=fixture.grid,
                options=fixture.options,
                sampler_descriptor=fixture.sampler.descriptor(),
                numeric_backend=fixture.backend,
            )

    def test_publication_rejects_cache_payload_hash_drift(self) -> None:
        fixture = create_product_fixture(self.root)
        first = fixture.run.results[0]
        payload = bytearray(first.payload_path.read_bytes())
        payload[0] ^= 1
        first.payload_path.write_bytes(payload)
        with self.assertRaisesRegex(SpectralProductError, "hash is inconsistent"):
            corrupt_output = self.root / "corrupt-product"
            publish_spectral_product(
                corrupt_output,
                job_spec=fixture.spec,
                job_run=fixture.run,
                layout=fixture.layout,
                grid=fixture.grid,
                options=fixture.options,
                sampler_descriptor=fixture.sampler.descriptor(),
                numeric_backend=fixture.backend,
            )
        self.assertFalse(corrupt_output.exists())

    def test_publication_rejects_symlinked_deep_cache_ancestor(self) -> None:
        fixture = create_product_fixture(self.root)
        linked_root = self.root / "linked-root"
        linked_root.symlink_to(self.root, target_is_directory=True)
        redirected_results = tuple(
            replace(
                result,
                payload_path=(
                    linked_root / result.payload_path.relative_to(self.root)
                ),
                receipt_path=(
                    linked_root / result.receipt_path.relative_to(self.root)
                ),
            )
            for result in fixture.run.results
        )
        redirected_run = replace(fixture.run, results=redirected_results)
        output = self.root / "ancestor-symlink-product"
        with self.assertRaisesRegex(
            SpectralProductError,
            "without following symlinks",
        ):
            publish_spectral_product(
                output,
                job_spec=fixture.spec,
                job_run=redirected_run,
                layout=fixture.layout,
                grid=fixture.grid,
                options=fixture.options,
                sampler_descriptor=fixture.sampler.descriptor(),
                numeric_backend=fixture.backend,
            )
        self.assertFalse(output.exists())

    def test_publication_rejects_final_cache_file_symlink(self) -> None:
        fixture = create_product_fixture(self.root)
        first = fixture.run.results[0]
        displaced = first.payload_path.with_name(
            f"{first.payload_path.name}.displaced"
        )
        first.payload_path.rename(displaced)
        first.payload_path.symlink_to(displaced.name)
        output = self.root / "final-symlink-product"
        with self.assertRaisesRegex(
            SpectralProductError,
            "without following symlinks",
        ):
            publish_spectral_product(
                output,
                job_spec=fixture.spec,
                job_run=fixture.run,
                layout=fixture.layout,
                grid=fixture.grid,
                options=fixture.options,
                sampler_descriptor=fixture.sampler.descriptor(),
                numeric_backend=fixture.backend,
            )
        self.assertFalse(output.exists())

    def test_oversized_cache_payload_is_rejected_before_read(self) -> None:
        fixture = create_product_fixture(self.root)
        first = fixture.run.results[0]
        first.payload_path.write_bytes(first.payload_path.read_bytes() + b"x")
        output = self.root / "oversized-cache-product"
        with patch(
            "offline.spectral_product.os.read",
            side_effect=AssertionError("oversized payload must not be read"),
        ) as mocked_read:
            with self.assertRaisesRegex(
                SpectralProductError,
                "hard byte limit",
            ):
                publish_spectral_product(
                    output,
                    job_spec=fixture.spec,
                    job_run=fixture.run,
                    layout=fixture.layout,
                    grid=fixture.grid,
                    options=fixture.options,
                    sampler_descriptor=fixture.sampler.descriptor(),
                    numeric_backend=fixture.backend,
                )
        mocked_read.assert_not_called()
        self.assertFalse(output.exists())

    def test_stable_reader_rejects_same_inode_mutation(self) -> None:
        path = self.root / "same-inode.bin"
        path.write_bytes(b"stable payload")
        original_inode = path.stat().st_ino
        real_read = os.read
        mutated = False

        def mutate_after_read(descriptor: int, count: int) -> bytes:
            nonlocal mutated
            payload = real_read(descriptor, count)
            if payload and not mutated:
                mutated = True
                current_mode = stat.S_IMODE(path.stat().st_mode)
                path.chmod(current_mode ^ stat.S_IXUSR)
                self.assertEqual(path.stat().st_ino, original_inode)
            return payload

        with patch(
            "offline.spectral_product.os.read",
            side_effect=mutate_after_read,
        ):
            with self.assertRaisesRegex(
                SpectralProductError,
                "changed while being read",
            ):
                spectral_product_module._read_stable_regular_file(
                    path,
                    "same-inode test file",
                    maximum_bytes=len(b"stable payload"),
                )
        self.assertTrue(mutated)

    def test_stable_reader_rejects_directory_swap_with_same_bytes(self) -> None:
        directory = self.root / "read-swap"
        nested = directory / "one" / "two"
        nested.mkdir(parents=True)
        path = nested / "payload.bin"
        payload = b"same scientific bytes"
        path.write_bytes(payload)
        displaced = self.root / "read-swap-displaced"
        real_read = os.read
        swapped = False

        def swap_after_read(descriptor: int, count: int) -> bytes:
            nonlocal swapped
            block = real_read(descriptor, count)
            if block and not swapped:
                swapped = True
                directory.rename(displaced)
                path.parent.mkdir(parents=True)
                path.write_bytes(payload)
            return block

        with patch(
            "offline.spectral_product.os.read",
            side_effect=swap_after_read,
        ):
            with self.assertRaisesRegex(
                SpectralProductError,
                "path changed while being read",
            ):
                spectral_product_module._read_stable_regular_file(
                    path,
                    "directory-swap test file",
                    maximum_bytes=len(payload),
                )
        self.assertTrue(swapped)

    def test_old_tile_receipts_cannot_be_relabelled_as_a_new_job(self) -> None:
        fixture = create_product_fixture(self.root)
        changed_sampler = MockSpectralSampler(intensity_scale=2.0)
        changed_spec = build_spectral_job_spec(
            fixture.layout,
            fixture.grid,
            fixture.options,
            changed_sampler.descriptor(),
            tile_width=2,
            tile_height=1,
            numeric_backend=fixture.backend,
            producer_source_hashes=(SOURCE_HASH,),
        )
        forged_run = JobRun(
            job_key=changed_spec.job_key,
            results=fixture.run.results,
            reused_tasks=len(fixture.run.results),
            executed_tasks=0,
            max_in_flight_observed=0,
        )
        output = self.root / "forged-product"
        with self.assertRaisesRegex(SpectralProductError, "receipt"):
            publish_spectral_product(
                output,
                job_spec=changed_spec,
                job_run=forged_run,
                layout=fixture.layout,
                grid=fixture.grid,
                options=fixture.options,
                sampler_descriptor=changed_sampler.descriptor(),
                numeric_backend=fixture.backend,
            )
        self.assertFalse(output.exists())

    def test_publication_failures_never_expose_final_output_and_can_retry(
        self,
    ) -> None:
        fixture = create_product_fixture(self.root)
        real_write = spectral_product_module._atomic_write_no_replace
        for fail_at in (2, 5, 6):
            output = self.root / f"fault-{fail_at}"
            calls = 0

            def injected_write(path: Path, payload: bytes) -> None:
                nonlocal calls
                calls += 1
                if calls == fail_at:
                    raise OSError(f"injected publication failure {fail_at}")
                real_write(path, payload)

            with self.subTest(fail_at=fail_at):
                with patch(
                    "offline.spectral_product._atomic_write_no_replace",
                    side_effect=injected_write,
                ):
                    with self.assertRaisesRegex(OSError, "injected"):
                        publish_spectral_product(
                            output,
                            job_spec=fixture.spec,
                            job_run=fixture.run,
                            layout=fixture.layout,
                            grid=fixture.grid,
                            options=fixture.options,
                            sampler_descriptor=fixture.sampler.descriptor(),
                            numeric_backend=fixture.backend,
                        )
                self.assertFalse(output.exists())
                self.assertEqual(
                    list(output.parent.glob(f".{output.name}.staging-*")),
                    [],
                )
                retried = publish_spectral_product(
                    output,
                    job_spec=fixture.spec,
                    job_run=fixture.run,
                    layout=fixture.layout,
                    grid=fixture.grid,
                    options=fixture.options,
                    sampler_descriptor=fixture.sampler.descriptor(),
                    numeric_backend=fixture.backend,
                )
                self.assertTrue(retried.manifest_path.is_file())

    def test_manifest_binds_job_source_hashes_and_every_tile_receipt(self) -> None:
        fixture = create_product_fixture(self.root)
        manifest = json.loads(fixture.publication.manifest_path.read_bytes())
        self.assertEqual(
            manifest["producer"]["jobSpec"]["producerSourceHashes"],
            [SOURCE_HASH],
        )
        self.assertEqual(manifest["producer"]["jobKey"], fixture.spec.job_key)
        self.assertEqual(
            [tile["recordCount"] for tile in manifest["tiles"]],
            [2, 2, 2, 2],
        )
        self.assertEqual(manifest["summary"]["recordCount"], 8)
        self.assertEqual(manifest["summary"]["sourceMaskUnion"], 5)


if __name__ == "__main__":
    unittest.main()
