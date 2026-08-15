from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import offline.cie_product as cie_product_module
import scripts.verify_offline_cie_xyz as cie_verifier_module
from offline.adaptive_frame import (
    AdaptivePixelOptions,
    RayConvergenceAudit,
    SpectralRaySample,
)
from offline.cie_color import (
    cie_1931_frequency_grid_hz,
    load_authenticated_cie_1931_2deg,
    spectral_i_nu_to_cie_xyz,
)
from offline.cie_product import (
    CONVERTER_SOURCE_FILES,
    CieProductError,
    CieXyzPixelRecord,
    RECORD_BYTES,
    SCIENTIFIC_STATUS,
    convert_spectral_product_to_cie_xyz,
    pack_cie_xyz_pixel,
    unpack_cie_xyz_pixel,
)
from offline.job import run_job
from offline.spectral_frame import (
    SpectralPixelLayout,
    unpack_spectral_pixel,
)
from offline.spectral_product import (
    AdaptiveSpectralTileProducer,
    SpectralFrameGrid,
    build_spectral_job_spec,
    publish_spectral_product,
)
from tests.test_offline_spectral_product import create_product_fixture


SOURCE_HASH = hashlib.sha256(b"exact CIE test sampler source").hexdigest()
ROOT = Path(__file__).resolve().parents[1]
TEST_CONVERTER_BACKEND = {
    "floatEvaluation": "test CPython binary64",
    "implementationId": "tests.cie-product-backend/v1",
}


@dataclass(frozen=True, slots=True)
class _ExactCieSampler:
    def descriptor(self) -> dict[str, object]:
        return {
            "implementationId": "tests.exact-cie-grid-sampler/v1",
            "scientificStatus": "deterministic constant test spectrum",
        }

    def sample(
        self,
        screen_x: float,
        screen_y: float,
        observer_frequencies_hz: tuple[float, ...],
    ) -> SpectralRaySample:
        del screen_x, screen_y
        intensities = tuple(
            1.0e-30 * (index + 1)
            for index in range(len(observer_frequencies_hz))
        )
        errors = tuple(value * 1.0e-3 for value in intensities)
        return SpectralRaySample(
            specific_intensities_nu=intensities,
            absolute_errors_nu=errors,
            visible_source="escaped-boundary",
            topology_signature="single-analytic-sky",
            escape_direction=(0.0, 0.0, 1.0),
            ray_converged=True,
            convergence_audit=RayConvergenceAudit(
                maximum_null_residual=1.0e-13,
                accepted_steps=5,
                rejected_steps=0,
                ray_gate_passed=True,
                source_gate_passed=True,
                transfer_gate_passed=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class ExactCieSpectralFixture:
    root: Path
    input_product: Path
    input_manifest: Path
    output_product: Path
    output_manifest: Path
    layout: SpectralPixelLayout
    publication: object


def create_exact_cie_spectral_fixture(root: Path) -> ExactCieSpectralFixture:
    table = load_authenticated_cie_1931_2deg()
    layout = SpectralPixelLayout(cie_1931_frequency_grid_hz(table))
    grid = SpectralFrameGrid(
        width_pixels=2,
        height_pixels=1,
        screen_x_min=-0.2,
        screen_x_max=0.2,
        screen_y_min=-0.1,
        screen_y_max=0.1,
    )
    options = AdaptivePixelOptions(
        minimum_depth=0,
        maximum_depth=0,
        maximum_ray_evaluations=64,
        radiance_absolute_tolerances=(1.0,) * layout.frequency_count,
        radiance_relative_tolerance=1.0e-8,
        radiance_guard_ceilings=(1.0,) * layout.frequency_count,
        weighted_log_g_tolerance=1.0,
        weighted_direction_tolerance_rad=1.0,
    )
    input_backend = {
        "implementationId": "tests.spectral-input-backend/v1",
    }
    sampler = _ExactCieSampler()
    specification = build_spectral_job_spec(
        layout,
        grid,
        options,
        sampler.descriptor(),
        tile_width=1,
        tile_height=1,
        numeric_backend=input_backend,
        producer_source_hashes=(SOURCE_HASH,),
    )
    producer = AdaptiveSpectralTileProducer(
        sampler,
        layout,
        grid,
        options,
        input_backend,
    )
    run = run_job(specification, producer, root / "cache", jobs=1)
    input_product = root / "spectral-product"
    publication = publish_spectral_product(
        input_product,
        job_spec=specification,
        job_run=run,
        layout=layout,
        grid=grid,
        options=options,
        sampler_descriptor=sampler.descriptor(),
        numeric_backend=input_backend,
    )
    output_product = root / "cie-product"
    output_publication = convert_spectral_product_to_cie_xyz(
        publication.manifest_path,
        output_product,
        numeric_backend=TEST_CONVERTER_BACKEND,
    )
    return ExactCieSpectralFixture(
        root=root,
        input_product=input_product,
        input_manifest=publication.manifest_path,
        output_product=output_product,
        output_manifest=output_publication.manifest_path,
        layout=layout,
        publication=output_publication,
    )


class OfflineCieProductTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temporary.name).resolve(strict=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_grid_conversion_publishes_bound_binary64_records(self) -> None:
        fixture = create_exact_cie_spectral_fixture(self.root)
        manifest = json.loads(fixture.output_manifest.read_bytes())
        input_manifest = json.loads(fixture.input_manifest.read_bytes())
        self.assertEqual(fixture.publication.record_count, 2)
        self.assertEqual(fixture.publication.tile_count, 2)
        self.assertEqual(manifest["pixelLayout"]["recordBytes"], RECORD_BYTES)
        self.assertEqual(manifest["scientificStatus"], dict(SCIENTIFIC_STATUS))
        self.assertFalse(manifest["scientificStatus"]["toneMappingApplied"])
        self.assertFalse(manifest["scientificStatus"]["linearSrgbStored"])
        self.assertEqual(
            manifest["inputSpectralProduct"]["id"],
            input_manifest["id"],
        )

        output_entry = manifest["tiles"][0]
        input_entry = input_manifest["tiles"][0]
        output_payload = (
            fixture.output_product / output_entry["outputPayload"]["uri"]
        ).read_bytes()
        input_payload = (
            fixture.input_product / input_entry["payload"]["uri"]
        ).read_bytes()
        output_record = unpack_cie_xyz_pixel(output_payload)
        input_record = unpack_spectral_pixel(fixture.layout, input_payload)
        expected = spectral_i_nu_to_cie_xyz(
            fixture.layout.observer_frequencies_hz,
            input_record.mean_specific_intensities_nu,
        )
        expected_error = spectral_i_nu_to_cie_xyz(
            fixture.layout.observer_frequencies_hz,
            input_record.mean_estimated_absolute_errors_nu,
        )
        self.assertEqual(
            output_record.mean_cie_xyz,
            (expected.x, expected.y, expected.z),
        )
        self.assertEqual(
            output_record.mean_estimated_absolute_error_xyz,
            (expected_error.x, expected_error.y, expected_error.z),
        )
        self.assertEqual(
            output_record.input_record_sha256,
            hashlib.sha256(input_payload).digest(),
        )
        self.assertEqual(output_record.source_mask, input_record.source_mask)
        self.assertEqual(
            output_record.convergence_mask,
            input_record.convergence_mask,
        )

    def test_publication_is_deterministic_and_never_overwrites(self) -> None:
        fixture = create_exact_cie_spectral_fixture(self.root)
        second = convert_spectral_product_to_cie_xyz(
            fixture.input_manifest,
            self.root / "second-cie-product",
            numeric_backend=TEST_CONVERTER_BACKEND,
        )
        self.assertEqual(second.product_sha256, fixture.publication.product_sha256)
        self.assertEqual(
            second.manifest_path.read_bytes(),
            fixture.output_manifest.read_bytes(),
        )
        with self.assertRaisesRegex(CieProductError, "refusing to overwrite"):
            convert_spectral_product_to_cie_xyz(
                fixture.input_manifest,
                fixture.output_product,
                numeric_backend=TEST_CONVERTER_BACKEND,
            )

    def test_converter_cli_publishes_a_verifiable_manifest(self) -> None:
        fixture = create_exact_cie_spectral_fixture(self.root)
        cli_output = self.root / "cli-cie-product"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "convert_offline_spectral_to_cie_xyz.py"),
                str(fixture.input_manifest),
                str(cli_output),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["recordCount"], 2)
        self.assertEqual(Path(report["manifest"]), cli_output / "manifest.json")
        self.assertTrue((cli_output / "manifest.json").is_file())

    def test_every_production_dependency_hash_changes_product_identity(self) -> None:
        fixture = create_exact_cie_spectral_fixture(self.root)
        manifest = json.loads(fixture.output_manifest.read_bytes())
        module_uris = tuple(
            item["moduleUri"]
            for item in manifest["converter"]["descriptor"]["sourceFiles"]
        )
        self.assertEqual(
            module_uris,
            CONVERTER_SOURCE_FILES,
        )
        self.assertEqual(
            tuple(
                entry["moduleUri"]
                for entry in cie_verifier_module._source_descriptor()
            ),
            CONVERTER_SOURCE_FILES,
        )
        schema = json.loads(
            (ROOT / "schemas" / "offline-cie-xyz-frame-v1.schema.json")
            .read_bytes()
        )
        source_files_schema = schema["$defs"]["converterDescriptor"][
            "properties"
        ]["sourceFiles"]
        self.assertEqual(source_files_schema["minItems"], len(CONVERTER_SOURCE_FILES))
        self.assertEqual(source_files_schema["maxItems"], len(CONVERTER_SOURCE_FILES))
        self.assertEqual(
            tuple(schema["$defs"]["sourceFile"]["properties"]["moduleUri"]["enum"]),
            CONVERTER_SOURCE_FILES,
        )
        original_descriptor = cie_product_module._source_file_descriptor
        changed_hashes: set[str] = set()
        for index, selected_uri in enumerate(module_uris):
            def altered_descriptor(
                path: Path,
                module_uri: str,
                *,
                selected_uri: str = selected_uri,
            ) -> dict[str, object]:
                descriptor = original_descriptor(path, module_uri)
                if module_uri == selected_uri:
                    descriptor["sha256"] = hashlib.sha256(
                        f"changed:{module_uri}".encode("ascii")
                    ).hexdigest()
                return descriptor

            with patch.object(
                cie_product_module,
                "_source_file_descriptor",
                side_effect=altered_descriptor,
            ):
                publication = convert_spectral_product_to_cie_xyz(
                    fixture.input_manifest,
                    self.root / f"dependency-change-{index}",
                    numeric_backend=TEST_CONVERTER_BACKEND,
                )
            self.assertNotEqual(
                publication.product_sha256,
                fixture.publication.product_sha256,
            )
            changed_hashes.add(publication.product_sha256)
        self.assertEqual(len(changed_hashes), len(module_uris))

    def test_transaction_failure_cleans_staging_and_same_path_can_retry(self) -> None:
        fixture = create_exact_cie_spectral_fixture(self.root)
        real_write = cie_product_module._atomic_write_no_replace
        for failure_call, label in ((2, "second-tile"), (3, "sidecar"), (4, "manifest")):
            with self.subTest(label=label):
                output = self.root / f"transaction-{label}"
                calls = 0

                def injected_write(path: Path, payload: bytes) -> None:
                    nonlocal calls
                    calls += 1
                    if calls == failure_call:
                        raise CieProductError(f"injected {label} write failure")
                    real_write(path, payload)

                with patch.object(
                    cie_product_module,
                    "_atomic_write_no_replace",
                    side_effect=injected_write,
                ):
                    with self.assertRaisesRegex(CieProductError, f"injected {label}"):
                        convert_spectral_product_to_cie_xyz(
                            fixture.input_manifest,
                            output,
                            numeric_backend=TEST_CONVERTER_BACKEND,
                        )
                self.assertFalse(output.exists())
                self.assertEqual(
                    tuple(self.root.glob(f".{output.name}.staging-*")),
                    (),
                )
                publication = convert_spectral_product_to_cie_xyz(
                    fixture.input_manifest,
                    output,
                    numeric_backend=TEST_CONVERTER_BACKEND,
                )
                self.assertTrue(publication.manifest_path.is_file())

    def test_non_cie_frequency_grid_fails_before_output_creation(self) -> None:
        wrong_root = self.root / "wrong-grid"
        wrong_root.mkdir()
        wrong = create_product_fixture(wrong_root)
        output = self.root / "must-not-exist"
        with self.assertRaisesRegex(CieProductError, "exact authenticated 471-bin"):
            convert_spectral_product_to_cie_xyz(
                wrong.publication.manifest_path,
                output,
                numeric_backend=TEST_CONVERTER_BACKEND,
            )
        self.assertFalse(output.exists())

    def test_pixel_abi_rejects_negative_nan_and_incomplete_gates(self) -> None:
        valid = CieXyzPixelRecord(
            mean_cie_xyz=(1.0, 2.0, 3.0),
            mean_estimated_absolute_error_xyz=(0.1, 0.2, 0.3),
            input_record_sha256=b"x" * 32,
            source_mask=4,
            convergence_mask=255,
        )
        self.assertEqual(unpack_cie_xyz_pixel(pack_cie_xyz_pixel(valid)), valid)
        for bad_value in (-1.0, float("nan")):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    CieXyzPixelRecord(
                        mean_cie_xyz=(bad_value, 2.0, 3.0),
                        mean_estimated_absolute_error_xyz=(0.1, 0.2, 0.3),
                        input_record_sha256=b"x" * 32,
                        source_mask=4,
                        convergence_mask=255,
                    )
        with self.assertRaisesRegex(ValueError, "required gate"):
            CieXyzPixelRecord(
                mean_cie_xyz=(1.0, 2.0, 3.0),
                mean_estimated_absolute_error_xyz=(0.1, 0.2, 0.3),
                input_record_sha256=b"x" * 32,
                source_mask=4,
                convergence_mask=254,
            )


if __name__ == "__main__":
    unittest.main()
