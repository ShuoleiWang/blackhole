from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from offline.adaptive_frame import AdaptivePixelOptions
from offline.cie_color import DEFAULT_CIE_CSV, DEFAULT_CIE_METADATA
from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.job import InputArtifact, JobSpec, TaskKey, canonical_json_bytes, run_job
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_disk import StationaryNovikovThorneDisk
from offline.kerr_disk_frame import DarkEscapedObserverSpectrum
from offline.kerr_finite_thickness import (
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_finite_thickness_frame import KerrFiniteThicknessRaySampler
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
import offline.kerr_returning_radiation_finite_thickness_frame as frame_module
from offline.kerr_returning_radiation_finite_thickness_frame import (
    KerrReturningRadiationFiniteThicknessRaySampler,
)
from offline.kerr_returning_radiation_frame_context import (
    ValidatedReturningThermalAuthority,
    authenticate_returning_thermal_emission,
)
import offline.kerr_returning_radiation_kernel as kernel_module
from offline.kerr_returning_radiation_kernel import (
    KerrReturningRadiationKernelPolicy,
    integrate_kerr_returning_radiation_energy_kernel,
)
import offline.kerr_returning_radiation_spectral_replay as replay_module
from offline.kerr_returning_radiation_spectral_replay import (
    DEFAULT_RETURNING_RADIATION_SPECTRAL_REPLAY_LIMITS,
    KerrReturningRadiationSpectralReplayError,
    ReturningRadiationSpectralReplayLimits,
    validate_returning_radiation_spectral_live_replay,
)
import offline.kerr_returning_radiation_live_replay_attestation as attestation_module
from offline.kerr_returning_radiation_live_replay_attestation import (
    publish_returning_radiation_live_replay_attestation,
)
from offline.kerr_returning_radiation_spectral_product import (
    ReturningRadiationAdaptiveSpectralTileProducer,
    build_returning_radiation_spectral_job_spec,
)
from offline.kerr_returning_radiation_thermal_profile import (
    solve_certified_kerr_returning_radiation_thermal_profile,
)
from offline.kerr_returning_radiation_thermal_spectrum import (
    build_certified_returning_radiation_thermal_spectrum_provider,
)
from offline.spectral_frame import (
    SpectralPixelLayout,
    pack_spectral_pixel,
    unpack_spectral_pixel,
)
import offline.spectral_product as spectral_product_module
from offline.spectral_product import (
    PRODUCT_SCHEMA,
    SpectralFrameGrid,
    default_numeric_backend_descriptor,
    publish_spectral_product,
)
import scripts.render_offline_kerr_returning_radiation_frame as renderer
import scripts.verify_offline_spectral_frame as verifier_module
from scripts.verify_nr_contract import ContractError
from scripts.verify_offline_kerr_returning_radiation_live_replay_attestation import (
    validate_returning_radiation_live_replay_attestation,
)


SOLAR_MASS_KG = 1.98847e30


def _identity_for(*values: object) -> str:
    return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


def _escaped_classifier(*arguments):
    return kernel_module._DirectionTransport(
        "escaped",
        None,
        None,
        None,
        0.0,
        _identity_for(*arguments[6:]),
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class _CodeCallCounter:
    def __init__(self, entry):
        self.code = entry.__code__
        self.calls = 0
        self.previous = None

    def __enter__(self):
        self.previous = sys.getprofile()

        def profile(frame, event, argument):
            del argument
            if event == "call" and frame.f_code is self.code:
                self.calls += 1
            if self.previous is not None:
                self.previous(frame, event, None)

        sys.setprofile(profile)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        sys.setprofile(self.previous)


class ReturningRadiationSpectralLiveReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name).resolve(strict=True)

        metric = KerrKerrSchildMetric(spin_a_m=0.7)
        calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.05,
            outer_radius_over_mass=8.0,
        )
        surface = KerrFiniteThicknessMultiSurface(metric, calibration)
        termination = KerrOblateTermination.horizon_worldtube(
            metric,
            escape_radius_m=15.0,
            offset_m=0.02,
        )
        ray_options = RayTraceOptions(
            absolute_tolerance=1.0e-8,
            relative_tolerance=1.0e-8,
            initial_step=0.1,
            maximum_step=0.5,
            maximum_affine_length=50.0,
            null_residual_limit=1.0e-6,
            record_path=True,
        )
        surface_options = SurfaceEventOptions(
            absolute_tolerance=1.0e-8,
            relative_tolerance=1.0e-8,
            null_residual_limit=1.0e-6,
            subdivisions_per_segment=2,
        )
        disk = StationaryNovikovThorneDisk(
            metric=metric,
            black_hole_mass_kg=1.0e8 * SOLAR_MASS_KG,
            mass_accretion_rate_kg_s=1.0e22,
            colour_correction=1.7,
        )
        kernel_policy = KerrReturningRadiationKernelPolicy(
            rho_order=4,
            mu_order=4,
            psi_count=4,
            absolute_tolerance=1.0e-10,
            relative_tolerance=1.0e-10,
            symmetry_absolute_tolerance=1.0e-10,
            symmetry_relative_tolerance=1.0e-10,
            maximum_direction_evaluations=20_000,
            maximum_whole_ray_traces=80_000,
        )
        area_policy = KerrFiniteThicknessAreaQuadraturePolicy(
            gauss_legendre_order=24,
            relative_tolerance=1.0e-8,
            absolute_tolerance_over_mass_squared=1.0e-8,
            maximum_point_evaluations=384,
        )
        with mock.patch.object(
            kernel_module,
            "_trace_direction",
            side_effect=_escaped_classifier,
        ):
            source = integrate_kerr_returning_radiation_energy_kernel(
                surface,
                termination=termination,
                annulus_edges_over_mass=(
                    float(calibration.isco_radius_over_mass),
                    6.0,
                    8.0,
                ),
                ray_options=ray_options,
                surface_options=surface_options,
                policy=kernel_policy,
                area_policy=area_policy,
            )
            profile = solve_certified_kerr_returning_radiation_thermal_profile(
                source,
                disk=disk,
            )
            provider = build_certified_returning_radiation_thermal_spectrum_provider(
                profile
            )
            authority = authenticate_returning_thermal_emission(surface, provider)
            cls.other_authority = authenticate_returning_thermal_emission(
                surface,
                provider,
            )
        base = KerrFiniteThicknessRaySampler(
            metric=metric,
            observer_radius_m=10.0,
            termination=termination,
            surface=surface,
            disk=disk,
            escaped_observer_spectrum=DarkEscapedObserverSpectrum(),
            fine_options=ray_options,
            surface_options=surface_options,
            observer_theta_rad=0.2,
            coarse_tolerance_multiplier=8.0,
            terminal_event_tolerance_m=2.0e-4,
            terminal_covector_tolerance=2.0e-4,
            specific_intensity_relative_tolerance=1.0e-3,
        )
        cls.sampler = KerrReturningRadiationFiniteThicknessRaySampler(base, authority)
        cls.layout = SpectralPixelLayout((3.0e14, 5.0e14))
        cls.grid = SpectralFrameGrid(
            width_pixels=1,
            height_pixels=1,
            screen_x_min=-1.0e-6,
            screen_x_max=1.0e-6,
            screen_y_min=-1.0e-6,
            screen_y_max=1.0e-6,
        )
        cls.options = AdaptivePixelOptions(
            minimum_depth=0,
            maximum_depth=0,
            maximum_ray_evaluations=32,
            radiance_absolute_tolerances=(1.0e100, 1.0e100),
            radiance_relative_tolerance=1.0,
            radiance_guard_ceilings=(1.0e200, 1.0e200),
            unresolved_solid_angle_fraction_tolerance=1.0,
            weighted_log_g_tolerance=1.0,
            weighted_direction_tolerance_rad=math.pi,
        )
        cls.backend = default_numeric_backend_descriptor()
        cls.sources = renderer.collect_source_artifacts()
        cls.science = renderer.collect_science_artifacts(
            Path(DEFAULT_CIE_CSV),
            Path(DEFAULT_CIE_METADATA),
        )
        inputs = tuple(sorted((*cls.sources, *cls.science)))
        source_hashes = tuple(sorted({artifact.sha256 for artifact in cls.sources}))
        spec = build_returning_radiation_spectral_job_spec(
            cls.sampler,
            cls.layout,
            cls.grid,
            cls.options,
            tile_width=1,
            tile_height=1,
            numeric_backend=cls.backend,
            inputs=inputs,
            producer_source_hashes=source_hashes,
        )
        producer = ReturningRadiationAdaptiveSpectralTileProducer(
            cls.sampler,
            cls.layout,
            cls.grid,
            cls.options,
            cls.backend,
            spec,
        )
        run = run_job(spec, producer, cls.root / "cache", jobs=1)
        cls.publication = publish_spectral_product(
            cls.root / "baseline",
            job_spec=spec,
            job_run=run,
            layout=cls.layout,
            grid=cls.grid,
            options=cls.options,
            sampler_descriptor=cls.sampler.descriptor(),
            numeric_backend=cls.backend,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def setUp(self) -> None:
        self.case = self.root / self._testMethodName
        shutil.copytree(self.publication.output_directory, self.case)
        self.manifest_path = self.case / "manifest.json"

    def _manifest(self) -> dict[str, object]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _rebuild_job_spec(self, manifest: dict[str, object]) -> None:
        producer = manifest["producer"]
        raw = producer["jobSpec"]
        tasks = tuple(
            TaskKey(
                item["sampleIndex"],
                item["y"],
                item["x"],
                item["width"],
                item["height"],
            )
            for item in raw["tasks"]
        )
        inputs = tuple(
            InputArtifact(item["uri"], item["byteLength"], item["sha256"])
            for item in raw["inputs"]
        )
        spec = JobSpec(
            producer=raw["producer"],
            algorithm_version=raw["algorithmVersion"],
            tasks=tasks,
            parameters=raw["parameters"],
            inputs=inputs,
            producer_source_hashes=tuple(raw["producerSourceHashes"]),
            record_bytes=raw["recordBytes"],
        )
        producer["id"] = spec.producer
        producer["algorithmVersion"] = spec.algorithm_version
        producer["jobSpec"] = spec.as_dict()
        producer["jobKey"] = spec.job_key

    def _reseal(self, manifest: dict[str, object], *, rebuild_job: bool = False) -> None:
        if rebuild_job:
            self._rebuild_job_spec(manifest)
        producer = manifest["producer"]
        parameters = producer["jobSpec"]["parameters"]
        configuration = {
            **parameters,
            "jobKey": producer["jobKey"],
            "jobSpec": producer["jobSpec"],
        }
        manifest["integrity"]["configurationSha256"] = _canonical_hash(configuration)
        identity = {
            "configurationSha256": manifest["integrity"]["configurationSha256"],
            "schema": PRODUCT_SCHEMA,
            "summary": manifest["summary"],
            "tiles": manifest["tiles"],
        }
        product_hash = _canonical_hash(identity)
        manifest["integrity"]["productSha256"] = product_hash
        manifest["id"] = f"scientific-spectral-frame-{product_hash[:24]}"
        payload = canonical_json_bytes(manifest)
        self.manifest_path.write_bytes(payload)
        sidecar = self.case / manifest["integrity"]["manifestSidecar"]
        sidecar.write_bytes(
            f"{hashlib.sha256(payload).hexdigest()}  manifest.json\n".encode("ascii")
        )

    def test_real_one_by_one_live_sampler_replays_exact_bytes_and_layers_report(self):
        report = validate_returning_radiation_spectral_live_replay(
            self.manifest_path,
            sampler=self.sampler,
        )
        self.assertEqual(report["recordCount"], 1)
        self.assertEqual(report["tileCount"], 1)
        for name in (
            "structuralContractVerified",
            "producerIdentityCurrentMatch",
            "jobSpecVerified",
            "samplerDescriptorLiveMatch",
            "numericBackendCurrentMatch",
            "sourceArtifactsCurrentMatch",
            "frameGeodesicsReplayed",
            "frozenThermalSnapshotBound",
            "pixelBytesExact",
        ):
            self.assertIs(report[name], True, name)
        for name in (
            "thermalFixedPointReplayed",
            "directionCacheRecordsReplayed",
            "directionRaysRetraced",
            "independentPhysicsOracle",
        ):
            self.assertIs(report[name], False, name)
        self.assertIn("same-code", report["replayScope"])
        self.assertNotIn("independent oracle", report["replayScope"].lower())
        self.assertEqual(
            report["totalFrameGeodesicsReplayed"],
            2 * report["totalRaySamples"],
        )

    def test_published_live_replay_attestation_is_closed_historical_evidence(self):
        output = self.root / f"{self._testMethodName}.live-replay-attestation-v1"
        publication = replace(
            self.publication,
            output_directory=self.case,
            manifest_path=self.manifest_path,
            manifest_sha256=hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
        )
        attestation = publish_returning_radiation_live_replay_attestation(
            output,
            spectral_publication=publication,
            sampler=self.sampler,
            spectral_schema_path=verifier_module.DEFAULT_SCHEMA,
        )
        self.assertEqual(
            set(path.name for path in output.iterdir()),
            {"manifest.json", "manifest.sha256"},
        )
        document = json.loads(attestation.manifest_path.read_text(encoding="utf-8"))
        self.assertIs(document["sameCodeReplayAttested"], True)
        for name in (
            "sameCodeReplayReperformed",
            "physicsVerified",
            "independentPhysicsOracle",
            "thermalFixedPointReplayed",
            "directionCacheRecordsReplayed",
            "directionRaysRetraced",
        ):
            self.assertIs(document[name], False, name)
        self.assertEqual(
            document["subject"]["manifestSha256"],
            publication.manifest_sha256,
        )
        with mock.patch.object(
            replay_module,
            "validate_returning_radiation_spectral_live_replay",
            side_effect=AssertionError("historical verifier must not replay"),
        ) as replay:
            report = validate_returning_radiation_live_replay_attestation(
                attestation.manifest_path,
                self.manifest_path,
            )
        replay.assert_not_called()
        self.assertIs(report["attestationVerified"], True)
        self.assertIs(report["sameCodeReplayReperformed"], False)
        self.assertIs(report["physicsVerified"], False)
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    renderer.ROOT
                    / "scripts"
                    / "verify_offline_kerr_returning_radiation_live_replay_attestation.py"
                ),
                str(attestation.manifest_path),
                str(self.manifest_path),
            ],
            cwd=renderer.ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), report)
        with self.assertRaises(FileExistsError):
            publish_returning_radiation_live_replay_attestation(
                output,
                spectral_publication=publication,
                sampler=self.sampler,
                spectral_schema_path=verifier_module.DEFAULT_SCHEMA,
            )

    def test_attestation_resealed_binding_and_extra_file_fail_closed(self):
        output = self.root / f"{self._testMethodName}.live-replay-attestation-v1"
        publication = replace(
            self.publication,
            output_directory=self.case,
            manifest_path=self.manifest_path,
            manifest_sha256=hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
        )
        attestation = publish_returning_radiation_live_replay_attestation(
            output,
            spectral_publication=publication,
            sampler=self.sampler,
            spectral_schema_path=verifier_module.DEFAULT_SCHEMA,
        )
        document = json.loads(attestation.manifest_path.read_text(encoding="utf-8"))
        document["bindings"]["inputArtifactsSha256"] = "0" * 64
        claims = {
            key: value
            for key, value in document.items()
            if key not in ("id", "integrity")
        }
        claims_hash = _canonical_hash(claims)
        document["id"] = (
            "returning-radiation-live-replay-attestation-" + claims_hash[:24]
        )
        document["integrity"] = {"claimsSha256": claims_hash}
        payload = canonical_json_bytes(document)
        attestation.manifest_path.write_bytes(payload)
        (output / "manifest.sha256").write_bytes(
            f"{hashlib.sha256(payload).hexdigest()}  manifest.json\n".encode("ascii")
        )
        with self.assertRaisesRegex(ContractError, "exact claim"):
            validate_returning_radiation_live_replay_attestation(
                attestation.manifest_path,
                self.manifest_path,
            )
        (output / "undeclared.txt").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "closed two-file tree"):
            validate_returning_radiation_live_replay_attestation(
                attestation.manifest_path,
                self.manifest_path,
            )

    def test_resealed_valid_tile_change_is_caught_by_live_byte_replay(self):
        manifest = self._manifest()
        entry = manifest["tiles"][0]
        tile_path = self.case / entry["payload"]["uri"]
        record = unpack_spectral_pixel(self.layout, tile_path.read_bytes())
        changed = replace(record, sample_count=record.sample_count + 1)
        payload = pack_spectral_pixel(self.layout, changed)
        tile_path.write_bytes(payload)
        entry["payload"]["sha256"] = hashlib.sha256(payload).hexdigest()
        manifest["summary"]["totalRaySamples"] += 1
        self._reseal(manifest)
        with self.assertRaisesRegex(
            KerrReturningRadiationSpectralReplayError,
            "not byte-identical",
        ):
            validate_returning_radiation_spectral_live_replay(
                self.manifest_path,
                sampler=self.sampler,
            )

    def test_mutation_during_final_generic_pass_is_caught_by_last_reread(self):
        manifest = self._manifest()
        tile_path = self.case / manifest["tiles"][0]["payload"]["uri"]
        structural_entry = replay_module._verify_structural_snapshot
        calls = 0

        def mutate_after_final(snapshot):
            nonlocal calls
            calls += 1
            report = structural_entry(snapshot)
            if calls == 2:
                payload = bytearray(tile_path.read_bytes())
                payload[-1] ^= 1
                tile_path.write_bytes(payload)
            return report

        with (
            mock.patch.object(
                replay_module,
                "_verify_structural_snapshot",
                side_effect=mutate_after_final,
            ),
            self.assertRaisesRegex(
                KerrReturningRadiationSpectralReplayError,
                "changed during final structural pass",
            ),
        ):
            validate_returning_radiation_spectral_live_replay(
                self.manifest_path,
                sampler=self.sampler,
            )
        self.assertEqual(calls, 2)

    def test_sampler_and_snapshot_binding_tamper_fail_closed(self):
        changed_multiplier = KerrReturningRadiationFiniteThicknessRaySampler(
            self.sampler.base_sampler,
            self.sampler.authority,
            annulus_edge_clearance_multiplier=5.0,
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationSpectralReplayError,
            "supplied live sampler descriptor",
        ):
            validate_returning_radiation_spectral_live_replay(
                self.manifest_path,
                sampler=changed_multiplier,
            )

        original_authority = self.sampler.authority
        object.__setattr__(self.sampler, "authority", self.other_authority)
        try:
            with self.assertRaisesRegex(
                KerrReturningRadiationSpectralReplayError,
                "sampler descriptor|snapshot identity|authority snapshot",
            ):
                validate_returning_radiation_spectral_live_replay(
                    self.manifest_path,
                    sampler=self.sampler,
                )
        finally:
            object.__setattr__(self.sampler, "authority", original_authority)

    def test_backend_and_source_or_jobspec_tamper_reject_before_rays(self):
        with (
            mock.patch.object(
                spectral_product_module.platform,
                "machine",
                return_value="stale-plan-machine",
            ),
            _CodeCallCounter(frame_module.integrate_returning_thermal_spectral_pixel) as calls,
            self.assertRaisesRegex(
                KerrReturningRadiationSpectralReplayError,
                "numeric backend",
            ),
        ):
            validate_returning_radiation_spectral_live_replay(
                self.manifest_path,
                sampler=self.sampler,
            )
        self.assertEqual(calls.calls, 0)

        manifest = self._manifest()
        raw = manifest["producer"]["jobSpec"]
        source = next(
            item for item in raw["inputs"] if item["uri"].startswith("repo-source://")
        )
        old_hash = source["sha256"]
        fake_hash = "0" * 64 if old_hash != "0" * 64 else "1" * 64
        source["sha256"] = fake_hash
        raw["producerSourceHashes"] = sorted(
            fake_hash if value == old_hash else value
            for value in raw["producerSourceHashes"]
        )
        self._reseal(manifest, rebuild_job=True)
        with (
            _CodeCallCounter(frame_module.integrate_returning_thermal_spectral_pixel) as calls,
            self.assertRaisesRegex(
                KerrReturningRadiationSpectralReplayError,
                "producer/CIE artifacts",
            ),
        ):
            validate_returning_radiation_spectral_live_replay(
                self.manifest_path,
                sampler=self.sampler,
            )
        self.assertEqual(calls.calls, 0)

    def test_resealed_producer_version_tamper_is_only_structural_not_replayable(self):
        manifest = self._manifest()
        manifest["producer"]["jobSpec"]["algorithmVersion"] = "9.9.9"
        self._reseal(manifest, rebuild_job=True)
        events: list[str] = []
        structural_entry = replay_module._verify_structural_snapshot
        strict_entry = replay_module._strict_configuration

        def structural(snapshot):
            events.append("generic")
            return structural_entry(snapshot)

        def strict(document, sampler):
            events.append("returning-strict")
            return strict_entry(document, sampler)

        with (
            mock.patch.object(
                replay_module,
                "_verify_structural_snapshot",
                side_effect=structural,
            ),
            mock.patch.object(
                replay_module,
                "_strict_configuration",
                side_effect=strict,
            ),
            _CodeCallCounter(frame_module.integrate_returning_thermal_spectral_pixel) as calls,
            self.assertRaisesRegex(
                KerrReturningRadiationSpectralReplayError,
                "unsupported returning tile producer/version",
            ),
        ):
            validate_returning_radiation_spectral_live_replay(
                self.manifest_path,
                sampler=self.sampler,
            )
        self.assertEqual(calls.calls, 0)
        self.assertEqual(events, ["generic", "returning-strict"])

    def test_resealed_irregular_tiling_not_emitted_by_current_builder_is_rejected(self):
        manifest = self._manifest()
        original_entry = manifest["tiles"][0]
        original_path = self.case / original_entry["payload"]["uri"]
        record_payload = original_path.read_bytes()
        first = TaskKey(0, 0, 0, 1, 1)
        second = TaskKey(0, 0, 1, 2, 1)
        second_uri = f"tiles/{second.file_stem}.spx"
        second_payload = record_payload * 2
        (self.case / second_uri).write_bytes(second_payload)

        frame = manifest["frame"]
        frame["widthPixels"] = 3
        raw_spec = manifest["producer"]["jobSpec"]
        raw_spec["parameters"]["frame"] = json.loads(json.dumps(frame))
        raw_spec["tasks"] = [first.as_dict(), second.as_dict()]

        def tile_entry(task: TaskKey, payload: bytes) -> dict[str, object]:
            return {
                "payload": {
                    "byteLength": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "uri": f"tiles/{task.file_stem}.spx",
                },
                "recordCount": task.width * task.height,
                "recordOrder": "row-major-local-y-then-x",
                "sampleIndex": task.sample_index,
                "tile": {
                    "height": task.height,
                    "width": task.width,
                    "x": task.x,
                    "y": task.y,
                },
            }

        manifest["tiles"] = [
            tile_entry(first, record_payload),
            tile_entry(second, second_payload),
        ]
        summary = verifier_module._Summary(self.layout.frequency_count)
        record = unpack_spectral_pixel(self.layout, record_payload)
        for _index in range(3):
            summary.add(record)
        manifest["summary"] = summary.descriptor()
        self._reseal(manifest, rebuild_job=True)

        with (
            _CodeCallCounter(frame_module.integrate_returning_thermal_spectral_pixel) as calls,
            self.assertRaisesRegex(
                KerrReturningRadiationSpectralReplayError,
                "exact output of the current returning builder",
            ),
        ):
            validate_returning_radiation_spectral_live_replay(
                self.manifest_path,
                sampler=self.sampler,
            )
        self.assertEqual(calls.calls, 0)

    def test_resource_limits_are_enforced_before_any_ray(self):
        manifest_payload = self.manifest_path.read_bytes()
        manifest = self._manifest()
        tile_path = self.case / manifest["tiles"][0]["payload"]["uri"]
        tile_bytes = tile_path.stat().st_size
        cases = (
            (
                replace(
                    DEFAULT_RETURNING_RADIATION_SPECTRAL_REPLAY_LIMITS,
                    maximum_manifest_bytes=len(manifest_payload) - 1,
                ),
                "fixed.*byte limit",
            ),
            (
                replace(
                    DEFAULT_RETURNING_RADIATION_SPECTRAL_REPLAY_LIMITS,
                    maximum_tile_bytes=tile_bytes - 1,
                ),
                "tile exceeds",
            ),
            (
                replace(
                    DEFAULT_RETURNING_RADIATION_SPECTRAL_REPLAY_LIMITS,
                    maximum_total_tile_bytes=tile_bytes - 1,
                ),
                "total tile bytes",
            ),
            (
                replace(
                    DEFAULT_RETURNING_RADIATION_SPECTRAL_REPLAY_LIMITS,
                    maximum_frequency_bins=1,
                ),
                "frequency-bin count",
            ),
            (
                replace(
                    DEFAULT_RETURNING_RADIATION_SPECTRAL_REPLAY_LIMITS,
                    maximum_annuli=1,
                ),
                "annulus count",
            ),
            (
                replace(
                    DEFAULT_RETURNING_RADIATION_SPECTRAL_REPLAY_LIMITS,
                    maximum_total_ray_evaluations=1,
                ),
                "geodesic budget",
            ),
        )
        with _CodeCallCounter(
            frame_module.integrate_returning_thermal_spectral_pixel
        ) as calls:
            for limits, message in cases:
                with self.subTest(limit=message), self.assertRaisesRegex(
                    KerrReturningRadiationSpectralReplayError,
                    message,
                ):
                    validate_returning_radiation_spectral_live_replay(
                        self.manifest_path,
                        sampler=self.sampler,
                        limits=limits,
                    )

            manifest = json.loads(manifest_payload)
            manifest["tiles"].append(json.loads(json.dumps(manifest["tiles"][0])))
            self.manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(
                KerrReturningRadiationSpectralReplayError,
                "tile count",
            ):
                validate_returning_radiation_spectral_live_replay(
                    self.manifest_path,
                    sampler=self.sampler,
                    limits=replace(
                        DEFAULT_RETURNING_RADIATION_SPECTRAL_REPLAY_LIMITS,
                        maximum_tiles=1,
                    ),
                )

            manifest = json.loads(manifest_payload)
            manifest["tiles"][0]["recordCount"] = 2
            self.manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(
                KerrReturningRadiationSpectralReplayError,
                "record count",
            ):
                validate_returning_radiation_spectral_live_replay(
                    self.manifest_path,
                    sampler=self.sampler,
                    limits=replace(
                        DEFAULT_RETURNING_RADIATION_SPECTRAL_REPLAY_LIMITS,
                        maximum_records=1,
                    ),
                )
        self.manifest_path.write_bytes(manifest_payload)
        self.assertEqual(calls.calls, 0)
        with self.assertRaisesRegex(ValueError, "no greater"):
            ReturningRadiationSpectralReplayLimits(
                maximum_tiles=replay_module.MAXIMUM_REPLAY_TILES + 1
            )

    def test_ancestor_manifest_and_tile_symlinks_are_rejected(self):
        ancestor = self.root / f"{self._testMethodName}-ancestor"
        ancestor.symlink_to(self.case, target_is_directory=True)
        with self.assertRaisesRegex(
            KerrReturningRadiationSpectralReplayError,
            "non-symlink path",
        ):
            validate_returning_radiation_spectral_live_replay(
                ancestor / "manifest.json",
                sampler=self.sampler,
            )

        original = self.manifest_path.read_bytes()
        target = self.case / "manifest.real"
        target.write_bytes(original)
        self.manifest_path.unlink()
        self.manifest_path.symlink_to(target.name)
        with self.assertRaisesRegex(
            KerrReturningRadiationSpectralReplayError,
            "traversal-safe regular artifact",
        ):
            validate_returning_radiation_spectral_live_replay(
                self.manifest_path,
                sampler=self.sampler,
            )

        schema_copy = self.root / f"{self._testMethodName}-schema.json"
        schema_copy.write_bytes(Path(replay_module.DEFAULT_SCHEMA).read_bytes() + b" ")
        with (
            _CodeCallCounter(frame_module.integrate_returning_thermal_spectral_pixel) as calls,
            self.assertRaisesRegex(
                KerrReturningRadiationSpectralReplayError,
                "repository's exact schema",
            ),
        ):
            validate_returning_radiation_spectral_live_replay(
                self.publication.manifest_path,
                sampler=self.sampler,
                schema_path=schema_copy,
            )
        self.assertEqual(calls.calls, 0)

    def test_product_root_replacement_during_generic_is_rejected_before_rays(self):
        structural_entry = replay_module._verify_structural_snapshot
        moved = self.root / f"{self._testMethodName}-original"

        def replace_root(snapshot):
            report = structural_entry(snapshot)
            self.case.rename(moved)
            shutil.copytree(moved, self.case)
            return report

        with (
            mock.patch.object(
                replay_module,
                "_verify_structural_snapshot",
                side_effect=replace_root,
            ),
            _CodeCallCounter(frame_module.integrate_returning_thermal_spectral_pixel) as calls,
            self.assertRaisesRegex(
                KerrReturningRadiationSpectralReplayError,
                "product root changed|no longer names the held product root",
            ),
        ):
            validate_returning_radiation_spectral_live_replay(
                self.manifest_path,
                sampler=self.sampler,
            )
        self.assertEqual(calls.calls, 0)

    def test_tile_symlink_and_same_fd_toctou_are_rejected(self):
        manifest = self._manifest()
        tile_path = self.case / manifest["tiles"][0]["payload"]["uri"]
        saved = tile_path.with_suffix(".saved")
        saved.write_bytes(tile_path.read_bytes())
        tile_path.unlink()
        tile_path.symlink_to(saved.name)
        with self.assertRaisesRegex(
            KerrReturningRadiationSpectralReplayError,
            "stat traversal-safe",
        ):
            validate_returning_radiation_spectral_live_replay(
                self.manifest_path,
                sampler=self.sampler,
            )

        tile_path.unlink()
        tile_path.write_bytes(saved.read_bytes())
        saved.unlink()
        target_identity = (tile_path.stat().st_dev, tile_path.stat().st_ino)
        original_read = replay_module.os.read
        changed = False

        def racing_read(descriptor: int, count: int) -> bytes:
            nonlocal changed
            block = original_read(descriptor, count)
            metadata = os.fstat(descriptor)
            if not changed and (metadata.st_dev, metadata.st_ino) == target_identity and block:
                changed = True
                payload = bytearray(tile_path.read_bytes())
                payload[-1] ^= 1
                tile_path.write_bytes(payload)
            return block

        with (
            mock.patch.object(replay_module.os, "read", side_effect=racing_read),
            self.assertRaisesRegex(
                KerrReturningRadiationSpectralReplayError,
                "changed while it was being read",
            ),
        ):
            validate_returning_radiation_spectral_live_replay(
                self.manifest_path,
                sampler=self.sampler,
            )
        self.assertTrue(changed)

    def test_public_integrator_and_packer_rebinding_rejects_before_replacement(self):
        for owner, name in (
            (frame_module, "integrate_returning_thermal_spectral_pixel"),
            (frame_module, "integrate_spectral_pixel"),
            (frame_module._TrustedPixelSampler, "sample"),
            (replay_module.spectral_frame_module, "pack_adaptive_pixel"),
            (replay_module.spectral_frame_module, "pack_spectral_pixel"),
            (replay_module.spectral_frame_module, "_source_coverage"),
            (SpectralFrameGrid, "pixel_bounds"),
        ):
            replacement = mock.Mock(side_effect=AssertionError("must not run"))
            with (
                self.subTest(binding=name),
                mock.patch.object(owner, name, replacement),
                self.assertRaisesRegex(
                    KerrReturningRadiationSpectralReplayError,
                    "callable identity|runtime binding",
                ),
            ):
                validate_returning_radiation_spectral_live_replay(
                    self.manifest_path,
                    sampler=self.sampler,
                )
            replacement.assert_not_called()

    def test_no_cli_and_exact_sampler_and_limits_types(self):
        self.assertFalse(hasattr(replay_module, "main"))
        with self.assertRaisesRegex(TypeError, "exact KerrReturning"):
            validate_returning_radiation_spectral_live_replay(
                self.manifest_path,
                sampler=object(),  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(TypeError, "limits must have exact"):
            validate_returning_radiation_spectral_live_replay(
                self.manifest_path,
                sampler=self.sampler,
                limits=object(),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
