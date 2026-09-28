from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from offline.adaptive_frame import (
    AdaptivePixelOptions,
    RayConvergenceAudit,
    SpectralRaySample,
    integrate_spectral_pixel,
)
from offline.job import InputArtifact, TaskKey, run_job
from offline.kerr_returning_radiation_finite_thickness_frame import (
    IMPLEMENTATION_ID as SAMPLER_IMPLEMENTATION_ID,
    KerrReturningRadiationFiniteThicknessRaySampler,
)
from offline import (
    kerr_returning_radiation_finite_thickness_frame as returning_frame_module,
)
import offline.kerr_returning_radiation_spectral_product as product_module
from offline.kerr_returning_radiation_spectral_product import (
    MAXIMUM_RETURNING_FRAME_PIXELS,
    MAXIMUM_RETURNING_TILE_PAYLOAD_BYTES,
    MAXIMUM_RETURNING_TILE_TASKS,
    RETURNING_ADAPTIVE_TILE_ALGORITHM_VERSION,
    RETURNING_ADAPTIVE_TILE_PRODUCER_ID,
    SCIENTIFIC_STATUS,
    ReturningRadiationAdaptiveSpectralTileProducer,
    build_returning_radiation_spectral_job_spec,
    invoke_returning_radiation_spectral_tile_producer,
)
import offline.spectral_frame as spectral_frame_module
from offline.spectral_frame import SpectralPixelLayout
from offline.spectral_product import (
    SpectralFrameGrid,
    SpectralProductError,
    publish_spectral_product,
)


SOURCE_HASH = hashlib.sha256(b"returning spectral producer test").hexdigest()
INPUT_HASH = hashlib.sha256(b"returning spectral input test").hexdigest()


class _AlwaysEqualStr(str):
    def __eq__(self, other):
        del other
        return True

    __hash__ = str.__hash__


class _PolicyBypassInt(int):
    def __eq__(self, other):
        del other
        return True

    def __lt__(self, other):
        del other
        return False

    __hash__ = int.__hash__


class _PolicyBypassFloat(float):
    def __eq__(self, other):
        del other
        return True

    __hash__ = float.__hash__


class _ConstantDiskSampler:
    def sample(self, screen_x, screen_y, observer_frequencies_hz):
        del screen_x, screen_y
        return SpectralRaySample(
            specific_intensities_nu=tuple(
                float(index + 1) for index in range(len(observer_frequencies_hz))
            ),
            absolute_errors_nu=(0.0,) * len(observer_frequencies_hz),
            visible_source="disk",
            topology_signature="mock-returning-disk",
            frequency_shift_g=1.0,
            escape_direction=None,
            ray_converged=True,
            convergence_audit=RayConvergenceAudit(
                accepted_steps=1,
                ray_gate_passed=True,
                source_gate_passed=True,
                transfer_gate_passed=True,
            ),
        )


class _Authority:
    def __init__(self) -> None:
        self.snapshot = object()
        self.live_calls = 0

    def require_live(self):
        self.live_calls += 1
        return self.snapshot


def _fake_sampler() -> KerrReturningRadiationFiniteThicknessRaySampler:
    sampler = object.__new__(KerrReturningRadiationFiniteThicknessRaySampler)
    authority = _Authority()
    object.__setattr__(sampler, "authority", authority)
    object.__setattr__(sampler, "_authenticated_snapshot", authority.snapshot)
    return sampler


def _descriptor() -> dict[str, object]:
    return {
        "implementationId": SAMPLER_IMPLEMENTATION_ID,
        "authority": {
            "cachedScientificEvidence": {
                "productionForwardOnly": True,
                "excludedOperationalIdentity": {
                    "cacheAbsolutePaths": True,
                    "executedOrReusedCounters": True,
                    "workerScheduling": True,
                },
                "scientificBindingSha256": "a" * 64,
            }
        },
        "scientificStatus": dict(SCIENTIFIC_STATUS),
    }


def _options(count: int) -> AdaptivePixelOptions:
    return AdaptivePixelOptions(
        minimum_depth=0,
        maximum_depth=0,
        maximum_ray_evaluations=32,
        radiance_absolute_tolerances=(0.0,) * count,
        radiance_relative_tolerance=1.0e-12,
        radiance_guard_ceilings=(10.0,) * count,
        weighted_log_g_tolerance=1.0,
        weighted_direction_tolerance_rad=1.0,
    )


class ReturningRadiationSpectralProductTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve(strict=True)
        self.layout = SpectralPixelLayout((4.0e14, 6.0e14))
        self.grid = SpectralFrameGrid(
            width_pixels=2,
            height_pixels=1,
            screen_x_min=-0.1,
            screen_x_max=0.1,
            screen_y_min=-0.1,
            screen_y_max=0.1,
        )
        self.options = _options(self.layout.frequency_count)
        self.backend = {
            "implementationId": "tests.returning-numeric-backend/v1",
            "binary64": True,
        }
        self.sampler = _fake_sampler()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make(self):
        specification = build_returning_radiation_spectral_job_spec(
            self.sampler,
            self.layout,
            self.grid,
            self.options,
            tile_width=1,
            tile_height=1,
            numeric_backend=self.backend,
            inputs=(InputArtifact("test://input", 3, INPUT_HASH),),
            producer_source_hashes=(SOURCE_HASH,),
        )
        producer = ReturningRadiationAdaptiveSpectralTileProducer(
            self.sampler,
            self.layout,
            self.grid,
            self.options,
            self.backend,
            specification,
        )
        return specification, producer

    def test_scope_and_constructor_fix_exact_sampler_and_pixel_integrator(self) -> None:
        for key in (
            "acceptsArbitraryPixelIntegrator",
            "acceptsGenericSpectralRaySampler",
            "isContinuumRadialReturningRadiationSolution",
            "hasIndependentPhysicsOracle",
            "includesReturningRadiationStressWorkFS",
            "includesScatteringOrSolvedAtmosphere",
            "isGeneralRelativisticMagnetohydrodynamics",
        ):
            self.assertIs(SCIENTIFIC_STATUS[key], False)
        parameters = inspect.signature(
            ReturningRadiationAdaptiveSpectralTileProducer
        ).parameters
        self.assertNotIn("integrator", parameters)
        self.assertNotIn("pixel_integrator", parameters)
        with self.assertRaisesRegex(TypeError, "exact returning-radiation"):
            ReturningRadiationAdaptiveSpectralTileProducer(
                object(),  # type: ignore[arg-type]
                self.layout,
                self.grid,
                self.options,
                self.backend,
                None,  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(TypeError, "cannot be subclassed"):
            class _ForeignProducer(ReturningRadiationAdaptiveSpectralTileProducer):
                pass

        class _ForeignGrid(SpectralFrameGrid):
            def descriptor(self):
                return SpectralFrameGrid.descriptor(self)

            def pixel_bounds(self, x, y):
                return (0.0, 1.0, 0.0, 1.0)

        foreign_grid = _ForeignGrid(
            width_pixels=2,
            height_pixels=1,
            screen_x_min=-0.1,
            screen_x_max=0.1,
            screen_y_min=-0.1,
            screen_y_max=0.1,
        )
        with (
            patch.object(
                KerrReturningRadiationFiniteThicknessRaySampler,
                "descriptor",
                return_value=_descriptor(),
            ),
            self.assertRaisesRegex(TypeError, "exact SpectralFrameGrid"),
        ):
            ReturningRadiationAdaptiveSpectralTileProducer(
                self.sampler,
                self.layout,
                foreign_grid,
                self.options,
                self.backend,
                self._make()[0],
            )

    def test_fixed_resource_bounds_reject_before_task_tuple_allocation(self) -> None:
        cases = (
            (
                SpectralFrameGrid(
                    MAXIMUM_RETURNING_FRAME_PIXELS + 1,
                    1,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                ),
                1,
                1,
                "frame pixel count",
            ),
            (
                SpectralFrameGrid(
                    MAXIMUM_RETURNING_TILE_TASKS + 1,
                    1,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                ),
                1,
                1,
                "tile task count",
            ),
            (
                SpectralFrameGrid(
                    MAXIMUM_RETURNING_TILE_PAYLOAD_BYTES
                    // self.layout.record_bytes
                    + 1,
                    1,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                ),
                MAXIMUM_RETURNING_TILE_PAYLOAD_BYTES
                // self.layout.record_bytes
                + 1,
                1,
                "tile payload",
            ),
        )
        with patch.object(
            KerrReturningRadiationFiniteThicknessRaySampler,
            "descriptor",
            side_effect=AssertionError("descriptor must follow resource checks"),
        ) as descriptor:
            for grid, tile_width, tile_height, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError,
                    message,
                ):
                    build_returning_radiation_spectral_job_spec(
                        self.sampler,
                        self.layout,
                        grid,
                        self.options,
                        tile_width=tile_width,
                        tile_height=tile_height,
                        numeric_backend=self.backend,
                        producer_source_hashes=(SOURCE_HASH,),
                    )
        descriptor.assert_not_called()

    def test_bool_tile_control_is_not_an_integer(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact positive integer"):
            build_returning_radiation_spectral_job_spec(
                self.sampler,
                self.layout,
                self.grid,
                self.options,
                tile_width=True,
                tile_height=1,
                numeric_backend=self.backend,
                producer_source_hashes=(SOURCE_HASH,),
            )

    def test_runtime_rebinding_rejected_before_alternative_entries(self) -> None:
        with patch.object(
            KerrReturningRadiationFiniteThicknessRaySampler,
            "descriptor",
            return_value=_descriptor(),
        ):
            specification, producer = self._make()
        key = specification.tasks[0]

        alternatives = (
            (
                returning_frame_module,
                "integrate_returning_thermal_spectral_pixel",
                "whole-pixel integrator",
            ),
            (spectral_frame_module, "pack_adaptive_pixel", "pixel packer"),
            (SpectralFrameGrid, "pixel_bounds", "pixel-bounds"),
            (
                ReturningRadiationAdaptiveSpectralTileProducer,
                "__call__",
                "producer call",
            ),
        )
        for owner, name, message in alternatives:
            with self.subTest(entry=name):
                replacement = Mock(return_value=b"foreign")
                with (
                    patch.object(owner, name, replacement),
                    self.assertRaisesRegex(SpectralProductError, message),
                ):
                    invoke_returning_radiation_spectral_tile_producer(
                        producer,
                        specification,
                        key,
                    )
                replacement.assert_not_called()

    def test_mutable_descriptor_and_backend_are_compared_as_canonical_bytes(self) -> None:
        descriptor = _descriptor()
        with patch.object(
            KerrReturningRadiationFiniteThicknessRaySampler,
            "descriptor",
            side_effect=lambda: descriptor,
        ):
            specification, producer = self._make()
            self.backend["binary64"] = False
            with (
                patch.object(returning_frame_module, "integrate_spectral_pixel") as entry,
                self.assertRaisesRegex(
                    SpectralProductError,
                    "configuration changed",
                ),
            ):
                invoke_returning_radiation_spectral_tile_producer(
                    producer,
                    specification,
                    specification.tasks[0],
                )
            entry.assert_not_called()

        self.backend["binary64"] = True
        descriptor = _descriptor()
        with patch.object(
            KerrReturningRadiationFiniteThicknessRaySampler,
            "descriptor",
            side_effect=lambda: descriptor,
        ):
            specification, producer = self._make()
            descriptor["authority"]["cachedScientificEvidence"][
                "scientificBindingSha256"
            ] = "b" * 64
            with self.assertRaisesRegex(
                SpectralProductError,
                "configuration changed",
            ):
                invoke_returning_radiation_spectral_tile_producer(
                    producer,
                    specification,
                    specification.tasks[0],
                )

    def test_same_parameters_with_fake_input_or_source_hash_are_rejected(self) -> None:
        with patch.object(
            KerrReturningRadiationFiniteThicknessRaySampler,
            "descriptor",
            return_value=_descriptor(),
        ):
            specification, producer = self._make()
            replacements = (
                (
                    "inputs",
                    (InputArtifact("test://fake", 4, "b" * 64),),
                ),
                ("producer_source_hashes", ("c" * 64,)),
            )
            for field_name, replacement in replacements:
                with self.subTest(field=field_name):
                    original = object.__getattribute__(specification, field_name)
                    object.__setattr__(specification, field_name, replacement)
                    try:
                        with (
                            patch.object(
                                returning_frame_module,
                                "integrate_spectral_pixel",
                            ) as entry,
                            self.assertRaisesRegex(
                                SpectralProductError,
                                "fields changed object identity",
                            ),
                        ):
                            invoke_returning_radiation_spectral_tile_producer(
                                producer,
                                specification,
                                specification.tasks[0],
                            )
                        entry.assert_not_called()
                    finally:
                        object.__setattr__(specification, field_name, original)

    def test_low_level_exact_schema_rejects_policy_bypass_subclasses(self) -> None:
        with patch.object(
            KerrReturningRadiationFiniteThicknessRaySampler,
            "descriptor",
            return_value=_descriptor(),
        ):
            specification, producer = self._make()
            key = specification.tasks[0]
            mutations = (
                (specification, "producer", _AlwaysEqualStr(specification.producer)),
                (
                    specification,
                    "record_bytes",
                    _PolicyBypassInt(specification.record_bytes),
                ),
                (self.grid, "width_pixels", _PolicyBypassInt(2)),
                (self.grid, "sample_indices", (_PolicyBypassInt(0),)),
                (
                    self.layout,
                    "observer_frequencies_hz",
                    (_PolicyBypassFloat(4.0e14), 6.0e14),
                ),
                (
                    self.options,
                    "maximum_ray_evaluations",
                    _PolicyBypassInt(32),
                ),
                (key, "width", _PolicyBypassInt(1)),
            )
            for owner, field_name, replacement in mutations:
                with self.subTest(owner=type(owner).__name__, field=field_name):
                    original = object.__getattribute__(owner, field_name)
                    object.__setattr__(owner, field_name, replacement)
                    try:
                        with (
                            patch.object(
                                returning_frame_module,
                                "integrate_spectral_pixel",
                            ) as entry,
                            self.assertRaises((TypeError, SpectralProductError)),
                        ):
                            invoke_returning_radiation_spectral_tile_producer(
                                producer,
                                specification,
                                key,
                            )
                        entry.assert_not_called()
                    finally:
                        object.__setattr__(owner, field_name, original)

    def test_direct_jobspec_subclass_fields_rejected_at_producer_binding(self) -> None:
        with patch.object(
            KerrReturningRadiationFiniteThicknessRaySampler,
            "descriptor",
            return_value=_descriptor(),
        ):
            specification, _producer = self._make()
            foreign = replace(
                specification,
                producer=_AlwaysEqualStr(specification.producer),
                record_bytes=_PolicyBypassInt(specification.record_bytes),
            )
            with self.assertRaisesRegex(TypeError, "exact str type"):
                ReturningRadiationAdaptiveSpectralTileProducer(
                    self.sampler,
                    self.layout,
                    self.grid,
                    self.options,
                    self.backend,
                    foreign,
                )

    def test_large_task_membership_is_logarithmic_not_linear(self) -> None:
        grid = SpectralFrameGrid(
            width_pixels=1024,
            height_pixels=1,
            screen_x_min=-1.0,
            screen_x_max=1.0,
            screen_y_min=-1.0,
            screen_y_max=1.0,
        )
        with patch.object(
            KerrReturningRadiationFiniteThicknessRaySampler,
            "descriptor",
            return_value=_descriptor(),
        ):
            specification = build_returning_radiation_spectral_job_spec(
                self.sampler,
                self.layout,
                grid,
                self.options,
                tile_width=1,
                tile_height=1,
                numeric_backend=self.backend,
                producer_source_hashes=(SOURCE_HASH,),
            )
            producer = ReturningRadiationAdaptiveSpectralTileProducer(
                self.sampler,
                self.layout,
                grid,
                self.options,
                self.backend,
                specification,
            )

        comparisons = 0
        original_lt = TaskKey.__lt__
        original_eq = TaskKey.__eq__

        def counted_lt(left, right):
            nonlocal comparisons
            comparisons += 1
            return original_lt(left, right)

        def counted_eq(left, right):
            nonlocal comparisons
            comparisons += 1
            return original_eq(left, right)

        def fake_adaptive(
            sampler,
            observer_frequencies_hz,
            *,
            x_min,
            x_max,
            y_min,
            y_max,
            options,
        ):
            del sampler
            return integrate_spectral_pixel(
                _ConstantDiskSampler(),
                observer_frequencies_hz,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                options=options,
            )

        with (
            patch.object(
                KerrReturningRadiationFiniteThicknessRaySampler,
                "descriptor",
                return_value=_descriptor(),
            ),
            patch.object(TaskKey, "__lt__", counted_lt),
            patch.object(TaskKey, "__eq__", counted_eq),
            patch.object(
                returning_frame_module,
                "integrate_spectral_pixel",
                side_effect=fake_adaptive,
            ),
        ):
            payload = invoke_returning_radiation_spectral_tile_producer(
                producer,
                specification,
                specification.tasks[-1],
            )
        self.assertEqual(len(payload), self.layout.record_bytes)
        self.assertLessEqual(comparisons, 16)

    def test_fixed_whole_pixel_entry_runs_once_per_pixel_and_reuses_cache(self) -> None:
        pixel_calls: list[tuple[float, float, float, float]] = []
        authority = self.sampler.authority

        def fixed_adaptive_entry(
            sampler,
            observer_frequencies_hz,
            *,
            x_min,
            x_max,
            y_min,
            y_max,
            options,
        ):
            del sampler
            result = integrate_spectral_pixel(
                _ConstantDiskSampler(),
                observer_frequencies_hz,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                options=options,
            )
            pixel_calls.append((x_min, x_max, y_min, y_max))
            return result

        with (
            patch.object(
                KerrReturningRadiationFiniteThicknessRaySampler,
                "descriptor",
                return_value=_descriptor(),
            ),
            patch.object(
                returning_frame_module,
                "integrate_spectral_pixel",
                side_effect=fixed_adaptive_entry,
            ),
        ):
            specification, producer = self._make()
            first = run_job(
                specification,
                producer,
                self.root / "tile-cache",
                jobs=1,
            )
            second = run_job(
                specification,
                producer,
                self.root / "tile-cache",
                jobs=1,
            )

        self.assertEqual(len(pixel_calls), self.grid.record_count)
        self.assertEqual(authority.live_calls, 2 * self.grid.record_count)
        self.assertEqual((first.executed_tasks, first.reused_tasks), (2, 0))
        self.assertEqual((second.executed_tasks, second.reused_tasks), (0, 2))
        self.assertEqual(specification.producer, RETURNING_ADAPTIVE_TILE_PRODUCER_ID)
        self.assertEqual(
            specification.algorithm_version,
            RETURNING_ADAPTIVE_TILE_ALGORITHM_VERSION,
        )

    def test_jobspec_tamper_fails_before_whole_pixel_integration(self) -> None:
        with patch.object(
            KerrReturningRadiationFiniteThicknessRaySampler,
            "descriptor",
            return_value=_descriptor(),
        ):
            specification, producer = self._make()
            tampered = replace(
                specification,
                parameters={**specification.parameters, "tampered": True},
            )
            with (
                patch.object(returning_frame_module, "integrate_spectral_pixel") as entry,
                self.assertRaisesRegex(
                    SpectralProductError,
                    "producer-bound JobSpec object",
                ),
            ):
                producer(tampered, tampered.tasks[0])
        entry.assert_not_called()

    def test_manifest_keeps_returning_sampler_and_cached_scientific_identity(self) -> None:
        def fixed_adaptive_entry(
            sampler,
            observer_frequencies_hz,
            *,
            x_min,
            x_max,
            y_min,
            y_max,
            options,
        ):
            del sampler
            return integrate_spectral_pixel(
                _ConstantDiskSampler(),
                observer_frequencies_hz,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                options=options,
            )

        with (
            patch.object(
                KerrReturningRadiationFiniteThicknessRaySampler,
                "descriptor",
                return_value=_descriptor(),
            ),
            patch.object(
                returning_frame_module,
                "integrate_spectral_pixel",
                side_effect=fixed_adaptive_entry,
            ),
        ):
            specification, producer = self._make()
            run = run_job(specification, producer, self.root / "cache", jobs=1)
            publication = publish_spectral_product(
                self.root / "product",
                job_spec=specification,
                job_run=run,
                layout=self.layout,
                grid=self.grid,
                options=self.options,
                sampler_descriptor=self.sampler.descriptor(),
                numeric_backend=self.backend,
            )

        manifest = json.loads(publication.manifest_path.read_text(encoding="utf-8"))
        descriptor = manifest["sampler"]["descriptor"]
        self.assertEqual(descriptor["implementationId"], SAMPLER_IMPLEMENTATION_ID)
        evidence = descriptor["authority"]["cachedScientificEvidence"]
        self.assertTrue(evidence["productionForwardOnly"])
        encoded = json.dumps(descriptor, allow_nan=False, sort_keys=True)
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn("reusedTasks", encoded)
        self.assertNotIn("workerCount", encoded)


if __name__ == "__main__":
    unittest.main()
