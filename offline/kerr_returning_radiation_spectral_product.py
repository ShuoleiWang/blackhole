"""Authenticated tile production for returning-radiation spectral frames.

This module is intentionally a narrow adapter over the repository's existing
scientific spectral-product ABI.  It fixes the exact
``KerrReturningRadiationFiniteThicknessRaySampler`` and the exact
whole-pixel returning-thermal integrator into one tile producer; callers
cannot inject a generic ray sampler or pixel integration callback.

Every pixel therefore enters
``integrate_returning_thermal_spectral_pixel`` exactly once.  That integration
owns the two live-authority checks for the complete adaptive pixel while its
per-ray hot path consumes the frozen authenticated emission table.  Packing,
``JobSpec`` identity, task receipts, publication, and structural verification
remain owned by :mod:`offline.spectral_product`.

The scientific scope is the same-code finite-grid, piecewise-annulus model of
the bound sampler.  It is not an independent oracle, a continuum radial
returning-radiation solution, complete KERRBB, a returning-radiation stress
term, a solved/scattering atmosphere, polarization transport, or GRMHD.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

from offline.adaptive_frame import AdaptivePixelOptions
from offline.job import (
    JOB_SCHEMA,
    InputArtifact,
    JobSpec,
    TaskKey,
    _FrozenJsonObject,
    canonical_json_bytes,
)
import offline.kerr_returning_radiation_finite_thickness_frame as returning_frame_module
from offline.kerr_returning_radiation_finite_thickness_frame import (
    KerrReturningRadiationFiniteThicknessRaySampler,
    integrate_returning_thermal_spectral_pixel,
)
import offline.spectral_frame as spectral_frame_module
from offline.spectral_frame import SpectralPixelLayout, pack_adaptive_pixel
import offline.spectral_product as spectral_product_module
from offline.spectral_product import (
    SpectralFrameGrid,
    SpectralProductError,
    adaptive_pixel_options_descriptor,
    build_spectral_job_spec,
    spectral_job_parameters,
)


RETURNING_ADAPTIVE_TILE_PRODUCER_ID: Final = (
    "blackhole.returning-radiation-adaptive-spectral-tile"
)
RETURNING_ADAPTIVE_TILE_ALGORITHM_VERSION: Final = "1.0.0"

# Product-local denial-of-service bounds.  These are checked arithmetically
# before ``SpectralFrameGrid.tasks`` is allowed to allocate the task tuple and
# are rechecked at the producer boundary before a tile payload is allocated.
MAXIMUM_RETURNING_FRAME_PIXELS: Final = 16_777_216
MAXIMUM_RETURNING_TILE_TASKS: Final = 65_536
MAXIMUM_RETURNING_TILE_PAYLOAD_BYTES: Final = 64 * 1024 * 1024
MAXIMUM_RETURNING_TILE_WORKERS: Final = 64
MAXIMUM_RETURNING_TILE_IN_FLIGHT_TASKS: Final = 256

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "same-code authenticated finite-grid piecewise-annulus "
            "returning-thermal spectral tile"
        ),
        "wholePixelIntegrator": (
            "integrate_returning_thermal_spectral_pixel"
        ),
        "wholePixelLiveAuthorityChecks": "exactly start and end",
        "acceptsArbitraryPixelIntegrator": False,
        "acceptsGenericSpectralRaySampler": False,
        "reusesScientificSpectralPixelAbi": True,
        "reusesSpectralProductPublicationContract": True,
        "isContinuumRadialReturningRadiationSolution": False,
        "hasIndependentPhysicsOracle": False,
        "isCompleteKerrbb": False,
        "includesReturningRadiationStressWorkFS": False,
        "includesScatteringOrSolvedAtmosphere": False,
        "includesPolarization": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "prohibitedClaim": (
            "Do not describe this same-code finite-grid piecewise-annulus "
            "tile path as a continuum returning-radiation solution, an "
            "independent oracle, complete KERRBB, F_S, a solved atmosphere, "
            "polarization, or GRMHD."
        ),
    }
)

# Freeze the reviewed whole-pixel boundary once.  There is deliberately no
# constructor field or public argument through which a caller can replace it.
_RETURNING_PIXEL_INTEGRATOR_ENTRY: Final = (
    integrate_returning_thermal_spectral_pixel
)
_RETURNING_PIXEL_INTEGRATOR: Final = _RETURNING_PIXEL_INTEGRATOR_ENTRY
_PIXEL_BOUNDS_ENTRY: Final = SpectralFrameGrid.pixel_bounds
_CONTAINS_TASK_ENTRY: Final = SpectralFrameGrid.contains_task
_PACK_ADAPTIVE_PIXEL_ENTRY: Final = pack_adaptive_pixel
_GRID_TASKS_ENTRY: Final = SpectralFrameGrid.tasks
_GRID_DESCRIPTOR_ENTRY: Final = SpectralFrameGrid.descriptor
_LAYOUT_DESCRIPTOR_ENTRY: Final = SpectralPixelLayout.descriptor
_ADAPTIVE_OPTIONS_DESCRIPTOR_ENTRY: Final = adaptive_pixel_options_descriptor
_BUILD_SPECTRAL_JOB_SPEC_ENTRY: Final = build_spectral_job_spec
_SPECTRAL_JOB_PARAMETERS_ENTRY: Final = spectral_job_parameters


def _exact_positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be an exact positive integer")
    return value


def _task_sort_key(value: TaskKey) -> tuple[int, int, int, int, int]:
    if type(value) is not TaskKey:
        raise SpectralProductError("JobSpec contains a foreign tile key type")
    result = (
        value.sample_index,
        value.y,
        value.x,
        value.width,
        value.height,
    )
    if any(type(item) is not int for item in result):
        raise SpectralProductError("JobSpec tile key fields changed type")
    return result


def _exact_float_tuple(value: object, label: str) -> tuple[float, ...]:
    if type(value) is not tuple or not value:
        raise TypeError(f"{label} must be a non-empty exact tuple")
    if any(type(item) is not float or not math.isfinite(item) for item in value):
        raise TypeError(f"{label} must contain finite exact floats")
    return value


def _validated_layout_and_grid(
    layout: SpectralPixelLayout,
    grid: SpectralFrameGrid,
) -> tuple[SpectralPixelLayout, SpectralFrameGrid]:
    if type(layout) is not SpectralPixelLayout:
        raise TypeError("layout must have the exact SpectralPixelLayout type")
    frequencies = _exact_float_tuple(
        object.__getattribute__(layout, "observer_frequencies_hz"),
        "observer frequencies",
    )
    rebuilt_layout = SpectralPixelLayout(frequencies)

    if type(grid) is not SpectralFrameGrid:
        raise TypeError("grid must have the exact SpectralFrameGrid type")
    width = object.__getattribute__(grid, "width_pixels")
    height = object.__getattribute__(grid, "height_pixels")
    if type(width) is not int or type(height) is not int:
        raise TypeError("frame dimensions must be exact integers")
    bounds = tuple(
        object.__getattribute__(grid, name)
        for name in (
            "screen_x_min",
            "screen_x_max",
            "screen_y_min",
            "screen_y_max",
        )
    )
    if any(type(value) is not float or not math.isfinite(value) for value in bounds):
        raise TypeError("frame bounds must be finite exact floats")
    samples = object.__getattribute__(grid, "sample_indices")
    if type(samples) is not tuple or not samples:
        raise TypeError("sample_indices must be a non-empty exact tuple")
    if any(type(value) is not int for value in samples):
        raise TypeError("sample_indices must contain exact integers")
    rebuilt_grid = SpectralFrameGrid(
        width,
        height,
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        samples,
    )
    return rebuilt_layout, rebuilt_grid


def _validated_options(options: AdaptivePixelOptions) -> AdaptivePixelOptions:
    if type(options) is not AdaptivePixelOptions:
        raise TypeError("options must have the exact AdaptivePixelOptions type")
    integer_names = (
        "minimum_depth",
        "maximum_depth",
        "maximum_ray_evaluations",
    )
    integers = tuple(object.__getattribute__(options, name) for name in integer_names)
    if any(type(value) is not int for value in integers):
        raise TypeError("adaptive integer fields must have exact int type")
    absolute = _exact_float_tuple(
        object.__getattribute__(options, "radiance_absolute_tolerances"),
        "radiance absolute tolerances",
    )
    ceilings = _exact_float_tuple(
        object.__getattribute__(options, "radiance_guard_ceilings"),
        "radiance guard ceilings",
    )
    scalar_names = (
        "radiance_relative_tolerance",
        "unresolved_solid_angle_fraction_tolerance",
        "weighted_log_g_tolerance",
        "weighted_direction_tolerance_rad",
    )
    scalars = tuple(object.__getattribute__(options, name) for name in scalar_names)
    if any(type(value) is not float or not math.isfinite(value) for value in scalars):
        raise TypeError("adaptive scalar fields must be finite exact floats")
    stencil = object.__getattribute__(options, "stencil_version")
    if type(stencil) is not str:
        raise TypeError("adaptive stencil_version must have exact str type")
    return AdaptivePixelOptions(
        minimum_depth=integers[0],
        maximum_depth=integers[1],
        maximum_ray_evaluations=integers[2],
        radiance_absolute_tolerances=absolute,
        radiance_relative_tolerance=scalars[0],
        unresolved_solid_angle_fraction_tolerance=scalars[1],
        weighted_log_g_tolerance=scalars[2],
        weighted_direction_tolerance_rad=scalars[3],
        radiance_guard_ceilings=ceilings,
        stencil_version=stencil,
    )


def _require_exact_json_input(value: object, label: str) -> None:
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return
    if value_type is float:
        if not math.isfinite(value):
            raise TypeError(f"{label} contains a non-finite float")
        return
    if value_type in (list, tuple):
        for index, item in enumerate(value):
            _require_exact_json_input(item, f"{label}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{label} contains a non-exact string key")
            _require_exact_json_input(item, f"{label}.{key}")
        return
    raise TypeError(f"{label} contains a non-exact JSON value type")


def _thaw_exact_frozen_json(
    value: object,
    mapping_type: type,
    label: str,
) -> Any:
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise TypeError(f"{label} contains a non-finite float")
        return value
    if value_type is tuple:
        return [
            _thaw_exact_frozen_json(item, mapping_type, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if value_type is mapping_type:
        items = object.__getattribute__(value, "_items")
        if type(items) is not tuple:
            raise TypeError(f"{label} mapping storage must be an exact tuple")
        result: dict[str, Any] = {}
        previous_key_bytes: bytes | None = None
        for index, item in enumerate(items):
            if type(item) is not tuple or len(item) != 2:
                raise TypeError(f"{label} mapping item {index} is malformed")
            key, child = item
            if type(key) is not str:
                raise TypeError(f"{label} mapping key must have exact str type")
            key_bytes = key.encode("utf-8")
            if previous_key_bytes is not None and key_bytes <= previous_key_bytes:
                raise ValueError(f"{label} mapping keys are not canonical")
            previous_key_bytes = key_bytes
            result[key] = _thaw_exact_frozen_json(
                child,
                mapping_type,
                f"{label}.{key}",
            )
        return result
    raise TypeError(f"{label} contains a non-exact frozen JSON value")


def _exact_input_documents(
    inputs: object,
) -> tuple[dict[str, Any], ...]:
    if type(inputs) is not tuple:
        raise TypeError("JobSpec inputs must be an exact tuple")
    result: list[dict[str, Any]] = []
    previous_uri_bytes: bytes | None = None
    for artifact in inputs:
        if type(artifact) is not InputArtifact:
            raise TypeError("JobSpec inputs must have exact InputArtifact type")
        uri = object.__getattribute__(artifact, "uri")
        byte_length = object.__getattribute__(artifact, "byte_length")
        sha256 = object.__getattribute__(artifact, "sha256")
        if type(uri) is not str or type(byte_length) is not int or type(sha256) is not str:
            raise TypeError("JobSpec input fields must have exact schema types")
        InputArtifact(uri, byte_length, sha256)
        uri_bytes = uri.encode("utf-8")
        if previous_uri_bytes is not None and uri_bytes <= previous_uri_bytes:
            raise ValueError("JobSpec inputs are not in canonical URI order")
        previous_uri_bytes = uri_bytes
        result.append(
            {"byteLength": byte_length, "sha256": sha256, "uri": uri}
        )
    return tuple(result)


def _exact_source_hashes(source_hashes: object) -> tuple[str, ...]:
    if type(source_hashes) is not tuple:
        raise TypeError("JobSpec producer source hashes must be an exact tuple")
    result: list[str] = []
    previous: bytes | None = None
    for value in source_hashes:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise TypeError("JobSpec producer source hash has a non-exact schema")
        encoded = value.encode("ascii")
        if previous is not None and encoded <= previous:
            raise ValueError("JobSpec producer source hashes are not canonical")
        previous = encoded
        result.append(value)
    if not result:
        raise ValueError("JobSpec needs at least one producer source hash")
    return tuple(result)


def _exact_job_spec_document(
    spec: JobSpec,
) -> tuple[dict[str, Any], tuple[tuple[int, int, int, int, int], ...]]:
    if type(spec) is not JobSpec:
        raise TypeError("bound job specification must have exact JobSpec type")
    producer = object.__getattribute__(spec, "producer")
    algorithm_version = object.__getattribute__(spec, "algorithm_version")
    record_bytes = object.__getattribute__(spec, "record_bytes")
    if type(producer) is not str or type(algorithm_version) is not str:
        raise TypeError("JobSpec producer and version must have exact str type")
    if type(record_bytes) is not int or record_bytes < 1:
        raise TypeError("JobSpec record_bytes must have exact positive int type")
    tasks = object.__getattribute__(spec, "tasks")
    if type(tasks) is not tuple or not tasks:
        raise TypeError("JobSpec tasks must be a non-empty exact tuple")
    task_keys = tuple(_task_sort_key(item) for item in tasks)
    if task_keys != tuple(sorted(task_keys)) or len(set(task_keys)) != len(task_keys):
        raise ValueError("JobSpec task keys are not unique canonical primitives")
    inputs = _exact_input_documents(object.__getattribute__(spec, "inputs"))
    source_hashes = _exact_source_hashes(
        object.__getattribute__(spec, "producer_source_hashes")
    )
    parameters = object.__getattribute__(spec, "parameters")
    mapping_type = type(parameters)
    if mapping_type is not _FrozenJsonObject:
        raise TypeError("JobSpec parameters do not have frozen mapping storage")
    parameter_document = _thaw_exact_frozen_json(
        parameters,
        mapping_type,
        "JobSpec.parameters",
    )
    if type(parameter_document) is not dict:
        raise TypeError("JobSpec parameters must decode to an exact object")
    document = {
        "algorithmVersion": algorithm_version,
        "inputs": list(inputs),
        "parameters": parameter_document,
        "producer": producer,
        "producerSourceHashes": list(source_hashes),
        "recordBytes": record_bytes,
        "schema": JOB_SCHEMA,
        "tasks": [
            {
                "height": key[4],
                "sampleIndex": key[0],
                "width": key[3],
                "x": key[2],
                "y": key[1],
            }
            for key in task_keys
        ],
    }
    return document, task_keys


def validate_returning_radiation_spectral_resources(
    layout: SpectralPixelLayout,
    grid: SpectralFrameGrid,
    *,
    tile_width: int,
    tile_height: int,
) -> tuple[int, int, int]:
    """Validate fixed product limits without materializing any task keys."""

    rebuilt_layout, rebuilt_grid = _validated_layout_and_grid(layout, grid)
    width = _exact_positive_integer(rebuilt_grid.width_pixels, "frame width")
    height = _exact_positive_integer(rebuilt_grid.height_pixels, "frame height")
    samples = len(rebuilt_grid.sample_indices)
    if samples < 1:
        raise ValueError("frame must contain at least one sample index")
    tile_width = _exact_positive_integer(tile_width, "tile_width")
    tile_height = _exact_positive_integer(tile_height, "tile_height")

    frame_pixels = width * height * samples
    if frame_pixels > MAXIMUM_RETURNING_FRAME_PIXELS:
        raise ValueError(
            "returning-radiation frame pixel count exceeds the fixed maximum"
        )
    task_count = (
        ((width + tile_width - 1) // tile_width)
        * ((height + tile_height - 1) // tile_height)
        * samples
    )
    if task_count > MAXIMUM_RETURNING_TILE_TASKS:
        raise ValueError(
            "returning-radiation tile task count exceeds the fixed maximum"
        )
    maximum_tile_payload_bytes = (
        min(width, tile_width)
        * min(height, tile_height)
        * rebuilt_layout.record_bytes
    )
    if maximum_tile_payload_bytes > MAXIMUM_RETURNING_TILE_PAYLOAD_BYTES:
        raise ValueError(
            "returning-radiation tile payload exceeds the fixed byte maximum"
        )
    return frame_pixels, task_count, maximum_tile_payload_bytes


def _assert_returning_pixel_runtime_bindings_stable() -> None:
    if (
        integrate_returning_thermal_spectral_pixel
        is not _RETURNING_PIXEL_INTEGRATOR_ENTRY
        or returning_frame_module.integrate_returning_thermal_spectral_pixel
        is not _RETURNING_PIXEL_INTEGRATOR_ENTRY
        or _RETURNING_PIXEL_INTEGRATOR is not _RETURNING_PIXEL_INTEGRATOR_ENTRY
    ):
        raise SpectralProductError(
            "returning-radiation whole-pixel integrator identity changed"
        )
    if (
        spectral_product_module.SpectralFrameGrid is not SpectralFrameGrid
        or SpectralFrameGrid.pixel_bounds is not _PIXEL_BOUNDS_ENTRY
        or SpectralFrameGrid.contains_task is not _CONTAINS_TASK_ENTRY
    ):
        raise SpectralProductError(
            "returning-radiation pixel-bounds identity changed"
        )
    if (
        spectral_frame_module.pack_adaptive_pixel
        is not _PACK_ADAPTIVE_PIXEL_ENTRY
        or pack_adaptive_pixel is not _PACK_ADAPTIVE_PIXEL_ENTRY
    ):
        raise SpectralProductError(
            "returning-radiation pixel packer identity changed"
        )


def build_returning_radiation_spectral_job_spec(
    sampler: KerrReturningRadiationFiniteThicknessRaySampler,
    layout: SpectralPixelLayout,
    grid: SpectralFrameGrid,
    options: AdaptivePixelOptions,
    *,
    tile_width: int,
    tile_height: int,
    numeric_backend: Mapping[str, Any],
    inputs: Sequence[InputArtifact] = (),
    producer_source_hashes: Sequence[str],
) -> JobSpec:
    """Build the existing spectral ``JobSpec`` for the fixed producer."""

    if type(sampler) is not KerrReturningRadiationFiniteThicknessRaySampler:
        raise TypeError(
            "sampler must have the exact returning-radiation finite-thickness "
            "type"
        )
    validate_returning_radiation_spectral_resources(
        layout,
        grid,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    _validated_options(options)
    _require_exact_json_input(numeric_backend, "numeric backend")
    sampler_descriptor = sampler.descriptor()
    _require_exact_json_input(sampler_descriptor, "sampler descriptor")
    assert_returning_radiation_spectral_runtime_bindings()
    return _BUILD_SPECTRAL_JOB_SPEC_ENTRY(
        layout,
        grid,
        options,
        sampler_descriptor,
        tile_width=tile_width,
        tile_height=tile_height,
        numeric_backend=numeric_backend,
        inputs=inputs,
        producer_source_hashes=producer_source_hashes,
        producer=RETURNING_ADAPTIVE_TILE_PRODUCER_ID,
        algorithm_version=RETURNING_ADAPTIVE_TILE_ALGORITHM_VERSION,
    )


@dataclass(frozen=True, slots=True)
class ReturningRadiationAdaptiveSpectralTileProducer:
    """Produce ABI-compatible tiles through the fixed returning pixel path."""

    sampler: KerrReturningRadiationFiniteThicknessRaySampler
    layout: SpectralPixelLayout
    grid: SpectralFrameGrid
    options: AdaptivePixelOptions
    numeric_backend: Mapping[str, Any]
    job_spec: JobSpec
    _sampler_descriptor_bytes: bytes = field(init=False, repr=False)
    _numeric_backend_bytes: bytes = field(init=False, repr=False)
    _parameters_bytes: bytes = field(init=False, repr=False)
    _job_spec_document_bytes: bytes = field(init=False, repr=False)
    _job_task_keys: tuple[tuple[int, int, int, int, int], ...] = field(
        init=False,
        repr=False,
    )
    _job_tasks_reference: tuple[TaskKey, ...] = field(init=False, repr=False)
    _job_parameters_reference: Mapping[str, Any] = field(init=False, repr=False)
    _job_inputs_reference: tuple[InputArtifact, ...] = field(init=False, repr=False)
    _job_source_hashes_reference: tuple[str, ...] = field(init=False, repr=False)
    _job_inputs_bytes: bytes = field(init=False, repr=False)
    _job_source_hashes_bytes: bytes = field(init=False, repr=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError(
            "ReturningRadiationAdaptiveSpectralTileProducer cannot be subclassed"
        )

    def __post_init__(self) -> None:
        if type(self) is not ReturningRadiationAdaptiveSpectralTileProducer:
            raise TypeError("returning tile producer must have its exact type")
        if type(self.sampler) is not KerrReturningRadiationFiniteThicknessRaySampler:
            raise TypeError(
                "sampler must have the exact returning-radiation "
                "finite-thickness type"
            )
        _rebuilt_layout, rebuilt_grid = _validated_layout_and_grid(
            self.layout,
            self.grid,
        )
        if rebuilt_grid.record_count > MAXIMUM_RETURNING_FRAME_PIXELS:
            raise ValueError(
                "returning-radiation frame pixel count exceeds the fixed maximum"
            )
        _validated_options(self.options)
        _require_exact_json_input(self.numeric_backend, "numeric backend")
        sampler_descriptor_input = self.sampler.descriptor()
        _require_exact_json_input(
            sampler_descriptor_input,
            "sampler descriptor",
        )
        assert_returning_radiation_spectral_runtime_bindings()
        parameters = _SPECTRAL_JOB_PARAMETERS_ENTRY(
            self.layout,
            self.grid,
            self.options,
            sampler_descriptor_input,
            self.numeric_backend,
        )
        sampler_descriptor = parameters["samplerDescriptor"]
        numeric_backend = parameters["numericBackend"]
        if type(sampler_descriptor) is not dict or type(numeric_backend) is not dict:
            raise AssertionError("spectral parameter canonicalization changed shape")
        job_document, task_keys = _exact_job_spec_document(self.job_spec)
        if len(task_keys) > MAXIMUM_RETURNING_TILE_TASKS:
            raise ValueError("bound JobSpec exceeds the fixed tile task maximum")
        if (
            self.job_spec.producer.encode("utf-8")
            != RETURNING_ADAPTIVE_TILE_PRODUCER_ID.encode("utf-8")
            or self.job_spec.algorithm_version.encode("utf-8")
            != RETURNING_ADAPTIVE_TILE_ALGORITHM_VERSION.encode("utf-8")
            or self.job_spec.record_bytes != self.layout.record_bytes
            or canonical_json_bytes(job_document["parameters"])
            != canonical_json_bytes(parameters)
        ):
            raise SpectralProductError(
                "bound JobSpec does not match the returning producer configuration"
            )
        for task_key, task in zip(task_keys, self.job_spec.tasks):
            if (
                not _CONTAINS_TASK_ENTRY(self.grid, task)
                or task_key[3] * task_key[4] * self.job_spec.record_bytes
                > MAXIMUM_RETURNING_TILE_PAYLOAD_BYTES
            ):
                raise SpectralProductError(
                    "bound JobSpec task lies outside fixed frame or payload limits"
                )
        object.__setattr__(
            self,
            "_sampler_descriptor_bytes",
            canonical_json_bytes(sampler_descriptor),
        )
        object.__setattr__(
            self,
            "_numeric_backend_bytes",
            canonical_json_bytes(numeric_backend),
        )
        object.__setattr__(
            self,
            "_parameters_bytes",
            canonical_json_bytes(parameters),
        )
        object.__setattr__(
            self,
            "_job_spec_document_bytes",
            canonical_json_bytes(job_document),
        )
        object.__setattr__(self, "_job_task_keys", task_keys)
        object.__setattr__(self, "_job_tasks_reference", self.job_spec.tasks)
        object.__setattr__(
            self,
            "_job_parameters_reference",
            self.job_spec.parameters,
        )
        object.__setattr__(self, "_job_inputs_reference", self.job_spec.inputs)
        object.__setattr__(
            self,
            "_job_source_hashes_reference",
            self.job_spec.producer_source_hashes,
        )
        object.__setattr__(
            self,
            "_job_inputs_bytes",
            canonical_json_bytes(job_document["inputs"]),
        )
        object.__setattr__(
            self,
            "_job_source_hashes_bytes",
            canonical_json_bytes(job_document["producerSourceHashes"]),
        )

    @property
    def sampler_descriptor(self) -> Mapping[str, Any]:
        # Do not expose nested mutable aliases into the sealed producer state.
        return MappingProxyType(
            json.loads(self._sampler_descriptor_bytes)
        )

    def _assert_configuration_stable(self) -> None:
        if type(self) is not ReturningRadiationAdaptiveSpectralTileProducer:
            raise SpectralProductError("returning tile producer type changed")
        if type(self.sampler) is not KerrReturningRadiationFiniteThicknessRaySampler:
            raise SpectralProductError(
                "returning-radiation sampler type changed after construction"
            )
        _validated_layout_and_grid(self.layout, self.grid)
        _validated_options(self.options)
        _require_exact_json_input(self.numeric_backend, "numeric backend")
        sampler_descriptor = self.sampler.descriptor()
        _require_exact_json_input(sampler_descriptor, "sampler descriptor")
        assert_returning_radiation_spectral_runtime_bindings()
        current = _SPECTRAL_JOB_PARAMETERS_ENTRY(
            self.layout,
            self.grid,
            self.options,
            sampler_descriptor,
            self.numeric_backend,
        )
        if (
            canonical_json_bytes(current["samplerDescriptor"])
            != self._sampler_descriptor_bytes
            or canonical_json_bytes(current["numericBackend"])
            != self._numeric_backend_bytes
            or canonical_json_bytes(current) != self._parameters_bytes
        ):
            raise SpectralProductError(
                "returning-radiation producer configuration changed after "
                "construction"
            )

    def assert_bound_job_spec_stable(
        self,
        spec: JobSpec | None = None,
        *,
        full_document: bool = False,
    ) -> None:
        if type(self) is not ReturningRadiationAdaptiveSpectralTileProducer:
            raise SpectralProductError("returning tile producer type changed")
        _RETURNING_CONFIGURATION_ASSERT_ENTRY(self)
        candidate = self.job_spec if spec is None else spec
        if type(candidate) is not JobSpec or candidate is not self.job_spec:
            raise SpectralProductError(
                "tile call does not use the producer-bound JobSpec object"
            )
        if (
            object.__getattribute__(candidate, "tasks")
            is not self._job_tasks_reference
            or object.__getattribute__(candidate, "parameters")
            is not self._job_parameters_reference
            or object.__getattribute__(candidate, "inputs")
            is not self._job_inputs_reference
            or object.__getattribute__(candidate, "producer_source_hashes")
            is not self._job_source_hashes_reference
        ):
            raise SpectralProductError("bound JobSpec fields changed object identity")
        producer = object.__getattribute__(candidate, "producer")
        version = object.__getattribute__(candidate, "algorithm_version")
        record_bytes = object.__getattribute__(candidate, "record_bytes")
        if (
            type(producer) is not str
            or type(version) is not str
            or type(record_bytes) is not int
            or producer.encode("utf-8")
            != RETURNING_ADAPTIVE_TILE_PRODUCER_ID.encode("utf-8")
            or version.encode("utf-8")
            != RETURNING_ADAPTIVE_TILE_ALGORITHM_VERSION.encode("utf-8")
            or record_bytes != self.layout.record_bytes
        ):
            raise SpectralProductError("bound JobSpec scalar schema changed")
        parameters = _thaw_exact_frozen_json(
            candidate.parameters,
            _FrozenJsonObject,
            "JobSpec.parameters",
        )
        inputs = _exact_input_documents(candidate.inputs)
        source_hashes = _exact_source_hashes(candidate.producer_source_hashes)
        if (
            canonical_json_bytes(parameters) != self._parameters_bytes
            or canonical_json_bytes(inputs) != self._job_inputs_bytes
            or canonical_json_bytes(source_hashes) != self._job_source_hashes_bytes
        ):
            raise SpectralProductError("bound JobSpec provenance changed")
        if full_document:
            document, task_keys = _exact_job_spec_document(candidate)
            if (
                task_keys != self._job_task_keys
                or canonical_json_bytes(document) != self._job_spec_document_bytes
            ):
                raise SpectralProductError("bound JobSpec document changed")

    def __call__(self, spec: JobSpec, key: TaskKey) -> bytes:
        if type(self) is not ReturningRadiationAdaptiveSpectralTileProducer:
            raise TypeError("returning tile producer must have its exact type")
        if type(spec) is not JobSpec or type(key) is not TaskKey:
            raise TypeError("tile producer expects exact JobSpec and TaskKey values")
        if (
            ReturningRadiationAdaptiveSpectralTileProducer.__call__
            is not _RETURNING_TILE_PRODUCER_CALL_ENTRY
        ):
            raise SpectralProductError(
                "returning-radiation tile producer call identity changed"
            )
        assert_returning_radiation_spectral_runtime_bindings()
        _RETURNING_JOB_SPEC_ASSERT_ENTRY(self, spec)
        key_sort_key = _task_sort_key(key)
        task_index = bisect_left(
            self._job_task_keys,
            key_sort_key,
        )
        if (
            task_index == len(self._job_task_keys)
            or self._job_task_keys[task_index] != key_sort_key
            or not _CONTAINS_TASK_ENTRY(self.grid, key)
        ):
            raise SpectralProductError("tile key is outside the bound frame")

        expected_payload_bytes = key.width * key.height * spec.record_bytes
        if expected_payload_bytes > MAXIMUM_RETURNING_TILE_PAYLOAD_BYTES:
            raise SpectralProductError(
                "returning-radiation tile payload exceeds the fixed byte maximum"
            )

        payload = bytearray()
        for y in range(key.y, key.y + key.height):
            for x in range(key.x, key.x + key.width):
                x_min, x_max, y_min, y_max = _PIXEL_BOUNDS_ENTRY(
                    self.grid,
                    x,
                    y,
                )
                # One call owns the complete adaptive pixel and therefore its
                # two live-authority boundary checks.
                result = _RETURNING_PIXEL_INTEGRATOR(
                    self.sampler,
                    self.layout.observer_frequencies_hz,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    options=self.options,
                )
                payload.extend(
                    _PACK_ADAPTIVE_PIXEL_ENTRY(
                        self.layout,
                        result,
                        self.options,
                    )
                )
        _RETURNING_CONFIGURATION_ASSERT_ENTRY(self)
        if len(payload) != expected_payload_bytes:
            raise SpectralProductError(
                "returning-radiation tile payload length disagrees with its key"
            )
        return bytes(payload)


_RETURNING_TILE_PRODUCER_CALL_ENTRY: Final = (
    ReturningRadiationAdaptiveSpectralTileProducer.__call__
)
_RETURNING_CONFIGURATION_ASSERT_ENTRY: Final = (
    ReturningRadiationAdaptiveSpectralTileProducer._assert_configuration_stable
)
_RETURNING_JOB_SPEC_ASSERT_ENTRY: Final = (
    ReturningRadiationAdaptiveSpectralTileProducer.assert_bound_job_spec_stable
)


def assert_returning_radiation_spectral_runtime_bindings() -> None:
    """Reject runtime replacement of any frozen whole-pixel tile boundary."""

    if (
        ReturningRadiationAdaptiveSpectralTileProducer.__call__
        is not _RETURNING_TILE_PRODUCER_CALL_ENTRY
        or ReturningRadiationAdaptiveSpectralTileProducer._assert_configuration_stable
        is not _RETURNING_CONFIGURATION_ASSERT_ENTRY
        or ReturningRadiationAdaptiveSpectralTileProducer.assert_bound_job_spec_stable
        is not _RETURNING_JOB_SPEC_ASSERT_ENTRY
    ):
        raise SpectralProductError(
            "returning-radiation tile producer call identity changed"
        )
    if (
        spectral_product_module.build_spectral_job_spec
        is not _BUILD_SPECTRAL_JOB_SPEC_ENTRY
        or build_spectral_job_spec is not _BUILD_SPECTRAL_JOB_SPEC_ENTRY
        or spectral_product_module.spectral_job_parameters
        is not _SPECTRAL_JOB_PARAMETERS_ENTRY
        or spectral_job_parameters is not _SPECTRAL_JOB_PARAMETERS_ENTRY
        or spectral_product_module.adaptive_pixel_options_descriptor
        is not _ADAPTIVE_OPTIONS_DESCRIPTOR_ENTRY
        or adaptive_pixel_options_descriptor
        is not _ADAPTIVE_OPTIONS_DESCRIPTOR_ENTRY
        or SpectralFrameGrid.tasks is not _GRID_TASKS_ENTRY
        or SpectralFrameGrid.descriptor is not _GRID_DESCRIPTOR_ENTRY
        or SpectralPixelLayout.descriptor is not _LAYOUT_DESCRIPTOR_ENTRY
    ):
        raise SpectralProductError(
            "returning-radiation spectral job construction identity changed"
        )
    _assert_returning_pixel_runtime_bindings_stable()


def invoke_returning_radiation_spectral_tile_producer(
    producer: ReturningRadiationAdaptiveSpectralTileProducer,
    spec: JobSpec,
    key: TaskKey,
) -> bytes:
    """Invoke the frozen unbound tile entry after checking public ownership."""

    if type(producer) is not ReturningRadiationAdaptiveSpectralTileProducer:
        raise TypeError("producer must have the exact returning tile type")
    if (
        ReturningRadiationAdaptiveSpectralTileProducer.__call__
        is not _RETURNING_TILE_PRODUCER_CALL_ENTRY
    ):
        raise SpectralProductError(
            "returning-radiation tile producer call identity changed"
        )
    return _RETURNING_TILE_PRODUCER_CALL_ENTRY(producer, spec, key)


__all__ = (
    "RETURNING_ADAPTIVE_TILE_ALGORITHM_VERSION",
    "RETURNING_ADAPTIVE_TILE_PRODUCER_ID",
    "MAXIMUM_RETURNING_FRAME_PIXELS",
    "MAXIMUM_RETURNING_TILE_IN_FLIGHT_TASKS",
    "MAXIMUM_RETURNING_TILE_PAYLOAD_BYTES",
    "MAXIMUM_RETURNING_TILE_TASKS",
    "MAXIMUM_RETURNING_TILE_WORKERS",
    "SCIENTIFIC_STATUS",
    "ReturningRadiationAdaptiveSpectralTileProducer",
    "assert_returning_radiation_spectral_runtime_bindings",
    "build_returning_radiation_spectral_job_spec",
    "invoke_returning_radiation_spectral_tile_producer",
    "validate_returning_radiation_spectral_resources",
)
