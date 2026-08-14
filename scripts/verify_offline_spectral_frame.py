#!/usr/bin/env python3
"""Independently verify a scientific adaptive spectral frame product.

Validation authenticates the manifest, every tile, the public binary pixel ABI,
complete tile topology, JobSpec identity, and all declared aggregate evidence.
It does not rerun the sampler or geodesic solver.  Unknown and recognized
producer identifiers therefore remain structural-only, never physics-verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, NoReturn, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.adaptive_frame import AdaptivePixelOptions
from offline.job import (
    InputArtifact,
    JobSpec,
    TaskKey,
    canonical_json_bytes,
)
from offline.spectral_frame import (
    HAS_ESCAPE_DIRECTION,
    HAS_FREQUENCY_SHIFT,
    REQUIRED_CONVERGENCE_MASK,
    SOURCE_DISK,
    SOURCE_ESCAPED_BOUNDARY,
    ScientificSpectralPixelRecord,
    SpectralFrameError,
    SpectralPixelLayout,
    unpack_spectral_pixel,
)

try:
    from scripts.verify_nr_contract import (
        ContractError,
        audit_schema_dialect,
        validate_json_schema,
    )
except ModuleNotFoundError:  # Direct ``python3 scripts/...`` execution.
    from verify_nr_contract import (
        ContractError,
        audit_schema_dialect,
        validate_json_schema,
    )


DEFAULT_SCHEMA = (
    ROOT / "schemas" / "offline-scientific-spectral-frame-v1.schema.json"
)
PRODUCT_SCHEMA = "blackhole.scientific-spectral-frame/v1"
SCHEMA_ID = (
    "https://github.com/ShuoleiWang/blackhole/schemas/"
    "offline-scientific-spectral-frame-v1.schema.json"
)
PIXEL_LAYOUT_ID = "blackhole.scientific-spectral-pixel/le-f64-v1"
ADAPTIVE_TILE_PRODUCER_ID = "blackhole.adaptive-spectral-tile"
ADAPTIVE_TILE_ALGORITHM_VERSION = "1.0.0"
KERR_SAMPLER_IMPLEMENTATION_ID = "exact-kerr-nt-spectral-ray-sampler/v2"


class OfflineSpectralFrameContractError(ContractError):
    """A deterministic scientific spectral frame validation failure."""


def fail(path: str, message: str) -> NoReturn:
    raise OfflineSpectralFrameContractError(f"{path}: {message}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(label, f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        fail(label, f"non-finite JSON number {value!r} is forbidden")

    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as error:
        fail(label, f"JSON is not UTF-8: {error}")
    except json.JSONDecodeError as error:
        fail(label, f"invalid JSON: {error}")


def _read_stable_file(path: Path, label: str) -> bytes:
    descriptor: int | None = None
    try:
        if path.is_symlink():
            fail(label, "symlinked files are forbidden")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(label, "expected a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(descriptor)
    except OSError as error:
        fail(label, f"unable to read file: {error}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(payload) != before.st_size:
        fail(label, "file changed while it was being read")
    return payload


def _normalized_uri(value: Any, path: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        fail(path, "URI must be a non-empty normalized relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or "." in pure.parts
        or ".." in pure.parts
        or pure.as_posix() != value
    ):
        fail(path, "URI must be normalized, relative, and traversal-free")
    return pure


def _read_relative_file(root: Path, uri: Any, path: str) -> bytes:
    pure = _normalized_uri(uri, path)
    if root.is_symlink():
        fail(path, "a symlinked artifact root is forbidden")
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            fail(path, "symlinked artifacts or path components are forbidden")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    directory_descriptors: list[int] = []
    file_descriptor: int | None = None
    try:
        directory_descriptors.append(os.open(root, directory_flags))
        for part in pure.parts[:-1]:
            directory_descriptors.append(
                os.open(
                    part,
                    directory_flags,
                    dir_fd=directory_descriptors[-1],
                )
            )
        file_descriptor = os.open(
            pure.parts[-1],
            file_flags,
            dir_fd=directory_descriptors[-1],
        )
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(path, "artifact must be a regular file")
        with os.fdopen(file_descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(file_descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity or len(payload) != before.st_size:
            fail(path, "artifact changed while it was being read")
        return payload
    except OSError as error:
        fail(path, f"unable to open traversal-safe artifact: {error}")
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)


def _load_json_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_stable_file(path, label)
    parsed = _strict_json_bytes(payload, label)
    if not isinstance(parsed, dict):
        fail(label, "JSON root must be an object")
    if _read_stable_file(path, label) != payload:
        fail(label, "file changed while it was being parsed")
    return parsed, payload


def _canonical_hash(value: Any) -> str:
    try:
        payload = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        fail("$", f"value is not finite canonical JSON: {error}")
    return _sha256_bytes(payload)


def _validate_manifest_sidecar(
    root: Path,
    manifest: dict[str, Any],
    payload: bytes,
) -> None:
    sidecar = _read_relative_file(
        root,
        manifest["integrity"]["manifestSidecar"],
        "$.integrity.manifestSidecar",
    )
    expected = f"{_sha256_bytes(payload)}  manifest.json\n".encode("ascii")
    if sidecar != expected:
        fail(
            "$.integrity.manifestSidecar",
            "sidecar must exactly bind the manifest SHA-256",
        )


def _adaptive_options(value: dict[str, Any]) -> AdaptivePixelOptions:
    try:
        options = AdaptivePixelOptions(
            minimum_depth=value["minimumDepth"],
            maximum_depth=value["maximumDepth"],
            maximum_ray_evaluations=value["maximumRayEvaluations"],
            radiance_absolute_tolerances=tuple(
                value["radianceAbsoluteTolerances"]
            ),
            radiance_relative_tolerance=value["radianceRelativeTolerance"],
            unresolved_solid_angle_fraction_tolerance=(
                value["unresolvedSolidAngleFractionTolerance"]
            ),
            weighted_log_g_tolerance=value["weightedLogGTolerance"],
            weighted_direction_tolerance_rad=(
                value["weightedDirectionToleranceRad"]
            ),
            radiance_guard_ceilings=tuple(value["radianceGuardCeilings"]),
            stencil_version=value["stencilVersion"],
        )
    except (KeyError, TypeError, ValueError) as error:
        fail("$.adaptivePixelOptions", f"invalid adaptive options: {error}")
    if _adaptive_options_descriptor(options) != value:
        fail("$.adaptivePixelOptions", "options do not round-trip canonically")
    return options


def _adaptive_options_descriptor(options: AdaptivePixelOptions) -> dict[str, Any]:
    return {
        "maximumDepth": options.maximum_depth,
        "maximumRayEvaluations": options.maximum_ray_evaluations,
        "minimumDepth": options.minimum_depth,
        "radianceAbsoluteTolerances": list(
            options.radiance_absolute_tolerances
        ),
        "radianceGuardCeilings": list(options.radiance_guard_ceilings),
        "radianceRelativeTolerance": options.radiance_relative_tolerance,
        "stencilVersion": options.stencil_version,
        "unresolvedSolidAngleFractionTolerance": (
            options.unresolved_solid_angle_fraction_tolerance
        ),
        "weightedDirectionToleranceRad": (
            options.weighted_direction_tolerance_rad
        ),
        "weightedLogGTolerance": options.weighted_log_g_tolerance,
    }


def _layout(manifest: dict[str, Any]) -> SpectralPixelLayout:
    try:
        layout = SpectralPixelLayout(
            tuple(manifest["observerFrequencyBinsHz"])
        )
    except (TypeError, ValueError) as error:
        fail("$.observerFrequencyBinsHz", f"invalid frequency layout: {error}")
    if dict(layout.descriptor()) != manifest["pixelLayout"]:
        fail("$.pixelLayout", "descriptor does not match the public pixel ABI")
    if manifest["pixelLayout"]["id"] != PIXEL_LAYOUT_ID:
        fail("$.pixelLayout.id", "unsupported pixel layout")
    return layout


def _task_from_dict(value: dict[str, Any], path: str) -> TaskKey:
    try:
        return TaskKey(
            value["sampleIndex"],
            value["y"],
            value["x"],
            value["width"],
            value["height"],
        )
    except (KeyError, TypeError, ValueError) as error:
        fail(path, f"invalid task: {error}")


def _job_spec(manifest: dict[str, Any]) -> JobSpec:
    raw = manifest["producer"]["jobSpec"]
    try:
        inputs = tuple(
            InputArtifact(
                item["uri"],
                item["byteLength"],
                item["sha256"],
            )
            for item in raw["inputs"]
        )
        tasks = tuple(
            _task_from_dict(item, f"$.producer.jobSpec.tasks[{index}]")
            for index, item in enumerate(raw["tasks"])
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
    except (KeyError, TypeError, ValueError) as error:
        fail("$.producer.jobSpec", f"invalid JobSpec: {error}")
    if spec.as_dict() != raw:
        fail("$.producer.jobSpec", "JobSpec is not in canonical order")
    producer = manifest["producer"]
    if (
        producer["id"] != spec.producer
        or producer["algorithmVersion"] != spec.algorithm_version
        or producer["jobKey"] != spec.job_key
    ):
        fail("$.producer", "producer wrapper disagrees with its JobSpec")
    return spec


def _frame_values(frame: dict[str, Any]) -> tuple[int, int, tuple[int, ...]]:
    width = frame["widthPixels"]
    height = frame["heightPixels"]
    samples = tuple(frame["sampleIndices"])
    bounds = frame["screenBounds"]
    if samples != tuple(sorted(set(samples))):
        fail("$.frame.sampleIndices", "sample indices must be unique and sorted")
    if bounds["xMax"] <= bounds["xMin"] or bounds["yMax"] <= bounds["yMin"]:
        fail("$.frame.screenBounds", "screen bounds must increase strictly")
    return width, height, samples


def _tile_task(entry: dict[str, Any], path: str) -> TaskKey:
    tile = entry["tile"]
    return _task_from_dict(
        {
            "height": tile["height"],
            "sampleIndex": entry["sampleIndex"],
            "width": tile["width"],
            "x": tile["x"],
            "y": tile["y"],
        },
        path,
    )


def _validate_topology(
    tasks: tuple[TaskKey, ...],
    width: int,
    height: int,
    samples: tuple[int, ...],
) -> None:
    if tasks != tuple(sorted(tasks)):
        fail("$.tiles", "tiles are not in canonical TaskKey order")
    if {task.sample_index for task in tasks} != set(samples):
        fail("$.tiles", "tile samples do not exactly match the frame")
    for task in tasks:
        if task.x + task.width > width or task.y + task.height > height:
            fail("$.tiles", "a tile extends outside the frame")
    for sample_index in samples:
        selected = tuple(task for task in tasks if task.sample_index == sample_index)
        boundaries = sorted(
            {
                0,
                height,
                *(task.y for task in selected),
                *(task.y + task.height for task in selected),
            }
        )
        for lower, upper in zip(boundaries, boundaries[1:]):
            intervals = sorted(
                (task.x, task.x + task.width)
                for task in selected
                if task.y <= lower and task.y + task.height >= upper
            )
            cursor = 0
            for start, end in intervals:
                if start != cursor or end <= start:
                    fail("$.tiles", "tiles overlap or leave a coverage gap")
                cursor = end
            if cursor != width:
                fail("$.tiles", "tiles do not cover a complete horizontal strip")


def _expected_tile_uri(task: TaskKey) -> str:
    return f"tiles/{task.file_stem}.spx"


def _positive_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) > 0.0


def _validate_record(record: ScientificSpectralPixelRecord, path: str) -> None:
    if (
        record.convergence_mask & REQUIRED_CONVERGENCE_MASK
        != REQUIRED_CONVERGENCE_MASK
    ):
        fail(path, "pixel is missing a required convergence gate")
    has_g = bool(record.convergence_mask & HAS_FREQUENCY_SHIFT)
    has_direction = bool(record.convergence_mask & HAS_ESCAPE_DIRECTION)
    if bool(record.source_mask & SOURCE_DISK) != has_g:
        fail(path, "g presence disagrees with disk coverage")
    if bool(record.source_mask & SOURCE_ESCAPED_BOUNDARY) != has_direction:
        fail(path, "direction presence disagrees with escaped coverage")
    if not has_g and not (
        _positive_zero(record.minimum_frequency_shift_g)
        and _positive_zero(record.maximum_frequency_shift_g)
    ):
        fail(path, "absent g requires positive-zero sentinels")
    if not has_direction and not (
        _positive_zero(record.maximum_escape_direction_span_rad)
        and _positive_zero(record.weighted_escape_direction_variation_rad)
    ):
        fail(path, "absent direction requires positive-zero diagnostics")


class _CompensatedSum:
    __slots__ = ("total", "correction")

    def __init__(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        updated = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - updated) + value
        else:
            self.correction += (value - updated) + self.total
        self.total = updated

    def value(self) -> float:
        return self.total + self.correction


class _Summary:
    def __init__(self, count: int) -> None:
        self.records = 0
        self.solid = _CompensatedSum()
        self.integrated = tuple(_CompensatedSum() for _ in range(count))
        self.errors = tuple(_CompensatedSum() for _ in range(count))
        self.disk = _CompensatedSum()
        self.captured = _CompensatedSum()
        self.escaped = _CompensatedSum()
        self.unresolved = _CompensatedSum()
        self.minimum_g: float | None = None
        self.maximum_g: float | None = None
        self.maximum_direction = 0.0
        self.maximum_weighted_g = 0.0
        self.maximum_weighted_direction = 0.0
        self.maximum_null = 0.0
        self.maximum_metric = 0.0
        self.maximum_event = 0.0
        self.maximum_covector = 0.0
        self.maximum_radius = 0.0
        self.maximum_relative_g = 0.0
        self.maximum_bracket = 0.0
        self.ray_samples = 0
        self.maximum_depth = 0
        self.maximum_accepted = 0
        self.maximum_rejected = 0
        self.source_union = 0

    def add(self, record: ScientificSpectralPixelRecord) -> None:
        self.records += 1
        solid = record.pixel_solid_angle_sr
        self.solid.add(solid)
        for index, value in enumerate(record.mean_specific_intensities_nu):
            self.integrated[index].add(value * solid)
        for index, value in enumerate(record.mean_estimated_absolute_errors_nu):
            self.errors[index].add(value * solid)
        self.disk.add(record.disk_coverage_fraction * solid)
        self.captured.add(record.captured_boundary_coverage_fraction * solid)
        self.escaped.add(record.escaped_boundary_coverage_fraction * solid)
        self.unresolved.add(record.unresolved_solid_angle_fraction * solid)
        if record.convergence_mask & HAS_FREQUENCY_SHIFT:
            self.minimum_g = (
                record.minimum_frequency_shift_g
                if self.minimum_g is None
                else min(self.minimum_g, record.minimum_frequency_shift_g)
            )
            self.maximum_g = (
                record.maximum_frequency_shift_g
                if self.maximum_g is None
                else max(self.maximum_g, record.maximum_frequency_shift_g)
            )
        self.maximum_direction = max(
            self.maximum_direction,
            record.maximum_escape_direction_span_rad,
        )
        self.maximum_weighted_g = max(
            self.maximum_weighted_g,
            record.weighted_log_g_variation,
        )
        self.maximum_weighted_direction = max(
            self.maximum_weighted_direction,
            record.weighted_escape_direction_variation_rad,
        )
        self.maximum_null = max(self.maximum_null, record.maximum_null_residual)
        self.maximum_metric = max(
            self.maximum_metric,
            record.maximum_metric_interpolation_error,
        )
        self.maximum_event = max(
            self.maximum_event,
            record.maximum_terminal_event_difference_m,
        )
        self.maximum_covector = max(
            self.maximum_covector,
            record.maximum_terminal_covector_relative_difference,
        )
        self.maximum_radius = max(
            self.maximum_radius,
            record.maximum_disk_radius_difference_m,
        )
        self.maximum_relative_g = max(
            self.maximum_relative_g,
            record.maximum_relative_g_difference,
        )
        self.maximum_bracket = max(
            self.maximum_bracket,
            record.maximum_surface_bracket_affine_width,
        )
        self.ray_samples += record.sample_count
        self.maximum_depth = max(self.maximum_depth, record.maximum_depth_reached)
        self.maximum_accepted = max(
            self.maximum_accepted,
            record.maximum_accepted_steps,
        )
        self.maximum_rejected = max(
            self.maximum_rejected,
            record.maximum_rejected_steps,
        )
        self.source_union |= record.source_mask

    def descriptor(self) -> dict[str, Any]:
        return {
            "estimatedAbsoluteErrorNuSr": [item.value() for item in self.errors],
            "integratedSpecificIntensityNuSr": [
                item.value() for item in self.integrated
            ],
            "maximumAcceptedSteps": self.maximum_accepted,
            "maximumAdaptiveDepth": self.maximum_depth,
            "maximumDiskRadiusDifferenceM": self.maximum_radius,
            "maximumEscapeDirectionSpanRad": self.maximum_direction,
            "maximumMetricInterpolationError": self.maximum_metric,
            "maximumNullResidual": self.maximum_null,
            "maximumRejectedSteps": self.maximum_rejected,
            "maximumRelativeGDifference": self.maximum_relative_g,
            "maximumSurfaceBracketAffineWidth": self.maximum_bracket,
            "maximumTerminalCovectorRelativeDifference": self.maximum_covector,
            "maximumTerminalEventDifferenceM": self.maximum_event,
            "maximumWeightedDirectionVariationRad": (
                self.maximum_weighted_direction
            ),
            "maximumWeightedLogGVariation": self.maximum_weighted_g,
            "maximumFrequencyShiftG": self.maximum_g,
            "minimumFrequencyShiftG": self.minimum_g,
            "recordCount": self.records,
            "requiredConvergenceMask": REQUIRED_CONVERGENCE_MASK,
            "sourceMaskUnion": self.source_union,
            "sourceSolidAngleSr": {
                "capturedBoundary": self.captured.value(),
                "disk": self.disk.value(),
                "escapedBoundary": self.escaped.value(),
            },
            "totalPixelSolidAngleSr": self.solid.value(),
            "totalRaySamples": self.ray_samples,
            "unresolvedSolidAngleSr": self.unresolved.value(),
        }


def _validate_tiles(
    manifest: dict[str, Any],
    root: Path,
    layout: SpectralPixelLayout,
    job_spec: JobSpec,
) -> tuple[dict[str, Any], set[str]]:
    width, height, samples = _frame_values(manifest["frame"])
    entries = manifest["tiles"]
    tasks = tuple(
        _tile_task(entry, f"$.tiles[{index}]")
        for index, entry in enumerate(entries)
    )
    if tasks != tuple(job_spec.tasks):
        fail("$.tiles", "tiles do not exactly match JobSpec tasks")
    _validate_topology(tasks, width, height, samples)
    summary = _Summary(layout.frequency_count)
    allowed = {"manifest.json", manifest["integrity"]["manifestSidecar"]}
    for tile_index, (entry, task) in enumerate(zip(entries, tasks)):
        path = f"$.tiles[{tile_index}]"
        expected_records = task.width * task.height
        if entry["recordCount"] != expected_records:
            fail(f"{path}.recordCount", "record count does not equal tile area")
        artifact = entry["payload"]
        expected_uri = _expected_tile_uri(task)
        if artifact["uri"] != expected_uri:
            fail(f"{path}.payload.uri", "URI does not match the canonical tile key")
        payload = _read_relative_file(root, artifact["uri"], f"{path}.payload.uri")
        expected_bytes = expected_records * layout.record_bytes
        if artifact["byteLength"] != expected_bytes or len(payload) != expected_bytes:
            fail(f"{path}.payload.byteLength", "tile byte length is invalid")
        actual_hash = _sha256_bytes(payload)
        if artifact["sha256"] != actual_hash:
            fail(f"{path}.payload.sha256", "tile hash mismatch")
        for record_index in range(expected_records):
            offset = record_index * layout.record_bytes
            record_path = f"{path}.records[{record_index}]"
            try:
                record = unpack_spectral_pixel(
                    layout,
                    payload[offset : offset + layout.record_bytes],
                )
            except (SpectralFrameError, TypeError, ValueError) as error:
                fail(record_path, f"invalid public pixel record: {error}")
            _validate_record(record, record_path)
            summary.add(record)
        allowed.add(artifact["uri"])
    return summary.descriptor(), allowed


def _validate_no_extra_files(root: Path, allowed_files: set[str]) -> None:
    allowed_directories = {
        PurePosixPath(uri).parent.as_posix()
        for uri in allowed_files
        if PurePosixPath(uri).parent.as_posix() != "."
    }
    if root.is_symlink():
        fail("$files", "a symlinked dataset root is forbidden")
    for directory, directories, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in tuple(directories):
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                fail("$files", f"symlinked directory {relative!r} is forbidden")
            if relative not in allowed_directories:
                fail("$files", f"undeclared output directory {relative!r}")
        for name in files:
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                fail("$files", f"symlinked file {relative!r} is forbidden")
            if relative not in allowed_files:
                fail("$files", f"undeclared output file {relative!r}")


def _validate_configuration_identity(
    manifest: dict[str, Any],
    layout: SpectralPixelLayout,
    options: AdaptivePixelOptions,
    spec: JobSpec,
) -> None:
    sampler = manifest["sampler"]
    backend = manifest["runtimeNumericBackend"]
    for value, path in (
        (sampler, "$.sampler"),
        (backend, "$.runtimeNumericBackend"),
    ):
        implementation_id = value["descriptor"].get("implementationId")
        if not isinstance(implementation_id, str) or not implementation_id:
            fail(f"{path}.descriptor", "a non-empty implementationId is required")
        if value["descriptorSha256"] != _canonical_hash(value["descriptor"]):
            fail(f"{path}.descriptorSha256", "descriptor hash mismatch")
    expected_parameters = {
        "adaptivePixelOptions": _adaptive_options_descriptor(options),
        "frame": manifest["frame"],
        "numericBackend": backend["descriptor"],
        "observerFrequencyBinsHz": list(layout.observer_frequencies_hz),
        "pixelLayout": dict(layout.descriptor()),
        "samplerDescriptor": sampler["descriptor"],
        "schema": PRODUCT_SCHEMA,
    }
    if spec.as_dict()["parameters"] != expected_parameters:
        fail(
            "$.producer.jobSpec.parameters",
            "JobSpec does not bind the manifest scientific configuration",
        )
    if spec.record_bytes != layout.record_bytes:
        fail("$.producer.jobSpec.recordBytes", "record size disagrees with ABI")
    configuration = {
        **expected_parameters,
        "jobKey": spec.job_key,
        "jobSpec": spec.as_dict(),
    }
    expected_hash = _canonical_hash(configuration)
    if manifest["integrity"]["configurationSha256"] != expected_hash:
        fail("$.integrity.configurationSha256", "configuration digest mismatch")


def _validate_product_identity(manifest: dict[str, Any]) -> None:
    identity = {
        "configurationSha256": manifest["integrity"][
            "configurationSha256"
        ],
        "schema": PRODUCT_SCHEMA,
        "summary": manifest["summary"],
        "tiles": manifest["tiles"],
    }
    product_hash = _canonical_hash(identity)
    if manifest["integrity"]["productSha256"] != product_hash:
        fail("$.integrity.productSha256", "product digest mismatch")
    expected_id = f"scientific-spectral-frame-{product_hash[:24]}"
    if manifest["id"] != expected_id:
        fail("$.id", "product id does not match the product digest")


def validate_scientific_spectral_frame(
    manifest_path: Path | str,
    schema_path: Path | str = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Validate one product and return deterministic structural evidence."""

    path = Path(manifest_path).absolute()
    if path.name != "manifest.json":
        fail("$", "v1 manifest must be named 'manifest.json'")
    schema, _schema_payload = _load_json_file(Path(schema_path).absolute(), "$schema")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail("$schema.$schema", "only JSON Schema Draft 2020-12 is supported")
    if schema.get("$id") != SCHEMA_ID:
        fail("$schema.$id", "unexpected scientific spectral frame schema id")
    audit_schema_dialect(schema)

    manifest, manifest_payload = _load_json_file(path, "$")
    validate_json_schema(manifest, schema, schema)
    try:
        expected_manifest_payload = canonical_json_bytes(manifest)
    except (TypeError, ValueError) as error:
        fail("$", f"manifest is not finite canonical JSON: {error}")
    if manifest_payload != expected_manifest_payload:
        fail("$", "manifest is not in canonical JSON encoding")
    root = path.parent
    _validate_manifest_sidecar(root, manifest, manifest_payload)
    layout = _layout(manifest)
    options = _adaptive_options(manifest["adaptivePixelOptions"])
    spec = _job_spec(manifest)
    _validate_configuration_identity(manifest, layout, options, spec)
    recomputed_summary, allowed = _validate_tiles(manifest, root, layout, spec)
    if canonical_json_bytes(manifest["summary"]) != canonical_json_bytes(
        recomputed_summary
    ):
        fail("$.summary", "declared summary does not match decoded records")
    _validate_product_identity(manifest)
    _validate_no_extra_files(root, allowed)

    producer_is_recognized = (
        spec.producer == ADAPTIVE_TILE_PRODUCER_ID
        and spec.algorithm_version == ADAPTIVE_TILE_ALGORITHM_VERSION
    )
    sampler_id = manifest["sampler"]["descriptor"]["implementationId"]
    identifiers_are_kerr = (
        producer_is_recognized
        and sampler_id == KERR_SAMPLER_IMPLEMENTATION_ID
    )
    provenance_scope = (
        "recognized-kerr-identifiers-structural-only"
        if identifiers_are_kerr
        else "unknown-producer-or-sampler-structural-only"
    )
    return {
        "id": manifest["id"],
        "physicsVerified": False,
        "provenanceScope": provenance_scope,
        "recordCount": recomputed_summary["recordCount"],
        "status": "scientific-spectral-frame-structural-contract-conformant",
        "tileCount": len(manifest["tiles"]),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        report = validate_scientific_spectral_frame(
            arguments.manifest,
            arguments.schema,
        )
    except ContractError as error:
        print(f"offline spectral frame validation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
