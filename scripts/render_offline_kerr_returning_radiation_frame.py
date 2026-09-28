#!/usr/bin/env python3
"""Publish an authenticated returning-radiation thermal spectral frame.

The renderer requires a complete pre-existing fixed production-forward Kerr
returning-radiation direction cache.  It authenticates that cache without
tracing missing directions and reduces it into a
finite-grid bolometric kernel, solves the absorbed returning-radiation thermal
fixed point, freezes a piecewise-annulus emission authority, and then renders
the exact 471-bin CIE grid through the finite-thickness camera sampler.  The
published artifact is structurally verified and then byte-replayed with that
same live process-local sampler before success is reported.

This is a same-code, finite-grid, piecewise-constant-annulus scalar ``I_nu``
product.  It is not a continuum radial returning-radiation solution, an
independent physics oracle, complete KERRBB, the returning-radiation stress
term F_S, a solved or scattering atmosphere, polarization transport, GRMHD,
or caustic-complete GRRT.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
while str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from offline.adaptive_frame import AdaptivePixelOptions  # noqa: E402
from offline.authenticated_artifact import (  # noqa: E402
    MAXIMUM_AUTHENTICATED_ARTIFACT_BYTE_LENGTH,
    MAXIMUM_AUTHENTICATED_ARTIFACT_TOTAL_BYTE_LENGTH,
    authenticate_stable_artifact,
)
from offline.cie_color import (  # noqa: E402
    CIE_ROW_COUNT,
    DEFAULT_CIE_CSV,
    DEFAULT_CIE_METADATA,
)
from offline.job import (  # noqa: E402
    InputArtifact,
    JobRun,
    JobSpec,
    TaskKey,
    TaskResult,
    canonical_json_bytes,
    run_job,
)
import offline.job as job_module  # noqa: E402
from offline.kerr_finite_thickness_area import (  # noqa: E402
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_finite_thickness_frame import (  # noqa: E402
    KerrFiniteThicknessRaySampler,
)
from offline.kerr_returning_radiation_finite_thickness_frame import (  # noqa: E402
    MAXIMUM_ANNULUS_EDGE_CLEARANCE_MULTIPLIER,
    KerrReturningRadiationFiniteThicknessRaySampler,
)
import offline.kerr_returning_radiation_finite_thickness_frame as returning_frame_module  # noqa: E402
from offline.kerr_returning_radiation_frame_context import (  # noqa: E402
    MAXIMUM_AUTHENTICATED_ANNULUS_COUNT,
    ValidatedReturningThermalAuthority,
    authenticate_returning_thermal_emission,
)
import offline.kerr_returning_radiation_frame_context as frame_context_module  # noqa: E402
from offline.kerr_returning_radiation_kernel import (  # noqa: E402
    KerrReturningRadiationKernelPolicy,
)
from offline.kerr_returning_radiation_kernel_cached import (  # noqa: E402
    KerrCachedKernelExecutionAudit,
    KerrCachedReturningRadiationKernelExecution,
    KerrKernelReductionConfiguration,
    build_forward_kerr_returning_radiation_kernel_cache_definition,
    integrate_existing_cached_kerr_returning_radiation_energy_kernel,
)
import offline.kerr_returning_radiation_kernel_cached as cached_kernel_module  # noqa: E402
from offline.kerr_returning_radiation_kernel_jobs import (  # noqa: E402
    FORWARD,
    KerrKernelDirectionCacheDefinition,
    KerrKernelDirectionTaskPlan,
    MAXIMUM_IN_FLIGHT_TASKS as MAXIMUM_KERNEL_IN_FLIGHT_TASKS,
    MAXIMUM_WORKERS as MAXIMUM_KERNEL_WORKERS,
)
from offline.kerr_returning_radiation_refinement_checkpoint import (  # noqa: E402
    KerrReturningRadiationVerifiedRefinementCheckpoint,
    verify_kerr_returning_radiation_refinement_checkpoint,
)
import offline.kerr_returning_radiation_refinement_checkpoint as refinement_checkpoint_module  # noqa: E402
import offline.kerr_returning_radiation_live_replay_attestation as attestation_module  # noqa: E402
from offline.kerr_returning_radiation_live_replay_attestation import (  # noqa: E402
    ReturningRadiationLiveReplayAttestationPublication,
    default_live_replay_attestation_directory,
    publish_returning_radiation_live_replay_attestation,
)
import offline.kerr_returning_radiation_spectral_product as returning_product_module  # noqa: E402
from offline.kerr_returning_radiation_spectral_product import (  # noqa: E402
    MAXIMUM_RETURNING_FRAME_PIXELS,
    MAXIMUM_RETURNING_TILE_IN_FLIGHT_TASKS,
    MAXIMUM_RETURNING_TILE_PAYLOAD_BYTES,
    MAXIMUM_RETURNING_TILE_TASKS,
    MAXIMUM_RETURNING_TILE_WORKERS,
    ReturningRadiationAdaptiveSpectralTileProducer,
    assert_returning_radiation_spectral_runtime_bindings,
    build_returning_radiation_spectral_job_spec,
    invoke_returning_radiation_spectral_tile_producer,
    validate_returning_radiation_spectral_resources,
)
from offline.kerr_returning_radiation_thermal_profile import (  # noqa: E402
    AxisymmetricReturningRadiationThermalProfile,
    solve_certified_kerr_returning_radiation_thermal_profile,
)
import offline.kerr_returning_radiation_thermal_profile as thermal_profile_module  # noqa: E402
from offline.kerr_returning_radiation_thermal_spectrum import (  # noqa: E402
    CertifiedReturningRadiationThermalSpectrumProvider,
    build_certified_returning_radiation_thermal_spectrum_provider,
)
import offline.kerr_returning_radiation_thermal_spectrum as thermal_spectrum_module  # noqa: E402
from offline.returning_radiation import (  # noqa: E402
    ReturningRadiationFixedPointPolicy,
)
from offline.spectral_frame import FIXED_RECORD_BYTES, SpectralPixelLayout  # noqa: E402
import offline.spectral_product as spectral_product_module  # noqa: E402
from offline.spectral_product import (  # noqa: E402
    SpectralFrameGrid,
    SpectralProductPublication,
    build_spectral_job_spec as generic_build_spectral_job_spec,
    default_numeric_backend_descriptor,
    publish_spectral_product,
)
import scripts.render_offline_kerr_finite_thickness_frame as base_renderer  # noqa: E402
from scripts.verify_offline_spectral_frame import (  # noqa: E402
    DEFAULT_SCHEMA as DEFAULT_SPECTRAL_SCHEMA,
    validate_scientific_spectral_frame,
)
import scripts.verify_offline_spectral_frame as spectral_verifier_module  # noqa: E402
import scripts.verify_offline_kerr_returning_radiation_live_replay_attestation as attestation_verifier_module  # noqa: E402,F401


CIE_CSV_INPUT_URI = base_renderer.CIE_CSV_INPUT_URI
CIE_METADATA_INPUT_URI = base_renderer.CIE_METADATA_INPUT_URI
RETURNING_CIE_RECORD_BYTES = 16 * CIE_ROW_COUNT + FIXED_RECORD_BYTES
MAXIMUM_KERNEL_DIRECTIONS_PER_TASK = 4096
MAXIMUM_PRODUCER_SOURCE_ARTIFACT_COUNT = 64
MAXIMUM_PRODUCER_SOURCE_ARTIFACT_BYTE_LENGTH = (
    MAXIMUM_AUTHENTICATED_ARTIFACT_BYTE_LENGTH
)
MAXIMUM_PRODUCER_SOURCE_ARTIFACT_TOTAL_BYTE_LENGTH = (
    MAXIMUM_AUTHENTICATED_ARTIFACT_TOTAL_BYTE_LENGTH
)
MAXIMUM_CIE_ARTIFACT_BYTE_LENGTH = MAXIMUM_AUTHENTICATED_ARTIFACT_BYTE_LENGTH
MAXIMUM_CIE_ARTIFACT_TOTAL_BYTE_LENGTH = (
    MAXIMUM_AUTHENTICATED_ARTIFACT_TOTAL_BYTE_LENGTH
)
RUNTIME_BINDING_SCOPE = (
    "public-and-owning-module-callable-rebinding-is-identity-gated",
    "private-call-entry-globals-are-trusted-in-process-test-seams",
    "no-resistance-claim-for-malicious-same-process-private-global-rewrite",
)

_RETURNING_TILE_PRODUCER_TYPE_ENTRY = ReturningRadiationAdaptiveSpectralTileProducer
_RETURNING_TILE_INVOKER_ENTRY = invoke_returning_radiation_spectral_tile_producer
_RETURNING_JOB_SPEC_BUILDER_ENTRY = build_returning_radiation_spectral_job_spec
_RETURNING_JOB_SPEC_STABILITY_ENTRY = (
    ReturningRadiationAdaptiveSpectralTileProducer.assert_bound_job_spec_stable
)
_GENERIC_SPECTRAL_JOB_SPEC_BUILDER_ENTRY = generic_build_spectral_job_spec
_SPECTRAL_GRID_TASKS_ENTRY = SpectralFrameGrid.tasks
_RUN_JOB_PUBLIC_ENTRY = run_job
_RUN_JOB_CALL_ENTRY = _RUN_JOB_PUBLIC_ENTRY
_PUBLISH_PUBLIC_ENTRY = publish_spectral_product
_PUBLISH_CALL_ENTRY = _PUBLISH_PUBLIC_ENTRY
_VERIFIER_PUBLIC_ENTRY = validate_scientific_spectral_frame
_VERIFIER_CALL_ENTRY = _VERIFIER_PUBLIC_ENTRY
_DEFAULT_NUMERIC_BACKEND_PUBLIC_ENTRY = default_numeric_backend_descriptor
_DEFAULT_NUMERIC_BACKEND_CALL_ENTRY = _DEFAULT_NUMERIC_BACKEND_PUBLIC_ENTRY
_RETURNING_RUNTIME_BINDING_ASSERTION_ENTRY = (
    assert_returning_radiation_spectral_runtime_bindings
)
_CACHED_KERNEL_INTEGRATOR_PUBLIC_ENTRY = (
    integrate_existing_cached_kerr_returning_radiation_energy_kernel
)
_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY = _CACHED_KERNEL_INTEGRATOR_PUBLIC_ENTRY
_REFINEMENT_CHECKPOINT_VERIFIER_PUBLIC_ENTRY = (
    verify_kerr_returning_radiation_refinement_checkpoint
)
_REFINEMENT_CHECKPOINT_VERIFIER_CALL_ENTRY = (
    _REFINEMENT_CHECKPOINT_VERIFIER_PUBLIC_ENTRY
)
_THERMAL_PROFILE_SOLVER_PUBLIC_ENTRY = (
    solve_certified_kerr_returning_radiation_thermal_profile
)
_THERMAL_PROFILE_SOLVER_CALL_ENTRY = _THERMAL_PROFILE_SOLVER_PUBLIC_ENTRY
_THERMAL_PROVIDER_BUILDER_PUBLIC_ENTRY = (
    build_certified_returning_radiation_thermal_spectrum_provider
)
_THERMAL_PROVIDER_BUILDER_CALL_ENTRY = _THERMAL_PROVIDER_BUILDER_PUBLIC_ENTRY
_THERMAL_AUTHENTICATOR_PUBLIC_ENTRY = authenticate_returning_thermal_emission
_THERMAL_AUTHENTICATOR_CALL_ENTRY = _THERMAL_AUTHENTICATOR_PUBLIC_ENTRY
_RETURNING_SAMPLER_TYPE_ENTRY = KerrReturningRadiationFiniteThicknessRaySampler
_RETURNING_SAMPLER_NEW_ENTRY = _RETURNING_SAMPLER_TYPE_ENTRY.__new__
_RETURNING_SAMPLER_INIT_ENTRY = _RETURNING_SAMPLER_TYPE_ENTRY.__init__
_RETURNING_SAMPLER_POST_INIT_ENTRY = _RETURNING_SAMPLER_TYPE_ENTRY.__post_init__
_RETURNING_SAMPLER_CONSTRUCTOR_CALL_ENTRY = _RETURNING_SAMPLER_TYPE_ENTRY
_ATTESTATION_PUBLISH_PUBLIC_ENTRY = publish_returning_radiation_live_replay_attestation
_ATTESTATION_PUBLISH_CALL_ENTRY = _ATTESTATION_PUBLISH_PUBLIC_ENTRY

# Complete repository-owned closure that can change kernel directions,
# thermal authentication, adaptive pixels, the binary ABI, or this orchestration
# path.  CIE data bytes are bound separately as scientific inputs.
PRODUCER_SOURCE_FILES = (
    Path("offline/__init__.py"),
    Path("offline/adaptive_frame.py"),
    Path("offline/authenticated_artifact.py"),
    Path("offline/cie_color.py"),
    Path("offline/disk_atmosphere.py"),
    Path("offline/geodesic.py"),
    Path("offline/job.py"),
    Path("offline/kerr.py"),
    Path("offline/kerr_disk.py"),
    Path("offline/kerr_disk_early_stop.py"),
    Path("offline/kerr_disk_frame.py"),
    Path("offline/kerr_disk_transfer.py"),
    Path("offline/kerr_finite_thickness.py"),
    Path("offline/kerr_finite_thickness_area.py"),
    Path("offline/kerr_finite_thickness_emitter.py"),
    Path("offline/kerr_finite_thickness_frame.py"),
    Path("offline/kerr_finite_thickness_launch.py"),
    Path("offline/kerr_finite_thickness_replay_certificate.py"),
    Path("offline/kerr_finite_thickness_surface.py"),
    Path("offline/kerr_finite_thickness_transfer.py"),
    Path("offline/kerr_returning_radiation_finite_thickness_frame.py"),
    Path("offline/kerr_returning_radiation_finite_thickness_transfer.py"),
    Path("offline/kerr_returning_radiation_frame_context.py"),
    Path("offline/kerr_returning_radiation_convergence_v2.py"),
    Path("offline/kerr_returning_radiation_kernel.py"),
    Path("offline/kerr_returning_radiation_kernel_cached.py"),
    Path("offline/kerr_returning_radiation_kernel_jobs.py"),
    Path("offline/kerr_returning_radiation_live_replay_attestation.py"),
    Path("offline/kerr_returning_radiation_rays.py"),
    Path("offline/kerr_returning_radiation_receiver_kernel.py"),
    Path("offline/kerr_returning_radiation_receiver_rays.py"),
    Path("offline/kerr_returning_radiation_refinement_checkpoint.py"),
    Path("offline/kerr_returning_radiation_spectral_product.py"),
    Path("offline/kerr_returning_radiation_spectral_replay.py"),
    Path("offline/kerr_returning_radiation_thermal_profile.py"),
    Path("offline/kerr_returning_radiation_thermal_spectrum.py"),
    Path("offline/novikov_thorne.py"),
    Path("offline/radiative_transfer.py"),
    Path("offline/returning_radiation.py"),
    Path("offline/returning_radiation_fate_quadrature.py"),
    Path("offline/spacetime.py"),
    Path("offline/spectral_frame.py"),
    Path("offline/spectral_product.py"),
    Path("scripts/render_offline_kerr_finite_thickness_frame.py"),
    Path("scripts/render_offline_kerr_returning_radiation_frame.py"),
    Path("scripts/verify_offline_kerr_returning_radiation_live_replay_attestation.py"),
    Path("scripts/verify_offline_spectral_frame.py"),
    Path("scripts/verify_nr_contract.py"),
    Path("schemas/offline-scientific-spectral-frame-v1.schema.json"),
    Path("schemas/offline-returning-radiation-live-replay-attestation-v1.schema.json"),
)


def _declared_source_module_name(relative: Path) -> str:
    """Map one fixed declared Python source path to its import name."""

    if type(relative) is not type(Path()) or relative.suffix != ".py":
        raise TypeError(
            "declared source module path must be an exact relative .py Path"
        )
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    if (
        not parts
        or parts[0] not in ("offline", "scripts")
        or any(not part.isidentifier() for part in parts)
    ):
        raise ValueError("declared source module path has no supported import name")
    return ".".join(parts)


PRODUCER_SOURCE_MODULES = tuple(
    (_declared_source_module_name(relative), relative)
    for relative in PRODUCER_SOURCE_FILES
    if relative.suffix == ".py"
)
if len(PRODUCER_SOURCE_MODULES) != 48 or len(
    {name for name, _relative in PRODUCER_SOURCE_MODULES}
) != len(PRODUCER_SOURCE_MODULES):
    raise RuntimeError("producer source module closure is not the fixed unique set")

_EXECUTING_SOURCE_RELATIVE = Path(
    "scripts/render_offline_kerr_returning_radiation_frame.py"
)


@dataclass(frozen=True, slots=True)
class BoundInputStableReturningSpectralTileProducer:
    """Recheck the complete source/CIE snapshot around every tile."""

    inner: ReturningRadiationAdaptiveSpectralTileProducer
    source_artifacts: tuple[InputArtifact, ...]
    science_artifacts: tuple[InputArtifact, ...]
    source_root: Path
    cie_csv_path: Path
    cie_metadata_path: Path
    _source_artifact_bytes: bytes = field(init=False, repr=False)
    _science_artifact_bytes: bytes = field(init=False, repr=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError(
            "BoundInputStableReturningSpectralTileProducer cannot be subclassed"
        )

    def __post_init__(self) -> None:
        if type(self) is not BoundInputStableReturningSpectralTileProducer:
            raise TypeError("bound returning tile producer must have its exact type")
        if type(self.inner) is not ReturningRadiationAdaptiveSpectralTileProducer:
            raise TypeError("inner must be the exact returning tile producer")
        object.__setattr__(
            self,
            "_source_artifact_bytes",
            _artifact_manifest_bytes(self.source_artifacts, "source artifacts"),
        )
        object.__setattr__(
            self,
            "_science_artifact_bytes",
            _artifact_manifest_bytes(self.science_artifacts, "science artifacts"),
        )

    def __call__(self, spec: JobSpec, key: TaskKey) -> bytes:
        if type(self) is not BoundInputStableReturningSpectralTileProducer:
            raise TypeError("bound returning tile producer must have its exact type")
        if (
            BoundInputStableReturningSpectralTileProducer.__call__
            is not _BOUND_TILE_PRODUCER_CALL_ENTRY
        ):
            raise RuntimeError("bound returning tile producer call identity changed")
        _assert_source_provenance(self.source_root)
        _assert_returning_tile_runtime_bindings()
        self._assert_artifact_fields_stable()
        planned_numeric_backend = object.__getattribute__(
            self.inner,
            "_numeric_backend_bytes",
        )
        if _current_numeric_backend_bytes() != planned_numeric_backend:
            raise RuntimeError(
                "numeric backend differs before returning tile production"
            )
        assert_bound_inputs_stable(
            self.source_artifacts,
            self.science_artifacts,
            source_root=self.source_root,
            cie_csv_path=self.cie_csv_path,
            cie_metadata_path=self.cie_metadata_path,
        )
        payload = _RETURNING_TILE_INVOKER_ENTRY(self.inner, spec, key)
        _assert_source_provenance(self.source_root)
        assert_bound_inputs_stable(
            self.source_artifacts,
            self.science_artifacts,
            source_root=self.source_root,
            cie_csv_path=self.cie_csv_path,
            cie_metadata_path=self.cie_metadata_path,
        )
        self._assert_artifact_fields_stable()
        if _current_numeric_backend_bytes() != planned_numeric_backend:
            raise RuntimeError(
                "numeric backend changed during returning tile production"
            )
        return payload

    def _assert_artifact_fields_stable(self) -> None:
        if (
            _artifact_manifest_bytes(self.source_artifacts, "source artifacts")
            != self._source_artifact_bytes
            or _artifact_manifest_bytes(
                self.science_artifacts,
                "science artifacts",
            )
            != self._science_artifact_bytes
        ):
            raise RuntimeError(
                "bound returning tile artifact descriptors changed"
            )


_BOUND_TILE_PRODUCER_CALL_ENTRY = (
    BoundInputStableReturningSpectralTileProducer.__call__
)


@dataclass(frozen=True, slots=True)
class _QualifiedV2CheckpointBinding:
    checkpoint_id: str
    input_artifact: InputArtifact
    kernel_descriptor_sha256: str
    scientific_job_key: str
    cache_job_key: str
    reduction_configuration_bytes: bytes


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationFramePlan:
    """Trace-free scientific configuration plus scheduling policy."""

    output_directory: Path
    live_replay_attestation_directory: Path
    tile_cache_root: Path
    kernel_cache_root: Path
    required_v2_checkpoint_manifest: Path
    required_v2_checkpoint_manifest_sha256: str
    required_v2_checkpoint_binding: _QualifiedV2CheckpointBinding
    tile_jobs: int
    tile_max_in_flight: int | None
    kernel_jobs: int
    kernel_max_in_flight: int | None
    kernel_directions_per_task: int
    tile_width: int
    tile_height: int
    layout: SpectralPixelLayout
    grid: SpectralFrameGrid
    adaptive_options: AdaptivePixelOptions
    base_sampler: KerrFiniteThicknessRaySampler
    base_sampler_descriptor: Mapping[str, object]
    annulus_edges_over_mass: tuple[float, ...]
    kernel_policy: KerrReturningRadiationKernelPolicy
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy
    kernel_reduction_configuration: KerrKernelReductionConfiguration
    kernel_reduction_configuration_bytes: bytes
    fixed_point_policy: ReturningRadiationFixedPointPolicy
    annulus_edge_clearance_multiplier: float
    numeric_backend: Mapping[str, object]
    source_artifacts: tuple[InputArtifact, ...]
    science_artifacts: tuple[InputArtifact, ...]
    source_root: Path
    cie_csv_path: Path
    cie_metadata_path: Path
    kernel_cache_definition: KerrKernelDirectionCacheDefinition
    verification_schema: Path


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationFrameExecution:
    cached_kernel: KerrCachedReturningRadiationKernelExecution
    thermal_profile: AxisymmetricReturningRadiationThermalProfile
    thermal_provider: CertifiedReturningRadiationThermalSpectrumProvider
    authority: ValidatedReturningThermalAuthority
    sampler: KerrReturningRadiationFiniteThicknessRaySampler
    job_spec: JobSpec
    job_run: JobRun
    publication: SpectralProductPublication
    verification: Mapping[str, Any]
    live_replay_attestation: ReturningRadiationLiveReplayAttestationPublication
    live_replay_report: Mapping[str, Any]


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _returning_tile_executor(max_workers: int) -> Executor:
    """Keep the deliberately process-local authority inside this process."""

    return ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="returning-spectral-tile",
    )


_RETURNING_TILE_EXECUTOR_PUBLIC_ENTRY = _returning_tile_executor
_RETURNING_TILE_EXECUTOR_CALL_ENTRY = _RETURNING_TILE_EXECUTOR_PUBLIC_ENTRY


def _exact_positive_runtime_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be an exact positive integer")
    return value


def _exact_json_bytes(value: object, label: str) -> bytes:
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return canonical_json_bytes(value)
    if value_type is float:
        if not math.isfinite(value):
            raise TypeError(f"{label} contains a non-finite float")
        return canonical_json_bytes(value)
    if value_type in (list, tuple):
        for index, item in enumerate(value):
            _exact_json_bytes(item, f"{label}[{index}]")
        return canonical_json_bytes(value)
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{label} contains a non-exact string key")
            _exact_json_bytes(item, f"{label}.{key}")
        return canonical_json_bytes(value)
    raise TypeError(f"{label} contains a non-exact JSON value")


def _current_numeric_backend_bytes() -> bytes:
    descriptor = _DEFAULT_NUMERIC_BACKEND_CALL_ENTRY()
    return _exact_json_bytes(descriptor, "current numeric backend")


_LIVE_REPLAY_FROZEN_ENTRY: Any | None = None


def _load_declared_source_modules_for_origin_gate() -> None:
    """Load the one reviewed lazy edge before exact module-origin checking."""

    from offline import kerr_returning_radiation_spectral_replay as replay_module

    if (
        sys.modules.get("offline.kerr_returning_radiation_spectral_replay")
        is not replay_module
    ):
        raise RuntimeError("canonical returning replay module import failed")
    missing = [
        module_name
        for module_name, _relative in PRODUCER_SOURCE_MODULES
        if module_name not in sys.modules
    ]
    if missing:
        raise RuntimeError(
            "declared producer source modules remain unloaded before origin gate: "
            + ", ".join(missing)
        )


def _frozen_live_replay_entry() -> Any:
    """Lazily import once so replay may import this fully initialized module."""

    from offline import kerr_returning_radiation_spectral_replay as replay_module

    global _LIVE_REPLAY_FROZEN_ENTRY
    current = replay_module.validate_returning_radiation_spectral_live_replay
    canonical = replay_module._CANONICAL_LIVE_REPLAY_VALIDATOR_ENTRY
    if current is not canonical:
        raise RuntimeError("returning live-replay callable identity changed")
    if _LIVE_REPLAY_FROZEN_ENTRY is None:
        _LIVE_REPLAY_FROZEN_ENTRY = canonical
    elif canonical is not _LIVE_REPLAY_FROZEN_ENTRY:
        raise RuntimeError("returning live-replay callable identity changed")
    return _LIVE_REPLAY_FROZEN_ENTRY


def _run_returning_radiation_spectral_live_replay(
    manifest_path: Path,
    *,
    sampler: KerrReturningRadiationFiniteThicknessRaySampler,
    schema_path: Path,
) -> Mapping[str, Any]:
    entry = _frozen_live_replay_entry()
    return entry(
        manifest_path,
        sampler=sampler,
        schema_path=schema_path,
    )


_LIVE_REPLAY_PUBLIC_ENTRY = _run_returning_radiation_spectral_live_replay
_LIVE_REPLAY_CALL_ENTRY = _LIVE_REPLAY_PUBLIC_ENTRY


def _validated_live_replay_report(
    report: object,
    publication: SpectralProductPublication,
) -> Mapping[str, Any]:
    """Keep structural and live evidence in separate exact report schemas."""

    if type(report) is not dict:
        raise TypeError("live replay must return an exact report object")
    if any(type(key) is not str for key in report):
        raise TypeError("live replay report keys must have exact string type")
    true_fields = (
        "structuralContractVerified",
        "producerIdentityCurrentMatch",
        "jobSpecVerified",
        "samplerDescriptorLiveMatch",
        "numericBackendCurrentMatch",
        "sourceArtifactsCurrentMatch",
        "frameGeodesicsReplayed",
        "frozenThermalSnapshotBound",
        "pixelBytesExact",
    )
    false_fields = (
        "thermalFixedPointReplayed",
        "directionCacheRecordsReplayed",
        "directionRaysRetraced",
        "independentPhysicsOracle",
    )
    integer_fields = (
        "recordCount",
        "tileCount",
        "totalRaySamples",
        "totalFrameGeodesicsReplayed",
    )
    text_fields = ("replayScope", "scientificScope", "sourceHashScope")
    expected_fields = {
        "id",
        "status",
        *true_fields,
        *false_fields,
        *integer_fields,
        *text_fields,
    }
    if set(report) != expected_fields:
        raise RuntimeError("live replay returned a non-exact report schema")
    if (
        type(report["id"]) is not str
        or report["id"].encode("utf-8")
        != publication.product_id.encode("utf-8")
        or type(report["status"]) is not str
        or report["status"].encode("utf-8")
        != b"returning-radiation-live-sampler-byte-replay-conformant"
    ):
        raise RuntimeError("live replay report identity or status is invalid")
    if any(report[name] is not True for name in true_fields):
        raise RuntimeError("live replay report lacks required positive evidence")
    if any(report[name] is not False for name in false_fields):
        raise RuntimeError("live replay report overclaims its scientific evidence")
    if any(
        type(report[name]) is not int or report[name] < 0
        for name in integer_fields
    ):
        raise RuntimeError("live replay report contains invalid counters")
    if (
        report["recordCount"] != publication.record_count
        or report["tileCount"] != publication.tile_count
        or report["totalFrameGeodesicsReplayed"]
        != 2 * report["totalRaySamples"]
    ):
        raise RuntimeError("live replay counters do not close")
    if any(type(report[name]) is not str or not report[name] for name in text_fields):
        raise RuntimeError("live replay report contains an invalid scope statement")
    return json.loads(canonical_json_bytes(report))


def _artifact_manifest_bytes(
    artifacts: Sequence[InputArtifact],
    label: str,
) -> bytes:
    if type(artifacts) is not tuple or any(
        type(artifact) is not InputArtifact for artifact in artifacts
    ):
        raise TypeError(f"{label} must be an exact tuple of InputArtifact values")
    return canonical_json_bytes(tuple(artifact.as_dict() for artifact in artifacts))


def _assert_returning_scientific_runtime_bindings() -> None:
    if (
        integrate_existing_cached_kerr_returning_radiation_energy_kernel
        is not _CACHED_KERNEL_INTEGRATOR_PUBLIC_ENTRY
        or cached_kernel_module.integrate_existing_cached_kerr_returning_radiation_energy_kernel
        is not _CACHED_KERNEL_INTEGRATOR_PUBLIC_ENTRY
        or solve_certified_kerr_returning_radiation_thermal_profile
        is not _THERMAL_PROFILE_SOLVER_PUBLIC_ENTRY
        or thermal_profile_module.solve_certified_kerr_returning_radiation_thermal_profile
        is not _THERMAL_PROFILE_SOLVER_PUBLIC_ENTRY
        or build_certified_returning_radiation_thermal_spectrum_provider
        is not _THERMAL_PROVIDER_BUILDER_PUBLIC_ENTRY
        or thermal_spectrum_module.build_certified_returning_radiation_thermal_spectrum_provider
        is not _THERMAL_PROVIDER_BUILDER_PUBLIC_ENTRY
        or authenticate_returning_thermal_emission
        is not _THERMAL_AUTHENTICATOR_PUBLIC_ENTRY
        or frame_context_module.authenticate_returning_thermal_emission
        is not _THERMAL_AUTHENTICATOR_PUBLIC_ENTRY
        or verify_kerr_returning_radiation_refinement_checkpoint
        is not _REFINEMENT_CHECKPOINT_VERIFIER_PUBLIC_ENTRY
        or refinement_checkpoint_module.verify_kerr_returning_radiation_refinement_checkpoint
        is not _REFINEMENT_CHECKPOINT_VERIFIER_PUBLIC_ENTRY
        or KerrReturningRadiationFiniteThicknessRaySampler
        is not _RETURNING_SAMPLER_TYPE_ENTRY
        or returning_frame_module.KerrReturningRadiationFiniteThicknessRaySampler
        is not _RETURNING_SAMPLER_TYPE_ENTRY
        or _RETURNING_SAMPLER_TYPE_ENTRY.__new__
        is not _RETURNING_SAMPLER_NEW_ENTRY
        or _RETURNING_SAMPLER_TYPE_ENTRY.__init__
        is not _RETURNING_SAMPLER_INIT_ENTRY
        or _RETURNING_SAMPLER_TYPE_ENTRY.__post_init__
        is not _RETURNING_SAMPLER_POST_INIT_ENTRY
    ):
        raise RuntimeError("returning scientific runtime binding changed")


def _require_qualified_v2_checkpoint(
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> KerrReturningRadiationVerifiedRefinementCheckpoint:
    """Preflight a qualified source-current checkpoint before cache access."""

    if (
        type(manifest_path) is not type(Path())
        or not manifest_path.is_absolute()
        or manifest_path != _lexical_absolute_path(manifest_path)
        or manifest_path.name != "manifest.json"
    ):
        raise ValueError(
            "required v2 checkpoint must be an exact canonical absolute "
            "manifest.json Path"
        )
    if (
        type(expected_manifest_sha256) is not str
        or len(expected_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_manifest_sha256)
    ):
        raise ValueError(
            "required v2 checkpoint SHA-256 must be an exact lowercase digest"
        )
    _assert_returning_scientific_runtime_bindings()
    verified = _REFINEMENT_CHECKPOINT_VERIFIER_CALL_ENTRY(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if type(verified) is not KerrReturningRadiationVerifiedRefinementCheckpoint:
        raise TypeError("required v2 checkpoint verifier returned a foreign type")
    if (
        type(verified.output_directory) is not type(Path())
        or verified.output_directory != manifest_path.parent
        or type(verified.manifest_path) is not type(Path())
        or verified.manifest_path != manifest_path
        or type(verified.manifest_sha256) is not str
        or verified.manifest_sha256 != expected_manifest_sha256
        or type(verified.checkpoint_id) is not str
        or not verified.checkpoint_id.startswith(
            "kerr-returning-radiation-refinement-"
        )
        or type(verified.document) is not dict
    ):
        raise RuntimeError(
            "required v2 checkpoint verifier returned a mismatched identity"
        )
    if verified.qualified is not True:
        raise RuntimeError("required v2 checkpoint is authenticated but non-qualified")
    return verified


def _bind_qualified_v2_checkpoint_to_kernel(
    verified: KerrReturningRadiationVerifiedRefinementCheckpoint,
    definition: KerrKernelDirectionCacheDefinition,
    reduction_configuration: KerrKernelReductionConfiguration,
) -> _QualifiedV2CheckpointBinding:
    """Bind the qualified evidence to the exact kernel this product will use."""

    if type(verified) is not KerrReturningRadiationVerifiedRefinementCheckpoint:
        raise TypeError("qualified v2 checkpoint must have its exact verified type")
    if type(definition) is not KerrKernelDirectionCacheDefinition:
        raise TypeError("product kernel definition must have its exact type")
    if type(reduction_configuration) is not KerrKernelReductionConfiguration:
        raise TypeError("product reduction configuration must have its exact type")
    document = verified.document
    if type(document) is not dict:
        raise TypeError("qualified v2 checkpoint document must be an exact dict")
    try:
        producer = document["producer"]
        cache_definition = producer["cacheDefinition"]
        reduction = producer["reductionConfiguration"]
        kernel = document["kernel"]
        if any(type(value) is not dict for value in (
            producer,
            cache_definition,
            reduction,
            kernel,
        )):
            raise TypeError("checkpoint kernel binding nodes must be exact dicts")
        scientific_job_key = cache_definition["scientificJobKey"]
        cache_job_key = cache_definition["cacheJobKey"]
        kernel_descriptor_sha256 = kernel["descriptorSha256"]
        reduction_descriptor = reduction["descriptor"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "qualified v2 checkpoint lacks its exact kernel binding"
        ) from error
    for label, digest in (
        ("scientific job key", scientific_job_key),
        ("cache job key", cache_job_key),
        ("kernel descriptor SHA-256", kernel_descriptor_sha256),
    ):
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"qualified v2 checkpoint has an invalid {label}")
    expected_reduction_bytes = canonical_json_bytes(
        reduction_configuration.as_dict()
    )
    if (
        scientific_job_key != definition.scientific_job_key
        or cache_job_key != definition.job_spec.job_key
        or _exact_json_bytes(
            reduction_descriptor,
            "qualified v2 checkpoint reduction configuration",
        )
        != expected_reduction_bytes
    ):
        raise RuntimeError(
            "qualified v2 checkpoint does not bind the product kernel definition"
        )
    manifest_payload = _exact_json_bytes(
        document,
        "qualified v2 checkpoint document",
    )
    if hashlib.sha256(manifest_payload).hexdigest() != verified.manifest_sha256:
        raise RuntimeError(
            "qualified v2 checkpoint document differs from its verified digest"
        )
    artifact = InputArtifact(
        (
            "urn:blackhole:kerr-returning-radiation-refinement-checkpoint:v1:"
            f"{verified.checkpoint_id}"
        ),
        len(manifest_payload),
        verified.manifest_sha256,
    )
    return _QualifiedV2CheckpointBinding(
        verified.checkpoint_id,
        artifact,
        kernel_descriptor_sha256,
        scientific_job_key,
        cache_job_key,
        expected_reduction_bytes,
    )


def _require_full_kernel_cache_hit(
    execution: KerrCachedReturningRadiationKernelExecution,
    definition: KerrKernelDirectionCacheDefinition,
) -> None:
    """Reject before thermal/product work unless every kernel task was reused."""

    if type(execution) is not KerrCachedReturningRadiationKernelExecution:
        raise TypeError("cached product kernel execution must have its exact type")
    if type(definition) is not KerrKernelDirectionCacheDefinition:
        raise TypeError("product kernel definition must have its exact type")
    audit = execution.execution_audit
    run = execution.job_run
    if type(audit) is not KerrCachedKernelExecutionAudit:
        raise TypeError("cached product kernel audit must have its exact type")
    if type(run) is not JobRun:
        raise TypeError("cached product kernel JobRun must have its exact type")
    if (
        type(run.results) is not tuple
        or len(run.results) != definition.plan.task_count
        or any(type(item) is not TaskResult for item in run.results)
        or tuple(item.key for item in run.results)
        != tuple(definition.job_spec.tasks)
        or any(item.reused is not True for item in run.results)
        or type(run.executed_tasks) is not int
        or run.executed_tasks != 0
        or type(run.reused_tasks) is not int
        or run.reused_tasks != definition.plan.task_count
        or type(run.max_in_flight_observed) is not int
        or run.max_in_flight_observed != 0
        or audit.executed_direction_records != 0
        or audit.executed_tasks != 0
        or audit.reused_direction_records != definition.plan.direction_count
        or audit.reused_tasks != definition.plan.task_count
        or audit.authenticated_direction_records != definition.plan.direction_count
        or audit.scientific_job_key != definition.scientific_job_key
        or audit.cache_job_key != definition.job_spec.job_key
    ):
        raise RuntimeError(
            "product rendering requires a complete pre-existing kernel cache hit"
        )


def _assert_returning_live_replay_runtime_binding() -> None:
    if (
        _run_returning_radiation_spectral_live_replay
        is not _LIVE_REPLAY_PUBLIC_ENTRY
        or publish_returning_radiation_live_replay_attestation
        is not _ATTESTATION_PUBLISH_PUBLIC_ENTRY
        or attestation_module.publish_returning_radiation_live_replay_attestation
        is not _ATTESTATION_PUBLISH_PUBLIC_ENTRY
        or attestation_module._PUBLISH_CANONICAL_ENTRY
        is not _ATTESTATION_PUBLISH_PUBLIC_ENTRY
    ):
        raise RuntimeError("returning live-replay or attestation identity changed")
    _frozen_live_replay_entry()


def _assert_returning_tile_runtime_bindings() -> None:
    if (
        BoundInputStableReturningSpectralTileProducer.__call__
        is not _BOUND_TILE_PRODUCER_CALL_ENTRY
        or
        ReturningRadiationAdaptiveSpectralTileProducer
        is not _RETURNING_TILE_PRODUCER_TYPE_ENTRY
        or returning_product_module.ReturningRadiationAdaptiveSpectralTileProducer
        is not _RETURNING_TILE_PRODUCER_TYPE_ENTRY
        or invoke_returning_radiation_spectral_tile_producer
        is not _RETURNING_TILE_INVOKER_ENTRY
        or returning_product_module.invoke_returning_radiation_spectral_tile_producer
        is not _RETURNING_TILE_INVOKER_ENTRY
        or build_returning_radiation_spectral_job_spec
        is not _RETURNING_JOB_SPEC_BUILDER_ENTRY
        or returning_product_module.build_returning_radiation_spectral_job_spec
        is not _RETURNING_JOB_SPEC_BUILDER_ENTRY
        or ReturningRadiationAdaptiveSpectralTileProducer.assert_bound_job_spec_stable
        is not _RETURNING_JOB_SPEC_STABILITY_ENTRY
        or generic_build_spectral_job_spec
        is not _GENERIC_SPECTRAL_JOB_SPEC_BUILDER_ENTRY
        or spectral_product_module.build_spectral_job_spec
        is not _GENERIC_SPECTRAL_JOB_SPEC_BUILDER_ENTRY
        or SpectralFrameGrid.tasks is not _SPECTRAL_GRID_TASKS_ENTRY
        or _returning_tile_executor
        is not _RETURNING_TILE_EXECUTOR_PUBLIC_ENTRY
        or run_job is not _RUN_JOB_PUBLIC_ENTRY
        or job_module.run_job is not _RUN_JOB_PUBLIC_ENTRY
        or publish_spectral_product is not _PUBLISH_PUBLIC_ENTRY
        or spectral_product_module.publish_spectral_product
        is not _PUBLISH_PUBLIC_ENTRY
        or validate_scientific_spectral_frame is not _VERIFIER_PUBLIC_ENTRY
        or spectral_verifier_module.validate_scientific_spectral_frame
        is not _VERIFIER_PUBLIC_ENTRY
        or default_numeric_backend_descriptor
        is not _DEFAULT_NUMERIC_BACKEND_PUBLIC_ENTRY
        or spectral_product_module.default_numeric_backend_descriptor
        is not _DEFAULT_NUMERIC_BACKEND_PUBLIC_ENTRY
        or assert_returning_radiation_spectral_runtime_bindings
        is not _RETURNING_RUNTIME_BINDING_ASSERTION_ENTRY
        or returning_product_module.assert_returning_radiation_spectral_runtime_bindings
        is not _RETURNING_RUNTIME_BINDING_ASSERTION_ENTRY
    ):
        raise RuntimeError("returning spectral product runtime binding changed")
    _RETURNING_RUNTIME_BINDING_ASSERTION_ENTRY()


def _assert_returning_renderer_runtime_bindings() -> None:
    _assert_returning_scientific_runtime_bindings()
    _assert_returning_tile_runtime_bindings()
    _assert_returning_live_replay_runtime_binding()


def _tile_control_limits(
    jobs: object,
    max_in_flight: object,
    task_count: int,
) -> tuple[int, int]:
    task_count = _exact_positive_runtime_integer(task_count, "tile task count")
    jobs = _exact_positive_runtime_integer(jobs, "tile jobs")
    max_in_flight = _exact_positive_runtime_integer(
        max_in_flight,
        "tile max_in_flight",
    )
    maximum_jobs = min(MAXIMUM_RETURNING_TILE_WORKERS, task_count)
    if jobs > maximum_jobs:
        raise ValueError(
            "tile jobs must not exceed min(64, task_count) "
            f"({maximum_jobs})"
        )
    maximum_in_flight = min(
        MAXIMUM_RETURNING_TILE_IN_FLIGHT_TASKS,
        task_count,
    )
    if max_in_flight > maximum_in_flight:
        raise ValueError(
            "tile max_in_flight must not exceed min(256, task_count) "
            f"({maximum_in_flight})"
        )
    return jobs, max_in_flight


def _kernel_control_limits(
    jobs: object,
    max_in_flight: object,
    task_count: object,
) -> None:
    task_count = _exact_positive_runtime_integer(task_count, "kernel task count")
    jobs = _exact_positive_runtime_integer(jobs, "kernel jobs")
    maximum_jobs = min(MAXIMUM_KERNEL_WORKERS, task_count)
    if jobs > maximum_jobs:
        raise ValueError(
            "kernel jobs must not exceed min(64, task_count) "
            f"({maximum_jobs})"
        )
    if max_in_flight is not None:
        max_in_flight = _exact_positive_runtime_integer(
            max_in_flight,
            "kernel max_in_flight",
        )
        maximum_in_flight = min(MAXIMUM_KERNEL_IN_FLIGHT_TASKS, task_count)
        if max_in_flight > maximum_in_flight:
            raise ValueError(
                "kernel max_in_flight must not exceed min(256, task_count) "
                f"({maximum_in_flight})"
            )


def _raw_annulus_count(arguments: argparse.Namespace) -> int:
    explicit = arguments.annulus_edges_over_mass
    if explicit is None:
        count = _exact_positive_runtime_integer(
            arguments.annulus_count,
            "annulus count",
        )
    else:
        if type(explicit) is not list or len(explicit) < 2:
            raise TypeError(
                "explicit annulus edges must be an exact list with at least two values"
            )
        if any(type(value) is not float or not math.isfinite(value) for value in explicit):
            raise TypeError("explicit annulus edges must contain finite exact floats")
        count = len(explicit) - 1
    if count > MAXIMUM_AUTHENTICATED_ANNULUS_COUNT:
        raise ValueError(
            "annulus count exceeds the authenticated frame-context maximum"
        )
    return count


def _reduction_configuration_from_arguments(
    arguments: argparse.Namespace,
) -> KerrKernelReductionConfiguration:
    policy = KerrReturningRadiationKernelPolicy(
        rho_order=arguments.kernel_rho_order,
        mu_order=arguments.kernel_mu_order,
        psi_count=arguments.kernel_psi_count,
        absolute_tolerance=arguments.kernel_absolute_tolerance,
        relative_tolerance=arguments.kernel_relative_tolerance,
        symmetry_absolute_tolerance=(
            arguments.kernel_symmetry_absolute_tolerance
        ),
        symmetry_relative_tolerance=(
            arguments.kernel_symmetry_relative_tolerance
        ),
        maximum_direction_evaluations=(
            arguments.kernel_maximum_direction_evaluations
        ),
        maximum_whole_ray_traces=arguments.kernel_maximum_whole_ray_traces,
    )
    area_policy = KerrFiniteThicknessAreaQuadraturePolicy(
        gauss_legendre_order=arguments.area_gauss_legendre_order,
        relative_tolerance=arguments.area_relative_tolerance,
        absolute_tolerance_over_mass_squared=(
            arguments.area_absolute_tolerance_over_mass_squared
        ),
        maximum_point_evaluations=arguments.area_maximum_point_evaluations,
    )
    return KerrKernelReductionConfiguration(policy, area_policy)


def _transport_plan_for_reduction_configuration(
    configuration: KerrKernelReductionConfiguration,
    annulus_count: int,
    directions_per_task: int,
) -> KerrKernelDirectionTaskPlan:
    if type(configuration) is not KerrKernelReductionConfiguration:
        raise TypeError("kernel reduction configuration must have its exact type")
    transport_plan = KerrKernelDirectionTaskPlan(
        FORWARD,
        annulus_count,
        configuration.policy.rho_order,
        configuration.policy.mu_order,
        configuration.policy.psi_count,
        directions_per_task,
    )
    if (
        transport_plan.direction_count
        > configuration.policy.maximum_direction_evaluations
    ):
        raise ValueError("kernel direction budget does not cover the transport plan")
    if (
        2 * transport_plan.direction_count
        > configuration.policy.maximum_whole_ray_traces
    ):
        raise ValueError("kernel whole-ray budget does not cover the transport plan")
    return transport_plan


def _validated_plan_reduction_configuration(
    plan: KerrReturningRadiationFramePlan,
) -> tuple[KerrKernelReductionConfiguration, KerrKernelDirectionTaskPlan]:
    stored = plan.kernel_reduction_configuration
    if type(stored) is not KerrKernelReductionConfiguration:
        raise TypeError("planned kernel reduction configuration has a foreign type")
    rebuilt = KerrKernelReductionConfiguration(
        object.__getattribute__(stored, "policy"),
        object.__getattribute__(stored, "area_policy"),
    )
    planned_bytes = plan.kernel_reduction_configuration_bytes
    if type(planned_bytes) is not bytes:
        raise TypeError("planned kernel reduction configuration bytes are not exact")
    if canonical_json_bytes(rebuilt.as_dict()) != planned_bytes:
        raise RuntimeError("planned kernel reduction configuration changed")
    current = KerrKernelReductionConfiguration(
        plan.kernel_policy,
        plan.area_policy,
    )
    if canonical_json_bytes(current.as_dict()) != planned_bytes:
        raise RuntimeError("planned kernel policy or area policy changed")
    if type(plan.annulus_edges_over_mass) is not tuple:
        raise TypeError("planned annulus edges must be an exact tuple")
    directions_per_task = _exact_positive_runtime_integer(
        plan.kernel_directions_per_task,
        "kernel directions_per_task",
    )
    if directions_per_task > MAXIMUM_KERNEL_DIRECTIONS_PER_TASK:
        raise ValueError("kernel directions_per_task exceeds the fixed maximum")
    transport_plan = _transport_plan_for_reduction_configuration(
        rebuilt,
        len(plan.annulus_edges_over_mass) - 1,
        directions_per_task,
    )
    return rebuilt, transport_plan


def _preflight_raw_render_controls(
    arguments: argparse.Namespace,
) -> tuple[int, KerrKernelReductionConfiguration]:
    """Reject unbounded plans before base planning can allocate task tuples."""

    width = _exact_positive_runtime_integer(
        arguments.width_pixels,
        "frame width",
    )
    height = _exact_positive_runtime_integer(
        arguments.height_pixels,
        "frame height",
    )
    tile_width = _exact_positive_runtime_integer(
        arguments.tile_width,
        "tile width",
    )
    tile_height = _exact_positive_runtime_integer(
        arguments.tile_height,
        "tile height",
    )
    frame_pixels = width * height
    if frame_pixels > MAXIMUM_RETURNING_FRAME_PIXELS:
        raise ValueError(
            "returning-radiation frame pixel count exceeds the fixed maximum"
        )
    task_count = (
        ((width + tile_width - 1) // tile_width)
        * ((height + tile_height - 1) // tile_height)
    )
    if task_count > MAXIMUM_RETURNING_TILE_TASKS:
        raise ValueError(
            "returning-radiation tile task count exceeds the fixed maximum"
        )
    tile_payload_bytes = (
        min(width, tile_width)
        * min(height, tile_height)
        * RETURNING_CIE_RECORD_BYTES
    )
    if tile_payload_bytes > MAXIMUM_RETURNING_TILE_PAYLOAD_BYTES:
        raise ValueError(
            "returning-radiation tile payload exceeds the fixed byte maximum"
        )
    _tile_control_limits(arguments.jobs, arguments.max_in_flight, task_count)

    kernel_jobs = _exact_positive_runtime_integer(
        arguments.kernel_jobs,
        "kernel jobs",
    )
    if kernel_jobs > MAXIMUM_KERNEL_WORKERS:
        raise ValueError("kernel jobs exceed the fixed worker maximum")
    if arguments.kernel_max_in_flight is not None:
        kernel_in_flight = _exact_positive_runtime_integer(
            arguments.kernel_max_in_flight,
            "kernel max_in_flight",
        )
        if kernel_in_flight > MAXIMUM_KERNEL_IN_FLIGHT_TASKS:
            raise ValueError("kernel max_in_flight exceeds the fixed maximum")
    directions = _exact_positive_runtime_integer(
        arguments.kernel_directions_per_task,
        "kernel directions_per_task",
    )
    if directions > MAXIMUM_KERNEL_DIRECTIONS_PER_TASK:
        raise ValueError("kernel directions_per_task exceeds the fixed maximum")
    reduction_configuration = _reduction_configuration_from_arguments(arguments)
    transport_plan = _transport_plan_for_reduction_configuration(
        reduction_configuration,
        _raw_annulus_count(arguments),
        directions,
    )
    _kernel_control_limits(
        arguments.kernel_jobs,
        arguments.kernel_max_in_flight,
        transport_plan.task_count,
    )
    return task_count, reduction_configuration


def _lexical_absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_exact_source_root(source_root: Path) -> Path:
    """Accept only the lexical repository root that supplied this renderer."""

    if type(source_root) is not type(Path()) or source_root != ROOT:
        raise ValueError("producer source_root must be the exact lexical renderer ROOT")
    return source_root


def _assert_exact_module_origin(
    module: object,
    module_name: str,
    relative: Path,
) -> None:
    if type(module) is not ModuleType:
        raise RuntimeError(
            f"producer source module {module_name} is missing or has a foreign type"
        )
    try:
        raw_origin = object.__getattribute__(module, "__file__")
    except AttributeError as error:
        raise RuntimeError(
            f"producer source module {module_name} has no file origin"
        ) from error
    if type(raw_origin) is not str:
        raise RuntimeError(
            f"producer source module {module_name} has a non-exact file origin"
        )
    origin = Path(raw_origin)
    expected = ROOT / relative
    if not origin.is_absolute() or origin != expected:
        raise RuntimeError(
            f"producer source module {module_name} was not loaded from "
            f"{relative.as_posix()} under the exact renderer ROOT"
        )


def _assert_producer_source_module_origins() -> None:
    """Bind every declared Python source to the exact loaded module origin."""

    for module_name, relative in PRODUCER_SOURCE_MODULES:
        _assert_exact_module_origin(
            sys.modules.get(module_name),
            module_name,
            relative,
        )
    # Direct CLI execution lives in ``__main__`` while lazy replay imports a
    # second canonical ``scripts.*`` module.  Bind the executing copy too so
    # checking the canonical copy cannot hide a foreign command entry point.
    _assert_exact_module_origin(
        sys.modules.get(__name__),
        __name__,
        _EXECUTING_SOURCE_RELATIVE,
    )


def _assert_source_provenance(source_root: Path) -> None:
    _require_exact_source_root(source_root)
    _assert_producer_source_module_origins()


def _reject_existing_symlink_components(path: Path, label: str) -> None:
    candidate = _lexical_absolute_path(path)
    current = Path(candidate.anchor)
    parts = candidate.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(
                f"{label} has an existing symbolic-link component: {current}"
            )
        if index + 1 < len(parts) and not stat.S_ISDIR(status.st_mode):
            raise NotADirectoryError(
                f"{label} has a non-directory path component: {current}"
            )


def _paths_equal_or_nested(first: Path, second: Path) -> bool:
    first = _lexical_absolute_path(first)
    second = _lexical_absolute_path(second)
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _preflight_product_paths(
    output: Path,
    tile_cache: Path,
    kernel_cache: Path,
    live_replay_attestation: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    output = _lexical_absolute_path(output)
    attestation = _lexical_absolute_path(
        default_live_replay_attestation_directory(output)
        if live_replay_attestation is None
        else live_replay_attestation
    )
    paths = (
        output,
        _lexical_absolute_path(tile_cache),
        _lexical_absolute_path(kernel_cache),
        attestation,
    )
    labels = (
        "output",
        "tile cache",
        "kernel cache",
        "live replay attestation",
    )
    for path, label in zip(paths, labels):
        _reject_existing_symlink_components(path, label)
    for left in range(len(paths)):
        for right in range(left + 1, len(paths)):
            if _paths_equal_or_nested(paths[left], paths[right]):
                raise ValueError(
                    f"{labels[left]} and {labels[right]} must be separate, "
                    "non-nested paths"
                )
    return paths


def collect_source_artifacts(
    source_root: Path = ROOT,
) -> tuple[InputArtifact, ...]:
    """Authenticate the fixed producer closure through anchored file fds."""

    _assert_source_provenance(source_root)
    if len(PRODUCER_SOURCE_FILES) > MAXIMUM_PRODUCER_SOURCE_ARTIFACT_COUNT:
        raise ValueError("producer source closure exceeds its fixed entry-count limit")
    if any(
        type(relative) is not type(Path())
        or relative.is_absolute()
        or not relative.parts
        or any(component in ("", ".", "..") for component in relative.parts)
        for relative in PRODUCER_SOURCE_FILES
    ):
        raise TypeError("producer source closure contains a non-canonical path")
    root = source_root
    artifacts: list[InputArtifact] = []
    total_byte_length = 0
    for relative in PRODUCER_SOURCE_FILES:
        authenticated = authenticate_stable_artifact(
            root / relative,
            f"producer source {relative.as_posix()}",
            maximum_byte_length=MAXIMUM_PRODUCER_SOURCE_ARTIFACT_BYTE_LENGTH,
            preceding_total_byte_length=total_byte_length,
            maximum_total_byte_length=(
                MAXIMUM_PRODUCER_SOURCE_ARTIFACT_TOTAL_BYTE_LENGTH
            ),
        )
        total_byte_length += authenticated.byte_length
        artifacts.append(
            InputArtifact(
                f"repo-source://{relative.as_posix()}",
                authenticated.byte_length,
                authenticated.sha256,
            )
        )
    return tuple(artifacts)


def collect_science_artifacts(
    cie_csv_path: Path,
    cie_metadata_path: Path,
) -> tuple[InputArtifact, ...]:
    """Authenticate both CIE artifacts with one fixed aggregate budget."""

    requested = (
        (CIE_CSV_INPUT_URI, _lexical_absolute_path(Path(cie_csv_path)), "CIE CSV"),
        (
            CIE_METADATA_INPUT_URI,
            _lexical_absolute_path(Path(cie_metadata_path)),
            "CIE metadata",
        ),
    )
    artifacts: list[InputArtifact] = []
    total_byte_length = 0
    for uri, path, label in requested:
        authenticated = authenticate_stable_artifact(
            path,
            label,
            maximum_byte_length=MAXIMUM_CIE_ARTIFACT_BYTE_LENGTH,
            preceding_total_byte_length=total_byte_length,
            maximum_total_byte_length=MAXIMUM_CIE_ARTIFACT_TOTAL_BYTE_LENGTH,
        )
        total_byte_length += authenticated.byte_length
        artifacts.append(
            InputArtifact(
                uri,
                authenticated.byte_length,
                authenticated.sha256,
            )
        )
    return tuple(sorted(artifacts))


def assert_bound_inputs_stable(
    expected_sources: Sequence[InputArtifact],
    expected_science: Sequence[InputArtifact],
    *,
    source_root: Path = ROOT,
    cie_csv_path: Path = DEFAULT_CIE_CSV,
    cie_metadata_path: Path = DEFAULT_CIE_METADATA,
) -> None:
    _assert_source_provenance(source_root)
    expected_source_bytes = _artifact_manifest_bytes(
        expected_sources,
        "expected source artifacts",
    )
    expected_science_bytes = _artifact_manifest_bytes(
        expected_science,
        "expected science artifacts",
    )
    if expected_source_bytes != _artifact_manifest_bytes(
        collect_source_artifacts(source_root),
        "current source artifacts",
    ):
        raise RuntimeError(
            "producer source files changed after returning-frame planning"
        )
    if expected_science_bytes != _artifact_manifest_bytes(
        collect_science_artifacts(cie_csv_path, cie_metadata_path),
        "current science artifacts",
    ):
        raise RuntimeError(
            "authenticated CIE inputs changed after returning-frame planning"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = base_renderer.build_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "cache":
            action.help = "resumable content-addressed spectral tile-cache root"
        elif action.dest == "jobs":
            action.help = (
                "in-process spectral tile worker threads; the authenticated "
                "emission authority is deliberately not process-picklable"
            )
        elif action.dest == "max_in_flight":
            action.help = "bounded in-flight spectral tile tasks"

    cache = parser.add_argument_group("returning-radiation cache execution")
    cache.add_argument(
        "--kernel-cache",
        required=True,
        type=Path,
        help=(
            "separate complete pre-existing production-forward direction-cache "
            "root; missing or invalid tasks are never traced by this renderer"
        ),
    )
    cache.add_argument(
        "--required-v2-checkpoint",
        required=True,
        type=Path,
        help=(
            "absolute qualified source-current refinement manifest required "
            "before kernel cache access"
        ),
    )
    cache.add_argument(
        "--required-v2-checkpoint-sha256",
        required=True,
        help="externally retained expected SHA-256 for the required checkpoint",
    )
    cache.add_argument("--kernel-jobs", type=_positive_integer, default=1)
    cache.add_argument(
        "--kernel-max-in-flight",
        type=_positive_integer,
        default=None,
    )
    cache.add_argument(
        "--kernel-directions-per-task",
        type=_positive_integer,
        default=64,
    )

    annuli = parser.add_argument_group("piecewise returning-radiation annuli")
    annulus_choice = annuli.add_mutually_exclusive_group()
    annulus_choice.add_argument(
        "--annulus-count",
        type=_positive_integer,
        default=1,
        help="uniform rho/M cells spanning the exact ISCO through R_out",
    )
    annulus_choice.add_argument(
        "--annulus-edges-over-mass",
        type=float,
        nargs="+",
        default=None,
        metavar="RHO_OVER_M",
        help="explicit complete edge list from exact ISCO through R_out",
    )
    annuli.add_argument(
        "--annulus-edge-clearance-multiplier",
        type=float,
        default=4.0,
    )

    kernel = parser.add_argument_group("finite-grid forward kernel policy")
    kernel.add_argument("--kernel-rho-order", type=_positive_integer, default=8)
    kernel.add_argument("--kernel-mu-order", type=_positive_integer, default=8)
    kernel.add_argument("--kernel-psi-count", type=_positive_integer, default=8)
    kernel.add_argument("--kernel-absolute-tolerance", type=float, default=2.0e-2)
    kernel.add_argument("--kernel-relative-tolerance", type=float, default=5.0e-2)
    kernel.add_argument(
        "--kernel-symmetry-absolute-tolerance",
        type=float,
        default=2.0e-8,
    )
    kernel.add_argument(
        "--kernel-symmetry-relative-tolerance",
        type=float,
        default=2.0e-7,
    )
    kernel.add_argument(
        "--kernel-maximum-direction-evaluations",
        type=_positive_integer,
        default=2_000_000,
    )
    kernel.add_argument(
        "--kernel-maximum-whole-ray-traces",
        type=_positive_integer,
        default=8_000_000,
    )

    area = parser.add_argument_group("finite-height comoving annulus area")
    area.add_argument(
        "--area-gauss-legendre-order",
        type=_positive_integer,
        default=24,
    )
    area.add_argument("--area-relative-tolerance", type=float, default=2.0e-10)
    area.add_argument(
        "--area-absolute-tolerance-over-mass-squared",
        type=float,
        default=2.0e-11,
    )
    area.add_argument(
        "--area-maximum-point-evaluations",
        type=_positive_integer,
        default=384,
    )

    fixed = parser.add_argument_group("absorbed returning-radiation fixed point")
    fixed.add_argument(
        "--fixed-point-maximum-iterations",
        type=_positive_integer,
        default=10_000,
    )
    fixed.add_argument(
        "--fixed-point-absolute-residual-tolerance",
        type=float,
        default=0.0,
    )
    fixed.add_argument(
        "--fixed-point-relative-residual-tolerance",
        type=float,
        default=1.0e-12,
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _annulus_edges(
    arguments: argparse.Namespace,
    sampler: KerrFiniteThicknessRaySampler,
) -> tuple[float, ...]:
    inner = float(sampler.surface.calibration.isco_radius_over_mass)
    outer = float(sampler.surface.calibration.outer_radius_over_mass)
    explicit = arguments.annulus_edges_over_mass
    if explicit is not None:
        edges = tuple(float(value) for value in explicit)
        if len(edges) < 2:
            raise ValueError("explicit annulus edges need at least two values")
    else:
        count = arguments.annulus_count
        if count > MAXIMUM_AUTHENTICATED_ANNULUS_COUNT:
            raise ValueError(
                "annulus count exceeds the authenticated frame-context maximum"
            )
        span = outer - inner
        edges = (
            inner,
            *(
                math.fsum((inner, span * index / count))
                for index in range(1, count)
            ),
            outer,
        )
    if len(edges) - 1 > MAXIMUM_AUTHENTICATED_ANNULUS_COUNT:
        raise ValueError(
            "annulus edges exceed the authenticated frame-context maximum"
        )
    if any(not math.isfinite(value) for value in edges):
        raise ValueError("annulus edges must be finite")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("annulus edges must increase strictly")
    if edges[0].hex() != inner.hex() or edges[-1].hex() != outer.hex():
        raise ValueError("annulus edges must exactly cover ISCO through R_out")
    return edges


def _kernel_cache_definition(
    plan_or_sampler: KerrReturningRadiationFramePlan | KerrFiniteThicknessRaySampler,
    *,
    annulus_edges_over_mass: tuple[float, ...] | None = None,
    kernel_policy: KerrReturningRadiationKernelPolicy | None = None,
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy | None = None,
    directions_per_task: int | None = None,
) -> KerrKernelDirectionCacheDefinition:
    if type(plan_or_sampler) is KerrReturningRadiationFramePlan:
        plan = plan_or_sampler
        sampler = plan.base_sampler
        edges = plan.annulus_edges_over_mass
        selected_kernel_policy = plan.kernel_policy
        selected_area_policy = plan.area_policy
        selected_directions = plan.kernel_directions_per_task
    else:
        sampler = plan_or_sampler
        if type(sampler) is not KerrFiniteThicknessRaySampler:
            raise TypeError("base sampler must have its exact certified type")
        if (
            annulus_edges_over_mass is None
            or kernel_policy is None
            or area_policy is None
            or directions_per_task is None
        ):
            raise TypeError("initial kernel definition needs every bound input")
        edges = annulus_edges_over_mass
        selected_kernel_policy = kernel_policy
        selected_area_policy = area_policy
        selected_directions = directions_per_task
    return build_forward_kerr_returning_radiation_kernel_cache_definition(
        sampler.surface,
        termination=sampler.termination,
        annulus_edges_over_mass=edges,
        ray_options=sampler.fine_options,
        surface_options=sampler.surface_options,
        policy=selected_kernel_policy,
        area_policy=selected_area_policy,
        directions_per_task=selected_directions,
    )


def build_render_plan(
    arguments: argparse.Namespace,
    *,
    source_root: Path = ROOT,
) -> KerrReturningRadiationFramePlan:
    """Build all trace-free identities without creating a cache or output."""

    _require_exact_source_root(source_root)
    raw_task_count, reduction_configuration = _preflight_raw_render_controls(
        arguments
    )
    _assert_returning_renderer_runtime_bindings()
    _assert_producer_source_module_origins()
    checkpoint_manifest = _lexical_absolute_path(
        Path(arguments.required_v2_checkpoint)
    )
    checkpoint_sha256 = arguments.required_v2_checkpoint_sha256
    verified_checkpoint = _require_qualified_v2_checkpoint(
        checkpoint_manifest,
        checkpoint_sha256,
    )
    numeric_backend_before = _current_numeric_backend_bytes()
    output, tile_cache, kernel_cache, live_replay_attestation = _preflight_product_paths(
        Path(arguments.output),
        Path(arguments.cache),
        Path(arguments.kernel_cache),
    )
    if live_replay_attestation.exists() or live_replay_attestation.is_symlink():
        raise FileExistsError(
            "refusing to overwrite existing live replay attestation "
            f"{live_replay_attestation}"
        )
    cie_csv = _lexical_absolute_path(Path(arguments.cie_csv))
    cie_metadata = _lexical_absolute_path(Path(arguments.cie_metadata))
    source_artifacts = collect_source_artifacts(source_root)
    science_artifacts = collect_science_artifacts(cie_csv, cie_metadata)

    base = base_renderer.build_render_plan(arguments, source_root=source_root)
    numeric_backend_after = _current_numeric_backend_bytes()
    base_numeric_backend_bytes = _exact_json_bytes(
        base.numeric_backend,
        "base numeric backend",
    )
    if (
        numeric_backend_before != numeric_backend_after
        or base_numeric_backend_bytes != numeric_backend_before
    ):
        raise RuntimeError("numeric backend changed during base planning")
    if _artifact_manifest_bytes(
        source_artifacts,
        "source artifacts before base planning",
    ) != _artifact_manifest_bytes(
        collect_source_artifacts(source_root),
        "source artifacts after base planning",
    ):
        raise RuntimeError("producer source files changed during base planning")
    if _artifact_manifest_bytes(
        science_artifacts,
        "science artifacts before base planning",
    ) != _artifact_manifest_bytes(
        collect_science_artifacts(cie_csv, cie_metadata),
        "science artifacts after base planning",
    ):
        raise RuntimeError("authenticated CIE inputs changed during base planning")
    if base.layout.frequency_count != CIE_ROW_COUNT:
        raise AssertionError("authenticated CIE grid does not contain 471 bins")
    if base.layout.record_bytes != RETURNING_CIE_RECORD_BYTES:
        raise AssertionError("authenticated CIE pixel record size changed")
    _frame_pixels, task_count, _tile_payload_bytes = (
        validate_returning_radiation_spectral_resources(
            base.layout,
            base.grid,
            tile_width=arguments.tile_width,
            tile_height=arguments.tile_height,
        )
    )
    if task_count != raw_task_count:
        raise AssertionError("base frame task count differs from preflight")
    _tile_control_limits(arguments.jobs, arguments.max_in_flight, task_count)
    edges = _annulus_edges(arguments, base.sampler)
    kernel_policy = reduction_configuration.policy
    area_policy = reduction_configuration.area_policy
    fixed_point_policy = ReturningRadiationFixedPointPolicy(
        maximum_iterations=arguments.fixed_point_maximum_iterations,
        absolute_residual_tolerance=(
            arguments.fixed_point_absolute_residual_tolerance
        ),
        relative_residual_tolerance=(
            arguments.fixed_point_relative_residual_tolerance
        ),
    )
    # Constructor validation fixes the policy before expensive work begins.
    edge_multiplier = float(arguments.annulus_edge_clearance_multiplier)
    if (
        not math.isfinite(edge_multiplier)
        or edge_multiplier < 1.0
        or edge_multiplier > MAXIMUM_ANNULUS_EDGE_CLEARANCE_MULTIPLIER
    ):
        raise ValueError(
            "annulus edge-clearance multiplier must lie in the certified "
            f"range [1, {MAXIMUM_ANNULUS_EDGE_CLEARANCE_MULTIPLIER}]"
        )

    definition = _kernel_cache_definition(
        base.sampler,
        annulus_edges_over_mass=edges,
        kernel_policy=kernel_policy,
        area_policy=area_policy,
        directions_per_task=arguments.kernel_directions_per_task,
    )
    checkpoint_binding = _bind_qualified_v2_checkpoint_to_kernel(
        verified_checkpoint,
        definition,
        reduction_configuration,
    )
    _kernel_control_limits(
        arguments.kernel_jobs,
        arguments.kernel_max_in_flight,
        definition.plan.task_count,
    )
    base_sampler_descriptor = json.loads(
        canonical_json_bytes(base.sampler.descriptor())
    )
    return KerrReturningRadiationFramePlan(
        output_directory=output,
        live_replay_attestation_directory=live_replay_attestation,
        tile_cache_root=tile_cache,
        kernel_cache_root=kernel_cache,
        required_v2_checkpoint_manifest=verified_checkpoint.manifest_path,
        required_v2_checkpoint_manifest_sha256=(
            verified_checkpoint.manifest_sha256
        ),
        required_v2_checkpoint_binding=checkpoint_binding,
        tile_jobs=arguments.jobs,
        tile_max_in_flight=arguments.max_in_flight,
        kernel_jobs=arguments.kernel_jobs,
        kernel_max_in_flight=arguments.kernel_max_in_flight,
        kernel_directions_per_task=arguments.kernel_directions_per_task,
        tile_width=arguments.tile_width,
        tile_height=arguments.tile_height,
        layout=base.layout,
        grid=base.grid,
        adaptive_options=base.adaptive_options,
        base_sampler=base.sampler,
        base_sampler_descriptor=base_sampler_descriptor,
        annulus_edges_over_mass=edges,
        kernel_policy=kernel_policy,
        area_policy=area_policy,
        kernel_reduction_configuration=reduction_configuration,
        kernel_reduction_configuration_bytes=canonical_json_bytes(
            reduction_configuration.as_dict()
        ),
        fixed_point_policy=fixed_point_policy,
        annulus_edge_clearance_multiplier=edge_multiplier,
        numeric_backend=base.numeric_backend,
        source_artifacts=source_artifacts,
        science_artifacts=science_artifacts,
        source_root=_lexical_absolute_path(Path(source_root)),
        cie_csv_path=cie_csv,
        cie_metadata_path=cie_metadata,
        kernel_cache_definition=definition,
        verification_schema=DEFAULT_SPECTRAL_SCHEMA.absolute(),
    )


def _assert_plan_inputs_stable(plan: KerrReturningRadiationFramePlan) -> None:
    assert_bound_inputs_stable(
        plan.source_artifacts,
        plan.science_artifacts,
        source_root=plan.source_root,
        cie_csv_path=plan.cie_csv_path,
        cie_metadata_path=plan.cie_metadata_path,
    )


def execute_render_plan(
    plan: KerrReturningRadiationFramePlan,
) -> KerrReturningRadiationFrameExecution:
    """Reduce a complete cached K, render, publish, verify, and live-replay."""

    if type(plan) is not KerrReturningRadiationFramePlan:
        raise TypeError("plan must have its exact returning-frame type")
    verified_checkpoint = _require_qualified_v2_checkpoint(
        plan.required_v2_checkpoint_manifest,
        plan.required_v2_checkpoint_manifest_sha256,
    )
    planned_reduction, planned_transport = (
        _validated_plan_reduction_configuration(plan)
    )
    _kernel_control_limits(
        plan.kernel_jobs,
        plan.kernel_max_in_flight,
        planned_transport.task_count,
    )
    _assert_source_provenance(plan.source_root)
    _frame_pixels, tile_task_count, _tile_payload_bytes = (
        validate_returning_radiation_spectral_resources(
            plan.layout,
            plan.grid,
            tile_width=plan.tile_width,
            tile_height=plan.tile_height,
        )
    )
    if (
        plan.layout.frequency_count != CIE_ROW_COUNT
        or plan.layout.record_bytes != RETURNING_CIE_RECORD_BYTES
    ):
        raise RuntimeError("planned authenticated CIE pixel layout changed")
    _tile_control_limits(
        plan.tile_jobs,
        plan.tile_max_in_flight,
        tile_task_count,
    )
    _assert_returning_renderer_runtime_bindings()
    planned_numeric_backend_bytes = _exact_json_bytes(
        plan.numeric_backend,
        "planned numeric backend",
    )
    if _current_numeric_backend_bytes() != planned_numeric_backend_bytes:
        raise RuntimeError("planned numeric backend differs from current runtime")
    output, tile_cache, kernel_cache, live_replay_attestation = _preflight_product_paths(
        plan.output_directory,
        plan.tile_cache_root,
        plan.kernel_cache_root,
        plan.live_replay_attestation_directory,
    )
    if (
        output != plan.output_directory
        or tile_cache != plan.tile_cache_root
        or kernel_cache != plan.kernel_cache_root
        or live_replay_attestation != plan.live_replay_attestation_directory
    ):
        raise ValueError(
            "planned output, attestation, and cache paths must be lexical absolutes"
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output {output}")
    if live_replay_attestation.exists() or live_replay_attestation.is_symlink():
        raise FileExistsError(
            "refusing to overwrite existing live replay attestation "
            f"{live_replay_attestation}"
        )

    _assert_plan_inputs_stable(plan)
    if canonical_json_bytes(plan.base_sampler.descriptor()) != canonical_json_bytes(
        plan.base_sampler_descriptor
    ):
        raise RuntimeError("base finite-thickness sampler changed after planning")
    current_definition = _kernel_cache_definition(plan)
    if current_definition != plan.kernel_cache_definition:
        raise RuntimeError("production forward kernel definition changed after planning")
    if current_definition.plan != planned_transport:
        raise RuntimeError("planned transport topology changed after planning")
    checkpoint_binding = _bind_qualified_v2_checkpoint_to_kernel(
        verified_checkpoint,
        current_definition,
        planned_reduction,
    )
    if checkpoint_binding != plan.required_v2_checkpoint_binding:
        raise RuntimeError(
            "required v2 checkpoint binding changed after renderer planning"
        )
    _assert_returning_scientific_runtime_bindings()
    cached_kernel = _CACHED_KERNEL_INTEGRATOR_CALL_ENTRY(
        plan.base_sampler.surface,
        termination=plan.base_sampler.termination,
        annulus_edges_over_mass=plan.annulus_edges_over_mass,
        cache_root=plan.kernel_cache_root,
        ray_options=plan.base_sampler.fine_options,
        surface_options=plan.base_sampler.surface_options,
        policy=plan.kernel_policy,
        area_policy=plan.area_policy,
        directions_per_task=plan.kernel_directions_per_task,
    )
    if type(cached_kernel) is not KerrCachedReturningRadiationKernelExecution:
        raise TypeError("production cached forward execution has a foreign type")
    if (
        type(cached_kernel.reduction_configuration)
        is not KerrKernelReductionConfiguration
        or canonical_json_bytes(cached_kernel.reduction_configuration.as_dict())
        != plan.kernel_reduction_configuration_bytes
        or canonical_json_bytes(planned_reduction.as_dict())
        != plan.kernel_reduction_configuration_bytes
    ):
        raise RuntimeError(
            "cached kernel reduction configuration differs from the planned one"
        )
    if (
        cached_kernel.cache_definition != plan.kernel_cache_definition
        or cached_kernel.cache_definition.scientific_job_key
        != plan.kernel_cache_definition.scientific_job_key
    ):
        raise RuntimeError("cached kernel execution differs from the planned definition")
    if (
        type(cached_kernel.kernel.model_descriptor_sha256) is not str
        or cached_kernel.kernel.model_descriptor_sha256
        != checkpoint_binding.kernel_descriptor_sha256
    ):
        raise RuntimeError(
            "cached product kernel differs from the qualified v2 checkpoint"
        )
    _require_full_kernel_cache_hit(cached_kernel, current_definition)
    _assert_plan_inputs_stable(plan)

    _assert_returning_scientific_runtime_bindings()
    profile = _THERMAL_PROFILE_SOLVER_CALL_ENTRY(
        cached_kernel,
        disk=plan.base_sampler.disk,
        policy=plan.fixed_point_policy,
    )
    _assert_returning_scientific_runtime_bindings()
    provider = _THERMAL_PROVIDER_BUILDER_CALL_ENTRY(
        profile
    )
    _assert_returning_scientific_runtime_bindings()
    authority = _THERMAL_AUTHENTICATOR_CALL_ENTRY(
        plan.base_sampler.surface,
        provider,
    )
    _assert_returning_scientific_runtime_bindings()
    sampler = _RETURNING_SAMPLER_CONSTRUCTOR_CALL_ENTRY(
        plan.base_sampler,
        authority,
        annulus_edge_clearance_multiplier=(
            plan.annulus_edge_clearance_multiplier
        ),
    )
    if type(sampler) is not _RETURNING_SAMPLER_TYPE_ENTRY:
        raise TypeError("returning sampler constructor returned a foreign type")
    _assert_plan_inputs_stable(plan)

    source_hashes = tuple(
        sorted({artifact.sha256 for artifact in plan.source_artifacts})
    )
    inputs = tuple(
        sorted(
            (
                *plan.source_artifacts,
                *plan.science_artifacts,
                checkpoint_binding.input_artifact,
            )
        )
    )
    _assert_returning_tile_runtime_bindings()
    job_spec = _RETURNING_JOB_SPEC_BUILDER_ENTRY(
        sampler,
        plan.layout,
        plan.grid,
        plan.adaptive_options,
        tile_width=plan.tile_width,
        tile_height=plan.tile_height,
        numeric_backend=plan.numeric_backend,
        inputs=inputs,
        producer_source_hashes=source_hashes,
    )
    inner = ReturningRadiationAdaptiveSpectralTileProducer(
        sampler,
        plan.layout,
        plan.grid,
        plan.adaptive_options,
        plan.numeric_backend,
        job_spec,
    )
    producer = BoundInputStableReturningSpectralTileProducer(
        inner=inner,
        source_artifacts=plan.source_artifacts,
        science_artifacts=plan.science_artifacts,
        source_root=plan.source_root,
        cie_csv_path=plan.cie_csv_path,
        cie_metadata_path=plan.cie_metadata_path,
    )
    frozen_producer = partial(_BOUND_TILE_PRODUCER_CALL_ENTRY, producer)
    _assert_returning_tile_runtime_bindings()
    _RETURNING_JOB_SPEC_STABILITY_ENTRY(
        inner,
        job_spec,
        full_document=True,
    )
    authority.require_live()
    try:
        job_run = _RUN_JOB_CALL_ENTRY(
            job_spec,
            frozen_producer,
            plan.tile_cache_root,
            jobs=plan.tile_jobs,
            max_in_flight=plan.tile_max_in_flight,
            executor_factory=_RETURNING_TILE_EXECUTOR_CALL_ENTRY,
        )
    finally:
        # A full cache hit never calls a tile producer, so the reviewed runtime
        # bindings must also be gated at the job boundary itself.
        _assert_returning_tile_runtime_bindings()
        _assert_source_provenance(plan.source_root)
        _RETURNING_JOB_SPEC_STABILITY_ENTRY(
            inner,
            job_spec,
            full_document=True,
        )
        if _current_numeric_backend_bytes() != planned_numeric_backend_bytes:
            raise RuntimeError("numeric backend changed during tile execution")
    authority.require_live()
    _assert_plan_inputs_stable(plan)

    publication = _PUBLISH_CALL_ENTRY(
        output,
        job_spec=job_spec,
        job_run=job_run,
        layout=plan.layout,
        grid=plan.grid,
        options=plan.adaptive_options,
        sampler_descriptor=sampler.descriptor(),
        numeric_backend=plan.numeric_backend,
    )
    _assert_plan_inputs_stable(plan)
    verification = _VERIFIER_CALL_ENTRY(
        publication.manifest_path,
        plan.verification_schema,
    )
    if verification.get("status") != (
        "scientific-spectral-frame-structural-contract-conformant"
    ):
        raise RuntimeError("strict spectral-frame verifier returned a bad status")
    _assert_plan_inputs_stable(plan)
    _assert_returning_live_replay_runtime_binding()
    live_replay_attestation = _ATTESTATION_PUBLISH_CALL_ENTRY(
        plan.live_replay_attestation_directory,
        spectral_publication=publication,
        sampler=sampler,
        spectral_schema_path=plan.verification_schema,
    )
    if type(live_replay_attestation) is not (
        ReturningRadiationLiveReplayAttestationPublication
    ):
        raise TypeError("live replay attestation publisher returned a foreign type")
    live_replay_report = live_replay_attestation.live_replay_report
    _assert_returning_live_replay_runtime_binding()
    _assert_plan_inputs_stable(plan)
    return KerrReturningRadiationFrameExecution(
        cached_kernel=cached_kernel,
        thermal_profile=profile,
        thermal_provider=provider,
        authority=authority,
        sampler=sampler,
        job_spec=job_spec,
        job_run=job_run,
        publication=publication,
        verification=verification,
        live_replay_attestation=live_replay_attestation,
        live_replay_report=live_replay_report,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        plan = build_render_plan(arguments)
        execution = execute_render_plan(plan)
    except Exception as error:
        print(
            f"Offline returning-radiation exact-Kerr frame failed: {error}",
            file=sys.stderr,
        )
        return 1

    publication = execution.publication
    run = execution.job_run
    kernel_audit = execution.cached_kernel.execution_audit
    print("Offline returning-radiation exact-Kerr CIE-grid frame completed")
    print(f"  manifest = {publication.manifest_path}")
    print(f"  manifest sha256 = {publication.manifest_sha256}")
    print(f"  product id = {publication.product_id}")
    print(f"  tile job key = {run.job_key}")
    print(
        f"  kernel scientific job key = "
        f"{execution.cached_kernel.cache_definition.scientific_job_key}"
    )
    print(f"  strict structural verifier = {execution.verification['status']}")
    print(
        "  live replay attestation = "
        f"{execution.live_replay_attestation.manifest_path}"
    )
    print(
        "  live replay attestation sha256 = "
        f"{execution.live_replay_attestation.manifest_sha256}"
    )
    live_report = execution.live_replay_report
    print(f"  live sampler replay = {live_report['status']}")
    print(
        "  live replay evidence = "
        f"pixelBytesExact={live_report['pixelBytesExact']}; "
        f"frameGeodesicsReplayed={live_report['frameGeodesicsReplayed']}; "
        f"totalFrameGeodesics={live_report['totalFrameGeodesicsReplayed']}"
    )
    print(
        "  live replay exclusions = "
        f"thermalFixedPointReplayed={live_report['thermalFixedPointReplayed']}; "
        f"directionCacheRecordsReplayed="
        f"{live_report['directionCacheRecordsReplayed']}; "
        f"directionRaysRetraced={live_report['directionRaysRetraced']}; "
        f"independentPhysicsOracle={live_report['independentPhysicsOracle']}"
    )
    print(
        f"  frequencies = {plan.layout.frequency_count} authenticated CIE bins; "
        f"annuli = {len(plan.annulus_edges_over_mass) - 1}; "
        f"records = {publication.record_count}; tiles = {publication.tile_count}"
    )
    print(
        f"  tile cache reused/executed = {run.reused_tasks}/{run.executed_tasks}; "
        f"kernel cache reused/executed directions = "
        f"{kernel_audit.reused_direction_records}/"
        f"{kernel_audit.executed_direction_records}"
    )
    print(
        "  scope = same-code finite-grid piecewise-annulus returning thermal "
        "I_nu; no independent oracle, continuum radial solution, complete "
        "KERRBB, F_S, solved/scattering atmosphere, polarization, or GRMHD"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
