"""Expanded, source-anchored Kerr phase-space golden qualification.

The narrow cache corpus in :mod:`tools.native.golden_cache` proves the exact
cached direction transport.  This module adds the implementation evidence a
native backend needs for differential qualification: every accepted fine and
coarse segment, terminal state, ordered surface crossing, work counter,
null/metric diagnostic, receiver quantity, and whole-ray convergence value.

Generation is intentionally a separate trust domain from comparison.  The
``generate`` command lazily imports the unchanged production Python backend,
rebuilds its exact source/runtime-bound cache definition, and traces eight
directions.  The ``verify`` and ``compare`` commands import no ``offline``
module and validate canonical JSON plus all embedded hashes themselves.

The original cache is opened only through the read-only milestone-1
authenticator.  No function in this module accepts a cache root as an output
and no generation path invokes a cache runner.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, Final, Mapping, Sequence, TYPE_CHECKING

if __package__ in (None, ""):
    _BOOTSTRAP_ROOT = Path(__file__).absolute().parents[2]
    if str(_BOOTSTRAP_ROOT) not in sys.path:
        sys.path.insert(0, str(_BOOTSTRAP_ROOT))
    from tools.native import golden_cache as gc
else:
    from . import golden_cache as gc

if TYPE_CHECKING:
    from offline.kerr_returning_radiation_kernel_jobs import (
        KerrKernelDirectionCacheDefinition,
        KerrKernelScientificContext,
    )


GOLDEN_SCHEMA: Final = "blackhole.native-kerr-phase-space-golden/v1"
CANDIDATE_SCHEMA: Final = "blackhole.native-kerr-phase-space-candidate/v1"
COMPARISON_SCHEMA: Final = "blackhole.native-kerr-phase-space-comparison/v1"
MANIFEST_SCHEMA: Final = "blackhole.native-kerr-phase-space-manifest/v1"
VERSION: Final = 1

CACHED_SELECTED_ORDINALS: Final = (369, 433, 726, 732, 736, 31_263, 37_331)
CRITICAL_NONCACHED_ORDINALS: Final = (75_785,)
REFERENCE_ORDINALS: Final = CACHED_SELECTED_ORDINALS + CRITICAL_NONCACHED_ORDINALS

REFERENCE_FILENAME: Final = "nested16_phase_space_golden.json"
# Filled after the checked-in reference is generated and independently
# replayed.  All-zero is deliberately rejected by checked-in loading.
FROZEN_REFERENCE_SHA256: Final = (
    "8ff7391df7a39992ba2733a61aec9c63460fa1ff33ad458f3e70dae8b137cc67"
)

_MAXIMUM_DOCUMENT_BYTES: Final = 128 * 1024 * 1024
_MAXIMUM_MISMATCHES: Final = 512
_SHA256_LENGTH: Final = 64


class PhaseSpaceGoldenError(RuntimeError):
    """Raised when a reference, candidate, or generation anchor fails closed."""


@dataclass(frozen=True, slots=True)
class Nested16NumericReplayContext:
    """Typed nested16 work with no frozen source/job-identity claim.

    This context exists only for same-configuration numerical differential
    replay after the production source closure has moved.  Its temporary job
    keys are intentionally absent from :meth:`descriptor` and must never be
    presented as the frozen d828/19c8 identities.
    """

    definition: KerrKernelDirectionCacheDefinition

    def __post_init__(self) -> None:
        from offline.kerr_returning_radiation_kernel_jobs import (
            KerrKernelDirectionCacheDefinition,
        )

        if type(self.definition) is not KerrKernelDirectionCacheDefinition:
            raise TypeError(
                "numeric replay definition must use the exact cache-definition type"
            )
        if (
            self.definition.plan.scientific_plan_sha256
            != gc.FROZEN_NESTED16_TRUST.scientific_plan_sha256
        ):
            raise PhaseSpaceGoldenError(
                "numeric replay scientific coordinate plan differs from nested16"
            )

    @property
    def scientific_context(self) -> KerrKernelScientificContext:
        return self.definition.scientific_context

    @property
    def plan(self) -> Any:
        return self.definition.plan

    def descriptor(self) -> dict[str, Any]:
        return {
            "classification": "non-production-nested16-numeric-replay-context",
            "directionCount": self.definition.plan.direction_count,
            "frozenCacheJobIdentityClaimed": False,
            "frozenScientificJobIdentityClaimed": False,
            "implementationId": "nested16-typed-numeric-replay-context/v1",
            "productionQualified": False,
            "scientificPlanSha256": (
                self.definition.plan.scientific_plan_sha256
            ),
            "temporaryJobKeysReported": False,
        }


def _canonical_bytes(value: Any) -> bytes:
    return gc.canonical_json_bytes(value)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _descriptor_bytes(value: Any) -> bytes:
    """Match production descriptor JSON: canonical, ASCII, no final newline."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PhaseSpaceGoldenError("descriptor cannot be canonicalized") from error


def _descriptor_sha256(value: Any) -> str:
    return hashlib.sha256(_descriptor_bytes(value)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    try:
        return gc._require_sha256(value, label)
    except gc.GoldenCacheError as error:
        raise PhaseSpaceGoldenError(str(error)) from error


def _require_exact_int(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PhaseSpaceGoldenError(f"{label} must be an exact int >= {minimum}")
    return value


def _require_finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise PhaseSpaceGoldenError(f"{label} must be an exact finite float")
    return value


def _require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise PhaseSpaceGoldenError(f"{label} has a non-exact schema")
    return value


def _require_finite_tree(value: Any, label: str) -> None:
    if type(value) in (str, int, bool, type(None)):
        return
    if type(value) is float:
        _require_finite_float(value, label)
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{label}[{index}]")
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise PhaseSpaceGoldenError(f"{label} contains a non-string key")
        for key, item in value.items():
            _require_finite_tree(item, f"{label}.{key}")
        return
    raise PhaseSpaceGoldenError(f"{label} contains unsupported type {type(value).__name__}")


def _read_document(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    if type(path) is not type(Path()) or not path.is_absolute():
        raise PhaseSpaceGoldenError(f"{label} path must be an exact absolute Path")
    try:
        parent, parent_fd = gc._open_absolute_directory(path.parent, f"{label} parent")
        del parent
        try:
            payload = gc._read_regular_file_at(
                parent_fd,
                path.name,
                maximum_bytes=_MAXIMUM_DOCUMENT_BYTES,
                label=label,
            )
        finally:
            os.close(parent_fd)
        document = gc.strict_json_bytes(payload, label, _MAXIMUM_DOCUMENT_BYTES)
    except gc.GoldenCacheError as error:
        raise PhaseSpaceGoldenError(str(error)) from error
    if type(document) is not dict:
        raise PhaseSpaceGoldenError(f"{label} must be a JSON object")
    return payload, document


def _state_document(state: Any) -> dict[str, Any]:
    return {
        "covector": list(state.covector),
        "event": list(state.event),
    }


def _segment_document(segment: Any) -> dict[str, Any]:
    return {
        "affineLength": segment.affine_length,
        "end": _state_document(segment.end),
        "midpoint": _state_document(segment.midpoint),
        "midpointNullResidual": segment.midpoint_null_residual,
        "start": _state_document(segment.start),
    }


def _surface_trace_document(trace: Any) -> dict[str, Any] | None:
    if trace is None:
        return None
    initial = trace.initial_contact
    return {
        "baseSubdivisionsPerStep": trace.base_subdivisions_per_step,
        "crossings": [
            {
                "bracketAffineWidth": entry.crossing.bracket_affine_width,
                "decision": {
                    "classification": entry.decision.classification,
                    "outcome": entry.decision.outcome,
                    "targetId": entry.decision.target_id,
                },
                "iterations": entry.crossing.iterations,
                "orientation": entry.crossing.orientation,
                "rayAffineLength": entry.crossing.ray_affine_length,
                "segmentAffineLength": entry.crossing.segment_affine_length,
                "segmentIndex": entry.crossing.segment_index,
                "state": _state_document(entry.crossing.state),
                "surfaceId": entry.surface_id,
                "surfaceValue": entry.crossing.surface_value,
            }
            for entry in trace.crossings
        ],
        "initialContact": None
        if initial is None
        else {
            "actualSurfaceValue": initial.actual_surface_value,
            "side": initial.side,
            "surfaceId": initial.surface_id,
            "surfaceValueTolerance": initial.surface_value_tolerance,
        },
        "maximumProbeCovectorRelativeDifference": (
            trace.maximum_probe_covector_relative_difference
        ),
        "maximumProbeEventDifference": trace.maximum_probe_event_difference,
        "probeReintegrations": trace.probe_reintegrations,
        "surfaceIds": list(trace.surface_ids),
        "surfaceValueEvaluations": trace.surface_value_evaluations,
        "topologyConverged": trace.topology_converged,
        "verificationSubdivisionsPerStep": trace.verification_subdivisions_per_step,
    }


def _ray_document(ray: Any) -> dict[str, Any]:
    return {
        "acceptedSteps": ray.accepted_steps,
        "affineLength": ray.affine_length,
        "failureReason": ray.failure_reason,
        "maximumMetricInterpolationError": ray.maximum_metric_interpolation_error,
        "maximumNullResidual": ray.maximum_null_residual,
        "outcome": ray.outcome,
        "rejectedSteps": ray.rejected_steps,
        "segments": [_segment_document(item) for item in ray.segments],
        "surfaceTrace": _surface_trace_document(ray.multi_surface_trace),
        "terminalState": _state_document(ray.terminal_state),
        "terminalTargetId": ray.terminal_target_id,
    }


def _receiver_document(primitive: Any) -> dict[str, Any] | None:
    receiver = primitive.receiver
    if receiver is None:
        return None
    descriptor = receiver.model_descriptor()
    return {
        "bolometricG4Factor": primitive.bolometric_g4_factor,
        "descriptor": descriptor,
        "descriptorSha256": receiver.model_descriptor_sha256,
        "emitterToReceiverFrequencyRatio": (
            primitive.emitter_to_receiver_frequency_ratio
        ),
        "eventKs": list(receiver.event),
        "face": primitive.receiver_face,
        "incidenceCosine": primitive.receiver_incidence_cosine,
        "incidenceWeightedG4": primitive.receiver_incidence_weighted_g4,
        "radiusOverMass": primitive.receiver_radius_over_mass,
        "surfaceId": primitive.receiver_surface_id,
    }


def _phase_space_document(primitive: Any) -> dict[str, Any]:
    return {
        "coarse": _ray_document(primitive.coarse_ray),
        "convergence": asdict(primitive.convergence),
        "fate": primitive.fate,
        "fine": _ray_document(primitive.ray),
        "receiver": _receiver_document(primitive),
    }


def _coordinate_for_ordinal(plan: Any, ordinal: int) -> Any:
    """Decode one canonical ordinal without materializing 229,376 addresses."""

    from offline.kerr_returning_radiation_kernel_jobs import (
        KerrKernelDirectionCoordinate,
    )

    _require_exact_int(ordinal, "ordinal")
    if ordinal >= plan.direction_count:
        raise PhaseSpaceGoldenError("ordinal lies outside the scientific plan")
    remaining = ordinal
    for pass_index, grid_pass in enumerate(plan.passes):
        angular_count = grid_pass.mu_order * grid_pass.psi_count
        pass_count = 2 * plan.annulus_count * grid_pass.rho_order * angular_count
        if remaining >= pass_count:
            remaining -= pass_count
            continue
        face_block = plan.annulus_count * grid_pass.rho_order * angular_count
        face_index, remaining = divmod(remaining, face_block)
        annulus_block = grid_pass.rho_order * angular_count
        annulus_index, remaining = divmod(remaining, annulus_block)
        rho_index, angular_index = divmod(remaining, angular_count)
        mu_index, psi_index = divmod(angular_index, grid_pass.psi_count)
        return KerrKernelDirectionCoordinate(
            ordinal,
            pass_index,
            grid_pass.name,
            face_index,
            ("upper", "lower")[face_index],
            annulus_index,
            rho_index,
            mu_index,
            psi_index,
        )
    raise AssertionError("canonical ordinal decoder exhausted the plan")


def _build_nested16_typed_definition() -> Any:
    """Build the exact nested16 physics/plan under the current source closure."""

    from offline.geodesic import RayTraceOptions, SurfaceEventOptions
    from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
    from offline.kerr_finite_thickness import (
        StationaryKerrFiniteThicknessCalibration,
    )
    from offline.kerr_finite_thickness_area import (
        KerrFiniteThicknessAreaQuadraturePolicy,
    )
    from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
    from offline.kerr_returning_radiation_kernel import (
        KerrReturningRadiationKernelPolicy,
    )
    import offline.kerr_returning_radiation_kernel_cached as cached

    metric = KerrKerrSchildMetric(
        mass_m=1.0,
        spin_a_m=0.7,
        singularity_guard_m=1.0e-9,
    )
    calibration = StationaryKerrFiniteThicknessCalibration(
        dimensionless_spin=0.7,
        eddington_scaled_mass_accretion_rate=0.05,
        orientation="prograde",
        outer_radius_over_mass=25.0,
        thinness_gate_maximum_h_over_rho=0.25,
    )
    surface = KerrFiniteThicknessMultiSurface(metric, calibration)
    termination = KerrOblateTermination.horizon_worldtube(
        metric,
        escape_radius_m=50.0,
        offset_m=0.02,
    )
    ray_options = RayTraceOptions(
        absolute_tolerance=6.25e-11,
        relative_tolerance=6.25e-11,
        initial_step=0.025,
        minimum_step=1.25e-9,
        maximum_step=0.125,
        maximum_affine_length=300.0,
        maximum_accepted_steps=100_000,
        maximum_rejected_steps=100_000,
        null_residual_limit=2.5e-8,
        metric_interpolation_error_limit=1.25e-8,
        event_value_tolerance=1.25e-10,
        event_affine_tolerance=1.25e-11,
        event_maximum_iterations=64,
        record_path=True,
    )
    surface_options = SurfaceEventOptions(
        absolute_tolerance=7.8125e-12,
        relative_tolerance=7.8125e-12,
        null_residual_limit=3.125e-9,
        metric_interpolation_error_limit=1.5625e-9,
        surface_value_tolerance=1.5625e-11,
        affine_tolerance=1.5625e-12,
        maximum_iterations=64,
        maximum_reintegrations=100_000,
        subdivisions_per_segment=16,
    )
    policy = KerrReturningRadiationKernelPolicy(
        rho_order=16,
        mu_order=32,
        psi_count=64,
        absolute_tolerance=2.0e-2,
        relative_tolerance=5.0e-2,
        symmetry_absolute_tolerance=2.0e-8,
        symmetry_relative_tolerance=2.0e-7,
        maximum_direction_evaluations=2_000_000,
        maximum_whole_ray_traces=8_000_000,
    )
    area = KerrFiniteThicknessAreaQuadraturePolicy(
        gauss_legendre_order=24,
        relative_tolerance=2.0e-10,
        absolute_tolerance_over_mass_squared=2.0e-11,
        maximum_point_evaluations=384,
    )
    definition = cached.build_forward_kerr_returning_radiation_kernel_cache_definition(
        surface,
        termination=termination,
        annulus_edges_over_mass=(
            float(calibration.isco_radius_over_mass),
            25.0,
        ),
        ray_options=ray_options,
        surface_options=surface_options,
        coarse_ray_options=None,
        coarse_surface_options=None,
        policy=policy,
        area_policy=area,
        directions_per_task=64,
    )
    return definition


def build_nested16_numeric_replay_context() -> Nested16NumericReplayContext:
    """Return typed nested16 numerical work without frozen job/source claims.

    No cache is opened or executed.  The current checkout necessarily produces
    temporary source-bound job keys; callers receive only a descriptor that
    labels this non-production numerical replay and omits those keys.
    """

    return Nested16NumericReplayContext(_build_nested16_typed_definition())


def _reference_definition() -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    """Rebuild the exact frozen nested16 typed context without cache execution."""

    import offline.kerr_returning_radiation_kernel_cached as cached

    definition = _build_nested16_typed_definition()
    context = definition.scientific_context
    if definition.job_spec.job_key != gc.FROZEN_NESTED16_TRUST.cache_job_key:
        raise PhaseSpaceGoldenError(
            "active/reference source-runtime closure does not reproduce the frozen cache job key"
        )
    if context.scientific_job_key != gc.FROZEN_NESTED16_TRUST.scientific_job_key:
        raise PhaseSpaceGoldenError(
            "active/reference backend does not reproduce the frozen scientific job key"
        )
    if context.scientific_plan_sha256 != gc.FROZEN_NESTED16_TRUST.scientific_plan_sha256:
        raise PhaseSpaceGoldenError("active/reference scientific plan SHA changed")

    closure = cached._source_closure_manifest()
    closure_binding = tuple(sorted(item.binding_sha256 for item in closure))
    if closure_binding != context.identity.source_closure_sha256:
        raise PhaseSpaceGoldenError("active source/runtime closure changed after definition build")
    runtime = cached._numeric_backend_descriptor()
    runtime_payload = _canonical_bytes(runtime)
    runtime_entries = [
        item for item in closure if item.logical_path == "runtime/numeric-backend.json"
    ]
    if len(runtime_entries) != 1:
        raise PhaseSpaceGoldenError("source closure lacks one numeric backend entry")
    runtime_entry = runtime_entries[0]
    if (
        runtime_entry.byte_length != len(runtime_payload)
        or runtime_entry.sha256 != hashlib.sha256(runtime_payload).hexdigest()
    ):
        raise PhaseSpaceGoldenError("numeric backend descriptor differs from its closure entry")
    reference = {
        "evaluatorArtifact": context.inputs[0].as_dict(),
        "numericBackend": runtime,
        "numericBackendSha256": runtime_entry.sha256,
        "scientificDocument": context.scientific_document(),
        "scientificDocumentSha256": _canonical_sha256(context.scientific_document()),
        "sourceRuntimeClosure": [
            {
                "bindingSha256": item.binding_sha256,
                "byteLength": item.byte_length,
                "logicalPath": item.logical_path,
                "sha256": item.sha256,
            }
            for item in closure
        ],
        "sourceRuntimeClosureManifestSha256": (
            cached._source_closure_manifest_sha256(closure)
        ),
    }
    return definition, closure, reference


def _trace_reference_record(
    definition: Any,
    coordinate: Any,
    cached_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from offline.kerr_finite_thickness_emitter import KerrFiniteThicknessFaceEmitter
    from offline.kerr_finite_thickness_launch import (
        KerrFiniteThicknessEmissionLaunch,
        KerrFiniteThicknessSurfaceFrame,
    )
    import offline.kerr_returning_radiation_kernel_cached as cached
    from offline.kerr_returning_radiation_rays import (
        trace_kerr_returning_radiation_direction,
        verify_kerr_returning_radiation_direction,
    )

    context = definition.scientific_context
    identity = context.identity
    node = cached._forward_coordinate_node(context, coordinate)
    sample = cached._forward_sample_document(node)
    if cached_record is not None:
        if cached_record["coordinate"] != coordinate.as_dict():
            raise PhaseSpaceGoldenError("cached coordinate differs from reference decoder")
        if cached_record["transport"]["sample"] != sample:
            raise PhaseSpaceGoldenError("cached quadrature sample differs from active reference")

    emitter = KerrFiniteThicknessFaceEmitter(
        metric=identity.surface.metric,
        calibration=identity.surface.calibration,
        pseudo_cylindrical_radius_over_mass=node.source_radius_over_mass,
        face=node.source_face,
    )
    launch = KerrFiniteThicknessEmissionLaunch(
        KerrFiniteThicknessSurfaceFrame(emitter),
        node.emission_angle_cosine,
        node.tangent_azimuth_rad,
        1.0,
    )
    primitive = trace_kerr_returning_radiation_direction(
        launch,
        identity.surface,
        termination=identity.termination,
        ray_options=identity.fine_ray_options,
        surface_options=identity.fine_surface_options,
        coarse_ray_options=identity.coarse_ray_options,
        coarse_surface_options=identity.coarse_surface_options,
    )
    # This independent mandatory replay detects mutable or substituted
    # production results before any reference bytes are accepted.
    verify_kerr_returning_radiation_direction(primitive)

    primitive_sha = primitive.model_descriptor_sha256
    cached_sha = None
    verified = False
    if cached_record is not None:
        cached_sha = cached_record["transport"]["transport"][
            "primitiveDescriptorSha256"
        ]
        if primitive_sha != cached_sha:
            raise PhaseSpaceGoldenError(
                f"ordinal {coordinate.ordinal} active primitive SHA differs from authenticated cache"
            )
        verified = True

    descriptor = primitive.model_descriptor()
    if _descriptor_sha256(descriptor) != primitive_sha:
        raise PhaseSpaceGoldenError("reference primitive descriptor hash is stale")
    fine_projection = _ray_document(primitive.ray)
    coarse_projection = _ray_document(primitive.coarse_ray)
    return {
        "cachePrimitiveDescriptorSha256": cached_sha,
        "cachePrimitiveDescriptorVerified": verified,
        "coordinate": coordinate.as_dict(),
        "phaseSpace": _phase_space_document(primitive),
        "referenceEvidence": {
            "coarseRayExecutionSha256": _descriptor_sha256(asdict(primitive.coarse_ray)),
            "coarseRayProjectionSha256": _canonical_sha256(coarse_projection),
            "fineRayExecutionSha256": _descriptor_sha256(asdict(primitive.ray)),
            "fineRayProjectionSha256": _canonical_sha256(fine_projection),
            "launchDescriptorSha256": launch.model_descriptor_sha256,
            "normalizedLaunchDescriptorSha256": (
                primitive.normalized_launch.model_descriptor_sha256
            ),
            "primitiveDescriptor": descriptor,
            "primitiveDescriptorSha256": primitive_sha,
        },
        "sample": sample,
    }


def _manifest_basis(document_without_manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = document_without_manifest["records"]
    comparison_records = [
        {
            "coordinate": item["coordinate"],
            "phaseSpace": item["phaseSpace"],
            "sample": item["sample"],
        }
        for item in records
    ]
    reference = document_without_manifest["reference"]
    return {
        "comparisonRecordsSha256": _canonical_sha256(comparison_records),
        "recordSha256": [_canonical_sha256(item) for item in records],
        "recordsSha256": _canonical_sha256(records),
        "referenceConfigurationSha256": _canonical_sha256(
            {
                "anchors": document_without_manifest["anchors"],
                "reference": reference,
                "selectedOrdinals": document_without_manifest["selectedOrdinals"],
            }
        ),
        "schema": MANIFEST_SCHEMA,
        "sourceRuntimeClosureManifestSha256": reference[
            "sourceRuntimeClosureManifestSha256"
        ],
    }


def _attach_manifest(document_without_manifest: dict[str, Any]) -> dict[str, Any]:
    basis = _manifest_basis(document_without_manifest)
    manifest = dict(basis)
    manifest["manifestSha256"] = _canonical_sha256(basis)
    return {**document_without_manifest, "manifest": manifest}


def generate_reference_document(cache_job_directory: Path) -> dict[str, Any]:
    """Generate eight fully expanded rays without writing or running the cache."""

    narrow = gc.extract_golden_corpus(
        cache_job_directory,
        selected_ordinals=CACHED_SELECTED_ORDINALS,
        trust=gc.FROZEN_NESTED16_TRUST,
    )
    cached_by_ordinal = {
        item["coordinate"]["ordinal"]: item for item in narrow["records"]
    }
    if tuple(cached_by_ordinal) != CACHED_SELECTED_ORDINALS:
        raise PhaseSpaceGoldenError("authenticated cache omitted a selected ordinal")

    definition, closure_before, reference = _reference_definition()
    records: list[dict[str, Any]] = []
    for ordinal in REFERENCE_ORDINALS:
        coordinate = _coordinate_for_ordinal(definition.plan, ordinal)
        records.append(
            _trace_reference_record(
                definition,
                coordinate,
                cached_by_ordinal.get(ordinal),
            )
        )

    import offline.kerr_returning_radiation_kernel_cached as cached

    closure_after = cached._source_closure_manifest()
    before_tree = tuple(
        (item.logical_path, item.byte_length, item.sha256, item.binding_sha256)
        for item in closure_before
    )
    after_tree = tuple(
        (item.logical_path, item.byte_length, item.sha256, item.binding_sha256)
        for item in closure_after
    )
    if before_tree != after_tree:
        raise PhaseSpaceGoldenError("source/runtime closure changed during reference tracing")

    anchors = {
        "authenticatedDirectionCount": narrow["evidence"][
            "authenticatedDirectionCount"
        ],
        "authenticatedDirectionStreamSha256": narrow["evidence"][
            "authenticatedDirectionStreamSha256"
        ],
        "authenticatedPayloadSetSha256": narrow["evidence"][
            "authenticatedPayloadSetSha256"
        ],
        "authenticatedTaskCount": narrow["evidence"]["authenticatedTaskCount"],
        "cacheJobKey": gc.FROZEN_NESTED16_TRUST.cache_job_key,
        "jobDocumentSha256": gc.FROZEN_NESTED16_TRUST.job_document_sha256,
        "narrowCorpusSha256": gc.FROZEN_NESTED16_CORPUS_SHA256,
        "scientificJobKey": gc.FROZEN_NESTED16_TRUST.scientific_job_key,
        "scientificPlanSha256": gc.FROZEN_NESTED16_TRUST.scientific_plan_sha256,
    }
    return _attach_manifest(
        {
            "anchors": anchors,
            "records": records,
            "reference": reference,
            "schema": GOLDEN_SCHEMA,
            "selectedOrdinals": list(REFERENCE_ORDINALS),
            "version": VERSION,
        }
    )


def _validate_state(value: Any, label: str) -> None:
    document = _require_exact_keys(value, {"covector", "event"}, label)
    for name in ("event", "covector"):
        vector = document[name]
        if type(vector) is not list or len(vector) != 4:
            raise PhaseSpaceGoldenError(f"{label}.{name} must contain four floats")
        for index, item in enumerate(vector):
            _require_finite_float(item, f"{label}.{name}[{index}]")


def _validate_surface_trace(value: Any, label: str, segment_count: int) -> None:
    if value is None:
        raise PhaseSpaceGoldenError(f"{label} must retain complete multi-surface evidence")
    keys = {
        "baseSubdivisionsPerStep",
        "crossings",
        "initialContact",
        "maximumProbeCovectorRelativeDifference",
        "maximumProbeEventDifference",
        "probeReintegrations",
        "surfaceIds",
        "surfaceValueEvaluations",
        "topologyConverged",
        "verificationSubdivisionsPerStep",
    }
    trace = _require_exact_keys(value, keys, label)
    base = _require_exact_int(trace["baseSubdivisionsPerStep"], f"{label}.base", 2)
    verification = _require_exact_int(
        trace["verificationSubdivisionsPerStep"], f"{label}.verification", 4
    )
    if base % 2 or verification != 2 * base:
        raise PhaseSpaceGoldenError(f"{label} has inconsistent N/2N subdivisions")
    if trace["topologyConverged"] is not True:
        raise PhaseSpaceGoldenError(f"{label} is not topology-converged")
    for name in (
        "maximumProbeCovectorRelativeDifference",
        "maximumProbeEventDifference",
    ):
        value_float = _require_finite_float(trace[name], f"{label}.{name}")
        if value_float < 0.0:
            raise PhaseSpaceGoldenError(f"{label}.{name} must be non-negative")
    _require_exact_int(trace["probeReintegrations"], f"{label}.probeReintegrations")
    _require_exact_int(
        trace["surfaceValueEvaluations"], f"{label}.surfaceValueEvaluations"
    )
    surface_ids = trace["surfaceIds"]
    if (
        type(surface_ids) is not list
        or not surface_ids
        or any(type(item) is not str or not item for item in surface_ids)
        or surface_ids != sorted(set(surface_ids))
    ):
        raise PhaseSpaceGoldenError(f"{label}.surfaceIds is not canonical")
    initial = trace["initialContact"]
    initial_doc = _require_exact_keys(
        initial,
        {"actualSurfaceValue", "side", "surfaceId", "surfaceValueTolerance"},
        f"{label}.initialContact",
    )
    if initial_doc["surfaceId"] not in surface_ids or initial_doc["side"] not in (-1, 1):
        raise PhaseSpaceGoldenError(f"{label}.initialContact is invalid")
    actual = _require_finite_float(
        initial_doc["actualSurfaceValue"], f"{label}.initialContact.actual"
    )
    tolerance = _require_finite_float(
        initial_doc["surfaceValueTolerance"], f"{label}.initialContact.tolerance"
    )
    if tolerance <= 0.0 or abs(actual) > tolerance:
        raise PhaseSpaceGoldenError(f"{label}.initialContact exceeds tolerance")
    crossings = trace["crossings"]
    if type(crossings) is not list:
        raise PhaseSpaceGoldenError(f"{label}.crossings must be an array")
    previous_affine = -math.inf
    terminal_seen = False
    crossing_keys = {
        "bracketAffineWidth",
        "decision",
        "iterations",
        "orientation",
        "rayAffineLength",
        "segmentAffineLength",
        "segmentIndex",
        "state",
        "surfaceId",
        "surfaceValue",
    }
    for index, item in enumerate(crossings):
        entry = _require_exact_keys(item, crossing_keys, f"{label}.crossings[{index}]")
        if entry["surfaceId"] not in surface_ids:
            raise PhaseSpaceGoldenError(f"{label}.crossings[{index}] has unknown surface")
        segment_index = _require_exact_int(
            entry["segmentIndex"], f"{label}.crossings[{index}].segmentIndex"
        )
        if segment_index >= segment_count:
            raise PhaseSpaceGoldenError(f"{label}.crossings[{index}] has invalid segment")
        _require_exact_int(entry["iterations"], f"{label}.crossings[{index}].iterations")
        if entry["orientation"] not in (-1, 1):
            raise PhaseSpaceGoldenError(f"{label}.crossings[{index}] has invalid orientation")
        for name in (
            "bracketAffineWidth",
            "rayAffineLength",
            "segmentAffineLength",
        ):
            scalar = _require_finite_float(entry[name], f"{label}.crossings[{index}].{name}")
            if scalar < 0.0:
                raise PhaseSpaceGoldenError(f"{label}.crossings[{index}].{name} is negative")
        _require_finite_float(
            entry["surfaceValue"], f"{label}.crossings[{index}].surfaceValue"
        )
        affine = entry["rayAffineLength"]
        if affine <= previous_affine or terminal_seen:
            raise PhaseSpaceGoldenError(f"{label}.crossings are not strictly terminal-ordered")
        previous_affine = affine
        decision = _require_exact_keys(
            entry["decision"],
            {"classification", "outcome", "targetId"},
            f"{label}.crossings[{index}].decision",
        )
        if type(decision["classification"]) is not str or not decision["classification"]:
            raise PhaseSpaceGoldenError(f"{label}.crossings[{index}] lacks classification")
        if (decision["outcome"] is None) != (decision["targetId"] is None):
            raise PhaseSpaceGoldenError(f"{label}.crossings[{index}] has partial terminal decision")
        if decision["outcome"] is not None:
            if type(decision["outcome"]) is not str or type(decision["targetId"]) is not str:
                raise PhaseSpaceGoldenError(f"{label}.crossings[{index}] has invalid decision")
            terminal_seen = True
        _validate_state(entry["state"], f"{label}.crossings[{index}].state")


def _validate_ray(value: Any, label: str) -> None:
    keys = {
        "acceptedSteps",
        "affineLength",
        "failureReason",
        "maximumMetricInterpolationError",
        "maximumNullResidual",
        "outcome",
        "rejectedSteps",
        "segments",
        "surfaceTrace",
        "terminalState",
        "terminalTargetId",
    }
    ray = _require_exact_keys(value, keys, label)
    if type(ray["outcome"]) is not str or not ray["outcome"]:
        raise PhaseSpaceGoldenError(f"{label}.outcome is invalid")
    if ray["failureReason"] is not None:
        raise PhaseSpaceGoldenError(f"{label} carries a failure reason")
    if type(ray["terminalTargetId"]) is not str or not ray["terminalTargetId"]:
        raise PhaseSpaceGoldenError(f"{label}.terminalTargetId is invalid")
    accepted = _require_exact_int(ray["acceptedSteps"], f"{label}.acceptedSteps", 1)
    _require_exact_int(ray["rejectedSteps"], f"{label}.rejectedSteps")
    affine = _require_finite_float(ray["affineLength"], f"{label}.affineLength")
    if affine <= 0.0:
        raise PhaseSpaceGoldenError(f"{label}.affineLength must be positive")
    for name in ("maximumMetricInterpolationError", "maximumNullResidual"):
        scalar = _require_finite_float(ray[name], f"{label}.{name}")
        if scalar < 0.0:
            raise PhaseSpaceGoldenError(f"{label}.{name} must be non-negative")
    segments = ray["segments"]
    if type(segments) is not list or len(segments) != accepted:
        raise PhaseSpaceGoldenError(f"{label}.segments must equal acceptedSteps")
    previous_end = None
    accumulated = 0.0
    segment_keys = {
        "affineLength",
        "end",
        "midpoint",
        "midpointNullResidual",
        "start",
    }
    for index, item in enumerate(segments):
        segment = _require_exact_keys(item, segment_keys, f"{label}.segments[{index}]")
        length = _require_finite_float(
            segment["affineLength"], f"{label}.segments[{index}].affineLength"
        )
        residual = _require_finite_float(
            segment["midpointNullResidual"],
            f"{label}.segments[{index}].midpointNullResidual",
        )
        if length <= 0.0 or residual < 0.0:
            raise PhaseSpaceGoldenError(f"{label}.segments[{index}] is invalid")
        for name in ("start", "midpoint", "end"):
            _validate_state(segment[name], f"{label}.segments[{index}].{name}")
        if previous_end is not None and segment["start"] != previous_end:
            raise PhaseSpaceGoldenError(f"{label}.segments are not contiguous")
        previous_end = segment["end"]
        accumulated = math.fsum((accumulated, length))
    _validate_state(ray["terminalState"], f"{label}.terminalState")
    if previous_end != ray["terminalState"]:
        raise PhaseSpaceGoldenError(f"{label}.terminalState differs from final segment")
    # Exact fsum equality is not required: the tracer's affine accumulator and
    # a post-hoc fsum can round at different points.  Their discrepancy must be
    # within a tiny binary64 accounting envelope.
    if abs(accumulated - affine) > 8.0 * math.ulp(max(1.0, affine)):
        raise PhaseSpaceGoldenError(f"{label}.affineLength disagrees with segments")
    _validate_surface_trace(ray["surfaceTrace"], f"{label}.surfaceTrace", accepted)


def _validate_phase_space(value: Any, label: str) -> None:
    phase = _require_exact_keys(
        value, {"coarse", "convergence", "fate", "fine", "receiver"}, label
    )
    if phase["fate"] not in (
        "captured",
        "escaped",
        "plunge-sink",
        "return-upper",
        "return-lower",
    ):
        raise PhaseSpaceGoldenError(f"{label}.fate is unsupported")
    _validate_ray(phase["fine"], f"{label}.fine")
    _validate_ray(phase["coarse"], f"{label}.coarse")
    convergence = phase["convergence"]
    if type(convergence) is not dict:
        raise PhaseSpaceGoldenError(f"{label}.convergence must be an object")
    _require_finite_tree(convergence, f"{label}.convergence")
    required_booleans = {
        "outcome_agrees",
        "target_agrees",
        "fate_agrees",
        "complete_topology_agrees",
        "converged",
    }
    if not required_booleans.issubset(convergence):
        raise PhaseSpaceGoldenError(f"{label}.convergence omits categorical gates")
    if any(convergence[name] is not True for name in required_booleans):
        raise PhaseSpaceGoldenError(f"{label}.convergence is not fully qualified")
    receiver = phase["receiver"]
    if phase["fate"].startswith("return-"):
        if type(receiver) is not dict:
            raise PhaseSpaceGoldenError(f"{label} returned fate lacks receiver evidence")
        _require_finite_tree(receiver, f"{label}.receiver")
        if receiver.get("face") != phase["fate"].removeprefix("return-"):
            raise PhaseSpaceGoldenError(f"{label}.receiver face disagrees with fate")
        descriptor = receiver.get("descriptor")
        digest = receiver.get("descriptorSha256")
        _require_sha256(digest, f"{label}.receiver.descriptorSha256")
        if _descriptor_sha256(descriptor) != digest:
            raise PhaseSpaceGoldenError(f"{label}.receiver descriptor SHA is stale")
    elif receiver is not None:
        raise PhaseSpaceGoldenError(f"{label} non-returning fate carries receiver evidence")


def validate_reference_document(document: Any) -> dict[str, Any]:
    keys = {
        "anchors",
        "manifest",
        "records",
        "reference",
        "schema",
        "selectedOrdinals",
        "version",
    }
    golden = _require_exact_keys(document, keys, "golden")
    if golden["schema"] != GOLDEN_SCHEMA or golden["version"] != VERSION:
        raise PhaseSpaceGoldenError("golden schema/version is unsupported")
    if golden["selectedOrdinals"] != list(REFERENCE_ORDINALS):
        raise PhaseSpaceGoldenError("golden selected ordinal set/order changed")

    anchors = _require_exact_keys(
        golden["anchors"],
        {
            "authenticatedDirectionCount",
            "authenticatedDirectionStreamSha256",
            "authenticatedPayloadSetSha256",
            "authenticatedTaskCount",
            "cacheJobKey",
            "jobDocumentSha256",
            "narrowCorpusSha256",
            "scientificJobKey",
            "scientificPlanSha256",
        },
        "golden.anchors",
    )
    expected_anchor_values = {
        "authenticatedDirectionCount": gc.FROZEN_NESTED16_TRUST.authenticated_direction_count,
        "authenticatedDirectionStreamSha256": gc.FROZEN_NESTED16_TRUST.authenticated_direction_stream_sha256,
        "authenticatedPayloadSetSha256": gc.FROZEN_NESTED16_TRUST.authenticated_payload_set_sha256,
        "authenticatedTaskCount": gc.FROZEN_NESTED16_TRUST.authenticated_task_count,
        "cacheJobKey": gc.FROZEN_NESTED16_TRUST.cache_job_key,
        "jobDocumentSha256": gc.FROZEN_NESTED16_TRUST.job_document_sha256,
        "narrowCorpusSha256": gc.FROZEN_NESTED16_CORPUS_SHA256,
        "scientificJobKey": gc.FROZEN_NESTED16_TRUST.scientific_job_key,
        "scientificPlanSha256": gc.FROZEN_NESTED16_TRUST.scientific_plan_sha256,
    }
    if anchors != expected_anchor_values:
        raise PhaseSpaceGoldenError("golden cache/scientific anchors changed")

    reference = _require_exact_keys(
        golden["reference"],
        {
            "evaluatorArtifact",
            "numericBackend",
            "numericBackendSha256",
            "scientificDocument",
            "scientificDocumentSha256",
            "sourceRuntimeClosure",
            "sourceRuntimeClosureManifestSha256",
        },
        "golden.reference",
    )
    _require_finite_tree(reference["numericBackend"], "golden.reference.numericBackend")
    runtime_payload = _canonical_bytes(reference["numericBackend"])
    runtime_sha = _require_sha256(
        reference["numericBackendSha256"], "golden.reference.numericBackendSha256"
    )
    if hashlib.sha256(runtime_payload).hexdigest() != runtime_sha:
        raise PhaseSpaceGoldenError("golden numeric backend SHA is stale")
    scientific_sha = _require_sha256(
        reference["scientificDocumentSha256"],
        "golden.reference.scientificDocumentSha256",
    )
    if _canonical_sha256(reference["scientificDocument"]) != scientific_sha:
        raise PhaseSpaceGoldenError("golden scientific document SHA is stale")
    scientific = reference["scientificDocument"]
    if (
        type(scientific) is not dict
        or _canonical_sha256(scientific) != anchors["scientificJobKey"]
    ):
        raise PhaseSpaceGoldenError("golden scientific document differs from job key")
    closure = reference["sourceRuntimeClosure"]
    if type(closure) is not list or not closure:
        raise PhaseSpaceGoldenError("golden source/runtime closure is empty")
    closure_lines: list[str] = []
    runtime_matches = 0
    binding_values: list[str] = []
    for index, item in enumerate(closure):
        entry = _require_exact_keys(
            item,
            {"bindingSha256", "byteLength", "logicalPath", "sha256"},
            f"golden.reference.sourceRuntimeClosure[{index}]",
        )
        logical = entry["logicalPath"]
        if type(logical) is not str or not logical:
            raise PhaseSpaceGoldenError("golden source closure has invalid logical path")
        byte_length = _require_exact_int(entry["byteLength"], "closure byteLength")
        sha = _require_sha256(entry["sha256"], "closure sha256")
        binding = _require_sha256(entry["bindingSha256"], "closure bindingSha256")
        expected_binding = hashlib.sha256(
            f"{logical}\0{byte_length}\0{sha}".encode("utf-8")
        ).hexdigest()
        if binding != expected_binding:
            raise PhaseSpaceGoldenError("golden source closure binding is stale")
        binding_values.append(binding)
        closure_lines.append(f"{logical}\0{byte_length}\0{sha}")
        if logical == "runtime/numeric-backend.json":
            runtime_matches += 1
            if byte_length != len(runtime_payload) or sha != runtime_sha:
                raise PhaseSpaceGoldenError("numeric backend closure entry is stale")
    logical_paths = [item["logicalPath"] for item in closure]
    expected_logical_paths = [
        *sorted(path for path in logical_paths if path != "runtime/numeric-backend.json"),
        "runtime/numeric-backend.json",
    ]
    if runtime_matches != 1 or logical_paths != expected_logical_paths:
        raise PhaseSpaceGoldenError("golden source/runtime closure is not canonical")
    manifest_sha = hashlib.sha256("\n".join(closure_lines).encode("utf-8")).hexdigest()
    if manifest_sha != reference["sourceRuntimeClosureManifestSha256"]:
        raise PhaseSpaceGoldenError("golden source/runtime closure manifest is stale")
    identity_closure = scientific.get("scientificIdentity", {}).get(
        "sourceClosureSha256"
    )
    if identity_closure != sorted(binding_values):
        raise PhaseSpaceGoldenError("scientific identity closure differs from manifest")

    records = golden["records"]
    if type(records) is not list or len(records) != len(REFERENCE_ORDINALS):
        raise PhaseSpaceGoldenError("golden record count is wrong")
    for index, (record, expected_ordinal) in enumerate(zip(records, REFERENCE_ORDINALS)):
        record_doc = _require_exact_keys(
            record,
            {
                "cachePrimitiveDescriptorSha256",
                "cachePrimitiveDescriptorVerified",
                "coordinate",
                "phaseSpace",
                "referenceEvidence",
                "sample",
            },
            f"golden.records[{index}]",
        )
        coordinate = record_doc["coordinate"]
        if type(coordinate) is not dict or coordinate.get("ordinal") != expected_ordinal:
            raise PhaseSpaceGoldenError(f"golden.records[{index}] coordinate/order changed")
        _require_finite_tree(coordinate, f"golden.records[{index}].coordinate")
        _require_finite_tree(record_doc["sample"], f"golden.records[{index}].sample")
        _validate_phase_space(record_doc["phaseSpace"], f"golden.records[{index}].phaseSpace")
        evidence = _require_exact_keys(
            record_doc["referenceEvidence"],
            {
                "coarseRayExecutionSha256",
                "coarseRayProjectionSha256",
                "fineRayExecutionSha256",
                "fineRayProjectionSha256",
                "launchDescriptorSha256",
                "normalizedLaunchDescriptorSha256",
                "primitiveDescriptor",
                "primitiveDescriptorSha256",
            },
            f"golden.records[{index}].referenceEvidence",
        )
        for name in evidence:
            if name.endswith("Sha256"):
                _require_sha256(evidence[name], f"golden.records[{index}].{name}")
        primitive_sha = evidence["primitiveDescriptorSha256"]
        primitive_descriptor = evidence["primitiveDescriptor"]
        if _descriptor_sha256(primitive_descriptor) != primitive_sha:
            raise PhaseSpaceGoldenError(f"golden.records[{index}] primitive SHA is stale")
        try:
            descriptor_emission = primitive_descriptor["emission"]
            descriptor_convergence = primitive_descriptor["wholeRayConvergence"]
        except (KeyError, TypeError) as error:
            raise PhaseSpaceGoldenError(
                f"golden.records[{index}] primitive descriptor omits replay evidence"
            ) from error
        if (
            primitive_descriptor.get("fate") != record_doc["phaseSpace"]["fate"]
            or descriptor_emission.get("launchDescriptorSha256")
            != evidence["launchDescriptorSha256"]
            or descriptor_emission.get("normalizedLaunchDescriptorSha256")
            != evidence["normalizedLaunchDescriptorSha256"]
            or descriptor_convergence.get("fineRayExecutionSha256")
            != evidence["fineRayExecutionSha256"]
            or descriptor_convergence.get("coarseRayExecutionSha256")
            != evidence["coarseRayExecutionSha256"]
            or descriptor_convergence.get("actual")
            != record_doc["phaseSpace"]["convergence"]
        ):
            raise PhaseSpaceGoldenError(
                f"golden.records[{index}] primitive descriptor disagrees with expanded evidence"
            )
        if _canonical_sha256(record_doc["phaseSpace"]["fine"]) != evidence[
            "fineRayProjectionSha256"
        ]:
            raise PhaseSpaceGoldenError(f"golden.records[{index}] fine projection SHA is stale")
        if _canonical_sha256(record_doc["phaseSpace"]["coarse"]) != evidence[
            "coarseRayProjectionSha256"
        ]:
            raise PhaseSpaceGoldenError(f"golden.records[{index}] coarse projection SHA is stale")
        cached_sha = record_doc["cachePrimitiveDescriptorSha256"]
        cached_verified = record_doc["cachePrimitiveDescriptorVerified"]
        if expected_ordinal in CACHED_SELECTED_ORDINALS:
            _require_sha256(cached_sha, f"golden.records[{index}].cachedPrimitive")
            if cached_verified is not True or cached_sha != primitive_sha:
                raise PhaseSpaceGoldenError(
                    f"golden.records[{index}] did not verify cached primitive SHA"
                )
        elif cached_sha is not None or cached_verified is not False:
            raise PhaseSpaceGoldenError(
                f"golden.records[{index}] fabricates cache evidence for noncached ordinal"
            )

    manifest = golden["manifest"]
    expected_manifest_keys = set(_manifest_basis({
        key: golden[key] for key in golden if key != "manifest"
    })) | {"manifestSha256"}
    manifest_doc = _require_exact_keys(manifest, expected_manifest_keys, "golden.manifest")
    basis = _manifest_basis({key: golden[key] for key in golden if key != "manifest"})
    expected_manifest = {**basis, "manifestSha256": _canonical_sha256(basis)}
    if manifest_doc != expected_manifest:
        raise PhaseSpaceGoldenError("golden manifest hashes are stale")
    return golden


def load_reference_document(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, "expected golden SHA-256")
    if expected == "0" * 64:
        raise PhaseSpaceGoldenError("checked-in golden SHA-256 has not been frozen")
    payload, document = _read_document(path, "phase-space golden")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise PhaseSpaceGoldenError("phase-space golden differs from external SHA-256 anchor")
    return validate_reference_document(document)


def candidate_document_from_reference(
    golden: Mapping[str, Any],
    *,
    golden_sha256: str,
    backend: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_reference_document(golden)
    _require_sha256(golden_sha256, "golden_sha256")
    selected_backend = (
        {"implementationId": "reference-copy/test-only"}
        if backend is None
        else deepcopy(dict(backend))
    )
    _require_finite_tree(selected_backend, "candidate.backend")
    return {
        "backend": selected_backend,
        "goldenSha256": golden_sha256,
        "records": [
            {
                "coordinate": deepcopy(item["coordinate"]),
                "phaseSpace": deepcopy(item["phaseSpace"]),
                "sample": deepcopy(item["sample"]),
            }
            for item in golden["records"]
        ],
        "schema": CANDIDATE_SCHEMA,
        "version": VERSION,
    }


def validate_candidate_document(
    document: Any,
    golden: Mapping[str, Any],
    golden_sha256: str,
) -> dict[str, Any]:
    candidate = _require_exact_keys(
        document,
        {"backend", "goldenSha256", "records", "schema", "version"},
        "candidate",
    )
    if candidate["schema"] != CANDIDATE_SCHEMA or candidate["version"] != VERSION:
        raise PhaseSpaceGoldenError("candidate schema/version is unsupported")
    if candidate["goldenSha256"] != golden_sha256:
        raise PhaseSpaceGoldenError("candidate is bound to a different golden SHA-256")
    if type(candidate["backend"]) is not dict or not candidate["backend"]:
        raise PhaseSpaceGoldenError("candidate backend provenance must be a non-empty object")
    _require_finite_tree(candidate["backend"], "candidate.backend")
    records = candidate["records"]
    if type(records) is not list or len(records) != len(REFERENCE_ORDINALS):
        raise PhaseSpaceGoldenError("candidate record count is wrong")
    for index, (record, expected) in enumerate(zip(records, golden["records"])):
        item = _require_exact_keys(
            record,
            {"coordinate", "phaseSpace", "sample"},
            f"candidate.records[{index}]",
        )
        if item["coordinate"] != expected["coordinate"]:
            raise PhaseSpaceGoldenError(f"candidate.records[{index}] coordinate/order changed")
        _require_finite_tree(item["sample"], f"candidate.records[{index}].sample")
        _validate_phase_space(item["phaseSpace"], f"candidate.records[{index}].phaseSpace")
    return candidate


def _ordered_float_bits(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return (~bits & 0xFFFFFFFFFFFFFFFF) if bits & (1 << 63) else bits | (1 << 63)


def _ulp_distance(left: float, right: float) -> int:
    return abs(_ordered_float_bits(left) - _ordered_float_bits(right))


def _compare_tree(
    expected: Any,
    actual: Any,
    path: str,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    maximum_ulp: int,
    mismatches: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> None:
    if type(actual) is not type(expected):
        mismatches.append(
            {"actualType": type(actual).__name__, "expectedType": type(expected).__name__, "path": path, "reason": "type"}
        )
        return
    if type(expected) is dict:
        if set(actual) != set(expected):
            mismatches.append({"path": path, "reason": "object-keys"})
            return
        for key in sorted(expected):
            if len(mismatches) >= _MAXIMUM_MISMATCHES:
                return
            _compare_tree(
                expected[key], actual[key], f"{path}.{key}",
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                maximum_ulp=maximum_ulp,
                mismatches=mismatches,
                diagnostics=diagnostics,
            )
        return
    if type(expected) is list:
        if len(actual) != len(expected):
            mismatches.append(
                {"actualLength": len(actual), "expectedLength": len(expected), "path": path, "reason": "array-length"}
            )
            return
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            if len(mismatches) >= _MAXIMUM_MISMATCHES:
                return
            _compare_tree(
                expected_item, actual_item, f"{path}[{index}]",
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                maximum_ulp=maximum_ulp,
                mismatches=mismatches,
                diagnostics=diagnostics,
            )
        return
    if type(expected) is float:
        if expected == 0.0 and actual == 0.0 and math.copysign(1.0, expected) != math.copysign(1.0, actual):
            mismatches.append({"actual": actual.hex(), "expected": expected.hex(), "path": path, "reason": "signed-zero"})
            return
        absolute = abs(actual - expected)
        scale = max(abs(expected), abs(actual))
        relative = 0.0 if absolute == 0.0 else absolute / max(scale, sys.float_info.min)
        ulp = _ulp_distance(expected, actual)
        diagnostics["comparedFloatCount"] += 1
        diagnostics["maximumAbsoluteDifference"] = max(
            diagnostics["maximumAbsoluteDifference"], absolute
        )
        diagnostics["maximumRelativeDifference"] = max(
            diagnostics["maximumRelativeDifference"], relative
        )
        diagnostics["maximumUlpDistance"] = max(diagnostics["maximumUlpDistance"], ulp)
        qualified = (
            absolute <= absolute_tolerance
            or relative <= relative_tolerance
            or ulp <= maximum_ulp
        )
        if not qualified:
            mismatches.append(
                {
                    "absoluteDifference": absolute,
                    "actual": actual.hex(),
                    "expected": expected.hex(),
                    "path": path,
                    "reason": "numeric",
                    "relativeDifference": relative,
                    "ulpDistance": ulp,
                }
            )
        return
    if type(expected) is bool:
        differs = actual is not expected
    else:
        differs = actual != expected
    if differs:
        mismatches.append({"actual": actual, "expected": expected, "path": path, "reason": "categorical"})


def compare_documents(
    golden: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    golden_sha256: str,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    maximum_ulp: int = 0,
    require_byte_exact: bool = False,
) -> dict[str, Any]:
    validate_reference_document(golden)
    validate_candidate_document(candidate, golden, golden_sha256)
    for name, value in (
        ("absolute_tolerance", absolute_tolerance),
        ("relative_tolerance", relative_tolerance),
    ):
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise PhaseSpaceGoldenError(f"{name} must be an exact finite non-negative float")
    _require_exact_int(maximum_ulp, "maximum_ulp")
    if type(require_byte_exact) is not bool:
        raise PhaseSpaceGoldenError("require_byte_exact must be an exact bool")
    if require_byte_exact and any(
        value != 0 for value in (absolute_tolerance, relative_tolerance, maximum_ulp)
    ):
        raise PhaseSpaceGoldenError("byte-exact comparison cannot enable numeric tolerances")

    expected_records = [
        {"coordinate": item["coordinate"], "phaseSpace": item["phaseSpace"], "sample": item["sample"]}
        for item in golden["records"]
    ]
    actual_records = candidate["records"]
    mismatches: list[dict[str, Any]] = []
    diagnostics = {
        "comparedFloatCount": 0,
        "maximumAbsoluteDifference": 0.0,
        "maximumRelativeDifference": 0.0,
        "maximumUlpDistance": 0,
    }
    _compare_tree(
        expected_records,
        actual_records,
        "records",
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        maximum_ulp=maximum_ulp,
        mismatches=mismatches,
        diagnostics=diagnostics,
    )
    byte_exact = _canonical_bytes(expected_records) == _canonical_bytes(actual_records)
    qualified = not mismatches and (byte_exact if require_byte_exact else True)
    return {
        "backend": candidate["backend"],
        "byteExact": byte_exact,
        "diagnostics": diagnostics,
        "goldenManifestSha256": golden["manifest"]["manifestSha256"],
        "goldenSha256": golden_sha256,
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "policy": {
            "absoluteTolerance": absolute_tolerance,
            "categoricalAndTopologyExact": True,
            "maximumUlp": maximum_ulp,
            "relativeTolerance": relative_tolerance,
            "requireByteExact": require_byte_exact,
            "signedZeroExact": True,
        },
        "qualified": qualified,
        "schema": COMPARISON_SCHEMA,
        "version": VERSION,
    }


def compare_files(
    golden_path: Path,
    candidate_path: Path,
    *,
    expected_golden_sha256: str,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    maximum_ulp: int = 0,
    require_byte_exact: bool = False,
) -> dict[str, Any]:
    golden = load_reference_document(
        golden_path, expected_sha256=expected_golden_sha256
    )
    _candidate_payload, candidate = _read_document(candidate_path, "phase-space candidate")
    return compare_documents(
        golden,
        candidate,
        golden_sha256=expected_golden_sha256,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        maximum_ulp=maximum_ulp,
        require_byte_exact=require_byte_exact,
    )


def _non_negative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _write(document: Mapping[str, Any], output: Path) -> None:
    try:
        gc._write_canonical_output(output, _canonical_bytes(document))
    except gc.GoldenCacheError as error:
        raise PhaseSpaceGoldenError(str(error)) from error


def _require_output_outside_cache(output: Path, cache_job_directory: Path) -> Path:
    output_absolute = Path(os.path.abspath(os.fspath(output)))
    cache_absolute = Path(os.path.abspath(os.fspath(cache_job_directory)))
    try:
        output_absolute.relative_to(cache_absolute)
    except ValueError:
        return output_absolute
    raise PhaseSpaceGoldenError("output must not be inside the golden cache")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("cache_job_directory", type=Path)
    generate.add_argument("--output", required=True, type=Path)

    verify = subparsers.add_parser("verify")
    verify.add_argument("golden", type=Path)
    verify.add_argument("--expected-sha256", default=FROZEN_REFERENCE_SHA256)

    candidate = subparsers.add_parser("make-reference-candidate")
    candidate.add_argument("golden", type=Path)
    candidate.add_argument("--expected-sha256", default=FROZEN_REFERENCE_SHA256)
    candidate.add_argument("--output", required=True, type=Path)

    compare = subparsers.add_parser("compare")
    compare.add_argument("golden", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--expected-golden-sha256", default=FROZEN_REFERENCE_SHA256)
    compare.add_argument("--absolute-tolerance", type=_non_negative_float, default=0.0)
    compare.add_argument("--relative-tolerance", type=_non_negative_float, default=0.0)
    compare.add_argument("--maximum-ulp", type=_non_negative_int, default=0)
    compare.add_argument("--require-byte-exact", action="store_true")
    compare.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "generate":
            cache_job_directory = arguments.cache_job_directory.absolute()
            output = _require_output_outside_cache(
                arguments.output, cache_job_directory
            )
            document = generate_reference_document(cache_job_directory)
            _write(document, output)
            print(f"phase-space golden sha256 = {hashlib.sha256(_canonical_bytes(document)).hexdigest()}")
            print(f"manifest sha256 = {document['manifest']['manifestSha256']}")
            return 0
        if arguments.command == "verify":
            document = load_reference_document(
                arguments.golden.absolute(), expected_sha256=arguments.expected_sha256
            )
            print(f"verified manifest sha256 = {document['manifest']['manifestSha256']}")
            return 0
        if arguments.command == "make-reference-candidate":
            golden = load_reference_document(
                arguments.golden.absolute(), expected_sha256=arguments.expected_sha256
            )
            candidate = candidate_document_from_reference(
                golden,
                golden_sha256=arguments.expected_sha256,
            )
            _write(candidate, arguments.output.absolute())
            return 0
        if arguments.command == "compare":
            report = compare_files(
                arguments.golden.absolute(),
                arguments.candidate.absolute(),
                expected_golden_sha256=arguments.expected_golden_sha256,
                absolute_tolerance=arguments.absolute_tolerance,
                relative_tolerance=arguments.relative_tolerance,
                maximum_ulp=arguments.maximum_ulp,
                require_byte_exact=arguments.require_byte_exact,
            )
            if arguments.output is None:
                sys.stdout.buffer.write(_canonical_bytes(report))
            else:
                _write(report, arguments.output.absolute())
            return 0 if report["qualified"] else 2
        raise AssertionError("unhandled command")
    except (OSError, PhaseSpaceGoldenError, gc.GoldenCacheError, ValueError) as error:
        print(f"phase-space golden failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
