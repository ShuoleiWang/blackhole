"""Opt-in strict-CPU returning-ray primitive and narrow forward transport.

Python retains all model validation, launch construction, fine/coarse policy,
receiver physics, convergence gates, descriptors, and cache document ownership.
Only each independently declared whole-ray resolution crosses the authenticated
ABI-v3 boundary.
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.kerr import KerrOblateTermination
from offline.kerr_finite_thickness import LOWER, UPPER
from offline.kerr_finite_thickness_emitter import KerrFiniteThicknessFaceEmitter
from offline.kerr_finite_thickness_launch import (
    KerrFiniteThicknessEmissionLaunch,
    KerrFiniteThicknessSurfaceFrame,
)
from offline.kerr_finite_thickness_surface import (
    LOWER_SURFACE_ID,
    UPPER_SURFACE_ID,
    KerrFiniteThicknessMultiSurface,
)
from offline.kerr_native_cpu_backend import (
    KerrNativeCpuRuntimeBinding,
    KerrNativeCpuOneResolution,
    StrictKerrCpuBackend,
    consume_kerr_native_cpu_runtime_binding,
)
import offline.kerr_returning_radiation_kernel as _forward
import offline.kerr_returning_radiation_rays as _rays

if TYPE_CHECKING:
    from offline.kerr_returning_radiation_kernel_jobs import (
        KerrKernelDirectionCoordinate,
        KerrKernelScientificContext,
    )


IMPLEMENTATION_ID = "finite-thickness-kerr-returning-ray/native-cpu-abi-v3"


def _require_completed_resolution(
    result: KerrNativeCpuOneResolution,
    *,
    normalized_launch: KerrFiniteThicknessEmissionLaunch,
    surface: KerrFiniteThicknessMultiSurface,
    surface_options: SurfaceEventOptions,
) -> None:
    ray = result.ray
    if ray.failure_reason is not None or ray.outcome in (
        "integrator-failure",
        "unresolved",
        "completed",
    ):
        raise _rays.KerrReturningRadiationRayError(
            "future native returning ray did not reach a certified fate: "
            f"{ray.outcome}: {ray.failure_reason or 'no terminal worldtube'}"
        )
    trace = ray.multi_surface_trace
    if trace is None or trace.initial_contact is None:
        raise _rays.KerrReturningRadiationRayError(
            "future native returning ray lost its authenticated initial contact"
        )
    emitting_surface_id = (
        UPPER_SURFACE_ID
        if normalized_launch.frame.emitter.face == UPPER
        else LOWER_SURFACE_ID
    )
    residual = surface.value(emitting_surface_id, normalized_launch.future_state)
    contact = trace.initial_contact
    if (
        contact.surface_id != emitting_surface_id
        or contact.side != 1
        or contact.actual_surface_value.hex() != float(residual).hex()
        or contact.surface_value_tolerance.hex()
        != surface_options.surface_value_tolerance.hex()
    ):
        raise _rays.KerrReturningRadiationRayError(
            "future native returning ray initial-contact provenance is stale"
        )


def trace_kerr_returning_radiation_direction_native_cpu(
    launch: KerrFiniteThicknessEmissionLaunch,
    surface: KerrFiniteThicknessMultiSurface,
    *,
    backend: StrictKerrCpuBackend,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
) -> _rays.KerrReturningRadiationRayPrimitive:
    """Build the exact public primitive from two native whole-ray resolutions."""

    if type(backend) is not StrictKerrCpuBackend:
        raise TypeError("backend must be the exact StrictKerrCpuBackend")
    backend.revalidate()
    surface = _rays._validated_surface(surface)
    launch = _rays._validated_launch(launch, surface)
    normalized_launch = KerrFiniteThicknessEmissionLaunch(
        launch.frame,
        launch.emission_angle_cosine,
        launch.tangent_azimuth_rad,
        1.0,
    )
    termination = _rays._validated_termination(
        termination,
        surface,
        normalized_launch.future_state,
    )
    fine_ray_options, fine_surface_options = _rays._validated_options(
        surface.metric,
        ray_options,
        surface_options,
    )
    _rays._validated_fine_options(
        surface.metric,
        fine_ray_options,
        fine_surface_options,
    )
    if (coarse_ray_options is None) != (coarse_surface_options is None):
        raise ValueError(
            "coarse_ray_options and coarse_surface_options must be supplied together"
        )
    if coarse_ray_options is None:
        coarse_ray_options, coarse_surface_options = _rays._derive_coarse_options(
            surface.metric,
            fine_ray_options,
            fine_surface_options,
        )
    assert coarse_surface_options is not None
    coarse_ray_options, coarse_surface_options = _rays._validated_options(
        surface.metric,
        coarse_ray_options,
        coarse_surface_options,
    )
    _rays._validate_coarse_relationship(
        fine_ray_options,
        fine_surface_options,
        coarse_ray_options,
        coarse_surface_options,
    )

    emitting_face = normalized_launch.frame.emitter.face
    emitting_surface_id = (
        UPPER_SURFACE_ID if emitting_face == UPPER else LOWER_SURFACE_ID
    )
    initial_contact_residual = surface.value(
        emitting_surface_id,
        normalized_launch.future_state,
    )
    if abs(initial_contact_residual) > _rays._INITIAL_CONTACT_MAXIMUM_RESIDUAL:
        raise _rays.KerrReturningRadiationRayError(
            "authenticated launch is not on its declared emitting face"
        )

    fine_native = backend.trace_one_resolution(
        normalized_launch.future_state,
        surface,
        termination,
        fine_ray_options,
        fine_surface_options,
        emitting_face=emitting_face,
    )
    _require_completed_resolution(
        fine_native,
        normalized_launch=normalized_launch,
        surface=surface,
        surface_options=fine_surface_options,
    )
    coarse_native = backend.trace_one_resolution(
        normalized_launch.future_state,
        surface,
        termination,
        coarse_ray_options,
        coarse_surface_options,
        emitting_face=emitting_face,
    )
    _require_completed_resolution(
        coarse_native,
        normalized_launch=normalized_launch,
        surface=surface,
        surface_options=coarse_surface_options,
    )

    fine = _rays._evaluate_returning_ray(
        fine_native.ray,
        surface,
        termination,
        fine_ray_options,
        fine_surface_options,
    )
    coarse = _rays._evaluate_returning_ray(
        coarse_native.ray,
        surface,
        termination,
        coarse_ray_options,
        coarse_surface_options,
    )
    convergence = _rays._compare_whole_rays(fine, coarse, surface.metric)
    return _rays._build_result(
        launch=launch,
        normalized_launch=normalized_launch,
        surface=surface,
        termination=termination,
        fine_ray_options=fine_ray_options,
        fine_surface_options=fine_surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        fine=fine,
        coarse=coarse,
        convergence=convergence,
        fine_initial_contact_residual=initial_contact_residual,
        coarse_initial_contact_residual=initial_contact_residual,
    )


(
    _trace_issued_kerr_returning_radiation_direction_native_cpu,
    _consume_issued_kerr_returning_radiation_direction_native_cpu,
) = _rays._build_process_local_issued_primitive_capability(
    trace_kerr_returning_radiation_direction_native_cpu
)


def trace_forward_direction_native_cpu(
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions | None,
    coarse_surface_options: SurfaceEventOptions | None,
    source_face: str,
    source_radius_over_mass: float,
    emission_angle_cosine: float,
    tangent_azimuth_rad: float,
    *,
    backend: StrictKerrCpuBackend,
) -> Any:
    """Return the existing narrow internal forward transport type."""

    if type(source_face) is not str or source_face not in (LOWER, UPPER):
        raise ValueError("source_face must be exact 'lower' or 'upper'")
    emitter = KerrFiniteThicknessFaceEmitter(
        metric=surface.metric,
        calibration=surface.calibration,
        pseudo_cylindrical_radius_over_mass=source_radius_over_mass,
        face=source_face,
    )
    launch = KerrFiniteThicknessEmissionLaunch(
        KerrFiniteThicknessSurfaceFrame(emitter),
        emission_angle_cosine,
        tangent_azimuth_rad,
        1.0,
    )
    primitive, issue_token = (
        _trace_issued_kerr_returning_radiation_direction_native_cpu(
            launch,
            surface,
            backend=backend,
            termination=termination,
            ray_options=ray_options,
            surface_options=surface_options,
            coarse_ray_options=coarse_ray_options,
            coarse_surface_options=coarse_surface_options,
        )
    )
    issued = _consume_issued_kerr_returning_radiation_direction_native_cpu(
        primitive,
        issue_token,
    )
    fate = issued.fate
    if fate.startswith("return-"):
        receiver_face = issued.receiver_face
        receiver_radius = issued.receiver_radius_over_mass
        ratio = issued.emitter_to_receiver_frequency_ratio
        if (
            type(receiver_face) is not str
            or receiver_face not in (LOWER, UPPER)
            or type(receiver_radius) is not float
            or type(ratio) is not float
            or receiver_radius <= 0.0
            or ratio <= 0.0
        ):
            raise _forward.KerrReturningRadiationKernelError(
                "native returned primitive has invalid receiver data"
            )
        coarse_face = issued.coarse_receiver_face
        coarse_radius = issued.coarse_receiver_radius_over_mass
        if type(coarse_face) is not str or type(coarse_radius) is not float:
            raise _forward.KerrReturningRadiationKernelError(
                "native coarse returned ray lacks receiver evidence"
            )
        g2 = ratio * ratio
        if not math.isfinite(g2) or g2 <= 0.0:
            raise _forward.KerrReturningRadiationKernelError(
                "native returned g^2 is invalid"
            )
        return _forward._DirectionTransport(
            fate,
            receiver_face,
            receiver_radius,
            ratio,
            g2,
            issued.primitive_descriptor_sha256,
            coarse_face,
            coarse_radius,
        )
    return _forward._DirectionTransport(
        fate,
        None,
        None,
        None,
        0.0,
        issued.primitive_descriptor_sha256,
        None,
        None,
    )


def consume_forward_direction_native_cpu_runtime_binding(
    runtime_binding: KerrNativeCpuRuntimeBinding,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions | None,
    coarse_surface_options: SurfaceEventOptions | None,
    source_face: str,
    source_radius_over_mass: float,
    emission_angle_cosine: float,
    tangent_azimuth_rad: float,
) -> Any:
    """Consume a path-free binding without any cache-node/document ownership."""

    with consume_kerr_native_cpu_runtime_binding(runtime_binding) as backend:
        return trace_forward_direction_native_cpu(
            surface,
            termination,
            ray_options,
            surface_options,
            coarse_ray_options,
            coarse_surface_options,
            source_face,
            source_radius_over_mass,
            emission_angle_cosine,
            tangent_azimuth_rad,
            backend=backend,
        )


def evaluate_forward_direction_native_cpu(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
    *,
    backend: StrictKerrCpuBackend,
) -> dict[str, Any]:
    """Evaluate one canonical coordinate without selecting or writing a cache."""

    import offline.kerr_returning_radiation_kernel_cached as _cached
    from offline.kerr_returning_radiation_kernel_jobs import (
        FORWARD,
        KerrKernelDirectionCoordinate,
        KerrKernelScientificContext,
    )

    if type(context) is not KerrKernelScientificContext:
        raise TypeError("context must be the exact KerrKernelScientificContext")
    if type(coordinate) is not KerrKernelDirectionCoordinate:
        raise TypeError("coordinate must be the exact KerrKernelDirectionCoordinate")
    node = _cached._forward_coordinate_node(context, coordinate)
    identity = context.identity
    if identity.formulation != FORWARD:
        raise ValueError("native forward helper requires a forward context")
    transport = trace_forward_direction_native_cpu(
        identity.surface,
        identity.termination,
        identity.fine_ray_options,
        identity.fine_surface_options,
        identity.coarse_ray_options,
        identity.coarse_surface_options,
        node.source_face,
        node.source_radius_over_mass,
        node.emission_angle_cosine,
        node.tangent_azimuth_rad,
        backend=backend,
    )
    return _cached._forward_transport_document(
        node,
        transport,
        identity.annulus_edges_over_mass,
    )


__all__ = (
    "IMPLEMENTATION_ID",
    "consume_forward_direction_native_cpu_runtime_binding",
    "evaluate_forward_direction_native_cpu",
    "trace_forward_direction_native_cpu",
    "trace_kerr_returning_radiation_direction_native_cpu",
)
