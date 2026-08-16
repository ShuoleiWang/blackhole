"""Opaque, process-local authentication of one deterministic ray replay.

The certificate in this module is deliberately *not* a serialized proof.  It
is a short-lived capability used only to avoid repeating the same expensive
geodesic integration inside one trusted call stack.  Authority comes from a
closure-private weak registry, not from fields carried by the token.  Public
transfer/result entry points still obtain authority by performing a complete
deterministic replay at least once.

Only geometry replay is covered.  Spectrum, source, frequency-shift,
photosphere, topology, and provenance validators remain the responsibility of
the transfer result and must run again when a certificate is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from threading import RLock
from typing import Any
from weakref import WeakKeyDictionary

from offline.geodesic import (
    HamiltonianState,
    RayTraceOptions,
    RayTraceResult,
    SurfaceEventOptions,
    trace_null_geodesic,
)
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_finite_thickness import (
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_surface import (
    KerrFiniteThicknessMultiSurface,
)


class ReplayCertificateError(ValueError):
    """Raised when a replay certificate is absent, stale, or misbound."""


class _ReplayCertificate:
    """Fieldless registry key that cannot be constructed through its API."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *args: object, **kwargs: object) -> "_ReplayCertificate":
        del args, kwargs
        raise TypeError("replay certificates can only be issued by a full replay")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("replay certificates may not be subclassed")

    def __copy__(self) -> "_ReplayCertificate":
        raise TypeError("replay certificates may not be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> "_ReplayCertificate":
        del memo
        raise TypeError("replay certificates may not be copied")

    def __reduce__(self) -> object:
        raise TypeError("replay certificates may not be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("replay certificates may not be serialized")


@dataclass(frozen=True, slots=True)
class _ReplayRecord:
    surface: KerrFiniteThicknessMultiSurface
    metric: KerrKerrSchildMetric
    calibration: StationaryKerrFiniteThicknessCalibration
    termination: KerrOblateTermination
    observer_initial_state: HamiltonianState
    claimed_ray: RayTraceResult
    ray_options: RayTraceOptions
    surface_options: SurfaceEventOptions
    canonical_replay: RayTraceResult
    context_snapshot: tuple[object, ...]


def _immutable_snapshot(value: object) -> object:
    """Freeze the small replay context without trusting dataclass equality."""

    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is tuple:
        return tuple(_immutable_snapshot(entry) for entry in value)
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple(
                (field.name, _immutable_snapshot(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    raise TypeError(
        "deterministic replay context contains an unsupported mutable value"
    )


def _validate_context_types(
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    observer_initial_state: HamiltonianState,
    ray: RayTraceResult,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
) -> None:
    if type(surface) is not KerrFiniteThicknessMultiSurface:
        raise TypeError("replay surface must be the exact finite-thickness adapter")
    if type(surface.metric) is not KerrKerrSchildMetric:
        raise TypeError("replay metric must be the exact built-in Kerr metric")
    if type(surface.calibration) is not StationaryKerrFiniteThicknessCalibration:
        raise TypeError(
            "replay calibration must be the exact finite-thickness calibration"
        )
    if type(termination) is not KerrOblateTermination:
        raise TypeError("replay termination must be the exact Kerr provider")
    if type(observer_initial_state) is not HamiltonianState:
        raise TypeError("replay initial state must be the exact HamiltonianState")
    if type(ray) is not RayTraceResult:
        raise TypeError("replay ray must be the exact RayTraceResult")
    if type(ray_options) is not RayTraceOptions:
        raise TypeError("replay ray options must be the exact RayTraceOptions")
    if type(surface_options) is not SurfaceEventOptions:
        raise TypeError(
            "replay surface options must be the exact SurfaceEventOptions"
        )


def _context_snapshot(
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    observer_initial_state: HamiltonianState,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
) -> tuple[object, ...]:
    return (
        _immutable_snapshot(surface),
        _immutable_snapshot(surface.metric),
        _immutable_snapshot(surface.calibration),
        _immutable_snapshot(termination),
        _immutable_snapshot(observer_initial_state),
        _immutable_snapshot(ray_options),
        _immutable_snapshot(surface_options),
    )


def _make_certificate_authority():
    registry: WeakKeyDictionary[_ReplayCertificate, _ReplayRecord] = (
        WeakKeyDictionary()
    )
    lock = RLock()

    def issue(
        surface: KerrFiniteThicknessMultiSurface,
        termination: KerrOblateTermination,
        observer_initial_state: HamiltonianState,
        ray: RayTraceResult,
        ray_options: RayTraceOptions,
        surface_options: SurfaceEventOptions,
    ) -> _ReplayCertificate:
        _validate_context_types(
            surface,
            termination,
            observer_initial_state,
            ray,
            ray_options,
            surface_options,
        )
        before = _context_snapshot(
            surface,
            termination,
            observer_initial_state,
            ray_options,
            surface_options,
        )
        try:
            replayed = trace_null_geodesic(
                surface.metric,
                observer_initial_state,
                termination=termination,
                multi_interior_surface=surface,
                surface_options=surface_options,
                options=ray_options,
            )
        except (ArithmeticError, RuntimeError, TypeError, ValueError) as error:
            raise ReplayCertificateError(
                f"deterministic finite-thickness ray replay failed: {error}"
            ) from error
        after = _context_snapshot(
            surface,
            termination,
            observer_initial_state,
            ray_options,
            surface_options,
        )
        if after != before:
            raise ReplayCertificateError(
                "deterministic replay context changed during authentication"
            )
        if replayed != ray:
            raise ReplayCertificateError(
                "input ray is not exactly equal to deterministic first-visible replay"
            )
        certificate = object.__new__(_ReplayCertificate)
        record = _ReplayRecord(
            surface=surface,
            metric=surface.metric,
            calibration=surface.calibration,
            termination=termination,
            observer_initial_state=observer_initial_state,
            claimed_ray=ray,
            ray_options=ray_options,
            surface_options=surface_options,
            canonical_replay=replayed,
            context_snapshot=after,
        )
        with lock:
            registry[certificate] = record
        return certificate

    def require(
        certificate: object,
        surface: KerrFiniteThicknessMultiSurface,
        termination: KerrOblateTermination,
        observer_initial_state: HamiltonianState,
        ray: RayTraceResult,
        ray_options: RayTraceOptions,
        surface_options: SurfaceEventOptions,
    ) -> None:
        _validate_context_types(
            surface,
            termination,
            observer_initial_state,
            ray,
            ray_options,
            surface_options,
        )
        if type(certificate) is not _ReplayCertificate:
            raise ReplayCertificateError("replay certificate type is not authentic")
        with lock:
            record = registry.get(certificate)
        if record is None:
            raise ReplayCertificateError("replay certificate was not issued here")
        if (
            surface is not record.surface
            or surface.metric is not record.metric
            or surface.calibration is not record.calibration
            or termination is not record.termination
            or observer_initial_state is not record.observer_initial_state
            or ray is not record.claimed_ray
            or ray_options is not record.ray_options
            or surface_options is not record.surface_options
        ):
            raise ReplayCertificateError("replay certificate is bound to other inputs")
        current = _context_snapshot(
            surface,
            termination,
            observer_initial_state,
            ray_options,
            surface_options,
        )
        if current != record.context_snapshot:
            raise ReplayCertificateError("replay certificate context is stale")
        if ray != record.canonical_replay:
            raise ReplayCertificateError("replay certificate ray is stale")

    return issue, require


_issue_replay_certificate, _require_replay_certificate = (
    _make_certificate_authority()
)
del _make_certificate_authority


# Deliberately no public API: the transfer module owns this process-local
# optimization boundary.  Private names remain importable for white-box tests,
# but importing the issuer never creates a certificate without a full replay.
__all__: tuple[str, ...] = ()
