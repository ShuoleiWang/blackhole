"""Typed composition boundary between offline rays and radiative transfer.

Geodesics are traced from the observer toward the source, while physical
radiative transfer propagates from the source toward the observer.  This module
owns that reversal explicitly so a caller cannot accidentally apply absorbing
layers in the wrong order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from offline.geodesic import HamiltonianState, RayTraceResult
from offline.radiative_transfer import (
    StokesInvariant,
    TransferCoefficients,
    TransferResult,
    TransferSegment,
    propagate_source_to_observer,
)
from offline.spacetime import Vector4


@runtime_checkable
class AffineMediumProvider(Protocol):
    """Scalar coefficients sampled on a past-traced ray midpoint.

    Implementations receive the past-directed covector stored by the geodesic
    tracer.  For a future-directed fluid four-velocity, ``u^mu p_mu`` is the
    positive comoving photon-frequency ratio when the camera ray is normalized
    to unit observer frequency.  Returned coefficients must use the invariant
    affine convention documented by :mod:`offline.radiative_transfer` and be
    constant over the accepted segment represented by the midpoint.  This
    bridge intentionally rejects polarized emissivity, dichroism, and Faraday
    terms until the geodesic record carries a parallel-transported screen
    basis and its own convergence evidence.
    """

    source_id: str

    def coefficients(
        self,
        state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> TransferCoefficients:
        """Return invariant coefficients at one ray midpoint."""


@runtime_checkable
class BoundarySpectrum(Protocol):
    """Source-boundary invariant Stokes spectrum."""

    source_id: str

    def invariant_stokes(
        self,
        terminal_state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> StokesInvariant:
        """Return boundary ``(I,Q,U,V)/nu^3`` for one observer bin."""


@dataclass(frozen=True)
class VacuumMedium:
    source_id: str = "vacuum"

    def coefficients(
        self,
        state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> TransferCoefficients:
        if not math.isfinite(observer_frequency_hz) or observer_frequency_hz <= 0.0:
            raise ValueError("observer frequency must be finite and positive")
        return TransferCoefficients()


@dataclass(frozen=True)
class DarkBoundary:
    source_id: str = "dark-boundary"

    def invariant_stokes(
        self,
        terminal_state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> StokesInvariant:
        if not math.isfinite(observer_frequency_hz) or observer_frequency_hz <= 0.0:
            raise ValueError("observer frequency must be finite and positive")
        return StokesInvariant()


@dataclass(frozen=True)
class RaySpectrumResult:
    observer_frequencies_hz: tuple[float, ...]
    transfers: tuple[TransferResult, ...]


def past_directed_comoving_frequency_ratio(
    state: HamiltonianState,
    fluid_four_velocity: Vector4,
) -> float:
    """Return ``u^mu p_mu > 0`` for the stored past-directed covector."""
    if not all(math.isfinite(value) for value in fluid_four_velocity):
        raise ValueError("fluid four-velocity must be finite")
    ratio = math.fsum(
        fluid_four_velocity[index] * state.covector[index]
        for index in range(4)
    )
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("comoving photon frequency ratio must be positive")
    return ratio


def source_to_observer_segments(
    ray: RayTraceResult,
    medium: AffineMediumProvider,
    observer_frequency_hz: float,
) -> tuple[TransferSegment, ...]:
    """Convert observer-traced segments to physical propagation order."""
    if not isinstance(medium, AffineMediumProvider):
        raise TypeError("medium must implement AffineMediumProvider")
    if not math.isfinite(observer_frequency_hz) or observer_frequency_hz <= 0.0:
        raise ValueError("observer frequency must be finite and positive")
    if not math.isfinite(ray.affine_length) or ray.affine_length < 0.0:
        raise ValueError("ray affine length must be finite and non-negative")
    if ray.outcome not in {"captured", "escaped", "completed"}:
        raise ValueError(
            f"ray outcome {ray.outcome!r} is not usable for radiative transfer"
        )
    if ray.failure_reason is not None:
        raise ValueError("failed ray may not enter radiative transfer")
    if ray.outcome in {"captured", "escaped"} and not ray.terminal_target_id:
        raise ValueError("terminal ray outcome requires a termination target id")
    if ray.affine_length > 0.0 and not ray.segments:
        raise ValueError("ray path was not recorded; trace with record_path=True")
    if ray.segments:
        if any(segment.affine_length <= 0.0 for segment in ray.segments):
            raise ValueError("recorded ray contains a non-positive segment")
        if any(
            previous.end != current.start
            for previous, current in zip(ray.segments, ray.segments[1:])
        ):
            raise ValueError("recorded ray segments are not contiguous")
        if ray.segments[-1].end != ray.terminal_state:
            raise ValueError("recorded ray does not end at its terminal state")
        recorded_length = math.fsum(
            segment.affine_length for segment in ray.segments
        )
        if not math.isclose(
            recorded_length,
            ray.affine_length,
            rel_tol=2.0e-13,
            abs_tol=2.0e-13,
        ):
            raise ValueError("recorded ray segments do not cover its affine length")

    result: list[TransferSegment] = []
    for segment in reversed(ray.segments):
        coefficients = medium.coefficients(
            segment.midpoint,
            observer_frequency_hz,
        )
        if (
            coefficients.invariant_emissivity.q != 0.0
            or coefficients.invariant_emissivity.u != 0.0
            or coefficients.invariant_emissivity.v != 0.0
            or coefficients.invariant_dichroism != (0.0, 0.0, 0.0)
            or coefficients.invariant_faraday != (0.0, 0.0, 0.0)
        ):
            raise ValueError(
                "polarized transfer requires a parallel-transported screen basis"
            )
        result.append(
            TransferSegment(
                length=segment.affine_length,
                coefficients=coefficients,
            )
        )
    return tuple(result)


def propagate_recorded_ray_spectrum(
    ray: RayTraceResult,
    observer_frequencies_hz: Sequence[float],
    medium: AffineMediumProvider,
    boundary: BoundarySpectrum,
    *,
    maximum_transfer_steps: int = 100_000,
    maximum_step_matrix_norm: float = 0.25,
    transfer_absolute_tolerance: float = 1.0e-10,
    transfer_relative_tolerance: float = 1.0e-5,
) -> RaySpectrumResult:
    """Propagate all declared observer-frequency bins along one recorded ray."""
    if not isinstance(boundary, BoundarySpectrum):
        raise TypeError("boundary must implement BoundarySpectrum")
    frequencies = tuple(float(value) for value in observer_frequencies_hz)
    if not frequencies:
        raise ValueError("at least one observer frequency is required")
    if any(not math.isfinite(value) or value <= 0.0 for value in frequencies):
        raise ValueError("observer frequencies must be finite and positive")
    if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
        raise ValueError("observer frequencies must be strictly increasing")

    transfers = []
    for frequency in frequencies:
        if ray.outcome == "captured":
            # A past-traced null ray terminating on a classical horizon has no
            # outgoing source boundary in this foundation.  Horizon/plasma
            # emission belongs in a future explicit medium model, not an
            # arbitrary BoundarySpectrum callback.
            source = StokesInvariant()
        else:
            source = boundary.invariant_stokes(ray.terminal_state, frequency)
        if source.q != 0.0 or source.u != 0.0 or source.v != 0.0:
            raise ValueError(
                "polarized boundary transfer requires a parallel-transported "
                "screen basis"
            )
        segments = source_to_observer_segments(ray, medium, frequency)
        transfers.append(
            propagate_source_to_observer(
                source,
                segments,
                maximum_steps=maximum_transfer_steps,
                maximum_step_matrix_norm=maximum_step_matrix_norm,
                absolute_tolerance=transfer_absolute_tolerance,
                relative_tolerance=transfer_relative_tolerance,
            )
        )
    return RaySpectrumResult(frequencies, tuple(transfers))


__all__ = (
    "AffineMediumProvider",
    "BoundarySpectrum",
    "DarkBoundary",
    "RaySpectrumResult",
    "VacuumMedium",
    "past_directed_comoving_frequency_ratio",
    "propagate_recorded_ray_spectrum",
    "source_to_observer_segments",
)
