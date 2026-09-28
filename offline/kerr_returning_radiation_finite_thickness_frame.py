"""Certified fine/coarse returning-thermal finite-thickness rays and pixels.

``KerrFiniteThicknessRaySampler`` remains the sole owner of the camera,
independent fine/coarse exact-Kerr traces, accepted-step multi-surface
topology, and all ray/surface convergence validation.  This module composes
that exact sampler and replaces only its equatorial Novikov--Thorne emission
with the frozen, authenticated returning-radiation thermal table.

For a disk hit, convergence additionally requires the fine and coarse rays to
land in the same piecewise-constant thermal annulus and to remain far enough
from every internal annulus edge.  The required clearance is

``multiplier * max(|delta rho|, declared rho tolerance, binary64 floor)``.

The standalone ``sample`` method checks the live authority exactly at its two
boundaries.  ``integrate_returning_thermal_spectral_pixel`` checks it exactly
at the whole-pixel boundaries and uses the private trusted hot path for every
adaptive ray; no per-ray live provider/kernel replay is introduced.

This remains a same-code, piecewise-annulus model.  It is not a continuum
radial returning-radiation solution, complete KERRBB, an independent physics
oracle, a returning-radiation stress/work ``F_S`` calculation, a solved or
scattering atmosphere, polarization transport, GRMHD, or a Sachs/Jacobi ray
bundle.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field, fields
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

from offline.adaptive_frame import (
    AdaptivePixelOptions,
    AdaptivePixelResult,
    RayConvergenceAudit,
    SpectralRaySample,
    integrate_spectral_pixel,
)
from offline.disk_atmosphere import FluxConservingLinearLimbDarkening
from offline.geodesic import RayTraceOptions, RayTraceResult, SurfaceEventOptions
from offline.kerr import kerr_zamo_camera_ray
import offline.kerr_finite_thickness_frame as base_frame_module
from offline.kerr_finite_thickness_frame import KerrFiniteThicknessRaySampler
from offline.kerr_finite_thickness_surface import OPAQUE_OUTCOME
from offline.kerr_finite_thickness_transfer import KerrFiniteThicknessSpectrumResult
from offline.kerr_returning_radiation_frame_context import (
    ReturningThermalEmissionSnapshotV1,
    ValidatedReturningThermalAuthority,
)
import offline.kerr_returning_radiation_finite_thickness_transfer as transfer_module
from offline.kerr_returning_radiation_finite_thickness_transfer import (
    KerrReturningRadiationFiniteThicknessSpectrumResult,
)


IMPLEMENTATION_ID: Final = (
    "kerr-returning-radiation-finite-thickness-spectral-ray-sampler/v1"
)
MAXIMUM_ANNULUS_EDGE_CLEARANCE_MULTIPLIER: Final = 64.0
BINARY64_RADIUS_FLOOR_ULPS: Final = 32
_TRUSTED_TRANSFER_ENTRY: Final = (
    transfer_module
    ._transfer_kerr_returning_radiation_finite_thickness_spectrum_trusted
)
_TRANSFER_RUNTIME_GATE_ENTRY: Final = (
    transfer_module._require_trusted_runtime_bindings
)
_FRAME_D20_CLASS: Final = FluxConservingLinearLimbDarkening
_FRAME_D20_INTENSITY_ENTRY: Final = (
    FluxConservingLinearLimbDarkening.intensity_multiplier
)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "same-code authenticated, fine/coarse converged returning-thermal "
            "finite-thickness scalar ray and adaptive pixel"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "geometryAndTraceOwner": (
            "offline.kerr_finite_thickness_frame.KerrFiniteThicknessRaySampler"
        ),
        "emissionOwner": "ValidatedReturningThermalAuthority frozen table",
        "fineCoarseWholeRayConvergence": True,
        "sameAnnulusFineCoarseRequired": True,
        "internalAnnulusEdgeClearanceRequired": True,
        "singleAnnulusEdgeClearance": "not-applicable-no-internal-edge",
        "thermalRadialDiscretization": (
            "piecewise-constant authenticated F_out and T_eff per annulus"
        ),
        "isContinuumRadialReturningRadiationSolution": False,
        "captureBoundary": "exact positive-sign zero",
        "escapeBoundary": "closed built-in observer-frame spectrum",
        "pixelAuthorityChecks": "exactly whole-pixel start and end",
        "perRayPixelAuthorityReplay": False,
        "trustedTransferCallableIdentityFrozenAtModuleLoad": True,
        "hasIndependentPhysicsOracle": False,
        "isCompleteKerrbb": False,
        "includesReturningRadiationStressWorkFS": False,
        "includesSpectralRedistribution": False,
        "includesScatteringOrSolvedAtmosphere": False,
        "includesPolarization": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "isSachsJacobiRayBundle": False,
        "prohibitedClaim": (
            "Do not describe this same-code piecewise-annulus calculation as "
            "a continuum returning-radiation solution, independent oracle, "
            "complete KERRBB, F_S, a solved atmosphere, polarization, GRMHD, "
            "or a Sachs/Jacobi ray bundle."
        ),
    }
)


class KerrReturningRadiationFiniteThicknessFrameError(RuntimeError):
    """Raised when a returning-thermal ray or pixel fails closed."""


def _canonical_json(value: Any, label: str) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise KerrReturningRadiationFiniteThicknessFrameError(
            f"{label} is not finite canonical JSON"
        ) from error


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _exact_multiplier(value: Any) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError("annulus_edge_clearance_multiplier must be an exact float")
    if value < 1.0:
        raise ValueError("annulus edge-clearance multiplier must be at least one")
    if value > MAXIMUM_ANNULUS_EDGE_CLEARANCE_MULTIPLIER:
        raise ValueError("annulus edge-clearance multiplier exceeds policy maximum")
    return value


def _exact_nonnegative(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise KerrReturningRadiationFiniteThicknessFrameError(
            f"{label} must be a finite non-negative exact float"
        )
    return value


def _exact_positive(value: Any, label: str) -> float:
    result = _exact_nonnegative(value, label)
    if result <= 0.0:
        raise KerrReturningRadiationFiniteThicknessFrameError(
            f"{label} must be positive"
        )
    return result


def _exact_digest(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise KerrReturningRadiationFiniteThicknessFrameError(
            f"{label} must be an exact lowercase SHA-256 string"
        )
    return value


def _snapshot_identity(snapshot: ReturningThermalEmissionSnapshotV1) -> dict[str, Any]:
    return {
        "axisymmetricKernelDescriptorSha256": (
            snapshot.axisymmetric_kernel_descriptor_sha256
        ),
        "descriptorSha256": snapshot.model_descriptor_sha256,
        "novikovThorneDiskDescriptorSha256": (
            snapshot.novikov_thorne_disk_descriptor_sha256
        ),
        "profileDescriptorSha256": snapshot.profile_descriptor_sha256,
        "providerDescriptorSha256": snapshot.provider_descriptor_sha256,
        "sourceEvidenceDescriptorSha256": (
            snapshot.source_evidence_descriptor_sha256
        ),
        "sourceKernelDescriptorSha256": snapshot.source_kernel_descriptor_sha256,
        "sourceKernelKind": snapshot.source_kernel_kind,
        "surfaceIdentityDescriptorSha256": (
            snapshot.surface_identity_descriptor_sha256
        ),
        "underlyingKernelDescriptorSha256": (
            snapshot.underlying_kernel_descriptor_sha256
        ),
    }


def _annulus_evidence(
    snapshot: ReturningThermalEmissionSnapshotV1,
    radius_over_mass: float,
) -> tuple[int, float, float, float | None]:
    """Independently derive the piecewise bin and internal-edge clearance."""

    radius = _exact_positive(radius_over_mass, "disk-hit radius")
    edges = snapshot.annulus_edges_over_mass
    if type(edges) is not tuple or len(edges) != snapshot.annulus_count + 1:
        raise KerrReturningRadiationFiniteThicknessFrameError(
            "authority annulus edges are malformed"
        )
    if radius < edges[0] or radius > edges[-1]:
        raise KerrReturningRadiationFiniteThicknessFrameError(
            "disk-hit radius lies outside the authenticated thermal domain"
        )
    index = bisect_right(edges, radius) - 1
    if index == snapshot.annulus_count:
        index -= 1
    adjacent_internal_edges = (
        *((edges[index],) if index > 0 else ()),
        *((edges[index + 1],) if index + 1 < len(edges) - 1 else ()),
    )
    nearest = (
        None
        if not adjacent_internal_edges
        else min(abs(radius - edge) for edge in adjacent_internal_edges)
    )
    if nearest is not None and (not math.isfinite(nearest) or nearest < 0.0):
        raise KerrReturningRadiationFiniteThicknessFrameError(
            "independent internal-edge distance is invalid"
        )
    return index, edges[index], edges[index + 1], nearest


def _recompute_transfer_descriptor(
    result: KerrReturningRadiationFiniteThicknessSpectrumResult,
) -> tuple[str, str]:
    values = {
        item.name: object.__getattribute__(result, item.name)
        for item in fields(KerrReturningRadiationFiniteThicknessSpectrumResult)
    }
    descriptor = transfer_module._result_descriptor(values)
    descriptor_json = transfer_module._canonical_json(descriptor)
    return descriptor_json, transfer_module._sha256_text(descriptor_json)


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationFiniteThicknessRaySampler:
    """Compose the exact base sampler with authenticated thermal emission."""

    base_sampler: KerrFiniteThicknessRaySampler
    authority: ValidatedReturningThermalAuthority
    annulus_edge_clearance_multiplier: float = 4.0
    _authenticated_snapshot: ReturningThermalEmissionSnapshotV1 = field(
        init=False,
        repr=False,
        compare=False,
    )
    _base_descriptor_json: str = field(init=False, repr=False, compare=False)
    _base_descriptor_sha256: str = field(init=False, repr=False, compare=False)
    _authenticated_edge_clearance_multiplier: float = field(
        init=False,
        repr=False,
        compare=False,
    )
    _authenticated_edge_clearance_multiplier_hex: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init_subclass__(cls, **kwargs):
        del cls, kwargs
        raise TypeError(
            "KerrReturningRadiationFiniteThicknessRaySampler cannot be subclassed"
        )

    def __post_init__(self) -> None:
        if type(self) is not KerrReturningRadiationFiniteThicknessRaySampler:
            raise TypeError("returning-thermal sampler must have its exact type")
        if type(self.base_sampler) is not KerrFiniteThicknessRaySampler:
            raise TypeError("base_sampler must have its exact certified type")
        if type(self.authority) is not ValidatedReturningThermalAuthority:
            raise TypeError("authority must have its exact process-local type")
        multiplier = _exact_multiplier(self.annulus_edge_clearance_multiplier)
        snapshot = self.authority.require_live()
        if type(snapshot) is not ReturningThermalEmissionSnapshotV1:
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "authority returned a foreign snapshot type"
            )
        snapshot.revalidate()
        transfer_module._validate_snapshot_ownership(
            self.base_sampler.surface,
            self.base_sampler.disk,
            snapshot,
        )
        base_descriptor = self.base_sampler.descriptor()
        base_json = _canonical_json(base_descriptor, "base sampler descriptor")
        object.__setattr__(self, "annulus_edge_clearance_multiplier", multiplier)
        object.__setattr__(self, "_authenticated_snapshot", snapshot)
        object.__setattr__(self, "_base_descriptor_json", base_json)
        object.__setattr__(self, "_base_descriptor_sha256", _sha256_text(base_json))
        object.__setattr__(
            self,
            "_authenticated_edge_clearance_multiplier",
            multiplier,
        )
        object.__setattr__(
            self,
            "_authenticated_edge_clearance_multiplier_hex",
            multiplier.hex(),
        )

    def _assert_configuration_stable(self) -> None:
        if type(self) is not KerrReturningRadiationFiniteThicknessRaySampler:
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "returning-thermal sampler type changed"
            )
        public_multiplier = self.annulus_edge_clearance_multiplier
        frozen_multiplier = self._authenticated_edge_clearance_multiplier
        frozen_hex = self._authenticated_edge_clearance_multiplier_hex
        if (
            type(public_multiplier) is not float
            or type(frozen_multiplier) is not float
            or type(frozen_hex) is not str
            or public_multiplier.hex() != frozen_hex
            or frozen_multiplier.hex() != frozen_hex
        ):
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "annulus edge-clearance multiplier changed after authentication"
            )
        if (
            FluxConservingLinearLimbDarkening is not _FRAME_D20_CLASS
            or _FRAME_D20_CLASS.intensity_multiplier
            is not _FRAME_D20_INTENSITY_ENTRY
        ):
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "frame D20 angular-law callable changed after module load"
            )
        if (
            transfer_module._require_trusted_runtime_bindings
            is not _TRANSFER_RUNTIME_GATE_ENTRY
        ):
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "transfer runtime-binding gate changed after module load"
            )
        _TRANSFER_RUNTIME_GATE_ENTRY()
        if (
            KerrReturningRadiationFiniteThicknessRaySampler._sample_trusted
            is not _SAMPLER_TRUSTED_ENTRY
        ):
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "trusted frame-sampler callable changed after module load"
            )
        current = _canonical_json(
            self.base_sampler.descriptor(),
            "base sampler descriptor",
        )
        if current.encode("utf-8") != self._base_descriptor_json.encode("utf-8"):
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "base sampler descriptor changed after authentication"
            )
        if self.authority.snapshot is not self._authenticated_snapshot:
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "authority snapshot identity changed after authentication"
            )

    def descriptor(self) -> Mapping[str, Any]:
        """Return content-complete sampler and thermal-authority identity."""

        self._assert_configuration_stable()
        base_descriptor = json.loads(self._base_descriptor_json)
        base_actual = base_descriptor["convergencePolicy"]["actual"]
        base_maxima = base_descriptor["convergencePolicy"]["maxima"]
        descriptor = {
            "annulusEdgeClearancePolicy": {
                "actual": {
                    "baseDiskRadiusAbsoluteToleranceM": (
                        base_actual["diskRadiusAbsoluteToleranceM"]
                    ),
                    "baseDiskRadiusRelativeTolerance": (
                        base_actual["diskRadiusRelativeTolerance"]
                    ),
                    "binary64RadiusFloorUlps": BINARY64_RADIUS_FLOOR_ULPS,
                    "multiplier": (
                        self._authenticated_edge_clearance_multiplier
                    ),
                },
                "equation": (
                    "clearance>=multiplier*max(abs(delta_rho),"
                    "base_abs_rho+base_rel_rho*max_abs_rho,binary64_floor)"
                ),
                "maxima": {
                    "baseDiskRadiusAbsoluteToleranceM": (
                        base_maxima["diskRadiusAbsoluteToleranceM"]
                    ),
                    "baseDiskRadiusRelativeTolerance": (
                        base_maxima["diskRadiusRelativeTolerance"]
                    ),
                    "binary64RadiusFloorUlps": BINARY64_RADIUS_FLOOR_ULPS,
                    "multiplier": MAXIMUM_ANNULUS_EDGE_CLEARANCE_MULTIPLIER,
                },
                "singleAnnulus": "not-applicable-no-internal-edge",
            },
            "authority": _snapshot_identity(self._authenticated_snapshot),
            "baseSampler": {
                "descriptor": base_descriptor,
                "descriptorSha256": self._base_descriptor_sha256,
                "exactType": (
                    "offline.kerr_finite_thickness_frame."
                    "KerrFiniteThicknessRaySampler"
                ),
            },
            "fineCoarseConvergenceGates": {
                "allSpectralBins": True,
                "emittingFace": True,
                "frequencyShiftG": True,
                "internalAnnulusEdgeClearance": True,
                "multiSurfaceTopology": True,
                "pseudoCylindricalRadius": True,
                "samePiecewiseThermalAnnulus": True,
                "signedOutgoingEmissionCosine": True,
                "visibleSource": True,
                "actualTolerances": {
                    "diskRadiusAbsoluteToleranceM": (
                        base_actual["diskRadiusAbsoluteToleranceM"]
                    ),
                    "diskRadiusRelativeTolerance": (
                        base_actual["diskRadiusRelativeTolerance"]
                    ),
                    "emissionCosineAbsoluteTolerance": (
                        base_actual["emissionCosineAbsoluteTolerance"]
                    ),
                    "frequencyShiftRelativeTolerance": (
                        base_actual["frequencyShiftRelativeTolerance"]
                    ),
                    "specificIntensityAbsoluteTolerance": (
                        base_actual["specificIntensityAbsoluteTolerance"]
                    ),
                    "specificIntensityRelativeTolerance": (
                        base_actual["specificIntensityRelativeTolerance"]
                    ),
                },
            },
            "implementationId": IMPLEMENTATION_ID,
            "liveAuthorityBoundaryPolicy": {
                "adaptivePixel": "one require_live at start and one at end",
                "partialResultPublishedBeforeEndGate": False,
                "standaloneRay": "one require_live at start and one at end",
                "trustedPerRayHotPathCallsRequireLive": False,
            },
            "trustedTransferBoundary": {
                "callableIdentityFrozenAtModuleLoad": True,
                "implementationId": transfer_module.IMPLEMENTATION_ID,
                "sameCodeTrustBoundary": True,
            },
            "scientificStatus": dict(SCIENTIFIC_STATUS),
            "version": 1,
        }
        # Canonical round-trip also prevents a mutable/custom mapping from
        # escaping through either composed descriptor.
        return json.loads(_canonical_json(descriptor, "returning frame descriptor"))

    def _validate_transfer_result(
        self,
        result: KerrReturningRadiationFiniteThicknessSpectrumResult,
        ray: RayTraceResult,
        initial: Any,
        frequencies: tuple[float, ...],
        snapshot: ReturningThermalEmissionSnapshotV1,
        *,
        ray_options: RayTraceOptions,
        surface_options: SurfaceEventOptions,
    ) -> KerrReturningRadiationFiniteThicknessSpectrumResult:
        if type(result) is not KerrReturningRadiationFiniteThicknessSpectrumResult:
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "returning transfer result type is foreign"
            )
        base = object.__getattribute__(result, "base_result")
        if type(base) is not KerrFiniteThicknessSpectrumResult:
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "returning transfer lacks its exact base result"
            )
        sampler = self.base_sampler
        path_absolute = max(
            sampler.recorded_path_absolute_tolerance,
            ray_options.absolute_tolerance,
        )
        path_relative = max(
            sampler.recorded_path_relative_tolerance,
            ray_options.relative_tolerance,
        )
        if (
            type(result.observer_frequencies_hz) is not tuple
            or result.observer_frequencies_hz != frequencies
            or any(type(value) is not float for value in frequencies)
            or type(base.observer_frequencies_hz) is not tuple
            or base.observer_frequencies_hz != (frequencies[0],)
            or type(result.geometry_witness_observer_frequency_hz) is not float
            or result.geometry_witness_observer_frequency_hz.hex()
            != frequencies[0].hex()
            or type(result.source_kind) is not str
            or result.source_kind not in (
                "finite-thickness-disk",
                "captured-boundary",
                "escaped-boundary",
            )
        ):
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "returning transfer frequency witness or source type is malformed"
            )
        for name in (
            "authority_snapshot_descriptor_sha256",
            "provider_descriptor_sha256",
            "profile_descriptor_sha256",
            "axisymmetric_kernel_descriptor_sha256",
            "source_kernel_descriptor_sha256",
            "underlying_kernel_descriptor_sha256",
            "novikov_thorne_disk_descriptor_sha256",
            "surface_identity_descriptor_sha256",
            "d20_certificate_descriptor_sha256",
            "base_result_identity_sha256",
            "ray_identity_sha256",
        ):
            _exact_digest(object.__getattribute__(result, name), f"result.{name}")
        if (
            object.__getattribute__(result, "_authority") is not self.authority
            or base.surface is not sampler.surface
            or base.disk is not sampler.disk
            or base.termination is not sampler.termination
            or base.ray is not ray
            or base.observer_initial_state is not initial
            or base.ray_options is not ray_options
            or base.surface_options is not surface_options
            or base.escaped_observer_spectrum is not sampler.escaped_observer_spectrum
            or base.observer_four_velocity != sampler._observer_tetrad.four_velocity
            or base.null_residual_limit != sampler.frequency_null_residual_limit
            or base.conserved_quantity_tolerance
            != sampler.conserved_quantity_tolerance
            or base.surface_value_tolerance != surface_options.surface_value_tolerance
            or base.recorded_path_absolute_tolerance != path_absolute
            or base.recorded_path_relative_tolerance != path_relative
            or base.boundary_value_tolerance_m
            != sampler._resolved_boundary_value_tolerance_m
            or base.emitter_event_tolerance_m
            != sampler._resolved_emitter_event_tolerance_m
            or result.observer_frequencies_hz != frequencies
            or result.source_kind != base.source_kind
            or result.authority_snapshot_descriptor_sha256
            != snapshot.model_descriptor_sha256
            or result.provider_descriptor_sha256
            != snapshot.provider_descriptor_sha256
            or result.profile_descriptor_sha256
            != snapshot.profile_descriptor_sha256
            or result.axisymmetric_kernel_descriptor_sha256
            != snapshot.axisymmetric_kernel_descriptor_sha256
            or result.source_kernel_descriptor_sha256
            != snapshot.source_kernel_descriptor_sha256
            or result.underlying_kernel_descriptor_sha256
            != snapshot.underlying_kernel_descriptor_sha256
            or result.novikov_thorne_disk_descriptor_sha256
            != snapshot.novikov_thorne_disk_descriptor_sha256
            or result.surface_identity_descriptor_sha256
            != snapshot.surface_identity_descriptor_sha256
            or result.d20_certificate_descriptor_sha256
            != snapshot.d20_certificate.model_descriptor_sha256
        ):
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "returning transfer is not bound to its ray, sampler, and authority"
            )
        if result.source_kind == "finite-thickness-disk":
            base_rho = _exact_positive(
                base.pseudo_cylindrical_radius_over_mass,
                "base disk-hit radius",
            )
            result_rho = _exact_positive(
                result.pseudo_cylindrical_radius_over_mass,
                "returning disk-hit radius",
            )
            base_shift = _exact_positive(base.frequency_shift_g, "base g")
            result_shift = _exact_positive(result.frequency_shift_g, "returning g")
            if base.photon_projection is None:
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "base disk hit lacks its photon projection"
                )
            base_mu = _exact_positive(
                base.photon_projection.outgoing_cosine,
                "base outgoing emission cosine",
            )
            result_mu = _exact_positive(
                result.outgoing_emission_angle_cosine,
                "returning outgoing emission cosine",
            )
            expected_annulus = _annulus_evidence(snapshot, result_rho)
            supplied_annulus = (
                result.annulus_index,
                result.annulus_inner_radius_over_mass,
                result.annulus_outer_radius_over_mass,
                result.nearest_internal_annulus_edge_distance_over_mass,
            )
            if (
                type(result.face) is not str
                or result.face != base.face
                or result_rho.hex() != base_rho.hex()
                or result_shift.hex() != base_shift.hex()
                or result_mu.hex() != base_mu.hex()
                or supplied_annulus != expected_annulus
                or result.annulus_query_boundary_convention
                != transfer_module.ANNULUS_QUERY_BOUNDARY_CONVENTION
            ):
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "returning disk diagnostics disagree with certified base geometry"
                )
            emitted_frequencies = result.emitted_frequencies_hz
            isotropic = result.isotropic_emitted_specific_intensities_nu
            emitted = result.emitted_specific_intensities_nu
            observed = result.observed_specific_intensities_nu
            if any(
                type(value) is not tuple or len(value) != len(frequencies)
                for value in (emitted_frequencies, isotropic, emitted, observed)
            ):
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "returning disk algebra has malformed spectral tuples"
                )
            multiplier = _exact_positive(
                result.angular_emission_multiplier,
                "returning angular multiplier",
            )
            d20_law = _FRAME_D20_CLASS()
            if type(d20_law) is not _FRAME_D20_CLASS:
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "frame D20 law has a foreign exact type"
                )
            expected_multiplier = _FRAME_D20_INTENSITY_ENTRY(
                d20_law,
                result_mu,
            )
            fixed_d20 = 0.5 + 0.75 * result_mu
            expected_emitted_frequencies = tuple(
                frequency / result_shift for frequency in frequencies
            )
            checked_isotropic = tuple(
                _exact_nonnegative(value, f"returning isotropic bin {index}")
                for index, value in enumerate(isotropic)
            )
            expected_emitted = tuple(
                value * multiplier for value in checked_isotropic
            )
            shift_cubed = result_shift * result_shift * result_shift
            expected_observed = tuple(
                shift_cubed * value for value in expected_emitted
            )
            if (
                multiplier.hex() != expected_multiplier.hex()
                or not math.isclose(
                    expected_multiplier,
                    fixed_d20,
                    rel_tol=4.0 * math.ulp(1.0),
                    abs_tol=4.0 * math.ulp(1.0),
                )
                or emitted_frequencies != expected_emitted_frequencies
                or emitted != expected_emitted
                or observed != expected_observed
            ):
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "returning disk spectral algebra is inconsistent"
                )
        else:
            if any(
                value is not None
                for value in (
                    result.annulus_index,
                    result.annulus_inner_radius_over_mass,
                    result.annulus_outer_radius_over_mass,
                    result.nearest_internal_annulus_edge_distance_over_mass,
                    result.annulus_query_boundary_convention,
                    result.face,
                    result.pseudo_cylindrical_radius_over_mass,
                    result.frequency_shift_g,
                    result.outgoing_emission_angle_cosine,
                    result.emitted_frequencies_hz,
                    result.isotropic_emitted_specific_intensities_nu,
                    result.angular_emission_multiplier,
                    result.emitted_specific_intensities_nu,
                )
            ):
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "returning boundary result invented disk diagnostics"
                )
        expected_json, expected_sha = _recompute_transfer_descriptor(result)
        if (
            type(object.__getattribute__(result, "_descriptor_json")) is not str
            or type(object.__getattribute__(result, "_descriptor_sha256")) is not str
            or object.__getattribute__(result, "_descriptor_json").encode("utf-8")
            != expected_json.encode("utf-8")
            or object.__getattribute__(result, "_descriptor_sha256").encode("ascii")
            != expected_sha.encode("ascii")
        ):
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "returning transfer descriptor is stale"
            )
        if ray.outcome == OPAQUE_OUTCOME:
            trace = ray.multi_surface_trace
            if (
                base.terminal_surface_entry is not trace.crossings[-1]
                or base.terminal_surface_entry.crossing.state != ray.terminal_state
            ):
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "returning transfer terminal face is not ray-owned"
                )
        return result

    def _transfer(
        self,
        ray: RayTraceResult,
        initial: Any,
        frequencies: tuple[float, ...],
        snapshot: ReturningThermalEmissionSnapshotV1,
        *,
        ray_options: RayTraceOptions,
        surface_options: SurfaceEventOptions,
    ) -> KerrReturningRadiationFiniteThicknessSpectrumResult:
        sampler = self.base_sampler
        if (
            transfer_module
            ._transfer_kerr_returning_radiation_finite_thickness_spectrum_trusted
            is not _TRUSTED_TRANSFER_ENTRY
        ):
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "trusted returning-transfer callable changed after module load"
            )
        result = _TRUSTED_TRANSFER_ENTRY(
            sampler.surface,
            sampler.disk,
            ray,
            initial,
            sampler._observer_tetrad.four_velocity,
            frequencies,
            termination=sampler.termination,
            ray_options=ray_options,
            surface_options=surface_options,
            escaped_observer_spectrum=sampler.escaped_observer_spectrum,
            authority=self.authority,
            authenticated_snapshot=snapshot,
            null_residual_limit=sampler.frequency_null_residual_limit,
            conserved_quantity_tolerance=sampler.conserved_quantity_tolerance,
            surface_value_tolerance=surface_options.surface_value_tolerance,
            recorded_path_absolute_tolerance=max(
                sampler.recorded_path_absolute_tolerance,
                ray_options.absolute_tolerance,
            ),
            recorded_path_relative_tolerance=max(
                sampler.recorded_path_relative_tolerance,
                ray_options.relative_tolerance,
            ),
            boundary_value_tolerance_m=(
                sampler._resolved_boundary_value_tolerance_m
            ),
            emitter_event_tolerance_m=sampler._resolved_emitter_event_tolerance_m,
        )
        return self._validate_transfer_result(
            result,
            ray,
            initial,
            frequencies,
            snapshot,
            ray_options=ray_options,
            surface_options=surface_options,
        )

    def _binary64_radius_floor(
        self,
        fine_radius: float,
        coarse_radius: float,
        snapshot: ReturningThermalEmissionSnapshotV1,
    ) -> float:
        ulp = max(
            math.ulp(value)
            for value in (
                fine_radius,
                coarse_radius,
                *snapshot.annulus_edges_over_mass,
            )
        )
        floor = float(BINARY64_RADIUS_FLOOR_ULPS) * ulp
        if not math.isfinite(floor) or floor <= 0.0:
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "binary64 annulus radius floor is invalid"
            )
        return floor

    def _sample_trusted(
        self,
        screen_x: float,
        screen_y: float,
        observer_frequencies_hz: tuple[float, ...],
        snapshot: ReturningThermalEmissionSnapshotV1,
    ) -> SpectralRaySample:
        """Hot per-ray path; caller owns the two live-authority gates."""

        if type(self) is not KerrReturningRadiationFiniteThicknessRaySampler:
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "trusted ray sampler must have its exact type"
            )
        if snapshot is not self._authenticated_snapshot:
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "trusted ray received a foreign authority snapshot"
            )
        self._assert_configuration_stable()
        sampler = self.base_sampler
        x = base_frame_module._finite_number(screen_x, "screen_x")
        y = base_frame_module._finite_number(screen_y, "screen_y")
        frequencies = base_frame_module._positive_frequencies(
            observer_frequencies_hz
        )
        sampler._assert_escape_descriptor_stable()
        initial = kerr_zamo_camera_ray(
            sampler.metric,
            observer_radius_m=sampler.observer_radius_m,
            screen_x=x,
            screen_y=y,
            theta_rad=sampler.observer_theta_rad,
            phi_ks_rad=sampler.observer_phi_ks_rad,
            coordinate_time_m=sampler.observer_coordinate_time_m,
        )
        if initial.event != sampler._observer_tetrad.event:
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "camera ray and authenticated base observer disagree"
            )
        refinement = base_frame_module.trace_refined_null_geodesic(
            sampler.metric,
            initial,
            termination=sampler.termination,
            multi_interior_surface=sampler.surface,
            surface_options=sampler.surface_options,
            fine_options=sampler.fine_options,
            record_coarse_path=True,
            coarse_tolerance_multiplier=sampler.coarse_tolerance_multiplier,
            terminal_event_tolerance=sampler.terminal_event_tolerance_m,
            terminal_covector_tolerance=sampler.terminal_covector_tolerance,
        )
        fine_ray, coarse_ray = sampler._validate_refinement(refinement, initial)
        fine = self._transfer(
            fine_ray,
            initial,
            frequencies,
            snapshot,
            ray_options=sampler.fine_options,
            surface_options=sampler.surface_options,
        )
        sampler._assert_escape_descriptor_stable()
        coarse = self._transfer(
            coarse_ray,
            initial,
            frequencies,
            snapshot,
            ray_options=sampler._coarse_ray_options,
            surface_options=sampler._coarse_surface_options,
        )
        sampler._assert_escape_descriptor_stable()

        if fine.source_kind != coarse.source_kind:
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "fine/coarse visible sources disagree"
            )
        fine_topology = base_frame_module._topology_token(fine_ray)
        if fine_topology != base_frame_module._topology_token(coarse_ray):
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "fine/coarse multi-surface topologies disagree"
            )
        if (
            fine.authority_snapshot_descriptor_sha256
            != coarse.authority_snapshot_descriptor_sha256
            or fine.profile_descriptor_sha256 != coarse.profile_descriptor_sha256
            or fine.source_kernel_descriptor_sha256
            != coarse.source_kernel_descriptor_sha256
            or fine.surface_identity_descriptor_sha256
            != coarse.surface_identity_descriptor_sha256
        ):
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "fine/coarse thermal or surface identities disagree"
            )
        if (
            fine.base_result.transfer_configuration_sha256
            == coarse.base_result.transfer_configuration_sha256
        ):
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "fine/coarse transfers do not bind distinct trace options"
            )

        disk_radius_difference_m = 0.0
        relative_g_difference = 0.0
        frequency_shift: float | None = None
        escape_direction: tuple[float, float, float] | None = None
        if fine.source_kind == "finite-thickness-disk":
            fine_rho = _exact_positive(
                fine.pseudo_cylindrical_radius_over_mass,
                "fine pseudo-cylindrical radius",
            )
            coarse_rho = _exact_positive(
                coarse.pseudo_cylindrical_radius_over_mass,
                "coarse pseudo-cylindrical radius",
            )
            delta_rho = abs(fine_rho - coarse_rho)
            declared_envelope = (
                sampler.disk_radius_absolute_tolerance_m / sampler.metric.mass_m
                + sampler.disk_radius_relative_tolerance
                * max(abs(fine_rho), abs(coarse_rho))
            )
            if delta_rho > declared_envelope:
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "fine/coarse pseudo-cylindrical radii disagree"
                )
            fine_annulus = _annulus_evidence(snapshot, fine_rho)
            coarse_annulus = _annulus_evidence(snapshot, coarse_rho)
            if fine_annulus[0] != coarse_annulus[0]:
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "fine/coarse disk hits occupy different thermal annuli"
                )
            if (
                fine_annulus[1].hex() != coarse_annulus[1].hex()
                or fine_annulus[2].hex() != coarse_annulus[2].hex()
            ):
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "fine/coarse annulus boundaries disagree"
                )
            if snapshot.annulus_count == 1:
                if fine_annulus[3] is not None or coarse_annulus[3] is not None:
                    raise KerrReturningRadiationFiniteThicknessFrameError(
                        "single-annulus transfer invented an internal edge"
                    )
            else:
                fine_clearance = _exact_nonnegative(
                    fine_annulus[3],
                    "fine internal-edge clearance",
                )
                coarse_clearance = _exact_nonnegative(
                    coarse_annulus[3],
                    "coarse internal-edge clearance",
                )
                binary_floor = self._binary64_radius_floor(
                    fine_rho,
                    coarse_rho,
                    snapshot,
                )
                required_clearance = (
                    self._authenticated_edge_clearance_multiplier
                    * max(
                        delta_rho,
                        declared_envelope,
                        binary_floor,
                    )
                )
                if (
                    not math.isfinite(required_clearance)
                    or min(fine_clearance, coarse_clearance) < required_clearance
                ):
                    raise KerrReturningRadiationFiniteThicknessFrameError(
                        "fine/coarse disk hit is too close to an internal thermal "
                        "annulus edge"
                    )
            if fine.face != coarse.face or fine.face is None:
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "fine/coarse visible photosphere faces disagree"
                )
            fine_shift = _exact_positive(fine.frequency_shift_g, "fine g")
            coarse_shift = _exact_positive(coarse.frequency_shift_g, "coarse g")
            if not base_frame_module._within_tolerance(
                fine_shift,
                coarse_shift,
                absolute_tolerance=0.0,
                relative_tolerance=sampler.frequency_shift_relative_tolerance,
            ):
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "fine/coarse frequency shifts disagree"
                )
            fine_mu = _exact_positive(
                fine.outgoing_emission_angle_cosine,
                "fine outgoing emission cosine",
            )
            coarse_mu = _exact_positive(
                coarse.outgoing_emission_angle_cosine,
                "coarse outgoing emission cosine",
            )
            if fine_mu > 1.0 or coarse_mu > 1.0:
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "fine/coarse outgoing emission cosine exceeds one"
                )
            if abs(fine_mu - coarse_mu) > sampler.emission_cosine_absolute_tolerance:
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "fine/coarse signed face emission cosines disagree"
                )
            disk_radius_difference_m = delta_rho * sampler.metric.mass_m
            relative_g_difference = abs(fine_shift - coarse_shift) / max(
                abs(fine_shift),
                abs(coarse_shift),
                1.0e-300,
            )
            frequency_shift = fine_shift
            visible_source = "disk"
        elif fine.source_kind == "captured-boundary":
            if any(
                value != 0.0 or math.copysign(1.0, value) < 0.0
                for value in (
                    *fine.observed_specific_intensities_nu,
                    *coarse.observed_specific_intensities_nu,
                )
            ):
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "captured boundary is not exactly positive-zero black"
                )
            visible_source = "captured-boundary"
        elif fine.source_kind == "escaped-boundary":
            for label, transfer, ray in (
                ("fine", fine, fine_ray),
                ("coarse", coarse, coarse_ray),
            ):
                if ray.terminal_target_id is None:
                    raise KerrReturningRadiationFiniteThicknessFrameError(
                        f"{label} escaped ray lacks a terminal target"
                    )
                expected = tuple(
                    sampler.escaped_observer_spectrum(
                        ray.terminal_state,
                        frequency,
                        ray.terminal_target_id,
                    )
                    for frequency in frequencies
                )
                if transfer.observed_specific_intensities_nu != expected:
                    raise KerrReturningRadiationFiniteThicknessFrameError(
                        f"{label} escape spectrum disagrees with its closed provider"
                    )
            fine_direction = base_frame_module._finite_worldtube_direction(
                sampler.metric,
                fine_ray.terminal_state,
            )
            coarse_direction = base_frame_module._finite_worldtube_direction(
                sampler.metric,
                coarse_ray.terminal_state,
            )
            if (
                base_frame_module._angular_separation(
                    fine_direction,
                    coarse_direction,
                )
                > sampler.escape_direction_tolerance_rad
            ):
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "fine/coarse finite-worldtube escape directions disagree"
                )
            escape_direction = fine_direction
            visible_source = "escaped-boundary"
        else:
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "returning transfer returned an unknown source kind"
            )

        fine_bins = fine.observed_specific_intensities_nu
        coarse_bins = coarse.observed_specific_intensities_nu
        if (
            type(fine_bins) is not tuple
            or type(coarse_bins) is not tuple
            or len(fine_bins) != len(frequencies)
            or len(coarse_bins) != len(frequencies)
        ):
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "returning transfer returned the wrong spectral bin count"
            )
        errors = tuple(abs(left - right) for left, right in zip(fine_bins, coarse_bins))
        for index, (left, right) in enumerate(zip(fine_bins, coarse_bins)):
            _exact_nonnegative(left, f"fine observed intensity {index}")
            _exact_nonnegative(right, f"coarse observed intensity {index}")
            if not base_frame_module._within_tolerance(
                left,
                right,
                absolute_tolerance=sampler.specific_intensity_absolute_tolerance,
                relative_tolerance=sampler.specific_intensity_relative_tolerance,
            ):
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    f"fine/coarse returning-thermal spectral bin {index} disagrees"
                )

        audit = RayConvergenceAudit(
            maximum_null_residual=max(
                fine_ray.maximum_null_residual,
                coarse_ray.maximum_null_residual,
            ),
            maximum_metric_interpolation_error=max(
                fine_ray.maximum_metric_interpolation_error,
                coarse_ray.maximum_metric_interpolation_error,
            ),
            terminal_event_difference_m=refinement.terminal_event_difference,
            terminal_covector_relative_difference=(
                refinement.terminal_covector_difference
            ),
            disk_radius_difference_m=disk_radius_difference_m,
            relative_g_difference=relative_g_difference,
            surface_bracket_affine_width=max(
                base_frame_module._maximum_surface_bracket(fine_ray),
                base_frame_module._maximum_surface_bracket(coarse_ray),
            ),
            accepted_steps=max(fine_ray.accepted_steps, coarse_ray.accepted_steps),
            rejected_steps=max(fine_ray.rejected_steps, coarse_ray.rejected_steps),
            ray_gate_passed=True,
            source_gate_passed=True,
            transfer_gate_passed=True,
        )
        return SpectralRaySample(
            specific_intensities_nu=fine_bins,
            absolute_errors_nu=errors,
            visible_source=visible_source,
            topology_signature=fine_topology,
            frequency_shift_g=frequency_shift,
            escape_direction=escape_direction,
            ray_converged=True,
            convergence_audit=audit,
        )

    def sample(
        self,
        screen_x: float,
        screen_y: float,
        observer_frequencies_hz: tuple[float, ...],
    ) -> SpectralRaySample:
        """Return one ray only after live-authority start and end gates."""

        if type(self) is not KerrReturningRadiationFiniteThicknessRaySampler:
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "public ray sampler must have its exact type"
            )
        start_snapshot = self.authority.require_live()
        try:
            if start_snapshot is not self._authenticated_snapshot:
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "authority no longer owns the sampler snapshot"
                )
            result = _SAMPLER_TRUSTED_ENTRY(
                self,
                screen_x,
                screen_y,
                observer_frequencies_hz,
                start_snapshot,
            )
        finally:
            end_snapshot = self.authority.require_live()
            if end_snapshot is not start_snapshot:
                raise KerrReturningRadiationFiniteThicknessFrameError(
                    "authority snapshot changed while sampling the ray"
                )
        return result


_SAMPLER_TRUSTED_ENTRY: Final = (
    KerrReturningRadiationFiniteThicknessRaySampler._sample_trusted
)


@dataclass(frozen=True, slots=True)
class _TrustedPixelSampler:
    sampler: KerrReturningRadiationFiniteThicknessRaySampler
    snapshot: ReturningThermalEmissionSnapshotV1

    def sample(
        self,
        screen_x: float,
        screen_y: float,
        observer_frequencies_hz: tuple[float, ...],
    ) -> SpectralRaySample:
        return _SAMPLER_TRUSTED_ENTRY(
            self.sampler,
            screen_x,
            screen_y,
            observer_frequencies_hz,
            self.snapshot,
        )


def integrate_returning_thermal_spectral_pixel(
    sampler: KerrReturningRadiationFiniteThicknessRaySampler,
    observer_frequencies_hz: Sequence[float],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    options: AdaptivePixelOptions,
) -> AdaptivePixelResult:
    """Integrate one adaptive pixel with only whole-pixel live checks."""

    if type(sampler) is not KerrReturningRadiationFiniteThicknessRaySampler:
        raise TypeError("sampler must have its exact returning-thermal type")
    start_snapshot = sampler.authority.require_live()
    try:
        if start_snapshot is not sampler._authenticated_snapshot:
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "authority no longer owns the pixel sampler snapshot"
            )
        result = integrate_spectral_pixel(
            _TrustedPixelSampler(sampler, start_snapshot),
            observer_frequencies_hz,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            options=options,
        )
    finally:
        end_snapshot = sampler.authority.require_live()
        if end_snapshot is not start_snapshot:
            raise KerrReturningRadiationFiniteThicknessFrameError(
                "authority snapshot changed while integrating the pixel"
            )
    return result


__all__ = (
    "BINARY64_RADIUS_FLOOR_ULPS",
    "IMPLEMENTATION_ID",
    "KerrReturningRadiationFiniteThicknessFrameError",
    "KerrReturningRadiationFiniteThicknessRaySampler",
    "MAXIMUM_ANNULUS_EDGE_CLEARANCE_MULTIPLIER",
    "SCIENTIFIC_STATUS",
    "integrate_returning_thermal_spectral_pixel",
)
