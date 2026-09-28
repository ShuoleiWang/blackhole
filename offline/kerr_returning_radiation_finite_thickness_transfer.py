"""Returning-radiation thermal spectrum on one certified finite-height ray.

The existing :mod:`offline.kerr_finite_thickness_transfer` remains the sole
owner of first-visible geometry, full deterministic ray replay, worldtube
authentication, finite-height emitter construction, invariant frequency shift
``g``, and signed outgoing face cosine ``mu``.  This module calls that transfer
once and replaces only its equatorial Novikov--Thorne emission sample with the
closure-private thermal table owned by a
``ValidatedReturningThermalAuthority``::

    nu_emit = nu_obs / g
    I_nu,emit = f_D20(mu) B_nu(f_col T_eff,out) / f_col**4
    I_nu,obs = g**3 I_nu,emit.

``F_out`` already equals ``F_0 + F_in`` in the authenticated thermal profile.
It is never added again, doubled, or multiplied by a bolometric ``g**4``.
Captured rays are exactly black.  Escaped rays are recomputed directly from
the closed observer-frame boundary provider at each true observer frequency;
the old transfer's stored spectral bins are not trusted as the new result.

The standalone entry performs expensive live-authority checks at both frame
boundaries.  A private trusted entry accepts the exact snapshot returned by a
prior boundary check so a future frame reducer can reuse it across many rays.
The per-ray hot path never replays the returning-radiation kernel/profile.

This is a same-code coupled transfer, not an independent oracle, complete
KERRBB, returning-radiation stress/work term ``F_S``, spectral redistribution,
scattering atmosphere, polarization calculation, or GRMHD result.  It also
does not establish fine/coarse whole-frame same-annulus or edge-clearance
convergence; it records the annulus and nearest internal edge distance for the
future frame layer to enforce those gates.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from types import FunctionType, MappingProxyType
from typing import Any, Final, Mapping

from offline.disk_atmosphere import FluxConservingLinearLimbDarkening
from offline.geodesic import (
    HamiltonianState,
    RayTraceOptions,
    RayTraceResult,
    SurfaceEventOptions,
)
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_disk import StationaryNovikovThorneDisk
from offline.kerr_finite_thickness import (
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_surface import (
    FINITE_THICKNESS_SURFACE_IDS,
    KerrFiniteThicknessMultiSurface,
)
from offline.kerr_finite_thickness_transfer import (
    BuiltInEscapedObserverSpectrum,
    KerrFiniteThicknessSpectrumResult,
    transfer_kerr_finite_thickness_spectrum,
)
from offline.kerr_returning_radiation_frame_context import (
    ReturningThermalEmissionSnapshotV1,
    ValidatedReturningThermalAuthority,
)


IMPLEMENTATION_ID: Final = (
    "kerr-returning-radiation-finite-thickness-first-visible-transfer/v1"
)
MAXIMUM_FREQUENCY_BINS: Final = 4096
ANNULUS_QUERY_BOUNDARY_CONVENTION: Final = (
    "left-closed right-open, with final outer edge included"
)
_BASE_TRANSFER_ENTRY: Final = transfer_kerr_finite_thickness_spectrum
_AUTHORITY_CLASS: Final = ValidatedReturningThermalAuthority
_AUTHORITY_BATCH_ENTRY: Final = (
    ValidatedReturningThermalAuthority.emitted_specific_intensity_nu_batch
)
_D20_CLASS: Final = FluxConservingLinearLimbDarkening
_D20_INTENSITY_ENTRY: Final = FluxConservingLinearLimbDarkening.intensity_multiplier

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "same-code authenticated returning-thermal finite-thickness "
            "single-ray transfer"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "geometryOwner": "offline.kerr_finite_thickness_transfer/v1",
        "emissionOwner": (
            "ValidatedReturningThermalAuthority private frozen table"
        ),
        "equation": (
            "I_nu,obs=g^3 f_D20(mu) "
            "B_nu(f_col*T_eff,out)/f_col^4"
        ),
        "emittedFrequency": "nu_emit=nu_obs/g",
        "angularLaw": "KERRBB D20 f(mu)=1/2+3mu/4",
        "outgoingFluxAlreadyIncludesReturningIncidentFlux": True,
        "addsIncidentFluxAgain": False,
        "implicitFactorOfTwo": False,
        "usesBolometricG4ForSpecificIntensity": False,
        "captureBoundary": "exact positive-sign zero",
        "escapeBoundary": "closed built-in observer-frame spectrum recomputed",
        "standaloneAuthorityChecks": "explicit frame start and frame end",
        "trustedHotEntryIsPublic": False,
        "baseTransferCallableIdentityFrozenAtModuleLoad": True,
        "authorityBatchCallableIdentityFrozenAtModuleLoad": True,
        "d20ClassAndCallableIdentityFrozenAtModuleLoad": True,
        "legacyBaseGeometryWitnessBins": 1,
        "legacyBaseRadianceUsedByCoupledResult": False,
        "maximumFrequencyBins": MAXIMUM_FREQUENCY_BINS,
        "recordsAnnulusIndexAndNearestInternalEdgeDistance": True,
        "thermalRadialDiscretization": (
            "piecewise-constant authenticated F_out and T_eff per annulus; "
            "left-closed right-open with the final outer edge included"
        ),
        "establishesFineCoarseSameAnnulus": False,
        "establishesEdgeClearance": False,
        "hasIndependentPhysicsOracle": False,
        "isCompleteKerrbb": False,
        "includesReturningRadiationStressWorkFS": False,
        "includesSpectralRedistribution": False,
        "includesScatteringOrSolvedAtmosphere": False,
        "includesPolarization": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "prohibitedClaim": (
            "Do not describe this shared-code coupling as an independent "
            "oracle, complete KERRBB, F_S, spectral redistribution, a solved "
            "atmosphere, polarization, GRMHD, or a fine/coarse frame proof."
        ),
    }
)


class KerrReturningRadiationFiniteThicknessTransferError(RuntimeError):
    """Base class for fail-closed coupled-transfer failures."""


class KerrReturningRadiationFiniteThicknessTransferVerificationError(
    KerrReturningRadiationFiniteThicknessTransferError
):
    """Raised when a coupled result differs from deterministic replay."""


def _require_trusted_runtime_bindings() -> None:
    """Fail before physics work if a frozen production entry was rebound."""

    if (
        type(_BASE_TRANSFER_ENTRY) is not FunctionType
        or transfer_kerr_finite_thickness_spectrum is not _BASE_TRANSFER_ENTRY
    ):
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "certified base-transfer callable changed after module load"
        )
    if (
        ValidatedReturningThermalAuthority is not _AUTHORITY_CLASS
        or type(_AUTHORITY_BATCH_ENTRY) is not FunctionType
        or _AUTHORITY_CLASS.emitted_specific_intensity_nu_batch
        is not _AUTHORITY_BATCH_ENTRY
    ):
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "authenticated thermal-batch callable changed after module load"
        )
    if (
        FluxConservingLinearLimbDarkening is not _D20_CLASS
        or type(_D20_INTENSITY_ENTRY) is not FunctionType
        or _D20_CLASS.intensity_multiplier is not _D20_INTENSITY_ENTRY
    ):
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "D20 angular-law callable changed after module load"
        )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise KerrReturningRadiationFiniteThicknessTransferError(
            "coupled-transfer descriptor is not finite canonical JSON"
        ) from error


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError(f"{label} must be an exact lowercase SHA-256 string")
    return value


def _exact_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{label} must be a finite exact float")
    return value


def _exact_positive_float(value: Any, label: str) -> float:
    result = _exact_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _exact_nonnegative_float(value: Any, label: str) -> float:
    result = _exact_float(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _exact_frequencies(value: Any) -> tuple[float, ...]:
    if type(value) is not tuple or not value:
        raise TypeError("observer_frequencies_hz must be a non-empty exact tuple")
    if len(value) > MAXIMUM_FREQUENCY_BINS:
        raise ValueError(
            f"observer frequency grid exceeds {MAXIMUM_FREQUENCY_BINS} bins"
        )
    checked = tuple(
        _exact_positive_float(frequency, f"observer frequency {index}")
        for index, frequency in enumerate(value)
    )
    if any(right <= left for left, right in zip(checked, checked[1:])):
        raise ValueError("observer frequencies must be strictly increasing")
    return checked


def _exact_observer_four_velocity(value: Any) -> tuple[float, float, float, float]:
    if type(value) is not tuple or len(value) != 4:
        raise TypeError("observer_four_velocity must be an exact four-float tuple")
    checked = tuple(
        _exact_float(component, f"observer four-velocity component {index}")
        for index, component in enumerate(value)
    )
    return checked  # type: ignore[return-value]


def _same_float(first: float, second: float) -> bool:
    return first.hex() == second.hex()


def _require_exact_tree(actual: Any, expected: Any, path: str) -> None:
    """Compare trusted dataclass/primitive evidence without equality hooks."""

    if type(actual) is not type(expected):
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            f"{path} has non-exact type {type(actual).__name__}; "
            f"expected {type(expected).__name__}"
        )
    if is_dataclass(expected) and not isinstance(expected, type):
        for item in fields(expected):
            _require_exact_tree(
                object.__getattribute__(actual, item.name),
                object.__getattribute__(expected, item.name),
                f"{path}.{item.name}",
            )
        return
    if type(expected) is tuple:
        if len(actual) != len(expected):
            raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
                f"{path} tuple length differs"
            )
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            _require_exact_tree(actual_item, expected_item, f"{path}[{index}]")
        return
    if type(expected) is float:
        differs = not math.isfinite(actual) or actual.hex() != expected.hex()
    elif type(expected) is str:
        differs = actual.encode("utf-8") != expected.encode("utf-8")
    elif type(expected) is int:
        differs = actual != expected
    elif type(expected) is bool:
        differs = actual is not expected
    elif expected is None:
        differs = False
    else:
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            f"{path} uses unsupported exact type {type(expected).__name__}"
        )
    if differs:
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            f"{path} differs from deterministic replay"
        )


def _canonical_identity_tree(value: Any, path: str) -> Any:
    value_type = type(value)
    if value_type is float:
        if not math.isfinite(value):
            raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
                f"{path} contains a non-finite float"
            )
        return {"floatHex": value.hex()}
    if value_type is str:
        return {"str": value}
    if value_type is bool:
        return {"bool": value}
    if value_type is int:
        return {"intDecimal": str(value)}
    if value is None:
        return None
    if value_type is tuple:
        return [
            _canonical_identity_tree(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "exactType": f"{value_type.__module__}.{value_type.__qualname__}",
            "fields": [
                [
                    item.name,
                    _canonical_identity_tree(
                        object.__getattribute__(value, item.name),
                        f"{path}.{item.name}",
                    ),
                ]
                for item in fields(value)
            ],
        }
    raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
        f"{path} contains unsupported type {value_type.__name__}"
    )


def _identity_sha256(value: Any, label: str) -> str:
    return _sha256_text(
        _canonical_json(_canonical_identity_tree(value, label))
    )


def _validate_snapshot_ownership(
    surface: KerrFiniteThicknessMultiSurface,
    disk: StationaryNovikovThorneDisk,
    snapshot: ReturningThermalEmissionSnapshotV1,
) -> None:
    """Bind render ownership to the authenticated metric/calibration/disk."""

    if type(surface) is not KerrFiniteThicknessMultiSurface:
        raise TypeError("surface must have its exact finite-thickness type")
    if type(surface.metric) is not KerrKerrSchildMetric:
        raise TypeError("surface metric must have its exact Kerr type")
    if type(surface.calibration) is not StationaryKerrFiniteThicknessCalibration:
        raise TypeError("surface calibration must have its exact stationary type")
    if type(disk) is not StationaryNovikovThorneDisk:
        raise TypeError("disk must have its exact StationaryNovikovThorneDisk type")
    if disk.metric is not surface.metric:
        raise ValueError("disk and surface must own the same exact metric object")
    if type(snapshot) is not ReturningThermalEmissionSnapshotV1:
        raise TypeError("authority snapshot must have its exact public type")
    try:
        metric = json.loads(snapshot.metric_identity_descriptor_json)
        calibration = json.loads(snapshot.calibration_identity_descriptor_json)
        surface_identity = json.loads(snapshot.surface_identity_descriptor_json)
        disk_identity = json.loads(snapshot.disk_identity_descriptor_json)
        disk_metric = disk_identity["metric"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "authority identity descriptors are malformed"
        ) from error
    exact_pairs = (
        (surface.metric.mass_m, metric["massM"], "metric mass"),
        (surface.metric.spin_a_m, metric["spinAM"], "signed Kerr spin"),
        (
            surface.metric.singularity_guard_m,
            metric["singularityGuardM"],
            "metric singularity guard",
        ),
        (
            surface.calibration.dimensionless_spin,
            calibration["dimensionlessSpin"],
            "calibration spin",
        ),
        (
            surface.calibration.eddington_scaled_mass_accretion_rate,
            calibration["eddingtonScaledMassAccretionRate"],
            "calibration dotm",
        ),
        (
            surface.calibration.isco_radius_over_mass,
            calibration["iscoRadiusOverMass"],
            "calibration ISCO",
        ),
        (
            surface.calibration.outer_radius_over_mass,
            calibration["outerRadiusOverMass"],
            "calibration outer radius",
        ),
        (
            surface.calibration.thinness_gate_maximum_h_over_rho,
            calibration["thinnessGateMaximumHOverRho"],
            "calibration thinness gate",
        ),
        (disk.black_hole_mass_kg, disk_identity["blackHoleMassKg"], "disk mass"),
        (
            disk.mass_accretion_rate_kg_s,
            disk_identity["massAccretionRateKgS"],
            "disk accretion rate",
        ),
        (
            disk.colour_correction,
            disk_identity["colourCorrection"],
            "disk colour correction",
        ),
        (disk.metric.mass_m, disk_metric["massM"], "disk metric mass"),
        (disk.metric.spin_a_m, disk_metric["spinAM"], "disk metric spin"),
        (
            disk.metric.singularity_guard_m,
            disk_metric["singularityGuardM"],
            "disk metric guard",
        ),
    )
    for actual, expected, label in exact_pairs:
        if type(expected) is not float or not _same_float(actual, expected):
            raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
                f"render {label} differs from authority snapshot"
            )
    string_pairs = (
        (surface.metric.source_id, metric["sourceId"], "metric source id"),
        (
            surface.calibration.orientation,
            calibration["orientation"],
            "calibration orientation",
        ),
        (disk.orientation, disk_identity["orientation"], "disk orientation"),
        (disk.metric.source_id, disk_metric["sourceId"], "disk metric source id"),
    )
    for actual, expected, label in string_pairs:
        if type(expected) is not str or actual.encode("utf-8") != expected.encode(
            "utf-8"
        ):
            raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
                f"render {label} differs from authority snapshot"
            )
    if (
        type(metric["timeDependent"]) is not bool
        or metric["timeDependent"] is not surface.metric.time_dependent
        or type(disk_metric["timeDependent"]) is not bool
        or disk_metric["timeDependent"] is not disk.metric.time_dependent
    ):
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "render metric stationarity differs from authority snapshot"
        )
    surface_ids = surface_identity.get("surfaceIds")
    if (
        type(surface_ids) is not list
        or tuple(surface_ids) != FINITE_THICKNESS_SURFACE_IDS
        or surface.surface_ids != FINITE_THICKNESS_SURFACE_IDS
    ):
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "render stable surface ids differ from authority snapshot"
        )


def _annulus_binding(
    snapshot: ReturningThermalEmissionSnapshotV1,
    radius_over_mass: float,
) -> tuple[int, float, float, float | None]:
    radius = _exact_positive_float(radius_over_mass, "disk-hit radius")
    edges = snapshot.annulus_edges_over_mass
    if type(edges) is not tuple or len(edges) < 2:
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "authority annulus edges are malformed"
        )
    if radius < edges[0] or radius > edges[-1]:
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "disk-hit radius lies outside the authenticated thermal domain"
        )
    index = bisect_right(edges, radius) - 1
    if index == len(edges) - 1:
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
        raise KerrReturningRadiationFiniteThicknessTransferError(
            "nearest internal annulus-edge distance is invalid"
        )
    return index, edges[index], edges[index + 1], nearest


@dataclass(frozen=True, slots=True, init=False)
class KerrReturningRadiationFiniteThicknessSpectrumResult:
    """One immutable coupled result with public deterministic revalidation."""

    base_result: KerrFiniteThicknessSpectrumResult
    observer_frequencies_hz: tuple[float, ...]
    geometry_witness_observer_frequency_hz: float
    observed_specific_intensities_nu: tuple[float, ...]
    source_kind: str
    authority_snapshot_descriptor_sha256: str
    provider_descriptor_sha256: str
    profile_descriptor_sha256: str
    axisymmetric_kernel_descriptor_sha256: str
    source_kernel_descriptor_sha256: str
    underlying_kernel_descriptor_sha256: str
    novikov_thorne_disk_descriptor_sha256: str
    surface_identity_descriptor_sha256: str
    d20_certificate_descriptor_sha256: str
    base_result_identity_sha256: str
    ray_identity_sha256: str
    annulus_index: int | None
    annulus_inner_radius_over_mass: float | None
    annulus_outer_radius_over_mass: float | None
    nearest_internal_annulus_edge_distance_over_mass: float | None
    annulus_query_boundary_convention: str | None
    face: str | None
    pseudo_cylindrical_radius_over_mass: float | None
    frequency_shift_g: float | None
    outgoing_emission_angle_cosine: float | None
    emitted_frequencies_hz: tuple[float, ...] | None
    isotropic_emitted_specific_intensities_nu: tuple[float, ...] | None
    angular_emission_multiplier: float | None
    emitted_specific_intensities_nu: tuple[float, ...] | None
    _authority: ValidatedReturningThermalAuthority
    _descriptor_json: str
    _descriptor_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "KerrReturningRadiationFiniteThicknessSpectrumResult is built "
            "only by the authenticated transfer entry point"
        )

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        if type(self._descriptor_json) is not str:
            raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
                "result descriptor has a non-exact type"
            )
        return json.loads(self._descriptor_json)

    def revalidate(self) -> None:
        verify_kerr_returning_radiation_finite_thickness_spectrum_result(self)


def _result_descriptor(
    values: Mapping[str, Any],
) -> Mapping[str, Any]:
    source_kind = values["source_kind"]
    disk_hit = source_kind == "finite-thickness-disk"
    return {
        "algebra": {
            "angularEmissionMultiplier": values["angular_emission_multiplier"],
            "emittedFrequenciesHz": values["emitted_frequencies_hz"],
            "emittedSpecificIntensitiesNu": (
                values["emitted_specific_intensities_nu"]
            ),
            "equation": (
                "I_nu,obs=g^3 f_D20(mu) "
                "B_nu(f_col*T_eff,out)/f_col^4"
            ),
            "frequencyShiftG": values["frequency_shift_g"],
            "isotropicEmittedSpecificIntensitiesNu": (
                values["isotropic_emitted_specific_intensities_nu"]
            ),
            "observedSpecificIntensitiesNu": (
                values["observed_specific_intensities_nu"]
            ),
            "observerFrequenciesHz": values["observer_frequencies_hz"],
            "outgoingEmissionAngleCosine": (
                values["outgoing_emission_angle_cosine"]
            ),
            "usesBolometricG4": False,
        },
        "annulus": {
            "index": values["annulus_index"],
            "innerRadiusOverMass": values["annulus_inner_radius_over_mass"],
            "nearestInternalEdgeDistanceOverMass": (
                values["nearest_internal_annulus_edge_distance_over_mass"]
            ),
            "outerRadiusOverMass": values["annulus_outer_radius_over_mass"],
            "queryBoundaryConvention": (
                values["annulus_query_boundary_convention"]
            ),
            "sameBinFineCoarseVerifiedHere": False,
            "edgeClearanceVerifiedHere": False,
            "valueModel": (
                "piecewise-constant authenticated F_out and T_eff per annulus"
            ),
        },
        "binding": {
            "authoritySnapshotDescriptorSha256": (
                values["authority_snapshot_descriptor_sha256"]
            ),
            "axisymmetricKernelDescriptorSha256": (
                values["axisymmetric_kernel_descriptor_sha256"]
            ),
            "baseResultIdentitySha256": values["base_result_identity_sha256"],
            "baseTransferConfigurationSha256": (
                values["base_result"].transfer_configuration_sha256
            ),
            "d20CertificateDescriptorSha256": (
                values["d20_certificate_descriptor_sha256"]
            ),
            "novikovThorneDiskDescriptorSha256": (
                values["novikov_thorne_disk_descriptor_sha256"]
            ),
            "profileDescriptorSha256": values["profile_descriptor_sha256"],
            "providerDescriptorSha256": values["provider_descriptor_sha256"],
            "rayIdentitySha256": values["ray_identity_sha256"],
            "sourceKernelDescriptorSha256": (
                values["source_kernel_descriptor_sha256"]
            ),
            "surfaceIdentityDescriptorSha256": (
                values["surface_identity_descriptor_sha256"]
            ),
            "underlyingKernelDescriptorSha256": (
                values["underlying_kernel_descriptor_sha256"]
            ),
        },
        "boundary": {
            "captureIsExactPositiveSignZero": True,
            "escapeSpectrumRecomputedAtObserverFrequencies": True,
        },
        "geometryWitness": {
            "observerFrequencyHz": (
                values["geometry_witness_observer_frequency_hz"]
            ),
            "purpose": (
                "single-bin legacy transfer witness for geometry, g, and mu; "
                "its Novikov-Thorne radiance is not used by this result"
            ),
        },
        "diskHit": {
            "face": values["face"],
            "pseudoCylindricalRadiusOverMass": (
                values["pseudo_cylindrical_radius_over_mass"]
            ),
            "present": disk_hit,
        },
        "implementationId": IMPLEMENTATION_ID,
        "scientificBoundary": {
            "addsIncidentFluxAgain": False,
            "hasIndependentPhysicsOracle": False,
            "implicitFactorOfTwo": False,
            "includesAtmosphere": False,
            "includesPolarization": False,
            "includesReturningRadiationStressWorkFS": False,
            "includesScattering": False,
            "includesSpectralRedistribution": False,
            "isCompleteKerrbb": False,
            "thermalProfileIsPiecewiseConstantByAnnulus": True,
        },
        "sourceKind": source_kind,
    }


_ANNULUS_BINDING_ENTRY: Final = _annulus_binding
_RESULT_DESCRIPTOR_ENTRY: Final = _result_descriptor
_CANONICAL_JSON_ENTRY: Final = _canonical_json
_SHA256_TEXT_ENTRY: Final = _sha256_text


def _build_result_from_base(
    base: KerrFiniteThicknessSpectrumResult,
    authority: ValidatedReturningThermalAuthority,
    snapshot: ReturningThermalEmissionSnapshotV1,
    observer_frequencies_hz: tuple[float, ...],
) -> KerrReturningRadiationFiniteThicknessSpectrumResult:
    _require_trusted_runtime_bindings()
    if (
        _annulus_binding is not _ANNULUS_BINDING_ENTRY
        or _result_descriptor is not _RESULT_DESCRIPTOR_ENTRY
        or _canonical_json is not _CANONICAL_JSON_ENTRY
        or _sha256_text is not _SHA256_TEXT_ENTRY
    ):
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "coupled-result identity helper changed after module load"
        )
    if type(base) is not KerrFiniteThicknessSpectrumResult:
        raise TypeError("base transfer must return its exact public result type")
    if type(authority) is not _AUTHORITY_CLASS:
        raise TypeError("authority must have its exact process-local type")
    if authority.snapshot is not snapshot:
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "trusted transfer snapshot is not owned by the authority"
        )
    _validate_snapshot_ownership(base.surface, base.disk, snapshot)
    frequencies = _exact_frequencies(observer_frequencies_hz)
    witness_frequencies = _exact_frequencies(base.observer_frequencies_hz)
    if (
        len(witness_frequencies) != 1
        or not _same_float(witness_frequencies[0], frequencies[0])
    ):
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "base geometry witness must use exactly the first observer bin"
        )
    common: dict[str, Any] = {
        "base_result": base,
        "observer_frequencies_hz": frequencies,
        "geometry_witness_observer_frequency_hz": witness_frequencies[0],
        "source_kind": base.source_kind,
        "authority_snapshot_descriptor_sha256": (
            snapshot.model_descriptor_sha256
        ),
        "provider_descriptor_sha256": snapshot.provider_descriptor_sha256,
        "profile_descriptor_sha256": snapshot.profile_descriptor_sha256,
        "axisymmetric_kernel_descriptor_sha256": (
            snapshot.axisymmetric_kernel_descriptor_sha256
        ),
        "source_kernel_descriptor_sha256": (
            snapshot.source_kernel_descriptor_sha256
        ),
        "underlying_kernel_descriptor_sha256": (
            snapshot.underlying_kernel_descriptor_sha256
        ),
        "novikov_thorne_disk_descriptor_sha256": (
            snapshot.novikov_thorne_disk_descriptor_sha256
        ),
        "surface_identity_descriptor_sha256": (
            snapshot.surface_identity_descriptor_sha256
        ),
        "d20_certificate_descriptor_sha256": (
            snapshot.d20_certificate.model_descriptor_sha256
        ),
        "base_result_identity_sha256": _identity_sha256(base, "base_result"),
        "ray_identity_sha256": _identity_sha256(base.ray, "base_result.ray"),
        "annulus_index": None,
        "annulus_inner_radius_over_mass": None,
        "annulus_outer_radius_over_mass": None,
        "nearest_internal_annulus_edge_distance_over_mass": None,
        "annulus_query_boundary_convention": None,
        "face": None,
        "pseudo_cylindrical_radius_over_mass": None,
        "frequency_shift_g": None,
        "outgoing_emission_angle_cosine": None,
        "emitted_frequencies_hz": None,
        "isotropic_emitted_specific_intensities_nu": None,
        "angular_emission_multiplier": None,
        "emitted_specific_intensities_nu": None,
        "_authority": authority,
    }
    if base.source_kind == "captured-boundary":
        observed = tuple(0.0 for _ in frequencies)
        if any(math.copysign(1.0, value) != 1.0 for value in observed):
            raise AssertionError("captured boundary zero lost its positive sign")
    elif base.source_kind == "escaped-boundary":
        observed = tuple(
            _exact_nonnegative_float(
                base.escaped_observer_spectrum(
                    base.ray.terminal_state,
                    frequency,
                    base.ray.terminal_target_id,
                ),
                f"escaped observer intensity {index}",
            )
            for index, frequency in enumerate(frequencies)
        )
    elif base.source_kind == "finite-thickness-disk":
        rho = _exact_positive_float(
            base.pseudo_cylindrical_radius_over_mass,
            "base disk-hit radius",
        )
        shift = _exact_positive_float(base.frequency_shift_g, "base frequency shift")
        projection = base.photon_projection
        if projection is None:
            raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
                "base disk hit lacks an authenticated photon projection"
            )
        mu = _exact_nonnegative_float(
            projection.outgoing_cosine,
            "base outgoing emission cosine",
        )
        if mu > 1.0:
            raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
                "base outgoing emission cosine exceeds one"
            )
        emitted_frequencies = tuple(frequency / shift for frequency in frequencies)
        if any(
            not math.isfinite(frequency) or frequency <= 0.0
            for frequency in emitted_frequencies
        ):
            raise KerrReturningRadiationFiniteThicknessTransferError(
                "emitter-frame frequency left the finite positive domain"
            )
        isotropic = _AUTHORITY_BATCH_ENTRY(
            authority,
            rho,
            emitted_frequencies,
        )
        if type(isotropic) is not tuple or len(isotropic) != len(frequencies):
            raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
                "authority batch returned a malformed spectrum"
            )
        checked_isotropic = tuple(
            _exact_nonnegative_float(value, f"isotropic emission {index}")
            for index, value in enumerate(isotropic)
        )
        law = _D20_CLASS()
        if type(law) is not _D20_CLASS:
            raise AssertionError("D20 law lost its exact implementation type")
        multiplier = _exact_positive_float(
            _D20_INTENSITY_ENTRY(law, mu),
            "D20 angular multiplier",
        )
        fixed_d20 = 0.5 + 0.75 * mu
        if not math.isclose(
            multiplier,
            fixed_d20,
            rel_tol=4.0 * math.ulp(1.0),
            abs_tol=4.0 * math.ulp(1.0),
        ):
            raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
                "D20 angular law disagrees with f(mu)=1/2+3mu/4"
            )
        emitted = tuple(value * multiplier for value in checked_isotropic)
        shift_cubed = shift * shift * shift
        observed = tuple(shift_cubed * value for value in emitted)
        if (
            not math.isfinite(shift_cubed)
            or shift_cubed <= 0.0
            or any(
                not math.isfinite(value) or value < 0.0
                for value in (*emitted, *observed)
            )
        ):
            raise KerrReturningRadiationFiniteThicknessTransferError(
                "returning-thermal specific-intensity chain overflowed"
            )
        annulus_index, inner, outer, nearest = _ANNULUS_BINDING_ENTRY(snapshot, rho)
        common.update(
            {
                "annulus_index": annulus_index,
                "annulus_inner_radius_over_mass": inner,
                "annulus_outer_radius_over_mass": outer,
                "nearest_internal_annulus_edge_distance_over_mass": nearest,
                "annulus_query_boundary_convention": (
                    ANNULUS_QUERY_BOUNDARY_CONVENTION
                ),
                "face": base.face,
                "pseudo_cylindrical_radius_over_mass": rho,
                "frequency_shift_g": shift,
                "outgoing_emission_angle_cosine": mu,
                "emitted_frequencies_hz": emitted_frequencies,
                "isotropic_emitted_specific_intensities_nu": checked_isotropic,
                "angular_emission_multiplier": multiplier,
                "emitted_specific_intensities_nu": emitted,
            }
        )
    else:
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "base transfer has an unsupported source kind"
        )
    common["observed_specific_intensities_nu"] = tuple(observed)
    descriptor_json = _CANONICAL_JSON_ENTRY(_RESULT_DESCRIPTOR_ENTRY(common))
    result = object.__new__(KerrReturningRadiationFiniteThicknessSpectrumResult)
    for name, value in (
        *common.items(),
        ("_descriptor_json", descriptor_json),
        ("_descriptor_sha256", _SHA256_TEXT_ENTRY(descriptor_json)),
    ):
        object.__setattr__(result, name, value)
    return result


_BUILD_RESULT_ENTRY: Final = _build_result_from_base


def _transfer_kerr_returning_radiation_finite_thickness_spectrum_trusted(
    surface: KerrFiniteThicknessMultiSurface,
    disk: StationaryNovikovThorneDisk,
    ray: RayTraceResult,
    observer_initial_state: HamiltonianState,
    observer_four_velocity: tuple[float, float, float, float],
    observer_frequencies_hz: tuple[float, ...],
    *,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    escaped_observer_spectrum: BuiltInEscapedObserverSpectrum,
    authority: ValidatedReturningThermalAuthority,
    authenticated_snapshot: ReturningThermalEmissionSnapshotV1,
    null_residual_limit: float = 2.0e-7,
    conserved_quantity_tolerance: float = 2.0e-7,
    surface_value_tolerance: float = 2.0e-8,
    recorded_path_absolute_tolerance: float = 2.0e-10,
    recorded_path_relative_tolerance: float = 2.0e-10,
    boundary_value_tolerance_m: float | None = None,
    emitter_event_tolerance_m: float | None = None,
) -> KerrReturningRadiationFiniteThicknessSpectrumResult:
    """Private per-ray entry for a frame that already checked authority live."""

    _require_trusted_runtime_bindings()
    if _build_result_from_base is not _BUILD_RESULT_ENTRY:
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "coupled-result builder changed after module load"
        )
    frequencies = _exact_frequencies(observer_frequencies_hz)
    checked_observer_four_velocity = _exact_observer_four_velocity(
        observer_four_velocity
    )
    if type(authority) is not _AUTHORITY_CLASS:
        raise TypeError("authority must have its exact process-local type")
    if type(authenticated_snapshot) is not ReturningThermalEmissionSnapshotV1:
        raise TypeError("authenticated_snapshot must have its exact public type")
    if authority.snapshot is not authenticated_snapshot:
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "trusted snapshot is not the authority's current frozen snapshot"
        )
    base = _BASE_TRANSFER_ENTRY(
        surface,
        disk,
        ray,
        observer_initial_state,
        checked_observer_four_velocity,
        (frequencies[0],),
        termination=termination,
        ray_options=ray_options,
        surface_options=surface_options,
        escaped_observer_spectrum=escaped_observer_spectrum,
        null_residual_limit=null_residual_limit,
        conserved_quantity_tolerance=conserved_quantity_tolerance,
        surface_value_tolerance=surface_value_tolerance,
        recorded_path_absolute_tolerance=recorded_path_absolute_tolerance,
        recorded_path_relative_tolerance=recorded_path_relative_tolerance,
        boundary_value_tolerance_m=boundary_value_tolerance_m,
        emitter_event_tolerance_m=emitter_event_tolerance_m,
    )
    return _BUILD_RESULT_ENTRY(
        base,
        authority,
        authenticated_snapshot,
        frequencies,
    )


_TRUSTED_COUPLED_TRANSFER_ENTRY: Final = (
    _transfer_kerr_returning_radiation_finite_thickness_spectrum_trusted
)


def transfer_kerr_returning_radiation_finite_thickness_spectrum(
    surface: KerrFiniteThicknessMultiSurface,
    disk: StationaryNovikovThorneDisk,
    ray: RayTraceResult,
    observer_initial_state: HamiltonianState,
    observer_four_velocity: tuple[float, float, float, float],
    observer_frequencies_hz: tuple[float, ...],
    *,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    escaped_observer_spectrum: BuiltInEscapedObserverSpectrum,
    authority: ValidatedReturningThermalAuthority,
    null_residual_limit: float = 2.0e-7,
    conserved_quantity_tolerance: float = 2.0e-7,
    surface_value_tolerance: float = 2.0e-8,
    recorded_path_absolute_tolerance: float = 2.0e-10,
    recorded_path_relative_tolerance: float = 2.0e-10,
    boundary_value_tolerance_m: float | None = None,
    emitter_event_tolerance_m: float | None = None,
) -> KerrReturningRadiationFiniteThicknessSpectrumResult:
    """Authenticate live emission at both boundaries and transfer one ray."""

    _require_trusted_runtime_bindings()
    if (
        _transfer_kerr_returning_radiation_finite_thickness_spectrum_trusted
        is not _TRUSTED_COUPLED_TRANSFER_ENTRY
    ):
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "trusted coupled-transfer callable changed after module load"
        )
    if type(authority) is not _AUTHORITY_CLASS:
        raise TypeError("authority must have its exact process-local type")
    checked_frequencies = _exact_frequencies(observer_frequencies_hz)
    checked_observer_four_velocity = _exact_observer_four_velocity(
        observer_four_velocity
    )
    start_snapshot = authority.require_live()
    try:
        result = (
            _TRUSTED_COUPLED_TRANSFER_ENTRY(
                surface,
                disk,
                ray,
                observer_initial_state,
                checked_observer_four_velocity,
                checked_frequencies,
                termination=termination,
                ray_options=ray_options,
                surface_options=surface_options,
                escaped_observer_spectrum=escaped_observer_spectrum,
                authority=authority,
                authenticated_snapshot=start_snapshot,
                null_residual_limit=null_residual_limit,
                conserved_quantity_tolerance=conserved_quantity_tolerance,
                surface_value_tolerance=surface_value_tolerance,
                recorded_path_absolute_tolerance=(
                    recorded_path_absolute_tolerance
                ),
                recorded_path_relative_tolerance=(
                    recorded_path_relative_tolerance
                ),
                boundary_value_tolerance_m=boundary_value_tolerance_m,
                emitter_event_tolerance_m=emitter_event_tolerance_m,
            )
        )
    finally:
        end_snapshot = authority.require_live()
        if end_snapshot is not start_snapshot:
            raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
                "authority snapshot changed while the ray transfer was running"
            )
    return result


_PUBLIC_COUPLED_TRANSFER_ENTRY: Final = (
    transfer_kerr_returning_radiation_finite_thickness_spectrum
)


def verify_kerr_returning_radiation_finite_thickness_spectrum_result(
    result: KerrReturningRadiationFiniteThicknessSpectrumResult,
) -> None:
    """Replay the base ray and rebuild the complete exact coupled result."""

    _require_trusted_runtime_bindings()
    if (
        transfer_kerr_returning_radiation_finite_thickness_spectrum
        is not _PUBLIC_COUPLED_TRANSFER_ENTRY
    ):
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "public coupled-transfer callable changed after module load"
        )
    if type(result) is not KerrReturningRadiationFiniteThicknessSpectrumResult:
        raise TypeError("result must have its exact coupled-transfer type")
    base = object.__getattribute__(result, "base_result")
    authority = object.__getattribute__(result, "_authority")
    if type(base) is not KerrFiniteThicknessSpectrumResult:
        raise TypeError("result base transfer has a non-exact type")
    if type(authority) is not _AUTHORITY_CLASS:
        raise TypeError("result authority has a non-exact type")
    rebuilt = _PUBLIC_COUPLED_TRANSFER_ENTRY(
        base.surface,
        base.disk,
        base.ray,
        base.observer_initial_state,
        base.observer_four_velocity,
        object.__getattribute__(result, "observer_frequencies_hz"),
        termination=base.termination,
        ray_options=base.ray_options,
        surface_options=base.surface_options,
        escaped_observer_spectrum=base.escaped_observer_spectrum,
        authority=authority,
        null_residual_limit=base.null_residual_limit,
        conserved_quantity_tolerance=base.conserved_quantity_tolerance,
        surface_value_tolerance=base.surface_value_tolerance,
        recorded_path_absolute_tolerance=base.recorded_path_absolute_tolerance,
        recorded_path_relative_tolerance=base.recorded_path_relative_tolerance,
        boundary_value_tolerance_m=base.boundary_value_tolerance_m,
        emitter_event_tolerance_m=base.emitter_event_tolerance_m,
    )
    for item in fields(result):
        actual = object.__getattribute__(result, item.name)
        expected = object.__getattribute__(rebuilt, item.name)
        if item.name == "_authority":
            if actual is not expected:
                raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
                    "result authority identity differs from replay"
                )
        else:
            _require_exact_tree(actual, expected, f"result.{item.name}")
    supplied_json = object.__getattribute__(result, "_descriptor_json")
    supplied_sha = object.__getattribute__(result, "_descriptor_sha256")
    if type(supplied_json) is not str or type(supplied_sha) is not str:
        raise KerrReturningRadiationFiniteThicknessTransferVerificationError(
            "result descriptor identity has non-exact types"
        )
    _digest(supplied_sha, "result descriptor SHA-256")


__all__ = (
    "ANNULUS_QUERY_BOUNDARY_CONVENTION",
    "IMPLEMENTATION_ID",
    "MAXIMUM_FREQUENCY_BINS",
    "SCIENTIFIC_STATUS",
    "KerrReturningRadiationFiniteThicknessSpectrumResult",
    "KerrReturningRadiationFiniteThicknessTransferError",
    "KerrReturningRadiationFiniteThicknessTransferVerificationError",
    "transfer_kerr_returning_radiation_finite_thickness_spectrum",
    "verify_kerr_returning_radiation_finite_thickness_spectrum_result",
)
