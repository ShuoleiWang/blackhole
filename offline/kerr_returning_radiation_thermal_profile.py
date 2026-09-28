"""Versioned axisymmetric absorbed-returning-radiation thermal profile.

This module joins two already public layers without changing either one:

* :class:`offline.returning_radiation.AxisymmetricReturningRadiationKernel`
  maps one-face, local-comoving outgoing bolometric flux to one-face incident
  bolometric flux; and
* :class:`offline.kerr_disk.StationaryNovikovThorneDisk` supplies the intrinsic
  zero-torque Page--Thorne flux from one disk face.

For piecewise-constant annulus values it solves

``F_in = K F_out`` and ``F_out = F_0 + F_in``.

The incident energy is assumed to be completely absorbed and instantaneously
thermally reradiated locally.  Thus the reported effective-temperature (and,
for the bound constant colour correction, colour-temperature) ratio is
``(F_out/F_0)**(1/4)``.  The module deliberately does *not* add the KERRBB
returning-radiation stress/work term ``F_S`` and does not solve a scattering
atmosphere, spectral redistribution, polarization, vertical structure, or
GRMHD.  It produces no frame or image.

Annulus proper areas are for one face and are expressed in ``M**2``.  They must
not be multiplied into ``K`` a second time.  The generic entry point treats an
``AxisymmetricReturningRadiationKernel`` and caller-declared areas as external,
unverified inputs and therefore does not publish provenance-certified watts.
The certified entry point instead either replays a complete Kerr forward
kernel (or its authenticated adjacent-annulus projection), or re-authenticates
an exact production-forward cached execution from its receipts without tracing
rays.  It derives the grid and areas, checks them against the NT disk, and only
then reports physical one-face watts using the area scale
``(G M_BH / c**2)**2``.  Cached projections, receiver caches, and synthetic
evaluators are not certified sources.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping

from offline.kerr import KerrKerrSchildMetric
from offline.kerr_disk import (
    GRAVITATIONAL_CONSTANT_M3_KG_S2,
    LIGHT_SPEED_M_S,
    StationaryNovikovThorneDisk,
)
from offline.kerr_returning_radiation_kernel import (
    KerrForwardReturningRadiationKernel,
    KerrForwardReturningRadiationKernelProjection,
    verify_and_reduce_kerr_returning_radiation_energy_kernel,
    verify_and_reduce_kerr_returning_radiation_kernel_projection,
)
from offline.kerr_returning_radiation_kernel_cached import (
    KerrCachedReturningRadiationKernelExecution,
    validate_and_reduce_cached_kerr_returning_radiation_energy_kernel,
)
from offline.novikov_thorne import PROGRADE, RETROGRADE, kerr_isco_radius_m
from offline.returning_radiation import (
    AxisymmetricReturningRadiationKernel,
    DEFAULT_FIXED_POINT_POLICY,
    ReturningRadiationConvergenceError,
    ReturningRadiationFixedPointPolicy,
    ReturningRadiationFluxSolution,
    solve_absorbed_returning_radiation,
    validate_returning_radiation_solution,
)


IMPLEMENTATION_ID: Final = (
    "kerr-nt-axisymmetric-absorbed-returning-thermal-annulus-profile/v1"
)
PROJECTION_IMPLEMENTATION_ID: Final = f"{IMPLEMENTATION_ID}/area-projection/v1"
MAXIMUM_RELATIVE_EQUATION_RESIDUAL: Final = 1.0e-10
EXTERNAL_UNVERIFIED_CLASSIFICATION: Final = (
    "external-unverified-axisymmetric-kernel-and-declared-geometry"
)
REPLAYED_KERR_FORWARD_CLASSIFICATION: Final = (
    "same-code-replayed-kerr-forward-kernel-and-derived-geometry"
)
CACHED_KERR_FORWARD_CLASSIFICATION: Final = (
    "same-code-cached-kerr-forward-kernel-and-derived-geometry"
)
CERTIFIED_KERR_FORWARD_CLASSIFICATIONS: Final = (
    REPLAYED_KERR_FORWARD_CLASSIFICATION,
    CACHED_KERR_FORWARD_CLASSIFICATION,
)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": "two-tier axisymmetric returning thermal profile",
        "externalEntryClassification": EXTERNAL_UNVERIFIED_CLASSIFICATION,
        "certifiedEntryClassification": REPLAYED_KERR_FORWARD_CLASSIFICATION,
        "cachedCertifiedEntryClassification": CACHED_KERR_FORWARD_CLASSIFICATION,
        "implementationId": IMPLEMENTATION_ID,
        "equation": "F_out=F_0+F_in; F_in=K F_out",
        "intrinsicFlux": "one-face zero-torque Novikov-Thorne/Page-Thorne flux",
        "fluxUnits": "W m^-2 of local comoving one-face proper area",
        "properAreaUnits": "M^2 for one disk face",
        "absorptionAlbedo": 0.0,
        "temperatureRatio": "(F_out/F_0)^(1/4)",
        "fixedColourCorrection": True,
        "piecewiseConstantAnnuli": True,
        "independentlyVerifiesDeclaredProperAreaProvenance": False,
        "externalEntryPublishesProvenanceCertifiedWatts": False,
        "certifiedEntryReplaysPhysicalAreaProvenance": True,
        "cachedCertifiedEntryRetracesRays": False,
        "cachedCertifiedEntryUsesLayoutFreeScientificIdentity": True,
        "maximumRelativeEquationResidual": (
            MAXIMUM_RELATIVE_EQUATION_RESIDUAL
        ),
        "includesReturningRadiationStressWorkFS": False,
        "includesSpectralRedistribution": False,
        "includesScattering": False,
        "includesSolvedAtmosphere": False,
        "includesPolarization": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "changesRenderedFrames": False,
        "prohibitedClaim": (
            "Do not describe this absorbed bolometric F_in fixed point as "
            "KERRBB F_S, a scattering atmosphere, spectral redistribution, "
            "polarized transfer, GRMHD, or a rendered-image update."
        ),
    }
)


class KerrReturningRadiationThermalProfileError(RuntimeError):
    """Base class for fail-closed thermal-profile failures."""


class KerrReturningRadiationThermalProfileVerificationError(
    KerrReturningRadiationThermalProfileError
):
    """Raised when an immutable profile cannot be reproduced exactly."""


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
        raise KerrReturningRadiationThermalProfileError(
            "thermal-profile descriptor is not finite canonical JSON"
        ) from error


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{label} must be a finite exact float")
    return value


def _exact_positive_float(value: Any, label: str) -> float:
    result = _exact_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _exact_tuple(value: Any, label: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact tuple")
    return value


def _digest(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError(f"{label} must be an exact lowercase SHA-256 string")
    return value


def _same_float(first: float, second: float) -> bool:
    return first.hex() == second.hex()


def _require_exact_tree(actual: Any, expected: Any, path: str) -> None:
    """Compare a trusted primitive tree without subclass dispatch."""

    if type(actual) is not type(expected):
        raise KerrReturningRadiationThermalProfileVerificationError(
            f"{path} has non-exact type {type(actual).__name__}; "
            f"expected {type(expected).__name__}"
        )
    if type(expected) is tuple:
        if len(actual) != len(expected):
            raise KerrReturningRadiationThermalProfileVerificationError(
                f"{path} tuple length differs from deterministic replay"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _require_exact_tree(actual_item, expected_item, f"{path}[{index}]")
        return
    if is_dataclass(expected) and not isinstance(expected, type):
        for item in fields(expected):
            _require_exact_tree(
                object.__getattribute__(actual, item.name),
                object.__getattribute__(expected, item.name),
                f"{path}.{item.name}",
            )
        return
    if type(expected) is float:
        differs = actual.hex() != expected.hex()
    elif type(expected) is int:
        differs = actual != expected
    elif type(expected) is bool:
        differs = actual is not expected
    elif type(expected) is str:
        differs = actual.encode("utf-8") != expected.encode("utf-8")
    elif expected is None:
        differs = False
    else:
        raise KerrReturningRadiationThermalProfileVerificationError(
            f"{path} uses unsupported trusted type {type(expected).__name__}"
        )
    if differs:
        raise KerrReturningRadiationThermalProfileVerificationError(
            f"{path} differs from deterministic replay"
        )


def _validated_kernel(
    value: Any,
) -> AxisymmetricReturningRadiationKernel:
    if type(value) is not AxisymmetricReturningRadiationKernel:
        raise TypeError(
            "kernel must be an exact AxisymmetricReturningRadiationKernel"
        )
    radii = _exact_tuple(
        object.__getattribute__(value, "annulus_radii_over_mass"),
        "kernel.annulus_radii_over_mass",
    )
    checked_radii = tuple(
        _exact_positive_float(radius, f"kernel radius {index}")
        for index, radius in enumerate(radii)
    )
    rows = _exact_tuple(
        object.__getattribute__(value, "receiver_emitter_coefficients"),
        "kernel.receiver_emitter_coefficients",
    )
    checked_rows: list[tuple[float, ...]] = []
    for receiver_index, row in enumerate(rows):
        entries = _exact_tuple(row, f"kernel row {receiver_index}")
        checked_rows.append(
            tuple(
                _exact_float(coefficient, f"kernel[{receiver_index},{index}]")
                for index, coefficient in enumerate(entries)
            )
        )
    producer = object.__getattribute__(value, "ray_kernel_producer_id")
    if type(producer) is not str or not producer.strip():
        raise TypeError("kernel.ray_kernel_producer_id must be an exact string")
    rebuilt = AxisymmetricReturningRadiationKernel(
        annulus_radii_over_mass=checked_radii,
        receiver_emitter_coefficients=tuple(checked_rows),
        ray_kernel_producer_id=producer,
    )
    supplied_json = object.__getattribute__(value, "_descriptor_json")
    supplied_sha = object.__getattribute__(value, "_descriptor_sha256")
    if type(supplied_json) is not str or type(supplied_sha) is not str:
        raise KerrReturningRadiationThermalProfileVerificationError(
            "kernel descriptor storage has non-exact types"
        )
    if (
        supplied_json.encode("utf-8")
        != rebuilt.canonical_descriptor_json.encode("utf-8")
        or supplied_sha.encode("ascii")
        != rebuilt.canonical_descriptor_sha256.encode("ascii")
    ):
        raise KerrReturningRadiationThermalProfileVerificationError(
            "kernel fields and authenticated descriptor are inconsistent"
        )
    return rebuilt


def _validated_disk(value: Any) -> StationaryNovikovThorneDisk:
    if type(value) is not StationaryNovikovThorneDisk:
        raise TypeError("disk must be an exact StationaryNovikovThorneDisk")
    metric = object.__getattribute__(value, "metric")
    if type(metric) is not KerrKerrSchildMetric:
        raise TypeError("disk.metric must be an exact KerrKerrSchildMetric")
    mass_m = _exact_positive_float(
        object.__getattribute__(metric, "mass_m"), "disk.metric.mass_m"
    )
    spin_a_m = _exact_float(
        object.__getattribute__(metric, "spin_a_m"), "disk.metric.spin_a_m"
    )
    guard_m = _exact_positive_float(
        object.__getattribute__(metric, "singularity_guard_m"),
        "disk.metric.singularity_guard_m",
    )
    source_id = object.__getattribute__(metric, "source_id")
    stationary = object.__getattribute__(metric, "time_dependent")
    if type(source_id) is not str or type(stationary) is not bool:
        raise TypeError("disk.metric ownership fields must have exact types")
    rebuilt_metric = KerrKerrSchildMetric(
        mass_m=mass_m,
        spin_a_m=spin_a_m,
        singularity_guard_m=guard_m,
        source_id=source_id,
        time_dependent=stationary,
    )
    black_hole_mass = _exact_positive_float(
        object.__getattribute__(value, "black_hole_mass_kg"),
        "disk.black_hole_mass_kg",
    )
    accretion_rate = _exact_positive_float(
        object.__getattribute__(value, "mass_accretion_rate_kg_s"),
        "disk.mass_accretion_rate_kg_s",
    )
    colour_correction = _exact_positive_float(
        object.__getattribute__(value, "colour_correction"),
        "disk.colour_correction",
    )
    orientation = object.__getattribute__(value, "orientation")
    if type(orientation) is not str or orientation not in (PROGRADE, RETROGRADE):
        raise TypeError("disk.orientation must be an exact supported string")
    return StationaryNovikovThorneDisk(
        metric=rebuilt_metric,
        black_hole_mass_kg=black_hole_mass,
        mass_accretion_rate_kg_s=accretion_rate,
        orientation=orientation,
        colour_correction=colour_correction,
    )


def _validated_policy(
    value: Any,
) -> ReturningRadiationFixedPointPolicy:
    if type(value) is not ReturningRadiationFixedPointPolicy:
        raise TypeError(
            "policy must be an exact ReturningRadiationFixedPointPolicy"
        )
    maximum_iterations = object.__getattribute__(value, "maximum_iterations")
    if type(maximum_iterations) is not int:
        raise TypeError("policy.maximum_iterations must be an exact int")
    absolute = _exact_float(
        object.__getattribute__(value, "absolute_residual_tolerance"),
        "policy.absolute_residual_tolerance",
    )
    relative = _exact_float(
        object.__getattribute__(value, "relative_residual_tolerance"),
        "policy.relative_residual_tolerance",
    )
    return ReturningRadiationFixedPointPolicy(
        maximum_iterations=maximum_iterations,
        absolute_residual_tolerance=absolute,
        relative_residual_tolerance=relative,
    )


def _validated_geometry(
    kernel: AxisymmetricReturningRadiationKernel,
    disk: StationaryNovikovThorneDisk,
    edges_value: Any,
    areas_value: Any,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    edges_raw = _exact_tuple(edges_value, "annulus_edges_over_mass")
    if len(edges_raw) != kernel.annulus_count + 1:
        raise ValueError("annulus edges must have one more entry than the kernel")
    edges = tuple(
        _exact_positive_float(value, f"annulus edge {index}")
        for index, value in enumerate(edges_raw)
    )
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("annulus edges must be strictly increasing")
    isco = kerr_isco_radius_m(
        disk.dimensionless_spin_magnitude,
        disk.orientation,
    )
    if not _same_float(edges[0], isco):
        raise ValueError(
            "annulus domain must start exactly at the disk ISCO; inside-ISCO "
            "or radially truncated intrinsic profiles are rejected"
        )
    for index, (inner, representative, outer) in enumerate(
        zip(edges, kernel.annulus_radii_over_mass, edges[1:])
    ):
        if not inner < representative < outer:
            raise ValueError(
                f"kernel representative radius {index} must lie strictly "
                "inside its declared annulus"
            )
    areas_raw = _exact_tuple(
        areas_value, "annulus_proper_areas_over_mass_squared"
    )
    if len(areas_raw) != kernel.annulus_count:
        raise ValueError("proper areas must have one entry per annulus")
    areas = tuple(
        _exact_positive_float(value, f"annulus proper area {index}")
        for index, value in enumerate(areas_raw)
    )
    return edges, areas


def _disk_descriptor(disk: StationaryNovikovThorneDisk) -> Mapping[str, Any]:
    return {
        "blackHoleMassKg": disk.black_hole_mass_kg,
        "colourCorrection": disk.colour_correction,
        "fluxBoundaryCondition": "zero torque at ISCO",
        "fluxFaceSemantics": "one disk face",
        "massAccretionRateKgS": disk.mass_accretion_rate_kg_s,
        "metric": {
            "dimensionlessSpin": disk.metric.dimensionless_spin,
            "massM": disk.metric.mass_m,
            "singularityGuardM": disk.metric.singularity_guard_m,
            "sourceId": disk.metric.source_id,
            "spinAM": disk.metric.spin_a_m,
            "timeDependent": disk.metric.time_dependent,
        },
        "orientation": disk.orientation,
        "radialScalar": "Page-Thorne 4 pi M^2 F / dot(M)",
        "thermalProvider": "offline.kerr_disk.StationaryNovikovThorneDisk",
    }


def _finite_sum(values: tuple[float, ...], label: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise KerrReturningRadiationThermalProfileError(
            f"{label} overflowed binary64"
        ) from error
    if not math.isfinite(result):
        raise KerrReturningRadiationThermalProfileError(
            f"{label} is not finite"
        )
    return result


@dataclass(frozen=True, slots=True)
class AxisymmetricThermalAnnulusValue:
    """One queried piecewise-constant, one-face annulus value."""

    annulus_index: int
    inner_radius_over_mass: float
    outer_radius_over_mass: float
    representative_radius_over_mass: float
    proper_area_over_mass_squared: float
    intrinsic_flux_w_m2: float
    incident_returning_flux_w_m2: float
    outgoing_flux_w_m2: float
    effective_temperature_ratio: float


@dataclass(frozen=True, slots=True, init=False)
class AxisymmetricReturningRadiationThermalProfile:
    """Versioned fixed point with an explicit provenance classification."""

    annulus_edges_over_mass: tuple[float, ...]
    annulus_representative_radii_over_mass: tuple[float, ...]
    annulus_proper_areas_over_mass_squared: tuple[float, ...]
    intrinsic_flux_w_m2: tuple[float, ...]
    incident_returning_flux_w_m2: tuple[float, ...]
    outgoing_flux_w_m2: tuple[float, ...]
    effective_temperature_ratio: tuple[float, ...]
    intrinsic_effective_temperature_k: tuple[float, ...]
    outgoing_effective_temperature_k: tuple[float, ...]
    maximum_relative_equation_residual: float
    maximum_allowed_relative_equation_residual: float
    provenance_classification: str
    geometry_provenance_certified: bool
    one_face_intrinsic_power_w: float | None
    one_face_incident_returning_power_w: float | None
    one_face_outgoing_power_w: float | None
    one_face_power_balance_residual_w: float | None
    one_face_power_balance_tolerance_w: float | None
    kernel_descriptor_sha256: str
    novikov_thorne_disk_descriptor_sha256: str
    fixed_point_policy_descriptor_sha256: str
    fixed_point_solution_descriptor_sha256: str
    source_authentication_descriptor_sha256: str | None
    _kernel: AxisymmetricReturningRadiationKernel = field(repr=False, compare=False)
    _disk: StationaryNovikovThorneDisk = field(repr=False, compare=False)
    _policy: ReturningRadiationFixedPointPolicy = field(repr=False, compare=False)
    _certified_source: (
        KerrForwardReturningRadiationKernel
        | KerrForwardReturningRadiationKernelProjection
        | KerrCachedReturningRadiationKernelExecution
        | None
    ) = field(repr=False, compare=False)
    _descriptor_json: str = field(repr=False, compare=False)
    _descriptor_sha256: str = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "AxisymmetricReturningRadiationThermalProfile is built only by "
            "the external-unverified or replay-certified solve entry point"
        )

    @property
    def annulus_count(self) -> int:
        return len(self.annulus_representative_radii_over_mass)

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        if type(self._descriptor_json) is not str:
            raise KerrReturningRadiationThermalProfileVerificationError(
                "profile descriptor has a non-exact type"
            )
        return json.loads(self._descriptor_json)

    @property
    def two_face_intrinsic_power_w(self) -> float | None:
        if self.one_face_intrinsic_power_w is None:
            return None
        return 2.0 * self.one_face_intrinsic_power_w

    @property
    def two_face_incident_returning_power_w(self) -> float | None:
        if self.one_face_incident_returning_power_w is None:
            return None
        return 2.0 * self.one_face_incident_returning_power_w

    @property
    def two_face_outgoing_power_w(self) -> float | None:
        if self.one_face_outgoing_power_w is None:
            return None
        return 2.0 * self.one_face_outgoing_power_w

    def annulus(self, annulus_index: int) -> AxisymmetricThermalAnnulusValue:
        if type(annulus_index) is not int:
            raise TypeError("annulus_index must be an exact int")
        if annulus_index < 0 or annulus_index >= self.annulus_count:
            raise IndexError("annulus_index lies outside the profile")
        return AxisymmetricThermalAnnulusValue(
            annulus_index=annulus_index,
            inner_radius_over_mass=self.annulus_edges_over_mass[annulus_index],
            outer_radius_over_mass=self.annulus_edges_over_mass[annulus_index + 1],
            representative_radius_over_mass=(
                self.annulus_representative_radii_over_mass[annulus_index]
            ),
            proper_area_over_mass_squared=(
                self.annulus_proper_areas_over_mass_squared[annulus_index]
            ),
            intrinsic_flux_w_m2=self.intrinsic_flux_w_m2[annulus_index],
            incident_returning_flux_w_m2=(
                self.incident_returning_flux_w_m2[annulus_index]
            ),
            outgoing_flux_w_m2=self.outgoing_flux_w_m2[annulus_index],
            effective_temperature_ratio=(
                self.effective_temperature_ratio[annulus_index]
            ),
        )

    def at_radius_over_mass(
        self, radius_over_mass: float
    ) -> AxisymmetricThermalAnnulusValue:
        radius = _exact_float(radius_over_mass, "radius_over_mass")
        if radius < self.annulus_edges_over_mass[0]:
            raise ValueError("radius lies inside the ISCO/profile domain")
        if radius > self.annulus_edges_over_mass[-1]:
            raise ValueError("radius lies outside the finite profile domain")
        index = bisect_right(self.annulus_edges_over_mass, radius) - 1
        if index == self.annulus_count:
            index -= 1
        return self.annulus(index)

    def revalidate(self) -> None:
        verify_axisymmetric_returning_radiation_thermal_profile(self)

    def coarsen_annuli(
        self, merged_annulus_edges_over_mass: tuple[float, ...]
    ) -> "AxisymmetricReturningRadiationThermalProfileProjection":
        """Return an area-conservative reporting projection.

        This operation does not pretend that averaging a solved fine-grid
        fixed point is the same as solving a newly projected coarse kernel.
        """

        self.revalidate()
        return _coarsen_profile(self, merged_annulus_edges_over_mass)


@dataclass(frozen=True, slots=True, init=False)
class AxisymmetricReturningRadiationThermalProfileProjection:
    """Area-conservative reporting projection, not a re-solved fixed point."""

    annulus_edges_over_mass: tuple[float, ...]
    annulus_representative_radii_over_mass: tuple[float, ...]
    annulus_proper_areas_over_mass_squared: tuple[float, ...]
    intrinsic_flux_w_m2: tuple[float, ...]
    incident_returning_flux_w_m2: tuple[float, ...]
    outgoing_flux_w_m2: tuple[float, ...]
    effective_temperature_ratio: tuple[float, ...]
    source_profile_descriptor_sha256: str
    is_resolved_coarse_fixed_point: bool
    _source: AxisymmetricReturningRadiationThermalProfile = field(
        repr=False, compare=False
    )
    _descriptor_json: str = field(repr=False, compare=False)
    _descriptor_sha256: str = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "AxisymmetricReturningRadiationThermalProfileProjection is built "
            "only by profile.coarsen_annuli"
        )

    @property
    def annulus_count(self) -> int:
        return len(self.annulus_edges_over_mass) - 1

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        if type(self._descriptor_json) is not str:
            raise KerrReturningRadiationThermalProfileVerificationError(
                "projection descriptor has a non-exact type"
            )
        return json.loads(self._descriptor_json)

    def annulus(self, annulus_index: int) -> AxisymmetricThermalAnnulusValue:
        if type(annulus_index) is not int:
            raise TypeError("annulus_index must be an exact int")
        if annulus_index < 0 or annulus_index >= self.annulus_count:
            raise IndexError("annulus_index lies outside the projection")
        return AxisymmetricThermalAnnulusValue(
            annulus_index=annulus_index,
            inner_radius_over_mass=self.annulus_edges_over_mass[annulus_index],
            outer_radius_over_mass=self.annulus_edges_over_mass[annulus_index + 1],
            representative_radius_over_mass=(
                self.annulus_representative_radii_over_mass[annulus_index]
            ),
            proper_area_over_mass_squared=(
                self.annulus_proper_areas_over_mass_squared[annulus_index]
            ),
            intrinsic_flux_w_m2=self.intrinsic_flux_w_m2[annulus_index],
            incident_returning_flux_w_m2=(
                self.incident_returning_flux_w_m2[annulus_index]
            ),
            outgoing_flux_w_m2=self.outgoing_flux_w_m2[annulus_index],
            effective_temperature_ratio=(
                self.effective_temperature_ratio[annulus_index]
            ),
        )

    def at_radius_over_mass(
        self, radius_over_mass: float
    ) -> AxisymmetricThermalAnnulusValue:
        radius = _exact_float(radius_over_mass, "radius_over_mass")
        if radius < self.annulus_edges_over_mass[0]:
            raise ValueError("radius lies inside the ISCO/projection domain")
        if radius > self.annulus_edges_over_mass[-1]:
            raise ValueError("radius lies outside the finite projection domain")
        index = bisect_right(self.annulus_edges_over_mass, radius) - 1
        if index == self.annulus_count:
            index -= 1
        return self.annulus(index)

    def revalidate(self) -> None:
        verify_axisymmetric_returning_radiation_thermal_profile_projection(self)


def _maximum_relative_residual(
    solution: ReturningRadiationFluxSolution,
) -> float:
    relative_values: list[float] = []
    for index, (intrinsic, incident, outgoing, residual) in enumerate(
        zip(
            solution.intrinsic_flux,
            solution.incident_returning_flux,
            solution.outgoing_flux,
            solution.equation_residual,
        )
    ):
        right_hand_side = _finite_sum(
            (intrinsic, incident),
            f"fixed-point right-hand side {index}",
        )
        scale = max(abs(right_hand_side), abs(outgoing))
        if not math.isfinite(scale) or scale <= 0.0:
            raise KerrReturningRadiationThermalProfileError(
                "relative fixed-point residual has no finite positive scale"
            )
        relative = abs(residual) / scale
        if not math.isfinite(relative):
            raise KerrReturningRadiationThermalProfileError(
                "relative fixed-point residual is not finite"
            )
        relative_values.append(relative)
    maximum = max(relative_values)
    if maximum > MAXIMUM_RELATIVE_EQUATION_RESIDUAL:
        raise KerrReturningRadiationThermalProfileError(
            "fixed-point solution exceeds the non-relaxable thermal-profile "
            f"relative residual limit ({maximum:.17g} > "
            f"{MAXIMUM_RELATIVE_EQUATION_RESIDUAL:.17g})"
        )
    return maximum


def _certified_source_inputs(
    source: Any,
    disk: StationaryNovikovThorneDisk,
) -> tuple[
    AxisymmetricReturningRadiationKernel,
    tuple[float, ...],
    tuple[float, ...],
    Mapping[str, Any],
    str,
]:
    if type(source) is KerrForwardReturningRadiationKernel:
        axisymmetric = verify_and_reduce_kerr_returning_radiation_energy_kernel(
            source
        )
        forward = source
        geometry_source = source
        source_kind = "KerrForwardReturningRadiationKernel"
        cached_binding = None
        authentication_sha = source.model_descriptor_sha256
    elif type(source) is KerrForwardReturningRadiationKernelProjection:
        axisymmetric = (
            verify_and_reduce_kerr_returning_radiation_kernel_projection(source)
        )
        forward = object.__getattribute__(source, "_source")
        if type(forward) is not KerrForwardReturningRadiationKernel:
            raise KerrReturningRadiationThermalProfileVerificationError(
                "projection source is not an exact forward Kerr kernel"
            )
        geometry_source = source
        source_kind = "KerrForwardReturningRadiationKernelProjection"
        cached_binding = None
        authentication_sha = source.model_descriptor_sha256
    elif type(source) is KerrCachedReturningRadiationKernelExecution:
        validated = (
            validate_and_reduce_cached_kerr_returning_radiation_energy_kernel(
                source
            )
        )
        axisymmetric = validated.axisymmetric_kernel
        forward = validated.forward_kernel
        geometry_source = forward
        source_kind = "KerrCachedReturningRadiationKernelExecution"
        cached_binding = dict(validated.scientific_binding())
        cached_binding_sha = validated.scientific_binding_sha256
        authentication_sha = cached_binding_sha
    else:
        raise TypeError(
            "source must be an exact replayable Kerr forward returning kernel "
            "or its authenticated projection, or an exact production-forward "
            "cached execution; an Axisymmetric kernel, receiver cache, test "
            "evaluator, cached projection, or producer string cannot certify "
            "physical provenance"
        )

    surface = object.__getattribute__(forward, "surface")
    source_metric = object.__getattribute__(surface, "metric")
    calibration = object.__getattribute__(surface, "calibration")
    for actual, expected, label in (
        (disk.metric.mass_m, source_metric.mass_m, "metric mass"),
        (disk.metric.spin_a_m, source_metric.spin_a_m, "signed Kerr spin"),
        (
            disk.metric.singularity_guard_m,
            source_metric.singularity_guard_m,
            "metric singularity guard",
        ),
        (
            disk.dimensionless_spin_magnitude,
            calibration.dimensionless_spin,
            "calibration spin magnitude",
        ),
    ):
        if not _same_float(actual, expected):
            raise ValueError(
                f"NT disk {label} does not exactly match the replayed kernel"
            )
    if (
        type(calibration.orientation) is not str
        or calibration.orientation.encode("utf-8")
        != disk.orientation.encode("utf-8")
    ):
        raise ValueError(
            "NT disk orientation does not match the replayed calibration"
        )
    edges = object.__getattribute__(geometry_source, "annulus_edges_over_mass")
    if type(edges) is not tuple:
        raise KerrReturningRadiationThermalProfileVerificationError(
            "replayed source annulus edges have a non-exact type"
        )
    source_isco = calibration.isco_radius_over_mass
    disk_isco = kerr_isco_radius_m(
        disk.dimensionless_spin_magnitude,
        disk.orientation,
    )
    if (
        not _same_float(edges[0], source_isco)
        or not _same_float(edges[0], disk_isco)
    ):
        raise ValueError(
            "replayed kernel, calibration, and NT disk ISCO do not match exactly"
        )
    if not _same_float(edges[-1], calibration.outer_radius_over_mass):
        raise ValueError(
            "replayed kernel outer edge does not match the calibrated R_out"
        )
    upper = object.__getattribute__(
        geometry_source, "upper_annulus_areas_over_mass_squared"
    )
    lower = object.__getattribute__(
        geometry_source, "lower_annulus_areas_over_mass_squared"
    )
    if type(upper) is not tuple or type(lower) is not tuple or len(upper) != len(lower):
        raise KerrReturningRadiationThermalProfileVerificationError(
            "replayed source proper-area arrays have invalid exact shapes"
        )
    areas = tuple(
        0.5
        * _finite_sum(
            (
                _exact_positive_float(up, f"replayed upper area {index}"),
                _exact_positive_float(down, f"replayed lower area {index}"),
            ),
            f"equatorially symmetrized one-face area {index}",
        )
        for index, (up, down) in enumerate(zip(upper, lower))
    )
    source_sha = object.__getattribute__(geometry_source, "_descriptor_sha256")
    _digest(source_sha, "replayed source descriptor SHA-256")
    binding = {
        "automaticGeometryDerivation": True,
        "geometryProvenanceCertified": True,
        "oneFaceAreaDerivation": "arithmetic mean of replayed symmetric faces",
        "sameCodeSourceReplayVerified": True,
        "sourceKernelDescriptorSha256": source_sha,
        "sourceKernelType": source_kind,
        "sourceMetricAndCalibrationMatchedToNovikovThorneDisk": True,
    }
    if cached_binding is not None:
        binding.update(
            {
                "cachedDirectionReplayVerified": True,
                "cachedReplayRetracedRays": False,
                "sourceAuthentication": cached_binding,
                "sourceAuthenticationDescriptorSha256": cached_binding_sha,
            }
        )
    return axisymmetric, tuple(edges), areas, binding, authentication_sha


def _certified_powers(
    *,
    disk: StationaryNovikovThorneDisk,
    areas: tuple[float, ...],
    solution: ReturningRadiationFluxSolution,
) -> tuple[float, float, float, float, float]:
    gravitational_radius_m = (
        GRAVITATIONAL_CONSTANT_M3_KG_S2
        * disk.black_hole_mass_kg
        / (LIGHT_SPEED_M_S * LIGHT_SPEED_M_S)
    )
    physical_areas = tuple(
        area * gravitational_radius_m * gravitational_radius_m for area in areas
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in physical_areas):
        raise KerrReturningRadiationThermalProfileError(
            "physical one-face annulus areas are not finite and positive"
        )
    p0 = _finite_sum(
        tuple(
            area * flux
            for area, flux in zip(physical_areas, solution.intrinsic_flux)
        ),
        "one-face intrinsic power",
    )
    pin = _finite_sum(
        tuple(
            area * flux
            for area, flux in zip(
                physical_areas, solution.incident_returning_flux
            )
        ),
        "one-face incident returning power",
    )
    pout = _finite_sum(
        tuple(
            area * flux for area, flux in zip(physical_areas, solution.outgoing_flux)
        ),
        "one-face outgoing power",
    )
    power_residual = _finite_sum((p0, pin, -pout), "one-face power residual")
    power_tolerance = _finite_sum(
        tuple(
            area * tolerance
            for area, tolerance in zip(
                physical_areas, solution.residual_tolerances
            )
        ),
        "one-face power tolerance",
    )
    roundoff_allowance = 32.0 * math.ulp(max(1.0, abs(p0), abs(pin), abs(pout)))
    if abs(power_residual) > power_tolerance + roundoff_allowance:
        raise KerrReturningRadiationThermalProfileError(
            "area-integrated one-face energy balance exceeds solver tolerance"
        )
    return p0, pin, pout, power_residual, power_tolerance


def _profile_descriptor(
    *,
    kernel: AxisymmetricReturningRadiationKernel,
    disk: StationaryNovikovThorneDisk,
    policy: ReturningRadiationFixedPointPolicy,
    edges: tuple[float, ...],
    areas: tuple[float, ...],
    solution: ReturningRadiationFluxSolution,
    intrinsic_temperatures: tuple[float, ...],
    outgoing_temperatures: tuple[float, ...],
    ratios: tuple[float, ...],
    maximum_relative_residual: float,
    provenance_classification: str,
    provenance_binding: Mapping[str, Any],
    source_authentication_descriptor_sha256: str | None,
    powers: tuple[float, float, float, float, float] | None,
) -> Mapping[str, Any]:
    disk_descriptor = _disk_descriptor(disk)
    power_result: Mapping[str, Any] | None
    if powers is None:
        power_result = None
    else:
        p0, pin, pout, power_residual, power_tolerance = powers
        power_result = {
            "balanceResidual": power_residual,
            "balanceTolerance": power_tolerance,
            "incidentReturning": pin,
            "intrinsic": p0,
            "outgoing": pout,
        }
    return {
        "annuli": {
            "edgesOverMass": edges,
            "properAreaFaceSemantics": "one face; no implicit factor of two",
            "properAreasOverMassSquared": areas,
            "queryBoundaryConvention": (
                "left-closed right-open, with final outer edge included"
            ),
            "representativeRadiiOverMass": kernel.annulus_radii_over_mass,
            "valueModel": "piecewise constant at declared representative radius",
        },
        "assumptions": {
            "absorptionAlbedo": 0.0,
            "colourCorrection": disk.colour_correction,
            "colourCorrectionHeldFixed": True,
            "includesReturningRadiationStressWorkFS": False,
            "includesScatteringOrSolvedAtmosphere": False,
            "includesSpectralRedistribution": False,
            "isGeneralRelativisticMagnetohydrodynamics": False,
            "temperatureLaw": "sigma T_eff^4=F_out",
        },
        "binding": {
            "fixedPointPolicyDescriptorSha256": policy.canonical_descriptor_sha256,
            "kernelDescriptorSha256": kernel.canonical_descriptor_sha256,
            "kernelProducerId": kernel.ray_kernel_producer_id,
            "novikovThorneDisk": disk_descriptor,
            "novikovThorneDiskDescriptorSha256": _sha256_json(disk_descriptor),
            "provenance": provenance_binding,
            "sourceAuthenticationDescriptorSha256": (
                source_authentication_descriptor_sha256
            ),
        },
        "classification": provenance_classification,
        "equation": "F_out=F_0+F_in; F_in=K F_out",
        "implementationId": IMPLEMENTATION_ID,
        "result": {
            "effectiveTemperatureRatio": ratios,
            "fixedPointSolution": solution.canonical_descriptor(),
            "incidentReturningFluxWM2": solution.incident_returning_flux,
            "intrinsicEffectiveTemperatureK": intrinsic_temperatures,
            "intrinsicFluxWM2": solution.intrinsic_flux,
            "maximumAllowedRelativeEquationResidual": (
                MAXIMUM_RELATIVE_EQUATION_RESIDUAL
            ),
            "maximumRelativeEquationResidual": maximum_relative_residual,
            "oneFacePowerW": power_result,
            "outgoingEffectiveTemperatureK": outgoing_temperatures,
            "outgoingFluxWM2": solution.outgoing_flux,
        },
        "units": {
            "flux": "W m^-2 local comoving one-face proper area",
            "physicalAreaScale": (
                "(G M_BH / c^2)^2 only for replay-certified geometry"
            ),
            "power": (
                "W per face only for replay-certified geometry; null for "
                "external-unverified declared areas"
            ),
            "properArea": "M^2 per face",
        },
    }


def _solve_profile(
    kernel: AxisymmetricReturningRadiationKernel,
    *,
    annulus_edges_over_mass: tuple[float, ...],
    annulus_proper_areas_over_mass_squared: tuple[float, ...],
    disk: StationaryNovikovThorneDisk,
    policy: ReturningRadiationFixedPointPolicy,
    certified_source: (
        KerrForwardReturningRadiationKernel
        | KerrForwardReturningRadiationKernelProjection
        | KerrCachedReturningRadiationKernelExecution
        | None
    ),
    provenance_binding: Mapping[str, Any],
    source_authentication_descriptor_sha256: str | None,
) -> AxisymmetricReturningRadiationThermalProfile:
    checked_kernel = _validated_kernel(kernel)
    checked_disk = _validated_disk(disk)
    checked_policy = _validated_policy(policy)
    edges, areas = _validated_geometry(
        checked_kernel,
        checked_disk,
        annulus_edges_over_mass,
        annulus_proper_areas_over_mass_squared,
    )
    states = tuple(
        checked_disk.thermal_state(radius * checked_disk.metric.mass_m)
        for radius in checked_kernel.annulus_radii_over_mass
    )
    intrinsic_flux = tuple(state.surface_flux_w_m2 for state in states)
    if any(value <= 0.0 for value in intrinsic_flux):
        raise KerrReturningRadiationThermalProfileError(
            "every representative radius must have positive finite NT flux"
        )
    try:
        solution = solve_absorbed_returning_radiation(
            checked_kernel,
            intrinsic_flux,
            checked_policy,
        )
    except ReturningRadiationConvergenceError:
        raise
    validate_returning_radiation_solution(
        checked_kernel,
        intrinsic_flux,
        solution,
        checked_policy,
    )
    maximum_relative_residual = _maximum_relative_residual(solution)
    ratios = tuple(
        math.exp(0.25 * (math.log(outgoing) - math.log(intrinsic)))
        for intrinsic, outgoing in zip(
            solution.intrinsic_flux, solution.outgoing_flux
        )
    )
    if any(not math.isfinite(value) or value < 1.0 for value in ratios):
        raise KerrReturningRadiationThermalProfileError(
            "effective-temperature ratios are not finite and at least one"
        )
    intrinsic_temperatures = tuple(state.effective_temperature_k for state in states)
    outgoing_temperatures = tuple(
        intrinsic * ratio
        for intrinsic, ratio in zip(intrinsic_temperatures, ratios)
    )
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in outgoing_temperatures
    ):
        raise KerrReturningRadiationThermalProfileError(
            "outgoing effective temperatures left the finite positive domain"
        )

    if type(certified_source) is KerrCachedReturningRadiationKernelExecution:
        provenance_classification = CACHED_KERR_FORWARD_CLASSIFICATION
    elif certified_source is not None:
        provenance_classification = REPLAYED_KERR_FORWARD_CLASSIFICATION
    else:
        provenance_classification = EXTERNAL_UNVERIFIED_CLASSIFICATION
    powers = (
        _certified_powers(
            disk=checked_disk,
            areas=areas,
            solution=solution,
        )
        if certified_source is not None
        else None
    )
    descriptor = _profile_descriptor(
        kernel=checked_kernel,
        disk=checked_disk,
        policy=checked_policy,
        edges=edges,
        areas=areas,
        solution=solution,
        intrinsic_temperatures=intrinsic_temperatures,
        outgoing_temperatures=outgoing_temperatures,
        ratios=ratios,
        maximum_relative_residual=maximum_relative_residual,
        provenance_classification=provenance_classification,
        provenance_binding=provenance_binding,
        source_authentication_descriptor_sha256=(
            source_authentication_descriptor_sha256
        ),
        powers=powers,
    )
    descriptor_json = _canonical_json(descriptor)
    disk_sha = _sha256_json(_disk_descriptor(checked_disk))
    result = object.__new__(AxisymmetricReturningRadiationThermalProfile)
    p0, pin, pout, power_residual, power_tolerance = (
        powers if powers is not None else (None, None, None, None, None)
    )
    for name, value in (
        ("annulus_edges_over_mass", edges),
        (
            "annulus_representative_radii_over_mass",
            checked_kernel.annulus_radii_over_mass,
        ),
        ("annulus_proper_areas_over_mass_squared", areas),
        ("intrinsic_flux_w_m2", solution.intrinsic_flux),
        ("incident_returning_flux_w_m2", solution.incident_returning_flux),
        ("outgoing_flux_w_m2", solution.outgoing_flux),
        ("effective_temperature_ratio", ratios),
        ("intrinsic_effective_temperature_k", intrinsic_temperatures),
        ("outgoing_effective_temperature_k", outgoing_temperatures),
        ("maximum_relative_equation_residual", maximum_relative_residual),
        (
            "maximum_allowed_relative_equation_residual",
            MAXIMUM_RELATIVE_EQUATION_RESIDUAL,
        ),
        ("provenance_classification", provenance_classification),
        ("geometry_provenance_certified", certified_source is not None),
        ("one_face_intrinsic_power_w", p0),
        ("one_face_incident_returning_power_w", pin),
        ("one_face_outgoing_power_w", pout),
        ("one_face_power_balance_residual_w", power_residual),
        ("one_face_power_balance_tolerance_w", power_tolerance),
        ("kernel_descriptor_sha256", checked_kernel.canonical_descriptor_sha256),
        ("novikov_thorne_disk_descriptor_sha256", disk_sha),
        (
            "fixed_point_policy_descriptor_sha256",
            checked_policy.canonical_descriptor_sha256,
        ),
        (
            "fixed_point_solution_descriptor_sha256",
            solution.canonical_descriptor_sha256,
        ),
        (
            "source_authentication_descriptor_sha256",
            source_authentication_descriptor_sha256,
        ),
        ("_kernel", checked_kernel),
        ("_disk", checked_disk),
        ("_policy", checked_policy),
        ("_certified_source", certified_source),
        ("_descriptor_json", descriptor_json),
        (
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        ),
    ):
        object.__setattr__(result, name, value)
    return result


def solve_axisymmetric_returning_radiation_thermal_profile(
    kernel: AxisymmetricReturningRadiationKernel,
    *,
    annulus_edges_over_mass: tuple[float, ...],
    annulus_proper_areas_over_mass_squared: tuple[float, ...],
    disk: StationaryNovikovThorneDisk,
    policy: ReturningRadiationFixedPointPolicy = DEFAULT_FIXED_POINT_POLICY,
) -> AxisymmetricReturningRadiationThermalProfile:
    """Solve an explicitly external, unverified axisymmetric profile.

    The coefficients, edges, and areas are sealed into the descriptor but are
    not promoted to Kerr ray-tracing provenance.  Consequently this entry
    never publishes integrated physical watts.
    """

    return _solve_profile(
        kernel,
        annulus_edges_over_mass=annulus_edges_over_mass,
        annulus_proper_areas_over_mass_squared=(
            annulus_proper_areas_over_mass_squared
        ),
        disk=disk,
        policy=policy,
        certified_source=None,
        provenance_binding={
            "automaticGeometryDerivation": False,
            "geometryProvenanceCertified": False,
            "sameCodeSourceReplayVerified": False,
            "warning": (
                "kernel coefficients, producer string, edges, and areas are "
                "externally declared and not physical-provenance certified"
            ),
        },
        source_authentication_descriptor_sha256=None,
    )


def solve_certified_kerr_returning_radiation_thermal_profile(
    source: (
        KerrForwardReturningRadiationKernel
        | KerrForwardReturningRadiationKernelProjection
        | KerrCachedReturningRadiationKernelExecution
    ),
    *,
    disk: StationaryNovikovThorneDisk,
    policy: ReturningRadiationFixedPointPolicy = DEFAULT_FIXED_POINT_POLICY,
) -> AxisymmetricReturningRadiationThermalProfile:
    """Authenticate a direct or production-cached Kerr source and its geometry."""

    checked_disk = _validated_disk(disk)
    kernel, edges, areas, binding, authentication_sha = _certified_source_inputs(
        source,
        checked_disk,
    )
    return _solve_profile(
        kernel,
        annulus_edges_over_mass=edges,
        annulus_proper_areas_over_mass_squared=areas,
        disk=checked_disk,
        policy=policy,
        certified_source=source,
        provenance_binding=binding,
        source_authentication_descriptor_sha256=authentication_sha,
    )


def _validated_profile_live_owners(
    profile: AxisymmetricReturningRadiationThermalProfile,
) -> tuple[
    AxisymmetricReturningRadiationKernel,
    StationaryNovikovThorneDisk,
    ReturningRadiationFixedPointPolicy,
]:
    """Deeply canonicalize the three live scientific owners without replay.

    The returned objects are fresh canonical values.  Exact-tree comparison
    against the live owners is intentional: it prevents a frozen dataclass
    changed through ``object.__setattr__`` from retaining a stale private
    descriptor while downstream code authenticates only the profile wrapper.
    Equal independently reconstructed owners remain valid here; process-local
    object-identity policy belongs to the frame authority.
    """

    if type(profile) is not AxisymmetricReturningRadiationThermalProfile:
        raise TypeError(
            "profile must be an exact AxisymmetricReturningRadiationThermalProfile"
        )
    live_kernel = object.__getattribute__(profile, "_kernel")
    live_disk = object.__getattribute__(profile, "_disk")
    live_policy = object.__getattribute__(profile, "_policy")
    checked_kernel = _validated_kernel(live_kernel)
    checked_disk = _validated_disk(live_disk)
    checked_policy = _validated_policy(live_policy)
    _require_exact_tree(
        live_kernel,
        checked_kernel,
        "profile._kernel/canonical_live_kernel",
    )
    _require_exact_tree(
        live_disk,
        checked_disk,
        "profile._disk/canonical_live_disk",
    )
    _require_exact_tree(
        live_policy,
        checked_policy,
        "profile._policy/canonical_live_policy",
    )
    return checked_kernel, checked_disk, checked_policy


def verify_axisymmetric_returning_radiation_thermal_profile(
    profile: AxisymmetricReturningRadiationThermalProfile,
) -> None:
    """Rebuild a profile and reject changed inputs, outputs, or bindings."""

    if type(profile) is not AxisymmetricReturningRadiationThermalProfile:
        raise TypeError(
            "profile must be an exact AxisymmetricReturningRadiationThermalProfile"
        )
    try:
        # Validate the live owners *before* an expensive direct or cached
        # source replay.  In particular, a stale axisymmetric-kernel
        # descriptor must fail without reading cache receipts or tracing rays.
        checked_kernel, checked_disk, checked_policy = (
            _validated_profile_live_owners(profile)
        )
        certified_source = object.__getattribute__(profile, "_certified_source")
        if certified_source is None:
            rebuilt = solve_axisymmetric_returning_radiation_thermal_profile(
                checked_kernel,
                annulus_edges_over_mass=object.__getattribute__(
                    profile, "annulus_edges_over_mass"
                ),
                annulus_proper_areas_over_mass_squared=object.__getattribute__(
                    profile, "annulus_proper_areas_over_mass_squared"
                ),
                disk=checked_disk,
                policy=checked_policy,
            )
        else:
            (
                expected_kernel,
                edges,
                areas,
                provenance_binding,
                authentication_sha,
            ) = _certified_source_inputs(certified_source, checked_disk)
            _require_exact_tree(
                checked_kernel,
                expected_kernel,
                "profile._kernel/certified_source.axisymmetric_reduction",
            )
            rebuilt = _solve_profile(
                expected_kernel,
                annulus_edges_over_mass=edges,
                annulus_proper_areas_over_mass_squared=areas,
                disk=checked_disk,
                policy=checked_policy,
                certified_source=certified_source,
                provenance_binding=provenance_binding,
                source_authentication_descriptor_sha256=authentication_sha,
            )
    except (TypeError, ValueError, RuntimeError) as error:
        raise KerrReturningRadiationThermalProfileVerificationError(
            f"profile could not be reproduced: {error}"
        ) from error
    supplied_json = object.__getattribute__(profile, "_descriptor_json")
    supplied_sha = object.__getattribute__(profile, "_descriptor_sha256")
    if type(supplied_json) is not str or type(supplied_sha) is not str:
        raise KerrReturningRadiationThermalProfileVerificationError(
            "profile descriptor storage has non-exact types"
        )
    if (
        supplied_json.encode("utf-8")
        != rebuilt._descriptor_json.encode("utf-8")
        or supplied_sha.encode("ascii") != rebuilt._descriptor_sha256.encode("ascii")
    ):
        raise KerrReturningRadiationThermalProfileVerificationError(
            "profile fields do not match deterministic same-code replay"
        )
    public_fields = (
        "annulus_edges_over_mass",
        "annulus_representative_radii_over_mass",
        "annulus_proper_areas_over_mass_squared",
        "intrinsic_flux_w_m2",
        "incident_returning_flux_w_m2",
        "outgoing_flux_w_m2",
        "effective_temperature_ratio",
        "intrinsic_effective_temperature_k",
        "outgoing_effective_temperature_k",
        "maximum_relative_equation_residual",
        "maximum_allowed_relative_equation_residual",
        "provenance_classification",
        "geometry_provenance_certified",
        "one_face_intrinsic_power_w",
        "one_face_incident_returning_power_w",
        "one_face_outgoing_power_w",
        "one_face_power_balance_residual_w",
        "one_face_power_balance_tolerance_w",
        "kernel_descriptor_sha256",
        "novikov_thorne_disk_descriptor_sha256",
        "fixed_point_policy_descriptor_sha256",
        "fixed_point_solution_descriptor_sha256",
        "source_authentication_descriptor_sha256",
    )
    for name in public_fields:
        actual = object.__getattribute__(profile, name)
        expected = object.__getattribute__(rebuilt, name)
        _require_exact_tree(actual, expected, f"profile.{name}")
    for name in (
        "kernel_descriptor_sha256",
        "novikov_thorne_disk_descriptor_sha256",
        "fixed_point_policy_descriptor_sha256",
        "fixed_point_solution_descriptor_sha256",
    ):
        _digest(object.__getattribute__(profile, name), f"profile.{name}")
    source_authentication_sha = object.__getattribute__(
        profile,
        "source_authentication_descriptor_sha256",
    )
    if source_authentication_sha is not None:
        _digest(
            source_authentication_sha,
            "profile.source_authentication_descriptor_sha256",
        )


def _edge_groups(
    fine_edges: tuple[float, ...], merged_value: Any
) -> tuple[tuple[float, ...], tuple[tuple[int, ...], ...]]:
    merged_raw = _exact_tuple(merged_value, "merged_annulus_edges_over_mass")
    if len(merged_raw) < 2:
        raise ValueError("merged annulus edges must contain at least two values")
    merged = tuple(
        _exact_positive_float(value, f"merged annulus edge {index}")
        for index, value in enumerate(merged_raw)
    )
    if any(right <= left for left, right in zip(merged, merged[1:])):
        raise ValueError("merged annulus edges must be strictly increasing")
    if not _same_float(merged[0], fine_edges[0]) or not _same_float(
        merged[-1], fine_edges[-1]
    ):
        raise ValueError("merged annuli must preserve the complete radial domain")
    fine_indices = {value.hex(): index for index, value in enumerate(fine_edges)}
    try:
        boundaries = tuple(fine_indices[value.hex()] for value in merged)
    except KeyError as error:
        raise ValueError("merged edges must be an exact subset of fine edges") from error
    groups = tuple(
        tuple(range(left, right))
        for left, right in zip(boundaries, boundaries[1:])
    )
    if any(not group for group in groups):
        raise ValueError("every merged annulus must contain a fine annulus")
    return merged, groups


def _coarsen_profile(
    source: AxisymmetricReturningRadiationThermalProfile,
    merged_value: Any,
) -> AxisymmetricReturningRadiationThermalProfileProjection:
    merged, groups = _edge_groups(source.annulus_edges_over_mass, merged_value)
    areas = tuple(
        math.fsum(source.annulus_proper_areas_over_mass_squared[index] for index in group)
        for group in groups
    )

    def averages(values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(
            math.fsum(
                source.annulus_proper_areas_over_mass_squared[index] * values[index]
                for index in group
            )
            / area
            for group, area in zip(groups, areas)
        )

    intrinsic = averages(source.intrinsic_flux_w_m2)
    incident = averages(source.incident_returning_flux_w_m2)
    outgoing = averages(source.outgoing_flux_w_m2)
    ratios = tuple(
        math.exp(0.25 * (math.log(out_value) - math.log(in_value)))
        for in_value, out_value in zip(intrinsic, outgoing)
    )
    representatives = tuple(
        0.5 * math.fsum((inner, outer))
        for inner, outer in zip(merged, merged[1:])
    )
    descriptor = {
        "annuli": {
            "edgesOverMass": merged,
            "properAreasOverMassSquared": areas,
            "representativeRadiiOverMass": representatives,
            "valueModel": "one-face proper-area-weighted reporting projection",
        },
        "implementationId": PROJECTION_IMPLEMENTATION_ID,
        "isResolvedCoarseFixedPoint": False,
        "sourceGeometryProvenanceCertified": (
            source.geometry_provenance_certified
        ),
        "sourceProvenanceClassification": source.provenance_classification,
        "result": {
            "effectiveTemperatureRatio": ratios,
            "incidentReturningFluxWM2": incident,
            "intrinsicFluxWM2": intrinsic,
            "outgoingFluxWM2": outgoing,
        },
        "sourceProfileDescriptorSha256": source.model_descriptor_sha256,
        "warning": (
            "area-averaging a solved fine profile is not a new coarse-kernel "
            "fixed-point solve"
        ),
    }
    descriptor_json = _canonical_json(descriptor)
    result = object.__new__(AxisymmetricReturningRadiationThermalProfileProjection)
    for name, value in (
        ("annulus_edges_over_mass", merged),
        ("annulus_representative_radii_over_mass", representatives),
        ("annulus_proper_areas_over_mass_squared", areas),
        ("intrinsic_flux_w_m2", intrinsic),
        ("incident_returning_flux_w_m2", incident),
        ("outgoing_flux_w_m2", outgoing),
        ("effective_temperature_ratio", ratios),
        ("source_profile_descriptor_sha256", source.model_descriptor_sha256),
        ("is_resolved_coarse_fixed_point", False),
        ("_source", source),
        ("_descriptor_json", descriptor_json),
        (
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        ),
    ):
        object.__setattr__(result, name, value)
    return result


def verify_axisymmetric_returning_radiation_thermal_profile_projection(
    projection: AxisymmetricReturningRadiationThermalProfileProjection,
) -> None:
    if type(projection) is not AxisymmetricReturningRadiationThermalProfileProjection:
        raise TypeError(
            "projection must be an exact thermal-profile projection"
        )
    try:
        source = object.__getattribute__(projection, "_source")
        verify_axisymmetric_returning_radiation_thermal_profile(source)
        rebuilt = _coarsen_profile(
            source,
            object.__getattribute__(projection, "annulus_edges_over_mass"),
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise KerrReturningRadiationThermalProfileVerificationError(
            f"projection could not be reproduced: {error}"
        ) from error
    for name in (
        "annulus_representative_radii_over_mass",
        "annulus_proper_areas_over_mass_squared",
        "intrinsic_flux_w_m2",
        "incident_returning_flux_w_m2",
        "outgoing_flux_w_m2",
        "effective_temperature_ratio",
        "source_profile_descriptor_sha256",
        "is_resolved_coarse_fixed_point",
        "_descriptor_json",
        "_descriptor_sha256",
    ):
        actual = object.__getattribute__(projection, name)
        expected = object.__getattribute__(rebuilt, name)
        _require_exact_tree(actual, expected, f"projection.{name}")


__all__ = [
    "AxisymmetricReturningRadiationThermalProfile",
    "AxisymmetricReturningRadiationThermalProfileProjection",
    "AxisymmetricThermalAnnulusValue",
    "CACHED_KERR_FORWARD_CLASSIFICATION",
    "CERTIFIED_KERR_FORWARD_CLASSIFICATIONS",
    "EXTERNAL_UNVERIFIED_CLASSIFICATION",
    "IMPLEMENTATION_ID",
    "KerrReturningRadiationThermalProfileError",
    "KerrReturningRadiationThermalProfileVerificationError",
    "MAXIMUM_RELATIVE_EQUATION_RESIDUAL",
    "PROJECTION_IMPLEMENTATION_ID",
    "REPLAYED_KERR_FORWARD_CLASSIFICATION",
    "SCIENTIFIC_STATUS",
    "solve_axisymmetric_returning_radiation_thermal_profile",
    "solve_certified_kerr_returning_radiation_thermal_profile",
    "verify_axisymmetric_returning_radiation_thermal_profile",
    "verify_axisymmetric_returning_radiation_thermal_profile_projection",
]
