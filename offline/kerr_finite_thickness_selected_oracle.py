"""Independent selected-ray oracle for the prescribed finite Kerr faces.

The geodesic is evolved in canonical Boyer--Lindquist coordinates with the
fixed-step classical RK4 implementation owned by :mod:`kerr_selected_oracle`.
This module never calls the production Cartesian Kerr--Schild geodesic
integrator, accepted-step multi-surface locator, finite-face emitter, spectral
transfer, or frame sampler.

Upper and lower Zhou/Taylor--Reynolds photospheres, their transparent radial
continuations, the actual-event circular-velocity matter frame, the face
normal, ``g``, signed ``mu``, and KERRBB-D20 transfer are independently
implemented here.  The thermal calculation intentionally shares only the
public Page--Thorne radial scalar and fundamental physical constants with the
existing thin selected-ray oracle.  The BL RK4 dynamics are also shared with
that *independent* oracle, not with production.

This is a finite list of selected calibration rays with explicit ``h`` versus
``h/2`` diagnostics.  It is not a full-frame proof, a surface-complete ray
bundle, an independent Page--Thorne derivation, hydrostatic vertical
structure, an atmosphere, returning radiation, NR, GRMHD, or complete GRRT.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import sys
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping, NoReturn, Sequence

from offline.kerr_selected_oracle import (
    BOLTZMANN_CONSTANT_J_K,
    GRAVITATIONAL_CONSTANT_M3_KG_S2,
    LIGHT_SPEED_M_S,
    PLANCK_CONSTANT_J_S,
    STEFAN_BOLTZMANN_W_M2_K4,
    FixedRk4Options,
    KerrPhotonConstants,
    _hamiltonian_residual as _shared_bl_hamiltonian_residual,
    _locate_crossing as _shared_partial_rk4_bisection,
    _rk4_step as _shared_bl_rk4_step,
    photon_constants,
)
from offline.novikov_thorne import page_thorne_flux_shape


SUPPORTED_SAMPLER_IMPLEMENTATION_ID: Final = (
    "kerr-finite-thickness-spectral-ray-sampler/v1"
)
SELECTED_ORACLE_IMPLEMENTATION_ID: Final = (
    "independent-bl-finite-thickness-fixed-rk4-selected-rays/v1"
)
UPPER: Final = "upper"
LOWER: Final = "lower"
_FACES: Final = (UPPER, LOWER)
_EXPECTED_SURFACE_IDS: Final = (
    "kerr-finite-thickness-lower-photosphere",
    "kerr-finite-thickness-upper-photosphere",
)
_ROOT_KEYS: Final = frozenset(
    {
        "convergence",
        "convergencePolicy",
        "diskThermalProxy",
        "escapeDirectionDiagnostic",
        "escapedObserverSpectrum",
        "finiteThicknessSurface",
        "frequencyTransfer",
        "implementationId",
        "metric",
        "observer",
        "observerFrequencyFrame",
        "rayOptions",
        "scientificStatus",
        "screenConvention",
        "surfaceOptions",
        "termination",
        "tolerancePolicy",
        "traceAccuracyPolicy",
        "version",
    }
)
_METRIC_KEYS: Final = frozenset(
    {"massM", "signedSpinAM", "singularityGuardM", "sourceId", "timeDependent"}
)
_OBSERVER_KEYS: Final = frozenset(
    {
        "coordinateTimeM",
        "event",
        "fourVelocity",
        "materialClearance",
        "phiKsRad",
        "radiusM",
        "thetaRad",
        "type",
    }
)
_MATERIAL_CLEARANCE_KEYS: Final = frozenset(
    {
        "lowerFaceSignedValue",
        "policy",
        "pseudoCylindricalRadiusOverMass",
        "status",
        "upperFaceSignedValue",
        "withinPhysicalAnnulus",
    }
)
_TERMINATION_KEYS: Final = frozenset(
    {
        "captureRadiusM",
        "captureTargetId",
        "escapeRadiusM",
        "escapeTargetId",
        "spinAM",
        "visibilityConstraints",
    }
)
_VISIBILITY_KEYS: Final = frozenset(
    {
        "captureStrictlyInsideDiskIsco",
        "escapeStrictlyOutsideMaximumPhotosphereOblateRadius",
        "maximumPhotosphereOblateRadiusM",
    }
)
_SURFACE_KEYS: Final = frozenset(
    {
        "dimensionlessSpinMagnitude",
        "eddingtonScaledMassAccretionRate",
        "heightRateIsIndependentOfThermalRate",
        "maximumPhotosphereOblateRadiusM",
        "orientation",
        "outerRadiusOverMass",
        "surfaceIds",
        "thinnessGateMaximumHOverRho",
        "type",
    }
)
_THERMAL_KEYS: Final = frozenset(
    {
        "blackHoleMassKg",
        "colourCorrection",
        "iscoRadiusM",
        "massAccretionRateKgS",
        "orientation",
        "radialReference",
    }
)
_SCREEN_KEYS: Final = frozenset(
    {"projection", "screenX", "screenY", "viewForward"}
)
_SCIENTIFIC_STATUS_KEYS: Final = frozenset(
    {
        "captureBoundary",
        "classification",
        "escapeBoundary",
        "fineCoarseWholeRayConvergence",
        "heightFluxRateBinding",
        "implementationId",
        "includesFineCoarseWholeRayConvergence",
        "includesReturningRadiation",
        "includesSolvedAtmosphere",
        "isCompleteGeneralRelativisticRadiativeTransfer",
        "isGeneralRelativisticMagnetohydrodynamics",
        "isHydrostaticVerticalStructureSolution",
        "isOffEquatorialGeodesicDisk",
        "isSachsJacobiRayBundle",
        "multiSurfaceTopologyCompared",
        "observerMaterialPolicy",
        "prohibitedClaim",
        "signedFaceEmissionCosineCompared",
        "spacetime",
        "surface",
        "thermalReference",
    }
)
_EXPECTED_SCREEN: Final = MappingProxyType(
    {
        "projection": "pinhole",
        "screenX": "ZAMO-right-negative-azimuthal",
        "screenY": "ZAMO-up-negative-polar",
        "viewForward": "ZAMO-negative-radial",
    }
)

SelectedOutcome = Literal[
    "upper",
    "lower",
    "captured",
    "escaped",
    "unresolved",
]

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "independent selected-ray canonical-BL fixed-RK4 oracle for the "
            "prescribed upper/lower finite Kerr photospheres"
        ),
        "implementationId": SELECTED_ORACLE_IMPLEMENTATION_ID,
        "supportedProductionSampler": SUPPORTED_SAMPLER_IMPLEMENTATION_ID,
        "sharedBlFramework": (
            "offline.kerr_selected_oracle independent canonical-BL analytic "
            "Hamiltonian derivatives, fixed RK4, and partial-RK4 bisection"
        ),
        "sharedPageThorneRadialScalar": True,
        "sharedFundamentalPhysicalConstants": True,
        "usesProductionKerrSchildGeodesic": False,
        "usesProductionAcceptedStepSurfaceLocator": False,
        "usesProductionFiniteFaceEmitter": False,
        "usesProductionFiniteThicknessTransfer": False,
        "usesProductionFiniteThicknessFrameSampler": False,
        "independentlyImplementsFinitePhotosphere": True,
        "independentlyImplementsOffEquatorialEmitterAndNormal": True,
        "independentlyImplementsGAndSignedMu": True,
        "independentlyImplementsPlanckD20Transfer": True,
        "requiresHAndHalfH": True,
        "isFullFrameProof": False,
        "isIndependentPageThorneDerivation": False,
        "isHydrostaticVerticalStructureSolution": False,
        "includesReturningRadiation": False,
        "includesSolvedAtmosphere": False,
        "isNumericalRelativity": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "isCompleteGeneralRelativisticRadiativeTransfer": False,
        "prohibitedClaim": (
            "Do not describe selected same-model calibration rays as a full "
            "frame proof, independent Page-Thorne derivation, hydrostatic "
            "solution, atmosphere, returning-radiation model, NR, GRMHD, or "
            "complete GRRT calculation."
        ),
    }
)

_EXPECTED_PRODUCTION_SCIENTIFIC_STATUS: Final = MappingProxyType(
    {
        "captureBoundary": "exactly black",
        "classification": (
            "independently fine/coarse converged exact-Kerr scalar ray for a "
            "stationary phenomenological finite-height photosphere"
        ),
        "escapeBoundary": "closed built-in observer-frame spectrum",
        "fineCoarseWholeRayConvergence": True,
        "heightFluxRateBinding": (
            "dimensionless height calibration rate and SI thermal disk rate "
            "are independently caller-supplied and are not silently equated"
        ),
        "implementationId": SUPPORTED_SAMPLER_IMPLEMENTATION_ID,
        "includesFineCoarseWholeRayConvergence": True,
        "includesReturningRadiation": False,
        "includesSolvedAtmosphere": False,
        "isCompleteGeneralRelativisticRadiativeTransfer": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "isHydrostaticVerticalStructureSolution": False,
        "isOffEquatorialGeodesicDisk": False,
        "isSachsJacobiRayBundle": False,
        "multiSurfaceTopologyCompared": True,
        "observerMaterialPolicy": (
            "observer may lie over the physical radial annulus only when "
            "strictly outside both photosphere faces"
        ),
        "prohibitedClaim": (
            "Do not describe this prescribed finite surface plus "
            "equatorial-NT proxy as hydrostatic structure, returning "
            "radiation, a solved atmosphere, GRMHD, complete GRRT, or a "
            "Sachs/Jacobi ray bundle."
        ),
        "signedFaceEmissionCosineCompared": True,
        "spacetime": "exact stationary Kerr in Cartesian Kerr-Schild coordinates",
        "surface": "Zhou prescribed stationary finite-height photosphere",
        "thermalReference": (
            "equatorial Novikov-Thorne/Page-Thorne spectrum at matching "
            "pseudo-cylindrical radius"
        ),
    }
)


class KerrFiniteThicknessSelectedOracleError(RuntimeError):
    """Fail-closed independent selected-ray configuration or trace error."""


def _fail(path: str, message: str) -> NoReturn:
    raise KerrFiniteThicknessSelectedOracleError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        _fail(path, "expected an exact object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    path: str,
) -> None:
    if any(type(key) is not str for key in value):
        _fail(path, "object keys must be exact strings")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        _fail(
            path,
            f"exact keys disagree; missing={missing!r}, unexpected={unexpected!r}",
        )


def _exact_json_tree(value: Any, path: str) -> None:
    """Reject Python subclasses and non-JSON nodes before policy parsing."""

    if type(value) is dict:
        if any(type(key) is not str for key in value):
            _fail(path, "object keys must be exact strings")
        for key, entry in value.items():
            _exact_json_tree(entry, f"{path}.{key}")
        return
    if type(value) is list:
        for index, entry in enumerate(value):
            _exact_json_tree(entry, f"{path}[{index}]")
        return
    if type(value) in (str, bool, int, type(None)):
        return
    if type(value) is float and math.isfinite(value):
        return
    _fail(path, "expected an exact finite JSON primitive, array, or object")


def _number(value: Any, path: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        _fail(path, "expected a finite built-in number")
    return float(value)


def _string(value: Any, path: str) -> str:
    if type(value) is not str or not value:
        _fail(path, "expected a non-empty exact string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "expected an exact boolean")
    return value


def _expected_primitives(
    value: Mapping[str, Any],
    expected: Mapping[str, str | bool],
    path: str,
) -> None:
    for key, expected_value in expected.items():
        actual = value.get(key)
        if type(actual) is not type(expected_value) or actual != expected_value:
            _fail(
                f"{path}.{key}",
                f"expected exact {expected_value!r}",
            )


def _sequence(value: Any, length: int, path: str) -> tuple[float, ...]:
    if type(value) is not list or len(value) != length:
        _fail(path, f"expected exactly {length} finite numbers")
    return tuple(
        _number(entry, f"{path}[{index}]")
        for index, entry in enumerate(value)
    )


def _close(first: float, second: float, tolerance: float = 2.0e-11) -> bool:
    return abs(first - second) <= tolerance * max(1.0, abs(first), abs(second))


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _independent_isco_radius(spin_magnitude: float, orientation: str) -> float:
    if not 0.0 <= spin_magnitude < 1.0:
        raise ValueError("spin magnitude must satisfy 0 <= a/M < 1")
    if orientation not in ("prograde", "retrograde"):
        raise ValueError("unsupported orientation")
    sign = 1.0 if orientation == "prograde" else -1.0
    z1 = 1.0 + (1.0 - spin_magnitude * spin_magnitude) ** (1.0 / 3.0) * (
        (1.0 + spin_magnitude) ** (1.0 / 3.0)
        + (1.0 - spin_magnitude) ** (1.0 / 3.0)
    )
    z2 = math.sqrt(3.0 * spin_magnitude * spin_magnitude + z1 * z1)
    return 3.0 + z2 - sign * math.sqrt(
        max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    )


def _equatorial_orbit(
    radius: float,
    signed_spin: float,
    orientation: str,
) -> tuple[float, float, float]:
    """Return signed-axis Omega, E, and Lz at matching cylindrical rho."""

    spin_magnitude = abs(signed_spin)
    orbit_sign = 1.0 if orientation == "prograde" else -1.0
    spin_axis_sign = -1.0 if signed_spin < 0.0 else 1.0
    root = math.sqrt(radius)
    radius_three_halves = radius * root
    radicand = radius_three_halves - 3.0 * root + 2.0 * orbit_sign * spin_magnitude
    if radicand <= 0.0:
        raise KerrFiniteThicknessSelectedOracleError(
            "equatorial circular reference is not timelike"
        )
    denominator = radius**0.75 * math.sqrt(radicand)
    relative_omega = orbit_sign / (
        radius_three_halves + orbit_sign * spin_magnitude
    )
    energy = (
        radius_three_halves - 2.0 * root + orbit_sign * spin_magnitude
    ) / denominator
    relative_lz = orbit_sign * (
        radius * radius
        - 2.0 * orbit_sign * spin_magnitude * root
        + spin_magnitude * spin_magnitude
    ) / denominator
    return (
        spin_axis_sign * relative_omega,
        energy,
        spin_axis_sign * relative_lz,
    )


@dataclass(frozen=True, slots=True)
class KerrFiniteThicknessSelectedConfiguration:
    mass_m: float
    spin_a_m: float
    observer_radius_m: float
    observer_theta_rad: float
    observer_phi_ks_rad: float
    capture_radius_m: float
    escape_radius_m: float
    isco_radius_m: float
    outer_radius_over_mass: float
    orientation: str
    height_accretion_rate_eddington: float
    thinness_gate_maximum_h_over_rho: float
    black_hole_mass_kg: float
    thermal_mass_accretion_rate_kg_s: float
    colour_correction: float
    sampler_descriptor_sha256: str

    @property
    def dimensionless_spin(self) -> float:
        return self.spin_a_m / self.mass_m

    @property
    def isco_radius_over_mass(self) -> float:
        return self.isco_radius_m / self.mass_m

    @property
    def disk_outer_radius_m(self) -> float:
        return self.outer_radius_over_mass * self.mass_m

    @property
    def radiative_efficiency(self) -> float:
        _omega, energy, _lz = _equatorial_orbit(
            self.isco_radius_over_mass,
            self.dimensionless_spin,
            self.orientation,
        )
        efficiency = 1.0 - energy
        if not math.isfinite(efficiency) or efficiency <= 0.0:
            raise KerrFiniteThicknessSelectedOracleError(
                "ISCO radiative efficiency is not finite and positive"
            )
        return efficiency

    @property
    def asymptotic_photosphere_height_over_mass(self) -> float:
        return (
            3.0
            * self.height_accretion_rate_eddington
            / self.radiative_efficiency
        )

    def photosphere_height_over_mass(self, rho: float) -> float:
        if not self.isco_radius_over_mass <= rho <= self.outer_radius_over_mass:
            raise ValueError("physical photosphere rho lies outside ISCO..Rout")
        return self.asymptotic_photosphere_height_over_mass * (
            1.0 - math.sqrt(self.isco_radius_over_mass / rho)
        )

    def photosphere_height_derivative(self, rho: float) -> float:
        if not self.isco_radius_over_mass <= rho <= self.outer_radius_over_mass:
            raise ValueError("physical photosphere rho lies outside ISCO..Rout")
        return (
            0.5
            * self.asymptotic_photosphere_height_over_mass
            * math.sqrt(self.isco_radius_over_mass)
            / rho**1.5
        )

    def auxiliary_height_over_mass(self, rho: float) -> float:
        if not math.isfinite(rho) or rho < 0.0:
            raise ValueError("auxiliary rho must be finite and non-negative")
        if rho < self.isco_radius_over_mass:
            return self.photosphere_height_derivative(
                self.isco_radius_over_mass
            ) * (self.isco_radius_over_mass - rho)
        if rho > self.outer_radius_over_mass:
            return self.photosphere_height_over_mass(
                self.outer_radius_over_mass
            )
        return self.photosphere_height_over_mass(rho)


def configuration_from_sampler_descriptor(
    descriptor: Mapping[str, Any],
) -> KerrFiniteThicknessSelectedConfiguration:
    """Strictly parse and independently cross-check a finite sampler v1."""

    raw = _mapping(descriptor, "$.sampler.descriptor")
    _exact_json_tree(raw, "$.sampler.descriptor")
    _exact_keys(raw, _ROOT_KEYS, "$.sampler.descriptor")
    implementation_id = _string(
        raw.get("implementationId"),
        "$.sampler.descriptor.implementationId",
    )
    if implementation_id != SUPPORTED_SAMPLER_IMPLEMENTATION_ID:
        _fail(
            "$.sampler.descriptor.implementationId",
            f"only {SUPPORTED_SAMPLER_IMPLEMENTATION_ID!r} is supported",
        )
    if type(raw.get("version")) is not int or raw.get("version") != 1:
        _fail("$.sampler.descriptor.version", "only version 1 is supported")
    try:
        metric = _mapping(raw["metric"], "$.sampler.descriptor.metric")
        observer = _mapping(raw["observer"], "$.sampler.descriptor.observer")
        material_clearance = _mapping(
            observer["materialClearance"],
            "$.sampler.descriptor.observer.materialClearance",
        )
        termination = _mapping(
            raw["termination"],
            "$.sampler.descriptor.termination",
        )
        visibility = _mapping(
            termination["visibilityConstraints"],
            "$.sampler.descriptor.termination.visibilityConstraints",
        )
        surface = _mapping(
            raw["finiteThicknessSurface"],
            "$.sampler.descriptor.finiteThicknessSurface",
        )
        thermal = _mapping(
            raw["diskThermalProxy"],
            "$.sampler.descriptor.diskThermalProxy",
        )
        screen = _mapping(
            raw["screenConvention"],
            "$.sampler.descriptor.screenConvention",
        )
        status = _mapping(
            raw["scientificStatus"],
            "$.sampler.descriptor.scientificStatus",
        )
    except KeyError as error:
        _fail("$.sampler.descriptor", f"missing field {error.args[0]!r}")

    _exact_keys(metric, _METRIC_KEYS, "$.sampler.descriptor.metric")
    _exact_keys(observer, _OBSERVER_KEYS, "$.sampler.descriptor.observer")
    _exact_keys(
        material_clearance,
        _MATERIAL_CLEARANCE_KEYS,
        "$.sampler.descriptor.observer.materialClearance",
    )
    _exact_keys(termination, _TERMINATION_KEYS, "$.sampler.descriptor.termination")
    _exact_keys(
        visibility,
        _VISIBILITY_KEYS,
        "$.sampler.descriptor.termination.visibilityConstraints",
    )
    _exact_keys(
        surface,
        _SURFACE_KEYS,
        "$.sampler.descriptor.finiteThicknessSurface",
    )
    _exact_keys(
        thermal,
        _THERMAL_KEYS,
        "$.sampler.descriptor.diskThermalProxy",
    )
    _exact_keys(screen, _SCREEN_KEYS, "$.sampler.descriptor.screenConvention")
    _exact_keys(
        status,
        _SCIENTIFIC_STATUS_KEYS,
        "$.sampler.descriptor.scientificStatus",
    )
    _expected_primitives(
        screen,
        _EXPECTED_SCREEN,
        "$.sampler.descriptor.screenConvention",
    )
    _expected_primitives(
        status,
        _EXPECTED_PRODUCTION_SCIENTIFIC_STATUS,
        "$.sampler.descriptor.scientificStatus",
    )
    if _string(
        raw.get("observerFrequencyFrame"),
        "$.sampler.descriptor.observerFrequencyFrame",
    ) != "observer-ZAMO":
        _fail(
            "$.sampler.descriptor.observerFrequencyFrame",
            "unsupported observer frequency frame",
        )
    if _string(metric.get("sourceId"), "$.metric.sourceId") != (
        "analytic-kerr-kerr-schild"
    ):
        _fail("$.metric.sourceId", "unsupported metric")
    if _boolean(metric.get("timeDependent"), "$.metric.timeDependent"):
        _fail("$.metric.timeDependent", "metric must be stationary")
    mass = _number(metric.get("massM"), "$.metric.massM")
    spin = _number(metric.get("signedSpinAM"), "$.metric.signedSpinAM")
    singularity_guard = _number(
        metric.get("singularityGuardM"),
        "$.metric.singularityGuardM",
    )
    if mass <= 0.0 or abs(spin) >= mass or singularity_guard <= 0.0:
        _fail("$.metric", "requires M>0 and |a|<M")

    if _string(observer.get("type"), "$.observer.type") != "Boyer-Lindquist-ZAMO":
        _fail("$.observer.type", "unsupported observer")
    observer_radius = _number(observer.get("radiusM"), "$.observer.radiusM")
    observer_theta = _number(observer.get("thetaRad"), "$.observer.thetaRad")
    observer_phi = _number(observer.get("phiKsRad"), "$.observer.phiKsRad")
    observer_time = _number(
        observer.get("coordinateTimeM"),
        "$.observer.coordinateTimeM",
    )
    observer_event = _sequence(observer.get("event"), 4, "$.observer.event")
    observer_four_velocity = _sequence(
        observer.get("fourVelocity"),
        4,
        "$.observer.fourVelocity",
    )
    if not 0.0 < observer_theta < math.pi or abs(math.cos(observer_theta)) <= 1e-10:
        _fail("$.observer.thetaRad", "observer is degenerate")
    sine = math.sin(observer_theta)
    expected_event = (
        observer_time,
        (observer_radius * math.cos(observer_phi) - spin * math.sin(observer_phi))
        * sine,
        (observer_radius * math.sin(observer_phi) + spin * math.cos(observer_phi))
        * sine,
        observer_radius * math.cos(observer_theta),
    )
    if any(
        not _close(actual, expected)
        for actual, expected in zip(observer_event, expected_event)
    ):
        _fail("$.observer.event", "event disagrees with BL ZAMO coordinates")
    cosine = math.cos(observer_theta)
    sigma = observer_radius * observer_radius + spin * spin * cosine * cosine
    delta = (
        observer_radius * observer_radius
        - 2.0 * mass * observer_radius
        + spin * spin
    )
    big_a = (
        (observer_radius * observer_radius + spin * spin) ** 2
        - spin * spin * delta * sine * sine
    )
    if delta <= 0.0 or big_a <= 0.0:
        _fail("$.observer", "ZAMO lies outside the supported Kerr exterior")
    lapse = math.sqrt(sigma * delta / big_a)
    frame_dragging = 2.0 * mass * spin * observer_radius / big_a
    expected_four_velocity = (
        1.0 / lapse,
        -observer_event[2] * frame_dragging / lapse,
        observer_event[1] * frame_dragging / lapse,
        0.0,
    )
    if any(
        not _close(actual, expected, 4.0e-12)
        for actual, expected in zip(
            observer_four_velocity,
            expected_four_velocity,
        )
    ):
        _fail(
            "$.observer.fourVelocity",
            "four-velocity disagrees with the independently rebuilt KS ZAMO",
        )

    termination_spin = _number(termination.get("spinAM"), "$.termination.spinAM")
    capture = _number(
        termination.get("captureRadiusM"),
        "$.termination.captureRadiusM",
    )
    escape = _number(
        termination.get("escapeRadiusM"),
        "$.termination.escapeRadiusM",
    )
    if not _close(termination_spin, spin) or not 0.0 < capture < observer_radius < escape:
        _fail("$.termination", "worldtube geometry is inconsistent")
    if _string(termination.get("escapeTargetId"), "$.termination.escapeTargetId") != (
        "analytic-kerr-escape-worldtube"
    ):
        _fail("$.termination.escapeTargetId", "unsupported escape target")
    capture_target = _string(
        termination.get("captureTargetId"),
        "$.termination.captureTargetId",
    )
    if capture_target not in (
        "analytic-kerr-event-horizon",
        "analytic-kerr-stretched-horizon",
    ):
        _fail("$.termination.captureTargetId", "unsupported capture target")
    if not _boolean(
        visibility.get("captureStrictlyInsideDiskIsco"),
        "$.termination.visibilityConstraints.captureStrictlyInsideDiskIsco",
    ):
        _fail(
            "$.termination.visibilityConstraints.captureStrictlyInsideDiskIsco",
            "capture/ISCO visibility contract is unavailable",
        )
    if not _boolean(
        visibility.get("escapeStrictlyOutsideMaximumPhotosphereOblateRadius"),
        "$.termination.visibilityConstraints."
        "escapeStrictlyOutsideMaximumPhotosphereOblateRadius",
    ):
        _fail(
            "$.termination.visibilityConstraints."
            "escapeStrictlyOutsideMaximumPhotosphereOblateRadius",
            "escape/photosphere visibility contract is unavailable",
        )
    visibility_maximum_radius = _number(
        visibility.get("maximumPhotosphereOblateRadiusM"),
        "$.termination.visibilityConstraints.maximumPhotosphereOblateRadiusM",
    )

    if _string(surface.get("type"), "$.finiteThicknessSurface.type") != (
        "Zhou-prescribed-stationary-photosphere"
    ):
        _fail("$.finiteThicknessSurface.type", "unsupported finite surface")
    surface_ids = surface.get("surfaceIds")
    if (
        type(surface_ids) is not list
        or len(surface_ids) != len(_EXPECTED_SURFACE_IDS)
        or any(type(value) is not str for value in surface_ids)
        or tuple(surface_ids) != _EXPECTED_SURFACE_IDS
    ):
        _fail("$.finiteThicknessSurface.surfaceIds", "surface ids disagree")
    if not _boolean(
        surface.get("heightRateIsIndependentOfThermalRate"),
        "$.finiteThicknessSurface.heightRateIsIndependentOfThermalRate",
    ):
        _fail(
            "$.finiteThicknessSurface.heightRateIsIndependentOfThermalRate",
            "height and thermal rates must remain independently supplied",
        )
    spin_magnitude = _number(
        surface.get("dimensionlessSpinMagnitude"),
        "$.finiteThicknessSurface.dimensionlessSpinMagnitude",
    )
    dotm = _number(
        surface.get("eddingtonScaledMassAccretionRate"),
        "$.finiteThicknessSurface.eddingtonScaledMassAccretionRate",
    )
    outer = _number(
        surface.get("outerRadiusOverMass"),
        "$.finiteThicknessSurface.outerRadiusOverMass",
    )
    thinness_gate = _number(
        surface.get("thinnessGateMaximumHOverRho"),
        "$.finiteThicknessSurface.thinnessGateMaximumHOverRho",
    )
    orientation = _string(
        surface.get("orientation"),
        "$.finiteThicknessSurface.orientation",
    )
    if orientation not in ("prograde", "retrograde"):
        _fail("$.finiteThicknessSurface.orientation", "unsupported orientation")
    if not _close(spin_magnitude, abs(spin / mass)) or dotm <= 0.0:
        _fail("$.finiteThicknessSurface", "spin or positive height rate disagrees")

    thermal_orientation = _string(
        thermal.get("orientation"),
        "$.diskThermalProxy.orientation",
    )
    if thermal_orientation != orientation:
        _fail("$.diskThermalProxy.orientation", "thermal orientation disagrees")
    if _string(
        thermal.get("radialReference"),
        "$.diskThermalProxy.radialReference",
    ) != "equatorial-NT-at-matching-rho":
        _fail(
            "$.diskThermalProxy.radialReference",
            "unsupported thermal radial reference",
        )
    isco_m = _number(thermal.get("iscoRadiusM"), "$.diskThermalProxy.iscoRadiusM")
    independent_isco = _independent_isco_radius(spin_magnitude, orientation)
    if not _close(isco_m / mass, independent_isco, 4.0e-12):
        _fail("$.diskThermalProxy.iscoRadiusM", "ISCO disagrees with independent root")
    if not independent_isco < outer or capture >= isco_m:
        _fail("$.finiteThicknessSurface", "radial domain or capture worldtube is invalid")
    black_hole_mass = _number(
        thermal.get("blackHoleMassKg"),
        "$.diskThermalProxy.blackHoleMassKg",
    )
    thermal_rate = _number(
        thermal.get("massAccretionRateKgS"),
        "$.diskThermalProxy.massAccretionRateKgS",
    )
    colour = _number(
        thermal.get("colourCorrection"),
        "$.diskThermalProxy.colourCorrection",
    )
    if black_hole_mass <= 0.0 or thermal_rate < 0.0 or colour <= 0.0:
        _fail("$.diskThermalProxy", "thermal SI parameters are invalid")
    configuration = KerrFiniteThicknessSelectedConfiguration(
        mass,
        spin,
        observer_radius,
        observer_theta,
        observer_phi,
        capture,
        escape,
        isco_m,
        outer,
        orientation,
        dotm,
        thinness_gate,
        black_hole_mass,
        thermal_rate,
        colour,
        _canonical_sha256(raw),
    )
    maximum_rho = min(2.25 * independent_isco, outer)
    maximum_h_over_rho = (
        0.5 * configuration.photosphere_height_over_mass(maximum_rho) / maximum_rho
    )
    if maximum_h_over_rho > thinness_gate * (1.0 + 64.0 * math.ulp(1.0)):
        _fail("$.finiteThicknessSurface", "independent thinness gate fails")
    declared_maximum_radius = _number(
        surface.get("maximumPhotosphereOblateRadiusM"),
        "$.finiteThicknessSurface.maximumPhotosphereOblateRadiusM",
    )
    outer_height = configuration.photosphere_height_over_mass(outer)
    independent_maximum_radius = math.hypot(outer, outer_height) * mass
    if not _close(declared_maximum_radius, independent_maximum_radius, 4.0e-12):
        _fail(
            "$.finiteThicknessSurface.maximumPhotosphereOblateRadiusM",
            "maximum photosphere radius disagrees",
        )
    if not _close(
        visibility_maximum_radius,
        independent_maximum_radius,
        4.0e-12,
    ):
        _fail(
            "$.termination.visibilityConstraints.maximumPhotosphereOblateRadiusM",
            "visibility maximum photosphere radius disagrees",
        )
    if independent_maximum_radius >= escape:
        _fail("$.termination.escapeRadiusM", "escape does not enclose photosphere")

    if _string(
        material_clearance.get("policy"),
        "$.observer.materialClearance.policy",
    ) != "outside both faces whenever rho is in the physical annulus":
        _fail(
            "$.observer.materialClearance.policy",
            "unsupported observer material policy",
        )
    if _string(
        material_clearance.get("status"),
        "$.observer.materialClearance.status",
    ) != "outside-certified":
        _fail(
            "$.observer.materialClearance.status",
            "observer material clearance is not certified",
        )
    declared_observer_rho = _number(
        material_clearance.get("pseudoCylindricalRadiusOverMass"),
        "$.observer.materialClearance.pseudoCylindricalRadiusOverMass",
    )
    declared_upper_value = _number(
        material_clearance.get("upperFaceSignedValue"),
        "$.observer.materialClearance.upperFaceSignedValue",
    )
    declared_lower_value = _number(
        material_clearance.get("lowerFaceSignedValue"),
        "$.observer.materialClearance.lowerFaceSignedValue",
    )
    declared_within_annulus = _boolean(
        material_clearance.get("withinPhysicalAnnulus"),
        "$.observer.materialClearance.withinPhysicalAnnulus",
    )
    observer_radius_over_mass = observer_radius / mass
    independent_observer_rho = observer_radius_over_mass * sine
    observer_auxiliary_height = configuration.auxiliary_height_over_mass(
        independent_observer_rho
    )
    independent_upper_value = (
        observer_radius_over_mass * cosine - observer_auxiliary_height
    )
    independent_lower_value = (
        -observer_radius_over_mass * cosine - observer_auxiliary_height
    )
    independent_within_annulus = (
        independent_isco <= independent_observer_rho <= outer
    )
    if (
        not _close(declared_observer_rho, independent_observer_rho, 4.0e-12)
        or not _close(declared_upper_value, independent_upper_value, 4.0e-12)
        or not _close(declared_lower_value, independent_lower_value, 4.0e-12)
        or declared_within_annulus is not independent_within_annulus
    ):
        _fail(
            "$.observer.materialClearance",
            "clearance disagrees with the independently rebuilt finite faces",
        )
    if (
        independent_within_annulus
        and independent_upper_value <= 0.0
        and independent_lower_value <= 0.0
    ):
        _fail(
            "$.observer.materialClearance",
            "observer lies on or inside both physical photosphere faces",
        )
    return configuration


@dataclass(frozen=True, slots=True)
class FiniteThicknessSelectedRayResult:
    screen_x: float
    screen_y: float
    outcome: SelectedOutcome
    affine_length_m: float
    terminal_radius_m: float
    face: str | None
    pseudo_cylindrical_radius_over_mass: float | None
    frequency_shift_g: float | None
    signed_emission_angle_cosine: float | None
    constants: KerrPhotonConstants
    maximum_hamiltonian_residual: float
    maximum_relative_carter_drift: float
    transparent_surface_crossings: int
    steps: int


@dataclass(frozen=True, slots=True)
class FiniteThicknessSelectedRayRefinement:
    coarse: FiniteThicknessSelectedRayResult
    fine: FiniteThicknessSelectedRayResult
    outcome_agrees: bool
    face_agrees: bool
    affine_length_difference_m: float
    terminal_radius_difference_m: float
    pseudo_cylindrical_radius_difference_m: float | None
    relative_g_difference: float | None
    signed_mu_difference: float | None


@dataclass(frozen=True, slots=True)
class FiniteThicknessSelectedSpectrumRefinement:
    coarse_intensities_nu: tuple[float, ...]
    fine_intensities_nu: tuple[float, ...]
    absolute_differences_nu: tuple[float, ...]
    maximum_relative_difference: float


_State = tuple[float, float, float, float, float, float, float, float]


def _initial_state(
    configuration: KerrFiniteThicknessSelectedConfiguration,
    screen_x: float,
    screen_y: float,
) -> _State:
    """Independent BL ZAMO pinhole launch with production screen signs."""

    radius = configuration.observer_radius_m / configuration.mass_m
    theta = configuration.observer_theta_rad
    spin = configuration.dimensionless_spin
    sine = math.sin(theta)
    cosine = math.cos(theta)
    sigma = radius * radius + spin * spin * cosine * cosine
    delta = radius * radius - 2.0 * radius + spin * spin
    big_a = (radius * radius + spin * spin) ** 2 - spin * spin * delta * sine * sine
    lapse = math.sqrt(sigma * delta / big_a)
    omega = 2.0 * spin * radius / big_a
    inverse_norm = 1.0 / math.sqrt(1.0 + screen_x * screen_x + screen_y * screen_y)
    p_phi = -inverse_norm * screen_x * sine * math.sqrt(big_a / sigma)
    p_theta = -inverse_norm * screen_y * math.sqrt(sigma)
    p_r = -inverse_norm * math.sqrt(sigma / delta)
    p_t = lapse - omega * p_phi
    return (0.0, radius, theta, 0.0, p_t, p_r, p_theta, p_phi)


def _face_value(
    configuration: KerrFiniteThicknessSelectedConfiguration,
    state: _State,
    face: str,
) -> float:
    radius = state[1]
    theta = state[2]
    rho = radius * math.sin(theta)
    sign = 1.0 if face == UPPER else -1.0
    return (
        sign * radius * math.cos(theta)
        - configuration.auxiliary_height_over_mass(rho)
    )


def _observer_frequency(
    configuration: KerrFiniteThicknessSelectedConfiguration,
    state: _State,
) -> float:
    radius = state[1]
    theta = state[2]
    spin = configuration.dimensionless_spin
    sine = math.sin(theta)
    cosine = math.cos(theta)
    sigma = radius * radius + spin * spin * cosine * cosine
    delta = radius * radius - 2.0 * radius + spin * spin
    big_a = (radius * radius + spin * spin) ** 2 - spin * spin * delta * sine * sine
    lapse = math.sqrt(sigma * delta / big_a)
    omega = 2.0 * spin * radius / big_a
    frequency = (state[4] + omega * state[7]) / lapse
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise KerrFiniteThicknessSelectedOracleError(
            "observer-frame past-directed frequency is invalid"
        )
    return frequency


def _face_transfer(
    configuration: KerrFiniteThicknessSelectedConfiguration,
    observer_state: _State,
    emitter_state: _State,
    face: str,
    rho: float,
) -> tuple[float, float]:
    radius = emitter_state[1]
    theta = emitter_state[2]
    spin = configuration.dimensionless_spin
    sine = math.sin(theta)
    cosine = math.cos(theta)
    sigma = radius * radius + spin * spin * cosine * cosine
    delta = radius * radius - 2.0 * radius + spin * spin
    big_a = (radius * radius + spin * spin) ** 2 - spin * spin * delta * sine * sine
    g_tt = -(1.0 - 2.0 * radius / sigma)
    g_t_phi = -2.0 * spin * radius * sine * sine / sigma
    g_phi_phi = big_a * sine * sine / sigma
    omega, _energy, _lz = _equatorial_orbit(
        rho,
        spin,
        configuration.orientation,
    )
    helical_norm = g_tt + 2.0 * omega * g_t_phi + omega * omega * g_phi_phi
    if not math.isfinite(helical_norm) or helical_norm >= 0.0:
        raise KerrFiniteThicknessSelectedOracleError(
            "actual-face circular-velocity reference is not timelike"
        )
    u_time = 1.0 / math.sqrt(-helical_norm)
    emitter_frequency = u_time * (emitter_state[4] + omega * emitter_state[7])
    observer_frequency = _observer_frequency(configuration, observer_state)
    if not math.isfinite(emitter_frequency) or emitter_frequency <= 0.0:
        raise KerrFiniteThicknessSelectedOracleError(
            "emitter-frame past-directed frequency is invalid"
        )
    shift = observer_frequency / emitter_frequency

    sign = 1.0 if face == UPPER else -1.0
    slope = configuration.photosphere_height_derivative(rho)
    radial_gradient = sign * cosine - slope * sine
    polar_gradient = -sign * radius * sine - slope * radius * cosine
    norm_squared = (
        delta * radial_gradient * radial_gradient
        + polar_gradient * polar_gradient
    ) / sigma
    if not math.isfinite(norm_squared) or norm_squared <= 0.0:
        raise KerrFiniteThicknessSelectedOracleError("face normal is invalid")
    inverse_norm = 1.0 / math.sqrt(norm_squared)
    normal_r = delta / sigma * radial_gradient * inverse_norm
    normal_theta = polar_gradient * inverse_norm / sigma
    normal_projection = (
        normal_r * emitter_state[5] + normal_theta * emitter_state[6]
    )
    signed_mu = -normal_projection / emitter_frequency
    if not math.isfinite(shift) or shift <= 0.0:
        raise KerrFiniteThicknessSelectedOracleError("frequency shift is invalid")
    if not math.isfinite(signed_mu) or signed_mu < -1.0 - 2e-9 or signed_mu > 1.0 + 2e-9:
        raise KerrFiniteThicknessSelectedOracleError(
            "signed face cosine lies outside the local null cone"
        )
    signed_mu = min(1.0, max(-1.0, signed_mu))
    if signed_mu <= 0.0:
        raise KerrFiniteThicknessSelectedOracleError(
            f"selected ray reaches the backside of the {face} face"
        )
    return shift, signed_mu


def trace_selected_ray(
    configuration: KerrFiniteThicknessSelectedConfiguration,
    screen_x: float,
    screen_y: float,
    options: FixedRk4Options = FixedRk4Options(),
) -> FiniteThicknessSelectedRayResult:
    """Trace one independent first-visible upper/lower finite-surface ray."""

    if type(configuration) is not KerrFiniteThicknessSelectedConfiguration:
        raise TypeError("configuration must be exact selected configuration")
    if type(options) is not FixedRk4Options:
        raise TypeError("options must be exact FixedRk4Options")
    for value, name in ((screen_x, "screen_x"), (screen_y, "screen_y")):
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite built-in number")
    x = float(screen_x)
    y = float(screen_y)
    mass = configuration.mass_m
    spin = configuration.dimensionless_spin
    step = options.step_m / mass
    maximum_affine = options.maximum_affine_length_m / mass
    capture = configuration.capture_radius_m / mass
    escape = configuration.escape_radius_m / mass
    state = _initial_state(configuration, x, y)
    observer_state = state
    initial_constants = photon_constants(state, spin)
    maximum_hamiltonian = _shared_bl_hamiltonian_residual(state, spin)
    maximum_carter_drift = 0.0
    transparent_surface_crossings = 0
    affine = 0.0

    def finish(
        outcome: SelectedOutcome,
        terminal: _State,
        steps: int,
        face: str | None = None,
    ) -> FiniteThicknessSelectedRayResult:
        rho = None
        shift = None
        signed_mu = None
        if face is not None:
            rho = terminal[1] * math.sin(terminal[2])
            shift, signed_mu = _face_transfer(
                configuration,
                observer_state,
                terminal,
                face,
                rho,
            )
        return FiniteThicknessSelectedRayResult(
            x,
            y,
            outcome,
            affine * mass,
            terminal[1] * mass,
            face,
            rho,
            shift,
            signed_mu,
            photon_constants(terminal, spin),
            maximum_hamiltonian,
            maximum_carter_drift,
            transparent_surface_crossings,
            steps,
        )

    maximum_steps = min(options.maximum_steps, math.ceil(maximum_affine / step) + 1)
    for step_index in range(1, maximum_steps + 1):
        remaining = maximum_affine - affine
        if remaining <= 0.0:
            return finish("unresolved", state, step_index - 1)
        actual_step = min(step, remaining)
        candidate = _shared_bl_rk4_step(state, actual_step, spin)
        if not all(math.isfinite(value) for value in candidate):
            raise KerrFiniteThicknessSelectedOracleError(
                "fixed RK4 state became non-finite"
            )
        if abs(math.sin(candidate[2])) <= options.pole_guard_sine:
            raise KerrFiniteThicknessSelectedOracleError(
                "fixed RK4 reached the guarded BL polar axis"
            )

        # Transparent auxiliary crossings participate in the same affine
        # ordering as opaque/capture/escape events.  Counting them immediately
        # while probing a whole RK4 step would incorrectly count a transparent
        # crossing that lies *after* an earlier terminal event in that step.
        events: list[
            tuple[float, SelectedOutcome | None, _State, str | None]
        ] = []
        if state[1] > capture and candidate[1] <= capture:
            fraction, located = _shared_partial_rk4_bisection(
                state,
                actual_step,
                spin,
                lambda entry: entry[1] - capture,
                options.event_bisection_iterations,
            )
            events.append((fraction, "captured", located, None))
        if (
            state[1] < escape
            and candidate[1] >= escape
            and candidate[5] > 0.0
        ):
            fraction, located = _shared_partial_rk4_bisection(
                state,
                actual_step,
                spin,
                lambda entry: entry[1] - escape,
                options.event_bisection_iterations,
            )
            events.append((fraction, "escaped", located, None))

        for face in _FACES:
            start_value = _face_value(configuration, state, face)
            end_value = _face_value(configuration, candidate, face)
            if start_value != 0.0 and (
                end_value == 0.0 or (start_value > 0.0) != (end_value > 0.0)
            ):
                fraction, located = _shared_partial_rk4_bisection(
                    state,
                    actual_step,
                    spin,
                    lambda entry, selected_face=face: _face_value(
                        configuration,
                        entry,
                        selected_face,
                    ),
                    options.event_bisection_iterations,
                )
                rho = located[1] * math.sin(located[2])
                if (
                    configuration.isco_radius_over_mass
                    <= rho
                    <= configuration.outer_radius_over_mass
                ):
                    events.append((fraction, face, located, face))
                else:
                    events.append((fraction, None, located, face))

        if events:
            events.sort(key=lambda entry: entry[0])
            terminal_index = next(
                (
                    index
                    for index, entry in enumerate(events)
                    if entry[1] is not None
                ),
                None,
            )
            if terminal_index is None:
                transparent_surface_crossings += len(events)
            else:
                transparent_surface_crossings += sum(
                    entry[1] is None for entry in events[:terminal_index]
                )
                fraction, outcome, terminal, face = events[terminal_index]
                if (
                    terminal_index + 1 < len(events)
                    and face is not None
                    and events[terminal_index + 1][1] is not None
                    and events[terminal_index + 1][3] is not None
                    and abs(fraction - events[terminal_index + 1][0])
                    <= 4.0
                    * math.ulp(
                        max(1.0, fraction, events[terminal_index + 1][0])
                    )
                ):
                    raise KerrFiniteThicknessSelectedOracleError(
                        "upper/lower first-visible ordering is ambiguous at the "
                        "ISCO seam"
                    )
                affine += fraction * actual_step
                maximum_hamiltonian = max(
                    maximum_hamiltonian,
                    _shared_bl_hamiltonian_residual(terminal, spin),
                )
                terminal_constants = photon_constants(terminal, spin)
                maximum_carter_drift = max(
                    maximum_carter_drift,
                    abs(terminal_constants.carter_q - initial_constants.carter_q)
                    / max(
                        1.0,
                        abs(initial_constants.carter_q),
                        abs(initial_constants.carter_k),
                    ),
                )
                if outcome is None:
                    raise AssertionError("terminal event lost its outcome")
                return finish(outcome, terminal, step_index, face)

        state = candidate
        affine += actual_step
        maximum_hamiltonian = max(
            maximum_hamiltonian,
            _shared_bl_hamiltonian_residual(state, spin),
        )
        current_constants = photon_constants(state, spin)
        maximum_carter_drift = max(
            maximum_carter_drift,
            abs(current_constants.carter_q - initial_constants.carter_q)
            / max(
                1.0,
                abs(initial_constants.carter_q),
                abs(initial_constants.carter_k),
            ),
        )
    return finish("unresolved", state, maximum_steps)


def trace_selected_ray_refined(
    configuration: KerrFiniteThicknessSelectedConfiguration,
    screen_x: float,
    screen_y: float,
    options: FixedRk4Options = FixedRk4Options(),
) -> FiniteThicknessSelectedRayRefinement:
    """Trace independent ``h`` and ``h/2`` rays and publish their differences."""

    coarse = trace_selected_ray(configuration, screen_x, screen_y, options)
    fine = trace_selected_ray(
        configuration,
        screen_x,
        screen_y,
        FixedRk4Options(
            step_m=0.5 * options.step_m,
            maximum_affine_length_m=options.maximum_affine_length_m,
            maximum_steps=2 * options.maximum_steps,
            event_bisection_iterations=options.event_bisection_iterations,
            pole_guard_sine=options.pole_guard_sine,
        ),
    )
    rho_difference = None
    relative_g_difference = None
    mu_difference = None
    if (
        coarse.pseudo_cylindrical_radius_over_mass is not None
        and fine.pseudo_cylindrical_radius_over_mass is not None
    ):
        rho_difference = (
            abs(
                coarse.pseudo_cylindrical_radius_over_mass
                - fine.pseudo_cylindrical_radius_over_mass
            )
            * configuration.mass_m
        )
    if coarse.frequency_shift_g is not None and fine.frequency_shift_g is not None:
        relative_g_difference = abs(
            coarse.frequency_shift_g - fine.frequency_shift_g
        ) / max(abs(coarse.frequency_shift_g), abs(fine.frequency_shift_g))
    if (
        coarse.signed_emission_angle_cosine is not None
        and fine.signed_emission_angle_cosine is not None
    ):
        mu_difference = abs(
            coarse.signed_emission_angle_cosine
            - fine.signed_emission_angle_cosine
        )
    return FiniteThicknessSelectedRayRefinement(
        coarse,
        fine,
        coarse.outcome == fine.outcome,
        coarse.face == fine.face,
        abs(coarse.affine_length_m - fine.affine_length_m),
        abs(coarse.terminal_radius_m - fine.terminal_radius_m),
        rho_difference,
        relative_g_difference,
        mu_difference,
    )


def selected_ray_observed_intensities_nu(
    configuration: KerrFiniteThicknessSelectedConfiguration,
    result: FiniteThicknessSelectedRayResult,
    observer_frequencies_hz: Sequence[float],
) -> tuple[float, ...]:
    """Independent Planck+D20+g^3 transfer with shared Page--Thorne radius."""

    if type(configuration) is not KerrFiniteThicknessSelectedConfiguration:
        raise TypeError("configuration must be exact selected configuration")
    if type(result) is not FiniteThicknessSelectedRayResult:
        raise TypeError("result must be exact selected-ray result")
    if type(observer_frequencies_hz) not in (tuple, list):
        raise TypeError("observer frequencies must be an exact tuple or list")
    if not observer_frequencies_hz:
        raise ValueError("observer frequencies must be non-empty")
    frequencies: list[float] = []
    for index, value in enumerate(observer_frequencies_hz):
        if (
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or value <= 0.0
        ):
            raise ValueError(
                f"observer frequency {index} must be a finite positive "
                "built-in number"
            )
        frequencies.append(float(value))
    if result.outcome in ("captured", "unresolved"):
        return tuple(0.0 for _frequency in frequencies)
    if result.outcome == "escaped":
        raise KerrFiniteThicknessSelectedOracleError(
            "escape I_nu belongs to the separately configured observer spectrum"
        )
    if (
        result.pseudo_cylindrical_radius_over_mass is None
        or result.frequency_shift_g is None
        or result.signed_emission_angle_cosine is None
    ):
        raise KerrFiniteThicknessSelectedOracleError(
            "finite-face ray lacks transfer diagnostics"
        )
    rho = result.pseudo_cylindrical_radius_over_mass
    flux_shape = page_thorne_flux_shape(
        rho,
        abs(configuration.dimensionless_spin),
        configuration.orientation,
    )
    if flux_shape == 0.0 or configuration.thermal_mass_accretion_rate_kg_s == 0.0:
        return tuple(0.0 for _frequency in frequencies)
    maximum_log = math.log(sys.float_info.max)
    minimum_log = math.log(math.ulp(0.0))
    log_surface_flux = (
        6.0 * math.log(LIGHT_SPEED_M_S)
        + math.log(configuration.thermal_mass_accretion_rate_kg_s)
        + math.log(flux_shape)
        - math.log(4.0 * math.pi)
        - 2.0 * math.log(GRAVITATIONAL_CONSTANT_M3_KG_S2)
        - 2.0 * math.log(configuration.black_hole_mass_kg)
    )
    if log_surface_flux > maximum_log or log_surface_flux < minimum_log:
        raise KerrFiniteThicknessSelectedOracleError(
            "shared Page-Thorne surface flux lies outside binary64"
        )
    log_temperature = math.log(configuration.colour_correction) + 0.25 * (
        log_surface_flux - math.log(STEFAN_BOLTZMANN_W_M2_K4)
    )
    mu = result.signed_emission_angle_cosine
    angular = 0.5 + 0.75 * mu
    shift = result.frequency_shift_g
    observed: list[float] = []
    for observer_frequency in frequencies:
        emitted_frequency = observer_frequency / shift
        log_exponent = (
            math.log(PLANCK_CONSTANT_J_S)
            + math.log(emitted_frequency)
            - math.log(BOLTZMANN_CONSTANT_J_K)
            - log_temperature
        )
        if log_exponent > maximum_log:
            observed.append(0.0)
            continue
        if log_exponent < minimum_log:
            log_denominator = log_exponent
        else:
            exponent = math.exp(log_exponent)
            log_denominator = exponent if exponent > 50.0 else math.log(math.expm1(exponent))
        log_value = (
            3.0 * math.log(shift)
            + math.log(angular)
            + math.log(2.0 * PLANCK_CONSTANT_J_S)
            - 2.0 * math.log(LIGHT_SPEED_M_S)
            + 3.0 * math.log(emitted_frequency)
            - log_denominator
            - 4.0 * math.log(configuration.colour_correction)
        )
        if log_value < minimum_log:
            value = 0.0
        elif log_value > maximum_log:
            raise KerrFiniteThicknessSelectedOracleError(
                "selected finite-face I_nu overflowed binary64"
            )
        else:
            value = math.exp(log_value)
        if not math.isfinite(value) or value < 0.0:
            raise KerrFiniteThicknessSelectedOracleError(
                "selected finite-face I_nu is invalid"
            )
        observed.append(value)
    return tuple(observed)


def selected_ray_refined_observed_intensities_nu(
    configuration: KerrFiniteThicknessSelectedConfiguration,
    refinement: FiniteThicknessSelectedRayRefinement,
    observer_frequencies_hz: Sequence[float],
) -> FiniteThicknessSelectedSpectrumRefinement:
    coarse = selected_ray_observed_intensities_nu(
        configuration,
        refinement.coarse,
        observer_frequencies_hz,
    )
    fine = selected_ray_observed_intensities_nu(
        configuration,
        refinement.fine,
        observer_frequencies_hz,
    )
    differences = tuple(abs(left - right) for left, right in zip(coarse, fine))
    relative = max(
        (
            difference / max(abs(left), abs(right), 1.0e-300)
            for left, right, difference in zip(coarse, fine, differences)
        ),
        default=0.0,
    )
    return FiniteThicknessSelectedSpectrumRefinement(
        coarse,
        fine,
        differences,
        relative,
    )


__all__ = (
    "FiniteThicknessSelectedRayRefinement",
    "FiniteThicknessSelectedRayResult",
    "FiniteThicknessSelectedSpectrumRefinement",
    "KerrFiniteThicknessSelectedConfiguration",
    "KerrFiniteThicknessSelectedOracleError",
    "SCIENTIFIC_STATUS",
    "SELECTED_ORACLE_IMPLEMENTATION_ID",
    "SUPPORTED_SAMPLER_IMPLEMENTATION_ID",
    "configuration_from_sampler_descriptor",
    "selected_ray_observed_intensities_nu",
    "selected_ray_refined_observed_intensities_nu",
    "trace_selected_ray",
    "trace_selected_ray_refined",
)
