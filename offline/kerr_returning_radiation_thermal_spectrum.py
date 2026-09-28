"""Certified local thermal spectrum for a returning-radiation profile.

This module is deliberately an emission-only adapter.  It turns the
piecewise-constant effective temperatures of a replay-certified
``AxisymmetricReturningRadiationThermalProfile`` into the same isotropic,
colour-corrected diluted blackbody used by
``StationaryNovikovThorneDisk``::

    I_nu = B_nu(f_col T_eff) / f_col**4.

The returned value is local-comoving, one-face specific intensity in
``W m^-2 sr^-1 Hz^-1`` at an *emitter-frame* frequency.  No ray-transfer
factor (including ``g**3`` or ``D20``), spectral redistribution, atmosphere,
scattering, polarization, returning-radiation stress/work term ``F_S``, or
GRMHD physics is applied here.  Annulus lookup has exactly the thermal
profile's piecewise-constant, left-closed/right-open convention, with its
final outer edge included.

Both providers and samples are sealed by canonical descriptors and expose
explicit same-code replay validation.  This is not an independent Planck or
Novikov--Thorne oracle: it intentionally shares constants and the binary64
branch structure of :mod:`offline.kerr_disk`.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping

from offline.kerr_disk import (
    BOLTZMANN_CONSTANT_J_K,
    COLOUR_CORRECTED_PLANCK_IMPLEMENTATION_ID,
    KerrDiskError,
    LIGHT_SPEED_M_S,
    PLANCK_CONSTANT_J_S,
    StationaryNovikovThorneDisk,
    _validated_colour_corrected_planck_specific_intensity_nu,
)
from offline.kerr_returning_radiation_thermal_profile import (
    CERTIFIED_KERR_FORWARD_CLASSIFICATIONS,
    REPLAYED_KERR_FORWARD_CLASSIFICATION,
    AxisymmetricReturningRadiationThermalProfile,
    verify_axisymmetric_returning_radiation_thermal_profile,
)


_shared_validated_planck_intensity = (
    _validated_colour_corrected_planck_specific_intensity_nu
)


IMPLEMENTATION_ID: Final = (
    "kerr-returning-radiation-certified-piecewise-thermal-spectrum/v1"
)
PLANCK_IMPLEMENTATION_ID: Final = COLOUR_CORRECTED_PLANCK_IMPLEMENTATION_ID
SPECIFIC_INTENSITY_UNITS: Final = "W m^-2 sr^-1 Hz^-1"
FREQUENCY_FRAME: Final = "local-comoving emitter frame"
QUERY_BOUNDARY_CONVENTION: Final = (
    "left-closed right-open, with final outer edge included"
)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "same-code replay-certified piecewise local thermal spectrum"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "requiredProfileClassification": REPLAYED_KERR_FORWARD_CLASSIFICATION,
        "acceptedProfileClassifications": CERTIFIED_KERR_FORWARD_CLASSIFICATIONS,
        "profileGeometryMustBeCertified": True,
        "emissionLaw": "I_nu=B_nu(f_col T_eff)/f_col^4",
        "specificIntensityUnits": SPECIFIC_INTENSITY_UNITS,
        "frequencyFrame": FREQUENCY_FRAME,
        "isotropicOneFaceLocalEmission": True,
        "piecewiseConstantAnnulusTemperature": True,
        "renderLoopRequiresPriorProviderRevalidation": True,
        "batchBuildsPerFrequencyEvidenceSamples": False,
        "sharesPlanckAndNovikovThorneConstantsWithKerrDisk": True,
        "hasIndependentPlanckOracle": False,
        "appliesD20AngularLaw": False,
        "appliesTransferG3": False,
        "includesReturningRadiationStressWorkFS": False,
        "includesSpectralRedistribution": False,
        "includesScattering": False,
        "includesSolvedAtmosphere": False,
        "includesPolarization": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "prohibitedClaim": (
            "Do not describe this shared-code, piecewise diluted-blackbody "
            "adapter as an independent oracle, D20/g^3 transfer, F_S, a "
            "solved atmosphere, scattering, polarization, or GRMHD."
        ),
    }
)


class KerrReturningRadiationThermalSpectrumError(RuntimeError):
    """Base class for fail-closed thermal-spectrum failures."""


class KerrReturningRadiationThermalSpectrumVerificationError(
    KerrReturningRadiationThermalSpectrumError
):
    """Raised when a provider or sample differs from same-code replay."""


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
        raise KerrReturningRadiationThermalSpectrumError(
            "thermal-spectrum descriptor is not finite canonical JSON"
        ) from error


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


def _require_exact_tree(actual: Any, expected: Any, path: str) -> None:
    """Compare a trusted primitive tree without subclass dispatch."""

    if type(actual) is not type(expected):
        raise KerrReturningRadiationThermalSpectrumVerificationError(
            f"{path} has non-exact type {type(actual).__name__}; "
            f"expected {type(expected).__name__}"
        )
    if type(expected) is tuple:
        if len(actual) != len(expected):
            raise KerrReturningRadiationThermalSpectrumVerificationError(
                f"{path} tuple length differs from deterministic replay"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _require_exact_tree(actual_item, expected_item, f"{path}[{index}]")
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
        raise KerrReturningRadiationThermalSpectrumVerificationError(
            f"{path} uses unsupported trusted type {type(expected).__name__}"
        )
    if differs:
        raise KerrReturningRadiationThermalSpectrumVerificationError(
            f"{path} differs from deterministic replay"
        )


def colour_corrected_planck_specific_intensity_nu(
    effective_temperature_k: float,
    colour_correction: float,
    emitted_frequency_hz: float,
) -> float:
    """Evaluate ``B_nu(f_col T_eff)/f_col**4`` in stable binary64.

    Inputs are exact finite floats.  Temperature may be zero, in which case
    the result is strictly positive-sign ``0.0``.  Frequency must be positive
    and ``f_col`` must be at least one.  Deep Rayleigh--Jeans and Wien limits
    are evaluated in log space, matching ``StationaryNovikovThorneDisk``.
    """

    effective_temperature = _exact_nonnegative_float(
        effective_temperature_k,
        "effective_temperature_k",
    )
    correction = _exact_positive_float(colour_correction, "colour_correction")
    if correction < 1.0:
        raise ValueError("colour_correction must be at least one")
    frequency = _exact_positive_float(
        emitted_frequency_hz,
        "emitted_frequency_hz",
    )
    if effective_temperature == 0.0:
        return 0.0

    return _evaluate_validated_planck_intensity(
        effective_temperature,
        correction,
        frequency,
    )


def _evaluate_validated_planck_intensity(
    effective_temperature_k: float,
    colour_correction: float,
    emitted_frequency_hz: float,
) -> float:
    """Map the shared validated kernel's failures into this module's API."""

    try:
        return _shared_validated_planck_intensity(
            effective_temperature_k,
            colour_correction,
            emitted_frequency_hz,
        )
    except KerrDiskError as error:
        raise KerrReturningRadiationThermalSpectrumError(str(error)) from error


def _planck_descriptor() -> Mapping[str, Any]:
    return {
        "binary64Branches": {
            "deepRayleighJeans": "log(x) retained below minimum subnormal",
            "moderate": "log(expm1(x)) for x<=50",
            "rangeBoundary": (
                "rare 120-digit Decimal replay from exact binary64 inputs"
            ),
            "underflowRounding": "half-minimum-subnormal threshold",
            "wien": "log(exp(x)-1)=x for x>50",
            "zeroTemperature": "strict positive-sign 0.0",
        },
        "constants": {
            "boltzmannConstantJK": BOLTZMANN_CONSTANT_J_K,
            "boltzmannConstantJKHex": BOLTZMANN_CONSTANT_J_K.hex(),
            "lightSpeedMS": LIGHT_SPEED_M_S,
            "lightSpeedMSHex": LIGHT_SPEED_M_S.hex(),
            "planckConstantJS": PLANCK_CONSTANT_J_S,
            "planckConstantJSHex": PLANCK_CONSTANT_J_S.hex(),
        },
        "formula": "I_nu=B_nu(f_col T_eff)/f_col^4",
        "implementationId": PLANCK_IMPLEMENTATION_ID,
        "independentOracle": False,
        "sharedCodeOwner": (
            "offline.kerr_disk.colour_corrected_planck_specific_intensity_nu"
        ),
    }


@dataclass(frozen=True, slots=True, init=False)
class CertifiedReturningRadiationThermalSpectrumSample:
    """One sealed piecewise-annulus local emission query result."""

    annulus_index: int
    radius_over_mass: float
    emitted_frequency_hz: float
    effective_temperature_k: float
    colour_temperature_k: float
    colour_correction: float
    isotropic_specific_intensity_nu: float
    specific_intensity_units: str
    frequency_frame: str
    provider_descriptor_sha256: str
    source_profile_descriptor_sha256: str
    _provider: "CertifiedReturningRadiationThermalSpectrumProvider" = field(
        repr=False,
        compare=False,
    )
    _descriptor_json: str = field(repr=False, compare=False)
    _descriptor_sha256: str = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "CertifiedReturningRadiationThermalSpectrumSample is built only "
            "by a certified spectrum provider"
        )

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        descriptor_json = object.__getattribute__(self, "_descriptor_json")
        if type(descriptor_json) is not str:
            raise KerrReturningRadiationThermalSpectrumVerificationError(
                "sample descriptor has a non-exact type"
            )
        return json.loads(descriptor_json)

    def revalidate(self) -> None:
        verify_certified_returning_radiation_thermal_spectrum_sample(self)


@dataclass(frozen=True, slots=True, init=False)
class CertifiedReturningRadiationThermalSpectrumProvider:
    """Replay-certified piecewise diluted-blackbody emission provider."""

    annulus_edges_over_mass: tuple[float, ...]
    outgoing_effective_temperature_k: tuple[float, ...]
    colour_correction: float
    source_profile_descriptor_sha256: str
    novikov_thorne_disk_descriptor_sha256: str
    specific_intensity_units: str
    frequency_frame: str
    query_boundary_convention: str
    _profile: AxisymmetricReturningRadiationThermalProfile = field(
        repr=False,
        compare=False,
    )
    _descriptor_json: str = field(repr=False, compare=False)
    _descriptor_sha256: str = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "CertifiedReturningRadiationThermalSpectrumProvider is built only "
            "by the replay-certified entry point"
        )

    @property
    def annulus_count(self) -> int:
        return len(self.outgoing_effective_temperature_k)

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        descriptor_json = object.__getattribute__(self, "_descriptor_json")
        if type(descriptor_json) is not str:
            raise KerrReturningRadiationThermalSpectrumVerificationError(
                "provider descriptor has a non-exact type"
            )
        return json.loads(descriptor_json)

    def _resolve_annulus_temperature(
        self,
        radius_over_mass: float,
    ) -> tuple[float, int, float, float, tuple[float, ...]]:
        """Resolve and validate radius-dependent state exactly once."""

        radius = _exact_float(radius_over_mass, "radius_over_mass")
        edges = object.__getattribute__(self, "annulus_edges_over_mass")
        temperatures = object.__getattribute__(
            self,
            "outgoing_effective_temperature_k",
        )
        if type(edges) is not tuple or type(temperatures) is not tuple:
            raise KerrReturningRadiationThermalSpectrumError(
                "provider annulus arrays need explicit revalidation"
            )
        if radius < edges[0]:
            raise ValueError("radius lies inside the ISCO/profile domain")
        if radius > edges[-1]:
            raise ValueError("radius lies outside the finite profile domain")
        annulus_index = bisect_right(edges, radius) - 1
        if annulus_index == len(temperatures):
            annulus_index -= 1
        effective_temperature = _exact_nonnegative_float(
            temperatures[annulus_index],
            "provider annulus effective temperature",
        )
        correction = _exact_positive_float(
            object.__getattribute__(self, "colour_correction"),
            "provider colour correction",
        )
        if correction < 1.0:
            raise KerrReturningRadiationThermalSpectrumError(
                "provider colour correction is below one"
            )
        return (
            radius,
            annulus_index,
            effective_temperature,
            correction,
            edges,
        )

    def sample(
        self,
        radius_over_mass: float,
        emitted_frequency_hz: float,
    ) -> CertifiedReturningRadiationThermalSpectrumSample:
        """Return a local one-face sample using the profile's annulus lookup."""

        frequency = _exact_positive_float(
            emitted_frequency_hz,
            "emitted_frequency_hz",
        )
        (
            radius,
            annulus_index,
            effective_temperature,
            correction,
            edges,
        ) = self._resolve_annulus_temperature(radius_over_mass)
        intensity = _evaluate_validated_planck_intensity(
            effective_temperature,
            correction,
            frequency,
        )
        colour_temperature = correction * effective_temperature
        descriptor = {
            "annulus": {
                "index": annulus_index,
                "innerRadiusOverMass": edges[annulus_index],
                "outerRadiusOverMass": edges[annulus_index + 1],
                "queryBoundaryConvention": QUERY_BOUNDARY_CONVENTION,
            },
            "emission": {
                "colourCorrection": correction,
                "colourTemperatureK": colour_temperature,
                "effectiveTemperatureK": effective_temperature,
                "emittedFrequencyHz": frequency,
                "frequencyFrame": FREQUENCY_FRAME,
                "isotropicSpecificIntensityNu": intensity,
                "specificIntensityUnits": SPECIFIC_INTENSITY_UNITS,
            },
            "implementationId": f"{IMPLEMENTATION_ID}/sample/v1",
            "providerDescriptorSha256": self.model_descriptor_sha256,
            "radiusOverMass": radius,
            "sourceProfileDescriptorSha256": (
                self.source_profile_descriptor_sha256
            ),
        }
        descriptor_json = _canonical_json(descriptor)
        result = object.__new__(CertifiedReturningRadiationThermalSpectrumSample)
        for name, value in (
            ("annulus_index", annulus_index),
            ("radius_over_mass", radius),
            ("emitted_frequency_hz", frequency),
            ("effective_temperature_k", effective_temperature),
            ("colour_temperature_k", colour_temperature),
            ("colour_correction", correction),
            ("isotropic_specific_intensity_nu", intensity),
            ("specific_intensity_units", SPECIFIC_INTENSITY_UNITS),
            ("frequency_frame", FREQUENCY_FRAME),
            ("provider_descriptor_sha256", self.model_descriptor_sha256),
            (
                "source_profile_descriptor_sha256",
                self.source_profile_descriptor_sha256,
            ),
            ("_provider", self),
            ("_descriptor_json", descriptor_json),
            (
                "_descriptor_sha256",
                hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
            ),
        ):
            object.__setattr__(result, name, value)
        return result

    def emitted_specific_intensity_nu(
        self,
        radius_over_mass: float,
        emitted_frequency_hz: float,
    ) -> float:
        """Return local ``I_nu`` without constructing an evidence sample.

        Call :meth:`revalidate` once before entering a trusted render loop.
        This hot path validates its scalar query but deliberately does not
        replay the expensive certified source profile on every invocation.
        """

        frequency = _exact_positive_float(
            emitted_frequency_hz,
            "emitted_frequency_hz",
        )
        (
            _radius,
            _annulus_index,
            effective_temperature,
            correction,
            _edges,
        ) = self._resolve_annulus_temperature(radius_over_mass)
        return _evaluate_validated_planck_intensity(
            effective_temperature,
            correction,
            frequency,
        )

    def emitted_specific_intensity_nu_batch(
        self,
        radius_over_mass: float,
        emitted_frequencies_hz: tuple[float, ...],
    ) -> tuple[float, ...]:
        """Return an exact tuple of intensities after one annulus lookup.

        The caller must invoke :meth:`revalidate` before entering a render
        loop and must not mutate the provider through low-level Python escape
        hatches.  This method resolves the radius/annulus once, validates the
        complete exact-tuple frequency grid once, and then calls the shared
        Planck kernel directly.  It intentionally constructs no per-bin
        sample descriptors or hashes; use :meth:`sample` for evidence records.
        """

        frequencies_raw = _exact_tuple(
            emitted_frequencies_hz,
            "emitted_frequencies_hz",
        )
        if not frequencies_raw:
            raise ValueError("emitted_frequencies_hz must not be empty")
        frequencies = tuple(
            _exact_positive_float(frequency, f"emitted frequency {index}")
            for index, frequency in enumerate(frequencies_raw)
        )
        (
            _radius,
            _annulus_index,
            effective_temperature,
            correction,
            _edges,
        ) = self._resolve_annulus_temperature(radius_over_mass)
        return tuple(
            _evaluate_validated_planck_intensity(
                effective_temperature,
                correction,
                frequency,
            )
            for frequency in frequencies
        )

    def revalidate(self) -> None:
        verify_certified_returning_radiation_thermal_spectrum_provider(self)


def _validated_certified_profile(
    value: Any,
) -> AxisymmetricReturningRadiationThermalProfile:
    if type(value) is not AxisymmetricReturningRadiationThermalProfile:
        raise TypeError(
            "profile must be an exact AxisymmetricReturningRadiationThermalProfile"
        )
    try:
        verify_axisymmetric_returning_radiation_thermal_profile(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise KerrReturningRadiationThermalSpectrumVerificationError(
            f"source thermal profile could not be replay-validated: {error}"
        ) from error
    certified = object.__getattribute__(value, "geometry_provenance_certified")
    classification = object.__getattribute__(value, "provenance_classification")
    if type(certified) is not bool or certified is not True:
        raise ValueError(
            "thermal spectrum requires geometry_provenance_certified=True"
        )
    if type(classification) is not str or classification not in (
        CERTIFIED_KERR_FORWARD_CLASSIFICATIONS
    ):
        raise ValueError(
            "thermal spectrum requires direct-replay or production-cached "
            "certified profile provenance"
        )
    disk = object.__getattribute__(value, "_disk")
    if type(disk) is not StationaryNovikovThorneDisk:
        raise KerrReturningRadiationThermalSpectrumVerificationError(
            "source profile disk has a non-exact type"
        )
    return value


def _build_provider_from_validated_profile(
    profile: AxisymmetricReturningRadiationThermalProfile,
) -> CertifiedReturningRadiationThermalSpectrumProvider:
    edges = object.__getattribute__(profile, "annulus_edges_over_mass")
    temperatures = object.__getattribute__(
        profile,
        "outgoing_effective_temperature_k",
    )
    disk = object.__getattribute__(profile, "_disk")
    correction = object.__getattribute__(disk, "colour_correction")
    source_sha = object.__getattribute__(profile, "_descriptor_sha256")
    disk_sha = object.__getattribute__(
        profile,
        "novikov_thorne_disk_descriptor_sha256",
    )
    classification = object.__getattribute__(
        profile,
        "provenance_classification",
    )
    if type(edges) is not tuple or type(temperatures) is not tuple:
        raise KerrReturningRadiationThermalSpectrumVerificationError(
            "source profile annulus arrays have non-exact types"
        )
    if len(edges) != len(temperatures) + 1 or not temperatures:
        raise KerrReturningRadiationThermalSpectrumVerificationError(
            "source profile annulus arrays have inconsistent shapes"
        )
    checked_edges = tuple(
        _exact_positive_float(edge, f"annulus edge {index}")
        for index, edge in enumerate(edges)
    )
    checked_temperatures = tuple(
        _exact_nonnegative_float(value, f"outgoing temperature {index}")
        for index, value in enumerate(temperatures)
    )
    if any(right <= left for left, right in zip(checked_edges, checked_edges[1:])):
        raise KerrReturningRadiationThermalSpectrumVerificationError(
            "source profile annulus edges are not strictly increasing"
        )
    checked_correction = _exact_positive_float(correction, "disk.colour_correction")
    if checked_correction < 1.0:
        raise KerrReturningRadiationThermalSpectrumVerificationError(
            "disk colour correction is below one"
        )
    _digest(source_sha, "source profile descriptor SHA-256")
    _digest(disk_sha, "Novikov-Thorne disk descriptor SHA-256")
    if type(classification) is not str or classification not in (
        CERTIFIED_KERR_FORWARD_CLASSIFICATIONS
    ):
        raise KerrReturningRadiationThermalSpectrumVerificationError(
            "source profile has an unsupported certified classification"
        )
    descriptor = {
        "annuli": {
            "edgesOverMass": checked_edges,
            "outgoingEffectiveTemperatureK": checked_temperatures,
            "queryBoundaryConvention": QUERY_BOUNDARY_CONVENTION,
            "valueModel": "piecewise constant thermal-profile annuli",
        },
        "binding": {
            "colourCorrection": checked_correction,
            "novikovThorneDiskDescriptorSha256": disk_sha,
            "profileGeometryProvenanceCertified": True,
            "profileProvenanceClassification": classification,
            "sourceProfileDescriptorSha256": source_sha,
        },
        "emission": {
            "angularLaw": "isotropic over the local outward hemisphere",
            "batchHotPath": {
                "buildsPerFrequencyEvidenceSamples": False,
                "callerMustRevalidateProviderBeforeRenderLoop": True,
                "radiusAndAnnulusResolvedOncePerBatch": True,
            },
            "frequencyFrame": FREQUENCY_FRAME,
            "includesD20": False,
            "includesTransferG3": False,
            "oneFaceLocalEmission": True,
            "planck": _planck_descriptor(),
            "specificIntensityUnits": SPECIFIC_INTENSITY_UNITS,
        },
        "implementationId": IMPLEMENTATION_ID,
        "scientificBoundary": {
            "includesAtmosphere": False,
            "includesGRMHD": False,
            "includesPolarization": False,
            "includesReturningRadiationStressWorkFS": False,
            "includesScattering": False,
            "includesSpectralRedistribution": False,
            "isIndependentOracle": False,
            "warning": (
                "shared Planck/NT constants and same-code replay; piecewise "
                "thermal profile only"
            ),
        },
    }
    descriptor_json = _canonical_json(descriptor)
    result = object.__new__(CertifiedReturningRadiationThermalSpectrumProvider)
    for name, stored_value in (
        ("annulus_edges_over_mass", checked_edges),
        ("outgoing_effective_temperature_k", checked_temperatures),
        ("colour_correction", checked_correction),
        ("source_profile_descriptor_sha256", source_sha),
        ("novikov_thorne_disk_descriptor_sha256", disk_sha),
        ("specific_intensity_units", SPECIFIC_INTENSITY_UNITS),
        ("frequency_frame", FREQUENCY_FRAME),
        ("query_boundary_convention", QUERY_BOUNDARY_CONVENTION),
        ("_profile", profile),
        ("_descriptor_json", descriptor_json),
        (
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        ),
    ):
        object.__setattr__(result, name, stored_value)
    return result


def build_certified_returning_radiation_thermal_spectrum_provider(
    profile: AxisymmetricReturningRadiationThermalProfile,
) -> CertifiedReturningRadiationThermalSpectrumProvider:
    """Build a provider only from an exactly replay-certified profile."""

    checked_profile = _validated_certified_profile(profile)
    return _build_provider_from_validated_profile(checked_profile)


def verify_certified_returning_radiation_thermal_spectrum_provider(
    provider: CertifiedReturningRadiationThermalSpectrumProvider,
) -> None:
    """Replay the source profile and provider descriptor exactly."""

    if type(provider) is not CertifiedReturningRadiationThermalSpectrumProvider:
        raise TypeError(
            "provider must be an exact "
            "CertifiedReturningRadiationThermalSpectrumProvider"
        )
    try:
        profile = _validated_certified_profile(
            object.__getattribute__(provider, "_profile")
        )
        rebuilt = _build_provider_from_validated_profile(profile)
    except (TypeError, ValueError, RuntimeError) as error:
        raise KerrReturningRadiationThermalSpectrumVerificationError(
            f"provider could not be reproduced: {error}"
        ) from error
    public_fields = (
        "annulus_edges_over_mass",
        "outgoing_effective_temperature_k",
        "colour_correction",
        "source_profile_descriptor_sha256",
        "novikov_thorne_disk_descriptor_sha256",
        "specific_intensity_units",
        "frequency_frame",
        "query_boundary_convention",
        "_descriptor_json",
        "_descriptor_sha256",
    )
    for name in public_fields:
        _require_exact_tree(
            object.__getattribute__(provider, name),
            object.__getattribute__(rebuilt, name),
            f"provider.{name}",
        )
    _digest(
        object.__getattribute__(provider, "source_profile_descriptor_sha256"),
        "provider.source_profile_descriptor_sha256",
    )
    _digest(
        object.__getattribute__(provider, "novikov_thorne_disk_descriptor_sha256"),
        "provider.novikov_thorne_disk_descriptor_sha256",
    )


def verify_certified_returning_radiation_thermal_spectrum_sample(
    sample: CertifiedReturningRadiationThermalSpectrumSample,
) -> None:
    """Replay a sample after fully replay-validating its provider."""

    if type(sample) is not CertifiedReturningRadiationThermalSpectrumSample:
        raise TypeError(
            "sample must be an exact "
            "CertifiedReturningRadiationThermalSpectrumSample"
        )
    try:
        provider = object.__getattribute__(sample, "_provider")
        verify_certified_returning_radiation_thermal_spectrum_provider(provider)
        rebuilt = provider.sample(
            object.__getattribute__(sample, "radius_over_mass"),
            object.__getattribute__(sample, "emitted_frequency_hz"),
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise KerrReturningRadiationThermalSpectrumVerificationError(
            f"sample could not be reproduced: {error}"
        ) from error
    fields = (
        "annulus_index",
        "radius_over_mass",
        "emitted_frequency_hz",
        "effective_temperature_k",
        "colour_temperature_k",
        "colour_correction",
        "isotropic_specific_intensity_nu",
        "specific_intensity_units",
        "frequency_frame",
        "provider_descriptor_sha256",
        "source_profile_descriptor_sha256",
        "_descriptor_json",
        "_descriptor_sha256",
    )
    for name in fields:
        _require_exact_tree(
            object.__getattribute__(sample, name),
            object.__getattribute__(rebuilt, name),
            f"sample.{name}",
        )
    for name in (
        "provider_descriptor_sha256",
        "source_profile_descriptor_sha256",
        "_descriptor_sha256",
    ):
        _digest(object.__getattribute__(sample, name), f"sample.{name}")


__all__ = [
    "CertifiedReturningRadiationThermalSpectrumProvider",
    "CertifiedReturningRadiationThermalSpectrumSample",
    "FREQUENCY_FRAME",
    "IMPLEMENTATION_ID",
    "KerrReturningRadiationThermalSpectrumError",
    "KerrReturningRadiationThermalSpectrumVerificationError",
    "PLANCK_IMPLEMENTATION_ID",
    "QUERY_BOUNDARY_CONVENTION",
    "SCIENTIFIC_STATUS",
    "SPECIFIC_INTENSITY_UNITS",
    "build_certified_returning_radiation_thermal_spectrum_provider",
    "colour_corrected_planck_specific_intensity_nu",
    "verify_certified_returning_radiation_thermal_spectrum_provider",
    "verify_certified_returning_radiation_thermal_spectrum_sample",
]
