"""Forward finite-volume returning-radiation energy kernel for Kerr faces.

This module constructs a *finite-grid*, local-comoving, bolometric energy
operator.  A source surface element ``dA_e`` emits one unit of local flux with
the KERRBB-D20 angular measure.  If a traced ray returns to receiver annulus
``i``, its contribution is

``dK[i,j] = (dA_e / A_i) w_emitted g**2``.

Consequently every source column has the directly auditable energy identity

``sum_i A_i K[i,j] / A_j = <g**2 1_return>_j``.

The two photosphere faces remain separate.  Matrices use the order
``K[receiver_annulus][emitter_annulus]`` and are published as ``UU``, ``UL``,
``LU``, and ``LL`` where the first letter is the receiver face.  There is no
implicit factor of two.  Only an explicitly verified equatorial-symmetry
reduction may combine the four blocks.

The ``g**2`` form is the forward photon-energy/current finite-volume measure;
it must not be replaced by the single-ray ``g**4`` or incidence-weighted
diagnostics.  This calculation is same-code replayable, but it has no
independent receiver-centred reverse-ray/Jacobian oracle and has not been
cross-validated coefficient-by-coefficient against the repository's separate
shared-code receiver-direction primitive.  It omits the KERRBB stress/work term
``F_S``, spectral redistribution, scattering, polarization, an atmosphere
solution, GRMHD, and continuum-error guarantees.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Final, Mapping

from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
    integrate_kerr_finite_thickness_annulus_area,
    kerr_finite_thickness_area_density,
)
from offline.kerr_finite_thickness_emitter import KerrFiniteThicknessFaceEmitter
from offline.kerr_finite_thickness_launch import (
    KerrFiniteThicknessEmissionLaunch,
    KerrFiniteThicknessSurfaceFrame,
)
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
from offline.kerr_returning_radiation_rays import (
    KerrReturningRadiationRayPrimitive,
    _consume_issued_kerr_returning_radiation_direction,
    _trace_issued_kerr_returning_radiation_direction,
)
from offline.returning_radiation import AxisymmetricReturningRadiationKernel
from offline.returning_radiation_fate_quadrature import (
    EmittedFluxDirectionNode,
    kerrbb_d20_emitted_flux_direction_nodes,
)


IMPLEMENTATION_ID: Final = (
    "kerr-finite-thickness-forward-finite-volume-returning-energy-kernel/v1"
)
KERRBB_SOURCE_URL: Final = "https://arxiv.org/abs/astro-ph/0411583"

_FACES: Final = (UPPER, LOWER)
_FATES: Final = (
    "return-upper",
    "return-lower",
    "captured",
    "escaped",
    "plunge-sink",
)
_MINIMUM_ORDER: Final = 4
_MAXIMUM_ORDER: Final = 64
_MAXIMUM_PSI_COUNT: Final = 256
_MAXIMUM_ABSOLUTE_TOLERANCE: Final = 0.25
_MAXIMUM_RELATIVE_TOLERANCE: Final = 0.25
_MAXIMUM_DIRECTION_EVALUATIONS: Final = 2_000_000
_MAXIMUM_WHOLE_RAY_TRACES: Final = 8_000_000
_GAUSS_MAXIMUM_ITERATIONS: Final = 64
_FATE_CLOSURE_MAXIMUM_ULPS_AT_UNITY: Final = 8
_FATE_CLOSURE_MAXIMUM_NEXTAFTER_STEPS: Final = 8
_FATE_CLOSURE_MAXIMUM_CORRECTION_ULPS_AT_UNITY: Final = 9
_FATE_CLOSURE_SINGLE_FSUM_MAXIMUM_CORRECT_ROUNDING_ERROR_AT_UNITY: Final = (
    0.5 * math.ulp(1.0)
)
_FATE_CLOSURE_MINIMUM_DIRECTION_WEIGHT: Final = (
    _FATE_CLOSURE_MAXIMUM_CORRECTION_ULPS_AT_UNITY * math.ulp(1.0)
)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "same-code finite-grid forward finite-volume local-comoving "
            "bolometric returning-radiation energy kernel"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "primarySource": KERRBB_SOURCE_URL,
        "coefficientEquation": "Delta K=(Delta A_e/A_i) w_emitted g^2",
        "columnEnergyEquation": (
            "sum_i A_i K[i,j]/A_j=<g^2 1_return>_j"
        ),
        "matrixIndexOrder": "K[receiverFace,receiverAnnulus][emitterFace,emitterAnnulus]",
        "angularLaw": "KERRBB D20 f(mu)=1/2+3mu/4",
        "isSameCodePhysicsReplayVerified": True,
        "isFiniteGridEnergyOnlyKernel": True,
        "hasIndependentReceiverReverseRayOracle": False,
        "hasIndependentPhysicsOracle": False,
        "sharedCodeReceiverDirectionPrimitiveExistsSeparately": True,
        "hasPerCoefficientForwardReverseCrossValidation": False,
        "requiresFineCoarseSameFiniteVolumeReceiverBin": True,
        "primitiveConsumptionBoundary": (
            "public ray primitives require public fine/coarse replay; this "
            "forward producer alone consumes a process-local fresh-issued "
            "immutable payload after the canonical tracer completed both rays"
        ),
        "fateFractionBinary64ClosureMethod": (
            "reject non-finite, negative, or residuals outside the hard "
            "binary64 representation gate; correct "
            "the largest positive fate with a compensated residual then bounded "
            "nextafter steps until math.fsum is exactly one"
        ),
        "fateFractionClosureMaximumUlpsAtUnity": (
            _FATE_CLOSURE_MAXIMUM_ULPS_AT_UNITY
        ),
        "fateFractionClosureMaximumNextafterSteps": (
            _FATE_CLOSURE_MAXIMUM_NEXTAFTER_STEPS
        ),
        "fateFractionClosureMaximumCorrectionUlpsAtUnity": (
            _FATE_CLOSURE_MAXIMUM_CORRECTION_ULPS_AT_UNITY
        ),
        "fateFractionClosureStableTieBreak": "first fate in declared order",
        "fateFractionClosureIsMissingFateAllowance": False,
        "fateFractionClosureInterpretation": (
            "binary64 representation closure after a hard raw-residual gate; "
            "not a physical fate-mass reallocation or missing-fate allowance"
        ),
        "fateFractionSingleFsumMaximumCorrectRoundingErrorAtUnity": (
            _FATE_CLOSURE_SINGLE_FSUM_MAXIMUM_CORRECT_ROUNDING_ERROR_AT_UNITY
        ),
        "fateFractionPrePostStructuralUndetectableMass": 0.0,
        "fateFractionStructuralExactOneFatePerDirection": True,
        "fateFractionStructuralAccounting": (
            "pre-classification ordinals are exactly 0..N-1; fate-bucket "
            "entries sorted by ordinal must match every pre-classification "
            "exact type, ordinal, and normalized-weight binary64 hex"
        ),
        "fateFractionMergeAccounting": (
            "exact ordered annulus-group coverage plus an independently "
            "flattened area-weighted total; closure remains representation-only"
        ),
        "fateFractionResidualCauseNumericallyIdentified": False,
        "fateFractionMinimumDirectionWeightExclusive": (
            _FATE_CLOSURE_MINIMUM_DIRECTION_WEIGHT
        ),
        "includesReturningRadiationStressWorkFS": False,
        "includesSpectralRedistribution": False,
        "includesScattering": False,
        "includesPolarization": False,
        "includesSolvedAtmosphere": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "isCompleteKerrbb": False,
        "rigorousContinuumErrorBound": False,
        "prohibitedClaim": (
            "Do not describe this same-code finite-grid energy-only K as an "
            "independent receiver reverse-ray oracle, complete KERRBB, F_S, "
            "atmosphere, spectral/polarized transfer, continuum theorem, or GRMHD."
        ),
    }
)


class KerrReturningRadiationKernelError(RuntimeError):
    """Base class for fail-closed finite-volume kernel failures."""


class KerrReturningRadiationKernelConvergenceError(
    KerrReturningRadiationKernelError
):
    """Raised when a mandatory grid or symmetry gate does not pass."""


class KerrReturningRadiationKernelVerificationError(
    KerrReturningRadiationKernelError
):
    """Raised when an immutable result does not match exact same-code replay."""


@dataclass(frozen=True, slots=True)
class _FateContribution:
    """Exact local direction identity and its normalized emitted-flux weight."""

    local_direction_ordinal: int
    normalized_weight: float

    def __post_init__(self) -> None:
        if (
            type(self.local_direction_ordinal) is not int
            or self.local_direction_ordinal < 0
        ):
            raise TypeError(
                "fate contribution ordinal must be a non-negative exact int"
            )
        if type(self.normalized_weight) is not float:
            raise TypeError("fate contribution weight must be an exact float")
        if (
            not math.isfinite(self.normalized_weight)
            or self.normalized_weight <= _FATE_CLOSURE_MINIMUM_DIRECTION_WEIGHT
        ):
            raise KerrReturningRadiationKernelError(
                "fate contribution is not finite-positive above the binary64 "
                "closure-correction bound"
            )


def _close_fate_fraction_binary64_roundoff(
    raw_values: tuple[float, ...],
    *,
    independently_accumulated_total: float,
) -> tuple[float, ...]:
    """Close a structurally audited five-fate binary64 representation.

    Only a residual within eight ULPs at unity is eligible for representation
    closure; its size alone does not identify its cause.  The correction target
    is the largest strictly positive fate; ties use ``_FATES`` order.  A
    compensated residual update is followed by a bounded representable-neighbour
    walk because one rounded ``+= residual`` need not make ``math.fsum`` exactly
    one.  Callers must independently prove structural accounting.  Every other
    case fails closed.
    """

    if type(raw_values) is not tuple or len(raw_values) != len(_FATES):
        raise TypeError("raw fate fractions must be an exact five-float tuple")
    for index, value in enumerate(raw_values):
        if type(value) is not float:
            raise TypeError(
                f"raw fate fraction {_FATES[index]} must be an exact float"
            )
        if (
            not math.isfinite(value)
            or value < 0.0
            or (value == 0.0 and math.copysign(1.0, value) < 0.0)
        ):
            raise KerrReturningRadiationKernelError(
                "raw fate fractions must be finite and use canonical "
                "non-negative binary64 values"
            )
    try:
        raw_total = math.fsum(raw_values)
    except OverflowError as error:
        raise KerrReturningRadiationKernelError(
            "raw fate partition overflows binary64"
        ) from error
    if not math.isfinite(raw_total) or raw_total <= 0.0:
        raise KerrReturningRadiationKernelError(
            "raw fate partition must have finite positive mass"
        )
    if type(independently_accumulated_total) is not float:
        raise TypeError(
            "independently accumulated fate total must be an exact float"
        )
    if (
        not math.isfinite(independently_accumulated_total)
        or independently_accumulated_total <= 0.0
        or (
            independently_accumulated_total == 0.0
            and math.copysign(1.0, independently_accumulated_total) < 0.0
        )
    ):
        raise KerrReturningRadiationKernelError(
            "independently accumulated fate total must be finite and positive"
        )

    unit_ulp = math.ulp(1.0)
    roundoff_bound = _FATE_CLOSURE_MAXIMUM_ULPS_AT_UNITY * unit_ulp
    independent_residual = 1.0 - independently_accumulated_total
    if abs(independent_residual) > roundoff_bound:
        raise KerrReturningRadiationKernelError(
            "independent fate total exceeds the strict binary64 representation gate"
        )
    if raw_total.hex() == 1.0.hex():
        if any(value > 1.0 for value in raw_values):
            raise KerrReturningRadiationKernelError(
                "exactly closed fate partition contains an invalid fraction"
            )
        return raw_values

    residual = 1.0 - raw_total
    if abs(residual) > roundoff_bound:
        raise KerrReturningRadiationKernelError(
            "fate partition residual exceeds the strict binary64 representation gate"
        )

    positive_indices = tuple(
        index for index, value in enumerate(raw_values) if value > 0.0
    )
    if not positive_indices:
        raise KerrReturningRadiationKernelError(
            "raw fate partition has no positive correction target"
        )
    correction_index = max(
        positive_indices,
        key=lambda index: raw_values[index],
    )
    original = raw_values[correction_index]
    corrected_values = list(raw_values)
    corrected_values[correction_index] = math.fsum((original, residual))

    for step in range(_FATE_CLOSURE_MAXIMUM_NEXTAFTER_STEPS + 1):
        corrected = corrected_values[correction_index]
        if not math.isfinite(corrected) or corrected < 0.0 or corrected > 1.0:
            raise KerrReturningRadiationKernelError(
                "binary64 fate closure produced an invalid fraction"
            )
        corrected_total = math.fsum(corrected_values)
        if corrected_total.hex() == 1.0.hex():
            correction = abs(math.fsum((corrected, -original)))
            correction_bound = (
                _FATE_CLOSURE_MAXIMUM_CORRECTION_ULPS_AT_UNITY * unit_ulp
            )
            if not math.isfinite(correction) or correction > correction_bound:
                raise KerrReturningRadiationKernelError(
                    "binary64 fate closure correction exceeds its hard bound"
                )
            return tuple(corrected_values)
        if step == _FATE_CLOSURE_MAXIMUM_NEXTAFTER_STEPS:
            break
        direction = math.inf if corrected_total < 1.0 else -math.inf
        next_corrected = math.nextafter(corrected, direction)
        if next_corrected.hex() == corrected.hex():
            break
        corrected_values[correction_index] = next_corrected

    raise KerrReturningRadiationKernelError(
        "fate partition could not close exactly within the binary64 step bound"
    )


def _audited_fate_fraction_partition(
    unclassified_contributions: list[_FateContribution],
    classified_contributions: list[list[_FateContribution]],
    *,
    expected_direction_count: int,
) -> tuple[float, ...]:
    """Prove structural one-fate accounting before representation closure."""

    if type(unclassified_contributions) is not list:
        raise TypeError("unclassified fate contributions must be an exact list")
    if (
        type(classified_contributions) is not list
        or len(classified_contributions) != len(_FATES)
        or any(type(bucket) is not list for bucket in classified_contributions)
    ):
        raise TypeError("classified fate contributions must be five exact lists")
    if type(expected_direction_count) is not int or expected_direction_count < 1:
        raise TypeError("expected fate direction count must be a positive exact int")

    flattened = [
        contribution
        for bucket in classified_contributions
        for contribution in bucket
    ]
    if (
        len(unclassified_contributions) != expected_direction_count
        or len(flattened) != expected_direction_count
    ):
        raise KerrReturningRadiationKernelError(
            "fate classification count differs from the exact direction grid"
        )
    for contributions in (unclassified_contributions, flattened):
        for contribution in contributions:
            if type(contribution) is not _FateContribution:
                raise TypeError(
                    "fate audit entries must use the exact contribution type"
                )
            ordinal = object.__getattribute__(
                contribution,
                "local_direction_ordinal",
            )
            weight = object.__getattribute__(contribution, "normalized_weight")
            if type(ordinal) is not int or ordinal < 0:
                raise TypeError(
                    "fate audit entry ordinal must be a non-negative exact int"
                )
            if type(weight) is not float:
                raise TypeError("fate audit entry weight must be an exact float")
            if (
                not math.isfinite(weight)
                or weight <= _FATE_CLOSURE_MINIMUM_DIRECTION_WEIGHT
            ):
                raise KerrReturningRadiationKernelError(
                    "fate audit entry weight violates its finite-positive bound"
                )
    expected_ordinals = tuple(range(expected_direction_count))
    pre_ordinals = tuple(
        contribution.local_direction_ordinal
        for contribution in unclassified_contributions
    )
    if pre_ordinals != expected_ordinals:
        raise KerrReturningRadiationKernelError(
            "pre-classification fate ordinals differ from the exact direction grid"
        )
    sorted_flattened = sorted(
        flattened,
        key=lambda contribution: contribution.local_direction_ordinal,
    )
    post_ordinals = tuple(
        contribution.local_direction_ordinal
        for contribution in sorted_flattened
    )
    if post_ordinals != expected_ordinals:
        raise KerrReturningRadiationKernelError(
            "post-classification fate ordinals are duplicated, missing, or reordered"
        )
    for pre, post in zip(unclassified_contributions, sorted_flattened):
        if (
            type(pre.local_direction_ordinal) is not int
            or type(post.local_direction_ordinal) is not int
            or type(pre.normalized_weight) is not float
            or type(post.normalized_weight) is not float
            or pre.local_direction_ordinal != post.local_direction_ordinal
            or pre.normalized_weight.hex() != post.normalized_weight.hex()
        ):
            raise KerrReturningRadiationKernelError(
                "post-classification fate entry differs from its exact pre-entry"
            )
    unclassified_total = math.fsum(
        contribution.normalized_weight
        for contribution in unclassified_contributions
    )
    classified_flat_total = math.fsum(
        contribution.normalized_weight for contribution in flattened
    )
    if unclassified_total.hex() != classified_flat_total.hex():
        raise KerrReturningRadiationKernelError(
            "pre-classification and flattened post-classification fate totals differ"
        )
    raw_values = tuple(
        math.fsum(contribution.normalized_weight for contribution in bucket)
        for bucket in classified_contributions
    )
    return _close_fate_fraction_binary64_roundoff(
        raw_values,
        independently_accumulated_total=unclassified_total,
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
        raise KerrReturningRadiationKernelError(
            "kernel descriptor is not finite canonical JSON"
        ) from error


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite exact float")
    return value


def _exact_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive exact int")
    return value


def _trusted_attribute(value: Any, name: str, path: str) -> Any:
    try:
        return object.__getattribute__(value, name)
    except (AttributeError, TypeError) as error:
        raise KerrReturningRadiationKernelVerificationError(
            f"{path}.{name} is missing"
        ) from error


def _require_exact_schema_types(actual: Any, template: Any, path: str) -> None:
    """Check trusted schema types before constructors or ``asdict`` run."""

    if type(actual) is not type(template):
        raise KerrReturningRadiationKernelVerificationError(
            f"{path} has non-exact type {type(actual).__name__}; "
            f"expected {type(template).__name__}"
        )
    if is_dataclass(template) and not isinstance(template, type):
        for item in fields(template):
            _require_exact_schema_types(
                _trusted_attribute(actual, item.name, path),
                _trusted_attribute(template, item.name, path),
                f"{path}.{item.name}",
            )
        return
    if type(template) is tuple:
        if len(actual) != len(template):
            raise KerrReturningRadiationKernelVerificationError(
                f"{path} tuple length differs from the trusted schema"
            )
        for index, (actual_item, template_item) in enumerate(zip(actual, template)):
            _require_exact_schema_types(
                actual_item,
                template_item,
                f"{path}[{index}]",
            )
        return
    if type(template) not in (float, int, bool, str, type(None)):
        raise KerrReturningRadiationKernelVerificationError(
            f"{path} uses unsupported schema type {type(template).__name__}"
        )


def _require_trusted_exact_tree(actual: Any, expected: Any, path: str) -> None:
    """Compare a replay tree without invoking attacker-controlled equality."""

    _require_exact_schema_types(actual, expected, path)
    if is_dataclass(expected) and not isinstance(expected, type):
        for item in fields(expected):
            _require_trusted_exact_tree(
                _trusted_attribute(actual, item.name, path),
                _trusted_attribute(expected, item.name, path),
                f"{path}.{item.name}",
            )
        return
    if type(expected) is tuple:
        if len(actual) != len(expected):
            raise KerrReturningRadiationKernelVerificationError(
                f"{path} tuple length differs"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _require_trusted_exact_tree(
                actual_item,
                expected_item,
                f"{path}[{index}]",
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
        raise KerrReturningRadiationKernelVerificationError(
            f"{path} has unsupported trusted type {type(expected).__name__}"
        )
    if differs:
        raise KerrReturningRadiationKernelVerificationError(
            f"{path} differs from trusted same-code replay"
        )


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationKernelPolicy:
    """Finite angular/radial resolution, convergence, and work gates."""

    rho_order: int = 8
    mu_order: int = 8
    psi_count: int = 8
    absolute_tolerance: float = 2.0e-2
    relative_tolerance: float = 5.0e-2
    symmetry_absolute_tolerance: float = 2.0e-8
    symmetry_relative_tolerance: float = 2.0e-7
    maximum_direction_evaluations: int = _MAXIMUM_DIRECTION_EVALUATIONS
    maximum_whole_ray_traces: int = _MAXIMUM_WHOLE_RAY_TRACES

    def __post_init__(self) -> None:
        for name in ("rho_order", "mu_order"):
            value = _exact_positive_int(getattr(self, name), name)
            if value < _MINIMUM_ORDER or value > _MAXIMUM_ORDER or value % 2:
                raise ValueError(
                    f"{name} must be even and lie in "
                    f"[{_MINIMUM_ORDER}, {_MAXIMUM_ORDER}]"
                )
        psi = _exact_positive_int(self.psi_count, "psi_count")
        if psi < _MINIMUM_ORDER or psi > _MAXIMUM_PSI_COUNT or psi % 2:
            raise ValueError(
                "psi_count must be even and lie in "
                f"[{_MINIMUM_ORDER}, {_MAXIMUM_PSI_COUNT}]"
            )
        for name, maximum in (
            ("absolute_tolerance", _MAXIMUM_ABSOLUTE_TOLERANCE),
            ("relative_tolerance", _MAXIMUM_RELATIVE_TOLERANCE),
            ("symmetry_absolute_tolerance", _MAXIMUM_ABSOLUTE_TOLERANCE),
            ("symmetry_relative_tolerance", _MAXIMUM_RELATIVE_TOLERANCE),
        ):
            value = _exact_finite_float(getattr(self, name), name)
            if value <= 0.0 or value > maximum:
                raise ValueError(f"{name} must lie in (0, {maximum}]")
        directions = _exact_positive_int(
            self.maximum_direction_evaluations,
            "maximum_direction_evaluations",
        )
        rays = _exact_positive_int(
            self.maximum_whole_ray_traces,
            "maximum_whole_ray_traces",
        )
        if directions > _MAXIMUM_DIRECTION_EVALUATIONS:
            raise ValueError("direction budget exceeds the hard policy maximum")
        if rays > _MAXIMUM_WHOLE_RAY_TRACES:
            raise ValueError("whole-ray budget exceeds the hard policy maximum")

    def descriptor(self) -> Mapping[str, Any]:
        return MappingProxyType(asdict(self))


@dataclass(frozen=True, slots=True)
class FiniteVolumeFateFractions:
    """Area-averaged one-face emitted-flux fate partition; no ``g`` factor."""

    return_upper: float
    return_lower: float
    captured: float
    escaped: float
    plunge_sink: float

    def __post_init__(self) -> None:
        for name in (
            "return_upper",
            "return_lower",
            "captured",
            "escaped",
            "plunge_sink",
        ):
            value = _exact_finite_float(getattr(self, name), name)
            if (
                value < 0.0
                or value > 1.0
                or (value == 0.0 and math.copysign(1.0, value) < 0.0)
            ):
                raise ValueError(
                    f"{name} must lie in [0, 1] and use canonical positive zero"
                )
        if self.total.hex() != 1.0.hex():
            raise ValueError("finite-volume fate fractions must sum exactly to one")

    @property
    def returning(self) -> float:
        return math.fsum((self.return_upper, self.return_lower))

    @property
    def total(self) -> float:
        return math.fsum(self.as_tuple())

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.return_upper,
            self.return_lower,
            self.captured,
            self.escaped,
            self.plunge_sink,
        )


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationGridDifference:
    """Full-grid difference from one independently coarsened/shifted grid."""

    matrix_maximum_absolute_difference: float
    matrix_maximum_scaled_difference: float
    g2_column_maximum_absolute_difference: float
    g2_column_maximum_scaled_difference: float
    fate_maximum_absolute_difference: float
    fate_maximum_scaled_difference: float
    converged: bool

    def __post_init__(self) -> None:
        for name in (
            "matrix_maximum_absolute_difference",
            "matrix_maximum_scaled_difference",
            "g2_column_maximum_absolute_difference",
            "g2_column_maximum_scaled_difference",
            "fate_maximum_absolute_difference",
            "fate_maximum_scaled_difference",
        ):
            value = _exact_finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if type(self.converged) is not bool:
            raise TypeError("converged must be an exact bool")
        expected = max(
            self.matrix_maximum_scaled_difference,
            self.g2_column_maximum_scaled_difference,
            self.fate_maximum_scaled_difference,
        ) <= 1.0
        if self.converged is not expected:
            raise ValueError("grid convergence flag disagrees with scaled gates")


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationKernelConvergence:
    half_rho: KerrReturningRadiationGridDifference
    half_mu: KerrReturningRadiationGridDifference
    half_psi: KerrReturningRadiationGridDifference
    phase_shifted: KerrReturningRadiationGridDifference
    converged: bool

    def __post_init__(self) -> None:
        for name in ("half_rho", "half_mu", "half_psi", "phase_shifted"):
            if type(getattr(self, name)) is not KerrReturningRadiationGridDifference:
                raise TypeError(f"{name} must be an exact grid difference")
        if type(self.converged) is not bool:
            raise TypeError("converged must be an exact bool")
        expected = all(
            item.converged
            for item in (self.half_rho, self.half_mu, self.half_psi, self.phase_shifted)
        )
        if self.converged is not expected:
            raise ValueError("aggregate convergence disagrees with component gates")


@dataclass(frozen=True, slots=True)
class _DirectionTransport:
    fate: str
    receiver_face: str | None
    receiver_radius_over_mass: float | None
    frequency_ratio: float | None
    g2: float
    primitive_descriptor_sha256: str
    coarse_receiver_face: str | None = None
    coarse_receiver_radius_over_mass: float | None = None

    def __post_init__(self) -> None:
        if type(self.fate) is not str or self.fate not in _FATES:
            raise ValueError("direction fate is unsupported")
        g2 = _exact_finite_float(self.g2, "g2")
        if self.fate.startswith("return-"):
            if (
                type(self.receiver_face) is not str
                or self.receiver_face.encode("utf-8")
                not in (UPPER.encode("utf-8"), LOWER.encode("utf-8"))
            ):
                raise ValueError("returned direction requires an exact receiver face")
            radius = _exact_finite_float(
                self.receiver_radius_over_mass,
                "receiver_radius_over_mass",
            )
            ratio = _exact_finite_float(self.frequency_ratio, "frequency_ratio")
            if radius <= 0.0 or ratio <= 0.0 or g2 <= 0.0:
                raise ValueError("returned direction radius and shifts must be positive")
            if g2.hex() != (ratio * ratio).hex():
                raise ValueError("g2 must be exactly recomputed from frequency ratio")
            expected = UPPER if self.fate == "return-upper" else LOWER
            if self.receiver_face.encode("utf-8") != expected.encode("utf-8"):
                raise ValueError("returned fate and receiver face disagree")
            if (
                type(self.coarse_receiver_face) is not str
                or self.coarse_receiver_face.encode("utf-8")
                != self.receiver_face.encode("utf-8")
            ):
                raise ValueError(
                    "returned direction requires the exact coarse receiver face"
                )
            coarse_radius = _exact_finite_float(
                self.coarse_receiver_radius_over_mass,
                "coarse_receiver_radius_over_mass",
            )
            if coarse_radius <= 0.0:
                raise ValueError("coarse receiver radius must be positive")
        elif any(
            value is not None
            for value in (
                self.receiver_face,
                self.receiver_radius_over_mass,
                self.frequency_ratio,
                self.coarse_receiver_face,
                self.coarse_receiver_radius_over_mass,
            )
        ) or g2.hex() != 0.0.hex():
            raise ValueError("non-returning direction cannot carry receiver data")
        if (
            type(self.primitive_descriptor_sha256) is not str
            or len(self.primitive_descriptor_sha256) != 64
            or self.primitive_descriptor_sha256.lower()
            != self.primitive_descriptor_sha256
        ):
            raise ValueError("primitive identity must be lowercase SHA-256")
        try:
            bytes.fromhex(self.primitive_descriptor_sha256)
        except ValueError as error:
            raise ValueError("primitive identity must be hexadecimal SHA-256") from error


@dataclass(frozen=True, slots=True)
class _ForwardDirectionSample:
    """Exact canonical address and quadrature node presented to a provider."""

    pass_index: int
    pass_name: str
    source_face: str
    source_annulus_index: int
    rho_index: int
    mu_index: int
    psi_index: int
    source_radius_over_mass: float
    rho_area_over_mass_squared: float
    emission_angle_cosine: float
    tangent_azimuth_rad: float
    normalized_emitted_flux_weight: float


_ForwardDirectionTransportProvider = Callable[
    [_ForwardDirectionSample],
    _DirectionTransport,
]


def _trace_direction(
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
    _issued_tracer: Any = _trace_issued_kerr_returning_radiation_direction,
    _issued_consumer: Any = _consume_issued_kerr_returning_radiation_direction,
) -> _DirectionTransport:
    """Build, trace, consume a fresh issued result, then narrow one direction."""

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
    primitive, issue_token = _issued_tracer(
        launch,
        surface,
        termination=termination,
        ray_options=ray_options,
        surface_options=surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
    )
    if type(primitive) is not KerrReturningRadiationRayPrimitive:
        raise TypeError("direction tracer returned a non-exact ray primitive")
    issued = _issued_consumer(
        primitive,
        issue_token,
    )
    fate = object.__getattribute__(issued, "fate")
    if type(fate) is not str or fate not in _FATES:
        raise KerrReturningRadiationKernelError(
            "issued primitive has an unsupported fate"
        )
    descriptor_sha = object.__getattribute__(
        issued,
        "primitive_descriptor_sha256",
    )
    if fate.startswith("return-"):
        receiver_face = object.__getattribute__(issued, "receiver_face")
        receiver_radius = object.__getattribute__(
            issued,
            "receiver_radius_over_mass",
        )
        ratio = object.__getattribute__(
            issued,
            "emitter_to_receiver_frequency_ratio",
        )
        if type(receiver_face) is not str or receiver_face not in _FACES:
            raise KerrReturningRadiationKernelError(
                "returned primitive lacks an exact receiver face"
            )
        receiver_radius = _exact_finite_float(
            receiver_radius,
            "receiver_radius_over_mass",
        )
        ratio = _exact_finite_float(ratio, "frequency_ratio")
        if receiver_radius <= 0.0 or ratio <= 0.0:
            raise KerrReturningRadiationKernelError(
                "returned primitive has invalid receiver data"
            )
        g2 = ratio * ratio
        if not math.isfinite(g2) or g2 <= 0.0:
            raise KerrReturningRadiationKernelError("returned g^2 is invalid")
        coarse_face = object.__getattribute__(issued, "coarse_receiver_face")
        coarse_radius = object.__getattribute__(
            issued,
            "coarse_receiver_radius_over_mass",
        )
        if type(coarse_face) is not str or type(coarse_radius) is not float:
            raise KerrReturningRadiationKernelError(
                "issued coarse receiver evidence has non-exact field types"
            )
        return _DirectionTransport(
            fate,
            receiver_face,
            receiver_radius,
            ratio,
            g2,
            descriptor_sha,
            coarse_face,
            coarse_radius,
        )
    return _DirectionTransport(
        fate,
        None,
        None,
        None,
        0.0,
        descriptor_sha,
        None,
        None,
    )


def _gauss_legendre_unit_interval(order: int) -> tuple[tuple[float, float], ...]:
    order = _exact_positive_int(order, "Gauss-Legendre order")
    nodes = [0.0] * order
    weights = [0.0] * order
    half = (order + 1) // 2
    for index in range(half):
        root = math.cos(math.pi * (index + 0.75) / (order + 0.5))
        derivative = 0.0
        visited_iterates: list[tuple[float, float, float]] = []
        for _iteration in range(_GAUSS_MAXIMUM_ITERATIONS):
            previous = 1.0
            current = root
            for degree in range(2, order + 1):
                following = (
                    (2.0 * degree - 1.0) * root * current
                    - (degree - 1.0) * previous
                ) / degree
                previous, current = current, following
            derivative = order * (root * current - previous) / (root * root - 1.0)
            update = current / derivative
            next_root = root - update
            if next_root == root or abs(update) <= 2.0 * math.ulp(root):
                root = next_root
                break
            visited_iterates.append((root, current, derivative))
            cycle_start = next(
                (
                    item_index
                    for item_index, (item_root, _residual, _derivative) in enumerate(
                        visited_iterates
                    )
                    if next_root == item_root
                ),
                None,
            )
            if cycle_start is not None:
                # Preserve the historical Newton path for ordinary roots, but
                # close any resolved finite binary64 cycle deterministically.
                root, _residual, derivative = min(
                    visited_iterates[cycle_start:],
                    key=lambda item: (abs(item[1]), item[0]),
                )
                break
            root = next_root
        else:
            raise KerrReturningRadiationKernelError(
                "Gauss-Legendre root solve did not converge"
            )
        weight = 1.0 / ((1.0 - root * root) * derivative * derivative)
        nodes[index] = 0.5 * (1.0 - root)
        nodes[order - 1 - index] = 0.5 * (1.0 + root)
        weights[index] = weight
        weights[order - 1 - index] = weight
    rule = tuple(zip(nodes, weights))
    if any(
        not (0.0 < node < 1.0 and math.isfinite(weight) and weight > 0.0)
        for node, weight in rule
    ):
        raise KerrReturningRadiationKernelError(
            "Gauss-Legendre rule is not interior-positive"
        )
    if abs(math.fsum(weight for _node, weight in rule) - 1.0) > 64.0 * math.ulp(1.0):
        raise KerrReturningRadiationKernelError(
            "Gauss-Legendre rule does not integrate a constant"
        )
    return rule


def _validated_edges(
    values: tuple[float, ...],
    surface: KerrFiniteThicknessMultiSurface,
) -> tuple[float, ...]:
    if type(values) is not tuple or len(values) < 2:
        raise TypeError("annulus edges must be an exact tuple with at least two values")
    edges = tuple(
        _exact_finite_float(value, f"annulus edge {index}")
        for index, value in enumerate(values)
    )
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("annulus edges must be strictly increasing")
    inner = float(surface.calibration.isco_radius_over_mass)
    outer = float(surface.calibration.outer_radius_over_mass)
    if edges[0].hex() != inner.hex() or edges[-1].hex() != outer.hex():
        raise ValueError(
            "annulus edges must exactly and completely cover ISCO through R_out"
        )
    return edges


def _receiver_bin_index(radius: float, edges: tuple[float, ...]) -> int:
    radius = _exact_finite_float(radius, "returned receiver radius")
    index = bisect_right(edges, radius) - 1
    if index == len(edges) - 1 and radius.hex() == edges[-1].hex():
        index -= 1
    if index < 0 or index >= len(edges) - 1:
        raise KerrReturningRadiationKernelError(
            "returned ray lies outside the declared finite receiver grid"
        )
    return index


def _validated_return_receiver_bin(
    transport: _DirectionTransport,
    edges: tuple[float, ...],
) -> int:
    """Require fine/coarse receiver topology to agree on the user grid."""

    if type(transport) is not _DirectionTransport:
        raise TypeError("transport must be the exact internal direction result")
    if not transport.fate.startswith("return-"):
        raise ValueError("receiver-bin validation requires a returned direction")
    fine_radius = _exact_finite_float(
        transport.receiver_radius_over_mass,
        "fine receiver radius",
    )
    coarse_radius = _exact_finite_float(
        transport.coarse_receiver_radius_over_mass,
        "coarse receiver radius",
    )
    if (
        type(transport.receiver_face) is not str
        or type(transport.coarse_receiver_face) is not str
        or transport.receiver_face.encode("utf-8")
        != transport.coarse_receiver_face.encode("utf-8")
    ):
        raise KerrReturningRadiationKernelError(
            "fine/coarse receiver faces disagree before finite-volume binning"
        )
    fine_bin = _receiver_bin_index(fine_radius, edges)
    coarse_bin = _receiver_bin_index(coarse_radius, edges)
    if fine_bin != coarse_bin:
        raise KerrReturningRadiationKernelConvergenceError(
            "fine/coarse returned rays land in different receiver annuli: "
            f"fine rho/M={fine_radius:.17g} -> bin {fine_bin}, "
            f"coarse rho/M={coarse_radius:.17g} -> bin {coarse_bin}"
        )
    return fine_bin


def _validated_surface(surface: KerrFiniteThicknessMultiSurface) -> KerrFiniteThicknessMultiSurface:
    if type(surface) is not KerrFiniteThicknessMultiSurface:
        raise TypeError("surface must be the exact KerrFiniteThicknessMultiSurface")
    metric = object.__getattribute__(surface, "metric")
    calibration = object.__getattribute__(surface, "calibration")
    if type(metric) is not KerrKerrSchildMetric:
        raise TypeError("surface metric must be exact KerrKerrSchildMetric")
    if type(calibration) is not StationaryKerrFiniteThicknessCalibration:
        raise TypeError(
            "surface calibration must be exact StationaryKerrFiniteThicknessCalibration"
        )
    _require_exact_schema_types(metric, KerrKerrSchildMetric(), "surface.metric")
    _require_exact_schema_types(
        calibration,
        StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.0,
            eddington_scaled_mass_accretion_rate=0.01,
            outer_radius_over_mass=10.0,
        ),
        "surface.calibration",
    )
    rebuilt_metric = KerrKerrSchildMetric(**asdict(metric))
    rebuilt_calibration = StationaryKerrFiniteThicknessCalibration(
        **asdict(calibration)
    )
    _require_trusted_exact_tree(metric, rebuilt_metric, "surface.metric")
    _require_trusted_exact_tree(
        calibration,
        rebuilt_calibration,
        "surface.calibration",
    )
    rebuilt = KerrFiniteThicknessMultiSurface(
        rebuilt_metric,
        rebuilt_calibration,
    )
    _require_trusted_exact_tree(surface, rebuilt, "surface")
    return rebuilt


def _validated_auxiliary_inputs(
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions | None,
    coarse_surface_options: SurfaceEventOptions | None,
) -> tuple[
    KerrOblateTermination,
    RayTraceOptions,
    SurfaceEventOptions,
    RayTraceOptions | None,
    SurfaceEventOptions | None,
]:
    if type(termination) is not KerrOblateTermination:
        raise TypeError("termination must be the exact KerrOblateTermination")
    if type(ray_options) is not RayTraceOptions:
        raise TypeError("ray_options must be the exact RayTraceOptions")
    if type(surface_options) is not SurfaceEventOptions:
        raise TypeError("surface_options must be the exact SurfaceEventOptions")
    if (coarse_ray_options is None) != (coarse_surface_options is None):
        raise ValueError("coarse ray and surface options must be supplied together")
    if coarse_ray_options is not None and type(coarse_ray_options) is not RayTraceOptions:
        raise TypeError("coarse_ray_options must be exact RayTraceOptions or None")
    if (
        coarse_surface_options is not None
        and type(coarse_surface_options) is not SurfaceEventOptions
    ):
        raise TypeError(
            "coarse_surface_options must be exact SurfaceEventOptions or None"
        )
    # Constructors accept values such as bools, ints, and float subclasses in
    # several numerically typed fields.  Prove the exact schema *before*
    # asdict, reconstruction, policy comparisons, descriptor generation, or
    # ray tracing so none of their overloaded arithmetic/equality hooks can
    # participate in a scientific decision.
    _require_exact_schema_types(
        termination,
        KerrOblateTermination(0.0, 1.0, 2.0),
        "termination",
    )
    _require_exact_schema_types(ray_options, RayTraceOptions(), "ray_options")
    _require_exact_schema_types(
        surface_options,
        SurfaceEventOptions(),
        "surface_options",
    )
    if coarse_ray_options is not None:
        assert coarse_surface_options is not None
        _require_exact_schema_types(
            coarse_ray_options,
            RayTraceOptions(),
            "coarse_ray_options",
        )
        _require_exact_schema_types(
            coarse_surface_options,
            SurfaceEventOptions(),
            "coarse_surface_options",
        )
    try:
        rebuilt_termination = KerrOblateTermination(**asdict(termination))
        rebuilt_ray = RayTraceOptions(**asdict(ray_options))
        rebuilt_surface = SurfaceEventOptions(**asdict(surface_options))
        rebuilt_coarse_ray = (
            None
            if coarse_ray_options is None
            else RayTraceOptions(**asdict(coarse_ray_options))
        )
        rebuilt_coarse_surface = (
            None
            if coarse_surface_options is None
            else SurfaceEventOptions(**asdict(coarse_surface_options))
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"kernel trace inputs are stale: {error}") from error
    _require_trusted_exact_tree(termination, rebuilt_termination, "termination")
    _require_trusted_exact_tree(ray_options, rebuilt_ray, "ray_options")
    _require_trusted_exact_tree(surface_options, rebuilt_surface, "surface_options")
    if coarse_ray_options is not None:
        assert coarse_surface_options is not None
        assert rebuilt_coarse_ray is not None
        assert rebuilt_coarse_surface is not None
        _require_trusted_exact_tree(
            coarse_ray_options,
            rebuilt_coarse_ray,
            "coarse_ray_options",
        )
        _require_trusted_exact_tree(
            coarse_surface_options,
            rebuilt_coarse_surface,
            "coarse_surface_options",
        )
    return (
        rebuilt_termination,
        rebuilt_ray,
        rebuilt_surface,
        rebuilt_coarse_ray,
        rebuilt_coarse_surface,
    )


def _validated_policy(
    policy: KerrReturningRadiationKernelPolicy | None,
) -> KerrReturningRadiationKernelPolicy:
    selected = KerrReturningRadiationKernelPolicy() if policy is None else policy
    if type(selected) is not KerrReturningRadiationKernelPolicy:
        raise TypeError("policy must be the exact KerrReturningRadiationKernelPolicy")
    _require_exact_schema_types(
        selected,
        KerrReturningRadiationKernelPolicy(),
        "policy",
    )
    return KerrReturningRadiationKernelPolicy(**asdict(selected))


def _validated_area_policy(
    policy: KerrFiniteThicknessAreaQuadraturePolicy | None,
) -> KerrFiniteThicknessAreaQuadraturePolicy:
    selected = KerrFiniteThicknessAreaQuadraturePolicy() if policy is None else policy
    if type(selected) is not KerrFiniteThicknessAreaQuadraturePolicy:
        raise TypeError(
            "area_policy must be exact KerrFiniteThicknessAreaQuadraturePolicy"
        )
    _require_exact_schema_types(
        selected,
        KerrFiniteThicknessAreaQuadraturePolicy(),
        "area_policy",
    )
    rebuilt = KerrFiniteThicknessAreaQuadraturePolicy(
        gauss_legendre_order=selected.gauss_legendre_order,
        relative_tolerance=selected.relative_tolerance,
        absolute_tolerance_over_mass_squared=(
            selected.absolute_tolerance_over_mass_squared
        ),
        maximum_point_evaluations=selected.maximum_point_evaluations,
    )
    _require_trusted_exact_tree(selected, rebuilt, "area_policy")
    return rebuilt


def _annulus_areas(
    surface: KerrFiniteThicknessMultiSurface,
    edges: tuple[float, ...],
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    values: dict[str, list[float]] = {UPPER: [], LOWER: []}
    identities: dict[str, list[str]] = {UPPER: [], LOWER: []}
    for face in _FACES:
        for inner, outer in zip(edges, edges[1:]):
            result = integrate_kerr_finite_thickness_annulus_area(
                metric=surface.metric,
                calibration=surface.calibration,
                inner_radius_over_mass=inner,
                outer_radius_over_mass=outer,
                face=face,
                policy=area_policy,
            )
            result.revalidate()
            values[face].append(result.proper_area_over_mass_squared)
            identities[face].append(result.model_descriptor_sha256)
    return (
        tuple(values[UPPER]),
        tuple(values[LOWER]),
        tuple(identities[UPPER]),
        tuple(identities[LOWER]),
    )


def _rho_area_nodes(
    surface: KerrFiniteThicknessMultiSurface,
    inner: float,
    outer: float,
    face: str,
    order: int,
    exact_annulus_area: float,
) -> tuple[tuple[float, float], ...]:
    width = outer - inner
    provisional: list[tuple[float, float]] = []
    rule = _gauss_legendre_unit_interval(order)
    rho_nodes = _rho_coordinate_nodes_from_rule(inner, outer, rule)
    for rho, (_node, weight) in zip(rho_nodes, rule):
        density = kerr_finite_thickness_area_density(
            metric=surface.metric,
            calibration=surface.calibration,
            pseudo_cylindrical_radius_over_mass=rho,
            face=face,
        )
        density.revalidate()
        raw_area = (
            2.0
            * math.pi
            * width
            * weight
            * density.proper_area_density_over_mass_squared
        )
        if not math.isfinite(raw_area) or raw_area <= 0.0:
            raise KerrReturningRadiationKernelError(
                "source radial proper-area weight is invalid"
            )
        provisional.append((rho, raw_area))
    raw_total = math.fsum(weight for _rho, weight in provisional)
    scale = exact_annulus_area / raw_total
    normalized = [(rho, weight * scale) for rho, weight in provisional]
    for _closure_iteration in range(16):
        current = math.fsum(weight for _rho, weight in normalized)
        if current.hex() == exact_annulus_area.hex():
            break
        final_rho, final_weight = normalized[-1]
        candidate = final_weight + (exact_annulus_area - current)
        if candidate.hex() == final_weight.hex():
            candidate = math.nextafter(
                final_weight,
                math.inf if current < exact_annulus_area else -math.inf,
            )
        normalized[-1] = (final_rho, candidate)
    normalized_total = math.fsum(weight for _rho, weight in normalized)
    if (
        abs(normalized_total - exact_annulus_area)
        > 2.0 * math.ulp(exact_annulus_area)
        or any(weight <= 0.0 or not math.isfinite(weight) for _rho, weight in normalized)
    ):
        raise KerrReturningRadiationKernelError(
            "source radial weights cannot close to proper annulus area within "
            "the float64 roundoff gate"
        )
    return tuple(normalized)


def _rho_coordinate_nodes(
    inner: float,
    outer: float,
    order: int,
) -> tuple[float, ...]:
    """Return the exact radial coordinates shared by tracing and cache jobs."""

    return _rho_coordinate_nodes_from_rule(
        inner,
        outer,
        _gauss_legendre_unit_interval(order),
    )


def _rho_coordinate_nodes_from_rule(
    inner: float,
    outer: float,
    rule: tuple[tuple[float, float], ...],
) -> tuple[float, ...]:
    width = outer - inner
    return tuple(
        math.fsum((inner, width * node))
        for node, _weight in rule
    )


def _empty_contributions(count: int) -> list[list[list[float]]]:
    return [[[] for _emitter in range(count)] for _receiver in range(count)]


def _freeze_matrix(values: list[list[list[float]]]) -> tuple[tuple[float, ...], ...]:
    matrix = tuple(tuple(math.fsum(cell) for cell in row) for row in values)
    if any(value < 0.0 or not math.isfinite(value) for row in matrix for value in row):
        raise KerrReturningRadiationKernelError("kernel matrix is not finite/non-negative")
    return matrix


@dataclass(frozen=True, slots=True)
class _KernelGridEstimate:
    uu: tuple[tuple[float, ...], ...]
    ul: tuple[tuple[float, ...], ...]
    lu: tuple[tuple[float, ...], ...]
    ll: tuple[tuple[float, ...], ...]
    upper_fates: tuple[FiniteVolumeFateFractions, ...]
    lower_fates: tuple[FiniteVolumeFateFractions, ...]
    upper_g2_columns: tuple[float, ...]
    lower_g2_columns: tuple[float, ...]
    upper_column_closure_residuals: tuple[float, ...]
    lower_column_closure_residuals: tuple[float, ...]
    direction_evaluations: int
    sample_audit_sha256: str
    maximum_normalized_sample_weight: float
    maximum_normalized_sample_weight_witness: (
        "KerrForwardReturningRadiationSampleWeightWitness"
    )

    def matrix_vector(self) -> tuple[float, ...]:
        return tuple(
            value
            for block in (self.uu, self.ul, self.lu, self.ll)
            for row in block
            for value in row
        )

    def g2_vector(self) -> tuple[float, ...]:
        return (*self.upper_g2_columns, *self.lower_g2_columns)

    def fate_vector(self) -> tuple[float, ...]:
        return tuple(
            value
            for fractions in (*self.upper_fates, *self.lower_fates)
            for value in fractions.as_tuple()
        )


@dataclass(frozen=True, slots=True, init=False)
class KerrForwardReturningRadiationSampleWeightWitness:
    """Producer-owned coordinate attaining one pass's maximum sample weight."""

    pass_index: int
    pass_name: str
    source_face: str
    source_annulus_index: int
    rho_index: int
    mu_index: int
    psi_index: int
    source_radius_over_mass: float
    rho_area_over_mass_squared: float
    emission_angle_cosine: float
    tangent_azimuth_rad: float
    normalized_emitted_flux_direction_weight: float
    normalized_sample_weight: float

    def __init__(self) -> None:
        raise TypeError(
            "sample-weight witnesses are built only by the certified "
            "finite-volume integrator"
        )


@dataclass(frozen=True, slots=True, init=False)
class KerrForwardReturningRadiationGridEvidence:
    """Factory-only complete evidence for one forward quadrature pass."""

    pass_index: int
    pass_name: str
    rho_order: int
    mu_order: int
    psi_count: int
    phase_cells: float
    upper_receiver_upper_emitter_coefficients: tuple[tuple[float, ...], ...]
    upper_receiver_lower_emitter_coefficients: tuple[tuple[float, ...], ...]
    lower_receiver_upper_emitter_coefficients: tuple[tuple[float, ...], ...]
    lower_receiver_lower_emitter_coefficients: tuple[tuple[float, ...], ...]
    upper_emitter_fate_fractions: tuple[FiniteVolumeFateFractions, ...]
    lower_emitter_fate_fractions: tuple[FiniteVolumeFateFractions, ...]
    upper_emitter_g2_returned_power_columns: tuple[float, ...]
    lower_emitter_g2_returned_power_columns: tuple[float, ...]
    upper_emitter_g2_column_closure_residuals: tuple[float, ...]
    lower_emitter_g2_column_closure_residuals: tuple[float, ...]
    direction_evaluations: int
    sample_audit_sha256: str
    maximum_normalized_sample_weight: float
    maximum_normalized_sample_weight_witness: (
        KerrForwardReturningRadiationSampleWeightWitness
    )

    def __init__(self) -> None:
        raise TypeError(
            "grid evidence is built only by the certified finite-volume integrator"
        )


@dataclass(frozen=True, slots=True, init=False)
class KerrForwardReturningRadiationConvergenceEvidence:
    """Factory-only five-pass evidence emitted with one canonical kernel build."""

    annulus_edges_over_mass: tuple[float, ...]
    upper_annulus_areas_over_mass_squared: tuple[float, ...]
    lower_annulus_areas_over_mass_squared: tuple[float, ...]
    kernel_descriptor_sha256: str
    policy: KerrReturningRadiationKernelPolicy
    passes: tuple[KerrForwardReturningRadiationGridEvidence, ...]

    def __init__(self) -> None:
        raise TypeError(
            "convergence evidence is built only by the certified "
            "finite-volume integrator"
        )


def _new_sample_weight_witness(
    sample: _ForwardDirectionSample,
    normalized_sample_weight: float,
) -> KerrForwardReturningRadiationSampleWeightWitness:
    result = object.__new__(KerrForwardReturningRadiationSampleWeightWitness)
    for name, value in (
        ("pass_index", sample.pass_index),
        ("pass_name", sample.pass_name),
        ("source_face", sample.source_face),
        ("source_annulus_index", sample.source_annulus_index),
        ("rho_index", sample.rho_index),
        ("mu_index", sample.mu_index),
        ("psi_index", sample.psi_index),
        ("source_radius_over_mass", sample.source_radius_over_mass),
        ("rho_area_over_mass_squared", sample.rho_area_over_mass_squared),
        ("emission_angle_cosine", sample.emission_angle_cosine),
        ("tangent_azimuth_rad", sample.tangent_azimuth_rad),
        (
            "normalized_emitted_flux_direction_weight",
            sample.normalized_emitted_flux_weight,
        ),
        ("normalized_sample_weight", normalized_sample_weight),
    ):
        object.__setattr__(result, name, value)
    return result


def _integrate_grid(
    *,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions | None,
    coarse_surface_options: SurfaceEventOptions | None,
    edges: tuple[float, ...],
    upper_areas: tuple[float, ...],
    lower_areas: tuple[float, ...],
    rho_order: int,
    mu_order: int,
    psi_count: int,
    phase_cells: float,
    pass_index: int,
    pass_name: str,
    direction_transport_provider: _ForwardDirectionTransportProvider | None,
    rho_area_node_cache: dict[
        tuple[str, str, str, int, str],
        tuple[tuple[float, float], ...],
    ],
) -> _KernelGridEstimate:
    count = len(edges) - 1
    blocks = {
        (UPPER, UPPER): _empty_contributions(count),
        (UPPER, LOWER): _empty_contributions(count),
        (LOWER, UPPER): _empty_contributions(count),
        (LOWER, LOWER): _empty_contributions(count),
    }
    areas = {UPPER: upper_areas, LOWER: lower_areas}
    fate_contributions: dict[
        str,
        list[list[list[_FateContribution]]],
    ] = {
        face: [[[] for _fate in _FATES] for _source in range(count)]
        for face in _FACES
    }
    unclassified_fate_contributions: dict[
        str,
        list[list[_FateContribution]],
    ] = {
        face: [[] for _source in range(count)] for face in _FACES
    }
    g2_contributions: dict[str, list[list[float]]] = {
        face: [[] for _source in range(count)] for face in _FACES
    }
    final_fate: dict[str, list[str | None]] = {
        face: [None] * count for face in _FACES
    }
    audit_hash = hashlib.sha256()
    directions = kerrbb_d20_emitted_flux_direction_nodes(
        mu_order,
        psi_count,
        phase_cells=phase_cells,
    )
    if type(directions) is not tuple or not directions:
        raise KerrReturningRadiationKernelError("D20 direction grid is unavailable")
    if len(directions) != mu_order * psi_count:
        raise KerrReturningRadiationKernelError(
            "D20 direction grid count disagrees with mu-order times psi-count"
        )
    expected_directions_per_source = rho_order * mu_order * psi_count
    evaluations = 0
    maximum_normalized_sample_weight = 0.0
    maximum_weight_witness: (
        KerrForwardReturningRadiationSampleWeightWitness | None
    ) = None
    for source_face in _FACES:
        for source_index, (inner, outer) in enumerate(zip(edges, edges[1:])):
            source_area = areas[source_face][source_index]
            cache_key = (
                source_face,
                inner.hex(),
                outer.hex(),
                rho_order,
                source_area.hex(),
            )
            rho_nodes = rho_area_node_cache.get(cache_key)
            if rho_nodes is None:
                rho_nodes = _rho_area_nodes(
                    surface,
                    inner,
                    outer,
                    source_face,
                    rho_order,
                    source_area,
                )
                rho_area_node_cache[cache_key] = rho_nodes
            for rho_index, (rho, rho_area) in enumerate(rho_nodes):
                for direction_index, node in enumerate(directions):
                    if type(node) is not EmittedFluxDirectionNode:
                        raise TypeError(
                            "D20 rule returned a non-exact EmittedFluxDirectionNode"
                        )
                    mu_index, psi_index = divmod(direction_index, psi_count)
                    sample = _ForwardDirectionSample(
                        pass_index,
                        pass_name,
                        source_face,
                        source_index,
                        rho_index,
                        mu_index,
                        psi_index,
                        rho,
                        rho_area,
                        node.emission_angle_cosine,
                        node.tangent_azimuth_rad,
                        node.normalized_emitted_flux_weight,
                    )
                    if direction_transport_provider is None:
                        transport = _trace_direction(
                            surface,
                            termination,
                            ray_options,
                            surface_options,
                            coarse_ray_options,
                            coarse_surface_options,
                            source_face,
                            rho,
                            node.emission_angle_cosine,
                            node.tangent_azimuth_rad,
                        )
                    else:
                        transport = direction_transport_provider(sample)
                    if type(transport) is not _DirectionTransport:
                        raise TypeError("direction transport must have exact internal type")
                    emitted_power_weight = (
                        rho_area * node.normalized_emitted_flux_weight
                    )
                    normalized_weight = emitted_power_weight / source_area
                    if (
                        not math.isfinite(normalized_weight)
                        or normalized_weight
                        <= _FATE_CLOSURE_MINIMUM_DIRECTION_WEIGHT
                    ):
                        raise KerrReturningRadiationKernelError(
                            "emitted finite-volume weight is not above the "
                            "binary64 closure-correction bound"
                        )
                    if normalized_weight > maximum_normalized_sample_weight:
                        maximum_normalized_sample_weight = normalized_weight
                        maximum_weight_witness = _new_sample_weight_witness(
                            sample,
                            normalized_weight,
                        )
                    fate_index = _FATES.index(transport.fate)
                    contribution = _FateContribution(
                        rho_index * len(directions) + direction_index,
                        normalized_weight,
                    )
                    unclassified_fate_contributions[source_face][
                        source_index
                    ].append(contribution)
                    fate_contributions[source_face][source_index][fate_index].append(
                        contribution
                    )
                    final_fate[source_face][source_index] = transport.fate
                    if transport.fate.startswith("return-"):
                        receiver_face = transport.receiver_face
                        assert receiver_face is not None
                        receiver_radius = transport.receiver_radius_over_mass
                        assert receiver_radius is not None
                        receiver_index = _validated_return_receiver_bin(
                            transport,
                            edges,
                        )
                        receiver_area = areas[receiver_face][receiver_index]
                        coefficient = (
                            emitted_power_weight * transport.g2 / receiver_area
                        )
                        if not math.isfinite(coefficient) or coefficient <= 0.0:
                            raise KerrReturningRadiationKernelError(
                                "returned finite-volume coefficient is invalid"
                            )
                        blocks[(receiver_face, source_face)][receiver_index][
                            source_index
                        ].append(coefficient)
                        g2_contributions[source_face][source_index].append(
                            normalized_weight * transport.g2
                        )
                    audit_hash.update(
                        _canonical_json(
                            {
                                "directionWeight": node.normalized_emitted_flux_weight,
                                "fate": transport.fate,
                                "g2": transport.g2,
                                "mu": node.emission_angle_cosine,
                                "primitiveDescriptorSha256": (
                                    transport.primitive_descriptor_sha256
                                ),
                                "psi": node.tangent_azimuth_rad,
                                "receiverFace": transport.receiver_face,
                                "receiverRadiusOverMass": (
                                    transport.receiver_radius_over_mass
                                ),
                                "coarseReceiverFace": (
                                    transport.coarse_receiver_face
                                ),
                                "coarseReceiverRadiusOverMass": (
                                    transport.coarse_receiver_radius_over_mass
                                ),
                                "rhoAreaOverMassSquared": rho_area,
                                "sourceAnnulus": source_index,
                                "sourceFace": source_face,
                                "sourceRadiusOverMass": rho,
                            }
                        ).encode("utf-8")
                    )
                    audit_hash.update(b"\n")
                    evaluations += 1

    frozen_blocks = {
        key: _freeze_matrix(value) for key, value in blocks.items()
    }
    frozen_fates: dict[str, tuple[FiniteVolumeFateFractions, ...]] = {}
    frozen_g2: dict[str, tuple[float, ...]] = {}
    closure_residuals: dict[str, tuple[float, ...]] = {}
    for source_face in _FACES:
        face_fates: list[FiniteVolumeFateFractions] = []
        face_g2: list[float] = []
        face_residuals: list[float] = []
        for source_index in range(count):
            last = final_fate[source_face][source_index]
            if last is None:
                raise KerrReturningRadiationKernelError(
                    "source annulus produced no directions"
                )
            closed_fate_values = _audited_fate_fraction_partition(
                unclassified_fate_contributions[source_face][source_index],
                fate_contributions[source_face][source_index],
                expected_direction_count=expected_directions_per_source,
            )
            face_fates.append(FiniteVolumeFateFractions(*closed_fate_values))
            direct_g2 = math.fsum(g2_contributions[source_face][source_index])
            if not math.isfinite(direct_g2) or direct_g2 < 0.0:
                raise KerrReturningRadiationKernelError(
                    "g2 returned-power column is invalid"
                )
            source_area = areas[source_face][source_index]
            area_reconstructed = math.fsum(
                areas[receiver_face][receiver_index]
                * frozen_blocks[(receiver_face, source_face)][receiver_index][
                    source_index
                ]
                / source_area
                for receiver_face in _FACES
                for receiver_index in range(count)
            )
            closure = abs(area_reconstructed - direct_g2)
            scale = max(1.0, abs(area_reconstructed), abs(direct_g2))
            if closure > 4096.0 * math.ulp(scale):
                raise KerrReturningRadiationKernelError(
                    "area-weighted K column does not close against direct g2 power"
                )
            face_g2.append(direct_g2)
            face_residuals.append(closure)
        frozen_fates[source_face] = tuple(face_fates)
        frozen_g2[source_face] = tuple(face_g2)
        closure_residuals[source_face] = tuple(face_residuals)

    if maximum_weight_witness is None:
        raise KerrReturningRadiationKernelError(
            "quadrature pass produced no sample-weight witness"
        )
    return _KernelGridEstimate(
        frozen_blocks[(UPPER, UPPER)],
        frozen_blocks[(UPPER, LOWER)],
        frozen_blocks[(LOWER, UPPER)],
        frozen_blocks[(LOWER, LOWER)],
        frozen_fates[UPPER],
        frozen_fates[LOWER],
        frozen_g2[UPPER],
        frozen_g2[LOWER],
        closure_residuals[UPPER],
        closure_residuals[LOWER],
        evaluations,
        audit_hash.hexdigest(),
        maximum_normalized_sample_weight,
        maximum_weight_witness,
    )


def _vector_difference(
    full: tuple[float, ...],
    comparison: tuple[float, ...],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[float, float]:
    if len(full) != len(comparison) or not full:
        raise KerrReturningRadiationKernelError(
            "convergence vectors have incompatible shapes"
        )
    absolute_differences: list[float] = []
    scaled_differences: list[float] = []
    for full_value, comparison_value in zip(full, comparison):
        difference = abs(full_value - comparison_value)
        threshold = max(
            absolute_tolerance,
            relative_tolerance * max(abs(full_value), abs(comparison_value)),
        )
        if not math.isfinite(difference) or not math.isfinite(threshold):
            raise KerrReturningRadiationKernelError(
                "grid convergence diagnostic is not finite"
            )
        absolute_differences.append(difference)
        scaled_differences.append(difference / threshold)
    return max(absolute_differences), max(scaled_differences)


def _grid_difference(
    full: _KernelGridEstimate,
    comparison: _KernelGridEstimate,
    policy: KerrReturningRadiationKernelPolicy,
) -> KerrReturningRadiationGridDifference:
    matrix_abs, matrix_scaled = _vector_difference(
        full.matrix_vector(),
        comparison.matrix_vector(),
        policy.absolute_tolerance,
        policy.relative_tolerance,
    )
    g2_abs, g2_scaled = _vector_difference(
        full.g2_vector(),
        comparison.g2_vector(),
        policy.absolute_tolerance,
        policy.relative_tolerance,
    )
    fate_abs, fate_scaled = _vector_difference(
        full.fate_vector(),
        comparison.fate_vector(),
        policy.absolute_tolerance,
        policy.relative_tolerance,
    )
    return KerrReturningRadiationGridDifference(
        matrix_abs,
        matrix_scaled,
        g2_abs,
        g2_scaled,
        fate_abs,
        fate_scaled,
        max(matrix_scaled, g2_scaled, fate_scaled) <= 1.0,
    )


def _new_grid_evidence(
    estimate: _KernelGridEstimate,
    *,
    pass_index: int,
    pass_name: str,
    rho_order: int,
    mu_order: int,
    psi_count: int,
    phase_cells: float,
) -> KerrForwardReturningRadiationGridEvidence:
    result = object.__new__(KerrForwardReturningRadiationGridEvidence)
    for name, value in (
        ("pass_index", pass_index),
        ("pass_name", pass_name),
        ("rho_order", rho_order),
        ("mu_order", mu_order),
        ("psi_count", psi_count),
        ("phase_cells", phase_cells),
        ("upper_receiver_upper_emitter_coefficients", estimate.uu),
        ("upper_receiver_lower_emitter_coefficients", estimate.ul),
        ("lower_receiver_upper_emitter_coefficients", estimate.lu),
        ("lower_receiver_lower_emitter_coefficients", estimate.ll),
        ("upper_emitter_fate_fractions", estimate.upper_fates),
        ("lower_emitter_fate_fractions", estimate.lower_fates),
        ("upper_emitter_g2_returned_power_columns", estimate.upper_g2_columns),
        ("lower_emitter_g2_returned_power_columns", estimate.lower_g2_columns),
        (
            "upper_emitter_g2_column_closure_residuals",
            estimate.upper_column_closure_residuals,
        ),
        (
            "lower_emitter_g2_column_closure_residuals",
            estimate.lower_column_closure_residuals,
        ),
        ("direction_evaluations", estimate.direction_evaluations),
        ("sample_audit_sha256", estimate.sample_audit_sha256),
        (
            "maximum_normalized_sample_weight",
            estimate.maximum_normalized_sample_weight,
        ),
        (
            "maximum_normalized_sample_weight_witness",
            estimate.maximum_normalized_sample_weight_witness,
        ),
    ):
        object.__setattr__(result, name, value)
    return result


@dataclass(frozen=True, slots=True, init=False)
class KerrForwardReturningRadiationKernel:
    """Authenticated four-face finite-volume returning energy operator."""

    annulus_edges_over_mass: tuple[float, ...]
    annulus_representative_radii_over_mass: tuple[float, ...]
    upper_annulus_areas_over_mass_squared: tuple[float, ...]
    lower_annulus_areas_over_mass_squared: tuple[float, ...]
    upper_receiver_upper_emitter_coefficients: tuple[tuple[float, ...], ...]
    upper_receiver_lower_emitter_coefficients: tuple[tuple[float, ...], ...]
    lower_receiver_upper_emitter_coefficients: tuple[tuple[float, ...], ...]
    lower_receiver_lower_emitter_coefficients: tuple[tuple[float, ...], ...]
    upper_emitter_fate_fractions: tuple[FiniteVolumeFateFractions, ...]
    lower_emitter_fate_fractions: tuple[FiniteVolumeFateFractions, ...]
    upper_emitter_g2_returned_power_columns: tuple[float, ...]
    lower_emitter_g2_returned_power_columns: tuple[float, ...]
    upper_emitter_g2_column_closure_residuals: tuple[float, ...]
    lower_emitter_g2_column_closure_residuals: tuple[float, ...]
    convergence: KerrReturningRadiationKernelConvergence
    fine_coarse_receiver_bin_topology_verified: bool
    direction_evaluations_consumed: int
    whole_ray_traces_consumed: int
    full_grid_sample_audit_sha256: str
    half_rho_sample_audit_sha256: str
    half_mu_sample_audit_sha256: str
    half_psi_sample_audit_sha256: str
    phase_shifted_sample_audit_sha256: str
    upper_annulus_area_descriptor_sha256: tuple[str, ...]
    lower_annulus_area_descriptor_sha256: tuple[str, ...]
    surface: KerrFiniteThicknessMultiSurface
    termination: KerrOblateTermination
    ray_options: RayTraceOptions
    surface_options: SurfaceEventOptions
    coarse_ray_options: RayTraceOptions | None
    coarse_surface_options: SurfaceEventOptions | None
    policy: KerrReturningRadiationKernelPolicy
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy
    _descriptor_json: str
    _descriptor_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "KerrForwardReturningRadiationKernel is built only by the "
            "certified finite-volume integrator"
        )

    @property
    def annulus_count(self) -> int:
        return len(self.annulus_edges_over_mass) - 1

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        if type(self._descriptor_json) is not str:
            raise KerrReturningRadiationKernelVerificationError(
                "kernel descriptor has a non-exact type"
            )
        return json.loads(self._descriptor_json)

    def revalidate(self) -> None:
        verify_kerr_returning_radiation_energy_kernel(self)

    def to_axisymmetric_energy_kernel(
        self,
        require_equatorial_symmetry: bool = True,
    ) -> AxisymmetricReturningRadiationKernel:
        """Replay, enforce face symmetry, then reduce the four blocks."""

        return verify_and_reduce_kerr_returning_radiation_energy_kernel(
            self,
            require_equatorial_symmetry=require_equatorial_symmetry,
        )

    def coarsen_annuli(
        self,
        merged_annulus_edges_over_mass: tuple[float, ...],
    ) -> "KerrForwardReturningRadiationKernelProjection":
        """Merge adjacent annuli with exact finite-volume area weighting.

        Receiver rows are proper-area averaged and emitter columns are summed,
        which is the unique projection preserving the local-flux action when
        the outgoing flux is constant inside each merged source annulus.
        """

        verify_kerr_returning_radiation_energy_kernel(self)
        return _coarsen_verified_kernel(self, merged_annulus_edges_over_mass)


def _require_equatorial_symmetry(
    result: KerrForwardReturningRadiationKernel
    | KerrForwardReturningRadiationKernelProjection,
) -> None:
    policy = result.policy
    for upper, lower in zip(
        result.upper_annulus_areas_over_mass_squared,
        result.lower_annulus_areas_over_mass_squared,
    ):
        difference = abs(upper - lower)
        threshold = max(
            policy.symmetry_absolute_tolerance,
            policy.symmetry_relative_tolerance * max(abs(upper), abs(lower)),
        )
        if difference > threshold:
            raise KerrReturningRadiationKernelConvergenceError(
                "upper/lower annulus proper areas fail equatorial symmetry"
            )
    for left, right, label in (
        (
            result.upper_receiver_upper_emitter_coefficients,
            result.lower_receiver_lower_emitter_coefficients,
            "UU versus LL",
        ),
        (
            result.upper_receiver_lower_emitter_coefficients,
            result.lower_receiver_upper_emitter_coefficients,
            "UL versus LU",
        ),
    ):
        for left_row, right_row in zip(left, right):
            for left_value, right_value in zip(left_row, right_row):
                difference = abs(left_value - right_value)
                threshold = max(
                    policy.symmetry_absolute_tolerance,
                    policy.symmetry_relative_tolerance
                    * max(abs(left_value), abs(right_value)),
                )
                if difference > threshold:
                    raise KerrReturningRadiationKernelConvergenceError(
                        f"{label} blocks fail equatorial symmetry"
                    )


@dataclass(frozen=True, slots=True, init=False)
class KerrForwardReturningRadiationKernelProjection:
    """Self-verifying adjacent-annulus projection of a full replayed kernel."""

    annulus_edges_over_mass: tuple[float, ...]
    annulus_representative_radii_over_mass: tuple[float, ...]
    upper_annulus_areas_over_mass_squared: tuple[float, ...]
    lower_annulus_areas_over_mass_squared: tuple[float, ...]
    upper_receiver_upper_emitter_coefficients: tuple[tuple[float, ...], ...]
    upper_receiver_lower_emitter_coefficients: tuple[tuple[float, ...], ...]
    lower_receiver_upper_emitter_coefficients: tuple[tuple[float, ...], ...]
    lower_receiver_lower_emitter_coefficients: tuple[tuple[float, ...], ...]
    upper_emitter_fate_fractions: tuple[FiniteVolumeFateFractions, ...]
    lower_emitter_fate_fractions: tuple[FiniteVolumeFateFractions, ...]
    upper_emitter_g2_returned_power_columns: tuple[float, ...]
    lower_emitter_g2_returned_power_columns: tuple[float, ...]
    upper_emitter_g2_column_closure_residuals: tuple[float, ...]
    lower_emitter_g2_column_closure_residuals: tuple[float, ...]
    policy: KerrReturningRadiationKernelPolicy
    source_kernel_descriptor_sha256: str
    _source: KerrForwardReturningRadiationKernel
    _descriptor_json: str
    _descriptor_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "KerrForwardReturningRadiationKernelProjection is built only by "
            "coarsen_annuli"
        )

    @property
    def annulus_count(self) -> int:
        return len(self.annulus_edges_over_mass) - 1

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        if type(self._descriptor_json) is not str:
            raise KerrReturningRadiationKernelVerificationError(
                "projection descriptor has a non-exact type"
            )
        return json.loads(self._descriptor_json)

    def revalidate(self) -> None:
        verify_kerr_returning_radiation_kernel_projection(self)

    def to_axisymmetric_energy_kernel(
        self,
        require_equatorial_symmetry: bool = True,
    ) -> AxisymmetricReturningRadiationKernel:
        return verify_and_reduce_kerr_returning_radiation_kernel_projection(
            self,
            require_equatorial_symmetry=require_equatorial_symmetry,
        )


def _reduce_verified_four_face_kernel(
    result: KerrForwardReturningRadiationKernel
    | KerrForwardReturningRadiationKernelProjection,
    *,
    producer_id: str,
) -> AxisymmetricReturningRadiationKernel:
    """Apply symmetry and reduce a result already replayed by its verifier."""

    _require_equatorial_symmetry(result)
    count = result.annulus_count
    reduced = tuple(
        tuple(
            0.5
            * math.fsum(
                (
                    result.upper_receiver_upper_emitter_coefficients[i][j],
                    result.upper_receiver_lower_emitter_coefficients[i][j],
                    result.lower_receiver_upper_emitter_coefficients[i][j],
                    result.lower_receiver_lower_emitter_coefficients[i][j],
                )
            )
            for j in range(count)
        )
        for i in range(count)
    )
    return AxisymmetricReturningRadiationKernel(
        annulus_radii_over_mass=result.annulus_representative_radii_over_mass,
        receiver_emitter_coefficients=reduced,
        ray_kernel_producer_id=producer_id,
    )


def verify_and_reduce_kerr_returning_radiation_energy_kernel(
    result: KerrForwardReturningRadiationKernel,
    *,
    require_equatorial_symmetry: bool = True,
) -> AxisymmetricReturningRadiationKernel:
    """Replay one full kernel exactly once, then symmetry-reduce it."""

    if type(require_equatorial_symmetry) is not bool:
        raise TypeError("require_equatorial_symmetry must be an exact bool")
    if require_equatorial_symmetry is not True:
        raise ValueError(
            "four-face reduction is prohibited without equatorial symmetry"
        )
    verify_kerr_returning_radiation_energy_kernel(result)
    return _reduce_verified_four_face_kernel(
        result,
        producer_id=f"{IMPLEMENTATION_ID}:{result.model_descriptor_sha256}",
    )


def verify_and_reduce_kerr_returning_radiation_kernel_projection(
    result: KerrForwardReturningRadiationKernelProjection,
    *,
    require_equatorial_symmetry: bool = True,
) -> AxisymmetricReturningRadiationKernel:
    """Replay one authenticated projection exactly once, then reduce it."""

    if type(require_equatorial_symmetry) is not bool:
        raise TypeError("require_equatorial_symmetry must be an exact bool")
    if require_equatorial_symmetry is not True:
        raise ValueError(
            "four-face reduction is prohibited without equatorial symmetry"
        )
    verify_kerr_returning_radiation_kernel_projection(result)
    return _reduce_verified_four_face_kernel(
        result,
        producer_id=(
            f"{IMPLEMENTATION_ID}:coarsened:{result.model_descriptor_sha256}"
        ),
    )


def _edge_groups(
    fine_edges: tuple[float, ...],
    merged_edges: tuple[float, ...],
) -> tuple[tuple[int, ...], ...]:
    if type(merged_edges) is not tuple or len(merged_edges) < 2:
        raise TypeError("merged annulus edges must be an exact tuple")
    checked = tuple(
        _exact_finite_float(value, f"merged annulus edge {index}")
        for index, value in enumerate(merged_edges)
    )
    if any(right <= left for left, right in zip(checked, checked[1:])):
        raise ValueError("merged annulus edges must be strictly increasing")
    if (
        checked[0].hex() != fine_edges[0].hex()
        or checked[-1].hex() != fine_edges[-1].hex()
    ):
        raise ValueError("merged annuli must preserve the complete radial domain")
    fine_hex = {edge.hex(): index for index, edge in enumerate(fine_edges)}
    try:
        boundary_indices = tuple(fine_hex[edge.hex()] for edge in checked)
    except KeyError as error:
        raise ValueError("merged edges must be an exact subset of fine-grid edges") from error
    groups = tuple(
        tuple(range(left, right))
        for left, right in zip(boundary_indices, boundary_indices[1:])
    )
    if any(not group for group in groups):
        raise ValueError("each merged annulus must contain a fine annulus")
    return groups


def _merge_block(
    block: tuple[tuple[float, ...], ...],
    receiver_areas: tuple[float, ...],
    groups: tuple[tuple[int, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    merged_receiver_areas = tuple(
        math.fsum(receiver_areas[index] for index in group) for group in groups
    )
    return tuple(
        tuple(
            math.fsum(
                receiver_areas[receiver_index]
                * block[receiver_index][emitter_index]
                for receiver_index in receiver_group
                for emitter_index in emitter_group
            )
            / merged_receiver_areas[coarse_receiver]
            for emitter_group in groups
        )
        for coarse_receiver, receiver_group in enumerate(groups)
    )


def _merge_fates(
    fates: tuple[FiniteVolumeFateFractions, ...],
    areas: tuple[float, ...],
    groups: tuple[tuple[int, ...], ...],
) -> tuple[FiniteVolumeFateFractions, ...]:
    if type(fates) is not tuple or not fates:
        raise TypeError("merged fates must be a non-empty exact tuple")
    if type(areas) is not tuple or len(areas) != len(fates):
        raise TypeError("merged fate areas must be a matching exact tuple")
    if any(type(fate) is not FiniteVolumeFateFractions for fate in fates):
        raise TypeError("merged fates must use the exact fate-fraction type")
    for area in areas:
        if type(area) is not float:
            raise TypeError("merged fate area must be an exact float")
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("merged fate areas must be finite and positive")
    if (
        type(groups) is not tuple
        or not groups
        or any(type(group) is not tuple or not group for group in groups)
    ):
        raise TypeError("merged fate groups must be non-empty exact tuples")
    flattened_indices: list[int] = []
    for group in groups:
        for index in group:
            if type(index) is not int:
                raise TypeError("merged fate group index must be an exact int")
            flattened_indices.append(index)
    if tuple(flattened_indices) != tuple(range(len(fates))):
        raise ValueError(
            "merged fate groups must exactly cover each input annulus in order"
        )

    merged: list[FiniteVolumeFateFractions] = []
    for group in groups:
        area = math.fsum(areas[index] for index in group)
        flattened_weighted_contributions = tuple(
            areas[index] * fates[index].as_tuple()[fate_index]
            for index in group
            for fate_index in range(len(_FATES))
        )
        independently_accumulated_total = (
            math.fsum(flattened_weighted_contributions) / area
        )
        raw_values = tuple(
            math.fsum(
                areas[index] * fates[index].as_tuple()[fate_index]
                for index in group
            )
            / area
            for fate_index in range(len(_FATES))
        )
        closed_values = _close_fate_fraction_binary64_roundoff(
            raw_values,
            independently_accumulated_total=independently_accumulated_total,
        )
        merged.append(FiniteVolumeFateFractions(*closed_values))
    return tuple(merged)


def _merge_columns(
    columns: tuple[float, ...],
    areas: tuple[float, ...],
    groups: tuple[tuple[int, ...], ...],
) -> tuple[float, ...]:
    return tuple(
        math.fsum(areas[index] * columns[index] for index in group)
        / math.fsum(areas[index] for index in group)
        for group in groups
    )


def _projection_closure(
    *,
    source_face: str,
    upper_areas: tuple[float, ...],
    lower_areas: tuple[float, ...],
    uu: tuple[tuple[float, ...], ...],
    ul: tuple[tuple[float, ...], ...],
    lu: tuple[tuple[float, ...], ...],
    ll: tuple[tuple[float, ...], ...],
    g2_columns: tuple[float, ...],
) -> tuple[float, ...]:
    source_areas = upper_areas if source_face == UPPER else lower_areas
    upper_block = uu if source_face == UPPER else ul
    lower_block = lu if source_face == UPPER else ll
    residuals: list[float] = []
    for source_index, direct in enumerate(g2_columns):
        reconstructed = math.fsum(
            (
                *(
                    upper_areas[i] * upper_block[i][source_index]
                    / source_areas[source_index]
                    for i in range(len(source_areas))
                ),
                *(
                    lower_areas[i] * lower_block[i][source_index]
                    / source_areas[source_index]
                    for i in range(len(source_areas))
                ),
            )
        )
        residual = abs(reconstructed - direct)
        if residual > 8192.0 * math.ulp(max(1.0, reconstructed, direct)):
            raise KerrReturningRadiationKernelError(
                "coarsened K column violates g2 area-energy closure"
            )
        residuals.append(residual)
    return tuple(residuals)


def _coarsen_verified_kernel(
    source: KerrForwardReturningRadiationKernel,
    merged_edges: tuple[float, ...],
) -> KerrForwardReturningRadiationKernelProjection:
    groups = _edge_groups(source.annulus_edges_over_mass, merged_edges)
    edges = tuple(merged_edges)
    upper_areas = tuple(
        math.fsum(source.upper_annulus_areas_over_mass_squared[index] for index in group)
        for group in groups
    )
    lower_areas = tuple(
        math.fsum(source.lower_annulus_areas_over_mass_squared[index] for index in group)
        for group in groups
    )
    uu = _merge_block(
        source.upper_receiver_upper_emitter_coefficients,
        source.upper_annulus_areas_over_mass_squared,
        groups,
    )
    ul = _merge_block(
        source.upper_receiver_lower_emitter_coefficients,
        source.upper_annulus_areas_over_mass_squared,
        groups,
    )
    lu = _merge_block(
        source.lower_receiver_upper_emitter_coefficients,
        source.lower_annulus_areas_over_mass_squared,
        groups,
    )
    ll = _merge_block(
        source.lower_receiver_lower_emitter_coefficients,
        source.lower_annulus_areas_over_mass_squared,
        groups,
    )
    upper_fates = _merge_fates(
        source.upper_emitter_fate_fractions,
        source.upper_annulus_areas_over_mass_squared,
        groups,
    )
    lower_fates = _merge_fates(
        source.lower_emitter_fate_fractions,
        source.lower_annulus_areas_over_mass_squared,
        groups,
    )
    upper_g2 = _merge_columns(
        source.upper_emitter_g2_returned_power_columns,
        source.upper_annulus_areas_over_mass_squared,
        groups,
    )
    lower_g2 = _merge_columns(
        source.lower_emitter_g2_returned_power_columns,
        source.lower_annulus_areas_over_mass_squared,
        groups,
    )
    upper_closure = _projection_closure(
        source_face=UPPER,
        upper_areas=upper_areas,
        lower_areas=lower_areas,
        uu=uu,
        ul=ul,
        lu=lu,
        ll=ll,
        g2_columns=upper_g2,
    )
    lower_closure = _projection_closure(
        source_face=LOWER,
        upper_areas=upper_areas,
        lower_areas=lower_areas,
        uu=uu,
        ul=ul,
        lu=lu,
        ll=ll,
        g2_columns=lower_g2,
    )
    representative = tuple(
        0.5 * math.fsum((inner, outer))
        for inner, outer in zip(edges, edges[1:])
    )
    descriptor = {
        "annulusEdgesOverMass": edges,
        "coefficientIndexOrder": "K[receiverAnnulus][emitterAnnulus]",
        "implementationId": f"{IMPLEMENTATION_ID}/adjacent-annulus-projection/v1",
        "projection": (
            "receiver proper-area average and emitter-column sum for "
            "piecewise-constant outgoing coarse-annulus flux"
        ),
        "result": {
            "fates": {
                "lower": tuple(asdict(item) for item in lower_fates),
                "upper": tuple(asdict(item) for item in upper_fates),
            },
            "g2Columns": {"lower": lower_g2, "upper": upper_g2},
            "lowerAreas": lower_areas,
            "matrices": {"LL": ll, "LU": lu, "UL": ul, "UU": uu},
            "upperAreas": upper_areas,
        },
        "sourceKernelDescriptorSha256": source.model_descriptor_sha256,
    }
    descriptor_json = _canonical_json(descriptor)
    result = object.__new__(KerrForwardReturningRadiationKernelProjection)
    for name, value in (
        ("annulus_edges_over_mass", edges),
        ("annulus_representative_radii_over_mass", representative),
        ("upper_annulus_areas_over_mass_squared", upper_areas),
        ("lower_annulus_areas_over_mass_squared", lower_areas),
        ("upper_receiver_upper_emitter_coefficients", uu),
        ("upper_receiver_lower_emitter_coefficients", ul),
        ("lower_receiver_upper_emitter_coefficients", lu),
        ("lower_receiver_lower_emitter_coefficients", ll),
        ("upper_emitter_fate_fractions", upper_fates),
        ("lower_emitter_fate_fractions", lower_fates),
        ("upper_emitter_g2_returned_power_columns", upper_g2),
        ("lower_emitter_g2_returned_power_columns", lower_g2),
        ("upper_emitter_g2_column_closure_residuals", upper_closure),
        ("lower_emitter_g2_column_closure_residuals", lower_closure),
        ("policy", source.policy),
        ("source_kernel_descriptor_sha256", source.model_descriptor_sha256),
        ("_source", source),
        ("_descriptor_json", descriptor_json),
        (
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        ),
    ):
        object.__setattr__(result, name, value)
    return result


def verify_kerr_returning_radiation_kernel_projection(
    result: KerrForwardReturningRadiationKernelProjection,
) -> None:
    if type(result) is not KerrForwardReturningRadiationKernelProjection:
        raise TypeError(
            "result must be exact KerrForwardReturningRadiationKernelProjection"
        )
    descriptor_json = object.__getattribute__(result, "_descriptor_json")
    descriptor_sha = object.__getattribute__(result, "_descriptor_sha256")
    if type(descriptor_json) is not str or type(descriptor_sha) is not str:
        raise KerrReturningRadiationKernelVerificationError(
            "projection descriptor identity has a non-exact type"
        )
    try:
        parsed = json.loads(descriptor_json)
    except (json.JSONDecodeError, TypeError) as error:
        raise KerrReturningRadiationKernelVerificationError(
            "projection descriptor is malformed"
        ) from error
    if _canonical_json(parsed).encode("utf-8") != descriptor_json.encode("utf-8"):
        raise KerrReturningRadiationKernelVerificationError(
            "projection descriptor is not canonical"
        )
    if (
        hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest().encode("ascii")
        != descriptor_sha.encode("ascii")
    ):
        raise KerrReturningRadiationKernelVerificationError(
            "projection descriptor SHA-256 is inconsistent"
        )
    source = object.__getattribute__(result, "_source")
    verify_kerr_returning_radiation_energy_kernel(source)
    rebuilt = _coarsen_verified_kernel(source, result.annulus_edges_over_mass)
    _require_trusted_exact_tree(result, rebuilt, "projection")


def _integrate_kerr_returning_radiation_energy_kernel_with_transport_provider_and_evidence(
    surface: KerrFiniteThicknessMultiSurface,
    *,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
    policy: KerrReturningRadiationKernelPolicy | None = None,
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy | None = None,
    direction_transport_provider: _ForwardDirectionTransportProvider | None = None,
) -> tuple[
    KerrForwardReturningRadiationKernel,
    KerrForwardReturningRadiationConvergenceEvidence,
]:
    """Integrate the four forward ``g^2`` finite-volume face blocks."""

    surface = _validated_surface(surface)
    (
        termination,
        ray_options,
        surface_options,
        coarse_ray_options,
        coarse_surface_options,
    ) = _validated_auxiliary_inputs(
        termination,
        ray_options,
        surface_options,
        coarse_ray_options,
        coarse_surface_options,
    )
    selected_policy = _validated_policy(policy)
    selected_area_policy = _validated_area_policy(area_policy)
    if direction_transport_provider is not None and not callable(
        direction_transport_provider
    ):
        raise TypeError("direction_transport_provider must be callable or None")
    edges = _validated_edges(annulus_edges_over_mass, surface)
    count = len(edges) - 1
    expected_directions = (
        7
        * count
        * selected_policy.rho_order
        * selected_policy.mu_order
        * selected_policy.psi_count
    )
    expected_whole_rays = 2 * expected_directions
    if expected_directions > selected_policy.maximum_direction_evaluations:
        raise ValueError(
            f"kernel requires {expected_directions} direction evaluations but "
            f"budget is {selected_policy.maximum_direction_evaluations}"
        )
    if expected_whole_rays > selected_policy.maximum_whole_ray_traces:
        raise ValueError(
            f"kernel requires {expected_whole_rays} whole rays but budget is "
            f"{selected_policy.maximum_whole_ray_traces}"
        )
    (
        upper_areas,
        lower_areas,
        upper_area_hashes,
        lower_area_hashes,
    ) = _annulus_areas(surface, edges, selected_area_policy)

    common = {
        "surface": surface,
        "termination": termination,
        "ray_options": ray_options,
        "surface_options": surface_options,
        "coarse_ray_options": coarse_ray_options,
        "coarse_surface_options": coarse_surface_options,
        "edges": edges,
        "upper_areas": upper_areas,
        "lower_areas": lower_areas,
        "rho_area_node_cache": {},
        "direction_transport_provider": direction_transport_provider,
    }
    full = _integrate_grid(
        **common,
        rho_order=selected_policy.rho_order,
        mu_order=selected_policy.mu_order,
        psi_count=selected_policy.psi_count,
        phase_cells=0.0,
        pass_index=0,
        pass_name="full",
    )
    half_rho = _integrate_grid(
        **common,
        rho_order=selected_policy.rho_order // 2,
        mu_order=selected_policy.mu_order,
        psi_count=selected_policy.psi_count,
        phase_cells=0.0,
        pass_index=1,
        pass_name="half-rho",
    )
    half_mu = _integrate_grid(
        **common,
        rho_order=selected_policy.rho_order,
        mu_order=selected_policy.mu_order // 2,
        psi_count=selected_policy.psi_count,
        phase_cells=0.0,
        pass_index=2,
        pass_name="half-mu",
    )
    half_psi = _integrate_grid(
        **common,
        rho_order=selected_policy.rho_order,
        mu_order=selected_policy.mu_order,
        psi_count=selected_policy.psi_count // 2,
        phase_cells=0.0,
        pass_index=3,
        pass_name="half-psi",
    )
    phase_shifted = _integrate_grid(
        **common,
        rho_order=selected_policy.rho_order,
        mu_order=selected_policy.mu_order,
        psi_count=selected_policy.psi_count,
        phase_cells=0.5,
        pass_index=4,
        pass_name="phase-shifted",
    )
    actual_directions = math.fsum(
        estimate.direction_evaluations
        for estimate in (full, half_rho, half_mu, half_psi, phase_shifted)
    )
    if type(actual_directions) is not float or not actual_directions.is_integer():
        raise KerrReturningRadiationKernelError("direction work accounting is invalid")
    actual_direction_count = int(actual_directions)
    if actual_direction_count != expected_directions:
        raise KerrReturningRadiationKernelError(
            "direction work accounting disagrees with the declared grid"
        )
    convergence = KerrReturningRadiationKernelConvergence(
        _grid_difference(full, half_rho, selected_policy),
        _grid_difference(full, half_mu, selected_policy),
        _grid_difference(full, half_psi, selected_policy),
        _grid_difference(full, phase_shifted, selected_policy),
        all(
            _grid_difference(full, comparison, selected_policy).converged
            for comparison in (half_rho, half_mu, half_psi, phase_shifted)
        ),
    )
    if not convergence.converged:
        raise KerrReturningRadiationKernelConvergenceError(
            "finite-volume K failed full/half-rho, full/half-mu, "
            "full/half-psi, or periodic-phase convergence"
        )

    representative = tuple(
        0.5 * math.fsum((inner, outer))
        for inner, outer in zip(edges, edges[1:])
    )
    descriptor = {
        "annuli": {
            "edgesOverMass": edges,
            "representativeRadiusPolicy": "arithmetic edge midpoint",
            "representativeRadiiOverMass": representative,
        },
        "area": {
            "measure": "actual-face comoving proper area",
            "quadraturePolicy": dict(selected_area_policy.descriptor()),
            "upperAnnulusAreaDescriptorSha256": upper_area_hashes,
            "upperAnnulusAreasOverMassSquared": upper_areas,
            "lowerAnnulusAreaDescriptorSha256": lower_area_hashes,
            "lowerAnnulusAreasOverMassSquared": lower_areas,
        },
        "capabilities": dict(SCIENTIFIC_STATUS),
        "coefficient": {
            "equation": "Delta K=(Delta A_e/A_i) w_emitted g^2",
            "forbiddenSingleRayFactors": (
                "g^4 and receiver-incidence-weighted g^4 are not consumed"
            ),
            "fineCoarseReceiverTopologyGate": (
                "every returned direction must retain the same receiver face "
                "and land in the same user-declared annulus for independent "
                "fine and coarse whole rays"
            ),
            "matrixIndexOrder": "K[receiverAnnulus][emitterAnnulus]",
            "surfaceBlockOrder": {
                "LL": "lower receiver, lower emitter",
                "LU": "lower receiver, upper emitter",
                "UL": "upper receiver, lower emitter",
                "UU": "upper receiver, upper emitter",
            },
            "unconditionalFaceMultiplicity": 1,
        },
        "convergence": {
            "actual": asdict(convergence),
            "absoluteTolerance": selected_policy.absolute_tolerance,
            "eachMatrixCoefficientAndG2ColumnGated": True,
            "fateDiagnosticsAlsoGated": True,
            "fineCoarseReceiverBinTopologyVerified": True,
            "relativeTolerance": selected_policy.relative_tolerance,
            "rigorousErrorBound": False,
        },
        "implementationId": IMPLEMENTATION_ID,
        "modelOwnership": {
            "calibration": asdict(surface.calibration),
            "metric": asdict(surface.metric),
            "termination": asdict(termination),
        },
        "quadrature": {
            "angularRule": "public kerrbb_d20_emitted_flux_direction_nodes",
            "fullMuOrder": selected_policy.mu_order,
            "fullPsiCount": selected_policy.psi_count,
            "fullRhoOrderPerSourceCell": selected_policy.rho_order,
            "periodicPhaseShiftCells": 0.5,
            "rhoMeasure": "2 pi sqrt(det q) d(rho/M)",
        },
        "result": {
            "estimateKind": "finite-grid-point-estimate",
            "fates": {
                "lowerEmitters": tuple(asdict(item) for item in full.lower_fates),
                "upperEmitters": tuple(asdict(item) for item in full.upper_fates),
            },
            "g2ReturnedPowerColumns": {
                "lowerEmitters": full.lower_g2_columns,
                "upperEmitters": full.upper_g2_columns,
            },
            "g2ColumnClosureResiduals": {
                "lowerEmitters": full.lower_column_closure_residuals,
                "upperEmitters": full.upper_column_closure_residuals,
            },
            "matrices": {
                "LL": full.ll,
                "LU": full.lu,
                "UL": full.ul,
                "UU": full.uu,
            },
        },
        "sampleAuditSha256": {
            "full": full.sample_audit_sha256,
            "halfMu": half_mu.sample_audit_sha256,
            "halfPsi": half_psi.sample_audit_sha256,
            "halfRho": half_rho.sample_audit_sha256,
            "phaseShifted": phase_shifted.sample_audit_sha256,
        },
        "workBudget": {
            "directionEvaluationsConsumed": actual_direction_count,
            "maximumDirectionEvaluations": (
                selected_policy.maximum_direction_evaluations
            ),
            "maximumWholeRayTraces": selected_policy.maximum_whole_ray_traces,
            "primitiveWholeRaysPerDirection": 2,
            "productionIssuedPrimitiveReplayWholeRaysPerDirection": 0,
            "publicPrimitiveReplayWholeRaysPerDirection": 2,
            "wholeRayTracesConsumed": 2 * actual_direction_count,
        },
    }
    descriptor_json = _canonical_json(descriptor)
    result = object.__new__(KerrForwardReturningRadiationKernel)
    for name, value in (
        ("annulus_edges_over_mass", edges),
        ("annulus_representative_radii_over_mass", representative),
        ("upper_annulus_areas_over_mass_squared", upper_areas),
        ("lower_annulus_areas_over_mass_squared", lower_areas),
        ("upper_receiver_upper_emitter_coefficients", full.uu),
        ("upper_receiver_lower_emitter_coefficients", full.ul),
        ("lower_receiver_upper_emitter_coefficients", full.lu),
        ("lower_receiver_lower_emitter_coefficients", full.ll),
        ("upper_emitter_fate_fractions", full.upper_fates),
        ("lower_emitter_fate_fractions", full.lower_fates),
        ("upper_emitter_g2_returned_power_columns", full.upper_g2_columns),
        ("lower_emitter_g2_returned_power_columns", full.lower_g2_columns),
        (
            "upper_emitter_g2_column_closure_residuals",
            full.upper_column_closure_residuals,
        ),
        (
            "lower_emitter_g2_column_closure_residuals",
            full.lower_column_closure_residuals,
        ),
        ("convergence", convergence),
        ("fine_coarse_receiver_bin_topology_verified", True),
        ("direction_evaluations_consumed", actual_direction_count),
        ("whole_ray_traces_consumed", 2 * actual_direction_count),
        ("full_grid_sample_audit_sha256", full.sample_audit_sha256),
        ("half_rho_sample_audit_sha256", half_rho.sample_audit_sha256),
        ("half_mu_sample_audit_sha256", half_mu.sample_audit_sha256),
        ("half_psi_sample_audit_sha256", half_psi.sample_audit_sha256),
        ("phase_shifted_sample_audit_sha256", phase_shifted.sample_audit_sha256),
        ("upper_annulus_area_descriptor_sha256", upper_area_hashes),
        ("lower_annulus_area_descriptor_sha256", lower_area_hashes),
        ("surface", surface),
        ("termination", termination),
        ("ray_options", ray_options),
        ("surface_options", surface_options),
        ("coarse_ray_options", coarse_ray_options),
        ("coarse_surface_options", coarse_surface_options),
        ("policy", selected_policy),
        ("area_policy", selected_area_policy),
        ("_descriptor_json", descriptor_json),
        (
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        ),
    ):
        object.__setattr__(result, name, value)
    pass_evidence = (
        _new_grid_evidence(
            full,
            pass_index=0,
            pass_name="full",
            rho_order=selected_policy.rho_order,
            mu_order=selected_policy.mu_order,
            psi_count=selected_policy.psi_count,
            phase_cells=0.0,
        ),
        _new_grid_evidence(
            half_rho,
            pass_index=1,
            pass_name="half-rho",
            rho_order=selected_policy.rho_order // 2,
            mu_order=selected_policy.mu_order,
            psi_count=selected_policy.psi_count,
            phase_cells=0.0,
        ),
        _new_grid_evidence(
            half_mu,
            pass_index=2,
            pass_name="half-mu",
            rho_order=selected_policy.rho_order,
            mu_order=selected_policy.mu_order // 2,
            psi_count=selected_policy.psi_count,
            phase_cells=0.0,
        ),
        _new_grid_evidence(
            half_psi,
            pass_index=3,
            pass_name="half-psi",
            rho_order=selected_policy.rho_order,
            mu_order=selected_policy.mu_order,
            psi_count=selected_policy.psi_count // 2,
            phase_cells=0.0,
        ),
        _new_grid_evidence(
            phase_shifted,
            pass_index=4,
            pass_name="phase-shifted",
            rho_order=selected_policy.rho_order,
            mu_order=selected_policy.mu_order,
            psi_count=selected_policy.psi_count,
            phase_cells=0.5,
        ),
    )
    evidence = object.__new__(KerrForwardReturningRadiationConvergenceEvidence)
    for name, value in (
        ("annulus_edges_over_mass", edges),
        ("upper_annulus_areas_over_mass_squared", upper_areas),
        ("lower_annulus_areas_over_mass_squared", lower_areas),
        ("kernel_descriptor_sha256", result.model_descriptor_sha256),
        ("policy", selected_policy),
        ("passes", pass_evidence),
    ):
        object.__setattr__(evidence, name, value)
    return result, evidence


def _integrate_kerr_returning_radiation_energy_kernel_with_transport_provider(
    surface: KerrFiniteThicknessMultiSurface,
    *,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
    policy: KerrReturningRadiationKernelPolicy | None = None,
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy | None = None,
    direction_transport_provider: _ForwardDirectionTransportProvider | None = None,
) -> KerrForwardReturningRadiationKernel:
    """Compatibility wrapper returning only the public kernel."""

    kernel, _evidence = (
        _integrate_kerr_returning_radiation_energy_kernel_with_transport_provider_and_evidence(
            surface,
            termination=termination,
            annulus_edges_over_mass=annulus_edges_over_mass,
            ray_options=ray_options,
            surface_options=surface_options,
            coarse_ray_options=coarse_ray_options,
            coarse_surface_options=coarse_surface_options,
            policy=policy,
            area_policy=area_policy,
            direction_transport_provider=direction_transport_provider,
        )
    )
    return kernel


def integrate_kerr_returning_radiation_energy_kernel(
    surface: KerrFiniteThicknessMultiSurface,
    *,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
    policy: KerrReturningRadiationKernelPolicy | None = None,
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy | None = None,
) -> KerrForwardReturningRadiationKernel:
    """Integrate directly; every direction uses the production ray tracer."""

    return _integrate_kerr_returning_radiation_energy_kernel_with_transport_provider(
        surface,
        termination=termination,
        annulus_edges_over_mass=annulus_edges_over_mass,
        ray_options=ray_options,
        surface_options=surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        policy=policy,
        area_policy=area_policy,
        direction_transport_provider=None,
    )


def verify_kerr_returning_radiation_energy_kernel(
    result: KerrForwardReturningRadiationKernel,
) -> None:
    """Replay every area and direction and compare the complete exact tree."""

    if type(result) is not KerrForwardReturningRadiationKernel:
        raise TypeError(
            "result must be the exact KerrForwardReturningRadiationKernel"
        )
    descriptor_json = object.__getattribute__(result, "_descriptor_json")
    descriptor_sha = object.__getattribute__(result, "_descriptor_sha256")
    if type(descriptor_json) is not str or type(descriptor_sha) is not str:
        raise KerrReturningRadiationKernelVerificationError(
            "kernel descriptor identity has a non-exact type"
        )
    try:
        parsed = json.loads(descriptor_json)
    except (json.JSONDecodeError, TypeError) as error:
        raise KerrReturningRadiationKernelVerificationError(
            "kernel descriptor is malformed"
        ) from error
    if _canonical_json(parsed).encode("utf-8") != descriptor_json.encode("utf-8"):
        raise KerrReturningRadiationKernelVerificationError(
            "kernel descriptor is not canonical"
        )
    expected_sha = hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest()
    if expected_sha.encode("ascii") != descriptor_sha.encode("ascii"):
        raise KerrReturningRadiationKernelVerificationError(
            "kernel descriptor SHA-256 is inconsistent"
        )
    rebuilt = _rebuild_verified_kerr_returning_radiation_energy_kernel(result)[0]
    _require_trusted_exact_tree(result, rebuilt, "result")


def _rebuild_verified_kerr_returning_radiation_energy_kernel(
    result: KerrForwardReturningRadiationKernel,
) -> tuple[
    KerrForwardReturningRadiationKernel,
    KerrForwardReturningRadiationConvergenceEvidence,
]:
    """Return a fresh canonical kernel and producer evidence after replay."""

    if type(result) is not KerrForwardReturningRadiationKernel:
        raise TypeError(
            "result must be the exact KerrForwardReturningRadiationKernel"
        )
    descriptor_json = object.__getattribute__(result, "_descriptor_json")
    descriptor_sha = object.__getattribute__(result, "_descriptor_sha256")
    if type(descriptor_json) is not str or type(descriptor_sha) is not str:
        raise KerrReturningRadiationKernelVerificationError(
            "kernel descriptor identity has a non-exact type"
        )
    try:
        parsed = json.loads(descriptor_json)
    except (json.JSONDecodeError, TypeError) as error:
        raise KerrReturningRadiationKernelVerificationError(
            "kernel descriptor is malformed"
        ) from error
    if _canonical_json(parsed).encode("utf-8") != descriptor_json.encode("utf-8"):
        raise KerrReturningRadiationKernelVerificationError(
            "kernel descriptor is not canonical"
        )
    expected_sha = hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest()
    if expected_sha.encode("ascii") != descriptor_sha.encode("ascii"):
        raise KerrReturningRadiationKernelVerificationError(
            "kernel descriptor SHA-256 is inconsistent"
        )
    rebuilt, evidence = (
        _integrate_kerr_returning_radiation_energy_kernel_with_transport_provider_and_evidence(
        result.surface,
        termination=result.termination,
        annulus_edges_over_mass=result.annulus_edges_over_mass,
        ray_options=result.ray_options,
        surface_options=result.surface_options,
        coarse_ray_options=result.coarse_ray_options,
        coarse_surface_options=result.coarse_surface_options,
        policy=result.policy,
        area_policy=result.area_policy,
        direction_transport_provider=None,
        )
    )
    _require_trusted_exact_tree(result, rebuilt, "result")
    return rebuilt, evidence


__all__ = (
    "IMPLEMENTATION_ID",
    "KERRBB_SOURCE_URL",
    "SCIENTIFIC_STATUS",
    "FiniteVolumeFateFractions",
    "KerrForwardReturningRadiationConvergenceEvidence",
    "KerrForwardReturningRadiationGridEvidence",
    "KerrForwardReturningRadiationKernel",
    "KerrForwardReturningRadiationKernelProjection",
    "KerrForwardReturningRadiationSampleWeightWitness",
    "KerrReturningRadiationGridDifference",
    "KerrReturningRadiationKernelConvergence",
    "KerrReturningRadiationKernelConvergenceError",
    "KerrReturningRadiationKernelError",
    "KerrReturningRadiationKernelPolicy",
    "KerrReturningRadiationKernelVerificationError",
    "integrate_kerr_returning_radiation_energy_kernel",
    "verify_and_reduce_kerr_returning_radiation_energy_kernel",
    "verify_and_reduce_kerr_returning_radiation_kernel_projection",
    "verify_kerr_returning_radiation_energy_kernel",
    "verify_kerr_returning_radiation_kernel_projection",
)
