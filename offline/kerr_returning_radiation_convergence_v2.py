"""Scale-aware convergence diagnostics for returning-radiation kernels.

The production forward kernel stores coefficients as local receiver flux per
unit local emitter flux.  Raw coefficients therefore change when receiver or
emitter finite-volume areas change.  This module compares two already-built
grids in the conserved, dimensionless proper-power coordinate

``P[i,j] = A_receiver[i] * K[i,j] / A_emitter[j]``.

For every emitter column, ``sum_i P[i,j]`` must reproduce that grid's direct
``<g**2 1_return>`` column before a comparison is accepted.  Matrix power,
direct returned power, and the five emitted-flux fate fractions have separate
gates.  Exact sampled zero is not treated as a proved physical zero: a
zero/non-zero disagreement above the declared support floor fails closed.

This is a deterministic, self-replaying *finite-grid diagnostic layer*.  It
does not authenticate caller-supplied summaries against an upstream kernel;
an authenticated adapter is required before product certification.  It is not
an independent geodesic or physics oracle, and passing it is not a rigorous
continuum error bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
from importlib.machinery import ModuleSpec, SourceFileLoader
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from types import MappingProxyType
from typing import Any, Final, Mapping


IMPLEMENTATION_ID: Final = "kerr-returning-radiation-convergence/v2"
AUTHENTICATED_IMPLEMENTATION_ID: Final = (
    f"{IMPLEMENTATION_ID}/authenticated-forward-five-pass/v1"
)
FATE_NAMES: Final = (
    "return-upper",
    "return-lower",
    "captured",
    "escaped",
    "plunge-sink",
)

_HARD_MAXIMUM_RECEIVER_CELLS: Final = 1024
_HARD_MAXIMUM_SOURCE_COLUMNS: Final = 1024
_HARD_MAXIMUM_MATRIX_CELLS: Final = 262_144
_HARD_MAXIMUM_DESCRIPTOR_BYTES: Final = 64 * 1024 * 1024
_COLUMN_CLOSURE_MAXIMUM_ULPS_AT_UNITY: Final = 4096
_MODULE_NAME: Final = "offline.kerr_returning_radiation_convergence_v2"
_TRUSTED_CHECKOUT_ROOT: Final = Path("/Users/shuolei/Documents/blackhole")
_OFFLINE_PACKAGE_OWNER: Final = sys.modules.get("offline")
_EXPECTED_MODULE_FILE: Final = (
    _TRUSTED_CHECKOUT_ROOT
    / "offline/kerr_returning_radiation_convergence_v2.py"
)
_MODULE_FILE: Final = Path(os.path.abspath(__file__))
_MODULE_OWNER: Final = sys.modules.get(_MODULE_NAME)
_MODULE_SPEC: Final = globals().get("__spec__")
_MODULE_LOADER: Final = globals().get("__loader__")


def _require_loaded_function_source_identity(function: Any, name: str) -> None:
    if type(function) is not type(_require_loaded_function_source_identity):
        raise KerrReturningRadiationConvergenceV2VerificationError(
            f"convergence-v2 {name} has a foreign callable type"
        )
    code = object.__getattribute__(function, "__code__")
    if (
        type(code.co_filename) is not str
        or Path(os.path.abspath(code.co_filename)) != _EXPECTED_MODULE_FILE
    ):
        raise KerrReturningRadiationConvergenceV2VerificationError(
            f"convergence-v2 {name} was not executed from the trusted source path"
        )


def _require_exact_module_origin() -> None:
    module_file = globals().get("__file__")
    module = sys.modules.get(_MODULE_NAME)
    offline_package = sys.modules.get("offline")
    if (
        type(globals().get("__name__")) is not str
        or globals()["__name__"] != _MODULE_NAME
        or type(module_file) is not str
        or Path(os.path.abspath(module_file)) != _MODULE_FILE
        or _MODULE_FILE != _EXPECTED_MODULE_FILE
        or type(offline_package) is not ModuleType
        or offline_package is not _OFFLINE_PACKAGE_OWNER
        or type(object.__getattribute__(offline_package, "__file__")) is not str
        or Path(object.__getattribute__(offline_package, "__file__"))
        != _TRUSTED_CHECKOUT_ROOT / "offline/__init__.py"
        or type(module) is not ModuleType
        or module is not _MODULE_OWNER
        or globals().get("__spec__") is not _MODULE_SPEC
        or globals().get("__loader__") is not _MODULE_LOADER
        or type(_MODULE_SPEC) is not ModuleSpec
        or type(_MODULE_LOADER) is not SourceFileLoader
        or _MODULE_SPEC.loader is not _MODULE_LOADER
        or type(_MODULE_SPEC.name) is not str
        or _MODULE_SPEC.name != _MODULE_NAME
        or type(_MODULE_SPEC.origin) is not str
        or Path(_MODULE_SPEC.origin) != _EXPECTED_MODULE_FILE
        or type(_MODULE_LOADER.name) is not str
        or _MODULE_LOADER.name != _MODULE_NAME
        or type(_MODULE_LOADER.path) is not str
        or Path(_MODULE_LOADER.path) != _EXPECTED_MODULE_FILE
    ):
        raise KerrReturningRadiationConvergenceV2VerificationError(
            "convergence-v2 module has no frozen source identity"
        )
    raw_origin = object.__getattribute__(module, "__file__")
    if (
        type(raw_origin) is not str
        or not Path(raw_origin).is_absolute()
        or Path(raw_origin) != _MODULE_FILE
    ):
        raise KerrReturningRadiationConvergenceV2VerificationError(
            "convergence-v2 module was loaded from another tree"
        )
    for name in (
        "_require_exact_module_origin",
        "authenticate_direct_kerr_returning_radiation_convergence_v2",
        "authenticate_cached_kerr_returning_radiation_convergence_v2",
        "verify_authenticated_kerr_returning_radiation_convergence_v2",
    ):
        _require_loaded_function_source_identity(globals().get(name), name)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "scale-aware finite-grid returning-radiation convergence diagnostic"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "comparisonCoordinate": "P[i,j]=A_receiver[i]*K[i,j]/A_emitter[j]",
        "gatesDirectG2Columns": True,
        "gatesColumnNormalizedL1": True,
        "gatesSignificantCellsAndInsignificantTailSeparately": True,
        "gatesSampledZeroSupportFlips": True,
        "gatesFateTotalVariationAbsoluteRelativeAndSupport": True,
        "gatesMaximumNormalizedSampleWeight": True,
        "acceptsCallerConstructedGridSummaries": True,
        "authenticatesUpstreamKernelProvenance": False,
        "maximumNormalizedSampleWeightIsCallerSuppliedEvidence": True,
        "requiresSameOrderedReceiverEmitterCellSemantics": True,
        "requiresAuthenticatedAdapterBeforeProductCertification": True,
        "requiresRevalidationBeforeConsumption": True,
        "descriptorAccessAloneIsAuthentication": False,
        "protectsAgainstMaliciousSameProcessObjectMutation": False,
        "isIndependentGeodesicOracle": False,
        "isIndependentPhysicsOracle": False,
        "rigorousContinuumErrorBound": False,
        "prohibitedClaim": (
            "Passing this declared finite-grid diagnostic must not be described "
            "as authenticated upstream provenance, an independent physics oracle, "
            "or a rigorous continuum error bound."
        ),
    }
)


class KerrReturningRadiationConvergenceV2Error(RuntimeError):
    """Base class for invalid or failed v2 diagnostic state."""


class KerrReturningRadiationConvergenceV2VerificationError(
    KerrReturningRadiationConvergenceV2Error
):
    """Raised when frozen state differs from exact reconstruction."""


def _canonical_json(value: Any) -> str:
    try:
        result = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise KerrReturningRadiationConvergenceV2Error(
            "v2 descriptor is not finite canonical JSON"
        ) from error
    if len(result.encode("utf-8")) > _HARD_MAXIMUM_DESCRIPTOR_BYTES:
        raise KerrReturningRadiationConvergenceV2Error(
            "v2 descriptor exceeds the hard byte limit"
        )
    return result


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_non_negative_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise TypeError(f"{label} must be a finite non-negative exact float")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{label} must use canonical positive zero")
    return value


def _exact_positive_float(value: Any, label: str) -> float:
    value = _exact_non_negative_float(value, label)
    if value <= 0.0:
        raise ValueError(f"{label} must be positive")
    return value


def _exact_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise TypeError(f"{label} must be a positive exact int")
    return value


def _exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact bool")
    return value


def _exact_grid_id(value: Any) -> str:
    if type(value) is not str or not value or len(value) > 128:
        raise TypeError("grid_id must be a non-empty exact str of at most 128 chars")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("grid_id must be ASCII") from error
    if any(byte < 0x21 or byte > 0x7E for byte in encoded):
        raise ValueError("grid_id must contain only visible ASCII without spaces")
    return value


def _require_exact_tree(actual: Any, expected: Any, path: str) -> None:
    """Compare reconstructed trusted state without equality hooks."""

    if type(actual) is not type(expected):
        raise KerrReturningRadiationConvergenceV2VerificationError(
            f"{path} has non-exact type {type(actual).__name__}; "
            f"expected {type(expected).__name__}"
        )
    if is_dataclass(expected) and not isinstance(expected, type):
        for item in fields(expected):
            try:
                actual_value = object.__getattribute__(actual, item.name)
                expected_value = object.__getattribute__(expected, item.name)
            except (AttributeError, TypeError) as error:
                raise KerrReturningRadiationConvergenceV2VerificationError(
                    f"{path}.{item.name} is missing"
                ) from error
            _require_exact_tree(
                actual_value,
                expected_value,
                f"{path}.{item.name}",
            )
        return
    if type(expected) is tuple:
        if len(actual) != len(expected):
            raise KerrReturningRadiationConvergenceV2VerificationError(
                f"{path} tuple length differs"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _require_exact_tree(actual_item, expected_item, f"{path}[{index}]")
        return
    if type(expected) is float:
        differs = actual.hex() != expected.hex()
    elif type(expected) in (int, str):
        differs = actual != expected
    elif type(expected) is bool:
        differs = actual is not expected
    elif expected is None:
        differs = False
    else:
        raise KerrReturningRadiationConvergenceV2VerificationError(
            f"{path} has unsupported exact type {type(expected).__name__}"
        )
    if differs:
        raise KerrReturningRadiationConvergenceV2VerificationError(
            f"{path} differs from exact reconstruction"
        )


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationConvergenceV2Policy:
    """Separate finite-grid gates for power, fate, support, and resolution."""

    g2_column_relative_tolerance: float = 1.0e-2
    g2_column_absolute_tolerance: float = 1.0e-6
    g2_column_relative_floor: float = 1.0e-4
    column_normalized_l1_tolerance: float = 1.0e-2
    significant_cell_fraction_of_column: float = 1.0e-3
    significant_cell_symmetric_relative_tolerance: float = 2.0e-2
    insignificant_tail_normalized_tolerance: float = 1.0e-3
    support_flip_absolute_tolerance: float = 1.0e-4
    fate_total_variation_tolerance: float = 1.0e-3
    fate_component_absolute_tolerance: float = 1.0e-3
    fate_component_relative_tolerance: float = 1.0e-1
    fate_component_relative_floor: float = 1.0e-3
    maximum_normalized_sample_weight: float = 1.0e-3
    maximum_receiver_cells: int = 512
    maximum_source_columns: int = 512
    maximum_matrix_cells: int = _HARD_MAXIMUM_MATRIX_CELLS
    _descriptor_json: str = field(init=False, repr=False, compare=False)
    _descriptor_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        fraction_fields = (
            "g2_column_relative_tolerance",
            "g2_column_absolute_tolerance",
            "g2_column_relative_floor",
            "column_normalized_l1_tolerance",
            "significant_cell_fraction_of_column",
            "significant_cell_symmetric_relative_tolerance",
            "insignificant_tail_normalized_tolerance",
            "support_flip_absolute_tolerance",
            "fate_total_variation_tolerance",
            "fate_component_absolute_tolerance",
            "fate_component_relative_tolerance",
            "fate_component_relative_floor",
            "maximum_normalized_sample_weight",
        )
        for name in fraction_fields:
            value = _exact_positive_float(getattr(self, name), name)
            if value > 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        receiver_limit = _exact_positive_int(
            self.maximum_receiver_cells,
            "maximum_receiver_cells",
        )
        source_limit = _exact_positive_int(
            self.maximum_source_columns,
            "maximum_source_columns",
        )
        matrix_limit = _exact_positive_int(
            self.maximum_matrix_cells,
            "maximum_matrix_cells",
        )
        if receiver_limit > _HARD_MAXIMUM_RECEIVER_CELLS:
            raise ValueError("maximum_receiver_cells exceeds the hard limit")
        if source_limit > _HARD_MAXIMUM_SOURCE_COLUMNS:
            raise ValueError("maximum_source_columns exceeds the hard limit")
        if matrix_limit > _HARD_MAXIMUM_MATRIX_CELLS:
            raise ValueError("maximum_matrix_cells exceeds the hard limit")
        descriptor_json = _canonical_json(self._descriptor_document())
        object.__setattr__(self, "_descriptor_json", descriptor_json)
        object.__setattr__(
            self,
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        )

    def _descriptor_document(self) -> dict[str, Any]:
        return {
            "columnNormalizedL1Tolerance": self.column_normalized_l1_tolerance,
            "fateComponentAbsoluteTolerance": self.fate_component_absolute_tolerance,
            "fateComponentRelativeFloor": self.fate_component_relative_floor,
            "fateComponentRelativeTolerance": self.fate_component_relative_tolerance,
            "fateTotalVariationTolerance": self.fate_total_variation_tolerance,
            "g2ColumnAbsoluteTolerance": self.g2_column_absolute_tolerance,
            "g2ColumnRelativeFloor": self.g2_column_relative_floor,
            "g2ColumnRelativeTolerance": self.g2_column_relative_tolerance,
            "insignificantTailNormalizedTolerance": (
                self.insignificant_tail_normalized_tolerance
            ),
            "maximumMatrixCells": self.maximum_matrix_cells,
            "maximumNormalizedSampleWeight": self.maximum_normalized_sample_weight,
            "maximumReceiverCells": self.maximum_receiver_cells,
            "maximumSourceColumns": self.maximum_source_columns,
            "schema": "blackhole.kerr-returning-radiation-convergence-policy/v2",
            "significantCellFractionOfColumn": (
                self.significant_cell_fraction_of_column
            ),
            "significantCellSymmetricRelativeTolerance": (
                self.significant_cell_symmetric_relative_tolerance
            ),
            "supportFlipAbsoluteTolerance": self.support_flip_absolute_tolerance,
        }

    @property
    def canonical_descriptor_json(self) -> str:
        return self._descriptor_json

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def descriptor(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(self._descriptor_json))

    def revalidate(self) -> None:
        expected = KerrReturningRadiationConvergenceV2Policy(
            self.g2_column_relative_tolerance,
            self.g2_column_absolute_tolerance,
            self.g2_column_relative_floor,
            self.column_normalized_l1_tolerance,
            self.significant_cell_fraction_of_column,
            self.significant_cell_symmetric_relative_tolerance,
            self.insignificant_tail_normalized_tolerance,
            self.support_flip_absolute_tolerance,
            self.fate_total_variation_tolerance,
            self.fate_component_absolute_tolerance,
            self.fate_component_relative_tolerance,
            self.fate_component_relative_floor,
            self.maximum_normalized_sample_weight,
            self.maximum_receiver_cells,
            self.maximum_source_columns,
            self.maximum_matrix_cells,
        )
        _require_exact_tree(self, expected, "policy")


def _summary_document(
    grid_id: str,
    receiver_areas: tuple[float, ...],
    emitter_areas: tuple[float, ...],
    coefficients: tuple[tuple[float, ...], ...],
    g2_columns: tuple[float, ...],
    fate_fractions: tuple[tuple[float, ...], ...],
    maximum_normalized_sample_weight: float,
) -> dict[str, Any]:
    return {
        "coefficientIndexOrder": "K[receiverCell][emitterColumn]",
        "emitterAreas": emitter_areas,
        "fateFractionOrder": FATE_NAMES,
        "fateFractions": fate_fractions,
        "g2ReturnedPowerColumns": g2_columns,
        "gridId": grid_id,
        "matrices": coefficients,
        "maximumNormalizedSampleWeight": maximum_normalized_sample_weight,
        "properPowerEquation": "P[i,j]=A_receiver[i]*K[i,j]/A_emitter[j]",
        "receiverAreas": receiver_areas,
        "schema": "blackhole.kerr-returning-radiation-grid-summary/v2",
    }


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationGridSummaryV2:
    """Finite-grid inputs needed by the independent v2 diagnostic."""

    grid_id: str
    receiver_areas: tuple[float, ...]
    emitter_areas: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    g2_columns: tuple[float, ...]
    fate_fractions: tuple[tuple[float, ...], ...]
    maximum_normalized_sample_weight: float
    _proper_power_matrix: tuple[tuple[float, ...], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _descriptor_json: str = field(init=False, repr=False, compare=False)
    _descriptor_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        grid_id = _exact_grid_id(self.grid_id)
        if type(self.receiver_areas) is not tuple or not self.receiver_areas:
            raise TypeError("receiver_areas must be a non-empty exact tuple")
        if type(self.emitter_areas) is not tuple or not self.emitter_areas:
            raise TypeError("emitter_areas must be a non-empty exact tuple")
        if len(self.receiver_areas) > _HARD_MAXIMUM_RECEIVER_CELLS:
            raise ValueError("receiver grid exceeds the hard cell limit")
        if len(self.emitter_areas) > _HARD_MAXIMUM_SOURCE_COLUMNS:
            raise ValueError("emitter grid exceeds the hard column limit")
        if len(self.receiver_areas) * len(self.emitter_areas) > (
            _HARD_MAXIMUM_MATRIX_CELLS
        ):
            raise ValueError("kernel matrix exceeds the hard cell limit")
        receiver_areas = tuple(
            _exact_positive_float(value, f"receiver_areas[{index}]")
            for index, value in enumerate(self.receiver_areas)
        )
        emitter_areas = tuple(
            _exact_positive_float(value, f"emitter_areas[{index}]")
            for index, value in enumerate(self.emitter_areas)
        )
        column_count = len(emitter_areas)
        if type(self.coefficients) is not tuple or len(self.coefficients) != len(
            receiver_areas
        ):
            raise TypeError("coefficients must have one exact tuple per receiver")
        checked_rows: list[tuple[float, ...]] = []
        proper_rows: list[tuple[float, ...]] = []
        for receiver_index, (area, row) in enumerate(
            zip(receiver_areas, self.coefficients)
        ):
            if type(row) is not tuple or len(row) != column_count:
                raise TypeError("every coefficient row must be an exact source tuple")
            checked_row = tuple(
                _exact_non_negative_float(
                    value,
                    f"coefficients[{receiver_index}][{source_index}]",
                )
                for source_index, value in enumerate(row)
            )
            proper_row: list[float] = []
            for source_index, value in enumerate(checked_row):
                power = area * value / emitter_areas[source_index]
                if not math.isfinite(power) or power < 0.0:
                    raise ValueError("proper-power transform is not finite/non-negative")
                if value > 0.0 and power == 0.0:
                    raise ValueError("proper-power transform underflowed a positive cell")
                proper_row.append(float(power))
            checked_rows.append(checked_row)
            proper_rows.append(tuple(proper_row))
        if type(self.g2_columns) is not tuple or len(self.g2_columns) != column_count:
            raise TypeError("g2_columns must be an exact source-column tuple")
        g2_columns = tuple(
            _exact_non_negative_float(value, f"g2_columns[{index}]")
            for index, value in enumerate(self.g2_columns)
        )
        if type(self.fate_fractions) is not tuple or len(
            self.fate_fractions
        ) != column_count:
            raise TypeError("fate_fractions must have one exact tuple per source")
        checked_fates: list[tuple[float, ...]] = []
        for source_index, fate_row in enumerate(self.fate_fractions):
            if type(fate_row) is not tuple or len(fate_row) != len(FATE_NAMES):
                raise TypeError("each fate row must be an exact five-fate tuple")
            checked = tuple(
                _exact_non_negative_float(
                    value,
                    f"fate_fractions[{source_index}][{fate_index}]",
                )
                for fate_index, value in enumerate(fate_row)
            )
            if any(value > 1.0 for value in checked):
                raise ValueError("fate fractions must lie in [0, 1]")
            if math.fsum(checked).hex() != 1.0.hex():
                raise ValueError("each fate row must sum exactly to one")
            checked_fates.append(checked)
        maximum_weight = _exact_positive_float(
            self.maximum_normalized_sample_weight,
            "maximum_normalized_sample_weight",
        )
        if maximum_weight > 1.0:
            raise ValueError("maximum_normalized_sample_weight must not exceed one")
        proper_matrix = tuple(proper_rows)
        for source_index, direct in enumerate(g2_columns):
            reconstructed = math.fsum(
                proper_matrix[receiver_index][source_index]
                for receiver_index in range(len(receiver_areas))
            )
            scale = max(1.0, reconstructed, direct)
            if abs(reconstructed - direct) > (
                _COLUMN_CLOSURE_MAXIMUM_ULPS_AT_UNITY * math.ulp(scale)
            ):
                raise ValueError(
                    f"proper-power column {source_index} does not close against g2"
                )
        descriptor_json = _canonical_json(
            _summary_document(
                grid_id,
                receiver_areas,
                emitter_areas,
                tuple(checked_rows),
                g2_columns,
                tuple(checked_fates),
                maximum_weight,
            )
        )
        object.__setattr__(self, "_proper_power_matrix", proper_matrix)
        object.__setattr__(self, "_descriptor_json", descriptor_json)
        object.__setattr__(
            self,
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        )

    @property
    def proper_power_matrix(self) -> tuple[tuple[float, ...], ...]:
        return self._proper_power_matrix

    @property
    def canonical_descriptor_json(self) -> str:
        return self._descriptor_json

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def descriptor(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(self._descriptor_json))

    def revalidate(self) -> None:
        expected = KerrReturningRadiationGridSummaryV2(
            self.grid_id,
            self.receiver_areas,
            self.emitter_areas,
            self.coefficients,
            self.g2_columns,
            self.fate_fractions,
            self.maximum_normalized_sample_weight,
        )
        _require_exact_tree(self, expected, "grid_summary")


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationColumnDiagnosticV2:
    source_column: int
    fine_g2: float
    comparison_g2: float
    g2_absolute_difference: float
    g2_relative_difference: float
    g2_uses_relative_gate: bool
    column_normalized_l1_difference: float
    significant_cell_count: int
    insignificant_cell_count: int
    maximum_significant_cell_symmetric_relative_difference: float
    insignificant_tail_normalized_difference: float
    support_flip_count: int
    g2_converged: bool
    column_l1_converged: bool
    significant_cells_converged: bool
    insignificant_tail_converged: bool
    support_converged: bool
    converged: bool

    def __post_init__(self) -> None:
        if type(self.source_column) is not int or self.source_column < 0:
            raise TypeError("source_column must be a non-negative exact int")
        for name in (
            "fine_g2",
            "comparison_g2",
            "g2_absolute_difference",
            "g2_relative_difference",
            "column_normalized_l1_difference",
            "maximum_significant_cell_symmetric_relative_difference",
            "insignificant_tail_normalized_difference",
        ):
            _exact_non_negative_float(getattr(self, name), name)
        for name in (
            "significant_cell_count",
            "insignificant_cell_count",
            "support_flip_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be a non-negative exact int")
        for name in (
            "g2_uses_relative_gate",
            "g2_converged",
            "column_l1_converged",
            "significant_cells_converged",
            "insignificant_tail_converged",
            "support_converged",
            "converged",
        ):
            _exact_bool(getattr(self, name), name)
        expected = all(
            (
                self.g2_converged,
                self.column_l1_converged,
                self.significant_cells_converged,
                self.insignificant_tail_converged,
                self.support_converged,
            )
        )
        if self.converged is not expected:
            raise ValueError("column convergence disagrees with component gates")

    def as_dict(self) -> dict[str, Any]:
        return {
            "columnL1Converged": self.column_l1_converged,
            "columnNormalizedL1Difference": self.column_normalized_l1_difference,
            "comparisonG2": self.comparison_g2,
            "converged": self.converged,
            "fineG2": self.fine_g2,
            "g2AbsoluteDifference": self.g2_absolute_difference,
            "g2Converged": self.g2_converged,
            "g2RelativeDifference": self.g2_relative_difference,
            "g2UsesRelativeGate": self.g2_uses_relative_gate,
            "insignificantCellCount": self.insignificant_cell_count,
            "insignificantTailConverged": self.insignificant_tail_converged,
            "insignificantTailNormalizedDifference": (
                self.insignificant_tail_normalized_difference
            ),
            "maximumSignificantCellSymmetricRelativeDifference": (
                self.maximum_significant_cell_symmetric_relative_difference
            ),
            "significantCellCount": self.significant_cell_count,
            "significantCellsConverged": self.significant_cells_converged,
            "sourceColumn": self.source_column,
            "supportConverged": self.support_converged,
            "supportFlipCount": self.support_flip_count,
        }


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationFateDiagnosticV2:
    source_column: int
    total_variation_distance: float
    maximum_absolute_difference: float
    maximum_relative_difference_above_floor: float
    relative_component_count: int
    support_flip_count: int
    total_variation_converged: bool
    absolute_components_converged: bool
    relative_components_converged: bool
    support_converged: bool
    converged: bool

    def __post_init__(self) -> None:
        if type(self.source_column) is not int or self.source_column < 0:
            raise TypeError("source_column must be a non-negative exact int")
        for name in (
            "total_variation_distance",
            "maximum_absolute_difference",
            "maximum_relative_difference_above_floor",
        ):
            _exact_non_negative_float(getattr(self, name), name)
        for name in ("relative_component_count", "support_flip_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be a non-negative exact int")
        for name in (
            "total_variation_converged",
            "absolute_components_converged",
            "relative_components_converged",
            "support_converged",
            "converged",
        ):
            _exact_bool(getattr(self, name), name)
        expected = all(
            (
                self.total_variation_converged,
                self.absolute_components_converged,
                self.relative_components_converged,
                self.support_converged,
            )
        )
        if self.converged is not expected:
            raise ValueError("fate convergence disagrees with component gates")

    def as_dict(self) -> dict[str, Any]:
        return {
            "absoluteComponentsConverged": self.absolute_components_converged,
            "converged": self.converged,
            "maximumAbsoluteDifference": self.maximum_absolute_difference,
            "maximumRelativeDifferenceAboveFloor": (
                self.maximum_relative_difference_above_floor
            ),
            "relativeComponentCount": self.relative_component_count,
            "relativeComponentsConverged": self.relative_components_converged,
            "sourceColumn": self.source_column,
            "supportConverged": self.support_converged,
            "supportFlipCount": self.support_flip_count,
            "totalVariationConverged": self.total_variation_converged,
            "totalVariationDistance": self.total_variation_distance,
        }


def _column_diagnostic(
    source_column: int,
    fine: KerrReturningRadiationGridSummaryV2,
    comparison: KerrReturningRadiationGridSummaryV2,
    policy: KerrReturningRadiationConvergenceV2Policy,
) -> KerrReturningRadiationColumnDiagnosticV2:
    fine_g2 = fine.g2_columns[source_column]
    comparison_g2 = comparison.g2_columns[source_column]
    actual_scale = max(fine_g2, comparison_g2)
    normalization_scale = max(actual_scale, policy.g2_column_relative_floor)
    g2_difference = abs(fine_g2 - comparison_g2)
    g2_relative = 0.0 if actual_scale == 0.0 else g2_difference / actual_scale
    uses_relative = actual_scale >= policy.g2_column_relative_floor
    g2_converged = (
        g2_relative <= policy.g2_column_relative_tolerance
        if uses_relative
        else g2_difference <= policy.g2_column_absolute_tolerance
    )
    differences: list[float] = []
    significant_relative: list[float] = []
    insignificant_differences: list[float] = []
    support_flips = 0
    significant_threshold = (
        policy.significant_cell_fraction_of_column * normalization_scale
    )
    for receiver_index in range(len(fine.receiver_areas)):
        fine_power = fine.proper_power_matrix[receiver_index][source_column]
        comparison_power = comparison.proper_power_matrix[receiver_index][
            source_column
        ]
        difference = abs(fine_power - comparison_power)
        differences.append(difference)
        power_scale = max(fine_power, comparison_power)
        if (fine_power == 0.0) != (comparison_power == 0.0) and (
            power_scale > policy.support_flip_absolute_tolerance
        ):
            support_flips += 1
        if power_scale >= significant_threshold and power_scale > 0.0:
            significant_relative.append(
                2.0 * difference / math.fsum((fine_power, comparison_power))
            )
        else:
            insignificant_differences.append(difference)
    column_l1 = math.fsum(differences) / normalization_scale
    maximum_significant_relative = (
        max(significant_relative) if significant_relative else 0.0
    )
    tail = math.fsum(insignificant_differences) / normalization_scale
    significant_converged = maximum_significant_relative <= (
        policy.significant_cell_symmetric_relative_tolerance
    )
    tail_converged = tail <= policy.insignificant_tail_normalized_tolerance
    support_converged = support_flips == 0
    l1_converged = column_l1 <= policy.column_normalized_l1_tolerance
    converged = all(
        (
            g2_converged,
            l1_converged,
            significant_converged,
            tail_converged,
            support_converged,
        )
    )
    return KerrReturningRadiationColumnDiagnosticV2(
        source_column,
        fine_g2,
        comparison_g2,
        g2_difference,
        g2_relative,
        uses_relative,
        column_l1,
        len(significant_relative),
        len(insignificant_differences),
        maximum_significant_relative,
        tail,
        support_flips,
        g2_converged,
        l1_converged,
        significant_converged,
        tail_converged,
        support_converged,
        converged,
    )


def _fate_diagnostic(
    source_column: int,
    fine: KerrReturningRadiationGridSummaryV2,
    comparison: KerrReturningRadiationGridSummaryV2,
    policy: KerrReturningRadiationConvergenceV2Policy,
) -> KerrReturningRadiationFateDiagnosticV2:
    fine_fates = fine.fate_fractions[source_column]
    comparison_fates = comparison.fate_fractions[source_column]
    differences: list[float] = []
    relative_differences: list[float] = []
    support_flips = 0
    for fine_value, comparison_value in zip(fine_fates, comparison_fates):
        difference = abs(fine_value - comparison_value)
        scale = max(fine_value, comparison_value)
        differences.append(difference)
        if scale >= policy.fate_component_relative_floor:
            relative_differences.append(difference / scale)
        if (fine_value == 0.0) != (comparison_value == 0.0) and (
            scale > policy.support_flip_absolute_tolerance
        ):
            support_flips += 1
    total_variation = 0.5 * math.fsum(differences)
    maximum_absolute = max(differences)
    maximum_relative = max(relative_differences) if relative_differences else 0.0
    total_variation_converged = (
        total_variation <= policy.fate_total_variation_tolerance
    )
    absolute_converged = (
        maximum_absolute <= policy.fate_component_absolute_tolerance
    )
    relative_converged = (
        maximum_relative <= policy.fate_component_relative_tolerance
    )
    support_converged = support_flips == 0
    converged = all(
        (
            total_variation_converged,
            absolute_converged,
            relative_converged,
            support_converged,
        )
    )
    return KerrReturningRadiationFateDiagnosticV2(
        source_column,
        total_variation,
        maximum_absolute,
        maximum_relative,
        len(relative_differences),
        support_flips,
        total_variation_converged,
        absolute_converged,
        relative_converged,
        support_converged,
        converged,
    )


def _comparison_document(
    fine: KerrReturningRadiationGridSummaryV2,
    comparison: KerrReturningRadiationGridSummaryV2,
    policy: KerrReturningRadiationConvergenceV2Policy,
    columns: tuple[KerrReturningRadiationColumnDiagnosticV2, ...],
    fates: tuple[KerrReturningRadiationFateDiagnosticV2, ...],
    resolution_qualified: bool,
    converged: bool,
) -> dict[str, Any]:
    return {
        "capabilities": dict(SCIENTIFIC_STATUS),
        "columnDiagnostics": tuple(item.as_dict() for item in columns),
        "comparisonGridId": comparison.grid_id,
        "comparisonGridSummarySha256": comparison.model_descriptor_sha256,
        "comparisonMaximumNormalizedSampleWeight": (
            comparison.maximum_normalized_sample_weight
        ),
        "converged": converged,
        "fateDiagnostics": tuple(item.as_dict() for item in fates),
        "fineGridId": fine.grid_id,
        "fineGridSummarySha256": fine.model_descriptor_sha256,
        "fineMaximumNormalizedSampleWeight": fine.maximum_normalized_sample_weight,
        "implementationId": IMPLEMENTATION_ID,
        "policySha256": policy.model_descriptor_sha256,
        "resolutionQualified": resolution_qualified,
        "schema": "blackhole.kerr-returning-radiation-convergence-report/v2",
    }


@dataclass(frozen=True, slots=True, init=False)
class KerrReturningRadiationGridComparisonV2:
    """Frozen diagnostic report bound to both summaries and the exact policy."""

    fine: KerrReturningRadiationGridSummaryV2
    comparison: KerrReturningRadiationGridSummaryV2
    policy: KerrReturningRadiationConvergenceV2Policy
    columns: tuple[KerrReturningRadiationColumnDiagnosticV2, ...]
    fates: tuple[KerrReturningRadiationFateDiagnosticV2, ...]
    resolution_qualified: bool
    converged: bool
    _descriptor_json: str = field(init=False, repr=False, compare=False)
    _descriptor_sha256: str = field(init=False, repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "KerrReturningRadiationGridComparisonV2 is built only by the "
            "v2 comparison factory"
        )

    def __post_init__(self) -> None:
        if type(self.fine) is not KerrReturningRadiationGridSummaryV2:
            raise TypeError("fine must be an exact v2 grid summary")
        if type(self.comparison) is not KerrReturningRadiationGridSummaryV2:
            raise TypeError("comparison must be an exact v2 grid summary")
        if type(self.policy) is not KerrReturningRadiationConvergenceV2Policy:
            raise TypeError("policy must be an exact v2 policy")
        if type(self.columns) is not tuple or any(
            type(item) is not KerrReturningRadiationColumnDiagnosticV2
            for item in self.columns
        ):
            raise TypeError("columns must be an exact diagnostic tuple")
        if type(self.fates) is not tuple or any(
            type(item) is not KerrReturningRadiationFateDiagnosticV2
            for item in self.fates
        ):
            raise TypeError("fates must be an exact diagnostic tuple")
        self.fine.revalidate()
        self.comparison.revalidate()
        self.policy.revalidate()
        if len(self.fine.receiver_areas) != len(self.comparison.receiver_areas):
            raise ValueError("comparison report receiver shapes differ")
        source_count = len(self.fine.emitter_areas)
        if source_count != len(self.comparison.emitter_areas):
            raise ValueError("comparison report source shapes differ")
        receiver_count = len(self.fine.receiver_areas)
        if receiver_count > self.policy.maximum_receiver_cells:
            raise ValueError("comparison report exceeds its receiver-cell limit")
        if source_count > self.policy.maximum_source_columns:
            raise ValueError("comparison report exceeds its source-column limit")
        if receiver_count * source_count > self.policy.maximum_matrix_cells:
            raise ValueError("comparison report exceeds its matrix-cell limit")
        expected_columns = tuple(
            _column_diagnostic(index, self.fine, self.comparison, self.policy)
            for index in range(source_count)
        )
        expected_fates = tuple(
            _fate_diagnostic(index, self.fine, self.comparison, self.policy)
            for index in range(source_count)
        )
        _require_exact_tree(self.columns, expected_columns, "report.columns")
        _require_exact_tree(self.fates, expected_fates, "report.fates")
        _exact_bool(self.resolution_qualified, "resolution_qualified")
        _exact_bool(self.converged, "converged")
        expected_resolution = max(
            self.fine.maximum_normalized_sample_weight,
            self.comparison.maximum_normalized_sample_weight,
        ) <= self.policy.maximum_normalized_sample_weight
        if self.resolution_qualified is not expected_resolution:
            raise ValueError("resolution qualification differs from exact inputs")
        expected = expected_resolution and all(
            item.converged for item in (*self.columns, *self.fates)
        )
        if self.converged is not expected:
            raise ValueError("aggregate convergence disagrees with component gates")
        descriptor_json = _canonical_json(
            _comparison_document(
                self.fine,
                self.comparison,
                self.policy,
                self.columns,
                self.fates,
                self.resolution_qualified,
                self.converged,
            )
        )
        object.__setattr__(self, "_descriptor_json", descriptor_json)
        object.__setattr__(
            self,
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        )

    @property
    def canonical_descriptor_json(self) -> str:
        return self._descriptor_json

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def descriptor(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(self._descriptor_json))

    def revalidate(self) -> None:
        verify_kerr_returning_radiation_grid_comparison_v2(self)


def compare_kerr_returning_radiation_grids_v2(
    fine: KerrReturningRadiationGridSummaryV2,
    comparison: KerrReturningRadiationGridSummaryV2,
    policy: KerrReturningRadiationConvergenceV2Policy | None = None,
) -> KerrReturningRadiationGridComparisonV2:
    """Compare two same-shape finite grids with scale-aware fail-closed gates."""

    if type(fine) is not KerrReturningRadiationGridSummaryV2:
        raise TypeError("fine must be an exact v2 grid summary")
    if type(comparison) is not KerrReturningRadiationGridSummaryV2:
        raise TypeError("comparison must be an exact v2 grid summary")
    selected = (
        KerrReturningRadiationConvergenceV2Policy()
        if policy is None
        else policy
    )
    if type(selected) is not KerrReturningRadiationConvergenceV2Policy:
        raise TypeError("policy must be an exact v2 policy or None")
    fine.revalidate()
    comparison.revalidate()
    selected.revalidate()
    if len(fine.receiver_areas) != len(comparison.receiver_areas):
        raise ValueError("grid summaries have different receiver-cell counts")
    if len(fine.emitter_areas) != len(comparison.emitter_areas):
        raise ValueError("grid summaries have different source-column counts")
    receiver_count = len(fine.receiver_areas)
    source_count = len(fine.emitter_areas)
    if receiver_count > selected.maximum_receiver_cells:
        raise ValueError("comparison exceeds the policy receiver-cell limit")
    if source_count > selected.maximum_source_columns:
        raise ValueError("comparison exceeds the policy source-column limit")
    if receiver_count * source_count > selected.maximum_matrix_cells:
        raise ValueError("comparison exceeds the policy matrix-cell limit")
    columns = tuple(
        _column_diagnostic(source_index, fine, comparison, selected)
        for source_index in range(source_count)
    )
    fates = tuple(
        _fate_diagnostic(source_index, fine, comparison, selected)
        for source_index in range(source_count)
    )
    resolution_qualified = max(
        fine.maximum_normalized_sample_weight,
        comparison.maximum_normalized_sample_weight,
    ) <= selected.maximum_normalized_sample_weight
    converged = resolution_qualified and all(
        item.converged for item in (*columns, *fates)
    )
    result = object.__new__(KerrReturningRadiationGridComparisonV2)
    for name, value in (
        ("fine", fine),
        ("comparison", comparison),
        ("policy", selected),
        ("columns", columns),
        ("fates", fates),
        ("resolution_qualified", resolution_qualified),
        ("converged", converged),
    ):
        object.__setattr__(result, name, value)
    result.__post_init__()
    return result


def verify_kerr_returning_radiation_grid_comparison_v2(
    result: KerrReturningRadiationGridComparisonV2,
) -> None:
    """Rebuild a report from its exact frozen inputs and reject tampering."""

    if type(result) is not KerrReturningRadiationGridComparisonV2:
        raise TypeError("result must be an exact v2 grid comparison")
    try:
        fine = object.__getattribute__(result, "fine")
        comparison = object.__getattribute__(result, "comparison")
        policy = object.__getattribute__(result, "policy")
    except (AttributeError, TypeError) as error:
        raise KerrReturningRadiationConvergenceV2VerificationError(
            "comparison inputs are missing"
        ) from error
    try:
        expected = compare_kerr_returning_radiation_grids_v2(
            fine,
            comparison,
            policy,
        )
    except (TypeError, ValueError, KerrReturningRadiationConvergenceV2Error) as error:
        raise KerrReturningRadiationConvergenceV2VerificationError(
            "comparison inputs cannot reproduce a trusted report"
        ) from error
    _require_exact_tree(result, expected, "comparison_report")


def _authenticated_resource_gate(annulus_count: int, policy: Any) -> None:
    if type(annulus_count) is not int or annulus_count < 1:
        raise KerrReturningRadiationConvergenceV2Error(
            "authenticated source has an invalid annulus count"
        )
    receiver_count = 2 * annulus_count
    source_count = receiver_count
    if receiver_count > min(
        _HARD_MAXIMUM_RECEIVER_CELLS, policy.maximum_receiver_cells
    ):
        raise KerrReturningRadiationConvergenceV2Error(
            "authenticated source exceeds the receiver-cell limit"
        )
    if source_count > min(
        _HARD_MAXIMUM_SOURCE_COLUMNS, policy.maximum_source_columns
    ):
        raise KerrReturningRadiationConvergenceV2Error(
            "authenticated source exceeds the source-column limit"
        )
    if receiver_count * source_count > min(
        _HARD_MAXIMUM_MATRIX_CELLS, policy.maximum_matrix_cells
    ):
        raise KerrReturningRadiationConvergenceV2Error(
            "authenticated source exceeds the matrix-cell limit"
        )


def _evidence_summary(
    evidence: Any,
    grid: Any,
    *,
    source_kind: str,
) -> KerrReturningRadiationGridSummaryV2:
    upper_areas = evidence.upper_annulus_areas_over_mass_squared
    lower_areas = evidence.lower_annulus_areas_over_mass_squared
    receiver_areas = (*upper_areas, *lower_areas)
    emitter_areas = receiver_areas
    coefficients = tuple(
        (*uu_row, *ul_row)
        for uu_row, ul_row in zip(
            grid.upper_receiver_upper_emitter_coefficients,
            grid.upper_receiver_lower_emitter_coefficients,
        )
    ) + tuple(
        (*lu_row, *ll_row)
        for lu_row, ll_row in zip(
            grid.lower_receiver_upper_emitter_coefficients,
            grid.lower_receiver_lower_emitter_coefficients,
        )
    )
    g2_columns = (
        *grid.upper_emitter_g2_returned_power_columns,
        *grid.lower_emitter_g2_returned_power_columns,
    )
    fates = tuple(
        item.as_tuple()
        for item in (
            *grid.upper_emitter_fate_fractions,
            *grid.lower_emitter_fate_fractions,
        )
    )
    return KerrReturningRadiationGridSummaryV2(
        f"{source_kind}-{grid.pass_name}",
        receiver_areas,
        emitter_areas,
        coefficients,
        g2_columns,
        fates,
        grid.maximum_normalized_sample_weight,
    )


def _witness_document(witness: Any) -> dict[str, Any]:
    return {
        "emissionAngleCosine": witness.emission_angle_cosine,
        "muIndex": witness.mu_index,
        "normalizedEmittedFluxDirectionWeight": (
            witness.normalized_emitted_flux_direction_weight
        ),
        "normalizedSampleWeight": witness.normalized_sample_weight,
        "passIndex": witness.pass_index,
        "passName": witness.pass_name,
        "psiIndex": witness.psi_index,
        "rhoAreaOverMassSquared": witness.rho_area_over_mass_squared,
        "rhoIndex": witness.rho_index,
        "sourceAnnulusIndex": witness.source_annulus_index,
        "sourceFace": witness.source_face,
        "sourceRadiusOverMass": witness.source_radius_over_mass,
        "tangentAzimuthRad": witness.tangent_azimuth_rad,
    }


def _authenticated_document(
    source_kind: str,
    evidence: Any,
    summaries: tuple[KerrReturningRadiationGridSummaryV2, ...],
    comparisons: tuple[KerrReturningRadiationGridComparisonV2, ...],
    policy: KerrReturningRadiationConvergenceV2Policy,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "authentication": {
            "acceptsCallerConstructedSummaries": False,
            "isIndependentGeodesicOrPhysicsOracle": False,
            "isRigorousContinuumErrorBound": False,
            "revalidationRebuildsFromRetainedSource": True,
            "sameCodeEvidence": True,
        },
        "cellSemantics": {
            "coefficientIndexOrder": "K[receiverCell][emitterColumn]",
            "emitterCellOrder": "upper annuli then lower annuli",
            "matrixBlockRows": "UU|UL then LU|LL",
            "receiverCellOrder": "upper annuli then lower annuli",
        },
        "gridGeometry": {
            "annulusEdgesOverMass": evidence.annulus_edges_over_mass,
            "lowerAnnulusAreasOverMassSquared": (
                evidence.lower_annulus_areas_over_mass_squared
            ),
            "upperAnnulusAreasOverMassSquared": (
                evidence.upper_annulus_areas_over_mass_squared
            ),
        },
        "comparisonReportSha256": tuple(
            item.model_descriptor_sha256 for item in comparisons
        ),
        "implementationId": AUTHENTICATED_IMPLEMENTATION_ID,
        "kernelDescriptorSha256": evidence.kernel_descriptor_sha256,
        "passEvidence": tuple(
            {
                "directionEvaluations": item.direction_evaluations,
                "maximumNormalizedSampleWeight": (
                    item.maximum_normalized_sample_weight
                ),
                "maximumNormalizedSampleWeightWitness": _witness_document(
                    item.maximum_normalized_sample_weight_witness
                ),
                "lowerEmitterG2ColumnClosureResiduals": (
                    item.lower_emitter_g2_column_closure_residuals
                ),
                "muOrder": item.mu_order,
                "passIndex": item.pass_index,
                "passName": item.pass_name,
                "phaseCells": item.phase_cells,
                "psiCount": item.psi_count,
                "rhoOrder": item.rho_order,
                "sampleAuditSha256": item.sample_audit_sha256,
                "upperEmitterG2ColumnClosureResiduals": (
                    item.upper_emitter_g2_column_closure_residuals
                ),
            }
            for item in evidence.passes
        ),
        "policySha256": policy.model_descriptor_sha256,
        "provenance": dict(provenance),
        "schema": "blackhole.kerr-returning-radiation-authenticated-convergence/v2",
        "sourceKind": source_kind,
        "summarySha256": tuple(item.model_descriptor_sha256 for item in summaries),
    }


@dataclass(frozen=True, slots=True, init=False)
class KerrAuthenticatedReturningRadiationConvergenceV2:
    """Factory-only five-pass convergence reports bound to a replayable source."""

    source_kind: str
    summaries: tuple[KerrReturningRadiationGridSummaryV2, ...]
    comparisons: tuple[KerrReturningRadiationGridComparisonV2, ...]
    policy: KerrReturningRadiationConvergenceV2Policy
    canonical_descriptor_json: str
    model_descriptor_sha256: str
    _source: Any

    def __init__(self) -> None:
        raise TypeError(
            "authenticated convergence is built only by a source adapter factory"
        )

    @property
    def full(self) -> KerrReturningRadiationGridSummaryV2:
        return self.summaries[0]

    @property
    def reports(self) -> tuple[KerrReturningRadiationGridComparisonV2, ...]:
        return self.comparisons

    @property
    def converged(self) -> bool:
        return all(item.converged for item in self.comparisons)

    def _fresh_authenticated(self) -> "KerrAuthenticatedReturningRadiationConvergenceV2":
        source_kind = object.__getattribute__(self, "source_kind")
        source = object.__getattribute__(self, "_source")
        policy = object.__getattribute__(self, "policy")
        if type(source_kind) is str and source_kind.encode("utf-8") == b"direct":
            return authenticate_direct_kerr_returning_radiation_convergence_v2(
                source, policy
            )
        if type(source_kind) is str and source_kind.encode("utf-8") == b"cached":
            return authenticate_cached_kerr_returning_radiation_convergence_v2(
                source, policy
            )
        raise KerrReturningRadiationConvergenceV2VerificationError(
            "authenticated source kind is unsupported"
        )

    def descriptor(self) -> Mapping[str, Any]:
        fresh = self._fresh_authenticated()
        return MappingProxyType(json.loads(fresh.canonical_descriptor_json))

    def revalidate(self) -> None:
        verify_authenticated_kerr_returning_radiation_convergence_v2(self)


def _build_authenticated_result(
    *,
    source_kind: str,
    source: Any,
    evidence: Any,
    policy: KerrReturningRadiationConvergenceV2Policy,
    provenance: Mapping[str, Any],
) -> KerrAuthenticatedReturningRadiationConvergenceV2:
    if type(evidence.passes) is not tuple or len(evidence.passes) != 5:
        raise KerrReturningRadiationConvergenceV2VerificationError(
            "producer evidence must contain exactly five passes"
        )
    producer_policy = evidence.policy
    from offline import kerr_returning_radiation_kernel as forward

    if type(producer_policy) is not forward.KerrReturningRadiationKernelPolicy:
        raise KerrReturningRadiationConvergenceV2VerificationError(
            "producer evidence policy has a non-exact type"
        )
    expected_passes = (
        (
            0,
            "full",
            producer_policy.rho_order,
            producer_policy.mu_order,
            producer_policy.psi_count,
            0.0,
        ),
        (
            1,
            "half-rho",
            producer_policy.rho_order // 2,
            producer_policy.mu_order,
            producer_policy.psi_count,
            0.0,
        ),
        (
            2,
            "half-mu",
            producer_policy.rho_order,
            producer_policy.mu_order // 2,
            producer_policy.psi_count,
            0.0,
        ),
        (
            3,
            "half-psi",
            producer_policy.rho_order,
            producer_policy.mu_order,
            producer_policy.psi_count // 2,
            0.0,
        ),
        (
            4,
            "phase-shifted",
            producer_policy.rho_order,
            producer_policy.mu_order,
            producer_policy.psi_count,
            0.5,
        ),
    )
    actual_passes = tuple(
        (
            item.pass_index,
            item.pass_name,
            item.rho_order,
            item.mu_order,
            item.psi_count,
            item.phase_cells,
        )
        for item in evidence.passes
    )
    if any(
        type(actual_index) is not int
        or type(actual_name) is not str
        or type(actual_rho) is not int
        or type(actual_mu) is not int
        or type(actual_psi) is not int
        or type(actual_phase) is not float
        or not math.isfinite(actual_phase)
        or actual_phase.hex() != expected_phase.hex()
        or actual_index != expected_index
        or actual_name.encode("utf-8") != expected_name.encode("utf-8")
        or actual_rho != expected_rho
        or actual_mu != expected_mu
        or actual_psi != expected_psi
        for (
            actual_index,
            actual_name,
            actual_rho,
            actual_mu,
            actual_psi,
            actual_phase,
        ), (
            expected_index,
            expected_name,
            expected_rho,
            expected_mu,
            expected_psi,
            expected_phase,
        ) in zip(actual_passes, expected_passes)
    ):
        raise KerrReturningRadiationConvergenceV2VerificationError(
            "producer pass identity, grid relationship, or order is non-canonical"
        )
    _authenticated_resource_gate(len(evidence.annulus_edges_over_mass) - 1, policy)
    annulus_count = len(evidence.annulus_edges_over_mass) - 1
    area_by_face = {
        "upper": evidence.upper_annulus_areas_over_mass_squared,
        "lower": evidence.lower_annulus_areas_over_mass_squared,
    }
    for item in evidence.passes:
        if type(item) is not forward.KerrForwardReturningRadiationGridEvidence:
            raise KerrReturningRadiationConvergenceV2VerificationError(
                "producer pass evidence has a non-exact type"
            )
        expected_directions = (
            2 * annulus_count * item.rho_order * item.mu_order * item.psi_count
        )
        if (
            type(item.direction_evaluations) is not int
            or item.direction_evaluations != expected_directions
        ):
            raise KerrReturningRadiationConvergenceV2VerificationError(
                "producer pass direction count disagrees with its fixed grid"
            )
        if (
            type(item.sample_audit_sha256) is not str
            or len(item.sample_audit_sha256) != 64
            or item.sample_audit_sha256.lower() != item.sample_audit_sha256
        ):
            raise KerrReturningRadiationConvergenceV2VerificationError(
                "producer pass sample audit is not a canonical SHA-256"
            )
        try:
            bytes.fromhex(item.sample_audit_sha256)
        except ValueError as error:
            raise KerrReturningRadiationConvergenceV2VerificationError(
                "producer pass sample audit is not a canonical SHA-256"
            ) from error
        witness = item.maximum_normalized_sample_weight_witness
        if type(witness) is not forward.KerrForwardReturningRadiationSampleWeightWitness:
            raise KerrReturningRadiationConvergenceV2VerificationError(
                "maximum sample-weight witness has a non-exact type"
            )
        source_areas = area_by_face.get(witness.source_face)
        if (
            type(witness.source_face) is not str
            or witness.source_face.encode("utf-8") not in (b"upper", b"lower")
            or source_areas is None
            or type(witness.source_annulus_index) is not int
            or witness.source_annulus_index < 0
            or witness.source_annulus_index >= annulus_count
            or type(witness.rho_index) is not int
            or not 0 <= witness.rho_index < item.rho_order
            or type(witness.mu_index) is not int
            or not 0 <= witness.mu_index < item.mu_order
            or type(witness.psi_index) is not int
            or not 0 <= witness.psi_index < item.psi_count
            or witness.pass_index != item.pass_index
            or type(witness.pass_name) is not str
            or witness.pass_name.encode("utf-8") != item.pass_name.encode("utf-8")
        ):
            raise KerrReturningRadiationConvergenceV2VerificationError(
                "maximum sample-weight witness has a non-canonical coordinate"
            )
        witness_floats = (
            witness.source_radius_over_mass,
            witness.rho_area_over_mass_squared,
            witness.emission_angle_cosine,
            witness.tangent_azimuth_rad,
            witness.normalized_emitted_flux_direction_weight,
            witness.normalized_sample_weight,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in witness_floats):
            raise KerrReturningRadiationConvergenceV2VerificationError(
                "maximum sample-weight witness contains a non-exact/non-finite float"
            )
        if (
            witness.source_radius_over_mass <= 0.0
            or witness.rho_area_over_mass_squared <= 0.0
            or not 0.0 < witness.emission_angle_cosine < 1.0
            or not 0.0 <= witness.tangent_azimuth_rad < 2.0 * math.pi
            or witness.normalized_emitted_flux_direction_weight <= 0.0
            or witness.normalized_sample_weight <= 0.0
        ):
            raise KerrReturningRadiationConvergenceV2VerificationError(
                "maximum sample-weight witness is outside its physical node domain"
            )
        reconstructed_weight = (
            witness.rho_area_over_mass_squared
            * witness.normalized_emitted_flux_direction_weight
            / source_areas[witness.source_annulus_index]
        )
        if (
            type(item.maximum_normalized_sample_weight) is not float
            or not math.isfinite(item.maximum_normalized_sample_weight)
            or item.maximum_normalized_sample_weight <= 0.0
            or type(witness.normalized_sample_weight) is not float
            or witness.normalized_sample_weight.hex()
            != item.maximum_normalized_sample_weight.hex()
            or reconstructed_weight.hex()
            != item.maximum_normalized_sample_weight.hex()
        ):
            raise KerrReturningRadiationConvergenceV2VerificationError(
                "maximum sample-weight witness does not reproduce its value"
            )
        for source_face, closure, g2, receiver_upper, receiver_lower in (
            (
                "upper",
                item.upper_emitter_g2_column_closure_residuals,
                item.upper_emitter_g2_returned_power_columns,
                item.upper_receiver_upper_emitter_coefficients,
                item.lower_receiver_upper_emitter_coefficients,
            ),
            (
                "lower",
                item.lower_emitter_g2_column_closure_residuals,
                item.lower_emitter_g2_returned_power_columns,
                item.upper_receiver_lower_emitter_coefficients,
                item.lower_receiver_lower_emitter_coefficients,
            ),
        ):
            source_area_values = area_by_face[source_face]
            if (
                type(closure) is not tuple
                or len(closure) != annulus_count
                or type(g2) is not tuple
                or len(g2) != annulus_count
            ):
                raise KerrReturningRadiationConvergenceV2VerificationError(
                    "pass closure evidence has a non-canonical shape"
                )
            for source_index, actual_closure in enumerate(closure):
                if (
                    type(actual_closure) is not float
                    or not math.isfinite(actual_closure)
                    or actual_closure < 0.0
                    or (
                        actual_closure == 0.0
                        and math.copysign(1.0, actual_closure) < 0.0
                    )
                ):
                    raise KerrReturningRadiationConvergenceV2VerificationError(
                        "pass closure residual is not canonical non-negative float"
                    )
                reconstructed = math.fsum(
                    tuple(
                        evidence.upper_annulus_areas_over_mass_squared[receiver_index]
                        * receiver_upper[receiver_index][source_index]
                        / source_area_values[source_index]
                        for receiver_index in range(annulus_count)
                    )
                    + tuple(
                        evidence.lower_annulus_areas_over_mass_squared[receiver_index]
                        * receiver_lower[receiver_index][source_index]
                        / source_area_values[source_index]
                        for receiver_index in range(annulus_count)
                    )
                )
                expected_closure = abs(reconstructed - g2[source_index])
                if actual_closure.hex() != expected_closure.hex():
                    raise KerrReturningRadiationConvergenceV2VerificationError(
                        "pass closure residual differs from its matrix and g2 column"
                    )
    summaries = tuple(
        _evidence_summary(evidence, item, source_kind=source_kind)
        for item in evidence.passes
    )
    comparisons = tuple(
        compare_kerr_returning_radiation_grids_v2(summaries[0], item, policy)
        for item in summaries[1:]
    )
    descriptor_json = _canonical_json(
        _authenticated_document(
            source_kind,
            evidence,
            summaries,
            comparisons,
            policy,
            provenance,
        )
    )
    result = object.__new__(KerrAuthenticatedReturningRadiationConvergenceV2)
    for name, value in (
        ("source_kind", source_kind),
        ("summaries", summaries),
        ("comparisons", comparisons),
        ("policy", policy),
        ("canonical_descriptor_json", descriptor_json),
        (
            "model_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        ),
        ("_source", source),
    ):
        object.__setattr__(result, name, value)
    return result


def _require_full_evidence_matches_kernel(evidence: Any, kernel: Any) -> None:
    full = evidence.passes[0]
    pairs = (
        (full.upper_receiver_upper_emitter_coefficients, kernel.upper_receiver_upper_emitter_coefficients),
        (full.upper_receiver_lower_emitter_coefficients, kernel.upper_receiver_lower_emitter_coefficients),
        (full.lower_receiver_upper_emitter_coefficients, kernel.lower_receiver_upper_emitter_coefficients),
        (full.lower_receiver_lower_emitter_coefficients, kernel.lower_receiver_lower_emitter_coefficients),
        (full.upper_emitter_fate_fractions, kernel.upper_emitter_fate_fractions),
        (full.lower_emitter_fate_fractions, kernel.lower_emitter_fate_fractions),
        (full.upper_emitter_g2_returned_power_columns, kernel.upper_emitter_g2_returned_power_columns),
        (full.lower_emitter_g2_returned_power_columns, kernel.lower_emitter_g2_returned_power_columns),
        (full.upper_emitter_g2_column_closure_residuals, kernel.upper_emitter_g2_column_closure_residuals),
        (full.lower_emitter_g2_column_closure_residuals, kernel.lower_emitter_g2_column_closure_residuals),
    )
    for index, (actual, expected) in enumerate(pairs):
        _require_exact_tree(actual, expected, f"full_kernel_parity[{index}]")
    if full.sample_audit_sha256.encode("ascii") != (
        kernel.full_grid_sample_audit_sha256.encode("ascii")
    ):
        raise KerrReturningRadiationConvergenceV2VerificationError(
            "full evidence sample audit differs from its kernel"
        )


def authenticate_direct_kerr_returning_radiation_convergence_v2(
    kernel: Any,
    policy: KerrReturningRadiationConvergenceV2Policy | None = None,
) -> KerrAuthenticatedReturningRadiationConvergenceV2:
    """Replay one exact direct forward kernel and adapt its fresh evidence."""

    _require_exact_module_origin()
    from offline import kerr_returning_radiation_kernel as forward

    if type(kernel) is not forward.KerrForwardReturningRadiationKernel:
        raise TypeError("kernel must be an exact direct forward kernel")
    selected = KerrReturningRadiationConvergenceV2Policy() if policy is None else policy
    if type(selected) is not KerrReturningRadiationConvergenceV2Policy:
        raise TypeError("policy must be an exact v2 policy or None")
    selected.revalidate()
    raw_edges = object.__getattribute__(kernel, "annulus_edges_over_mass")
    if type(raw_edges) is not tuple:
        raise KerrReturningRadiationConvergenceV2VerificationError(
            "direct source edges have a non-exact type"
        )
    _authenticated_resource_gate(len(raw_edges) - 1, selected)
    rebuilt, evidence = forward._rebuild_verified_kerr_returning_radiation_energy_kernel(
        kernel
    )
    _require_full_evidence_matches_kernel(evidence, rebuilt)
    return _build_authenticated_result(
        source_kind="direct",
        source=kernel,
        evidence=evidence,
        policy=selected,
        provenance={
            "forwardKernelDescriptorSha256": rebuilt.model_descriptor_sha256,
            "producerAuthentication": "direct exact whole-kernel replay",
            "producerImplementationId": forward.IMPLEMENTATION_ID,
        },
    )


def authenticate_cached_kerr_returning_radiation_convergence_v2(
    execution: Any,
    policy: KerrReturningRadiationConvergenceV2Policy | None = None,
) -> KerrAuthenticatedReturningRadiationConvergenceV2:
    """Authenticate one production-forward cache and adapt its replay evidence."""

    _require_exact_module_origin()
    from offline import kerr_returning_radiation_kernel_cached as cached

    if type(execution) is not cached.KerrCachedReturningRadiationKernelExecution:
        raise TypeError("execution must be an exact cached execution")
    selected = KerrReturningRadiationConvergenceV2Policy() if policy is None else policy
    if type(selected) is not KerrReturningRadiationConvergenceV2Policy:
        raise TypeError("policy must be an exact v2 policy or None")
    selected.revalidate()
    raw_kernel = object.__getattribute__(execution, "kernel")
    raw_edges = object.__getattribute__(raw_kernel, "annulus_edges_over_mass")
    if type(raw_edges) is not tuple:
        raise KerrReturningRadiationConvergenceV2VerificationError(
            "cached source edges have a non-exact type"
        )
    _authenticated_resource_gate(len(raw_edges) - 1, selected)
    rebuilt, evidence = cached._rebuild_verified_cached_forward_execution_with_evidence(
        execution
    )
    _require_full_evidence_matches_kernel(evidence, rebuilt.kernel)
    binding_json, binding_sha = cached._production_forward_scientific_binding(
        rebuilt
    )
    context = rebuilt.cache_definition.scientific_context
    return _build_authenticated_result(
        source_kind="cached",
        source=execution,
        evidence=evidence,
        policy=selected,
        provenance={
            "forwardKernelDescriptorSha256": rebuilt.kernel.model_descriptor_sha256,
            "producerAuthentication": "same-code authenticated cache replay",
            "producerImplementationId": cached.IMPLEMENTATION_ID,
            "reductionConfigurationSha256": rebuilt.reduction_configuration_sha256,
            "scientificBindingSha256": binding_sha,
            "scientificBindingJsonSha256": hashlib.sha256(
                binding_json.encode("utf-8")
            ).hexdigest(),
            "sourceClosureManifestSha256": (
                cached._source_closure_manifest_sha256(rebuilt.source_closure)
            ),
            "transportScientificJobKey": context.scientific_job_key,
        },
    )


def verify_authenticated_kerr_returning_radiation_convergence_v2(
    result: KerrAuthenticatedReturningRadiationConvergenceV2,
) -> None:
    """Rebuild from the retained source; never trust post-validation fields."""

    _require_exact_module_origin()
    if type(result) is not KerrAuthenticatedReturningRadiationConvergenceV2:
        raise TypeError("result must be an exact authenticated v2 result")
    source_kind = object.__getattribute__(result, "source_kind")
    source = object.__getattribute__(result, "_source")
    policy = object.__getattribute__(result, "policy")
    try:
        if type(source_kind) is not str:
            raise TypeError("source kind must be exact str")
        if source_kind.encode("utf-8") == b"direct":
            expected = authenticate_direct_kerr_returning_radiation_convergence_v2(
                source, policy
            )
        elif source_kind.encode("utf-8") == b"cached":
            expected = authenticate_cached_kerr_returning_radiation_convergence_v2(
                source, policy
            )
        else:
            raise TypeError("source kind is unsupported")
    except Exception as error:
        if isinstance(error, KerrReturningRadiationConvergenceV2VerificationError):
            raise
        raise KerrReturningRadiationConvergenceV2VerificationError(
            "retained source cannot reproduce authenticated convergence"
        ) from error
    for name in (
        "source_kind",
        "summaries",
        "comparisons",
        "policy",
        "canonical_descriptor_json",
        "model_descriptor_sha256",
    ):
        _require_exact_tree(
            object.__getattribute__(result, name),
            object.__getattribute__(expected, name),
            f"authenticated_convergence.{name}",
        )


# Explicit grid aliases retain the terminology used by the diagnostic layer.
authenticate_direct_kerr_returning_radiation_grid_v2 = (
    authenticate_direct_kerr_returning_radiation_convergence_v2
)
authenticate_cached_kerr_returning_radiation_grid_v2 = (
    authenticate_cached_kerr_returning_radiation_convergence_v2
)


__all__ = (
    "AUTHENTICATED_IMPLEMENTATION_ID",
    "FATE_NAMES",
    "IMPLEMENTATION_ID",
    "SCIENTIFIC_STATUS",
    "KerrReturningRadiationColumnDiagnosticV2",
    "KerrReturningRadiationConvergenceV2Error",
    "KerrReturningRadiationConvergenceV2Policy",
    "KerrReturningRadiationConvergenceV2VerificationError",
    "KerrReturningRadiationFateDiagnosticV2",
    "KerrReturningRadiationGridComparisonV2",
    "KerrReturningRadiationGridSummaryV2",
    "KerrAuthenticatedReturningRadiationConvergenceV2",
    "authenticate_cached_kerr_returning_radiation_convergence_v2",
    "authenticate_cached_kerr_returning_radiation_grid_v2",
    "authenticate_direct_kerr_returning_radiation_convergence_v2",
    "authenticate_direct_kerr_returning_radiation_grid_v2",
    "compare_kerr_returning_radiation_grids_v2",
    "verify_kerr_returning_radiation_grid_comparison_v2",
    "verify_authenticated_kerr_returning_radiation_convergence_v2",
)
