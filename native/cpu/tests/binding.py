"""Minimal ctypes binding used only by native CPU tests and benchmarks."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Iterable, Sequence


CPU_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = CPU_ROOT / "build" / "libblackhole_cpu.dylib"

BH_CPU_OK = 0
BH_CPU_INVALID_ARGUMENT = 1
BH_CPU_INVALID_FP_ENVIRONMENT = 2
BH_CPU_NONFINITE_INPUT = 3
BH_CPU_NUMERIC_OVERFLOW = 4
BH_CPU_INCONSISTENT_METRIC = 5
BH_CPU_TOO_MANY_TERMS = 6
BH_CPU_GUARDED_SINGULARITY = 7
BH_CPU_CAPACITY_EXCEEDED = 8

BH_CPU_ABI_VERSION = 3

BH_CPU_RAY_OUTCOME_NONE = 0
BH_CPU_RAY_OUTCOME_CAPTURED = 1
BH_CPU_RAY_OUTCOME_ESCAPED = 2
BH_CPU_RAY_OUTCOME_RETURNED = 3
BH_CPU_RAY_OUTCOME_PLUNGE_SINK = 4
BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE = 5
BH_CPU_RAY_OUTCOME_UNRESOLVED = 6
BH_CPU_RAY_OUTCOME_COMPLETED = 7

BH_CPU_TARGET_NONE = 0
BH_CPU_TARGET_KERR_STRETCHED_HORIZON = 1
BH_CPU_TARGET_KERR_ESCAPE_WORLDTUBE = 2
BH_CPU_TARGET_OPAQUE_LOWER_FACE = 3
BH_CPU_TARGET_OPAQUE_UPPER_FACE = 4
BH_CPU_TARGET_LOWER_PLUNGE_ENTRY = 5
BH_CPU_TARGET_UPPER_PLUNGE_ENTRY = 6

BH_CPU_SURFACE_LOWER = 0
BH_CPU_SURFACE_UPPER = 1

BH_CPU_SURFACE_CLASSIFICATION_NONE = 0
BH_CPU_SURFACE_INSIDE_ISCO_TRANSPARENT = 1
BH_CPU_SURFACE_OUTSIDE_OUTER_TRANSPARENT = 2
BH_CPU_SURFACE_OUTWARD_LOWER_PLUNGE_EXIT = 3
BH_CPU_SURFACE_OUTWARD_UPPER_PLUNGE_EXIT = 4
BH_CPU_SURFACE_SUBSEQUENT_LOWER_CONTACT = 5
BH_CPU_SURFACE_SUBSEQUENT_UPPER_CONTACT = 6
BH_CPU_SURFACE_INWARD_LOWER_PLUNGE_ENTRY = 7
BH_CPU_SURFACE_INWARD_UPPER_PLUNGE_ENTRY = 8

BH_CPU_RAY_FAILURE_NONE = 0

DoublePointer = ctypes.POINTER(ctypes.c_double)


class RuntimeContract(ctypes.Structure):
    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("double_size", ctypes.c_uint32),
        ("double_mantissa_bits", ctypes.c_uint32),
        ("float_radix", ctypes.c_uint32),
        ("float_evaluation_method", ctypes.c_int32),
        ("rounding_mode", ctypes.c_int32),
        ("iec_60559_binary64", ctypes.c_uint32),
        ("fast_math_enabled", ctypes.c_uint32),
        ("little_endian", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    )


class KerrRayOptions(ctypes.Structure):
    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("absolute_tolerance", ctypes.c_double),
        ("relative_tolerance", ctypes.c_double),
        ("initial_step", ctypes.c_double),
        ("minimum_step", ctypes.c_double),
        ("maximum_step", ctypes.c_double),
        ("maximum_affine_length", ctypes.c_double),
        ("null_residual_limit", ctypes.c_double),
        ("metric_interpolation_error_limit", ctypes.c_double),
        ("event_value_tolerance", ctypes.c_double),
        ("event_affine_tolerance", ctypes.c_double),
        ("maximum_accepted_steps", ctypes.c_uint64),
        ("maximum_rejected_steps", ctypes.c_uint64),
        ("event_maximum_iterations", ctypes.c_uint32),
        ("record_path", ctypes.c_uint32),
    )


class KerrSurfaceOptions(ctypes.Structure):
    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("absolute_tolerance", ctypes.c_double),
        ("relative_tolerance", ctypes.c_double),
        ("null_residual_limit", ctypes.c_double),
        ("metric_interpolation_error_limit", ctypes.c_double),
        ("surface_value_tolerance", ctypes.c_double),
        ("affine_tolerance", ctypes.c_double),
        ("maximum_reintegrations", ctypes.c_uint64),
        ("maximum_iterations", ctypes.c_uint32),
        ("subdivisions_per_segment", ctypes.c_uint32),
    )


class KerrReturningModel(ctypes.Structure):
    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("mass_m", ctypes.c_double),
        ("spin_a_m", ctypes.c_double),
        ("singularity_guard_m", ctypes.c_double),
        ("capture_radius_m", ctypes.c_double),
        ("escape_radius_m", ctypes.c_double),
        ("isco_radius_over_mass", ctypes.c_double),
        ("outer_radius_over_mass", ctypes.c_double),
        ("asymptotic_pressure_scale_height_over_mass", ctypes.c_double),
        ("emitting_surface_id", ctypes.c_uint32),
        ("initial_contact_side", ctypes.c_int32),
    )


class RayPathSegment(ctypes.Structure):
    _fields_ = (
        ("start", ctypes.c_double * 8),
        ("end", ctypes.c_double * 8),
        ("midpoint", ctypes.c_double * 8),
        ("affine_length", ctypes.c_double),
        ("midpoint_null_residual", ctypes.c_double),
    )


class RecordedSurfaceCrossing(ctypes.Structure):
    _fields_ = (
        ("state", ctypes.c_double * 8),
        ("ray_affine_length", ctypes.c_double),
        ("segment_affine_length", ctypes.c_double),
        ("surface_value", ctypes.c_double),
        ("bracket_affine_width", ctypes.c_double),
        ("segment_index", ctypes.c_uint64),
        ("iterations", ctypes.c_uint32),
        ("orientation", ctypes.c_int32),
        ("surface_id", ctypes.c_uint32),
        ("classification", ctypes.c_uint32),
        ("outcome", ctypes.c_uint32),
        ("target", ctypes.c_uint32),
    )


class KerrRayResult(ctypes.Structure):
    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("outcome", ctypes.c_uint32),
        ("terminal_target", ctypes.c_uint32),
        ("failure", ctypes.c_uint32),
        ("topology_converged", ctypes.c_uint32),
        ("initial_contact_surface_id", ctypes.c_uint32),
        ("initial_contact_side", ctypes.c_int32),
        ("terminal_state", ctypes.c_double * 8),
        ("affine_length", ctypes.c_double),
        ("maximum_null_residual", ctypes.c_double),
        ("maximum_metric_interpolation_error", ctypes.c_double),
        ("maximum_probe_event_difference", ctypes.c_double),
        ("maximum_probe_covector_relative_difference", ctypes.c_double),
        ("initial_contact_actual_surface_value", ctypes.c_double),
        ("initial_contact_surface_value_tolerance", ctypes.c_double),
        ("accepted_steps", ctypes.c_uint64),
        ("rejected_steps", ctypes.c_uint64),
        ("segment_count", ctypes.c_uint64),
        ("crossing_count", ctypes.c_uint64),
        ("probe_reintegrations", ctypes.c_uint64),
        ("surface_value_evaluations", ctypes.c_uint64),
        ("metric_sample_evaluations", ctypes.c_uint64),
        ("hamiltonian_rhs_evaluations", ctypes.c_uint64),
    )


def double_array(values: Iterable[float]) -> ctypes.Array[ctypes.c_double]:
    sequence = tuple(values)
    return (ctypes.c_double * len(sequence))(*sequence)


def flatten_matrix(matrix: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(value for row in matrix for value in row)


def flatten_derivatives(
    derivatives: Sequence[Sequence[Sequence[float]]],
) -> tuple[float, ...]:
    return tuple(
        value for derivative in derivatives for row in derivative for value in row
    )


def load_library() -> ctypes.CDLL:
    library = ctypes.CDLL(str(LIBRARY_PATH))
    library.bh_cpu_get_runtime_contract.argtypes = (
        ctypes.POINTER(RuntimeContract),
    )
    library.bh_cpu_get_runtime_contract.restype = ctypes.c_int
    library.bh_cpu_require_strict_fp.argtypes = ()
    library.bh_cpu_require_strict_fp.restype = ctypes.c_int
    library.bh_cpu_fsum.argtypes = (
        DoublePointer,
        ctypes.c_size_t,
        DoublePointer,
    )
    library.bh_cpu_fsum.restype = ctypes.c_int
    library.bh_cpu_matrix4_vector4.argtypes = (
        DoublePointer,
        DoublePointer,
        DoublePointer,
    )
    library.bh_cpu_matrix4_vector4.restype = ctypes.c_int
    library.bh_cpu_bilinear4.argtypes = (
        DoublePointer,
        DoublePointer,
        DoublePointer,
        DoublePointer,
    )
    library.bh_cpu_bilinear4.restype = ctypes.c_int
    library.bh_cpu_metric_algebra_audit.argtypes = (
        DoublePointer,
        DoublePointer,
        DoublePointer,
        ctypes.c_double,
        ctypes.c_double,
    )
    library.bh_cpu_metric_algebra_audit.restype = ctypes.c_int
    library.bh_cpu_metric_sample_audit.argtypes = (
        DoublePointer,
        DoublePointer,
        DoublePointer,
        ctypes.c_double,
        ctypes.c_double,
    )
    library.bh_cpu_metric_sample_audit.restype = ctypes.c_int
    library.bh_cpu_kerr_metric_sample.argtypes = (
        DoublePointer,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        DoublePointer,
        DoublePointer,
        DoublePointer,
    )
    library.bh_cpu_kerr_metric_sample.restype = ctypes.c_int
    library.bh_cpu_normalized_null_residual.argtypes = (
        DoublePointer,
        DoublePointer,
        DoublePointer,
    )
    library.bh_cpu_normalized_null_residual.restype = ctypes.c_int
    library.bh_cpu_hamiltonian_rhs.argtypes = (
        DoublePointer,
        DoublePointer,
        DoublePointer,
        DoublePointer,
    )
    library.bh_cpu_hamiltonian_rhs.restype = ctypes.c_int
    library.bh_cpu_hamiltonian_rhs_batch.argtypes = (
        DoublePointer,
        DoublePointer,
        DoublePointer,
        ctypes.c_size_t,
        DoublePointer,
        ctypes.POINTER(ctypes.c_size_t),
    )
    library.bh_cpu_hamiltonian_rhs_batch.restype = ctypes.c_int
    library.bh_cpu_kerr_hamiltonian_rhs.argtypes = (
        DoublePointer,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        DoublePointer,
    )
    library.bh_cpu_kerr_hamiltonian_rhs.restype = ctypes.c_int
    library.bh_cpu_kerr_dopri54_step.argtypes = (
        DoublePointer,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        DoublePointer,
        DoublePointer,
    )
    library.bh_cpu_kerr_dopri54_step.restype = ctypes.c_int
    library.bh_cpu_kerr_dopri54_probe_batch.argtypes = (
        DoublePointer,
        DoublePointer,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        DoublePointer,
        DoublePointer,
        ctypes.POINTER(ctypes.c_size_t),
    )
    library.bh_cpu_kerr_dopri54_probe_batch.restype = ctypes.c_int
    library.bh_cpu_trace_kerr_returning_ray.argtypes = (
        DoublePointer,
        ctypes.POINTER(KerrReturningModel),
        ctypes.POINTER(KerrRayOptions),
        ctypes.POINTER(KerrSurfaceOptions),
        ctypes.POINTER(RayPathSegment),
        ctypes.c_size_t,
        ctypes.POINTER(RecordedSurfaceCrossing),
        ctypes.c_size_t,
        ctypes.POINTER(KerrRayResult),
    )
    library.bh_cpu_trace_kerr_returning_ray.restype = ctypes.c_int
    library.bh_cpu_status_string.argtypes = (ctypes.c_int,)
    library.bh_cpu_status_string.restype = ctypes.c_char_p
    return library
