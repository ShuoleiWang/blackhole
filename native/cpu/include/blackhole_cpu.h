#ifndef BLACKHOLE_NATIVE_CPU_H
#define BLACKHOLE_NATIVE_CPU_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define BH_CPU_API __declspec(dllexport)
#else
#define BH_CPU_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

/* Increment this whenever a public layout or numerical contract changes. */
#define BH_CPU_ABI_VERSION UINT32_C(3)
#define BH_CPU_FSUM_MAX_TERMS ((size_t)128)

typedef enum bh_cpu_status {
    BH_CPU_OK = 0,
    BH_CPU_INVALID_ARGUMENT = 1,
    BH_CPU_INVALID_FP_ENVIRONMENT = 2,
    BH_CPU_NONFINITE_INPUT = 3,
    BH_CPU_NUMERIC_OVERFLOW = 4,
    BH_CPU_INCONSISTENT_METRIC = 5,
    BH_CPU_TOO_MANY_TERMS = 6,
    BH_CPU_GUARDED_SINGULARITY = 7,
    BH_CPU_CAPACITY_EXCEEDED = 8
} bh_cpu_status;

/* Exact product-owned outcomes/targets for the one-resolution returning ray. */
typedef enum bh_cpu_ray_outcome {
    BH_CPU_RAY_OUTCOME_NONE = 0,
    BH_CPU_RAY_OUTCOME_CAPTURED = 1,
    BH_CPU_RAY_OUTCOME_ESCAPED = 2,
    BH_CPU_RAY_OUTCOME_RETURNED = 3,
    BH_CPU_RAY_OUTCOME_PLUNGE_SINK = 4,
    BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE = 5,
    BH_CPU_RAY_OUTCOME_UNRESOLVED = 6,
    BH_CPU_RAY_OUTCOME_COMPLETED = 7
} bh_cpu_ray_outcome;

typedef enum bh_cpu_terminal_target {
    BH_CPU_TARGET_NONE = 0,
    BH_CPU_TARGET_KERR_STRETCHED_HORIZON = 1,
    BH_CPU_TARGET_KERR_ESCAPE_WORLDTUBE = 2,
    BH_CPU_TARGET_OPAQUE_LOWER_FACE = 3,
    BH_CPU_TARGET_OPAQUE_UPPER_FACE = 4,
    BH_CPU_TARGET_LOWER_PLUNGE_ENTRY = 5,
    BH_CPU_TARGET_UPPER_PLUNGE_ENTRY = 6
} bh_cpu_terminal_target;

typedef enum bh_cpu_surface_id {
    BH_CPU_SURFACE_LOWER = 0,
    BH_CPU_SURFACE_UPPER = 1
} bh_cpu_surface_id;

typedef enum bh_cpu_surface_classification {
    BH_CPU_SURFACE_CLASSIFICATION_NONE = 0,
    BH_CPU_SURFACE_INSIDE_ISCO_TRANSPARENT = 1,
    BH_CPU_SURFACE_OUTSIDE_OUTER_TRANSPARENT = 2,
    BH_CPU_SURFACE_OUTWARD_LOWER_PLUNGE_EXIT = 3,
    BH_CPU_SURFACE_OUTWARD_UPPER_PLUNGE_EXIT = 4,
    BH_CPU_SURFACE_SUBSEQUENT_LOWER_CONTACT = 5,
    BH_CPU_SURFACE_SUBSEQUENT_UPPER_CONTACT = 6,
    BH_CPU_SURFACE_INWARD_LOWER_PLUNGE_ENTRY = 7,
    BH_CPU_SURFACE_INWARD_UPPER_PLUNGE_ENTRY = 8
} bh_cpu_surface_classification;

typedef struct bh_cpu_runtime_contract {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t double_size;
    uint32_t double_mantissa_bits;
    uint32_t float_radix;
    int32_t float_evaluation_method;
    int32_t rounding_mode;
    uint32_t iec_60559_binary64;
    uint32_t fast_math_enabled;
    uint32_t little_endian;
    uint32_t reserved;
} bh_cpu_runtime_contract;

/* Layout-checked counterparts of RayTraceOptions and SurfaceEventOptions. */
typedef struct bh_cpu_kerr_ray_options {
    uint32_t abi_version;
    uint32_t struct_size;
    double absolute_tolerance;
    double relative_tolerance;
    double initial_step;
    double minimum_step;
    double maximum_step;
    double maximum_affine_length;
    double null_residual_limit;
    double metric_interpolation_error_limit;
    double event_value_tolerance;
    double event_affine_tolerance;
    uint64_t maximum_accepted_steps;
    uint64_t maximum_rejected_steps;
    uint32_t event_maximum_iterations;
    uint32_t record_path;
} bh_cpu_kerr_ray_options;

typedef struct bh_cpu_kerr_surface_options {
    uint32_t abi_version;
    uint32_t struct_size;
    double absolute_tolerance;
    double relative_tolerance;
    double null_residual_limit;
    double metric_interpolation_error_limit;
    double surface_value_tolerance;
    double affine_tolerance;
    uint64_t maximum_reintegrations;
    uint32_t maximum_iterations;
    uint32_t subdivisions_per_segment;
} bh_cpu_kerr_surface_options;

/*
 * Exact analytic Kerr + finite-thickness product parameters.  The caller
 * supplies the already-authenticated calibration scalars so this isolated
 * core does not duplicate Novikov--Thorne policy construction.
 */
typedef struct bh_cpu_kerr_returning_model {
    uint32_t abi_version;
    uint32_t struct_size;
    double mass_m;
    double spin_a_m;
    double singularity_guard_m;
    double capture_radius_m;
    double escape_radius_m;
    double isco_radius_over_mass;
    double outer_radius_over_mass;
    double asymptotic_pressure_scale_height_over_mass;
    uint32_t emitting_surface_id;
    int32_t initial_contact_side;
} bh_cpu_kerr_returning_model;

typedef struct bh_cpu_ray_path_segment {
    double start[8];
    double end[8];
    double midpoint[8];
    double affine_length;
    double midpoint_null_residual;
} bh_cpu_ray_path_segment;

typedef struct bh_cpu_recorded_surface_crossing {
    double state[8];
    double ray_affine_length;
    double segment_affine_length;
    double surface_value;
    double bracket_affine_width;
    uint64_t segment_index;
    uint32_t iterations;
    int32_t orientation;
    uint32_t surface_id;
    uint32_t classification;
    uint32_t outcome;
    uint32_t target;
} bh_cpu_recorded_surface_crossing;

typedef enum bh_cpu_ray_failure {
    BH_CPU_RAY_FAILURE_NONE = 0,
    BH_CPU_RAY_FAILURE_INITIAL_NULL = 1,
    BH_CPU_RAY_FAILURE_MINIMUM_STEP = 2,
    BH_CPU_RAY_FAILURE_ACCEPTED_BUDGET = 3,
    BH_CPU_RAY_FAILURE_REJECTED_BUDGET = 4,
    BH_CPU_RAY_FAILURE_EVENT_REFINEMENT_BUDGET = 5,
    BH_CPU_RAY_FAILURE_SURFACE_REFINEMENT_BUDGET = 6,
    BH_CPU_RAY_FAILURE_SURFACE = 7,
    BH_CPU_RAY_FAILURE_NUMERIC = 8,
    BH_CPU_RAY_FAILURE_NULL_LIMIT = 9,
    BH_CPU_RAY_FAILURE_AFFINE_BUDGET = 10
} bh_cpu_ray_failure;

typedef struct bh_cpu_kerr_ray_result {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t outcome;
    uint32_t terminal_target;
    uint32_t failure;
    uint32_t topology_converged;
    uint32_t initial_contact_surface_id;
    int32_t initial_contact_side;
    double terminal_state[8];
    double affine_length;
    double maximum_null_residual;
    double maximum_metric_interpolation_error;
    double maximum_probe_event_difference;
    double maximum_probe_covector_relative_difference;
    double initial_contact_actual_surface_value;
    double initial_contact_surface_value_tolerance;
    uint64_t accepted_steps;
    uint64_t rejected_steps;
    uint64_t segment_count;
    uint64_t crossing_count;
    uint64_t probe_reintegrations;
    uint64_t surface_value_evaluations;
    uint64_t metric_sample_evaluations;
    uint64_t hamiltonian_rhs_evaluations;
} bh_cpu_kerr_ray_result;

/*
 * Inspect, but never modify, the calling thread's floating-point environment.
 * Every numerical entry point rejects anything except round-to-nearest/even.
 */
BH_CPU_API bh_cpu_status
bh_cpu_get_runtime_contract(bh_cpu_runtime_contract *out_contract);

BH_CPU_API bh_cpu_status bh_cpu_require_strict_fp(void);

/* Finite-input counterpart of CPython 3.12 math.fsum. */
BH_CPU_API bh_cpu_status
bh_cpu_fsum(const double *values, size_t count, double *out_sum);

/* Row-major matrix helpers matching offline.spacetime evaluation order. */
BH_CPU_API bh_cpu_status
bh_cpu_matrix4_vector4(
    const double matrix[16],
    const double vector[4],
    double out_vector[4]
);

BH_CPU_API bh_cpu_status
bh_cpu_bilinear4(
    const double left[4],
    const double matrix[16],
    const double right[4],
    double *out_value
);

/*
 * Audit the symmetry and covariant/inverse product gates used by MetricSample.
 * This algebra-only compatibility entry point deliberately does not imply the
 * Lorentzian-signature certification provided by bh_cpu_metric_sample_audit.
 */
BH_CPU_API bh_cpu_status
bh_cpu_metric_algebra_audit(
    const double covariant[16],
    const double inverse[16],
    const double inverse_derivatives[64],
    double relative_tolerance,
    double absolute_tolerance
);

/*
 * Complete generic MetricSample audit.  This preserves the algebra gates
 * above and additionally certifies one negative and three positive
 * eigenvalues with the reference Jacobi-sweep contract.
 */
BH_CPU_API bh_cpu_status
bh_cpu_metric_sample_audit(
    const double covariant[16],
    const double inverse[16],
    const double inverse_derivatives[64],
    double relative_tolerance,
    double absolute_tolerance
);

/*
 * Exact analytic ingoing-Cartesian Kerr--Schild MetricSample construction.
 * Layouts are row-major and derivatives are [coordinate][row][column].  The
 * result is returned only after the complete generic MetricSample audit.
 */
BH_CPU_API bh_cpu_status
bh_cpu_kerr_metric_sample(
    const double event[4],
    double mass_m,
    double spin_a_m,
    double singularity_guard_m,
    double out_covariant[16],
    double out_inverse[16],
    double out_inverse_derivatives[64]
);

/* Match offline.geodesic._normalized_null_residual_from_sample. */
BH_CPU_API bh_cpu_status
bh_cpu_normalized_null_residual(
    const double inverse[16],
    const double covector[4],
    double *out_residual
);

/*
 * Hamiltonian RHS from an already authenticated MetricSample.  The derivative
 * layout is [coordinate][row][column], all row-major.
 */
BH_CPU_API bh_cpu_status
bh_cpu_hamiltonian_rhs(
    const double inverse[16],
    const double inverse_derivatives[64],
    const double covector[4],
    double out_rhs[8]
);

/* Structure-of-arrays-by-record batch form; no Python callback or allocation. */
BH_CPU_API bh_cpu_status
bh_cpu_hamiltonian_rhs_batch(
    const double *inverse_records,
    const double *inverse_derivative_records,
    const double *covector_records,
    size_t record_count,
    double *out_rhs_records,
    size_t *out_completed_records
);

/* Hamiltonian RHS including exact, generically audited Kerr construction. */
BH_CPU_API bh_cpu_status
bh_cpu_kerr_hamiltonian_rhs(
    const double state[8],
    double mass_m,
    double spin_a_m,
    double singularity_guard_m,
    double out_rhs[8]
);

/*
 * One canonical Dormand--Prince 5(4) step over the exact Kerr Hamiltonian.
 * Both returned states match offline.geodesic._dormand_prince_step byte for
 * byte under the declared strict floating-point contract.
 */
BH_CPU_API bh_cpu_status
bh_cpu_kerr_dopri54_step(
    const double state[8],
    double step,
    double mass_m,
    double spin_a_m,
    double singularity_guard_m,
    double out_fifth[8],
    double out_error[8]
);

/*
 * Independently reintegrate every affine offset from one common start.  This
 * is the allocation-free boundary needed by N/2N accepted-step probe grids;
 * it does not introduce dense-output interpolation or share stage states.
 */
BH_CPU_API bh_cpu_status
bh_cpu_kerr_dopri54_probe_batch(
    const double state[8],
    const double *affine_offsets,
    size_t probe_count,
    double mass_m,
    double spin_a_m,
    double singularity_guard_m,
    double *out_fifth_records,
    double *out_error_records,
    size_t *out_completed_probes
);

/*
 * Complete one-resolution returning-radiation whole ray.  The routine owns
 * the adaptive controller, Kerr-oblate termination localization, shared
 * two-face N/2N surface topology, audits, and path/crossing serialization.
 * The initial state must be an authenticated physical emitter contact with
 * ISCO < rho <= outer radius and must lie strictly between both worldtubes;
 * auxiliary continuation contacts and initial terminal states are rejected.
 * Caller-owned arrays make allocation ownership explicit; capacity failure
 * is fail-closed and never publishes a partial result.
 */
BH_CPU_API bh_cpu_status
bh_cpu_trace_kerr_returning_ray(
    const double initial_state[8],
    const bh_cpu_kerr_returning_model *model,
    const bh_cpu_kerr_ray_options *ray_options,
    const bh_cpu_kerr_surface_options *surface_options,
    bh_cpu_ray_path_segment *out_segments,
    size_t segment_capacity,
    bh_cpu_recorded_surface_crossing *out_crossings,
    size_t crossing_capacity,
    bh_cpu_kerr_ray_result *out_result
);

BH_CPU_API const char *bh_cpu_status_string(bh_cpu_status status);

#ifdef __cplusplus
}
#endif

#endif
