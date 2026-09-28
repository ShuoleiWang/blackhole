#include "blackhole_cpu.h"

#include <assert.h>
#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON

static void
test_runtime_contract(void)
{
    bh_cpu_runtime_contract contract;
    assert(bh_cpu_get_runtime_contract(&contract) == BH_CPU_OK);
    assert(contract.abi_version == BH_CPU_ABI_VERSION);
    assert(contract.struct_size == sizeof(contract));
    assert(contract.double_size == 8);
    assert(contract.double_mantissa_bits == 53);
    assert(contract.float_radix == 2);
    assert(contract.float_evaluation_method == 0);
    assert(contract.fast_math_enabled == 0);
    assert(contract.iec_60559_binary64 == 1);
    assert(bh_cpu_require_strict_fp() == BH_CPU_OK);
}

static void
test_rounding_mode_rejection(void)
{
    const int original_rounding_mode = fegetround();
    assert(original_rounding_mode == FE_TONEAREST);
    assert(fesetround(FE_DOWNWARD) == 0);
    assert(bh_cpu_require_strict_fp() == BH_CPU_INVALID_FP_ENVIRONMENT);
    const double event[4] = {0.0, 8.0, -1.5, 2.5};
    const double state[8] = {
        0.0, 8.0, -1.5, 2.5, -1.0, 0.15, -0.25, 0.35,
    };
    double covariant[16];
    double inverse[16];
    double derivatives[64];
    double fifth[8];
    double error[8];
    assert(bh_cpu_kerr_metric_sample(
               event,
               1.0,
               0.7,
               1.0e-9,
               covariant,
               inverse,
               derivatives) == BH_CPU_INVALID_FP_ENVIRONMENT);
    assert(bh_cpu_kerr_dopri54_step(
               state,
               0.025,
               1.0,
               0.7,
               1.0e-9,
               fifth,
               error) == BH_CPU_INVALID_FP_ENVIRONMENT);
    assert(fesetround(original_rounding_mode) == 0);
    assert(bh_cpu_require_strict_fp() == BH_CPU_OK);
}

static void
test_gradual_underflow_rejection(void)
{
#if defined(__aarch64__)
    uint64_t original_fpcr = UINT64_C(0);
    __asm__ volatile("mrs %0, fpcr" : "=r"(original_fpcr));
    const uint64_t flushed_fpcr =
        original_fpcr | (UINT64_C(1) << UINT64_C(24));
    __asm__ volatile("msr fpcr, %0" : : "r"(flushed_fpcr));
    __asm__ volatile("isb");
    assert(bh_cpu_require_strict_fp() == BH_CPU_INVALID_FP_ENVIRONMENT);
    __asm__ volatile("msr fpcr, %0" : : "r"(original_fpcr));
    __asm__ volatile("isb");
#elif defined(__x86_64__) || defined(__i386__)
    const unsigned int original_mxcsr = _mm_getcsr();
    _mm_setcsr(original_mxcsr | UINT32_C(0x00008040));
    assert(bh_cpu_require_strict_fp() == BH_CPU_INVALID_FP_ENVIRONMENT);
    _mm_setcsr(original_mxcsr);
#endif
    assert(bh_cpu_require_strict_fp() == BH_CPU_OK);
}

static void
test_status_and_argument_failures(void)
{
    double output = 0.0;
    const double nonfinite[] = {1.0, HUGE_VAL};
    assert(bh_cpu_fsum(NULL, 1, &output) == BH_CPU_INVALID_ARGUMENT);
    assert(bh_cpu_fsum(nonfinite, 2, &output) == BH_CPU_NONFINITE_INPUT);
    assert(bh_cpu_fsum(NULL, 0, &output) == BH_CPU_OK);
    assert(output == 0.0);
    assert(bh_cpu_status_string(BH_CPU_OK) != NULL);
    assert(bh_cpu_status_string(BH_CPU_GUARDED_SINGULARITY) != NULL);
}

static void
test_kerr_and_dopri_contract(void)
{
    const double event[4] = {0.0, 8.0, -1.5, 2.5};
    const double state[8] = {
        0.0, 8.0, -1.5, 2.5, -1.0, 0.15, -0.25, 0.35,
    };
    double covariant[16];
    double inverse[16];
    double derivatives[64];
    assert(bh_cpu_kerr_metric_sample(
               event,
               1.0,
               0.7,
               1.0e-9,
               covariant,
               inverse,
               derivatives) == BH_CPU_OK);
    assert(bh_cpu_metric_sample_audit(
               covariant,
               inverse,
               derivatives,
               2.0e-10,
               2.0e-12) == BH_CPU_OK);

    double rhs[8];
    double fifth[8];
    double error[8];
    assert(bh_cpu_kerr_hamiltonian_rhs(
               state, 1.0, 0.7, 1.0e-9, rhs) == BH_CPU_OK);
    assert(bh_cpu_kerr_dopri54_step(
               state,
               0.025,
               1.0,
               0.7,
               1.0e-9,
               fifth,
               error) == BH_CPU_OK);

    const double offsets[4] = {0.0, 0.001, 0.0125, 0.025};
    double fifth_records[32];
    double error_records[32];
    size_t completed = 99;
    assert(bh_cpu_kerr_dopri54_probe_batch(
               state,
               offsets,
               4,
               1.0,
               0.7,
               1.0e-9,
               fifth_records,
               error_records,
               &completed) == BH_CPU_OK);
    assert(completed == 4);

    const double guarded_event[4] = {0.0, 0.0, 0.0, 0.0};
    assert(bh_cpu_kerr_metric_sample(
               guarded_event,
               1.0,
               0.7,
               1.0e-9,
               covariant,
               inverse,
               derivatives) == BH_CPU_GUARDED_SINGULARITY);
}

static void
test_returning_whole_ray_contract(void)
{
    /* Authenticated nested16 ordinal 75785 fine-resolution launch. */
    const double initial_state[8] = {
        0.0,
        19.8741198679998,
        0.6993612853398631,
        0.8495807960850076,
        -1.1594875240991946,
        0.7152930947421235,
        1.0889771608731338,
        0.01792675599391145,
    };
    const bh_cpu_kerr_returning_model model = {
        .abi_version = BH_CPU_ABI_VERSION,
        .struct_size = (uint32_t)sizeof(bh_cpu_kerr_returning_model),
        .mass_m = 1.0,
        .spin_a_m = 0.7,
        .singularity_guard_m = 1.0e-9,
        .capture_radius_m = 1.7341428428542849,
        .escape_radius_m = 50.0,
        .isco_radius_over_mass = 3.3931284701816304,
        .outer_radius_over_mass = 25.0,
        .asymptotic_pressure_scale_height_over_mass = 0.7239051887604214,
        .emitting_surface_id = BH_CPU_SURFACE_UPPER,
        .initial_contact_side = 1,
    };
    const bh_cpu_kerr_ray_options ray = {
        .abi_version = BH_CPU_ABI_VERSION,
        .struct_size = (uint32_t)sizeof(bh_cpu_kerr_ray_options),
        .absolute_tolerance = 6.25e-11,
        .relative_tolerance = 6.25e-11,
        .initial_step = 0.025,
        .minimum_step = 1.25e-9,
        .maximum_step = 0.125,
        .maximum_affine_length = 300.0,
        .null_residual_limit = 2.5e-8,
        .metric_interpolation_error_limit = 1.25e-8,
        .event_value_tolerance = 1.25e-10,
        .event_affine_tolerance = 1.25e-11,
        .maximum_accepted_steps = UINT64_C(100000),
        .maximum_rejected_steps = UINT64_C(100000),
        .event_maximum_iterations = UINT32_C(64),
        .record_path = UINT32_C(0),
    };
    const bh_cpu_kerr_surface_options surface = {
        .abi_version = BH_CPU_ABI_VERSION,
        .struct_size = (uint32_t)sizeof(bh_cpu_kerr_surface_options),
        .absolute_tolerance = 7.8125e-12,
        .relative_tolerance = 7.8125e-12,
        .null_residual_limit = 3.125e-9,
        .metric_interpolation_error_limit = 1.5625e-9,
        .surface_value_tolerance = 1.5625e-11,
        .affine_tolerance = 1.5625e-12,
        .maximum_reintegrations = UINT64_C(100000),
        .maximum_iterations = UINT32_C(64),
        .subdivisions_per_segment = UINT32_C(16),
    };
    bh_cpu_recorded_surface_crossing crossings[4];
    bh_cpu_kerr_ray_result result;
    assert(bh_cpu_trace_kerr_returning_ray(
               initial_state,
               &model,
               &ray,
               &surface,
               NULL,
               0,
               crossings,
               4,
               &result) == BH_CPU_OK);
    assert(result.abi_version == BH_CPU_ABI_VERSION);
    assert(result.struct_size == sizeof(result));
    assert(result.outcome == BH_CPU_RAY_OUTCOME_RETURNED);
    assert(result.terminal_target == BH_CPU_TARGET_OPAQUE_UPPER_FACE);
    assert(result.failure == BH_CPU_RAY_FAILURE_NONE);
    assert(result.topology_converged == 1);
    assert(result.accepted_steps == UINT64_C(59));
    assert(result.rejected_steps == UINT64_C(0));
    assert(result.segment_count == UINT64_C(0));
    assert(result.crossing_count == UINT64_C(1));
    assert(crossings[0].surface_id == BH_CPU_SURFACE_UPPER);
    assert(crossings[0].classification ==
           BH_CPU_SURFACE_SUBSEQUENT_UPPER_CONTACT);

    bh_cpu_kerr_ray_options recorded_ray = ray;
    recorded_ray.record_path = UINT32_C(1);
    bh_cpu_kerr_ray_result capacity_result;
    bh_cpu_kerr_ray_result unchanged_result;
    memset(&capacity_result, 0xa5, sizeof(capacity_result));
    memcpy(&unchanged_result, &capacity_result, sizeof(unchanged_result));
    assert(bh_cpu_trace_kerr_returning_ray(
               initial_state,
               &model,
               &recorded_ray,
               &surface,
               NULL,
               0,
               NULL,
               0,
               &capacity_result) == BH_CPU_CAPACITY_EXCEEDED);
    assert(memcmp(
               &capacity_result,
               &unchanged_result,
               sizeof(capacity_result)) == 0);
}

int
main(void)
{
    test_runtime_contract();
    test_rounding_mode_rejection();
    test_gradual_underflow_rejection();
    test_status_and_argument_failures();
    test_kerr_and_dopri_contract();
    test_returning_whole_ray_contract();
    puts("strict CPU contract tests passed");
    return 0;
}
