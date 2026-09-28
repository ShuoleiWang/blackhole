#include "blackhole_cpu.h"

#include <float.h>
#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FENV_ACCESS ON
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#pragma clang fp exceptions(strict)
#endif

_Static_assert(CHAR_BIT == 8, "blackhole native ABI requires 8-bit bytes");
_Static_assert(sizeof(double) == 8, "blackhole native ABI requires binary64");
_Static_assert(DBL_MANT_DIG == 53, "blackhole native ABI requires 53-bit double");
_Static_assert(FLT_RADIX == 2, "blackhole native ABI requires radix-2 floats");
_Static_assert(DBL_MAX_EXP == 1024, "blackhole native ABI requires binary64 max");
_Static_assert(DBL_MIN_EXP == -1021, "blackhole native ABI requires binary64 min");
_Static_assert(DBL_HAS_SUBNORM == 1,
               "blackhole native ABI requires gradual underflow");
_Static_assert(sizeof(bh_cpu_runtime_contract) == 44,
               "runtime-contract ABI layout changed");
_Static_assert(sizeof(bh_cpu_kerr_ray_options) == 112,
               "ray-options ABI layout changed");
_Static_assert(sizeof(bh_cpu_kerr_surface_options) == 72,
               "surface-options ABI layout changed");
_Static_assert(sizeof(bh_cpu_kerr_returning_model) == 80,
               "returning-model ABI layout changed");
_Static_assert(sizeof(bh_cpu_ray_path_segment) == 208,
               "ray-segment ABI layout changed");
_Static_assert(sizeof(bh_cpu_recorded_surface_crossing) == 128,
               "surface-crossing ABI layout changed");
_Static_assert(sizeof(bh_cpu_kerr_ray_result) == 216,
               "ray-result ABI layout changed");

static bool
bh_is_little_endian(void)
{
    const uint16_t marker = UINT16_C(1);
    uint8_t first = 0;
    memcpy(&first, &marker, sizeof(first));
    return first == UINT8_C(1);
}

static bool
bh_all_finite(const double *values, size_t count)
{
    if (values == NULL) {
        return false;
    }
    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

static bool
bh_runtime_has_gradual_underflow(void)
{
    const uint64_t smallest_subnormal_bits = UINT64_C(0x0000000000000001);
    double smallest_subnormal = 0.0;
    memcpy(
        &smallest_subnormal,
        &smallest_subnormal_bits,
        sizeof(smallest_subnormal));

    volatile double subnormal_operand = smallest_subnormal;
    volatile double doubled_subnormal =
        subnormal_operand + subnormal_operand;
    volatile double minimum_normal_operand = DBL_MIN;
    volatile double halved_minimum_normal =
        minimum_normal_operand * 0.5;

    const double doubled_snapshot = doubled_subnormal;
    const double halved_snapshot = halved_minimum_normal;
    uint64_t doubled_bits = UINT64_C(0);
    uint64_t halved_bits = UINT64_C(0);
    memcpy(&doubled_bits, &doubled_snapshot, sizeof(doubled_bits));
    memcpy(&halved_bits, &halved_snapshot, sizeof(halved_bits));
    return doubled_bits == UINT64_C(0x0000000000000002) &&
        halved_bits == UINT64_C(0x0008000000000000);
}

bh_cpu_status
bh_cpu_get_runtime_contract(bh_cpu_runtime_contract *out_contract)
{
    if (out_contract == NULL) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    *out_contract = (bh_cpu_runtime_contract){
        .abi_version = BH_CPU_ABI_VERSION,
        .struct_size = (uint32_t)sizeof(bh_cpu_runtime_contract),
        .double_size = (uint32_t)sizeof(double),
        .double_mantissa_bits = (uint32_t)DBL_MANT_DIG,
        .float_radix = (uint32_t)FLT_RADIX,
        .float_evaluation_method = (int32_t)FLT_EVAL_METHOD,
        .rounding_mode = (int32_t)fegetround(),
        /* Exponents are compile-time checked; extrema complete the proof. */
        .iec_60559_binary64 =
            (DBL_MAX == 0x1.fffffffffffffp+1023 &&
             DBL_TRUE_MIN == 0x0.0000000000001p-1022 &&
             isinf(HUGE_VAL))
                ? UINT32_C(1)
                : UINT32_C(0),
#if defined(__FAST_MATH__)
        .fast_math_enabled = UINT32_C(1),
#else
        .fast_math_enabled = UINT32_C(0),
#endif
        .little_endian = bh_is_little_endian() ? UINT32_C(1) : UINT32_C(0),
        .reserved = UINT32_C(0),
    };
    return BH_CPU_OK;
}

bh_cpu_status
bh_cpu_require_strict_fp(void)
{
    bh_cpu_runtime_contract contract;
    const bh_cpu_status status = bh_cpu_get_runtime_contract(&contract);
    if (status != BH_CPU_OK) {
        return status;
    }
    if (contract.abi_version != BH_CPU_ABI_VERSION ||
        contract.struct_size != sizeof(bh_cpu_runtime_contract) ||
        contract.double_size != 8 ||
        contract.double_mantissa_bits != 53 ||
        contract.float_radix != 2 ||
        contract.float_evaluation_method != 0 ||
        contract.rounding_mode != FE_TONEAREST ||
        contract.iec_60559_binary64 != UINT32_C(1) ||
        contract.fast_math_enabled != UINT32_C(0) ||
        contract.little_endian != UINT32_C(1) ||
        !bh_runtime_has_gradual_underflow()) {
        return BH_CPU_INVALID_FP_ENVIRONMENT;
    }
    return BH_CPU_OK;
}

/*
 * Shewchuk-style partial expansion and final half-even fixup, adapted from the
 * finite-input path of CPython 3.12 math.fsum (PSF-2.0; see
 * THIRD_PARTY_NOTICES.md).  Intermediates are intentionally named so strict
 * compilation rounds every primitive binary64 operation.
 */
static bh_cpu_status
bh_fsum_impl(const double *values, size_t count, double *out_sum)
{
    double partials[BH_CPU_FSUM_MAX_TERMS];
    size_t partial_count = 0;

    for (size_t value_index = 0; value_index < count; ++value_index) {
        double x = values[value_index];
        if (!isfinite(x)) {
            return BH_CPU_NONFINITE_INPUT;
        }
        size_t next_partial = 0;
        for (size_t partial_index = 0;
             partial_index < partial_count;
             ++partial_index) {
            double y = partials[partial_index];
            if (fabs(x) < fabs(y)) {
                const double swap = x;
                x = y;
                y = swap;
            }
            const double high = x + y;
            const double rounded_y = high - x;
            const double low = y - rounded_y;
            if (low != 0.0) {
                partials[next_partial] = low;
                ++next_partial;
            }
            x = high;
        }
        partial_count = next_partial;
        if (!isfinite(x)) {
            return BH_CPU_NUMERIC_OVERFLOW;
        }
        if (x != 0.0) {
            if (partial_count == BH_CPU_FSUM_MAX_TERMS) {
                return BH_CPU_TOO_MANY_TERMS;
            }
            partials[partial_count] = x;
            ++partial_count;
        }
    }

    double high = 0.0;
    double low = 0.0;
    double y = 0.0;
    if (partial_count > 0) {
        --partial_count;
        high = partials[partial_count];
        while (partial_count > 0) {
            const double x = high;
            --partial_count;
            y = partials[partial_count];
            high = x + y;
            const double rounded_y = high - x;
            low = y - rounded_y;
            if (low != 0.0) {
                break;
            }
        }
        if (partial_count > 0 &&
            ((low < 0.0 && partials[partial_count - 1] < 0.0) ||
             (low > 0.0 && partials[partial_count - 1] > 0.0))) {
            y = low * 2.0;
            const double x = high + y;
            const double rounded_y = x - high;
            if (y == rounded_y) {
                high = x;
            }
        }
    }
    *out_sum = high;
    return BH_CPU_OK;
}

typedef struct bh_double_length {
    double high;
    double low;
} bh_double_length;

static bh_double_length
bh_double_length_fast_sum(double first, double second)
{
    const double high = first + second;
    const double low = (first - high) + second;
    return (bh_double_length){.high = high, .low = low};
}

/* Dekker error-free product, used instead of a fused hardware instruction. */
static bh_double_length
bh_double_length_multiply(double first, double second)
{
    const double splitter = 134217729.0;
    const double first_split = first * splitter;
    const double first_high = first_split - (first_split - first);
    const double first_low = first - first_high;
    const double second_split = second * splitter;
    const double second_high = second_split - (second_split - second);
    const double second_low = second - second_high;
    const double cross =
        first_high * second_low + first_low * second_high;
    const double product_sum = first_high * second_high + cross;
    const double product_low =
        first_high * second_high - product_sum + cross +
        first_low * second_low;
    return (bh_double_length){
        .high = product_sum,
        .low = product_low,
    };
}

/* Finite two-coordinate CPython 3.12 math.hypot/vector_norm path. */
static double
bh_hypot2_impl(double first, double second)
{
    double coordinates[2] = {fabs(first), fabs(second)};
    double maximum = coordinates[0];
    if (coordinates[1] > maximum) {
        maximum = coordinates[1];
    }
    if (maximum == 0.0) {
        return maximum;
    }

    int maximum_exponent = 0;
    (void)frexp(maximum, &maximum_exponent);
    if (maximum_exponent < -1023) {
        coordinates[0] = coordinates[0] / DBL_MIN;
        coordinates[1] = coordinates[1] / DBL_MIN;
        return DBL_MIN * bh_hypot2_impl(
            coordinates[0], coordinates[1]);
    }
    const double scale = ldexp(1.0, -maximum_exponent);
    double compensated_sum = 1.0;
    double fraction_one = 0.0;
    double fraction_two = 0.0;
    for (size_t index = 0; index < 2; ++index) {
        const double scaled = coordinates[index] * scale;
        const bh_double_length product =
            bh_double_length_multiply(scaled, scaled);
        const bh_double_length sum =
            bh_double_length_fast_sum(compensated_sum, product.high);
        compensated_sum = sum.high;
        fraction_one = fraction_one + product.low;
        fraction_two = fraction_two + sum.low;
    }
    double combined_fraction = fraction_one + fraction_two;
    double radicand = compensated_sum - 1.0;
    radicand = radicand + combined_fraction;
    double result = sqrt(radicand);
    const bh_double_length negative_square =
        bh_double_length_multiply(-result, result);
    const bh_double_length corrected =
        bh_double_length_fast_sum(compensated_sum, negative_square.high);
    compensated_sum = corrected.high;
    fraction_one = fraction_one + negative_square.low;
    fraction_two = fraction_two + corrected.low;
    combined_fraction = fraction_one + fraction_two;
    double correction = compensated_sum - 1.0;
    correction = correction + combined_fraction;
    result = result + correction / (2.0 * result);
    return result / scale;
}

bh_cpu_status
bh_cpu_fsum(const double *values, size_t count, double *out_sum)
{
    if (out_sum == NULL || (count > 0 && values == NULL)) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    if (count > BH_CPU_FSUM_MAX_TERMS) {
        return BH_CPU_TOO_MANY_TERMS;
    }
    const bh_cpu_status environment_status = bh_cpu_require_strict_fp();
    if (environment_status != BH_CPU_OK) {
        return environment_status;
    }
    return bh_fsum_impl(values, count, out_sum);
}

static bh_cpu_status
bh_matrix4_vector4_impl(
    const double matrix[16],
    const double vector[4],
    double out_vector[4]
)
{
    for (size_t row = 0; row < 4; ++row) {
        double products[4];
        for (size_t column = 0; column < 4; ++column) {
            products[column] = matrix[row * 4 + column] * vector[column];
        }
        const bh_cpu_status status = bh_fsum_impl(products, 4, &out_vector[row]);
        if (status != BH_CPU_OK) {
            return status;
        }
    }
    return BH_CPU_OK;
}

bh_cpu_status
bh_cpu_matrix4_vector4(
    const double matrix[16],
    const double vector[4],
    double out_vector[4]
)
{
    if (matrix == NULL || vector == NULL || out_vector == NULL) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    const bh_cpu_status environment_status = bh_cpu_require_strict_fp();
    if (environment_status != BH_CPU_OK) {
        return environment_status;
    }
    if (!bh_all_finite(matrix, 16) || !bh_all_finite(vector, 4)) {
        return BH_CPU_NONFINITE_INPUT;
    }
    /* Make the scalar ABI safe even if out_vector aliases an input. */
    double result[4];
    const bh_cpu_status status =
        bh_matrix4_vector4_impl(matrix, vector, result);
    if (status == BH_CPU_OK) {
        memcpy(out_vector, result, sizeof(result));
    }
    return status;
}

static bh_cpu_status
bh_bilinear4_impl(
    const double left[4],
    const double matrix[16],
    const double right[4],
    double *out_value
)
{
    double product[4];
    bh_cpu_status status = bh_matrix4_vector4_impl(matrix, right, product);
    if (status != BH_CPU_OK) {
        return status;
    }
    double terms[4];
    for (size_t index = 0; index < 4; ++index) {
        terms[index] = left[index] * product[index];
    }
    status = bh_fsum_impl(terms, 4, out_value);
    return status;
}

bh_cpu_status
bh_cpu_bilinear4(
    const double left[4],
    const double matrix[16],
    const double right[4],
    double *out_value
)
{
    if (left == NULL || matrix == NULL || right == NULL || out_value == NULL) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    const bh_cpu_status environment_status = bh_cpu_require_strict_fp();
    if (environment_status != BH_CPU_OK) {
        return environment_status;
    }
    if (!bh_all_finite(left, 4) || !bh_all_finite(matrix, 16) ||
        !bh_all_finite(right, 4)) {
        return BH_CPU_NONFINITE_INPUT;
    }
    return bh_bilinear4_impl(left, matrix, right, out_value);
}

static bool
bh_matrix4_is_symmetric(
    const double matrix[16],
    double relative_tolerance,
    double absolute_tolerance
)
{
    for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 4; ++column) {
            const double left = matrix[row * 4 + column];
            const double right = matrix[column * 4 + row];
            const double scale = fmax(fabs(left), fabs(right));
            const double tolerance =
                absolute_tolerance + relative_tolerance * scale;
            if (fabs(left - right) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

static bh_cpu_status
bh_metric_algebra_audit_impl(
    const double covariant[16],
    const double inverse[16],
    const double inverse_derivatives[64],
    double relative_tolerance,
    double absolute_tolerance
)
{
    if (!bh_matrix4_is_symmetric(
            covariant, relative_tolerance, absolute_tolerance) ||
        !bh_matrix4_is_symmetric(
            inverse, relative_tolerance, absolute_tolerance)) {
        return BH_CPU_INCONSISTENT_METRIC;
    }
    for (size_t derivative = 0; derivative < 4; ++derivative) {
        if (!bh_matrix4_is_symmetric(
                inverse_derivatives + derivative * 16,
                relative_tolerance,
                absolute_tolerance)) {
            return BH_CPU_INCONSISTENT_METRIC;
        }
    }

    for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 4; ++column) {
            double terms[4];
            double absolute_terms[4];
            for (size_t inner = 0; inner < 4; ++inner) {
                terms[inner] = covariant[row * 4 + inner] *
                               inverse[inner * 4 + column];
                absolute_terms[inner] = fabs(terms[inner]);
            }
            double product = 0.0;
            double absolute_sum = 0.0;
            bh_cpu_status status = bh_fsum_impl(terms, 4, &product);
            if (status != BH_CPU_OK) {
                return status;
            }
            status = bh_fsum_impl(absolute_terms, 4, &absolute_sum);
            if (status != BH_CPU_OK || !isfinite(absolute_sum)) {
                return status == BH_CPU_OK ? BH_CPU_NUMERIC_OVERFLOW : status;
            }
            const double expected = row == column ? 1.0 : 0.0;
            const double tolerance = absolute_tolerance +
                                     relative_tolerance *
                                         fmax(1.0, fabs(expected));
            if (!isfinite(product) || fabs(product - expected) > tolerance) {
                return BH_CPU_INCONSISTENT_METRIC;
            }
        }
    }
    return BH_CPU_OK;
}

/* Match offline.spacetime._require_lorentzian_minus_plus_signature. */
static bh_cpu_status
bh_lorentzian_minus_plus_audit_impl(const double covariant[16])
{
    double scale = fabs(covariant[0]);
    for (size_t index = 1; index < 16; ++index) {
        const double magnitude = fabs(covariant[index]);
        if (magnitude > scale) {
            scale = magnitude;
        }
    }
    if (!isfinite(scale) || scale == 0.0) {
        return BH_CPU_INCONSISTENT_METRIC;
    }

    double working[16];
    for (size_t index = 0; index < 16; ++index) {
        working[index] = covariant[index] / scale;
    }

    bool converged = false;
    for (size_t sweep = 0; sweep < 64; ++sweep) {
        size_t first = 0;
        size_t second = 1;
        double off_diagonal = fabs(working[1]);
        for (size_t row = 0; row < 4; ++row) {
            for (size_t column = row + 1; column < 4; ++column) {
                const double candidate = fabs(working[row * 4 + column]);
                /* Python max(..., key=...) retains the first item on ties. */
                if (candidate > off_diagonal) {
                    off_diagonal = candidate;
                    first = row;
                    second = column;
                }
            }
        }
        if (off_diagonal <= 2.0e-14) {
            converged = true;
            break;
        }

        const double coupling = working[first * 4 + second];
        const double diagonal_difference =
            working[second * 4 + second] - working[first * 4 + first];
        const double twice_coupling = 2.0 * coupling;
        const double tau = diagonal_difference / twice_coupling;
        const double tau_square = tau * tau;
        const double tangent_denominator =
            fabs(tau) + sqrt(1.0 + tau_square);
        const double tangent_sign = tau >= 0.0 ? 1.0 : -1.0;
        const double tangent = tangent_sign / tangent_denominator;
        const double tangent_square = tangent * tangent;
        const double cosine = 1.0 / sqrt(1.0 + tangent_square);
        const double sine = tangent * cosine;
        const double first_diagonal = working[first * 4 + first];
        const double second_diagonal = working[second * 4 + second];

        double first_value = cosine * cosine;
        first_value = first_value * first_diagonal;
        double first_coupling = 2.0 * sine;
        first_coupling = first_coupling * cosine;
        first_coupling = first_coupling * coupling;
        first_value = first_value - first_coupling;
        double first_second = sine * sine;
        first_second = first_second * second_diagonal;
        first_value = first_value + first_second;

        double second_value = sine * sine;
        second_value = second_value * first_diagonal;
        double second_coupling = 2.0 * sine;
        second_coupling = second_coupling * cosine;
        second_coupling = second_coupling * coupling;
        second_value = second_value + second_coupling;
        double second_first = cosine * cosine;
        second_first = second_first * second_diagonal;
        second_value = second_value + second_first;

        working[first * 4 + first] = first_value;
        working[second * 4 + second] = second_value;
        working[first * 4 + second] = 0.0;
        working[second * 4 + first] = 0.0;
        for (size_t index = 0; index < 4; ++index) {
            if (index == first || index == second) {
                continue;
            }
            const double old_first = working[index * 4 + first];
            const double old_second = working[index * 4 + second];
            const double new_first =
                cosine * old_first - sine * old_second;
            const double new_second =
                sine * old_first + cosine * old_second;
            working[index * 4 + first] = new_first;
            working[first * 4 + index] = new_first;
            working[index * 4 + second] = new_second;
            working[second * 4 + index] = new_second;
        }
        if (!bh_all_finite(working, 16)) {
            return BH_CPU_INCONSISTENT_METRIC;
        }
    }
    if (!converged) {
        return BH_CPU_INCONSISTENT_METRIC;
    }

    size_t negative_count = 0;
    for (size_t index = 0; index < 4; ++index) {
        const double eigenvalue = working[index * 4 + index];
        if (fabs(eigenvalue) <= 2.0e-12) {
            return BH_CPU_INCONSISTENT_METRIC;
        }
        if (eigenvalue < 0.0) {
            ++negative_count;
        }
    }
    return negative_count == 1 ? BH_CPU_OK : BH_CPU_INCONSISTENT_METRIC;
}

static bh_cpu_status
bh_validate_metric_audit_arguments(
    const double covariant[16],
    const double inverse[16],
    const double inverse_derivatives[64],
    double relative_tolerance,
    double absolute_tolerance
)
{
    if (covariant == NULL || inverse == NULL || inverse_derivatives == NULL) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    if (!isfinite(relative_tolerance) || relative_tolerance < 0.0 ||
        !isfinite(absolute_tolerance) || absolute_tolerance < 0.0) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    if (!bh_all_finite(covariant, 16) || !bh_all_finite(inverse, 16) ||
        !bh_all_finite(inverse_derivatives, 64)) {
        return BH_CPU_NONFINITE_INPUT;
    }
    return BH_CPU_OK;
}

bh_cpu_status
bh_cpu_metric_algebra_audit(
    const double covariant[16],
    const double inverse[16],
    const double inverse_derivatives[64],
    double relative_tolerance,
    double absolute_tolerance
)
{
    const bh_cpu_status environment_status = bh_cpu_require_strict_fp();
    if (environment_status != BH_CPU_OK) {
        return environment_status;
    }
    const bh_cpu_status argument_status = bh_validate_metric_audit_arguments(
        covariant,
        inverse,
        inverse_derivatives,
        relative_tolerance,
        absolute_tolerance);
    if (argument_status != BH_CPU_OK) {
        return argument_status;
    }
    return bh_metric_algebra_audit_impl(
        covariant,
        inverse,
        inverse_derivatives,
        relative_tolerance,
        absolute_tolerance);
}

bh_cpu_status
bh_cpu_metric_sample_audit(
    const double covariant[16],
    const double inverse[16],
    const double inverse_derivatives[64],
    double relative_tolerance,
    double absolute_tolerance
)
{
    const bh_cpu_status environment_status = bh_cpu_require_strict_fp();
    if (environment_status != BH_CPU_OK) {
        return environment_status;
    }
    const bh_cpu_status argument_status = bh_validate_metric_audit_arguments(
        covariant,
        inverse,
        inverse_derivatives,
        relative_tolerance,
        absolute_tolerance);
    if (argument_status != BH_CPU_OK) {
        return argument_status;
    }
    bh_cpu_status status = bh_metric_algebra_audit_impl(
        covariant,
        inverse,
        inverse_derivatives,
        relative_tolerance,
        absolute_tolerance);
    if (status != BH_CPU_OK) {
        return status;
    }
    status = bh_lorentzian_minus_plus_audit_impl(covariant);
    return status;
}

static bh_cpu_status
bh_validate_kerr_parameters(
    double mass_m,
    double spin_a_m,
    double singularity_guard_m
)
{
    if (!isfinite(mass_m) || !isfinite(spin_a_m) ||
        !isfinite(singularity_guard_m)) {
        return BH_CPU_NONFINITE_INPUT;
    }
    if (mass_m <= 0.0 || fabs(spin_a_m) > mass_m ||
        singularity_guard_m <= 0.0) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    return BH_CPU_OK;
}

/*
 * Match KerrKerrSchildMetric.sample primitive-for-primitive.  Every named
 * temporary below corresponds to a Python binary64 operation; the strict
 * compile contract prevents contraction and reassociation.
 */
static bh_cpu_status
bh_kerr_metric_sample_impl(
    const double event[4],
    double mass_m,
    double spin_a_m,
    double singularity_guard_m,
    double out_covariant[16],
    double out_inverse[16],
    double out_inverse_derivatives[64]
)
{
    const double x_m = event[1];
    const double y_m = event[2];
    const double z_m = event[3];
    const double spin_squared = spin_a_m * spin_a_m;
    const double squared_coordinates[3] = {
        x_m * x_m,
        y_m * y_m,
        z_m * z_m,
    };
    double rho_squared = 0.0;
    bh_cpu_status status =
        bh_fsum_impl(squared_coordinates, 3, &rho_squared);
    if (status != BH_CPU_OK) {
        return status;
    }
    const double difference = rho_squared - spin_squared;
    const double spin_position = spin_a_m * z_m;
    const double twice_spin_position = 2.0 * spin_position;
    if (!isfinite(difference) || !isfinite(twice_spin_position)) {
        return BH_CPU_NUMERIC_OVERFLOW;
    }
    const double discriminant_root =
        bh_hypot2_impl(difference, twice_spin_position);
    if (!isfinite(discriminant_root)) {
        return BH_CPU_NUMERIC_OVERFLOW;
    }

    double radius_squared = 0.0;
    if (difference >= 0.0) {
        radius_squared = 0.5 * (difference + discriminant_root);
    } else if (spin_position != 0.0) {
        double numerator = 2.0 * spin_position;
        numerator = numerator * spin_position;
        radius_squared = numerator / (discriminant_root - difference);
    }
    const double nonnegative_radius_squared =
        radius_squared > 0.0 ? radius_squared : 0.0;
    const double radius = sqrt(nonnegative_radius_squared);
    const double guard_squared = pow(singularity_guard_m, 2.0);
    if (radius <= singularity_guard_m ||
        discriminant_root <= guard_squared) {
        return BH_CPU_GUARDED_SINGULARITY;
    }

    const double radius_discriminant = radius * discriminant_root;
    const double radius_x = radius_squared * x_m;
    const double radius_y = radius_squared * y_m;
    const double radius_plus_spin = radius_squared + spin_squared;
    const double radius_z = radius_plus_spin * z_m;
    const double radius_derivatives[3] = {
        radius_x / radius_discriminant,
        radius_y / radius_discriminant,
        radius_z / radius_discriminant,
    };

    const double radius_fourth = radius_squared * radius_squared;
    double spin_z_squared = spin_squared * z_m;
    spin_z_squared = spin_z_squared * z_m;
    const double h_denominator = radius_fourth + spin_z_squared;
    const double guard_fourth = pow(singularity_guard_m, 4.0);
    if (!isfinite(h_denominator) || h_denominator <= guard_fourth) {
        return BH_CPU_GUARDED_SINGULARITY;
    }
    double h = mass_m * radius;
    h = h * radius_squared;
    h = h / h_denominator;

    const double spatial_denominator = radius_squared + spin_squared;
    const double numerator_x = radius * x_m + spin_a_m * y_m;
    const double numerator_y = radius * y_m - spin_a_m * x_m;
    const double spatial_null[3] = {
        numerator_x / spatial_denominator,
        numerator_y / spatial_denominator,
        z_m / radius,
    };
    const double l_covariant[4] = {
        1.0,
        spatial_null[0],
        spatial_null[1],
        spatial_null[2],
    };
    const double l_contravariant[4] = {
        -1.0,
        spatial_null[0],
        spatial_null[1],
        spatial_null[2],
    };

    double covariant[16];
    double inverse[16];
    for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 4; ++column) {
            const double minkowski =
                row == column ? (row == 0 ? -1.0 : 1.0) : 0.0;
            double covariant_term = 2.0 * h;
            covariant_term = covariant_term * l_covariant[row];
            covariant_term = covariant_term * l_covariant[column];
            covariant[row * 4 + column] = minkowski + covariant_term;

            double inverse_term = 2.0 * h;
            inverse_term = inverse_term * l_contravariant[row];
            inverse_term = inverse_term * l_contravariant[column];
            inverse[row * 4 + column] = minkowski - inverse_term;
        }
    }

    double inverse_derivatives[64];
    memset(inverse_derivatives, 0, 16 * sizeof(double));
    for (size_t axis = 0; axis < 3; ++axis) {
        const double derivative_radius = radius_derivatives[axis];
        double derivative_h_denominator = 4.0 * radius;
        derivative_h_denominator =
            derivative_h_denominator * radius_squared;
        derivative_h_denominator =
            derivative_h_denominator * derivative_radius;
        double z_derivative_term = 0.0;
        if (axis == 2) {
            z_derivative_term = 2.0 * spin_squared;
            z_derivative_term = z_derivative_term * z_m;
        }
        derivative_h_denominator =
            derivative_h_denominator + z_derivative_term;
        double logarithmic_h_derivative = 3.0 * derivative_radius;
        logarithmic_h_derivative = logarithmic_h_derivative / radius;
        logarithmic_h_derivative = logarithmic_h_derivative -
            derivative_h_denominator / h_denominator;
        const double derivative_h = h * logarithmic_h_derivative;

        double derivative_spatial_denominator = 2.0 * radius;
        derivative_spatial_denominator =
            derivative_spatial_denominator * derivative_radius;
        double derivative_numerator_x = derivative_radius * x_m;
        derivative_numerator_x = derivative_numerator_x +
            (axis == 0 ? radius : 0.0);
        derivative_numerator_x = derivative_numerator_x +
            (axis == 1 ? spin_a_m : 0.0);
        double derivative_numerator_y = derivative_radius * y_m;
        derivative_numerator_y = derivative_numerator_y +
            (axis == 1 ? radius : 0.0);
        derivative_numerator_y = derivative_numerator_y -
            (axis == 0 ? spin_a_m : 0.0);

        double derivative_l_x =
            derivative_numerator_x * spatial_denominator;
        derivative_l_x = derivative_l_x -
            numerator_x * derivative_spatial_denominator;
        derivative_l_x = derivative_l_x /
            (spatial_denominator * spatial_denominator);
        double derivative_l_y =
            derivative_numerator_y * spatial_denominator;
        derivative_l_y = derivative_l_y -
            numerator_y * derivative_spatial_denominator;
        derivative_l_y = derivative_l_y /
            (spatial_denominator * spatial_denominator);
        double derivative_l_z = (axis == 2 ? 1.0 : 0.0) / radius;
        double derivative_l_z_second = z_m * derivative_radius;
        derivative_l_z_second = derivative_l_z_second / radius_squared;
        derivative_l_z = derivative_l_z - derivative_l_z_second;
        const double derivative_l[4] = {
            0.0,
            derivative_l_x,
            derivative_l_y,
            derivative_l_z,
        };

        const size_t derivative_offset = (axis + 1) * 16;
        for (size_t row = 0; row < 4; ++row) {
            for (size_t column = 0; column < 4; ++column) {
                double first_term = derivative_h * l_contravariant[row];
                first_term = first_term * l_contravariant[column];
                double second_term = h * derivative_l[row];
                second_term = second_term * l_contravariant[column];
                double third_term = h * l_contravariant[row];
                third_term = third_term * derivative_l[column];
                double derivative_value = first_term + second_term;
                derivative_value = derivative_value + third_term;
                derivative_value = -2.0 * derivative_value;
                inverse_derivatives[
                    derivative_offset + row * 4 + column] = derivative_value;
            }
        }
    }
    if (!bh_all_finite(covariant, 16) || !bh_all_finite(inverse, 16) ||
        !bh_all_finite(inverse_derivatives, 64)) {
        return BH_CPU_NUMERIC_OVERFLOW;
    }

    status = bh_metric_algebra_audit_impl(
        covariant,
        inverse,
        inverse_derivatives,
        2.0e-10,
        2.0e-12);
    if (status != BH_CPU_OK) {
        return status;
    }
    status = bh_lorentzian_minus_plus_audit_impl(covariant);
    if (status != BH_CPU_OK) {
        return status;
    }

    memcpy(out_covariant, covariant, sizeof(covariant));
    memcpy(out_inverse, inverse, sizeof(inverse));
    memcpy(
        out_inverse_derivatives,
        inverse_derivatives,
        sizeof(inverse_derivatives));
    return BH_CPU_OK;
}

bh_cpu_status
bh_cpu_kerr_metric_sample(
    const double event[4],
    double mass_m,
    double spin_a_m,
    double singularity_guard_m,
    double out_covariant[16],
    double out_inverse[16],
    double out_inverse_derivatives[64]
)
{
    if (event == NULL || out_covariant == NULL || out_inverse == NULL ||
        out_inverse_derivatives == NULL) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    const bh_cpu_status environment_status = bh_cpu_require_strict_fp();
    if (environment_status != BH_CPU_OK) {
        return environment_status;
    }
    if (!bh_all_finite(event, 4)) {
        return BH_CPU_NONFINITE_INPUT;
    }
    const bh_cpu_status parameter_status = bh_validate_kerr_parameters(
        mass_m,
        spin_a_m,
        singularity_guard_m);
    if (parameter_status != BH_CPU_OK) {
        return parameter_status;
    }
    return bh_kerr_metric_sample_impl(
        event,
        mass_m,
        spin_a_m,
        singularity_guard_m,
        out_covariant,
        out_inverse,
        out_inverse_derivatives);
}

bh_cpu_status
bh_cpu_normalized_null_residual(
    const double inverse[16],
    const double covector[4],
    double *out_residual
)
{
    if (inverse == NULL || covector == NULL || out_residual == NULL) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    const bh_cpu_status environment_status = bh_cpu_require_strict_fp();
    if (environment_status != BH_CPU_OK) {
        return environment_status;
    }
    if (!bh_all_finite(inverse, 16) || !bh_all_finite(covector, 4)) {
        return BH_CPU_NONFINITE_INPUT;
    }

    double momentum_scale = 0.0;
    for (size_t index = 0; index < 4; ++index) {
        momentum_scale = fmax(momentum_scale, fabs(covector[index]));
    }
    double metric_scale = 0.0;
    for (size_t index = 0; index < 16; ++index) {
        metric_scale = fmax(metric_scale, fabs(inverse[index]));
    }
    if (momentum_scale == 0.0 || metric_scale == 0.0) {
        *out_residual = HUGE_VAL;
        return BH_CPU_OK;
    }

    double normalized_covector[4];
    for (size_t index = 0; index < 4; ++index) {
        normalized_covector[index] = covector[index] / momentum_scale;
    }
    double terms[16];
    double absolute_terms[16];
    for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 4; ++column) {
            const size_t index = row * 4 + column;
            double term = inverse[index] / metric_scale;
            term = term * normalized_covector[row];
            term = term * normalized_covector[column];
            terms[index] = term;
            absolute_terms[index] = fabs(term);
        }
    }
    double numerator = 0.0;
    double denominator = 0.0;
    bh_cpu_status status = bh_fsum_impl(terms, 16, &numerator);
    if (status != BH_CPU_OK) {
        return status;
    }
    status = bh_fsum_impl(absolute_terms, 16, &denominator);
    if (status != BH_CPU_OK) {
        return status;
    }
    if (denominator == 0.0) {
        *out_residual = HUGE_VAL;
        return BH_CPU_OK;
    }
    const double result = fabs(numerator) / denominator;
    *out_residual = isfinite(result) ? result : HUGE_VAL;
    return BH_CPU_OK;
}

static bh_cpu_status
bh_hamiltonian_rhs_impl(
    const double inverse[16],
    const double inverse_derivatives[64],
    const double covector[4],
    double out_rhs[8]
)
{
    bh_cpu_status status =
        bh_matrix4_vector4_impl(inverse, covector, out_rhs);
    if (status != BH_CPU_OK) {
        return status;
    }
    for (size_t coordinate = 0; coordinate < 4; ++coordinate) {
        double quadratic = 0.0;
        status = bh_bilinear4_impl(
            covector,
            inverse_derivatives + coordinate * 16,
            covector,
            &quadratic);
        if (status != BH_CPU_OK) {
            return status;
        }
        out_rhs[4 + coordinate] = -0.5 * quadratic;
    }
    return bh_all_finite(out_rhs, 8) ? BH_CPU_OK : BH_CPU_NUMERIC_OVERFLOW;
}

bh_cpu_status
bh_cpu_hamiltonian_rhs(
    const double inverse[16],
    const double inverse_derivatives[64],
    const double covector[4],
    double out_rhs[8]
)
{
    if (inverse == NULL || inverse_derivatives == NULL || covector == NULL ||
        out_rhs == NULL) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    const bh_cpu_status environment_status = bh_cpu_require_strict_fp();
    if (environment_status != BH_CPU_OK) {
        return environment_status;
    }
    if (!bh_all_finite(inverse, 16) ||
        !bh_all_finite(inverse_derivatives, 64) ||
        !bh_all_finite(covector, 4)) {
        return BH_CPU_NONFINITE_INPUT;
    }
    /* Make the scalar ABI safe even if out_rhs aliases an input. */
    double result[8];
    const bh_cpu_status status = bh_hamiltonian_rhs_impl(
        inverse, inverse_derivatives, covector, result);
    if (status == BH_CPU_OK) {
        memcpy(out_rhs, result, sizeof(result));
    }
    return status;
}

bh_cpu_status
bh_cpu_hamiltonian_rhs_batch(
    const double *inverse_records,
    const double *inverse_derivative_records,
    const double *covector_records,
    size_t record_count,
    double *out_rhs_records,
    size_t *out_completed_records
)
{
    if (out_completed_records != NULL) {
        *out_completed_records = 0;
    }
    if ((record_count > 0 &&
         (inverse_records == NULL || inverse_derivative_records == NULL ||
          covector_records == NULL || out_rhs_records == NULL)) ||
        out_completed_records == NULL) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    if (record_count > SIZE_MAX / 64 || record_count > SIZE_MAX / 16 ||
        record_count > SIZE_MAX / 8 || record_count > SIZE_MAX / 4) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    const bh_cpu_status environment_status = bh_cpu_require_strict_fp();
    if (environment_status != BH_CPU_OK) {
        return environment_status;
    }
    for (size_t record = 0; record < record_count; ++record) {
        const double *inverse = inverse_records + record * 16;
        const double *derivatives = inverse_derivative_records + record * 64;
        const double *covector = covector_records + record * 4;
        if (!bh_all_finite(inverse, 16) ||
            !bh_all_finite(derivatives, 64) ||
            !bh_all_finite(covector, 4)) {
            return BH_CPU_NONFINITE_INPUT;
        }
        const bh_cpu_status status = bh_hamiltonian_rhs_impl(
            inverse,
            derivatives,
            covector,
            out_rhs_records + record * 8);
        if (status != BH_CPU_OK) {
            return status;
        }
        *out_completed_records = record + 1;
    }
    return BH_CPU_OK;
}

static bh_cpu_status
bh_kerr_hamiltonian_rhs_impl(
    const double state[8],
    double mass_m,
    double spin_a_m,
    double singularity_guard_m,
    double out_rhs[8],
    uint64_t *metric_sample_evaluations,
    uint64_t *hamiltonian_rhs_evaluations
)
{
    if (metric_sample_evaluations != NULL) {
        *metric_sample_evaluations += UINT64_C(1);
    }
    double covariant[16];
    double inverse[16];
    double inverse_derivatives[64];
    const bh_cpu_status sample_status = bh_kerr_metric_sample_impl(
        state,
        mass_m,
        spin_a_m,
        singularity_guard_m,
        covariant,
        inverse,
        inverse_derivatives);
    if (sample_status != BH_CPU_OK) {
        return sample_status;
    }
    const bh_cpu_status status = bh_hamiltonian_rhs_impl(
        inverse,
        inverse_derivatives,
        state + 4,
        out_rhs);
    if (status == BH_CPU_OK && hamiltonian_rhs_evaluations != NULL) {
        *hamiltonian_rhs_evaluations += UINT64_C(1);
    }
    return status;
}

bh_cpu_status
bh_cpu_kerr_hamiltonian_rhs(
    const double state[8],
    double mass_m,
    double spin_a_m,
    double singularity_guard_m,
    double out_rhs[8]
)
{
    if (state == NULL || out_rhs == NULL) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    const bh_cpu_status environment_status = bh_cpu_require_strict_fp();
    if (environment_status != BH_CPU_OK) {
        return environment_status;
    }
    if (!bh_all_finite(state, 8)) {
        return BH_CPU_NONFINITE_INPUT;
    }
    const bh_cpu_status parameter_status = bh_validate_kerr_parameters(
        mass_m,
        spin_a_m,
        singularity_guard_m);
    if (parameter_status != BH_CPU_OK) {
        return parameter_status;
    }
    double result[8];
    const bh_cpu_status status = bh_kerr_hamiltonian_rhs_impl(
        state,
        mass_m,
        spin_a_m,
        singularity_guard_m,
        result,
        NULL,
        NULL);
    if (status == BH_CPU_OK) {
        memcpy(out_rhs, result, sizeof(result));
    }
    return status;
}

static bh_cpu_status
bh_linear_combination_impl(
    const double base[8],
    double step,
    const double *coefficients,
    const double vectors[][8],
    size_t term_count,
    double out_state[8]
)
{
    for (size_t component = 0; component < 8; ++component) {
        double products[6];
        for (size_t term = 0; term < term_count; ++term) {
            products[term] = coefficients[term] * vectors[term][component];
        }
        double sum = 0.0;
        const bh_cpu_status status =
            bh_fsum_impl(products, term_count, &sum);
        if (status != BH_CPU_OK) {
            return status;
        }
        const double increment = step * sum;
        out_state[component] = base[component] + increment;
    }
    return bh_all_finite(out_state, 8)
        ? BH_CPU_OK
        : BH_CPU_NUMERIC_OVERFLOW;
}

static bh_cpu_status
bh_kerr_dopri54_step_impl(
    const double state[8],
    double step,
    double mass_m,
    double spin_a_m,
    double singularity_guard_m,
    double out_fifth[8],
    double out_error[8],
    uint64_t *metric_sample_evaluations,
    uint64_t *hamiltonian_rhs_evaluations
)
{
    static const double stage2_coefficients[1] = {
        0x1.999999999999ap-3,
    };
    static const double stage3_coefficients[2] = {
        0x1.3333333333333p-4,
        0x1.ccccccccccccdp-3,
    };
    static const double stage4_coefficients[3] = {
        0x1.f49f49f49f49fp-1,
        -0x1.ddddddddddddep+1,
        0x1.c71c71c71c71cp+1,
    };
    static const double stage5_coefficients[4] = {
        0x1.79eec0fc37181p+1,
        -0x1.7310bd29520e4p+3,
        0x1.3a552363c5290p+3,
        -0x1.29c9eba1e3345p-2,
    };
    static const double stage6_coefficients[5] = {
        0x1.6c52bf5a814b0p+1,
        -0x1.583e0f83e0f84p+3,
        0x1.1d016a3721e8bp+3,
        0x1.1d1745d1745d1p-2,
        -0x1.1818970d9cc2fp-2,
    };
    static const double fifth_coefficients[5] = {
        0x1.7555555555555p-4,
        0x1.cc0499a5605fbp-2,
        0x1.4d55555555555p-1,
        -0x1.4a1cfb2b78c13p-2,
        0x1.0c30c30c30c31p-3,
    };
    static const double fourth_coefficients[6] = {
        0x1.7048d159e26afp-4,
        0x1.d05f703aa30fap-2,
        0x1.3a66666666666p-1,
        -0x1.16075785e4908p-2,
        0x1.6cbd323989ff0p-4,
        0x1.999999999999ap-6,
    };

    double k[7][8];
    double stage_state[8];
    bh_cpu_status status = bh_kerr_hamiltonian_rhs_impl(
        state,
        mass_m,
        spin_a_m,
        singularity_guard_m,
        k[0],
        metric_sample_evaluations,
        hamiltonian_rhs_evaluations);
    if (status != BH_CPU_OK) {
        return status;
    }

    status = bh_linear_combination_impl(
        state, step, stage2_coefficients, k, 1, stage_state);
    if (status != BH_CPU_OK) {
        return status;
    }
    status = bh_kerr_hamiltonian_rhs_impl(
        stage_state,
        mass_m,
        spin_a_m,
        singularity_guard_m,
        k[1],
        metric_sample_evaluations,
        hamiltonian_rhs_evaluations);
    if (status != BH_CPU_OK) {
        return status;
    }

    status = bh_linear_combination_impl(
        state, step, stage3_coefficients, k, 2, stage_state);
    if (status != BH_CPU_OK) {
        return status;
    }
    status = bh_kerr_hamiltonian_rhs_impl(
        stage_state,
        mass_m,
        spin_a_m,
        singularity_guard_m,
        k[2],
        metric_sample_evaluations,
        hamiltonian_rhs_evaluations);
    if (status != BH_CPU_OK) {
        return status;
    }

    status = bh_linear_combination_impl(
        state, step, stage4_coefficients, k, 3, stage_state);
    if (status != BH_CPU_OK) {
        return status;
    }
    status = bh_kerr_hamiltonian_rhs_impl(
        stage_state,
        mass_m,
        spin_a_m,
        singularity_guard_m,
        k[3],
        metric_sample_evaluations,
        hamiltonian_rhs_evaluations);
    if (status != BH_CPU_OK) {
        return status;
    }

    status = bh_linear_combination_impl(
        state, step, stage5_coefficients, k, 4, stage_state);
    if (status != BH_CPU_OK) {
        return status;
    }
    status = bh_kerr_hamiltonian_rhs_impl(
        stage_state,
        mass_m,
        spin_a_m,
        singularity_guard_m,
        k[4],
        metric_sample_evaluations,
        hamiltonian_rhs_evaluations);
    if (status != BH_CPU_OK) {
        return status;
    }

    status = bh_linear_combination_impl(
        state, step, stage6_coefficients, k, 5, stage_state);
    if (status != BH_CPU_OK) {
        return status;
    }
    status = bh_kerr_hamiltonian_rhs_impl(
        stage_state,
        mass_m,
        spin_a_m,
        singularity_guard_m,
        k[5],
        metric_sample_evaluations,
        hamiltonian_rhs_evaluations);
    if (status != BH_CPU_OK) {
        return status;
    }

    const double fifth_vectors[5][8] = {
        {k[0][0], k[0][1], k[0][2], k[0][3],
         k[0][4], k[0][5], k[0][6], k[0][7]},
        {k[2][0], k[2][1], k[2][2], k[2][3],
         k[2][4], k[2][5], k[2][6], k[2][7]},
        {k[3][0], k[3][1], k[3][2], k[3][3],
         k[3][4], k[3][5], k[3][6], k[3][7]},
        {k[4][0], k[4][1], k[4][2], k[4][3],
         k[4][4], k[4][5], k[4][6], k[4][7]},
        {k[5][0], k[5][1], k[5][2], k[5][3],
         k[5][4], k[5][5], k[5][6], k[5][7]},
    };
    double fifth[8];
    status = bh_linear_combination_impl(
        state, step, fifth_coefficients, fifth_vectors, 5, fifth);
    if (status != BH_CPU_OK) {
        return status;
    }
    status = bh_kerr_hamiltonian_rhs_impl(
        fifth,
        mass_m,
        spin_a_m,
        singularity_guard_m,
        k[6],
        metric_sample_evaluations,
        hamiltonian_rhs_evaluations);
    if (status != BH_CPU_OK) {
        return status;
    }

    const double fourth_vectors[6][8] = {
        {k[0][0], k[0][1], k[0][2], k[0][3],
         k[0][4], k[0][5], k[0][6], k[0][7]},
        {k[2][0], k[2][1], k[2][2], k[2][3],
         k[2][4], k[2][5], k[2][6], k[2][7]},
        {k[3][0], k[3][1], k[3][2], k[3][3],
         k[3][4], k[3][5], k[3][6], k[3][7]},
        {k[4][0], k[4][1], k[4][2], k[4][3],
         k[4][4], k[4][5], k[4][6], k[4][7]},
        {k[5][0], k[5][1], k[5][2], k[5][3],
         k[5][4], k[5][5], k[5][6], k[5][7]},
        {k[6][0], k[6][1], k[6][2], k[6][3],
         k[6][4], k[6][5], k[6][6], k[6][7]},
    };
    double fourth[8];
    status = bh_linear_combination_impl(
        state, step, fourth_coefficients, fourth_vectors, 6, fourth);
    if (status != BH_CPU_OK) {
        return status;
    }
    double error[8];
    for (size_t component = 0; component < 8; ++component) {
        error[component] = fifth[component] - fourth[component];
    }
    if (!bh_all_finite(error, 8)) {
        return BH_CPU_NUMERIC_OVERFLOW;
    }
    memcpy(out_fifth, fifth, sizeof(fifth));
    memcpy(out_error, error, sizeof(error));
    return BH_CPU_OK;
}

bh_cpu_status
bh_cpu_kerr_dopri54_step(
    const double state[8],
    double step,
    double mass_m,
    double spin_a_m,
    double singularity_guard_m,
    double out_fifth[8],
    double out_error[8]
)
{
    if (state == NULL || out_fifth == NULL || out_error == NULL ||
        out_fifth == out_error) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    const bh_cpu_status environment_status = bh_cpu_require_strict_fp();
    if (environment_status != BH_CPU_OK) {
        return environment_status;
    }
    if (!bh_all_finite(state, 8) || !isfinite(step)) {
        return BH_CPU_NONFINITE_INPUT;
    }
    const bh_cpu_status parameter_status = bh_validate_kerr_parameters(
        mass_m,
        spin_a_m,
        singularity_guard_m);
    if (parameter_status != BH_CPU_OK) {
        return parameter_status;
    }
    double fifth[8];
    double error[8];
    const bh_cpu_status status = bh_kerr_dopri54_step_impl(
        state,
        step,
        mass_m,
        spin_a_m,
        singularity_guard_m,
        fifth,
        error,
        NULL,
        NULL);
    if (status == BH_CPU_OK) {
        memcpy(out_fifth, fifth, sizeof(fifth));
        memcpy(out_error, error, sizeof(error));
    }
    return status;
}

bh_cpu_status
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
)
{
    if (out_completed_probes != NULL) {
        *out_completed_probes = 0;
    }
    if (state == NULL || out_completed_probes == NULL ||
        (probe_count > 0 &&
         (affine_offsets == NULL || out_fifth_records == NULL ||
          out_error_records == NULL)) ||
        (probe_count > 0 && out_fifth_records == out_error_records) ||
        probe_count > SIZE_MAX / 8) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    const bh_cpu_status environment_status = bh_cpu_require_strict_fp();
    if (environment_status != BH_CPU_OK) {
        return environment_status;
    }
    if (!bh_all_finite(state, 8)) {
        return BH_CPU_NONFINITE_INPUT;
    }
    const bh_cpu_status parameter_status = bh_validate_kerr_parameters(
        mass_m,
        spin_a_m,
        singularity_guard_m);
    if (parameter_status != BH_CPU_OK) {
        return parameter_status;
    }

    double base[8];
    memcpy(base, state, sizeof(base));
    for (size_t probe = 0; probe < probe_count; ++probe) {
        if (!isfinite(affine_offsets[probe])) {
            return BH_CPU_NONFINITE_INPUT;
        }
        double fifth[8];
        double error[8];
        const bh_cpu_status status = bh_kerr_dopri54_step_impl(
            base,
            affine_offsets[probe],
            mass_m,
            spin_a_m,
            singularity_guard_m,
            fifth,
            error,
            NULL,
            NULL);
        if (status != BH_CPU_OK) {
            return status;
        }
        memcpy(out_fifth_records + probe * 8, fifth, sizeof(fifth));
        memcpy(out_error_records + probe * 8, error, sizeof(error));
        *out_completed_probes = probe + 1;
    }
    return BH_CPU_OK;
}

/* ------------------------------------------------------------------------- */
/* Exact one-resolution Kerr returning-radiation whole-ray core (ABI v3).    */

typedef enum bh_surface_flow {
    BH_SURFACE_FLOW_OK = 0,
    BH_SURFACE_FLOW_RETRY_SMALLER = 1,
    BH_SURFACE_FLOW_FAILURE = 2
} bh_surface_flow;

typedef struct bh_ray_trace_context {
    const bh_cpu_kerr_returning_model *model;
    const bh_cpu_kerr_ray_options *ray;
    const bh_cpu_kerr_surface_options *surface;
    uint64_t probe_reintegrations;
    uint64_t surface_value_evaluations;
    uint64_t metric_sample_evaluations;
    uint64_t hamiltonian_rhs_evaluations;
    double maximum_probe_null_residual;
} bh_ray_trace_context;

typedef struct bh_surface_probe {
    double state[8];
    double values[2];
    double affine_offset;
} bh_surface_probe;

typedef struct bh_surface_probe_cache {
    bh_ray_trace_context *context;
    double start[8];
    bh_surface_probe *items;
    size_t count;
    size_t capacity;
} bh_surface_probe_cache;

typedef struct bh_surface_candidate {
    uint32_t surface_id;
    double lower_offset;
    double upper_offset;
    bool exact;
    bh_cpu_recorded_surface_crossing exact_crossing;
} bh_surface_candidate;

typedef struct bh_crossing_vector {
    bh_cpu_recorded_surface_crossing *items;
    size_t count;
    size_t capacity;
} bh_crossing_vector;

static bool
bh_exact_struct_header(uint32_t abi_version, uint32_t struct_size, size_t expected)
{
    return abi_version == BH_CPU_ABI_VERSION &&
        (size_t)struct_size == expected;
}

static bh_cpu_status
bh_validate_whole_ray_arguments(
    const double initial_state[8],
    const bh_cpu_kerr_returning_model *model,
    const bh_cpu_kerr_ray_options *ray,
    const bh_cpu_kerr_surface_options *surface,
    const bh_cpu_ray_path_segment *segments,
    size_t segment_capacity,
    const bh_cpu_recorded_surface_crossing *crossings,
    size_t crossing_capacity,
    const bh_cpu_kerr_ray_result *result
)
{
    if (initial_state == NULL || model == NULL || ray == NULL ||
        surface == NULL || result == NULL ||
        (ray->record_path != UINT32_C(0) && segment_capacity > 0 &&
         segments == NULL) ||
        (crossing_capacity > 0 && crossings == NULL)) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    if (!bh_exact_struct_header(
            model->abi_version, model->struct_size, sizeof(*model)) ||
        !bh_exact_struct_header(
            ray->abi_version, ray->struct_size, sizeof(*ray)) ||
        !bh_exact_struct_header(
            surface->abi_version, surface->struct_size, sizeof(*surface))) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    if (!bh_all_finite(initial_state, 8)) {
        return BH_CPU_NONFINITE_INPUT;
    }
    const double model_values[8] = {
        model->mass_m,
        model->spin_a_m,
        model->singularity_guard_m,
        model->capture_radius_m,
        model->escape_radius_m,
        model->isco_radius_over_mass,
        model->outer_radius_over_mass,
        model->asymptotic_pressure_scale_height_over_mass,
    };
    const double ray_values[10] = {
        ray->absolute_tolerance,
        ray->relative_tolerance,
        ray->initial_step,
        ray->minimum_step,
        ray->maximum_step,
        ray->maximum_affine_length,
        ray->null_residual_limit,
        ray->metric_interpolation_error_limit,
        ray->event_value_tolerance,
        ray->event_affine_tolerance,
    };
    const double surface_values[6] = {
        surface->absolute_tolerance,
        surface->relative_tolerance,
        surface->null_residual_limit,
        surface->metric_interpolation_error_limit,
        surface->surface_value_tolerance,
        surface->affine_tolerance,
    };
    if (!bh_all_finite(model_values, 8) ||
        !bh_all_finite(ray_values, 10) ||
        !bh_all_finite(surface_values, 6)) {
        return BH_CPU_NONFINITE_INPUT;
    }
    bh_cpu_status status = bh_validate_kerr_parameters(
        model->mass_m,
        model->spin_a_m,
        model->singularity_guard_m);
    if (status != BH_CPU_OK) {
        return status;
    }
    if (model->capture_radius_m <= 0.0 ||
        model->escape_radius_m <= model->capture_radius_m ||
        model->isco_radius_over_mass <= 0.0 ||
        model->outer_radius_over_mass <= model->isco_radius_over_mass ||
        model->asymptotic_pressure_scale_height_over_mass <= 0.0 ||
        model->emitting_surface_id > BH_CPU_SURFACE_UPPER ||
        model->initial_contact_side != 1) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    const double spin_magnitude = fabs(model->spin_a_m);
    const double horizon = model->mass_m + sqrt(
        (model->mass_m - spin_magnitude) *
        (model->mass_m + spin_magnitude));
    const double outer_height_factor = -expm1(
        0.5 *
        (log(model->isco_radius_over_mass) -
         log(model->outer_radius_over_mass)));
    const double outer_height =
        2.0 * model->asymptotic_pressure_scale_height_over_mass *
        outer_height_factor;
    const double outer_worldtube_radius = model->mass_m *
        bh_hypot2_impl(model->outer_radius_over_mass, outer_height);
    if (!isfinite(horizon) || !isfinite(outer_worldtube_radius) ||
        model->capture_radius_m < horizon ||
        model->capture_radius_m >=
            model->isco_radius_over_mass * model->mass_m ||
        model->escape_radius_m <= outer_worldtube_radius) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < 10; ++index) {
        if (ray_values[index] <= 0.0) {
            return BH_CPU_INVALID_ARGUMENT;
        }
    }
    for (size_t index = 0; index < 6; ++index) {
        if (surface_values[index] <= 0.0) {
            return BH_CPU_INVALID_ARGUMENT;
        }
    }
    if (ray->minimum_step > ray->initial_step ||
        ray->initial_step > ray->maximum_step ||
        ray->maximum_accepted_steps == 0 ||
        ray->maximum_rejected_steps == 0 ||
        ray->event_maximum_iterations == 0 ||
        ray->record_path > UINT32_C(1) ||
        surface->maximum_reintegrations == 0 ||
        surface->maximum_iterations == 0 ||
        surface->subdivisions_per_segment < UINT32_C(2) ||
        surface->subdivisions_per_segment > UINT32_C(128) ||
        surface->subdivisions_per_segment % UINT32_C(2) != UINT32_C(0)) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    const double policy_scale = fmax(1.0, model->mass_m);
    if (ray->absolute_tolerance > 1.0e-7 * policy_scale ||
        ray->relative_tolerance > 1.0e-7 ||
        ray->maximum_step > 2.0 * policy_scale ||
        ray->null_residual_limit > 1.0e-6 ||
        ray->metric_interpolation_error_limit > 1.0e-6 ||
        ray->event_value_tolerance > 1.0e-7 * policy_scale ||
        ray->event_affine_tolerance > 1.0e-7 * policy_scale ||
        ray->maximum_affine_length > 1.0e6 * policy_scale ||
        ray->maximum_accepted_steps > UINT64_C(1000000) ||
        ray->maximum_rejected_steps > UINT64_C(1000000) ||
        ray->event_maximum_iterations > UINT32_C(256) ||
        surface->absolute_tolerance > 1.0e-7 * policy_scale ||
        surface->relative_tolerance > 1.0e-7 ||
        surface->null_residual_limit > 1.0e-6 ||
        surface->metric_interpolation_error_limit > 1.0e-6 ||
        surface->surface_value_tolerance > 1.0e-7 ||
        surface->affine_tolerance > 1.0e-7 * policy_scale ||
        surface->maximum_iterations > UINT32_C(256) ||
        surface->maximum_reintegrations > UINT64_C(2000000)) {
        return BH_CPU_INVALID_ARGUMENT;
    }
    return BH_CPU_OK;
}

static bh_cpu_status
bh_kerr_oblate_radius_impl(
    const double state[8],
    double spin_a_m,
    double *out_radius
)
{
    const double x_m = state[1];
    const double y_m = state[2];
    const double z_m = state[3];
    const double squared_coordinates[3] = {
        x_m * x_m,
        y_m * y_m,
        z_m * z_m,
    };
    double rho_squared = 0.0;
    bh_cpu_status status =
        bh_fsum_impl(squared_coordinates, 3, &rho_squared);
    if (status != BH_CPU_OK) {
        return status;
    }
    const double spin_squared = spin_a_m * spin_a_m;
    const double difference = rho_squared - spin_squared;
    const double spin_position = spin_a_m * z_m;
    const double twice_spin_position = 2.0 * spin_position;
    if (!isfinite(difference) || !isfinite(twice_spin_position)) {
        return BH_CPU_NUMERIC_OVERFLOW;
    }
    const double discriminant_root =
        bh_hypot2_impl(difference, twice_spin_position);
    if (!isfinite(discriminant_root)) {
        return BH_CPU_NUMERIC_OVERFLOW;
    }
    double radius_squared = 0.0;
    if (difference >= 0.0) {
        radius_squared = 0.5 * (difference + discriminant_root);
    } else if (spin_position != 0.0) {
        double numerator = 2.0 * spin_position;
        numerator = numerator * spin_position;
        radius_squared = numerator / (discriminant_root - difference);
    }
    const double scale = fmax(1.0, fmax(rho_squared, spin_squared));
    const double scale_ulp = nextafter(scale, HUGE_VAL) - scale;
    if (radius_squared < -16.0 * scale_ulp) {
        return BH_CPU_NUMERIC_OVERFLOW;
    }
    const double radius = sqrt(fmax(0.0, radius_squared));
    if (!isfinite(radius)) {
        return BH_CPU_NUMERIC_OVERFLOW;
    }
    *out_radius = radius;
    return BH_CPU_OK;
}

static double
bh_scaled_state_error_norm(
    const double before[8],
    const double after[8],
    const double error[8],
    double absolute_tolerance,
    double relative_tolerance
)
{
    double result = 0.0;
    for (size_t index = 0; index < 8; ++index) {
        const double magnitude = fmax(fabs(before[index]), fabs(after[index]));
        const double scale = absolute_tolerance +
            relative_tolerance * magnitude;
        const double value = fabs(error[index]) / scale;
        if (value > result) {
            result = value;
        }
    }
    return result;
}

static bh_cpu_status
bh_trace_dopri_step(
    bh_ray_trace_context *context,
    const double state[8],
    double step,
    double out_fifth[8],
    double out_error[8]
)
{
    return bh_kerr_dopri54_step_impl(
        state,
        step,
        context->model->mass_m,
        context->model->spin_a_m,
        context->model->singularity_guard_m,
        out_fifth,
        out_error,
        &context->metric_sample_evaluations,
        &context->hamiltonian_rhs_evaluations);
}

static bh_cpu_status
bh_trace_state_null_residual(
    bh_ray_trace_context *context,
    const double state[8],
    double *out_residual
)
{
    double covariant[16];
    double inverse[16];
    double derivatives[64];
    context->metric_sample_evaluations += UINT64_C(1);
    bh_cpu_status status = bh_kerr_metric_sample_impl(
        state,
        context->model->mass_m,
        context->model->spin_a_m,
        context->model->singularity_guard_m,
        covariant,
        inverse,
        derivatives);
    if (status != BH_CPU_OK) {
        return status;
    }
    return bh_cpu_normalized_null_residual(
        inverse,
        state + 4,
        out_residual);
}

static bh_cpu_status
bh_surface_coordinates(
    const bh_cpu_kerr_returning_model *model,
    const double state[8],
    double *out_rho_over_mass,
    double *out_height_over_mass,
    bool require_azimuth
)
{
    double radius = 0.0;
    bh_cpu_status status =
        bh_kerr_oblate_radius_impl(state, model->spin_a_m, &radius);
    if (status != BH_CPU_OK) {
        return status;
    }
    if (radius <= model->singularity_guard_m) {
        return BH_CPU_GUARDED_SINGULARITY;
    }
    const double equatorial_scale =
        bh_hypot2_impl(radius, model->spin_a_m);
    const double sine_theta =
        bh_hypot2_impl(state[1], state[2]) / equatorial_scale;
    const double cosine_theta = state[3] / radius;
    const double theta = atan2(sine_theta, cosine_theta);
    if (!isfinite(theta)) {
        return BH_CPU_NUMERIC_OVERFLOW;
    }
    if (require_azimuth) {
        const double cosine_numerator =
            radius * state[1] + model->spin_a_m * state[2];
        const double sine_numerator =
            radius * state[2] - model->spin_a_m * state[1];
        if (cosine_numerator == 0.0 && sine_numerator == 0.0) {
            return BH_CPU_INVALID_ARGUMENT;
        }
        const double azimuth = atan2(sine_numerator, cosine_numerator);
        if (!isfinite(azimuth)) {
            return BH_CPU_NUMERIC_OVERFLOW;
        }
    }
    const double normalized_radius = radius / model->mass_m;
    const double rho = normalized_radius * sin(theta);
    const double height = normalized_radius * cos(theta);
    if (!isfinite(rho) || !isfinite(height) || normalized_radius <= 0.0 ||
        rho < 0.0) {
        return BH_CPU_NUMERIC_OVERFLOW;
    }
    *out_rho_over_mass = rho;
    *out_height_over_mass = height;
    return BH_CPU_OK;
}

static bh_cpu_status
bh_photosphere_height(
    const bh_cpu_kerr_returning_model *model,
    double rho,
    double *out_height
)
{
    const double inner = model->isco_radius_over_mass;
    const double outer = model->outer_radius_over_mass;
    double value_radius = rho;
    if (rho < inner) {
        const double pressure_derivative =
            0.5 * model->asymptotic_pressure_scale_height_over_mass *
            sqrt(inner) / pow(inner, 1.5);
        const double inner_slope = 2.0 * pressure_derivative;
        const double height = inner_slope * (inner - rho);
        if (!isfinite(height)) {
            return BH_CPU_NUMERIC_OVERFLOW;
        }
        *out_height = height;
        return BH_CPU_OK;
    }
    if (rho > outer) {
        value_radius = outer;
    }
    const double height_factor = -expm1(
        0.5 * (log(inner) - log(value_radius)));
    const double pressure_height =
        model->asymptotic_pressure_scale_height_over_mass * height_factor;
    const double height = 2.0 * pressure_height;
    if (!isfinite(height) || height < 0.0) {
        return BH_CPU_NUMERIC_OVERFLOW;
    }
    *out_height = height;
    return BH_CPU_OK;
}

static bh_cpu_status
bh_surface_value(
    bh_ray_trace_context *context,
    uint32_t surface_id,
    const double state[8],
    double *out_value
)
{
    double rho = 0.0;
    double signed_height = 0.0;
    bh_cpu_status status = bh_surface_coordinates(
        context->model,
        state,
        &rho,
        &signed_height,
        false);
    if (status != BH_CPU_OK) {
        return status;
    }
    double face_height = 0.0;
    status = bh_photosphere_height(context->model, rho, &face_height);
    if (status != BH_CPU_OK) {
        return status;
    }
    const double value = surface_id == BH_CPU_SURFACE_UPPER
        ? signed_height - face_height
        : -signed_height - face_height;
    if (!isfinite(value)) {
        return BH_CPU_NUMERIC_OVERFLOW;
    }
    context->surface_value_evaluations += UINT64_C(1);
    *out_value = value;
    return BH_CPU_OK;
}

static bh_cpu_status
bh_surface_values(
    bh_ray_trace_context *context,
    const double state[8],
    double out_values[2]
)
{
    bh_cpu_status status = bh_surface_value(
        context, BH_CPU_SURFACE_LOWER, state, &out_values[0]);
    if (status != BH_CPU_OK) {
        return status;
    }
    return bh_surface_value(
        context, BH_CPU_SURFACE_UPPER, state, &out_values[1]);
}

static void
bh_probe_cache_destroy(bh_surface_probe_cache *cache)
{
    free(cache->items);
    cache->items = NULL;
    cache->count = 0;
    cache->capacity = 0;
}

static bool
bh_probe_cache_reserve(bh_surface_probe_cache *cache, size_t required)
{
    if (required <= cache->capacity) {
        return true;
    }
    size_t next_capacity = cache->capacity == 0 ? 64 : cache->capacity;
    while (next_capacity < required) {
        if (next_capacity > SIZE_MAX / 2) {
            return false;
        }
        next_capacity *= 2;
    }
    if (next_capacity > SIZE_MAX / sizeof(*cache->items)) {
        return false;
    }
    void *replacement = realloc(
        cache->items,
        next_capacity * sizeof(*cache->items));
    if (replacement == NULL) {
        return false;
    }
    cache->items = replacement;
    cache->capacity = next_capacity;
    return true;
}

static bh_surface_flow
bh_probe_cache_initialize(
    bh_surface_probe_cache *cache,
    bh_ray_trace_context *context,
    const double start[8],
    bool apply_initial_contact,
    bh_cpu_status *out_status
)
{
    memset(cache, 0, sizeof(*cache));
    cache->context = context;
    memcpy(cache->start, start, sizeof(cache->start));
    if (!bh_probe_cache_reserve(cache, 1)) {
        *out_status = BH_CPU_CAPACITY_EXCEEDED;
        return BH_SURFACE_FLOW_FAILURE;
    }
    bh_surface_probe *probe = &cache->items[0];
    memcpy(probe->state, start, sizeof(probe->state));
    probe->affine_offset = 0.0;
    *out_status = bh_surface_values(context, start, probe->values);
    if (*out_status != BH_CPU_OK) {
        return BH_SURFACE_FLOW_FAILURE;
    }
    if (apply_initial_contact) {
        probe->values[context->model->emitting_surface_id] =
            (double)context->model->initial_contact_side;
    }
    cache->count = 1;
    return BH_SURFACE_FLOW_OK;
}

static bh_surface_probe *
bh_probe_cache_find(bh_surface_probe_cache *cache, double affine_offset)
{
    for (size_t index = 0; index < cache->count; ++index) {
        if (cache->items[index].affine_offset == affine_offset) {
            return &cache->items[index];
        }
    }
    return NULL;
}

static double
bh_state_mismatch_norm(
    const double expected[8],
    const double actual[8],
    double absolute_tolerance,
    double relative_tolerance
)
{
    double difference[8];
    for (size_t index = 0; index < 8; ++index) {
        difference[index] = actual[index] - expected[index];
    }
    return bh_scaled_state_error_norm(
        expected,
        actual,
        difference,
        absolute_tolerance,
        relative_tolerance);
}

static bh_surface_flow
bh_probe_cache_at(
    bh_surface_probe_cache *cache,
    double affine_offset,
    const double expected[8],
    bool has_expected,
    bh_surface_probe **out_probe,
    bh_cpu_status *out_status
)
{
    if (!isfinite(affine_offset) || affine_offset <= 0.0) {
        *out_status = BH_CPU_INVALID_ARGUMENT;
        return BH_SURFACE_FLOW_FAILURE;
    }
    bh_surface_probe *cached =
        bh_probe_cache_find(cache, affine_offset);
    if (cached != NULL) {
        if (has_expected) {
            const double mismatch = bh_state_mismatch_norm(
                expected,
                cached->state,
                cache->context->surface->absolute_tolerance,
                cache->context->surface->relative_tolerance);
            if (!isfinite(mismatch)) {
                *out_status = BH_CPU_NUMERIC_OVERFLOW;
                return BH_SURFACE_FLOW_FAILURE;
            }
            if (mismatch > 1.0) {
                *out_status = BH_CPU_OK;
                return BH_SURFACE_FLOW_RETRY_SMALLER;
            }
        }
        *out_probe = cached;
        *out_status = BH_CPU_OK;
        return BH_SURFACE_FLOW_OK;
    }
    if (cache->context->probe_reintegrations >=
        cache->context->surface->maximum_reintegrations) {
        *out_status = BH_CPU_CAPACITY_EXCEEDED;
        return BH_SURFACE_FLOW_FAILURE;
    }
    cache->context->probe_reintegrations += UINT64_C(1);
    double state[8];
    double error[8];
    *out_status = bh_trace_dopri_step(
        cache->context,
        cache->start,
        affine_offset,
        state,
        error);
    if (*out_status != BH_CPU_OK) {
        return BH_SURFACE_FLOW_FAILURE;
    }
    const double error_norm = bh_scaled_state_error_norm(
        cache->start,
        state,
        error,
        cache->context->surface->absolute_tolerance,
        cache->context->surface->relative_tolerance);
    if (!isfinite(error_norm)) {
        *out_status = BH_CPU_NUMERIC_OVERFLOW;
        return BH_SURFACE_FLOW_FAILURE;
    }
    if (error_norm > 1.0) {
        *out_status = BH_CPU_OK;
        return BH_SURFACE_FLOW_RETRY_SMALLER;
    }
    double residual = 0.0;
    *out_status = bh_trace_state_null_residual(
        cache->context,
        state,
        &residual);
    if (*out_status != BH_CPU_OK) {
        return BH_SURFACE_FLOW_FAILURE;
    }
    if (residual > cache->context->surface->null_residual_limit) {
        *out_status = BH_CPU_INCONSISTENT_METRIC;
        return BH_SURFACE_FLOW_FAILURE;
    }
    if (residual > cache->context->maximum_probe_null_residual) {
        cache->context->maximum_probe_null_residual = residual;
    }
    if (has_expected) {
        const double mismatch = bh_state_mismatch_norm(
            expected,
            state,
            cache->context->surface->absolute_tolerance,
            cache->context->surface->relative_tolerance);
        if (!isfinite(mismatch)) {
            *out_status = BH_CPU_NUMERIC_OVERFLOW;
            return BH_SURFACE_FLOW_FAILURE;
        }
        if (mismatch > 1.0) {
            *out_status = BH_CPU_OK;
            return BH_SURFACE_FLOW_RETRY_SMALLER;
        }
    }
    if (!bh_probe_cache_reserve(cache, cache->count + 1)) {
        *out_status = BH_CPU_CAPACITY_EXCEEDED;
        return BH_SURFACE_FLOW_FAILURE;
    }
    bh_surface_probe *probe = &cache->items[cache->count];
    memcpy(probe->state, state, sizeof(probe->state));
    probe->affine_offset = affine_offset;
    *out_status = bh_surface_values(
        cache->context,
        state,
        probe->values);
    if (*out_status != BH_CPU_OK) {
        return BH_SURFACE_FLOW_FAILURE;
    }
    cache->count += 1;
    *out_probe = probe;
    return BH_SURFACE_FLOW_OK;
}

static bool
bh_crossing_vector_append(
    bh_crossing_vector *vector,
    const bh_cpu_recorded_surface_crossing *crossing
)
{
    if (vector->count == vector->capacity) {
        size_t next_capacity = vector->capacity == 0 ? 8 : vector->capacity * 2;
        if (next_capacity < vector->capacity ||
            next_capacity > SIZE_MAX / sizeof(*vector->items)) {
            return false;
        }
        void *replacement = realloc(
            vector->items,
            next_capacity * sizeof(*vector->items));
        if (replacement == NULL) {
            return false;
        }
        vector->items = replacement;
        vector->capacity = next_capacity;
    }
    vector->items[vector->count] = *crossing;
    vector->count += 1;
    return true;
}

static bool
bh_candidate_append(
    bh_surface_candidate **items,
    size_t *count,
    size_t *capacity,
    const bh_surface_candidate *candidate
)
{
    if (*count == *capacity) {
        size_t next_capacity = *capacity == 0 ? 16 : *capacity * 2;
        if (next_capacity < *capacity ||
            next_capacity > SIZE_MAX / sizeof(**items)) {
            return false;
        }
        void *replacement = realloc(
            *items,
            next_capacity * sizeof(**items));
        if (replacement == NULL) {
            return false;
        }
        *items = replacement;
        *capacity = next_capacity;
    }
    (*items)[*count] = *candidate;
    *count += 1;
    return true;
}

static bh_cpu_status
bh_classify_surface_crossing(
    bh_ray_trace_context *context,
    bh_cpu_recorded_surface_crossing *crossing
)
{
    double rho = 0.0;
    double height = 0.0;
    bh_cpu_status status = bh_surface_coordinates(
        context->model,
        crossing->state,
        &rho,
        &height,
        true);
    if (status != BH_CPU_OK) {
        return status;
    }
    (void)height;
    const bool upper = crossing->surface_id == BH_CPU_SURFACE_UPPER;
    crossing->outcome = BH_CPU_RAY_OUTCOME_NONE;
    crossing->target = BH_CPU_TARGET_NONE;
    if (rho <= context->model->isco_radius_over_mass) {
        if (crossing->orientation == -1) {
            crossing->classification = upper
                ? BH_CPU_SURFACE_INWARD_UPPER_PLUNGE_ENTRY
                : BH_CPU_SURFACE_INWARD_LOWER_PLUNGE_ENTRY;
            crossing->outcome = BH_CPU_RAY_OUTCOME_PLUNGE_SINK;
            crossing->target = upper
                ? BH_CPU_TARGET_UPPER_PLUNGE_ENTRY
                : BH_CPU_TARGET_LOWER_PLUNGE_ENTRY;
        } else {
            crossing->classification = upper
                ? BH_CPU_SURFACE_OUTWARD_UPPER_PLUNGE_EXIT
                : BH_CPU_SURFACE_OUTWARD_LOWER_PLUNGE_EXIT;
        }
        return BH_CPU_OK;
    }
    if (rho > context->model->outer_radius_over_mass) {
        crossing->classification = BH_CPU_SURFACE_OUTSIDE_OUTER_TRANSPARENT;
        return BH_CPU_OK;
    }
    crossing->classification = upper
        ? BH_CPU_SURFACE_SUBSEQUENT_UPPER_CONTACT
        : BH_CPU_SURFACE_SUBSEQUENT_LOWER_CONTACT;
    crossing->outcome = BH_CPU_RAY_OUTCOME_RETURNED;
    crossing->target = upper
        ? BH_CPU_TARGET_OPAQUE_UPPER_FACE
        : BH_CPU_TARGET_OPAQUE_LOWER_FACE;
    return BH_CPU_OK;
}

static bool
bh_crossing_is_terminal(const bh_cpu_recorded_surface_crossing *crossing)
{
    return crossing->outcome != BH_CPU_RAY_OUTCOME_NONE;
}

static bh_surface_flow
bh_locate_surface_root(
    bh_surface_probe_cache *cache,
    uint32_t surface_id,
    double lower_offset,
    double upper_offset,
    double ray_affine_start,
    uint64_t segment_index,
    bh_cpu_recorded_surface_crossing *out_crossing,
    bh_cpu_status *out_status
)
{
    bh_surface_probe *lower_probe =
        bh_probe_cache_find(cache, lower_offset);
    bh_surface_probe *upper_probe =
        bh_probe_cache_find(cache, upper_offset);
    if (lower_probe == NULL || upper_probe == NULL) {
        *out_status = BH_CPU_INVALID_ARGUMENT;
        return BH_SURFACE_FLOW_FAILURE;
    }
    double lower_state[8];
    double upper_state[8];
    memcpy(lower_state, lower_probe->state, sizeof(lower_state));
    memcpy(upper_state, upper_probe->state, sizeof(upper_state));
    double lower_value = lower_probe->values[surface_id];
    double upper_value = upper_probe->values[surface_id];
    if (lower_value == 0.0 || upper_value == 0.0 ||
        signbit(lower_value) == signbit(upper_value)) {
        *out_status = BH_CPU_INVALID_ARGUMENT;
        return BH_SURFACE_FLOW_FAILURE;
    }
    const int32_t orientation =
        lower_value < 0.0 && upper_value > 0.0 ? 1 : -1;
    for (uint32_t iteration = 1;
         iteration <= cache->context->surface->maximum_iterations;
         ++iteration) {
        const double middle_offset = 0.5 * (lower_offset + upper_offset);
        if (middle_offset <= lower_offset || middle_offset >= upper_offset) {
            *out_status = BH_CPU_NUMERIC_OVERFLOW;
            return BH_SURFACE_FLOW_FAILURE;
        }
        bh_surface_probe *middle = NULL;
        const bh_surface_flow flow = bh_probe_cache_at(
            cache,
            middle_offset,
            NULL,
            false,
            &middle,
            out_status);
        if (flow != BH_SURFACE_FLOW_OK) {
            return flow;
        }
        const double middle_value = middle->values[surface_id];
        const double bracket_width = upper_offset - lower_offset;
        if (middle_value == 0.0) {
            memset(out_crossing, 0, sizeof(*out_crossing));
            memcpy(out_crossing->state, middle->state, sizeof(out_crossing->state));
            out_crossing->ray_affine_length =
                ray_affine_start + middle_offset;
            out_crossing->segment_affine_length = middle_offset;
            out_crossing->surface_value = 0.0;
            out_crossing->bracket_affine_width = bracket_width;
            out_crossing->segment_index = segment_index;
            out_crossing->iterations = iteration;
            out_crossing->orientation = orientation;
            out_crossing->surface_id = surface_id;
            *out_status = bh_classify_surface_crossing(
                cache->context, out_crossing);
            return *out_status == BH_CPU_OK
                ? BH_SURFACE_FLOW_OK
                : BH_SURFACE_FLOW_FAILURE;
        }
        if (signbit(middle_value) == signbit(lower_value)) {
            lower_offset = middle_offset;
            memcpy(lower_state, middle->state, sizeof(lower_state));
            lower_value = middle_value;
        } else {
            upper_offset = middle_offset;
            memcpy(upper_state, middle->state, sizeof(upper_state));
            upper_value = middle_value;
        }
        if (upper_offset - lower_offset <=
            cache->context->surface->affine_tolerance) {
            const bool choose_lower =
                fabs(lower_value) <= fabs(upper_value);
            const double *chosen_state = choose_lower
                ? lower_state
                : upper_state;
            const double chosen_offset = choose_lower
                ? lower_offset
                : upper_offset;
            const double chosen_value = choose_lower
                ? lower_value
                : upper_value;
            if (fabs(chosen_value) >
                cache->context->surface->surface_value_tolerance) {
                *out_status = BH_CPU_INCONSISTENT_METRIC;
                return BH_SURFACE_FLOW_FAILURE;
            }
            memset(out_crossing, 0, sizeof(*out_crossing));
            memcpy(out_crossing->state, chosen_state, sizeof(out_crossing->state));
            out_crossing->ray_affine_length =
                ray_affine_start + chosen_offset;
            out_crossing->segment_affine_length = chosen_offset;
            out_crossing->surface_value = chosen_value;
            out_crossing->bracket_affine_width =
                upper_offset - lower_offset;
            out_crossing->segment_index = segment_index;
            out_crossing->iterations = iteration;
            out_crossing->orientation = orientation;
            out_crossing->surface_id = surface_id;
            *out_status = bh_classify_surface_crossing(
                cache->context, out_crossing);
            return *out_status == BH_CPU_OK
                ? BH_SURFACE_FLOW_OK
                : BH_SURFACE_FLOW_FAILURE;
        }
    }
    *out_status = BH_CPU_CAPACITY_EXCEEDED;
    return BH_SURFACE_FLOW_FAILURE;
}

static bool
bh_terminal_endpoint_contact_is_transparent(
    bh_ray_trace_context *context,
    uint32_t surface_id,
    const bh_surface_probe *probe,
    double ray_affine_start,
    uint64_t segment_index,
    bh_cpu_status *out_status
)
{
    bh_cpu_recorded_surface_crossing decisions[2];
    const int32_t orientations[2] = {-1, 1};
    for (size_t index = 0; index < 2; ++index) {
        memset(&decisions[index], 0, sizeof(decisions[index]));
        memcpy(
            decisions[index].state,
            probe->state,
            sizeof(decisions[index].state));
        decisions[index].ray_affine_length =
            ray_affine_start + probe->affine_offset;
        decisions[index].segment_affine_length = probe->affine_offset;
        decisions[index].segment_index = segment_index;
        decisions[index].orientation = orientations[index];
        decisions[index].surface_id = surface_id;
        *out_status = bh_classify_surface_crossing(
            context, &decisions[index]);
        if (*out_status != BH_CPU_OK) {
            return false;
        }
    }
    return decisions[0].classification == decisions[1].classification &&
        decisions[0].outcome == decisions[1].outcome &&
        decisions[0].target == decisions[1].target &&
        !bh_crossing_is_terminal(&decisions[0]);
}

static void
bh_sort_candidates_stable(bh_surface_candidate *items, size_t count)
{
    for (size_t index = 1; index < count; ++index) {
        const bh_surface_candidate value = items[index];
        size_t position = index;
        while (position > 0) {
            const bh_surface_candidate *previous = &items[position - 1];
            const bool follows =
                previous->lower_offset < value.lower_offset ||
                (previous->lower_offset == value.lower_offset &&
                 previous->upper_offset <= value.upper_offset);
            if (follows) {
                break;
            }
            items[position] = items[position - 1];
            --position;
        }
        items[position] = value;
    }
}

static void
bh_sort_crossings_stable(bh_cpu_recorded_surface_crossing *items, size_t count)
{
    for (size_t index = 1; index < count; ++index) {
        const bh_cpu_recorded_surface_crossing value = items[index];
        size_t position = index;
        while (position > 0 &&
               items[position - 1].ray_affine_length >
                   value.ray_affine_length) {
            items[position] = items[position - 1];
            --position;
        }
        items[position] = value;
    }
}

static bh_surface_flow
bh_scan_surface_grid(
    bh_surface_probe_cache *cache,
    const double *offsets,
    size_t grid_count,
    double ray_affine_start,
    uint64_t segment_index,
    bool allow_transparent_terminal_contact,
    bh_crossing_vector *out_crossings,
    bh_cpu_status *out_status
)
{
    if (grid_count < 3 || grid_count > 257) {
        *out_status = BH_CPU_INVALID_ARGUMENT;
        return BH_SURFACE_FLOW_FAILURE;
    }
    const size_t subdivisions = grid_count - 1;
    bh_surface_candidate *candidates = NULL;
    size_t candidate_count = 0;
    size_t candidate_capacity = 0;
    for (uint32_t surface_id = BH_CPU_SURFACE_LOWER;
         surface_id <= BH_CPU_SURFACE_UPPER;
         ++surface_id) {
        bool zero[257] = {false};
        bool ignored[257] = {false};
        size_t zero_indices[257];
        size_t zero_count = 0;
        for (size_t index = 0; index < grid_count; ++index) {
            bh_surface_probe *probe =
                bh_probe_cache_find(cache, offsets[index]);
            if (probe == NULL) {
                free(candidates);
                *out_status = BH_CPU_INVALID_ARGUMENT;
                return BH_SURFACE_FLOW_FAILURE;
            }
            zero[index] = probe->values[surface_id] == 0.0;
        }
        if (allow_transparent_terminal_contact && zero[subdivisions]) {
            bh_surface_probe *probe =
                bh_probe_cache_find(cache, offsets[subdivisions]);
            if (probe == NULL) {
                free(candidates);
                *out_status = BH_CPU_INVALID_ARGUMENT;
                return BH_SURFACE_FLOW_FAILURE;
            }
            const bool transparent =
                bh_terminal_endpoint_contact_is_transparent(
                    cache->context,
                    surface_id,
                    probe,
                    ray_affine_start,
                    segment_index,
                    out_status);
            if (*out_status != BH_CPU_OK) {
                free(candidates);
                return BH_SURFACE_FLOW_FAILURE;
            }
            ignored[subdivisions] = transparent;
        }
        for (size_t index = 0; index < grid_count; ++index) {
            if (zero[index] && !ignored[index]) {
                zero_indices[zero_count] = index;
                ++zero_count;
            }
        }
        if (zero_count > 0 &&
            (zero_indices[0] == 0 ||
             zero_indices[zero_count - 1] == subdivisions)) {
            free(candidates);
            *out_status = BH_CPU_INCONSISTENT_METRIC;
            return BH_SURFACE_FLOW_FAILURE;
        }
        for (size_t index = 1; index < zero_count; ++index) {
            if (zero_indices[index] == zero_indices[index - 1] + 1) {
                free(candidates);
                *out_status = BH_CPU_INCONSISTENT_METRIC;
                return BH_SURFACE_FLOW_FAILURE;
            }
        }
        for (size_t zero_index = 0;
             zero_index < zero_count;
             ++zero_index) {
            const size_t index = zero_indices[zero_index];
            bh_surface_probe *left =
                bh_probe_cache_find(cache, offsets[index - 1]);
            bh_surface_probe *root =
                bh_probe_cache_find(cache, offsets[index]);
            bh_surface_probe *right =
                bh_probe_cache_find(cache, offsets[index + 1]);
            if (left == NULL || root == NULL || right == NULL) {
                free(candidates);
                *out_status = BH_CPU_INVALID_ARGUMENT;
                return BH_SURFACE_FLOW_FAILURE;
            }
            const double left_value = left->values[surface_id];
            const double right_value = right->values[surface_id];
            if (signbit(left_value) == signbit(right_value)) {
                free(candidates);
                *out_status = BH_CPU_INCONSISTENT_METRIC;
                return BH_SURFACE_FLOW_FAILURE;
            }
            bh_surface_candidate candidate;
            memset(&candidate, 0, sizeof(candidate));
            candidate.surface_id = surface_id;
            candidate.lower_offset = root->affine_offset;
            candidate.upper_offset = root->affine_offset;
            candidate.exact = true;
            memcpy(
                candidate.exact_crossing.state,
                root->state,
                sizeof(candidate.exact_crossing.state));
            candidate.exact_crossing.ray_affine_length =
                ray_affine_start + root->affine_offset;
            candidate.exact_crossing.segment_affine_length =
                root->affine_offset;
            candidate.exact_crossing.segment_index = segment_index;
            candidate.exact_crossing.orientation =
                left_value < 0.0 && right_value > 0.0 ? 1 : -1;
            candidate.exact_crossing.surface_id = surface_id;
            if (!bh_candidate_append(
                    &candidates,
                    &candidate_count,
                    &candidate_capacity,
                    &candidate)) {
                free(candidates);
                *out_status = BH_CPU_CAPACITY_EXCEEDED;
                return BH_SURFACE_FLOW_FAILURE;
            }
        }
        for (size_t index = 0; index < subdivisions; ++index) {
            if (zero[index] || zero[index + 1] ||
                ignored[index] || ignored[index + 1]) {
                continue;
            }
            bh_surface_probe *lower =
                bh_probe_cache_find(cache, offsets[index]);
            bh_surface_probe *upper =
                bh_probe_cache_find(cache, offsets[index + 1]);
            if (lower == NULL || upper == NULL) {
                free(candidates);
                *out_status = BH_CPU_INVALID_ARGUMENT;
                return BH_SURFACE_FLOW_FAILURE;
            }
            if (signbit(lower->values[surface_id]) ==
                signbit(upper->values[surface_id])) {
                continue;
            }
            const bh_surface_candidate candidate = {
                .surface_id = surface_id,
                .lower_offset = lower->affine_offset,
                .upper_offset = upper->affine_offset,
                .exact = false,
            };
            if (!bh_candidate_append(
                    &candidates,
                    &candidate_count,
                    &candidate_capacity,
                    &candidate)) {
                free(candidates);
                *out_status = BH_CPU_CAPACITY_EXCEEDED;
                return BH_SURFACE_FLOW_FAILURE;
            }
        }
    }

    bh_sort_candidates_stable(candidates, candidate_count);
    size_t group_start = 0;
    while (group_start < candidate_count) {
        size_t group_end = group_start + 1;
        double group_upper = candidates[group_start].upper_offset;
        while (group_end < candidate_count &&
               candidates[group_end].lower_offset <=
                   group_upper +
                   2.0 * cache->context->surface->affine_tolerance) {
            group_upper = fmax(
                group_upper,
                candidates[group_end].upper_offset);
            ++group_end;
        }
        const size_t root_count = group_end - group_start;
        if (root_count > SIZE_MAX /
            sizeof(bh_cpu_recorded_surface_crossing)) {
            free(candidates);
            *out_status = BH_CPU_CAPACITY_EXCEEDED;
            return BH_SURFACE_FLOW_FAILURE;
        }
        bh_cpu_recorded_surface_crossing *roots = calloc(
            root_count,
            sizeof(*roots));
        if (roots == NULL) {
            free(candidates);
            *out_status = BH_CPU_CAPACITY_EXCEEDED;
            return BH_SURFACE_FLOW_FAILURE;
        }
        for (size_t index = 0; index < root_count; ++index) {
            const bh_surface_candidate *candidate =
                &candidates[group_start + index];
            if (candidate->exact) {
                roots[index] = candidate->exact_crossing;
                *out_status = bh_classify_surface_crossing(
                    cache->context,
                    &roots[index]);
                if (*out_status != BH_CPU_OK) {
                    free(roots);
                    free(candidates);
                    return BH_SURFACE_FLOW_FAILURE;
                }
            } else {
                const bh_surface_flow flow = bh_locate_surface_root(
                    cache,
                    candidate->surface_id,
                    candidate->lower_offset,
                    candidate->upper_offset,
                    ray_affine_start,
                    segment_index,
                    &roots[index],
                    out_status);
                if (flow != BH_SURFACE_FLOW_OK) {
                    free(roots);
                    free(candidates);
                    return flow;
                }
            }
        }
        bh_sort_crossings_stable(roots, root_count);
        for (size_t index = 0; index < root_count; ++index) {
            if (index + 1 < root_count &&
                roots[index].surface_id != roots[index + 1].surface_id) {
                const double separation =
                    roots[index + 1].ray_affine_length -
                    roots[index].ray_affine_length;
                const double uncertainty = fmax(
                    2.0 * cache->context->surface->affine_tolerance,
                    roots[index].bracket_affine_width +
                        roots[index + 1].bracket_affine_width);
                if (separation <= uncertainty) {
                    free(roots);
                    free(candidates);
                    *out_status = BH_CPU_INCONSISTENT_METRIC;
                    return BH_SURFACE_FLOW_FAILURE;
                }
            }
            if (!bh_crossing_vector_append(out_crossings, &roots[index])) {
                free(roots);
                free(candidates);
                *out_status = BH_CPU_CAPACITY_EXCEEDED;
                return BH_SURFACE_FLOW_FAILURE;
            }
            if (bh_crossing_is_terminal(&roots[index])) {
                free(roots);
                free(candidates);
                *out_status = BH_CPU_OK;
                return BH_SURFACE_FLOW_OK;
            }
        }
        free(roots);
        group_start = group_end;
    }
    free(candidates);
    *out_status = BH_CPU_OK;
    return BH_SURFACE_FLOW_OK;
}

static bool
bh_crossing_prefix_keys_equal(
    const bh_crossing_vector *first,
    const bh_crossing_vector *second
)
{
    if (first->count != second->count) {
        return false;
    }
    for (size_t index = 0; index < first->count; ++index) {
        const bh_cpu_recorded_surface_crossing *left = &first->items[index];
        const bh_cpu_recorded_surface_crossing *right = &second->items[index];
        if (left->surface_id != right->surface_id ||
            left->orientation != right->orientation ||
            left->classification != right->classification ||
            left->outcome != right->outcome ||
            left->target != right->target) {
            return false;
        }
    }
    return true;
}

static bh_cpu_status
bh_crossing_phase_difference(
    const bh_cpu_recorded_surface_crossing *coarse,
    const bh_cpu_recorded_surface_crossing *fine,
    double *out_event_difference,
    double *out_covector_difference
)
{
    double event_terms[4];
    double coarse_covector_terms[4];
    double fine_covector_terms[4];
    double covector_difference_terms[4];
    for (size_t index = 0; index < 4; ++index) {
        const double event_delta =
            coarse->state[index] - fine->state[index];
        event_terms[index] = event_delta * event_delta;
        coarse_covector_terms[index] =
            coarse->state[4 + index] * coarse->state[4 + index];
        fine_covector_terms[index] =
            fine->state[4 + index] * fine->state[4 + index];
        const double covector_delta =
            coarse->state[4 + index] - fine->state[4 + index];
        covector_difference_terms[index] =
            covector_delta * covector_delta;
    }
    double event_sum = 0.0;
    double coarse_sum = 0.0;
    double fine_sum = 0.0;
    double difference_sum = 0.0;
    bh_cpu_status status = bh_fsum_impl(event_terms, 4, &event_sum);
    if (status == BH_CPU_OK) {
        status = bh_fsum_impl(coarse_covector_terms, 4, &coarse_sum);
    }
    if (status == BH_CPU_OK) {
        status = bh_fsum_impl(fine_covector_terms, 4, &fine_sum);
    }
    if (status == BH_CPU_OK) {
        status = bh_fsum_impl(
            covector_difference_terms, 4, &difference_sum);
    }
    if (status != BH_CPU_OK) {
        return status;
    }
    const double event_difference = sqrt(event_sum);
    const double covector_scale = fmax(
        sqrt(coarse_sum),
        fmax(sqrt(fine_sum), 1.0e-300));
    const double covector_difference =
        sqrt(difference_sum) / covector_scale;
    if (!isfinite(event_difference) || !isfinite(covector_difference)) {
        return BH_CPU_NUMERIC_OVERFLOW;
    }
    *out_event_difference = event_difference;
    *out_covector_difference = covector_difference;
    return BH_CPU_OK;
}

static bh_surface_flow
bh_locate_converged_surface_prefix(
    bh_ray_trace_context *context,
    const double start[8],
    const double end[8],
    double full_step,
    double ray_affine_start,
    uint64_t segment_index,
    bool allow_transparent_terminal_contact,
    bh_crossing_vector *out_verified,
    double *out_maximum_event_difference,
    double *out_maximum_covector_difference,
    bh_cpu_status *out_status
)
{
    bh_surface_probe_cache cache;
    const bool apply_initial_contact =
        ray_affine_start == 0.0 && segment_index == 0;
    bh_surface_flow flow = bh_probe_cache_initialize(
        &cache,
        context,
        start,
        apply_initial_contact,
        out_status);
    if (flow != BH_SURFACE_FLOW_OK) {
        bh_probe_cache_destroy(&cache);
        return flow;
    }
    const size_t base_subdivisions =
        (size_t)context->surface->subdivisions_per_segment;
    const size_t verified_subdivisions = 2 * base_subdivisions;
    double verified_offsets[257];
    double base_offsets[129];
    verified_offsets[0] = 0.0;
    for (size_t subdivision = 1;
         subdivision <= verified_subdivisions;
         ++subdivision) {
        const double offset = full_step * (double)subdivision /
            (double)verified_subdivisions;
        verified_offsets[subdivision] = offset;
        bh_surface_probe *probe = NULL;
        flow = bh_probe_cache_at(
            &cache,
            offset,
            end,
            subdivision == verified_subdivisions,
            &probe,
            out_status);
        if (flow != BH_SURFACE_FLOW_OK) {
            bh_probe_cache_destroy(&cache);
            return flow;
        }
        (void)probe;
    }
    for (size_t subdivision = 0;
         subdivision <= base_subdivisions;
         ++subdivision) {
        base_offsets[subdivision] = verified_offsets[2 * subdivision];
    }
    bh_crossing_vector base = {0};
    flow = bh_scan_surface_grid(
        &cache,
        base_offsets,
        base_subdivisions + 1,
        ray_affine_start,
        segment_index,
        allow_transparent_terminal_contact,
        &base,
        out_status);
    if (flow == BH_SURFACE_FLOW_OK) {
        flow = bh_scan_surface_grid(
            &cache,
            verified_offsets,
            verified_subdivisions + 1,
            ray_affine_start,
            segment_index,
            allow_transparent_terminal_contact,
            out_verified,
            out_status);
    }
    if (flow != BH_SURFACE_FLOW_OK) {
        free(base.items);
        bh_probe_cache_destroy(&cache);
        return flow;
    }
    if (!bh_crossing_prefix_keys_equal(&base, out_verified)) {
        free(base.items);
        bh_probe_cache_destroy(&cache);
        *out_status = BH_CPU_OK;
        return BH_SURFACE_FLOW_RETRY_SMALLER;
    }
    double maximum_event = 0.0;
    double maximum_covector = 0.0;
    for (size_t index = 0; index < base.count; ++index) {
        const double affine_difference = fabs(
            base.items[index].ray_affine_length -
            out_verified->items[index].ray_affine_length);
        if (affine_difference >
            2.0 * context->surface->affine_tolerance) {
            free(base.items);
            bh_probe_cache_destroy(&cache);
            *out_status = BH_CPU_OK;
            return BH_SURFACE_FLOW_RETRY_SMALLER;
        }
        double event_difference = 0.0;
        double covector_difference = 0.0;
        *out_status = bh_crossing_phase_difference(
            &base.items[index],
            &out_verified->items[index],
            &event_difference,
            &covector_difference);
        if (*out_status != BH_CPU_OK) {
            free(base.items);
            bh_probe_cache_destroy(&cache);
            return BH_SURFACE_FLOW_FAILURE;
        }
        maximum_event = fmax(maximum_event, event_difference);
        maximum_covector = fmax(maximum_covector, covector_difference);
    }
    free(base.items);
    bh_probe_cache_destroy(&cache);
    *out_maximum_event_difference = maximum_event;
    *out_maximum_covector_difference = maximum_covector;
    *out_status = BH_CPU_OK;
    return BH_SURFACE_FLOW_OK;
}

typedef struct bh_termination_crossing {
    bool exists;
    uint32_t outcome;
    uint32_t target;
} bh_termination_crossing;

typedef struct bh_segment_vector {
    bh_cpu_ray_path_segment *items;
    size_t count;
    size_t capacity;
} bh_segment_vector;

static bool
bh_segment_vector_append(
    bh_segment_vector *vector,
    const bh_cpu_ray_path_segment *segment
)
{
    if (vector->count == vector->capacity) {
        size_t next_capacity = vector->capacity == 0 ? 256 : vector->capacity * 2;
        if (next_capacity < vector->capacity ||
            next_capacity > SIZE_MAX / sizeof(*vector->items)) {
            return false;
        }
        void *replacement = realloc(
            vector->items,
            next_capacity * sizeof(*vector->items));
        if (replacement == NULL) {
            return false;
        }
        vector->items = replacement;
        vector->capacity = next_capacity;
    }
    vector->items[vector->count] = *segment;
    vector->count += 1;
    return true;
}

static bh_cpu_status
bh_termination_crossing_for_step(
    const bh_cpu_kerr_returning_model *model,
    const double previous[8],
    const double current[8],
    bh_termination_crossing *out_crossing
)
{
    double previous_radius = 0.0;
    double current_radius = 0.0;
    bh_cpu_status status = bh_kerr_oblate_radius_impl(
        previous, model->spin_a_m, &previous_radius);
    if (status == BH_CPU_OK) {
        status = bh_kerr_oblate_radius_impl(
            current, model->spin_a_m, &current_radius);
    }
    if (status != BH_CPU_OK) {
        return status;
    }
    *out_crossing = (bh_termination_crossing){0};
    const double capture_before =
        previous_radius - model->capture_radius_m;
    const double capture_after =
        current_radius - model->capture_radius_m;
    if (capture_before > 0.0 && capture_after <= 0.0) {
        out_crossing->exists = true;
        out_crossing->outcome = BH_CPU_RAY_OUTCOME_CAPTURED;
        out_crossing->target = BH_CPU_TARGET_KERR_STRETCHED_HORIZON;
        return BH_CPU_OK;
    }
    const double escape_before = previous_radius - model->escape_radius_m;
    const double escape_after = current_radius - model->escape_radius_m;
    if (escape_before < 0.0 && escape_after >= 0.0) {
        out_crossing->exists = true;
        out_crossing->outcome = BH_CPU_RAY_OUTCOME_ESCAPED;
        out_crossing->target = BH_CPU_TARGET_KERR_ESCAPE_WORLDTUBE;
    }
    return BH_CPU_OK;
}

static bh_cpu_status
bh_termination_needs_refinement(
    const bh_cpu_kerr_returning_model *model,
    const double previous[8],
    const double current[8],
    bool *out_needs_refinement
)
{
    double previous_radius = 0.0;
    double current_radius = 0.0;
    bh_cpu_status status = bh_kerr_oblate_radius_impl(
        previous, model->spin_a_m, &previous_radius);
    if (status == BH_CPU_OK) {
        status = bh_kerr_oblate_radius_impl(
            current, model->spin_a_m, &current_radius);
    }
    if (status != BH_CPU_OK) {
        return status;
    }
    if (previous_radius <= model->capture_radius_m ||
        current_radius <= model->capture_radius_m) {
        *out_needs_refinement = false;
        return BH_CPU_OK;
    }
    const double radius_squared =
        model->capture_radius_m * model->capture_radius_m;
    const double equatorial_squared = radius_squared +
        model->spin_a_m * model->spin_a_m;
    const double delta[3] = {
        current[1] - previous[1],
        current[2] - previous[2],
        current[3] - previous[3],
    };
    const double weights[3] = {
        1.0 / equatorial_squared,
        1.0 / equatorial_squared,
        1.0 / radius_squared,
    };
    double quadratic_terms[3];
    double linear_terms[3];
    for (size_t index = 0; index < 3; ++index) {
        quadratic_terms[index] = weights[index] * delta[index] * delta[index];
        linear_terms[index] = weights[index] * previous[index + 1] * delta[index];
    }
    double quadratic = 0.0;
    double linear_half = 0.0;
    status = bh_fsum_impl(quadratic_terms, 3, &quadratic);
    if (status == BH_CPU_OK) {
        status = bh_fsum_impl(linear_terms, 3, &linear_half);
    }
    if (status != BH_CPU_OK) {
        return status;
    }
    if (quadratic == 0.0) {
        *out_needs_refinement = false;
        return BH_CPU_OK;
    }
    double fraction = -linear_half / quadratic;
    fraction = fmin(1.0, fmax(0.0, fraction));
    double minimum_terms[3];
    for (size_t index = 0; index < 3; ++index) {
        const double coordinate =
            previous[index + 1] + fraction * delta[index];
        minimum_terms[index] = weights[index] * coordinate * coordinate;
    }
    double minimum_level = 0.0;
    status = bh_fsum_impl(minimum_terms, 3, &minimum_level);
    if (status != BH_CPU_OK) {
        return status;
    }
    *out_needs_refinement = minimum_level <= 1.0;
    return BH_CPU_OK;
}

static bh_cpu_status
bh_termination_value(
    const bh_cpu_kerr_returning_model *model,
    const double state[8],
    uint32_t outcome,
    double *out_value
)
{
    double radius = 0.0;
    bh_cpu_status status = bh_kerr_oblate_radius_impl(
        state, model->spin_a_m, &radius);
    if (status != BH_CPU_OK) {
        return status;
    }
    if (outcome == BH_CPU_RAY_OUTCOME_CAPTURED) {
        *out_value = radius - model->capture_radius_m;
    } else if (outcome == BH_CPU_RAY_OUTCOME_ESCAPED) {
        *out_value = radius - model->escape_radius_m;
    } else {
        return BH_CPU_INVALID_ARGUMENT;
    }
    return isfinite(*out_value) ? BH_CPU_OK : BH_CPU_NUMERIC_OVERFLOW;
}

static bh_cpu_status
bh_locate_termination_event(
    bh_ray_trace_context *context,
    const double previous[8],
    const double candidate[8],
    double full_step,
    const bh_termination_crossing *crossing,
    double out_state[8],
    double *out_step
)
{
    double lower_step = 0.0;
    double upper_step = full_step;
    double lower_state[8];
    double upper_state[8];
    memcpy(lower_state, previous, sizeof(lower_state));
    memcpy(upper_state, candidate, sizeof(upper_state));
    double lower_value = 0.0;
    double upper_value = 0.0;
    bh_cpu_status status = bh_termination_value(
        context->model,
        lower_state,
        crossing->outcome,
        &lower_value);
    if (status == BH_CPU_OK) {
        status = bh_termination_value(
            context->model,
            upper_state,
            crossing->outcome,
            &upper_value);
    }
    if (status != BH_CPU_OK) {
        return status;
    }
    if (fabs(lower_value) <= context->ray->event_value_tolerance) {
        memcpy(out_state, lower_state, sizeof(lower_state));
        *out_step = lower_step;
        return BH_CPU_OK;
    }
    if (fabs(upper_value) <= context->ray->event_value_tolerance) {
        memcpy(out_state, upper_state, sizeof(upper_state));
        *out_step = upper_step;
        return BH_CPU_OK;
    }
    if (signbit(lower_value) == signbit(upper_value)) {
        return BH_CPU_INCONSISTENT_METRIC;
    }
    for (uint32_t iteration = 0;
         iteration < context->ray->event_maximum_iterations;
         ++iteration) {
        const double middle_step = 0.5 * (lower_step + upper_step);
        double middle_state[8];
        double middle_error[8];
        status = bh_trace_dopri_step(
            context,
            previous,
            middle_step,
            middle_state,
            middle_error);
        if (status != BH_CPU_OK) {
            return status;
        }
        (void)middle_error;
        double middle_value = 0.0;
        status = bh_termination_value(
            context->model,
            middle_state,
            crossing->outcome,
            &middle_value);
        if (status != BH_CPU_OK) {
            return status;
        }
        if (fabs(middle_value) <= context->ray->event_value_tolerance ||
            upper_step - lower_step <=
                context->ray->event_affine_tolerance) {
            memcpy(out_state, middle_state, sizeof(middle_state));
            *out_step = middle_step;
            return BH_CPU_OK;
        }
        if (signbit(lower_value) == signbit(middle_value)) {
            lower_step = middle_step;
            memcpy(lower_state, middle_state, sizeof(lower_state));
            lower_value = middle_value;
        } else {
            upper_step = middle_step;
            memcpy(upper_state, middle_state, sizeof(upper_state));
            upper_value = middle_value;
        }
    }
    const bool choose_lower = fabs(lower_value) <= fabs(upper_value);
    const double chosen_value = choose_lower ? lower_value : upper_value;
    if (fabs(chosen_value) > context->ray->event_value_tolerance) {
        return BH_CPU_CAPACITY_EXCEEDED;
    }
    memcpy(out_state, choose_lower ? lower_state : upper_state, 8 * sizeof(double));
    *out_step = choose_lower ? lower_step : upper_step;
    return BH_CPU_OK;
}

static void
bh_fill_ray_result(
    const bh_ray_trace_context *context,
    const double state[8],
    uint32_t outcome,
    uint32_t target,
    uint32_t failure,
    double affine_length,
    uint64_t accepted,
    uint64_t rejected,
    double maximum_null,
    size_t segment_count,
    size_t crossing_count,
    double initial_contact_value,
    bh_cpu_kerr_ray_result *out_result
)
{
    *out_result = (bh_cpu_kerr_ray_result){
        .abi_version = BH_CPU_ABI_VERSION,
        .struct_size = (uint32_t)sizeof(*out_result),
        .outcome = outcome,
        .terminal_target = target,
        .failure = failure,
        .topology_converged =
            outcome != BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE &&
            outcome != BH_CPU_RAY_OUTCOME_UNRESOLVED
                ? UINT32_C(1)
                : UINT32_C(0),
        .initial_contact_surface_id = context->model->emitting_surface_id,
        .initial_contact_side = context->model->initial_contact_side,
        .affine_length = affine_length,
        .maximum_null_residual = maximum_null,
        .maximum_metric_interpolation_error = 0.0,
        .maximum_probe_event_difference = 0.0,
        .maximum_probe_covector_relative_difference = 0.0,
        .initial_contact_actual_surface_value = initial_contact_value,
        .initial_contact_surface_value_tolerance =
            context->surface->surface_value_tolerance,
        .accepted_steps = accepted,
        .rejected_steps = rejected,
        .segment_count = (uint64_t)segment_count,
        .crossing_count = (uint64_t)crossing_count,
        .probe_reintegrations = context->probe_reintegrations,
        .surface_value_evaluations = context->surface_value_evaluations,
        .metric_sample_evaluations = context->metric_sample_evaluations,
        .hamiltonian_rhs_evaluations = context->hamiltonian_rhs_evaluations,
    };
    memcpy(out_result->terminal_state, state, sizeof(out_result->terminal_state));
}

static bh_cpu_status
bh_publish_whole_ray(
    const bh_segment_vector *segments,
    const bh_crossing_vector *crossings,
    bh_cpu_ray_path_segment *out_segments,
    size_t segment_capacity,
    bh_cpu_recorded_surface_crossing *out_crossings,
    size_t crossing_capacity,
    const bh_cpu_kerr_ray_result *result,
    bh_cpu_kerr_ray_result *out_result
)
{
    if (segments->count > segment_capacity ||
        crossings->count > crossing_capacity ||
        (segments->count > 0 && out_segments == NULL) ||
        (crossings->count > 0 && out_crossings == NULL)) {
        return BH_CPU_CAPACITY_EXCEEDED;
    }
    if (segments->count > 0) {
        memcpy(
            out_segments,
            segments->items,
            segments->count * sizeof(*segments->items));
    }
    if (crossings->count > 0) {
        memcpy(
            out_crossings,
            crossings->items,
            crossings->count * sizeof(*crossings->items));
    }
    *out_result = *result;
    return BH_CPU_OK;
}

bh_cpu_status
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
)
{
    const bh_cpu_status environment_status = bh_cpu_require_strict_fp();
    if (environment_status != BH_CPU_OK) {
        return environment_status;
    }
    bh_cpu_status status = bh_validate_whole_ray_arguments(
        initial_state,
        model,
        ray_options,
        surface_options,
        out_segments,
        segment_capacity,
        out_crossings,
        crossing_capacity,
        out_result);
    if (status != BH_CPU_OK) {
        return status;
    }

    bh_ray_trace_context context = {
        .model = model,
        .ray = ray_options,
        .surface = surface_options,
    };
    bh_segment_vector segments = {0};
    bh_crossing_vector crossings = {0};
    bh_cpu_kerr_ray_result result;
    double state[8];
    memcpy(state, initial_state, sizeof(state));
    double affine_length = 0.0;
    double step = ray_options->initial_step;
    uint64_t accepted = 0;
    uint64_t rejected = 0;
    double maximum_null = 0.0;
    double initial_contact_value = 0.0;
    double maximum_probe_event_difference = 0.0;
    double maximum_probe_covector_difference = 0.0;

    status = bh_trace_state_null_residual(
        &context, state, &maximum_null);
    if (status != BH_CPU_OK ||
        maximum_null > ray_options->null_residual_limit) {
        bh_fill_ray_result(
            &context,
            state,
            BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE,
            BH_CPU_TARGET_NONE,
            BH_CPU_RAY_FAILURE_INITIAL_NULL,
            affine_length,
            accepted,
            rejected,
            maximum_null,
            segments.count,
            crossings.count,
            initial_contact_value,
            &result);
        status = bh_publish_whole_ray(
            &segments, &crossings, out_segments, segment_capacity,
            out_crossings, crossing_capacity, &result, out_result);
        free(segments.items);
        free(crossings.items);
        return status;
    }
    status = bh_surface_value(
        &context,
        model->emitting_surface_id,
        state,
        &initial_contact_value);
    if (status != BH_CPU_OK ||
        fabs(initial_contact_value) >
            surface_options->surface_value_tolerance) {
        free(segments.items);
        free(crossings.items);
        return status == BH_CPU_OK ? BH_CPU_INVALID_ARGUMENT : status;
    }
    double initial_rho = 0.0;
    double initial_height = 0.0;
    status = bh_surface_coordinates(
        model,
        state,
        &initial_rho,
        &initial_height,
        false);
    if (status != BH_CPU_OK ||
        initial_rho <= model->isco_radius_over_mass ||
        initial_rho > model->outer_radius_over_mass) {
        free(segments.items);
        free(crossings.items);
        return status == BH_CPU_OK ? BH_CPU_INVALID_ARGUMENT : status;
    }
    (void)initial_height;
    double initial_radius = 0.0;
    status = bh_kerr_oblate_radius_impl(
        state, model->spin_a_m, &initial_radius);
    if (status != BH_CPU_OK) {
        free(segments.items);
        free(crossings.items);
        return status;
    }
    if (initial_radius <= model->capture_radius_m ||
        initial_radius >= model->escape_radius_m) {
        free(segments.items);
        free(crossings.items);
        return BH_CPU_INVALID_ARGUMENT;
    }

    uint32_t final_outcome = BH_CPU_RAY_OUTCOME_NONE;
    uint32_t final_target = BH_CPU_TARGET_NONE;
    uint32_t final_failure = BH_CPU_RAY_FAILURE_NONE;
    while (affine_length < ray_options->maximum_affine_length) {
        if (accepted >= ray_options->maximum_accepted_steps) {
            final_outcome = BH_CPU_RAY_OUTCOME_UNRESOLVED;
            final_failure = BH_CPU_RAY_FAILURE_ACCEPTED_BUDGET;
            break;
        }
        const double remaining =
            ray_options->maximum_affine_length - affine_length;
        step = fmin(step, remaining);
        if (step < ray_options->minimum_step) {
            final_outcome = BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE;
            final_failure = BH_CPU_RAY_FAILURE_MINIMUM_STEP;
            break;
        }
        double candidate[8];
        double error[8];
        status = bh_trace_dopri_step(
            &context, state, step, candidate, error);
        if (status != BH_CPU_OK) {
            final_outcome = BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE;
            final_failure = BH_CPU_RAY_FAILURE_NUMERIC;
            break;
        }
        double error_norm = bh_scaled_state_error_norm(
            state,
            candidate,
            error,
            ray_options->absolute_tolerance,
            ray_options->relative_tolerance);
        if (!isfinite(error_norm)) {
            error_norm = HUGE_VAL;
        }
        if (error_norm > 1.0) {
            rejected += UINT64_C(1);
            if (rejected > ray_options->maximum_rejected_steps) {
                final_outcome = BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE;
                final_failure = BH_CPU_RAY_FAILURE_REJECTED_BUDGET;
                break;
            }
            const double factor =
                fmax(0.1, 0.9 * pow(error_norm, -0.2));
            step *= factor;
            continue;
        }

        double previous[8];
        memcpy(previous, state, sizeof(previous));
        double accepted_step = step;
        bh_termination_crossing termination_crossing = {0};
        status = bh_termination_crossing_for_step(
            model,
            previous,
            candidate,
            &termination_crossing);
        if (status != BH_CPU_OK) {
            final_outcome = BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE;
            final_failure = BH_CPU_RAY_FAILURE_NUMERIC;
            break;
        }
        if (!termination_crossing.exists) {
            bool needs_refinement = false;
            status = bh_termination_needs_refinement(
                model,
                previous,
                candidate,
                &needs_refinement);
            if (status != BH_CPU_OK) {
                final_outcome = BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE;
                final_failure = BH_CPU_RAY_FAILURE_NUMERIC;
                break;
            }
            if (needs_refinement) {
                rejected += UINT64_C(1);
                if (rejected > ray_options->maximum_rejected_steps) {
                    final_outcome = BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE;
                    final_failure =
                        BH_CPU_RAY_FAILURE_EVENT_REFINEMENT_BUDGET;
                    break;
                }
                step *= 0.5;
                continue;
            }
        } else {
            double localized[8];
            status = bh_locate_termination_event(
                &context,
                previous,
                candidate,
                step,
                &termination_crossing,
                localized,
                &accepted_step);
            if (status != BH_CPU_OK) {
                final_outcome = BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE;
                final_failure = BH_CPU_RAY_FAILURE_NUMERIC;
                break;
            }
            memcpy(candidate, localized, sizeof(candidate));
        }

        bh_crossing_vector step_crossings = {0};
        double step_probe_event_difference = 0.0;
        double step_probe_covector_difference = 0.0;
        const bh_surface_flow surface_flow =
            bh_locate_converged_surface_prefix(
                &context,
                previous,
                candidate,
                accepted_step,
                affine_length,
                accepted,
                termination_crossing.exists,
                &step_crossings,
                &step_probe_event_difference,
                &step_probe_covector_difference,
                &status);
        if (surface_flow == BH_SURFACE_FLOW_RETRY_SMALLER) {
            free(step_crossings.items);
            rejected += UINT64_C(1);
            if (rejected > ray_options->maximum_rejected_steps) {
                final_outcome = BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE;
                final_failure =
                    BH_CPU_RAY_FAILURE_SURFACE_REFINEMENT_BUDGET;
                break;
            }
            step *= 0.5;
            continue;
        }
        if (surface_flow == BH_SURFACE_FLOW_FAILURE) {
            free(step_crossings.items);
            final_outcome = BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE;
            final_failure = BH_CPU_RAY_FAILURE_SURFACE;
            break;
        }
        if (step_probe_event_difference > maximum_probe_event_difference) {
            maximum_probe_event_difference = step_probe_event_difference;
        }
        if (step_probe_covector_difference >
            maximum_probe_covector_difference) {
            maximum_probe_covector_difference = step_probe_covector_difference;
        }

        const bh_cpu_recorded_surface_crossing *interior_terminal = NULL;
        if (step_crossings.count > 0 &&
            bh_crossing_is_terminal(
                &step_crossings.items[step_crossings.count - 1])) {
            interior_terminal =
                &step_crossings.items[step_crossings.count - 1];
            memcpy(candidate, interior_terminal->state, sizeof(candidate));
            accepted_step = interior_terminal->segment_affine_length;
            if (accepted_step <= 0.0) {
                free(step_crossings.items);
                final_outcome = BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE;
                final_failure = BH_CPU_RAY_FAILURE_SURFACE;
                break;
            }
        }

        double endpoint_null = 0.0;
        status = bh_trace_state_null_residual(
            &context, candidate, &endpoint_null);
        if (status != BH_CPU_OK) {
            free(step_crossings.items);
            final_outcome = BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE;
            final_failure = BH_CPU_RAY_FAILURE_NUMERIC;
            break;
        }
        double midpoint[8];
        double midpoint_error[8];
        status = bh_trace_dopri_step(
            &context,
            previous,
            0.5 * accepted_step,
            midpoint,
            midpoint_error);
        if (status != BH_CPU_OK) {
            free(step_crossings.items);
            final_outcome = BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE;
            final_failure = BH_CPU_RAY_FAILURE_NUMERIC;
            break;
        }
        (void)midpoint_error;
        double midpoint_null = 0.0;
        status = bh_trace_state_null_residual(
            &context, midpoint, &midpoint_null);
        if (status != BH_CPU_OK) {
            free(step_crossings.items);
            final_outcome = BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE;
            final_failure = BH_CPU_RAY_FAILURE_NUMERIC;
            break;
        }

        memcpy(state, candidate, sizeof(state));
        affine_length += accepted_step;
        accepted += UINT64_C(1);
        maximum_null = fmax(maximum_null, endpoint_null);
        maximum_null = fmax(maximum_null, midpoint_null);
        maximum_null = fmax(
            maximum_null,
            context.maximum_probe_null_residual);
        if (maximum_null > ray_options->null_residual_limit) {
            free(step_crossings.items);
            final_outcome = BH_CPU_RAY_OUTCOME_INTEGRATOR_FAILURE;
            final_failure = BH_CPU_RAY_FAILURE_NULL_LIMIT;
            break;
        }
        if (ray_options->record_path != UINT32_C(0)) {
            bh_cpu_ray_path_segment segment;
            memcpy(segment.start, previous, sizeof(segment.start));
            memcpy(segment.end, state, sizeof(segment.end));
            memcpy(segment.midpoint, midpoint, sizeof(segment.midpoint));
            segment.affine_length = accepted_step;
            segment.midpoint_null_residual = midpoint_null;
            if (!bh_segment_vector_append(&segments, &segment)) {
                free(step_crossings.items);
                free(segments.items);
                free(crossings.items);
                return BH_CPU_CAPACITY_EXCEEDED;
            }
        }
        for (size_t index = 0; index < step_crossings.count; ++index) {
            if (!bh_crossing_vector_append(
                    &crossings, &step_crossings.items[index])) {
                free(step_crossings.items);
                free(segments.items);
                free(crossings.items);
                return BH_CPU_CAPACITY_EXCEEDED;
            }
        }
        if (interior_terminal != NULL) {
            final_outcome = interior_terminal->outcome;
            final_target = interior_terminal->target;
            free(step_crossings.items);
            break;
        }
        free(step_crossings.items);
        if (termination_crossing.exists) {
            final_outcome = termination_crossing.outcome;
            final_target = termination_crossing.target;
            break;
        }
        const double factor = error_norm == 0.0
            ? 5.0
            : fmin(5.0, fmax(0.2, 0.9 * pow(error_norm, -0.2)));
        step = fmin(ray_options->maximum_step, step * factor);
    }
    if (final_outcome == BH_CPU_RAY_OUTCOME_NONE) {
        final_outcome = BH_CPU_RAY_OUTCOME_UNRESOLVED;
        final_failure = BH_CPU_RAY_FAILURE_AFFINE_BUDGET;
    }
    bh_fill_ray_result(
        &context,
        state,
        final_outcome,
        final_target,
        final_failure,
        affine_length,
        accepted,
        rejected,
        maximum_null,
        segments.count,
        crossings.count,
        initial_contact_value,
        &result);
    result.maximum_probe_event_difference = maximum_probe_event_difference;
    result.maximum_probe_covector_relative_difference =
        maximum_probe_covector_difference;
    status = bh_publish_whole_ray(
        &segments,
        &crossings,
        out_segments,
        segment_capacity,
        out_crossings,
        crossing_capacity,
        &result,
        out_result);
    free(segments.items);
    free(crossings.items);
    return status;
}

const char *
bh_cpu_status_string(bh_cpu_status status)
{
    switch (status) {
    case BH_CPU_OK:
        return "ok";
    case BH_CPU_INVALID_ARGUMENT:
        return "invalid argument";
    case BH_CPU_INVALID_FP_ENVIRONMENT:
        return "invalid floating-point environment";
    case BH_CPU_NONFINITE_INPUT:
        return "non-finite input";
    case BH_CPU_NUMERIC_OVERFLOW:
        return "numeric overflow";
    case BH_CPU_INCONSISTENT_METRIC:
        return "inconsistent metric";
    case BH_CPU_TOO_MANY_TERMS:
        return "too many summation terms";
    case BH_CPU_GUARDED_SINGULARITY:
        return "guarded Kerr singularity";
    case BH_CPU_CAPACITY_EXCEEDED:
        return "output or work capacity exceeded";
    default:
        return "unknown status";
    }
}
