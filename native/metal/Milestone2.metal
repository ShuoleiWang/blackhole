#include <metal_stdlib>

using namespace metal;

// A scaled float-float expansion.  The represented value is
//
//     (hi + lo) * 2^exponent.
//
// Non-zero values are normalized so 0.5 <= abs(hi) < 1.0.  Consequently the
// low word is a mantissa residual (normally around 2^-24), rather than a word
// at the value's absolute exponent.  This avoids Metal's FP32 subnormal flush
// for low words of small binary64 values without changing the scientific
// tolerance policy.
struct ScaledDD {
    float hi;
    float lo;
    int exponent;
    uint status;
};

constant uint sdd_status_ok = 0u;
constant uint sdd_status_nonfinite = 1u;
constant uint sdd_status_divide_by_zero = 2u;
constant uint sdd_status_negative_sqrt = 3u;

struct RawDD {
    float hi;
    float lo;
};

static inline RawDD raw_make(float hi, float lo) {
    return RawDD{hi, lo};
}

static inline RawDD two_sum(float a, float b) {
    const float s = a + b;
    const float bb = s - a;
    const float e = (a - (s - bb)) + (b - bb);
    return raw_make(s, e);
}

// The caller guarantees |a| >= |b|.
static inline RawDD quick_two_sum(float a, float b) {
    const float s = a + b;
    const float e = b - (s - a);
    return raw_make(s, e);
}

static inline RawDD two_product(float a, float b) {
    const float p = a * b;
    // Explicit FMA recovers the product rounding residual.  The shader is
    // built with safe math; surrounding expressions are not implicitly
    // contracted.
    const float e = fma(a, b, -p);
    return raw_make(p, e);
}

static inline RawDD raw_add(RawDD a, RawDD b) {
    RawDD s = two_sum(a.hi, b.hi);
    RawDD t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return quick_two_sum(s.hi, s.lo);
}

static inline RawDD raw_negate(RawDD a) {
    return raw_make(-a.hi, -a.lo);
}

static inline RawDD raw_subtract(RawDD a, RawDD b) {
    return raw_add(a, raw_negate(b));
}

static inline RawDD raw_multiply(RawDD a, RawDD b) {
    RawDD p = two_product(a.hi, b.hi);
    p.lo += a.hi * b.lo;
    p.lo += a.lo * b.hi;
    p = quick_two_sum(p.hi, p.lo);
    p.lo += a.lo * b.lo;
    return quick_two_sum(p.hi, p.lo);
}

static inline RawDD raw_divide(RawDD a, RawDD b) {
    const float q1 = a.hi / b.hi;
    RawDD q = raw_make(q1, 0.0f);
    RawDD r = raw_subtract(a, raw_multiply(b, q));
    const float q2 = r.hi / b.hi;
    q = raw_add(q, raw_make(q2, 0.0f));
    r = raw_subtract(r, raw_multiply(b, raw_make(q2, 0.0f)));
    const float q3 = r.hi / b.hi;
    return raw_add(q, raw_make(q3, 0.0f));
}

static inline ScaledDD sdd_invalid(uint status) {
    return ScaledDD{as_type<float>(0x7fc00000u), 0.0f, 0, status};
}

static inline ScaledDD sdd_zero() {
    return ScaledDD{0.0f, 0.0f, 0, sdd_status_ok};
}

static inline bool sdd_valid(ScaledDD value) {
    return value.status == sdd_status_ok
        && isfinite(value.hi)
        && isfinite(value.lo);
}

static inline ScaledDD sdd_failure(ScaledDD value) {
    return value.status == sdd_status_ok
        ? sdd_invalid(sdd_status_nonfinite)
        : value;
}

static inline ScaledDD sdd_normalize(RawDD raw, int exponent) {
    if (!isfinite(raw.hi) || !isfinite(raw.lo)) {
        return sdd_invalid(sdd_status_nonfinite);
    }

    raw = two_sum(raw.hi, raw.lo);
    if (raw.hi == 0.0f) {
        if (raw.lo == 0.0f) {
            return sdd_zero();
        }
        raw = raw_make(raw.lo, 0.0f);
    }

    // All arithmetic starts from normalized mantissas.  Even after worst-case
    // float-float cancellation, at most about 64 exact power-of-two shifts are
    // needed.  The generous fixed bound keeps malformed inputs fail-visible.
    for (uint iteration = 0; iteration < 256u; ++iteration) {
        const float magnitude = abs(raw.hi);
        if (magnitude >= 1.0f) {
            raw.hi *= 0.5f;
            raw.lo *= 0.5f;
            exponent += 1;
            continue;
        }
        if (magnitude < 0.5f) {
            raw.hi *= 2.0f;
            raw.lo *= 2.0f;
            exponent -= 1;
            continue;
        }
        raw = quick_two_sum(raw.hi, raw.lo);
        return ScaledDD{raw.hi, raw.lo, exponent, sdd_status_ok};
    }
    return sdd_invalid(sdd_status_nonfinite);
}

static inline ScaledDD sdd_negate(ScaledDD a) {
    if (!sdd_valid(a)) {
        return sdd_failure(a);
    }
    return ScaledDD{-a.hi, -a.lo, a.exponent, a.status};
}

static inline RawDD sdd_scaled_mantissa(ScaledDD value, int shift) {
    // Callers never request shifts beyond the retained float-float precision,
    // so both words remain normal-or-zero FP32 values.
    float hi = value.hi;
    float lo = value.lo;
    for (int index = 0; index < shift; ++index) {
        hi *= 0.5f;
        lo *= 0.5f;
    }
    return raw_make(hi, lo);
}

static inline ScaledDD sdd_add(ScaledDD a, ScaledDD b) {
    if (!sdd_valid(a)) {
        return sdd_failure(a);
    }
    if (!sdd_valid(b)) {
        return sdd_failure(b);
    }
    if (a.hi == 0.0f && a.lo == 0.0f) {
        return b;
    }
    if (b.hi == 0.0f && b.lo == 0.0f) {
        return a;
    }
    if (a.exponent < b.exponent) {
        const ScaledDD temporary = a;
        a = b;
        b = temporary;
    }
    const int shift = a.exponent - b.exponent;
    // A normalized float-float expansion carries about 48 bits.  Values more
    // than 64 binary orders below the leading word cannot affect that result;
    // ignoring them also avoids creating a subnormal alignment lane.
    if (shift > 64) {
        return a;
    }
    return sdd_normalize(
        raw_add(raw_make(a.hi, a.lo), sdd_scaled_mantissa(b, shift)),
        a.exponent);
}

static inline ScaledDD sdd_subtract(ScaledDD a, ScaledDD b) {
    return sdd_add(a, sdd_negate(b));
}

static inline ScaledDD sdd_multiply(ScaledDD a, ScaledDD b) {
    if (!sdd_valid(a)) {
        return sdd_failure(a);
    }
    if (!sdd_valid(b)) {
        return sdd_failure(b);
    }
    if ((a.hi == 0.0f && a.lo == 0.0f)
        || (b.hi == 0.0f && b.lo == 0.0f)) {
        return sdd_zero();
    }
    return sdd_normalize(
        raw_multiply(raw_make(a.hi, a.lo), raw_make(b.hi, b.lo)),
        a.exponent + b.exponent);
}

static inline ScaledDD sdd_divide(ScaledDD a, ScaledDD b) {
    if (!sdd_valid(a)) {
        return sdd_failure(a);
    }
    if (!sdd_valid(b)) {
        return sdd_failure(b);
    }
    if (b.hi == 0.0f && b.lo == 0.0f) {
        return sdd_invalid(sdd_status_divide_by_zero);
    }
    if (a.hi == 0.0f && a.lo == 0.0f) {
        return sdd_zero();
    }
    return sdd_normalize(
        raw_divide(raw_make(a.hi, a.lo), raw_make(b.hi, b.lo)),
        a.exponent - b.exponent);
}

static inline ScaledDD sdd_half() {
    return ScaledDD{0.5f, 0.0f, 0, sdd_status_ok};
}

static inline ScaledDD sdd_square_root(ScaledDD a) {
    if (!sdd_valid(a)) {
        return sdd_failure(a);
    }
    if (a.hi == 0.0f && a.lo == 0.0f) {
        return sdd_zero();
    }
    if (a.hi < 0.0f) {
        return sdd_invalid(sdd_status_negative_sqrt);
    }

    RawDD mantissa = raw_make(a.hi, a.lo);
    int exponent = a.exponent;
    if ((exponent & 1) != 0) {
        mantissa.hi *= 2.0f;
        mantissa.lo *= 2.0f;
        exponent -= 1;
    }
    ScaledDD x = sdd_normalize(
        raw_make(metal::precise::sqrt(mantissa.hi), 0.0f),
        exponent / 2);
    // Two Newton refinements in the scaled expansion.
    x = sdd_multiply(sdd_add(x, sdd_divide(a, x)), sdd_half());
    x = sdd_multiply(sdd_add(x, sdd_divide(a, x)), sdd_half());
    return x;
}

enum ScaledArithmeticOperation : uint {
    scaled_operation_add = 0,
    scaled_operation_subtract = 1,
    scaled_operation_multiply = 2,
    scaled_operation_divide = 3,
    scaled_operation_square_root = 4,
};

kernel void scaled_dd_arithmetic(
    const device ScaledDD *input_a [[buffer(0)]],
    const device ScaledDD *input_b [[buffer(1)]],
    device ScaledDD *output [[buffer(2)]],
    constant uint &operation [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    const ScaledDD a = input_a[gid];
    const ScaledDD b = input_b[gid];
    switch (operation) {
        case scaled_operation_add:
            output[gid] = sdd_add(a, b);
            break;
        case scaled_operation_subtract:
            output[gid] = sdd_subtract(a, b);
            break;
        case scaled_operation_multiply:
            output[gid] = sdd_multiply(a, b);
            break;
        case scaled_operation_divide:
            output[gid] = sdd_divide(a, b);
            break;
        case scaled_operation_square_root:
            output[gid] = sdd_square_root(a);
            break;
        default:
            output[gid] = sdd_invalid(sdd_status_nonfinite);
            break;
    }
}

static inline ScaledDD sdd_dot4(
    const device ScaledDD *matrix,
    const device ScaledDD *vector,
    uint row) {
    ScaledDD sum = sdd_zero();
    for (uint column = 0; column < 4u; ++column) {
        sum = sdd_add(
            sum,
            sdd_multiply(matrix[row * 4u + column], vector[column]));
    }
    return sum;
}

// Hamiltonian RHS from a precomputed, host-authenticated MetricSample.  Input
// layout per record is inverse[16], inverseDerivatives[4][16], covector[4].
// Output is (dx^0..dx^3, dp_0..dp_3).  This deliberately does not claim metric
// construction, DOPRI acceptance, event topology, or renderer qualification.
kernel void scaled_dd_hamiltonian_rhs(
    const device ScaledDD *inverse_records [[buffer(0)]],
    const device ScaledDD *derivative_records [[buffer(1)]],
    const device ScaledDD *covector_records [[buffer(2)]],
    device ScaledDD *rhs_records [[buffer(3)]],
    device uint *record_status [[buffer(4)]],
    constant uint &record_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= record_count) {
        return;
    }
    const device ScaledDD *inverse = inverse_records + gid * 16u;
    const device ScaledDD *derivatives = derivative_records + gid * 64u;
    const device ScaledDD *covector = covector_records + gid * 4u;
    device ScaledDD *rhs = rhs_records + gid * 8u;

    uint status = sdd_status_ok;
    for (uint row = 0; row < 4u; ++row) {
        rhs[row] = sdd_dot4(inverse, covector, row);
        status = max(status, rhs[row].status);
    }
    for (uint coordinate = 0; coordinate < 4u; ++coordinate) {
        const device ScaledDD *derivative = derivatives + coordinate * 16u;
        ScaledDD quadratic = sdd_zero();
        for (uint row = 0; row < 4u; ++row) {
            const ScaledDD row_product = sdd_dot4(derivative, covector, row);
            quadratic = sdd_add(
                quadratic,
                sdd_multiply(covector[row], row_product));
        }
        rhs[4u + coordinate] = sdd_negate(sdd_multiply(quadratic, sdd_half()));
        status = max(status, rhs[4u + coordinate].status);
    }
    record_status[gid] = status;
}

// Batched DOPRI5(4) result/error algebra from already-computed stage vectors.
// Stages are record-major [7][8].  Coefficients are supplied as scaled DD so
// their binary64 values are not truncated to one FP32 literal.
kernel void scaled_dd_dopri_combine(
    const device ScaledDD *state_records [[buffer(0)]],
    const device ScaledDD *stage_records [[buffer(1)]],
    const device ScaledDD *step_records [[buffer(2)]],
    const device ScaledDD *coefficients [[buffer(3)]],
    device ScaledDD *result_records [[buffer(4)]],
    device uint *record_status [[buffer(5)]],
    constant uint &record_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= record_count) {
        return;
    }
    const device ScaledDD *state = state_records + gid * 8u;
    const device ScaledDD *stages = stage_records + gid * 56u;
    device ScaledDD *result = result_records + gid * 16u;
    const ScaledDD step = step_records[gid];
    constexpr uint fifth_stage[5] = {0u, 2u, 3u, 4u, 5u};
    constexpr uint fourth_stage[6] = {0u, 2u, 3u, 4u, 5u, 6u};

    uint status = sdd_status_ok;
    for (uint component = 0; component < 8u; ++component) {
        ScaledDD fifth_sum = sdd_zero();
        for (uint term = 0; term < 5u; ++term) {
            fifth_sum = sdd_add(
                fifth_sum,
                sdd_multiply(
                    coefficients[term],
                    stages[fifth_stage[term] * 8u + component]));
        }
        ScaledDD fourth_sum = sdd_zero();
        for (uint term = 0; term < 6u; ++term) {
            fourth_sum = sdd_add(
                fourth_sum,
                sdd_multiply(
                    coefficients[5u + term],
                    stages[fourth_stage[term] * 8u + component]));
        }
        const ScaledDD fifth = sdd_add(
            state[component],
            sdd_multiply(step, fifth_sum));
        const ScaledDD fourth = sdd_add(
            state[component],
            sdd_multiply(step, fourth_sum));
        result[component] = fifth;
        result[8u + component] = sdd_subtract(fifth, fourth);
        status = max(status, fifth.status);
        status = max(status, result[8u + component].status);
    }
    record_status[gid] = status;
}
