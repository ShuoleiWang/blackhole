#include <metal_stdlib>

using namespace metal;

// Milestone 3 deliberately keeps the milestone-2 wire inputs unchanged.  Each
// input therefore represents (hi + lo) * 2^exponent.  Products, reductions,
// and DOPRI result/error combinations are promoted to a three-word expansion:
//
//     (hi + middle + lo) * 2^exponent.
//
// The shared exponent keeps all retained mantissa words normal FP32 values.
// The expansion helpers use only error-free two_sum/two_product transforms and
// explicit FMA under Metal safe/precise math; no fast-math reassociation is
// permitted by the build contract.
struct ScaledDD {
    float hi;
    float lo;
    int exponent;
    uint status;
};

struct ScaledTD {
    float hi;
    float middle;
    float lo;
    int exponent;
    uint status;
};

constant uint std_status_ok = 0u;
constant uint std_status_nonfinite = 1u;
constant uint std_status_invalid_input = 4u;

struct Pair {
    float hi;
    float lo;
};

struct RawTD {
    float hi;
    float middle;
    float lo;
};

static inline Pair pair_make(float hi, float lo) {
    return Pair{hi, lo};
}

static inline RawTD raw_td_make(float hi, float middle, float lo) {
    return RawTD{hi, middle, lo};
}

static inline Pair two_sum(float a, float b) {
    const float sum = a + b;
    const float virtual_b = sum - a;
    const float error = (a - (sum - virtual_b)) + (b - virtual_b);
    return pair_make(sum, error);
}

static inline Pair two_product(float a, float b) {
    const float product = a * b;
    const float error = fma(a, b, -product);
    return pair_make(product, error);
}

// Grow a non-overlapping expansion stored from least to most significant.
// The largest call site supplies 18 exact product terms, so 32 slots leave a
// fail-visible implementation margin without dynamic allocation.
static inline uint expansion_grow(
    thread float *expansion,
    uint length,
    float term) {
    thread float next[32];
    uint next_length = 0u;
    float accumulator = term;
    for (uint index = 0u; index < length; ++index) {
        const Pair sum = two_sum(accumulator, expansion[index]);
        if (sum.lo != 0.0f) {
            next[next_length++] = sum.lo;
        }
        accumulator = sum.hi;
    }
    if (accumulator != 0.0f || next_length == 0u) {
        next[next_length++] = accumulator;
    }
    for (uint index = 0u; index < next_length; ++index) {
        expansion[index] = next[index];
    }
    return next_length;
}

static inline RawTD expansion_to_raw_td(
    thread const float *expansion,
    uint length) {
    // Do not merely take the three largest expansion components: the omitted
    // tail can carry the guard/round information that decides the third word.
    // Fold every component, largest first, through a three-level error-free
    // accumulator and discard only the residual below the third word.
    float hi = 0.0f;
    float middle = 0.0f;
    float lo = 0.0f;
    for (int index = int(length) - 1; index >= 0; --index) {
        const Pair high_sum = two_sum(hi, expansion[index]);
        hi = high_sum.hi;
        const Pair middle_sum = two_sum(middle, high_sum.lo);
        middle = middle_sum.hi;
        const Pair low_sum = two_sum(lo, middle_sum.lo);
        lo = low_sum.hi;
    }
    return raw_td_make(hi, middle, lo);
}

static inline RawTD raw_td_renormalize(RawTD value) {
    thread float expansion[8];
    uint length = 0u;
    length = expansion_grow(expansion, length, value.lo);
    length = expansion_grow(expansion, length, value.middle);
    length = expansion_grow(expansion, length, value.hi);
    return expansion_to_raw_td(expansion, length);
}

static inline RawTD raw_td_add(RawTD a, RawTD b) {
    thread float expansion[16];
    uint length = 0u;
    length = expansion_grow(expansion, length, a.lo);
    length = expansion_grow(expansion, length, a.middle);
    length = expansion_grow(expansion, length, a.hi);
    length = expansion_grow(expansion, length, b.lo);
    length = expansion_grow(expansion, length, b.middle);
    length = expansion_grow(expansion, length, b.hi);
    return expansion_to_raw_td(expansion, length);
}

static inline RawTD raw_td_multiply(RawTD a, RawTD b) {
    const float av[3] = {a.hi, a.middle, a.lo};
    const float bv[3] = {b.hi, b.middle, b.lo};
    thread float expansion[32];
    uint length = 0u;
    for (uint row = 0u; row < 3u; ++row) {
        for (uint column = 0u; column < 3u; ++column) {
            const Pair product = two_product(av[row], bv[column]);
            length = expansion_grow(expansion, length, product.lo);
            length = expansion_grow(expansion, length, product.hi);
        }
    }
    return expansion_to_raw_td(expansion, length);
}

// The scientific reference executes binary64 operations.  A raw three-float
// expansion carries about 72 bits, so leaving every intermediate unrounded can
// disagree with the reference solely because it computes a different (more
// exact) associativity.  Round each public arithmetic result to the normalized
// binary64 grid (2^-53 in this mantissa convention), while retaining the third
// float word as the guard/round carrier.  This is precision promotion over the
// 48-bit milestone-2 DD, not a tolerance relaxation.
static inline RawTD raw_td_round_binary64(RawTD value) {
    constexpr float quantum = 0x1.0p-53f;
    const float units = value.lo / quantum;
    const float rounded_lo = rint(units) * quantum;
    return raw_td_renormalize(
        raw_td_make(value.hi, value.middle, rounded_lo));
}

static inline ScaledTD std_invalid(uint status) {
    return ScaledTD{
        as_type<float>(0x7fc00000u), 0.0f, 0.0f, 0, status
    };
}

static inline ScaledTD std_zero() {
    return ScaledTD{0.0f, 0.0f, 0.0f, 0, std_status_ok};
}

static inline bool sdd_valid(ScaledDD value) {
    return value.status == std_status_ok
        && isfinite(value.hi)
        && isfinite(value.lo)
        && ((value.hi == 0.0f && value.lo == 0.0f)
            || (abs(value.hi) >= 0.5f && abs(value.hi) < 1.0f));
}

static inline bool std_valid(ScaledTD value) {
    return value.status == std_status_ok
        && isfinite(value.hi)
        && isfinite(value.middle)
        && isfinite(value.lo);
}

static inline ScaledTD std_failure(ScaledTD value) {
    return value.status == std_status_ok
        ? std_invalid(std_status_nonfinite)
        : value;
}

static inline ScaledTD std_normalize(RawTD raw, int exponent) {
    if (!isfinite(raw.hi) || !isfinite(raw.middle) || !isfinite(raw.lo)) {
        return std_invalid(std_status_nonfinite);
    }
    raw = raw_td_renormalize(raw);
    if (raw.hi == 0.0f) {
        if (raw.middle == 0.0f && raw.lo == 0.0f) {
            return std_zero();
        }
        raw = raw_td_renormalize(raw_td_make(raw.middle, raw.lo, 0.0f));
    }
    for (uint iteration = 0u; iteration < 256u; ++iteration) {
        const float magnitude = abs(raw.hi);
        if (magnitude >= 1.0f) {
            raw.hi *= 0.5f;
            raw.middle *= 0.5f;
            raw.lo *= 0.5f;
            exponent += 1;
            continue;
        }
        if (magnitude < 0.5f) {
            raw.hi *= 2.0f;
            raw.middle *= 2.0f;
            raw.lo *= 2.0f;
            exponent -= 1;
            continue;
        }
        raw = raw_td_round_binary64(raw);
        if (abs(raw.hi) >= 1.0f) {
            raw.hi *= 0.5f;
            raw.middle *= 0.5f;
            raw.lo *= 0.5f;
            exponent += 1;
        }
        return ScaledTD{
            raw.hi, raw.middle, raw.lo, exponent, std_status_ok
        };
    }
    return std_invalid(std_status_nonfinite);
}

static inline ScaledTD std_promote(ScaledDD value) {
    if (!sdd_valid(value)) {
        return std_invalid(
            value.status == std_status_ok ? std_status_invalid_input : value.status
        );
    }
    if (value.hi == 0.0f && value.lo == 0.0f) {
        return std_zero();
    }
    return std_normalize(raw_td_make(value.hi, value.lo, 0.0f), value.exponent);
}

static inline ScaledTD std_negate(ScaledTD value) {
    if (!std_valid(value)) {
        return std_failure(value);
    }
    return ScaledTD{
        -value.hi, -value.middle, -value.lo, value.exponent, value.status
    };
}

static inline RawTD std_scaled_mantissa(ScaledTD value, int shift) {
    RawTD raw = raw_td_make(value.hi, value.middle, value.lo);
    for (int index = 0; index < shift; ++index) {
        raw.hi *= 0.5f;
        raw.middle *= 0.5f;
        raw.lo *= 0.5f;
    }
    return raw;
}

static inline ScaledTD std_add(ScaledTD a, ScaledTD b) {
    if (!std_valid(a)) {
        return std_failure(a);
    }
    if (!std_valid(b)) {
        return std_failure(b);
    }
    if (a.hi == 0.0f && a.middle == 0.0f && a.lo == 0.0f) {
        return b;
    }
    if (b.hi == 0.0f && b.middle == 0.0f && b.lo == 0.0f) {
        return a;
    }
    if (a.exponent < b.exponent) {
        const ScaledTD temporary = a;
        a = b;
        b = temporary;
    }
    const int shift = a.exponent - b.exponent;
    // Three float words retain at most about 72 significant bits.  A value
    // more than 96 orders below the leading word cannot alter the retained
    // expansion; ignoring it also avoids relying on flushed subnormal lanes.
    if (shift > 96) {
        return a;
    }
    return std_normalize(
        raw_td_add(
            raw_td_make(a.hi, a.middle, a.lo),
            std_scaled_mantissa(b, shift)),
        a.exponent);
}

static inline ScaledTD std_subtract(ScaledTD a, ScaledTD b) {
    return std_add(a, std_negate(b));
}

static inline ScaledTD std_multiply(ScaledTD a, ScaledTD b) {
    if (!std_valid(a)) {
        return std_failure(a);
    }
    if (!std_valid(b)) {
        return std_failure(b);
    }
    if ((a.hi == 0.0f && a.middle == 0.0f && a.lo == 0.0f)
        || (b.hi == 0.0f && b.middle == 0.0f && b.lo == 0.0f)) {
        return std_zero();
    }
    return std_normalize(
        raw_td_multiply(
            raw_td_make(a.hi, a.middle, a.lo),
            raw_td_make(b.hi, b.middle, b.lo)),
        a.exponent + b.exponent);
}

// One binary64-grid rounding after a*b+c.  Retaining the exact product residual
// in the accumulator prevents the old two-word product rounding from becoming
// the dominant DOPRI error.  The full differential corpus, rather than an
// assumption about host compiler contraction, remains the acceptance oracle.
static inline ScaledTD std_fused_multiply_add(
    ScaledTD a,
    ScaledTD b,
    ScaledTD c) {
    if (!std_valid(a)) {
        return std_failure(a);
    }
    if (!std_valid(b)) {
        return std_failure(b);
    }
    if (!std_valid(c)) {
        return std_failure(c);
    }
    RawTD product = raw_td_multiply(
        raw_td_make(a.hi, a.middle, a.lo),
        raw_td_make(b.hi, b.middle, b.lo));
    int product_exponent = a.exponent + b.exponent;
    if (product.hi == 0.0f && product.middle == 0.0f && product.lo == 0.0f) {
        return c;
    }
    while (abs(product.hi) < 0.5f) {
        product.hi *= 2.0f;
        product.middle *= 2.0f;
        product.lo *= 2.0f;
        product_exponent -= 1;
    }
    while (abs(product.hi) >= 1.0f) {
        product.hi *= 0.5f;
        product.middle *= 0.5f;
        product.lo *= 0.5f;
        product_exponent += 1;
    }
    if (c.hi == 0.0f && c.middle == 0.0f && c.lo == 0.0f) {
        return std_normalize(product, product_exponent);
    }
    if (product_exponent >= c.exponent) {
        const int shift = product_exponent - c.exponent;
        if (shift > 96) {
            return std_normalize(product, product_exponent);
        }
        return std_normalize(
            raw_td_add(product, std_scaled_mantissa(c, shift)),
            product_exponent);
    }
    const int shift = c.exponent - product_exponent;
    if (shift > 96) {
        return c;
    }
    ScaledTD product_value = ScaledTD{
        product.hi, product.middle, product.lo,
        product_exponent, std_status_ok
    };
    return std_normalize(
        raw_td_add(
            raw_td_make(c.hi, c.middle, c.lo),
            std_scaled_mantissa(product_value, shift)),
        c.exponent);
}

static inline ScaledTD std_half() {
    return ScaledTD{0.5f, 0.0f, 0.0f, 0, std_status_ok};
}

static inline ScaledTD std_dot4(
    const device ScaledDD *matrix,
    const device ScaledDD *vector,
    uint row) {
    ScaledTD sum = std_zero();
    for (uint column = 0u; column < 4u; ++column) {
        sum = std_add(
            sum,
            std_multiply(
                std_promote(matrix[row * 4u + column]),
                std_promote(vector[column])));
    }
    return sum;
}

kernel void scaled_td_hamiltonian_rhs(
    const device ScaledDD *inverse_records [[buffer(0)]],
    const device ScaledDD *derivative_records [[buffer(1)]],
    const device ScaledDD *covector_records [[buffer(2)]],
    device ScaledTD *rhs_records [[buffer(3)]],
    device uint *record_status [[buffer(4)]],
    constant uint &record_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= record_count) {
        return;
    }
    const device ScaledDD *inverse = inverse_records + gid * 16u;
    const device ScaledDD *derivatives = derivative_records + gid * 64u;
    const device ScaledDD *covector = covector_records + gid * 4u;
    device ScaledTD *rhs = rhs_records + gid * 8u;

    uint status = std_status_ok;
    for (uint row = 0u; row < 4u; ++row) {
        rhs[row] = std_dot4(inverse, covector, row);
        status = max(status, rhs[row].status);
    }
    for (uint coordinate = 0u; coordinate < 4u; ++coordinate) {
        const device ScaledDD *derivative = derivatives + coordinate * 16u;
        ScaledTD quadratic = std_zero();
        for (uint row = 0u; row < 4u; ++row) {
            const ScaledTD row_product = std_dot4(derivative, covector, row);
            quadratic = std_add(
                quadratic,
                std_multiply(std_promote(covector[row]), row_product));
        }
        rhs[4u + coordinate] = std_negate(
            std_multiply(quadratic, std_half()));
        status = max(status, rhs[4u + coordinate].status);
    }
    record_status[gid] = status;
}

// Batched DOPRI5(4) result/error algebra.  The fifth-order state is accumulated
// in triple-word precision.  The embedded error is evaluated directly from
// (b5-b4)*stage rather than by subtracting two state-sized rounded results.
kernel void scaled_td_dopri_combine(
    const device ScaledDD *state_records [[buffer(0)]],
    const device ScaledDD *stage_records [[buffer(1)]],
    const device ScaledDD *step_records [[buffer(2)]],
    const device ScaledDD *fifth_coefficients [[buffer(3)]],
    const device ScaledDD *error_coefficients [[buffer(4)]],
    device ScaledTD *result_records [[buffer(5)]],
    device uint *record_status [[buffer(6)]],
    constant uint &record_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= record_count) {
        return;
    }
    const device ScaledDD *state = state_records + gid * 8u;
    const device ScaledDD *stages = stage_records + gid * 56u;
    device ScaledTD *result = result_records + gid * 16u;
    const ScaledTD step = std_promote(step_records[gid]);
    constexpr uint fifth_stage[5] = {0u, 2u, 3u, 4u, 5u};

    uint status = std_status_ok;
    for (uint component = 0u; component < 8u; ++component) {
        ScaledTD fifth_sum = std_zero();
        for (uint term = 0u; term < 5u; ++term) {
            fifth_sum = std_fused_multiply_add(
                std_promote(fifth_coefficients[term]),
                std_promote(stages[fifth_stage[term] * 8u + component]),
                fifth_sum);
        }
        ScaledTD error_sum = std_zero();
        // Error coefficients are laid out for all seven stages, including the
        // two explicit zeros.  Keeping this fixed topology makes host/shader
        // coefficient ordering fail-visible.
        for (uint stage = 0u; stage < 7u; ++stage) {
            error_sum = std_fused_multiply_add(
                std_promote(error_coefficients[stage]),
                std_promote(stages[stage * 8u + component]),
                error_sum);
        }
        const ScaledTD fifth = std_fused_multiply_add(
            step, fifth_sum, std_promote(state[component]));
        const ScaledTD error = std_multiply(step, error_sum);
        result[component] = fifth;
        result[8u + component] = error;
        status = max(status, fifth.status);
        status = max(status, error.status);
    }
    record_status[gid] = status;
}

// First post-gate batching boundary: one DOPRI combine plus a signed planar
// surface probe value z - surface_z.  It does not perform adaptive acceptance
// or root localization, and the report must not claim those policies.
kernel void scaled_td_one_resolution_step_probe(
    const device ScaledDD *state_records [[buffer(0)]],
    const device ScaledDD *stage_records [[buffer(1)]],
    const device ScaledDD *step_records [[buffer(2)]],
    const device ScaledDD *fifth_coefficients [[buffer(3)]],
    const device ScaledDD *surface_z_records [[buffer(4)]],
    device ScaledTD *fifth_state_records [[buffer(5)]],
    device ScaledTD *probe_records [[buffer(6)]],
    device uint *record_status [[buffer(7)]],
    constant uint &record_count [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= record_count) {
        return;
    }
    const device ScaledDD *state = state_records + gid * 8u;
    const device ScaledDD *stages = stage_records + gid * 56u;
    device ScaledTD *output = fifth_state_records + gid * 8u;
    const ScaledTD step = std_promote(step_records[gid]);
    constexpr uint fifth_stage[5] = {0u, 2u, 3u, 4u, 5u};
    uint status = std_status_ok;
    for (uint component = 0u; component < 8u; ++component) {
        ScaledTD sum = std_zero();
        for (uint term = 0u; term < 5u; ++term) {
            sum = std_fused_multiply_add(
                std_promote(fifth_coefficients[term]),
                std_promote(stages[fifth_stage[term] * 8u + component]),
                sum);
        }
        output[component] = std_fused_multiply_add(
            step, sum, std_promote(state[component]));
        status = max(status, output[component].status);
    }
    // State layout is (t,x,y,z,p_t,p_x,p_y,p_z).
    probe_records[gid] = std_subtract(
        output[3u], std_promote(surface_z_records[gid]));
    status = max(status, probe_records[gid].status);
    record_status[gid] = status;
}
