#include <metal_stdlib>

using namespace metal;

// A non-overlapping float-float expansion.  This milestone deliberately uses
// only operations whose error terms are reconstructed explicitly; it does not
// enable Metal's unsafe/fast floating-point mode.
struct DD {
    float hi;
    float lo;
};

static inline DD dd_make(float hi, float lo) {
    return DD{hi, lo};
}
static inline DD two_sum(float a, float b) {
    const float s = a + b;
    const float bb = s - a;
    const float e = (a - (s - bb)) + (b - bb);
    return dd_make(s, e);
}

// The caller guarantees |a| >= |b|.
static inline DD quick_two_sum(float a, float b) {
    const float s = a + b;
    const float e = b - (s - a);
    return dd_make(s, e);
}

static inline DD two_product(float a, float b) {
    const float p = a * b;
    // An explicit fma is required here: it recovers the product rounding error
    // without relying on the compiler to contract surrounding expressions.
    const float e = fma(a, b, -p);
    return dd_make(p, e);
}

static inline DD dd_add(DD a, DD b) {
    DD s = two_sum(a.hi, b.hi);
    DD t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return quick_two_sum(s.hi, s.lo);
}

static inline DD dd_negate(DD a) {
    return dd_make(-a.hi, -a.lo);
}

static inline DD dd_subtract(DD a, DD b) {
    return dd_add(a, dd_negate(b));
}

static inline DD dd_multiply(DD a, DD b) {
    DD p = two_product(a.hi, b.hi);
    p.lo += a.hi * b.lo;
    p.lo += a.lo * b.hi;
    p = quick_two_sum(p.hi, p.lo);
    p.lo += a.lo * b.lo;
    return quick_two_sum(p.hi, p.lo);
}

static inline DD dd_multiply_float(DD a, float b) {
    return dd_multiply(a, dd_make(b, 0.0f));
}

static inline DD dd_divide(DD a, DD b) {
    // Three quotient digits, each carrying approximately one float mantissa.
    // The final digit is needed to make the residual robust near cancellation.
    const float q1 = a.hi / b.hi;
    DD q = dd_make(q1, 0.0f);
    DD r = dd_subtract(a, dd_multiply(b, q));

    const float q2 = r.hi / b.hi;
    q = dd_add(q, dd_make(q2, 0.0f));
    r = dd_subtract(r, dd_multiply(b, dd_make(q2, 0.0f)));

    const float q3 = r.hi / b.hi;
    return dd_add(q, dd_make(q3, 0.0f));
}

static inline DD dd_square_root(DD a) {
    if (a.hi == 0.0f && a.lo == 0.0f) {
        return dd_make(0.0f, 0.0f);
    }
    if (a.hi < 0.0f) {
        return dd_make(as_type<float>(0x7fc00000u), 0.0f);
    }

    DD x = dd_make(metal::precise::sqrt(a.hi), 0.0f);
    // Two Newton refinements in float-float arithmetic.
    x = dd_multiply_float(dd_add(x, dd_divide(a, x)), 0.5f);
    x = dd_multiply_float(dd_add(x, dd_divide(a, x)), 0.5f);
    return x;
}

enum ArithmeticOperation : uint {
    operation_add = 0,
    operation_subtract = 1,
    operation_multiply = 2,
    operation_divide = 3,
    operation_square_root = 4,
};

kernel void dd_arithmetic(
    const device float2 *input_a [[buffer(0)]],
    const device float2 *input_b [[buffer(1)]],
    device float2 *output [[buffer(2)]],
    constant uint &operation [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }

    const float2 av = input_a[gid];
    const float2 bv = input_b[gid];
    const DD a = dd_make(av.x, av.y);
    const DD b = dd_make(bv.x, bv.y);
    DD result;

    switch (operation) {
        case operation_add:
            result = dd_add(a, b);
            break;
        case operation_subtract:
            result = dd_subtract(a, b);
            break;
        case operation_multiply:
            result = dd_multiply(a, b);
            break;
        case operation_divide:
            result = dd_divide(a, b);
            break;
        case operation_square_root:
            result = dd_square_root(a);
            break;
        default:
            result = dd_make(as_type<float>(0x7fc00000u), 0.0f);
            break;
    }
    output[gid] = float2(result.hi, result.lo);
}
