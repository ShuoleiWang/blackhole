#!/usr/bin/env python3
"""Generate deterministic scaled-DD RHS records from the frozen Python code.

The corpus is intentionally generated outside the shader.  Real records use
``KerrKerrSchildMetric.sample`` from the current checkout.  Adversarial records
exercise exponent spread and cancellation at the precomputed-MetricSample RHS
boundary.  CPU references use the same ``matrix_vector``/``bilinear`` functions
as the production Python Hamiltonian derivative, after every input has been
quantized to the exact scaled float-float wire representation sent to Metal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from offline.kerr import KerrKerrSchildMetric
from offline.spacetime import bilinear, matrix_vector


def binary32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def encode_scaled(value: float) -> list[float | int]:
    if not math.isfinite(value):
        raise ValueError("scaled-DD corpus accepts only finite binary64")
    if value == 0.0:
        return [binary32(value), 0.0, 0]
    mantissa, exponent = math.frexp(value)
    # A binary64 mantissa just below one can round to exactly +/-1 in binary32.
    # Shift it once so the wire high word remains in the canonical half-open
    # interval without losing the residual.
    if abs(binary32(mantissa)) >= 1.0:
        mantissa *= 0.5
        exponent += 1
    hi = binary32(mantissa)
    lo = binary32(mantissa - float(hi))
    if not (0.5 <= abs(hi) < 1.0):
        raise AssertionError("scaled-DD high word is not normalized")
    if lo != 0.0 and abs(lo) < 2.0**-126:
        raise AssertionError("scaled-DD low word unexpectedly became subnormal")
    return [hi, lo, exponent]


def decode_scaled(encoded: list[float | int]) -> float:
    hi, lo, exponent = encoded
    return math.ldexp(float(hi) + float(lo), int(exponent))


def quantize(value: float) -> tuple[list[float | int], float]:
    encoded = encode_scaled(value)
    return encoded, decode_scaled(encoded)


def reference_rhs(
    inverse: tuple[tuple[float, ...], ...],
    derivatives: tuple[tuple[tuple[float, ...], ...], ...],
    covector: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    coordinate = matrix_vector(inverse, covector)  # type: ignore[arg-type]
    momentum = tuple(
        -0.5 * bilinear(covector, derivative, covector)  # type: ignore[arg-type]
        for derivative in derivatives
    )
    result = (*coordinate, *momentum)
    if not all(math.isfinite(value) for value in result):
        raise ArithmeticError("non-finite CPU RHS reference")
    coordinate_scales = tuple(
        math.fsum(
            abs(inverse[row][column] * covector[column])
            for column in range(4)
        )
        for row in range(4)
    )
    momentum_scales = tuple(
        0.5
        * math.fsum(
            abs(
                covector[row]
                * derivatives[coordinate][row][column]
                * covector[column]
            )
            for row in range(4)
            for column in range(4)
        )
        for coordinate in range(4)
    )
    scales = (*coordinate_scales, *momentum_scales)
    if not all(math.isfinite(value) and value >= 0.0 for value in scales):
        raise ArithmeticError("non-finite CPU RHS conditioning scale")
    return result, scales


def encode_record(
    category: str,
    label: str,
    inverse_values: tuple[float, ...],
    derivative_values: tuple[float, ...],
    covector_values: tuple[float, ...],
) -> dict[str, object]:
    inverse_pairs = tuple(quantize(value) for value in inverse_values)
    derivative_pairs = tuple(quantize(value) for value in derivative_values)
    covector_pairs = tuple(quantize(value) for value in covector_values)
    inverse_quantized = tuple(pair[1] for pair in inverse_pairs)
    derivative_quantized = tuple(pair[1] for pair in derivative_pairs)
    covector_quantized = tuple(pair[1] for pair in covector_pairs)
    inverse = tuple(
        inverse_quantized[row * 4 : (row + 1) * 4] for row in range(4)
    )
    derivatives = tuple(
        tuple(
            derivative_quantized[
                coordinate * 16 + row * 4 : coordinate * 16 + (row + 1) * 4
            ]
            for row in range(4)
        )
        for coordinate in range(4)
    )
    reference, scales = reference_rhs(inverse, derivatives, covector_quantized)
    return {
        "category": category,
        "label": label,
        "inverse": [pair[0] for pair in inverse_pairs],
        "derivatives": [pair[0] for pair in derivative_pairs],
        "covector": [pair[0] for pair in covector_pairs],
        "reference": reference,
        "referenceScale": scales,
    }


def real_kerr_records(count: int) -> list[dict[str, object]]:
    rng = random.Random(0x4D4554414C4B455252)
    metric = KerrKerrSchildMetric(
        mass_m=1.0,
        spin_a_m=0.7,
        singularity_guard_m=1.0e-9,
    )
    records: list[dict[str, object]] = []
    for index in range(count):
        # Include near-horizon, grazing-equatorial, axial, and broad exterior
        # samples instead of drawing only an easy uniform shell.
        mode = index % 4
        if mode == 0:
            radius = rng.uniform(metric.outer_horizon_radius_m + 0.02, 2.5)
            cosine = rng.uniform(-0.999999, 0.999999)
        elif mode == 1:
            radius = rng.uniform(2.2, 25.0)
            cosine = rng.choice((-1.0, 1.0)) * rng.uniform(0.999, 0.9999999)
        elif mode == 2:
            radius = rng.uniform(2.2, 25.0)
            cosine = rng.uniform(-1.0e-7, 1.0e-7)
        else:
            radius = rng.uniform(2.2, 50.0)
            cosine = rng.uniform(-0.98, 0.98)
        sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
        azimuth = rng.uniform(-math.pi, math.pi)
        # Kerr-Schild Cartesian coordinates corresponding to an oblate r shell.
        x = math.sqrt(radius * radius + metric.spin_a_m**2) * sine * math.cos(azimuth)
        y = math.sqrt(radius * radius + metric.spin_a_m**2) * sine * math.sin(azimuth)
        z = radius * cosine
        event = (rng.uniform(-20.0, 20.0), x, y, z)
        sample = metric.sample(event)
        if index % 5 == 0:
            # Unequal exponent components exercise the stationary zero time
            # derivative together with strong spatial cancellation.
            covector = (
                rng.choice((-1.0, 1.0)) * math.ldexp(rng.uniform(0.5, 1.0), rng.randint(-8, 8)),
                rng.choice((-1.0, 1.0)) * math.ldexp(rng.uniform(0.5, 1.0), rng.randint(-8, 8)),
                rng.choice((-1.0, 1.0)) * math.ldexp(rng.uniform(0.5, 1.0), rng.randint(-8, 8)),
                rng.choice((-1.0, 1.0)) * math.ldexp(rng.uniform(0.5, 1.0), rng.randint(-8, 8)),
            )
        else:
            covector = tuple(rng.uniform(-3.0, 3.0) for _axis in range(4))
        inverse = tuple(value for row in sample.inverse for value in row)
        derivatives = tuple(
            value
            for derivative in sample.inverse_derivatives
            for row in derivative
            for value in row
        )
        records.append(
            encode_record(
                "real-kerr",
                f"real-kerr-{index}",
                inverse,
                derivatives,
                covector,
            )
        )
    return records


def random_scaled(rng: random.Random, exponent: int) -> float:
    return math.ldexp(rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 1.0), exponent)


def cancellation_matrix(
    rng: random.Random,
    covector: tuple[float, ...],
    exponent_limit: int,
) -> tuple[float, ...]:
    values = [
        random_scaled(rng, rng.randint(-exponent_limit, exponent_limit))
        for _ in range(16)
    ]
    # Craft two nearly cancelling terms in every row.  Perturbations near the
    # 48-bit wire precision expose any silent fall-back to single FP32.
    for row in range(4):
        first = 2 * row % 4
        second = (first + 1) % 4
        if covector[second] == 0.0:
            continue
        leading = random_scaled(rng, rng.randint(-exponent_limit, exponent_limit))
        perturbation = math.ldexp(rng.choice((-1.0, 1.0)), -rng.randint(38, 47))
        values[row * 4 + first] = leading
        values[row * 4 + second] = (
            -leading * covector[first] / covector[second] * (1.0 + perturbation)
        )
    return tuple(values)


def adversarial_records(count: int) -> list[dict[str, object]]:
    rng = random.Random(0x414456455253415249)
    records: list[dict[str, object]] = []
    while len(records) < count:
        index = len(records)
        exponent_limit = 104 if index % 2 == 0 else 40
        covector = tuple(
            random_scaled(rng, rng.randint(-40, 40)) for _axis in range(4)
        )
        inverse = cancellation_matrix(rng, covector, exponent_limit)
        derivatives = tuple(
            value
            for _coordinate in range(4)
            for value in cancellation_matrix(rng, covector, exponent_limit)
        )
        try:
            record = encode_record(
                "adversarial",
                f"adversarial-{index}",
                inverse,
                derivatives,
                covector,
            )
        except (ArithmeticError, OverflowError, ValueError):
            continue
        records.append(record)
    return records


def decoded_rhs_inputs(
    record: dict[str, object],
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[tuple[float, ...], ...], ...],
    tuple[float, ...],
]:
    inverse_values = tuple(
        decode_scaled(value) for value in record["inverse"]  # type: ignore[arg-type]
    )
    derivative_values = tuple(
        decode_scaled(value)
        for value in record["derivatives"]  # type: ignore[arg-type]
    )
    covector = tuple(
        decode_scaled(value) for value in record["covector"]  # type: ignore[arg-type]
    )
    inverse = tuple(
        inverse_values[row * 4 : (row + 1) * 4] for row in range(4)
    )
    derivatives = tuple(
        tuple(
            derivative_values[
                coordinate * 16 + row * 4 : coordinate * 16 + (row + 1) * 4
            ]
            for row in range(4)
        )
        for coordinate in range(4)
    )
    return inverse, derivatives, covector


def benchmark_cpu_reference(
    records: list[dict[str, object]],
    count: int,
) -> dict[str, object]:
    inputs = tuple(decoded_rhs_inputs(record) for record in records)
    best_seconds = math.inf
    checksum = 0.0
    for _repeat in range(3):
        start = time.perf_counter()
        current_checksum = 0.0
        for index in range(count):
            inverse, derivatives, covector = inputs[index % len(inputs)]
            result, _scales = reference_rhs(inverse, derivatives, covector)
            current_checksum += math.ldexp(result[index & 7], -(index & 31))
        elapsed = time.perf_counter() - start
        best_seconds = min(best_seconds, elapsed)
        checksum = current_checksum
    if not math.isfinite(checksum):
        # The adversarial corpus intentionally spans a broad exponent range;
        # the checksum is only an execution barrier, not scientific output.
        checksum = math.copysign(sys.float_info.max, checksum)
    return {
        "recordCount": count,
        "repeats": 3,
        "bestSeconds": best_seconds,
        "recordsPerSecond": count / best_seconds,
        "scope": (
            "current-checkout Python binary64 matrix_vector/bilinear over "
            "predecoded quantized records"
        ),
        "checksum": checksum,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu-benchmark-output", type=Path)
    parser.add_argument("--real-count", type=int, default=2048)
    parser.add_argument("--adversarial-count", type=int, default=2048)
    parser.add_argument("--cpu-benchmark-count", type=int, default=20_000)
    arguments = parser.parse_args()
    if (
        arguments.real_count <= 0
        or arguments.adversarial_count <= 0
        or arguments.cpu_benchmark_count <= 0
    ):
        parser.error("corpus counts must be positive")

    records = [
        *real_kerr_records(arguments.real_count),
        *adversarial_records(arguments.adversarial_count),
    ]
    payload = {
        "schema": "blackhole-metal-scaled-dd-rhs-corpus-v1",
        "encoding": "(binary32-hi + binary32-lo) * 2^signedExponent",
        "reference": (
            "current-checkout offline.spacetime matrix_vector/bilinear over "
            "the exact quantized wire inputs"
        ),
        "realKerrCount": arguments.real_count,
        "adversarialCount": arguments.adversarial_count,
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(arguments.output)
    print(f"wrote {len(records)} records to {arguments.output}")
    benchmark_path = arguments.cpu_benchmark_output
    if benchmark_path is None:
        benchmark_path = arguments.output.with_name(
            arguments.output.stem + "-cpu-benchmark.json"
        )
    benchmark_payload = {
        "schema": "blackhole-metal-milestone2-cpu-benchmark-v1",
        "corpusSHA256": hashlib.sha256(arguments.output.read_bytes()).hexdigest(),
        **benchmark_cpu_reference(records, arguments.cpu_benchmark_count),
    }
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_temporary = benchmark_path.with_suffix(
        benchmark_path.suffix + ".tmp"
    )
    benchmark_temporary.write_text(
        json.dumps(benchmark_payload, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    benchmark_temporary.replace(benchmark_path)
    print(f"wrote CPU benchmark to {benchmark_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
