#!/usr/bin/env python3
"""Microbenchmark the authenticated-sample RHS boundary, not full ray tracing."""

from __future__ import annotations

import ctypes
import json
import math
import random
import struct
import time

from native.cpu.tests.binding import (
    BH_CPU_OK,
    double_array,
    flatten_derivatives,
    flatten_matrix,
    load_library,
)
from offline.kerr import KerrKerrSchildMetric
from offline.spacetime import bilinear, matrix_vector


def bits(value: float) -> bytes:
    return struct.pack(">d", value)


def python_rhs(sample, covector):
    return (
        *matrix_vector(sample.inverse, covector),
        *(
            -0.5 * bilinear(covector, derivative, covector)
            for derivative in sample.inverse_derivatives
        ),
    )


def main() -> int:
    library = load_library()
    rng = random.Random(0x42454E4348)
    metric = KerrKerrSchildMetric(mass_m=1.0, spin_a_m=0.7)
    unique = []
    for _index in range(256):
        radius = rng.uniform(2.2, 50.0)
        cosine = rng.uniform(-0.98, 0.98)
        sine = math.sqrt(1.0 - cosine * cosine)
        azimuth = rng.uniform(-math.pi, math.pi)
        event = (
            0.0,
            radius * sine * math.cos(azimuth),
            radius * sine * math.sin(azimuth),
            radius * cosine,
        )
        unique.append(
            (
                metric.sample(event),
                tuple(rng.uniform(-3.0, 3.0) for _axis in range(4)),
            )
        )
    record_count = 20_000
    records = tuple(unique[index % len(unique)] for index in range(record_count))

    inverse_records = double_array(
        value
        for sample, _covector in records
        for value in flatten_matrix(sample.inverse)
    )
    derivative_records = double_array(
        value
        for sample, _covector in records
        for value in flatten_derivatives(sample.inverse_derivatives)
    )
    covector_records = double_array(
        value for _sample, covector in records for value in covector
    )
    output = (ctypes.c_double * (record_count * 8))()
    completed = ctypes.c_size_t()

    expected = ()
    best_python_seconds = math.inf
    for _repeat in range(3):
        start = time.perf_counter()
        current_expected = tuple(
            value
            for sample, covector in records
            for value in python_rhs(sample, covector)
        )
        elapsed = time.perf_counter() - start
        best_python_seconds = min(best_python_seconds, elapsed)
        expected = current_expected

    best_native_seconds = math.inf
    for _repeat in range(7):
        completed.value = 0
        start = time.perf_counter()
        status = library.bh_cpu_hamiltonian_rhs_batch(
            inverse_records,
            derivative_records,
            covector_records,
            record_count,
            output,
            ctypes.byref(completed),
        )
        elapsed = time.perf_counter() - start
        if status != BH_CPU_OK or completed.value != record_count:
            raise RuntimeError(
                f"native batch failed: status={status}, completed={completed.value}"
            )
        best_native_seconds = min(best_native_seconds, elapsed)

    if tuple(bits(value) for value in output) != tuple(
        bits(value) for value in expected
    ):
        raise RuntimeError("benchmark output lost byte parity")

    result = {
        "scope": "precomputed-MetricSample Hamiltonian RHS only",
        "records": record_count,
        "pythonBestSeconds": best_python_seconds,
        "nativeBestSeconds": best_native_seconds,
        "speedup": best_python_seconds / best_native_seconds,
        "nativeRecordsPerSecond": record_count / best_native_seconds,
        "byteParity": True,
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
