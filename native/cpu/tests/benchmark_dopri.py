#!/usr/bin/env python3
"""Benchmark the exact audited Kerr DOPRI boundary, not whole-ray tracing."""

from __future__ import annotations

import ctypes
import json
import math
import random
import struct
import time

from native.cpu.tests.binding import BH_CPU_OK, double_array, load_library
from offline.geodesic import _MetricAudit, _derivative, _dormand_prince_step
from offline.kerr import KerrKerrSchildMetric


def bits(value: float) -> bytes:
    return struct.pack(">d", value)


def python_step(metric, state, step):
    audit = _MetricAudit(metric)
    return _dormand_prince_step(
        lambda values: _derivative(audit, values),
        state,
        step,
    )


def best_seconds(repeats: int, operation) -> float:
    best = math.inf
    for _repeat in range(repeats):
        start = time.perf_counter()
        operation()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> int:
    library = load_library()
    metric = KerrKerrSchildMetric(1.0, 0.7, 1.0e-9)
    rng = random.Random(0x444F50524942454E)
    unique_states = []
    for _case in range(64):
        radius = rng.uniform(3.0, 50.0)
        cosine = rng.uniform(-0.98, 0.98)
        sine = math.sqrt(1.0 - cosine * cosine)
        azimuth = rng.uniform(-math.pi, math.pi)
        unique_states.append(
            (
                rng.uniform(-10.0, 10.0),
                radius * sine * math.cos(azimuth),
                radius * sine * math.sin(azimuth),
                radius * cosine,
                *(rng.uniform(-1.0, 1.0) for _axis in range(4)),
            )
        )
    states = tuple(unique_states[index % 64] for index in range(512))
    step = 0.025
    native_fifth = (ctypes.c_double * 8)()
    native_error = (ctypes.c_double * 8)()

    scalar_reference = []

    def run_python_scalar() -> None:
        current = [python_step(metric, state, step) for state in states]
        scalar_reference.clear()
        scalar_reference.extend(current)

    def run_native_scalar() -> None:
        for state in states:
            status = library.bh_cpu_kerr_dopri54_step(
                double_array(state),
                step,
                1.0,
                0.7,
                1.0e-9,
                native_fifth,
                native_error,
            )
            if status != BH_CPU_OK:
                raise RuntimeError(f"native scalar DOPRI failed: status={status}")

    python_scalar_seconds = best_seconds(3, run_python_scalar)
    native_scalar_seconds = best_seconds(7, run_native_scalar)
    last_expected_fifth, last_expected_error = scalar_reference[-1]
    if tuple(bits(value) for value in native_fifth) != tuple(
        bits(value) for value in last_expected_fifth
    ) or tuple(bits(value) for value in native_error) != tuple(
        bits(value) for value in last_expected_error
    ):
        raise RuntimeError("scalar benchmark output lost byte parity")

    probe_state = unique_states[0]
    offsets = tuple(step * (index + 1) / 48 for index in range(48))
    offsets_c = double_array(offsets)
    probe_fifth = (ctypes.c_double * (len(offsets) * 8))()
    probe_error = (ctypes.c_double * (len(offsets) * 8))()
    completed = ctypes.c_size_t()
    probe_reference = []

    def run_python_probes() -> None:
        audit = _MetricAudit(metric)
        derivative = lambda values: _derivative(audit, values)
        current = [
            _dormand_prince_step(derivative, probe_state, offset)
            for offset in offsets
        ]
        probe_reference.clear()
        probe_reference.extend(current)

    def run_native_probes() -> None:
        completed.value = 0
        status = library.bh_cpu_kerr_dopri54_probe_batch(
            double_array(probe_state),
            offsets_c,
            len(offsets),
            1.0,
            0.7,
            1.0e-9,
            probe_fifth,
            probe_error,
            ctypes.byref(completed),
        )
        if status != BH_CPU_OK or completed.value != len(offsets):
            raise RuntimeError(
                f"native probe batch failed: status={status}, "
                f"completed={completed.value}"
            )

    python_probe_seconds = best_seconds(5, run_python_probes)
    native_probe_seconds = best_seconds(11, run_native_probes)
    expected_probe_fifth = tuple(
        value for fifth, _error in probe_reference for value in fifth
    )
    expected_probe_error = tuple(
        value for _fifth, error in probe_reference for value in error
    )
    if tuple(bits(value) for value in probe_fifth) != tuple(
        bits(value) for value in expected_probe_fifth
    ) or tuple(bits(value) for value in probe_error) != tuple(
        bits(value) for value in expected_probe_error
    ):
        raise RuntimeError("probe benchmark output lost byte parity")

    result = {
        "byteParity": True,
        "excludes": [
            "adaptive accept/reject controller",
            "surface values and root topology",
            "terminal fate policy",
            "path recording and cache I/O",
        ],
        "probeBatch": {
            "nativeBestSeconds": native_probe_seconds,
            "probeCount": len(offsets),
            "pythonBestSeconds": python_probe_seconds,
            "speedup": python_probe_seconds / native_probe_seconds,
        },
        "scalar": {
            "calls": len(states),
            "nativeBestSeconds": native_scalar_seconds,
            "pythonBestSeconds": python_scalar_seconds,
            "speedup": python_scalar_seconds / native_scalar_seconds,
        },
        "scope": (
            "exact Kerr MetricSample construction, complete generic audits, "
            "Hamiltonian RHS, and one seven-stage DOPRI5(4) step"
        ),
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
