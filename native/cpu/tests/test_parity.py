"""Differential parity tests against the frozen Python numeric backend."""

from __future__ import annotations

import ctypes
import math
import random
import struct
import unittest

from native.cpu.tests.binding import (
    BH_CPU_GUARDED_SINGULARITY,
    BH_CPU_INCONSISTENT_METRIC,
    BH_CPU_INVALID_ARGUMENT,
    BH_CPU_NONFINITE_INPUT,
    BH_CPU_NUMERIC_OVERFLOW,
    BH_CPU_OK,
    RuntimeContract,
    double_array,
    flatten_derivatives,
    flatten_matrix,
    load_library,
)
from offline.geodesic import (
    _MetricAudit,
    _derivative,
    _dormand_prince_step,
    _normalized_null_residual_from_sample,
)
from offline.kerr import KerrKerrSchildMetric
from offline.spacetime import (
    METRIC_CONSISTENCY_ABSOLUTE_TOLERANCE,
    METRIC_CONSISTENCY_TOLERANCE,
    MINKOWSKI_COVARIANT,
    ZERO_DERIVATIVES,
    bilinear,
    matrix_vector,
)


def bits(value: float) -> bytes:
    return struct.pack(">d", value)


def python_rhs(sample, covector: tuple[float, float, float, float]):
    coordinate = matrix_vector(sample.inverse, covector)
    momentum = tuple(
        -0.5 * bilinear(covector, sample.inverse_derivatives[index], covector)
        for index in range(4)
    )
    return (*coordinate, *momentum)


def python_kerr_dopri(metric, state: tuple[float, ...], step: float):
    audit = _MetricAudit(metric)
    return _dormand_prince_step(
        lambda values: _derivative(audit, values),
        state,
        step,
    )


def random_finite(rng: random.Random, exponent_limit: int = 200) -> float:
    mantissa = rng.uniform(-1.0, 1.0)
    exponent = rng.randint(-exponent_limit, exponent_limit)
    return math.ldexp(mantissa, exponent)


def sample_cases(count: int, seed: int = 0x4B455252):
    rng = random.Random(seed)
    metric = KerrKerrSchildMetric(
        mass_m=1.0,
        spin_a_m=0.7,
        singularity_guard_m=1.0e-9,
    )
    for _ in range(count):
        radius = rng.uniform(2.2, 50.0)
        cosine = rng.uniform(-0.98, 0.98)
        sine = math.sqrt(1.0 - cosine * cosine)
        azimuth = rng.uniform(-math.pi, math.pi)
        event = (
            rng.uniform(-20.0, 20.0),
            radius * sine * math.cos(azimuth),
            radius * sine * math.sin(azimuth),
            radius * cosine,
        )
        sample = metric.sample(event)
        covector = tuple(rng.uniform(-3.0, 3.0) for _axis in range(4))
        yield sample, covector


class StrictCpuParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.library = load_library()

    def test_runtime_contract_is_binary64_round_to_nearest(self) -> None:
        contract = RuntimeContract()
        self.assertEqual(
            self.library.bh_cpu_get_runtime_contract(ctypes.byref(contract)),
            BH_CPU_OK,
        )
        self.assertEqual(contract.abi_version, 3)
        self.assertEqual(contract.struct_size, ctypes.sizeof(RuntimeContract))
        self.assertEqual(contract.double_size, 8)
        self.assertEqual(contract.double_mantissa_bits, 53)
        self.assertEqual(contract.float_radix, 2)
        self.assertEqual(contract.float_evaluation_method, 0)
        self.assertEqual(contract.fast_math_enabled, 0)
        self.assertEqual(contract.iec_60559_binary64, 1)
        self.assertEqual(contract.reserved, 0)
        self.assertEqual(self.library.bh_cpu_require_strict_fp(), BH_CPU_OK)

    def test_fsum_curated_byte_parity(self) -> None:
        cases = (
            (),
            (-0.0,),
            (1.0,),
            (1.0e100, 1.0, -1.0e100),
            (2.0**53, 1.0, 1.0, -(2.0**53)),
            (1.0, 2.0**-53, 2.0**-54),
            (2.0**-1074, 2.0**-1074, -(2.0**-1074)),
            tuple([0.1] * 100),
        )
        for values in cases:
            with self.subTest(values=values[:8]):
                source = double_array(values)
                output = ctypes.c_double()
                status = self.library.bh_cpu_fsum(
                    source,
                    len(values),
                    ctypes.byref(output),
                )
                self.assertEqual(status, BH_CPU_OK)
                self.assertEqual(bits(output.value), bits(math.fsum(values)))

    def test_fsum_random_byte_parity(self) -> None:
        rng = random.Random(0x4653554D)
        for _case in range(20_000):
            values = tuple(
                random_finite(rng) for _term in range(rng.randint(0, 64))
            )
            source = double_array(values)
            output = ctypes.c_double()
            status = self.library.bh_cpu_fsum(
                source,
                len(values),
                ctypes.byref(output),
            )
            self.assertEqual(status, BH_CPU_OK)
            self.assertEqual(bits(output.value), bits(math.fsum(values)))

    def test_fsum_rejects_nonfinite_input(self) -> None:
        for value in (math.inf, -math.inf, math.nan):
            source = double_array((1.0, value))
            output = ctypes.c_double()
            self.assertEqual(
                self.library.bh_cpu_fsum(source, 2, ctypes.byref(output)),
                BH_CPU_NONFINITE_INPUT,
            )

    def test_matrix_vector_and_bilinear_byte_parity(self) -> None:
        rng = random.Random(0x4D4154524958)
        for _case in range(4_000):
            matrix = tuple(
                tuple(random_finite(rng, 20) for _column in range(4))
                for _row in range(4)
            )
            left = tuple(random_finite(rng, 20) for _index in range(4))
            right = tuple(random_finite(rng, 20) for _index in range(4))
            matrix_c = double_array(flatten_matrix(matrix))
            left_c = double_array(left)
            right_c = double_array(right)
            output_vector = (ctypes.c_double * 4)()
            output_scalar = ctypes.c_double()
            self.assertEqual(
                self.library.bh_cpu_matrix4_vector4(
                    matrix_c, right_c, output_vector
                ),
                BH_CPU_OK,
            )
            self.assertEqual(
                tuple(bits(value) for value in output_vector),
                tuple(bits(value) for value in matrix_vector(matrix, right)),
            )
            self.assertEqual(
                self.library.bh_cpu_bilinear4(
                    left_c, matrix_c, right_c, ctypes.byref(output_scalar)
                ),
                BH_CPU_OK,
            )
            self.assertEqual(
                bits(output_scalar.value), bits(bilinear(left, matrix, right))
            )

    def test_matrix_vector_scalar_abi_is_alias_safe(self) -> None:
        matrix = (
            (2.0, 3.0, 5.0, 7.0),
            (11.0, 13.0, 17.0, 19.0),
            (23.0, 29.0, 31.0, 37.0),
            (41.0, 43.0, 47.0, 53.0),
        )
        vector = (0.25, -0.5, 1.5, -2.0)
        matrix_c = double_array(flatten_matrix(matrix))
        vector_and_output = double_array(vector)
        self.assertEqual(
            self.library.bh_cpu_matrix4_vector4(
                matrix_c, vector_and_output, vector_and_output
            ),
            BH_CPU_OK,
        )
        self.assertEqual(
            tuple(bits(value) for value in vector_and_output),
            tuple(bits(value) for value in matrix_vector(matrix, vector)),
        )

    def test_real_kerr_metric_audit_and_rhs_byte_parity(self) -> None:
        for sample, covector in sample_cases(2_000):
            covariant_c = double_array(flatten_matrix(sample.covariant))
            inverse_c = double_array(flatten_matrix(sample.inverse))
            derivatives_c = double_array(
                flatten_derivatives(sample.inverse_derivatives)
            )
            covector_c = double_array(covector)
            output = (ctypes.c_double * 8)()
            self.assertEqual(
                self.library.bh_cpu_metric_algebra_audit(
                    covariant_c,
                    inverse_c,
                    derivatives_c,
                    METRIC_CONSISTENCY_TOLERANCE,
                    METRIC_CONSISTENCY_ABSOLUTE_TOLERANCE,
                ),
                BH_CPU_OK,
            )
            self.assertEqual(
                self.library.bh_cpu_hamiltonian_rhs(
                    inverse_c, derivatives_c, covector_c, output
                ),
                BH_CPU_OK,
            )
            self.assertEqual(
                tuple(bits(value) for value in output),
                tuple(bits(value) for value in python_rhs(sample, covector)),
            )

    def test_normalized_null_residual_byte_parity(self) -> None:
        for sample, covector in sample_cases(4_000, seed=0x4E554C4C):
            inverse_c = double_array(flatten_matrix(sample.inverse))
            covector_c = double_array(covector)
            output = ctypes.c_double()
            self.assertEqual(
                self.library.bh_cpu_normalized_null_residual(
                    inverse_c, covector_c, ctypes.byref(output)
                ),
                BH_CPU_OK,
            )
            self.assertEqual(
                bits(output.value),
                bits(_normalized_null_residual_from_sample(sample, covector)),
            )

    def test_batch_rhs_matches_scalar_and_python(self) -> None:
        cases = tuple(sample_cases(257, seed=0x4241544348))
        inverse_records = double_array(
            value
            for sample, _covector in cases
            for value in flatten_matrix(sample.inverse)
        )
        derivative_records = double_array(
            value
            for sample, _covector in cases
            for value in flatten_derivatives(sample.inverse_derivatives)
        )
        covector_records = double_array(
            value for _sample, covector in cases for value in covector
        )
        output = (ctypes.c_double * (len(cases) * 8))()
        completed = ctypes.c_size_t()
        self.assertEqual(
            self.library.bh_cpu_hamiltonian_rhs_batch(
                inverse_records,
                derivative_records,
                covector_records,
                len(cases),
                output,
                ctypes.byref(completed),
            ),
            BH_CPU_OK,
        )
        self.assertEqual(completed.value, len(cases))
        expected = tuple(
            value
            for sample, covector in cases
            for value in python_rhs(sample, covector)
        )
        self.assertEqual(
            tuple(bits(value) for value in output),
            tuple(bits(value) for value in expected),
        )

    def test_scalar_rhs_abi_is_covector_alias_safe(self) -> None:
        sample, covector = next(sample_cases(1, seed=0x414C494153))
        inverse_c = double_array(flatten_matrix(sample.inverse))
        derivatives_c = double_array(flatten_derivatives(sample.inverse_derivatives))
        storage = (ctypes.c_double * 8)(*covector, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(
            self.library.bh_cpu_hamiltonian_rhs(
                inverse_c, derivatives_c, storage, storage
            ),
            BH_CPU_OK,
        )
        self.assertEqual(
            tuple(bits(value) for value in storage),
            tuple(bits(value) for value in python_rhs(sample, covector)),
        )

    def test_complete_metric_audit_preserves_lorentzian_gate(self) -> None:
        minkowski = double_array(flatten_matrix(MINKOWSKI_COVARIANT))
        derivatives = double_array(flatten_derivatives(ZERO_DERIVATIVES))
        self.assertEqual(
            self.library.bh_cpu_metric_sample_audit(
                minkowski,
                minkowski,
                derivatives,
                METRIC_CONSISTENCY_TOLERANCE,
                METRIC_CONSISTENCY_ABSOLUTE_TOLERANCE,
            ),
            BH_CPU_OK,
        )
        euclidean = double_array(
            value
            for row in range(4)
            for value in tuple(1.0 if row == column else 0.0 for column in range(4))
        )
        self.assertEqual(
            self.library.bh_cpu_metric_algebra_audit(
                euclidean,
                euclidean,
                derivatives,
                METRIC_CONSISTENCY_TOLERANCE,
                METRIC_CONSISTENCY_ABSOLUTE_TOLERANCE,
            ),
            BH_CPU_OK,
        )
        self.assertEqual(
            self.library.bh_cpu_metric_sample_audit(
                euclidean,
                euclidean,
                derivatives,
                METRIC_CONSISTENCY_TOLERANCE,
                METRIC_CONSISTENCY_ABSOLUTE_TOLERANCE,
            ),
            BH_CPU_INCONSISTENT_METRIC,
        )

    def test_exact_kerr_metric_sample_random_byte_parity(self) -> None:
        rng = random.Random(0x4B45525253414D50)
        spins = (-1.0, -0.7, 0.0, 0.7, 1.0)
        for case in range(10_000):
            spin = spins[case % len(spins)]
            metric = KerrKerrSchildMetric(
                mass_m=1.0,
                spin_a_m=spin,
                singularity_guard_m=1.0e-9,
            )
            radius = rng.uniform(1.8, 80.0)
            cosine = rng.uniform(-0.999, 0.999)
            sine = math.sqrt(1.0 - cosine * cosine)
            azimuth = rng.uniform(-math.pi, math.pi)
            event = (
                rng.uniform(-1.0e6, 1.0e6),
                radius * sine * math.cos(azimuth),
                radius * sine * math.sin(azimuth),
                radius * cosine,
            )
            expected = metric.sample(event)
            covariant = (ctypes.c_double * 16)()
            inverse = (ctypes.c_double * 16)()
            derivatives = (ctypes.c_double * 64)()
            self.assertEqual(
                self.library.bh_cpu_kerr_metric_sample(
                    double_array(event),
                    metric.mass_m,
                    metric.spin_a_m,
                    metric.singularity_guard_m,
                    covariant,
                    inverse,
                    derivatives,
                ),
                BH_CPU_OK,
            )
            actual_payload = tuple(covariant) + tuple(inverse) + tuple(derivatives)
            expected_payload = (
                flatten_matrix(expected.covariant)
                + flatten_matrix(expected.inverse)
                + flatten_derivatives(expected.inverse_derivatives)
            )
            self.assertEqual(
                tuple(bits(value) for value in actual_payload),
                tuple(bits(value) for value in expected_payload),
            )

    def test_exact_kerr_metric_sample_branch_edges_byte_parity(self) -> None:
        cases = (
            (0.0, (0.0, 0.0, 0.0, 2.1)),
            (0.7, (1.0e200, 2.1, 0.0, 0.0)),
            (-0.7, (-1.0e200, 0.0, -2.1, 0.0)),
            (1.0, (0.0, 1.0e-2, -2.0e-2, 1.0e-1)),
            (-1.0, (0.0, -1.0e-2, 2.0e-2, -1.0e-1)),
            (1.0, (0.0, 0.0, 0.0, 1.0e-6)),
            (-1.0, (0.0, 0.0, 0.0, -1.0e-6)),
            (0.999999999999, (0.0, 1.0e5, -2.0e5, 3.0e5)),
        )
        for spin, event in cases:
            with self.subTest(spin=spin, event=event):
                metric = KerrKerrSchildMetric(1.0, spin, 1.0e-9)
                expected = metric.sample(event)
                covariant = (ctypes.c_double * 16)()
                inverse = (ctypes.c_double * 16)()
                derivatives = (ctypes.c_double * 64)()
                self.assertEqual(
                    self.library.bh_cpu_kerr_metric_sample(
                        double_array(event),
                        1.0,
                        spin,
                        1.0e-9,
                        covariant,
                        inverse,
                        derivatives,
                    ),
                    BH_CPU_OK,
                )
                actual = tuple(covariant) + tuple(inverse) + tuple(derivatives)
                reference = (
                    flatten_matrix(expected.covariant)
                    + flatten_matrix(expected.inverse)
                    + flatten_derivatives(expected.inverse_derivatives)
                )
                self.assertEqual(
                    tuple(bits(value) for value in actual),
                    tuple(bits(value) for value in reference),
                )

    def test_exact_kerr_metric_rationalized_branch_random_byte_parity(self) -> None:
        rng = random.Random(0x524154494F4E414C)
        for case in range(2_000):
            spin = 1.0 if case % 2 == 0 else -1.0
            metric = KerrKerrSchildMetric(1.0, spin, 1.0e-9)
            z_magnitude = rng.uniform(1.0e-6, 0.4)
            event = (
                rng.uniform(-20.0, 20.0),
                rng.uniform(-0.4, 0.4),
                rng.uniform(-0.4, 0.4),
                z_magnitude if case % 4 < 2 else -z_magnitude,
            )
            rho_squared = math.fsum(value * value for value in event[1:])
            self.assertLess(rho_squared - spin * spin, 0.0)
            expected = metric.sample(event)
            covariant = (ctypes.c_double * 16)()
            inverse = (ctypes.c_double * 16)()
            derivatives = (ctypes.c_double * 64)()
            self.assertEqual(
                self.library.bh_cpu_kerr_metric_sample(
                    double_array(event),
                    1.0,
                    spin,
                    1.0e-9,
                    covariant,
                    inverse,
                    derivatives,
                ),
                BH_CPU_OK,
            )
            actual = tuple(covariant) + tuple(inverse) + tuple(derivatives)
            reference = (
                flatten_matrix(expected.covariant)
                + flatten_matrix(expected.inverse)
                + flatten_derivatives(expected.inverse_derivatives)
            )
            self.assertEqual(
                tuple(bits(value) for value in actual),
                tuple(bits(value) for value in reference),
            )

    def test_kerr_metric_and_rhs_fail_closed_at_guard(self) -> None:
        sentinel = float.fromhex("0x1.23456789abcdfp+12")
        covariant = (ctypes.c_double * 16)(*[sentinel] * 16)
        inverse = (ctypes.c_double * 16)(*[sentinel] * 16)
        derivatives = (ctypes.c_double * 64)(*[sentinel] * 64)
        self.assertEqual(
            self.library.bh_cpu_kerr_metric_sample(
                double_array((0.0, 0.0, 0.0, 0.0)),
                1.0,
                0.7,
                1.0e-9,
                covariant,
                inverse,
                derivatives,
            ),
            BH_CPU_GUARDED_SINGULARITY,
        )
        self.assertTrue(all(value == sentinel for value in covariant))
        self.assertTrue(all(value == sentinel for value in inverse))
        self.assertTrue(all(value == sentinel for value in derivatives))

        output = (ctypes.c_double * 8)(*[sentinel] * 8)
        self.assertEqual(
            self.library.bh_cpu_kerr_hamiltonian_rhs(
                double_array((0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
                1.0,
                0.7,
                1.0e-9,
                output,
            ),
            BH_CPU_GUARDED_SINGULARITY,
        )
        self.assertTrue(all(value == sentinel for value in output))

    def test_direct_kerr_rhs_random_byte_parity(self) -> None:
        rng = random.Random(0x4B455252524853)
        metric = KerrKerrSchildMetric(1.0, 0.7, 1.0e-9)
        for _case in range(2_000):
            radius = rng.uniform(2.2, 50.0)
            cosine = rng.uniform(-0.98, 0.98)
            sine = math.sqrt(1.0 - cosine * cosine)
            azimuth = rng.uniform(-math.pi, math.pi)
            event = (
                rng.uniform(-20.0, 20.0),
                radius * sine * math.cos(azimuth),
                radius * sine * math.sin(azimuth),
                radius * cosine,
            )
            covector = tuple(rng.uniform(-3.0, 3.0) for _axis in range(4))
            state = (*event, *covector)
            output = (ctypes.c_double * 8)()
            self.assertEqual(
                self.library.bh_cpu_kerr_hamiltonian_rhs(
                    double_array(state),
                    1.0,
                    0.7,
                    1.0e-9,
                    output,
                ),
                BH_CPU_OK,
            )
            self.assertEqual(
                tuple(bits(value) for value in output),
                tuple(bits(value) for value in python_rhs(metric.sample(event), covector)),
            )

    def test_kerr_dopri54_random_and_edge_byte_parity(self) -> None:
        rng = random.Random(0x444F5052493534)
        spins = (-1.0, -0.7, 0.0, 0.7, 1.0)
        steps = (0.0, 2.0**-40, 1.0e-6, 1.0e-4, 1.0e-3, 0.025, -1.0e-3)
        for case in range(2_000):
            spin = spins[case % len(spins)]
            metric = KerrKerrSchildMetric(1.0, spin, 1.0e-9)
            radius = rng.uniform(3.0, 60.0)
            cosine = rng.uniform(-0.98, 0.98)
            sine = math.sqrt(1.0 - cosine * cosine)
            azimuth = rng.uniform(-math.pi, math.pi)
            state = (
                rng.uniform(-20.0, 20.0),
                radius * sine * math.cos(azimuth),
                radius * sine * math.sin(azimuth),
                radius * cosine,
                *(rng.uniform(-1.0, 1.0) for _axis in range(4)),
            )
            step = steps[case % len(steps)]
            expected_fifth, expected_error = python_kerr_dopri(metric, state, step)
            fifth = (ctypes.c_double * 8)()
            error = (ctypes.c_double * 8)()
            self.assertEqual(
                self.library.bh_cpu_kerr_dopri54_step(
                    double_array(state),
                    step,
                    1.0,
                    spin,
                    1.0e-9,
                    fifth,
                    error,
                ),
                BH_CPU_OK,
            )
            self.assertEqual(
                tuple(bits(value) for value in fifth),
                tuple(bits(value) for value in expected_fifth),
            )
            self.assertEqual(
                tuple(bits(value) for value in error),
                tuple(bits(value) for value in expected_error),
            )

    def test_kerr_dopri54_rationalized_branch_edges_byte_parity(self) -> None:
        cases = (
            (1.0, (0.0, 0.1, -0.2, 0.1, -1.0, 0.01, -0.02, 0.03), 1.0e-6),
            (-1.0, (0.0, -0.15, 0.05, -0.2, -1.0, -0.03, 0.02, -0.01), 1.0e-6),
            (1.0, (0.0, 0.0, 0.0, 1.0e-5, -1.0, 0.0, 0.0, 1.0e-8), 1.0e-10),
        )
        for spin, state, step in cases:
            with self.subTest(spin=spin, state=state, step=step):
                metric = KerrKerrSchildMetric(1.0, spin, 1.0e-9)
                expected_fifth, expected_error = python_kerr_dopri(
                    metric,
                    state,
                    step,
                )
                fifth = (ctypes.c_double * 8)()
                error = (ctypes.c_double * 8)()
                self.assertEqual(
                    self.library.bh_cpu_kerr_dopri54_step(
                        double_array(state),
                        step,
                        1.0,
                        spin,
                        1.0e-9,
                        fifth,
                        error,
                    ),
                    BH_CPU_OK,
                )
                self.assertEqual(
                    tuple(bits(value) for value in fifth),
                    tuple(bits(value) for value in expected_fifth),
                )
                self.assertEqual(
                    tuple(bits(value) for value in error),
                    tuple(bits(value) for value in expected_error),
                )

    def test_kerr_dopri54_scalar_abi_is_state_alias_safe(self) -> None:
        metric = KerrKerrSchildMetric(1.0, 0.7, 1.0e-9)
        state = (0.0, 7.0, -2.0, 3.0, -1.0, 0.2, -0.3, 0.4)
        expected_fifth, expected_error = python_kerr_dopri(metric, state, 0.025)
        state_and_fifth = double_array(state)
        error = (ctypes.c_double * 8)()
        self.assertEqual(
            self.library.bh_cpu_kerr_dopri54_step(
                state_and_fifth,
                0.025,
                1.0,
                0.7,
                1.0e-9,
                state_and_fifth,
                error,
            ),
            BH_CPU_OK,
        )
        self.assertEqual(
            tuple(bits(value) for value in state_and_fifth),
            tuple(bits(value) for value in expected_fifth),
        )
        self.assertEqual(
            tuple(bits(value) for value in error),
            tuple(bits(value) for value in expected_error),
        )

    def test_probe_batch_is_independent_same_start_byte_parity(self) -> None:
        metric = KerrKerrSchildMetric(1.0, 0.7, 1.0e-9)
        state = (0.0, 8.0, -1.5, 2.5, -1.0, 0.15, -0.25, 0.35)
        offsets = tuple(0.025 * (index + 1) / 64 for index in range(64))
        expected = tuple(python_kerr_dopri(metric, state, offset) for offset in offsets)
        fifth_records = (ctypes.c_double * (len(offsets) * 8))()
        error_records = (ctypes.c_double * (len(offsets) * 8))()
        completed = ctypes.c_size_t()
        self.assertEqual(
            self.library.bh_cpu_kerr_dopri54_probe_batch(
                double_array(state),
                double_array(offsets),
                len(offsets),
                1.0,
                0.7,
                1.0e-9,
                fifth_records,
                error_records,
                ctypes.byref(completed),
            ),
            BH_CPU_OK,
        )
        self.assertEqual(completed.value, len(offsets))
        expected_fifth = tuple(value for fifth, _error in expected for value in fifth)
        expected_error = tuple(value for _fifth, error in expected for value in error)
        self.assertEqual(
            tuple(bits(value) for value in fifth_records),
            tuple(bits(value) for value in expected_fifth),
        )
        self.assertEqual(
            tuple(bits(value) for value in error_records),
            tuple(bits(value) for value in expected_error),
        )

    def test_probe_batch_reports_exact_completed_prefix(self) -> None:
        state = (0.0, 8.0, -1.5, 2.5, -1.0, 0.15, -0.25, 0.35)
        offsets = double_array((0.001, math.nan, 0.003))
        fifth_records = (ctypes.c_double * 24)()
        error_records = (ctypes.c_double * 24)()
        completed = ctypes.c_size_t(99)
        self.assertEqual(
            self.library.bh_cpu_kerr_dopri54_probe_batch(
                double_array(state),
                offsets,
                3,
                1.0,
                0.7,
                1.0e-9,
                fifth_records,
                error_records,
                ctypes.byref(completed),
            ),
            BH_CPU_NONFINITE_INPUT,
        )
        self.assertEqual(completed.value, 1)
        self.assertEqual(
            self.library.bh_cpu_kerr_dopri54_step(
                double_array(state),
                0.001,
                1.0,
                0.7,
                1.0e-9,
                fifth_records,
                fifth_records,
            ),
            BH_CPU_INVALID_ARGUMENT,
        )

    def test_kerr_api_rejects_invalid_parameters_and_empty_batch_is_valid(self) -> None:
        event = double_array((0.0, 8.0, -1.5, 2.5))
        covariant = (ctypes.c_double * 16)()
        inverse = (ctypes.c_double * 16)()
        derivatives = (ctypes.c_double * 64)()
        for mass, spin, guard, expected_status in (
            (0.0, 0.0, 1.0e-9, BH_CPU_INVALID_ARGUMENT),
            (1.0, 1.01, 1.0e-9, BH_CPU_INVALID_ARGUMENT),
            (1.0, 0.7, 0.0, BH_CPU_INVALID_ARGUMENT),
            (math.inf, 0.7, 1.0e-9, BH_CPU_NONFINITE_INPUT),
            (float.fromhex("0x1.fffffffffffffp+1023"),
             float.fromhex("0x1.fffffffffffffp+1023"),
             1.0e-9,
             BH_CPU_NUMERIC_OVERFLOW),
        ):
            with self.subTest(mass=mass, spin=spin, guard=guard):
                self.assertEqual(
                    self.library.bh_cpu_kerr_metric_sample(
                        event,
                        mass,
                        spin,
                        guard,
                        covariant,
                        inverse,
                        derivatives,
                    ),
                    expected_status,
                )

        state = double_array((0.0, 8.0, -1.5, 2.5, -1.0, 0.15, -0.25, 0.35))
        completed = ctypes.c_size_t(99)
        self.assertEqual(
            self.library.bh_cpu_kerr_dopri54_probe_batch(
                state,
                None,
                0,
                1.0,
                0.7,
                1.0e-9,
                None,
                None,
                ctypes.byref(completed),
            ),
            BH_CPU_OK,
        )
        self.assertEqual(completed.value, 0)

    def test_metric_audit_rejects_asymmetry_and_bad_inverse(self) -> None:
        sample, _covector = next(sample_cases(1))
        covariant = list(flatten_matrix(sample.covariant))
        inverse = list(flatten_matrix(sample.inverse))
        derivatives = flatten_derivatives(sample.inverse_derivatives)
        covariant[1] = covariant[1] + 1.0e-3
        self.assertEqual(
            self.library.bh_cpu_metric_algebra_audit(
                double_array(covariant),
                double_array(inverse),
                double_array(derivatives),
                METRIC_CONSISTENCY_TOLERANCE,
                METRIC_CONSISTENCY_ABSOLUTE_TOLERANCE,
            ),
            BH_CPU_INCONSISTENT_METRIC,
        )
        covariant = list(flatten_matrix(sample.covariant))
        inverse[0] = inverse[0] + 1.0e-3
        self.assertEqual(
            self.library.bh_cpu_metric_algebra_audit(
                double_array(covariant),
                double_array(inverse),
                double_array(derivatives),
                METRIC_CONSISTENCY_TOLERANCE,
                METRIC_CONSISTENCY_ABSOLUTE_TOLERANCE,
            ),
            BH_CPU_INCONSISTENT_METRIC,
        )


if __name__ == "__main__":
    unittest.main()
