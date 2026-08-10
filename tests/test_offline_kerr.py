from __future__ import annotations

import math
import unittest

from offline.geodesic import (
    HamiltonianState,
    RayTraceOptions,
    hamiltonian_null_residual,
    trace_null_geodesic,
)
from offline.kerr import (
    KerrKerrSchildMetric,
    KerrOblateTermination,
    kerr_bl_vector_to_ks_cartesian,
    kerr_bl_zamo_tetrad,
    kerr_constants_of_motion,
    kerr_ks_event_to_oblate,
    kerr_ks_event_to_oblate_meridional,
    kerr_oblate_event_to_ks_cartesian,
    kerr_oblate_radius_m,
    kerr_zamo_camera_ray,
    stationary_axisymmetric_constants,
)
from offline.spacetime import MinkowskiMetric, SchwarzschildKerrSchildMetric, bilinear
from scripts.generate_kerr_transfer_map import (
    OUTCOME_CAPTURED,
    OUTCOME_ESCAPED,
    SPIN_A_M,
    VERTICAL_FOV_RAD,
    _initial_constants_and_state,
    _integrate_primary,
    solve_ray,
)
from scripts.verify_kerr_transfer_map import _oracle_integrate


def _determinant(matrix: tuple[tuple[float, ...], ...]) -> float:
    working = [list(row) for row in matrix]
    determinant = 1.0
    for column in range(4):
        pivot = max(range(column, 4), key=lambda row: abs(working[row][column]))
        if pivot != column:
            working[column], working[pivot] = working[pivot], working[column]
            determinant = -determinant
        value = working[column][column]
        determinant *= value
        for row in range(column + 1, 4):
            factor = working[row][column] / value
            for inner in range(column + 1, 4):
                working[row][inner] -= factor * working[column][inner]
    return determinant


def _ks_position(
    radius_m: float,
    theta_rad: float,
    phi_ks_rad: float,
    spin_a_m: float,
) -> tuple[float, float, float]:
    sine = math.sin(theta_rad)
    return (
        (radius_m * math.cos(phi_ks_rad) - spin_a_m * math.sin(phi_ks_rad))
        * sine,
        (radius_m * math.sin(phi_ks_rad) + spin_a_m * math.cos(phi_ks_rad))
        * sine,
        radius_m * math.cos(theta_rad),
    )


def _angular_separation(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    cross = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    return math.atan2(
        math.sqrt(math.fsum(value * value for value in cross)),
        math.fsum(first[index] * second[index] for index in range(3)),
    )


def _terminal_bl_angles(
    state: HamiltonianState,
    spin_a_m: float,
) -> tuple[float, float]:
    radius = kerr_oblate_radius_m(*state.event[1:], spin_a_m)
    return (
        math.acos(state.event[3] / radius),
        math.atan2(state.event[2], state.event[1]) - math.atan2(spin_a_m, radius),
    )


def _icrs_direction(theta_rad: float, phi_rad: float) -> tuple[float, float, float]:
    world = (
        math.sin(theta_rad) * math.cos(phi_rad),
        math.sin(theta_rad) * math.sin(phi_rad),
        math.cos(theta_rad),
    )
    return (-world[1], world[0], world[2])


class OfflineKerrTests(unittest.TestCase):
    def test_metric_and_worldtube_inputs_are_canonical_exact_primitives(self) -> None:
        class SplitSpin(float):
            def __abs__(self):
                return 0.7

            def __mul__(self, other):
                return 0.7 * other

            __rmul__ = __mul__

            def __truediv__(self, other):
                return 0.7 / other

        metric = KerrKerrSchildMetric(
            mass_m=2,
            spin_a_m=1,
            singularity_guard_m=1,
        )
        self.assertIs(type(metric.mass_m), float)
        self.assertIs(type(metric.spin_a_m), float)
        self.assertIs(type(metric.singularity_guard_m), float)
        with self.assertRaises(TypeError):
            KerrKerrSchildMetric(spin_a_m=SplitSpin(0.6))
        with self.assertRaises(TypeError):
            KerrKerrSchildMetric(mass_m=True)
        with self.assertRaises(TypeError):
            KerrKerrSchildMetric(singularity_guard_m=True)
        with self.assertRaisesRegex(ValueError, "exact analytic provider"):
            KerrKerrSchildMetric(source_id="claimed-kerr")
        with self.assertRaisesRegex(ValueError, "exactly stationary"):
            KerrKerrSchildMetric(time_dependent=True)

        termination = KerrOblateTermination(0, 2, 10)
        self.assertIs(type(termination.spin_a_m), float)
        self.assertIs(type(termination.capture_radius_m), float)
        self.assertIs(type(termination.escape_radius_m), float)
        with self.assertRaises(TypeError):
            KerrOblateTermination(False, 2.0, 10.0)
        with self.assertRaises(ValueError):
            KerrOblateTermination(0.0, 2.0, 10.0, capture_target_id="")

    def test_public_oblate_event_round_trip_and_vector_transform(self) -> None:
        for spin in (0.0, 0.73, -0.73):
            metric = KerrKerrSchildMetric(mass_m=1.2, spin_a_m=spin)
            for theta, phi in ((0.41, -2.7), (math.pi / 2.0, 2.9), (2.4, 0.2)):
                event = kerr_oblate_event_to_ks_cartesian(
                    coordinate_time_m=-7.0,
                    radius_m=8.4,
                    theta_rad=theta,
                    phi_ks_rad=phi,
                    spin_a_m=spin,
                )
                recovered = kerr_ks_event_to_oblate(metric, event)
                self.assertAlmostEqual(recovered.coordinate_time_m, -7.0)
                self.assertAlmostEqual(recovered.radius_m, 8.4, places=14)
                self.assertAlmostEqual(recovered.theta_rad, theta, places=14)
                self.assertAlmostEqual(
                    math.remainder(recovered.phi_ks_rad - phi, 2.0 * math.pi),
                    0.0,
                    places=14,
                )

                stationary = kerr_bl_vector_to_ks_cartesian(
                    (1.0, 0.0, 0.0, 0.0),
                    mass_m=metric.mass_m,
                    spin_a_m=metric.spin_a_m,
                    radius_m=8.4,
                    theta_rad=theta,
                    phi_ks_rad=phi,
                )
                self.assertEqual(stationary, (1.0, 0.0, 0.0, 0.0))

    def test_meridional_inverse_is_defined_on_spin_axis_without_azimuth(self) -> None:
        metric = KerrKerrSchildMetric(mass_m=1.2, spin_a_m=0.73)
        for theta in (0.0, math.pi):
            event = (
                -7.0,
                0.0,
                0.0,
                8.4 if theta == 0.0 else -8.4,
            )
            recovered = kerr_ks_event_to_oblate_meridional(metric, event)
            with self.subTest(theta=theta):
                self.assertAlmostEqual(recovered.coordinate_time_m, -7.0)
                self.assertAlmostEqual(recovered.radius_m, 8.4, places=14)
                self.assertAlmostEqual(recovered.theta_rad, theta, places=14)
                with self.assertRaisesRegex(ValueError, "azimuth is undefined"):
                    kerr_ks_event_to_oblate(metric, event)

    def test_metric_is_exact_inverse_unit_determinant_and_analytic_derivative(self) -> None:
        metric = KerrKerrSchildMetric(mass_m=1.7, spin_a_m=0.91)
        event = (4.0, 8.3, -2.7, 3.1)
        sample = metric.sample(event)
        self.assertAlmostEqual(_determinant(sample.covariant), -1.0, delta=3.0e-14)
        for row in range(4):
            for column in range(4):
                product = math.fsum(
                    sample.covariant[row][inner] * sample.inverse[inner][column]
                    for inner in range(4)
                )
                self.assertAlmostEqual(product, float(row == column), delta=3.0e-14)

        epsilon = 2.0e-5
        for derivative_axis in range(1, 4):
            plus = list(event)
            minus = list(event)
            plus[derivative_axis] += epsilon
            minus[derivative_axis] -= epsilon
            inverse_plus = metric.sample(tuple(plus)).inverse
            inverse_minus = metric.sample(tuple(minus)).inverse
            for row in range(4):
                for column in range(4):
                    finite_difference = (
                        inverse_plus[row][column] - inverse_minus[row][column]
                    ) / (2.0 * epsilon)
                    self.assertAlmostEqual(
                        sample.inverse_derivatives[derivative_axis][row][column],
                        finite_difference,
                        delta=3.0e-10,
                    )
        self.assertEqual(sample.inverse_derivatives[0], ((0.0,) * 4,) * 4)

    def test_zero_spin_is_the_schwarzschild_kerr_schild_metric(self) -> None:
        kerr = KerrKerrSchildMetric(mass_m=1.3, spin_a_m=0.0)
        schwarzschild = SchwarzschildKerrSchildMetric(mass_m=1.3)
        for event in (
            (0.0, 11.0, -3.0, 2.0),
            (-7.0, -4.0, 9.0, 6.0),
        ):
            first = kerr.sample(event)
            second = schwarzschild.sample(event)
            for first_matrix, second_matrix in (
                (first.covariant, second.covariant),
                (first.inverse, second.inverse),
                *zip(first.inverse_derivatives, second.inverse_derivatives),
            ):
                self.assertLess(
                    max(
                        abs(first_matrix[row][column] - second_matrix[row][column])
                        for row in range(4)
                        for column in range(4)
                    ),
                    3.0e-16,
                )

    def test_spin_sign_horizon_and_ring_are_fail_closed(self) -> None:
        positive = KerrKerrSchildMetric(spin_a_m=0.7)
        negative = KerrKerrSchildMetric(spin_a_m=-0.7)
        positive_sample = positive.sample((0.0, 8.0, 0.0, 0.0))
        negative_sample = negative.sample((0.0, 8.0, 0.0, 0.0))
        self.assertLess(positive_sample.covariant[0][2], 0.0)
        self.assertGreater(negative_sample.covariant[0][2], 0.0)
        self.assertAlmostEqual(
            positive.outer_horizon_radius_m,
            1.0 + math.sqrt(1.0 - 0.7**2),
            places=15,
        )
        self.assertRaises(ValueError, positive.sample, (0.0, 0.7, 0.0, 0.0))
        self.assertRaises(ValueError, KerrKerrSchildMetric, spin_a_m=1.0001)

    def test_null_vector_radius_gradient_and_horizon_normal_invariants(self) -> None:
        for spin in (0.0, SPIN_A_M, 0.999999999999, 1.0, -0.7):
            metric = KerrKerrSchildMetric(spin_a_m=spin)
            for radius in (metric.outer_horizon_radius_m, 3.7, 12.0):
                if radius < metric.outer_horizon_radius_m:
                    continue
                theta = 0.83
                position = _ks_position(radius, theta, -0.47, spin)
                event = (0.0, *position)
                sample = metric.sample(event)
                h = 0.5 * (sample.covariant[0][0] + 1.0)
                spatial_null = tuple(
                    sample.covariant[0][index + 1] / (2.0 * h)
                    for index in range(3)
                )
                self.assertAlmostEqual(
                    math.fsum(value * value for value in spatial_null),
                    1.0,
                    delta=3.0e-14,
                )
                x_m, y_m, z_m = position
                radius_squared = radius * radius
                discriminant_root = math.hypot(
                    x_m * x_m + y_m * y_m + z_m * z_m - spin * spin,
                    2.0 * spin * z_m,
                )
                radius_gradient = (
                    radius_squared * x_m / (radius * discriminant_root),
                    radius_squared * y_m / (radius * discriminant_root),
                    (radius_squared + spin * spin)
                    * z_m
                    / (radius * discriminant_root),
                )
                self.assertAlmostEqual(
                    math.fsum(
                        spatial_null[index] * radius_gradient[index]
                        for index in range(3)
                    ),
                    1.0,
                    delta=3.0e-14,
                )
                normal = (0.0, *radius_gradient)
                normal_norm = bilinear(normal, sample.inverse, normal)
                sigma = radius_squared + spin * spin * math.cos(theta) ** 2
                delta = radius_squared - 2.0 * radius + spin * spin
                self.assertAlmostEqual(normal_norm, delta / sigma, delta=3.0e-14)

            exact = KerrOblateTermination.horizon_worldtube(
                metric,
                escape_radius_m=100.0,
                offset_m=0.0,
            )
            stretched = KerrOblateTermination.horizon_worldtube(
                metric,
                escape_radius_m=100.0,
                offset_m=0.02,
            )
            self.assertEqual(exact.capture_target_id, "analytic-kerr-event-horizon")
            self.assertEqual(
                stretched.capture_target_id,
                "analytic-kerr-stretched-horizon",
            )

    def test_zamo_camera_is_orthonormal_and_matches_separated_constants(self) -> None:
        for spin in (0.0, SPIN_A_M, -SPIN_A_M):
            metric = KerrKerrSchildMetric(spin_a_m=spin)
            tetrad = kerr_bl_zamo_tetrad(metric, observer_radius_m=40.0)
            sample = metric.sample(tetrad.event)
            basis = (
                tetrad.four_velocity,
                tetrad.right,
                tetrad.up,
                tetrad.forward,
            )
            for first in range(4):
                for second in range(4):
                    expected = -1.0 if first == second == 0 else float(first == second)
                    self.assertAlmostEqual(
                        bilinear(basis[first], sample.covariant, basis[second]),
                        expected,
                        delta=7.0e-15,
                    )

            for screen_x, screen_y in ((0.0, 0.0), (0.19, -0.08), (-0.23, 0.11)):
                state = kerr_zamo_camera_ray(
                    metric,
                    observer_radius_m=40.0,
                    screen_x=screen_x,
                    screen_y=screen_y,
                )
                local_frequency = math.fsum(
                    state.covector[index] * tetrad.four_velocity[index]
                    for index in range(4)
                )
                self.assertAlmostEqual(local_frequency, 1.0, delta=5.0e-15)
                self.assertLess(hamiltonian_null_residual(metric, state), 2.0e-15)
                expected, _separated_state = _initial_constants_and_state(
                    screen_x,
                    screen_y,
                    spin,
                )
                energy, angular_momentum = stationary_axisymmetric_constants(state)
                self.assertAlmostEqual(energy, expected.energy, delta=3.0e-15)
                self.assertAlmostEqual(
                    angular_momentum,
                    expected.angular_momentum_z,
                    delta=3.0e-14,
                )
                recovered = kerr_constants_of_motion(metric, state)
                self.assertAlmostEqual(recovered.carter_q, expected.carter_q, delta=4.0e-14)
                self.assertAlmostEqual(recovered.carter_k, expected.carter_k, delta=6.0e-14)

    def test_near_horizon_zamo_transform_fails_closed_when_ill_conditioned(self) -> None:
        for spin in (0.999999999999, 1.0):
            metric = KerrKerrSchildMetric(spin_a_m=spin)
            radius = metric.outer_horizon_radius_m + 1.0e-8
            with self.subTest(spin=spin):
                with self.assertRaisesRegex(ValueError, "ill-conditioned"):
                    kerr_bl_zamo_tetrad(
                        metric,
                        observer_radius_m=radius,
                        theta_rad=1.2,
                    )
                with self.assertRaises(ValueError):
                    kerr_zamo_camera_ray(
                        metric,
                        observer_radius_m=radius,
                        theta_rad=1.2,
                        screen_x=0.1,
                        screen_y=-0.05,
                    )

    def test_carter_recovery_is_stable_near_the_spin_axis(self) -> None:
        metric = KerrKerrSchildMetric(spin_a_m=0.9)
        radius = 40.0
        screen_x = 0.05
        screen_y = 0.08
        inverse_norm = 1.0 / math.sqrt(
            1.0 + screen_x * screen_x + screen_y * screen_y
        )
        for theta in (1.0e-6, 1.0e-8, math.pi - 1.0e-8):
            state = kerr_zamo_camera_ray(
                metric,
                observer_radius_m=radius,
                theta_rad=theta,
                screen_x=screen_x,
                screen_y=screen_y,
            )
            recovered = kerr_constants_of_motion(metric, state)
            p_theta = -screen_y * inverse_norm * math.sqrt(
                radius * radius + metric.spin_a_m**2 * math.cos(theta) ** 2
            )
            expected_q = p_theta * p_theta + math.cos(theta) ** 2 * (
                recovered.angular_momentum_z**2 / math.sin(theta) ** 2
                - metric.spin_a_m**2 * recovered.energy**2
            )
            with self.subTest(theta=theta):
                self.assertTrue(
                    math.isclose(
                        recovered.carter_q,
                        expected_q,
                        rel_tol=2.0e-10,
                        abs_tol=2.0e-10,
                    )
                )

    def test_oblate_event_refinement_catches_an_interior_leap(self) -> None:
        termination = KerrOblateTermination(
            spin_a_m=0.8,
            capture_radius_m=1.0,
            escape_radius_m=10.0,
        )
        start = HamiltonianState(
            event=(0.0, -2.0, 0.0, 0.0),
            covector=(1.0, 1.0, 0.0, 0.0),
        )
        end = HamiltonianState(
            event=(-4.0, 2.0, 0.0, 0.0),
            covector=start.covector,
        )
        self.assertTrue(termination.needs_refinement(start, end))
        result = trace_null_geodesic(
            MinkowskiMetric(),
            start,
            termination=termination,
            options=RayTraceOptions(
                initial_step=4.0,
                maximum_step=4.0,
                maximum_affine_length=6.0,
            ),
        )
        self.assertEqual(result.outcome, "captured", result.failure_reason)
        self.assertEqual(result.terminal_target_id, termination.capture_target_id)
        self.assertAlmostEqual(termination.radius(result.terminal_state), 1.0, delta=2.0e-9)

    def test_full_cartesian_hamiltonian_matches_separated_kerr_outcomes(self) -> None:
        options = RayTraceOptions(
            absolute_tolerance=2.0e-9,
            relative_tolerance=2.0e-9,
            maximum_step=0.5,
            maximum_affine_length=1_000.0,
            null_residual_limit=2.0e-7,
        )
        focal_pixels = 576.0 / (2.0 * math.tan(0.5 * VERTICAL_FOV_RAD))
        outcome_names = {OUTCOME_CAPTURED: "captured", OUTCOME_ESCAPED: "escaped"}
        cases = (
            (
                SPIN_A_M,
                (
                    (0.0, 0.0),
                    (0.30, 0.0),
                    (-0.30, 0.0),
                    (0.15, 0.0),
                    (-0.15, 0.0),
                ),
            ),
            (-SPIN_A_M, ((0.0, 0.0), (0.15, 0.0), (-0.15, 0.0))),
        )
        for spin, screen_points in cases:
            metric = KerrKerrSchildMetric(spin_a_m=spin)
            termination = KerrOblateTermination.horizon_worldtube(
                metric,
                escape_radius_m=120.0,
            )
            for screen_x, screen_y in screen_points:
                separated = solve_ray(
                    screen_x,
                    screen_y,
                    focal_pixels,
                    spin=spin,
                    refinement_check=False,
                )
                cartesian = trace_null_geodesic(
                    metric,
                    kerr_zamo_camera_ray(
                        metric,
                        observer_radius_m=40.0,
                        screen_x=screen_x,
                        screen_y=screen_y,
                    ),
                    termination=termination,
                    options=options,
                )
                self.assertEqual(cartesian.outcome, outcome_names[separated.outcome])
                self.assertLess(
                    cartesian.maximum_null_residual,
                    options.null_residual_limit,
                )

    def test_escape_event_frequency_time_and_position_match_separated_oracle(self) -> None:
        metric = KerrKerrSchildMetric(spin_a_m=SPIN_A_M)
        screen_x, screen_y = 0.30, 0.08
        initial = kerr_zamo_camera_ray(
            metric,
            observer_radius_m=40.0,
            screen_x=screen_x,
            screen_y=screen_y,
        )
        result = trace_null_geodesic(
            metric,
            initial,
            termination=KerrOblateTermination.horizon_worldtube(
                metric,
                escape_radius_m=1_000.0,
            ),
            options=RayTraceOptions(
                absolute_tolerance=2.0e-10,
                relative_tolerance=2.0e-10,
                maximum_step=1.0,
                maximum_affine_length=3_000.0,
                null_residual_limit=1.0e-7,
            ),
        )
        self.assertEqual(result.outcome, "escaped", result.failure_reason)

        constants, separated_initial = _initial_constants_and_state(
            screen_x,
            screen_y,
            SPIN_A_M,
        )
        separated_outcome, terminal, *_diagnostics = _integrate_primary(
            separated_initial,
            constants,
            absolute_tolerance=2.0e-12,
            relative_tolerance=2.0e-11,
        )
        self.assertEqual(separated_outcome, "escaped")
        radius = 1.0 / terminal[0]
        theta = terminal[2]
        phi_ks = terminal[4]
        expected_event = (
            terminal[5],
            (radius * math.cos(phi_ks) - SPIN_A_M * math.sin(phi_ks))
            * math.sin(theta),
            (radius * math.sin(phi_ks) + SPIN_A_M * math.cos(phi_ks))
            * math.sin(theta),
            radius * math.cos(theta),
        )
        self.assertLess(
            max(
                abs(result.terminal_state.event[index] - expected_event[index])
                for index in range(4)
            ),
            1.0e-6,
        )

        terminal_radius = kerr_oblate_radius_m(
            *result.terminal_state.event[1:],
            SPIN_A_M,
        )
        terminal_theta = math.acos(result.terminal_state.event[3] / terminal_radius)
        terminal_phi = (
            math.atan2(result.terminal_state.event[2], result.terminal_state.event[1])
            - math.atan2(SPIN_A_M, terminal_radius)
        )
        boundary_zamo = kerr_bl_zamo_tetrad(
            metric,
            observer_radius_m=terminal_radius,
            theta_rad=terminal_theta,
            phi_ks_rad=terminal_phi,
            coordinate_time_m=result.terminal_state.event[0],
        )
        boundary_frequency = math.fsum(
            result.terminal_state.covector[index] * boundary_zamo.four_velocity[index]
            for index in range(4)
        )
        expected_frequency = (
            -constants.energy
            + 2.0
            * SPIN_A_M
            * terminal_radius
            / (
                (terminal_radius**2 + SPIN_A_M**2) ** 2
                - SPIN_A_M**2
                * (
                    terminal_radius**2
                    - 2.0 * terminal_radius
                    + SPIN_A_M**2
                )
                * math.sin(terminal_theta) ** 2
            )
            * constants.angular_momentum_z
        ) / math.sqrt(
            (
                terminal_radius**2
                + SPIN_A_M**2 * math.cos(terminal_theta) ** 2
            )
            * (
                terminal_radius**2
                - 2.0 * terminal_radius
                + SPIN_A_M**2
            )
            / (
                (terminal_radius**2 + SPIN_A_M**2) ** 2
                - SPIN_A_M**2
                * (
                    terminal_radius**2
                    - 2.0 * terminal_radius
                    + SPIN_A_M**2
                )
                * math.sin(terminal_theta) ** 2
            )
        )
        self.assertAlmostEqual(boundary_frequency, expected_frequency, delta=2.0e-12)

        farther = trace_null_geodesic(
            metric,
            initial,
            termination=KerrOblateTermination.horizon_worldtube(
                metric,
                escape_radius_m=2_000.0,
            ),
            options=RayTraceOptions(
                absolute_tolerance=2.0e-10,
                relative_tolerance=2.0e-10,
                maximum_step=4.0,
                maximum_affine_length=4_000.0,
                null_residual_limit=1.0e-7,
            ),
        )
        self.assertEqual(farther.outcome, "escaped", farther.failure_reason)
        theta_1, phi_1 = _terminal_bl_angles(result.terminal_state, SPIN_A_M)
        theta_2, phi_2 = _terminal_bl_angles(farther.terminal_state, SPIN_A_M)
        phi_increment = math.remainder(phi_2 - phi_1, 2.0 * math.pi)
        richardson_direction = _icrs_direction(
            2.0 * theta_2 - theta_1,
            phi_2 + phi_increment,
        )
        independent = _oracle_integrate(
            screen_x,
            screen_y,
            spin=SPIN_A_M,
            step=1.0e-3,
        )
        self.assertEqual(independent.outcome, OUTCOME_ESCAPED)
        self.assertLess(
            _angular_separation(richardson_direction, independent.direction_icrs),
            1.0e-6,
        )
        self.assertAlmostEqual(
            1.0 / boundary_frequency,
            independent.frequency_shift_g,
            delta=3.0e-12,
        )

    def test_stationary_energy_and_axial_angular_momentum_are_conserved(self) -> None:
        metric = KerrKerrSchildMetric(spin_a_m=SPIN_A_M)
        initial = kerr_zamo_camera_ray(
            metric,
            observer_radius_m=40.0,
            screen_x=0.30,
            screen_y=0.08,
        )
        result = trace_null_geodesic(
            metric,
            initial,
            termination=KerrOblateTermination.horizon_worldtube(
                metric,
                escape_radius_m=120.0,
            ),
            options=RayTraceOptions(
                absolute_tolerance=5.0e-10,
                relative_tolerance=5.0e-10,
                maximum_step=0.4,
                maximum_affine_length=1_000.0,
                null_residual_limit=2.0e-7,
                record_path=True,
            ),
        )
        initial_constants = kerr_constants_of_motion(metric, initial)
        constants = tuple(
            kerr_constants_of_motion(metric, segment.end)
            for segment in result.segments
        )
        self.assertEqual(result.outcome, "escaped", result.failure_reason)
        self.assertLess(
            max(abs(value.energy - initial_constants.energy) for value in constants),
            2.0e-14,
        )
        self.assertLess(
            max(
                abs(value.angular_momentum_z - initial_constants.angular_momentum_z)
                for value in constants
            ),
            3.0e-10,
        )
        self.assertLess(
            max(abs(value.carter_q - initial_constants.carter_q) for value in constants),
            4.0e-9,
        )


if __name__ == "__main__":
    unittest.main()
