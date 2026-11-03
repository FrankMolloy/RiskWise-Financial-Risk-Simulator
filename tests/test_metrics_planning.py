import unittest

import numpy as np

from metrics import prob_reach_goal, summarise
from planning import required_monthly
from projections import apply_crash, deterministic_forecast
from simulator import simulate


class MetricsTests(unittest.TestCase):
    def test_known_percentiles(self):
        self.assertEqual(summarise(np.arange(11)),
                         dict(median=5, mean=5, p10=1, p90=9, min=0, max=10))

    def test_interpolated_percentiles_of_unsorted_samples(self):
        # Sorted [0, 10, 20, 100]: rank 0.3 gives 3, rank 2.7 gives 76.
        summary = summarise(np.array([100, 0, 20, 10]))
        self.assertAlmostEqual(summary['p10'], 3)
        self.assertAlmostEqual(summary['p90'], 76)
        self.assertEqual(summary['median'], 15)

    def test_nonfinite_goal_is_rejected(self):
        with self.assertRaises(ValueError):
            prob_reach_goal(np.array([1, 2]), np.nan)

    def test_goal_includes_equality_and_disabled_goal(self):
        self.assertEqual(prob_reach_goal(np.array([1, 2, 3, 4]), 3), .5)
        self.assertEqual(prob_reach_goal(np.array([1, 2]), 0), 0)
        self.assertEqual(prob_reach_goal(np.array([1, 2]), 10), 0)

    def test_invalid_samples(self):
        for values in [[], [np.nan], [np.inf], [[1, 2]]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                summarise(np.array(values))
            with self.subTest(values=values), self.assertRaises(ValueError):
                prob_reach_goal(np.array(values), 1)


class PlanningTests(unittest.TestCase):
    def test_unreachable_target(self):
        self.assertIsNone(required_monthly(lambda monthly: .1, .75))

    def test_already_funded(self):
        self.assertEqual(required_monthly(lambda monthly: 1, .75), 0)

    def test_rounds_up_to_feasible_ten_pounds(self):
        probability = lambda monthly: float(monthly >= 1004)
        result = required_monthly(probability, .75)
        self.assertEqual(result, 1010)
        self.assertGreaterEqual(probability(result), .75)

    def test_upper_bound_is_reachable(self):
        self.assertEqual(required_monthly(lambda m: float(m >= 5000), .75), 5000)

    def test_invalid_target(self):
        for target in [0, -1, 1.1, np.nan]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                required_monthly(lambda m: 1, target)

    def test_closed_form_contribution_boundary(self):
        # With no growth/inflation, £1,000 + 12*C must reach £13,000.
        def probability(monthly):
            paths = simulate(1, 1000, monthly, 0, 0, 0, simulations=3)
            return prob_reach_goal(paths[:, -1], 13000)
        result = required_monthly(probability, .75)
        self.assertGreaterEqual(result, 1000)
        self.assertLessEqual(result, 1010)  # bounded search may overshoot one £10 step
        self.assertGreaterEqual(probability(result), .75)

    def test_stochastic_recommendation_meets_sample_target_with_stress(self):
        def probability(monthly):
            paths = simulate(10, 1000, monthly, .067, .15, .02,
                             simulations=900, seed=123)
            paths = apply_crash(paths, .3, seed=2001)
            return prob_reach_goal(paths[:, -1], 50000)
        result = required_monthly(probability, .75)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(probability(result), .75)
        self.assertLess(probability(result - 20), .75)

    def test_invalid_search_configuration(self):
        for changes in [dict(upper=0), dict(upper=999), dict(iterations=0),
                        dict(iterations=True)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                required_monthly(lambda m: 1, .75, **changes)


class ProjectionTests(unittest.TestCase):
    def test_crash_is_repeatable_without_mutating_input(self):
        paths = np.full((5, 121), 100.0)
        actual = apply_crash(paths, .3)
        np.testing.assert_array_equal(paths, 100)
        np.testing.assert_array_equal(actual[:, 0], 100)
        np.testing.assert_array_equal(actual[:, -1], 70)
        np.testing.assert_array_equal(actual, apply_crash(paths, .3))

    def test_crash_disabled(self):
        paths = np.ones((2, 13))
        np.testing.assert_array_equal(apply_crash(paths, 0), paths)

    def test_crash_zero_horizon_preserves_initial_balance(self):
        np.testing.assert_array_equal(apply_crash(np.array([[1000.]]), .3), [[1000.]])

    def test_one_month_crash_preserves_initial_balance(self):
        np.testing.assert_array_equal(apply_crash(np.array([[1000., 1100.]]), .3),
                                      [[1000., 770.]])

    def test_crash_rejects_invalid_severity(self):
        for severity in [-.1, 1.1, np.nan]:
            with self.subTest(severity=severity), self.assertRaises(ValueError):
                apply_crash(np.ones((2, 13)), severity)

    def test_deterministic_compounds_annual_rate(self):
        path = deterministic_forecast(1, 1000, 0, .07, .02, .01)
        self.assertAlmostEqual(path[-1], 1000 * 1.06 / 1.02)
