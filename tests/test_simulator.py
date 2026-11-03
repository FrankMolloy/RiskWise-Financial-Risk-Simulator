import unittest

import numpy as np

from simulator import simulate


class SimulationTests(unittest.TestCase):
    def run_model(self, **changes):
        params = dict(years=1, start_balance=1000, monthly_contribution=100,
                      expected_return_annual=.07, volatility_annual=.15,
                      inflation_annual=.02, simulations=2, seed=42)
        return simulate(**(params | changes))

    def test_seeded_legacy_regression(self):
        # Captured from the original engine before RNG isolation.
        expected = [[2355.54599240, 2416.39139342, 2475.86843140],
                    [1748.16450902, 1858.76728534, 1850.08198402]]
        np.testing.assert_allclose(self.run_model()[:, -3:], expected, rtol=1e-11)

    def test_no_growth_has_exact_end_of_month_contributions(self):
        actual = self.run_model(expected_return_annual=0, volatility_annual=0,
                                inflation_annual=0)
        np.testing.assert_array_equal(actual, np.tile(1000 + 100 * np.arange(13), (2, 1)))

    def test_zero_volatility_matches_annuity_formula(self):
        months = np.arange(13)
        growth = 1 + .07 / 12
        nominal = 1000 * growth ** months + 100 * (growth ** months - 1) / (growth - 1)
        expected = nominal / 1.02 ** (months / 12)
        np.testing.assert_allclose(self.run_model(volatility_annual=0)[0], expected)

    def test_zero_year_horizon(self):
        np.testing.assert_array_equal(self.run_model(years=0), [[1000], [1000]])

    def test_local_rng_does_not_change_global_state(self):
        np.random.seed(73)
        before = np.random.get_state()
        self.run_model()
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_repeatable_but_seed_changes_paths(self):
        np.testing.assert_array_equal(self.run_model(), self.run_model())
        self.assertFalse(np.array_equal(self.run_model(), self.run_model(seed=43)))

    def test_expected_terminal_mean_under_independent_returns(self):
        # No contributions: E[B_T] = B_0 (1 + mu/12)^T.
        final = self.run_model(simulations=50000, monthly_contribution=0,
                               inflation_annual=0)[:, -1]
        expected = 1000 * (1 + .07 / 12) ** 12
        self.assertLess(abs(final.mean() - expected), 5 * final.std(ddof=1) / np.sqrt(final.size))

    def test_invalid_parameters(self):
        for change in [dict(years=-1), dict(years=1.5), dict(years=True),
                       dict(simulations=0), dict(simulations=2.5),
                       dict(start_balance=-1), dict(monthly_contribution=-1),
                       dict(volatility_annual=-.1), dict(inflation_annual=-1),
                       dict(expected_return_annual=np.nan), dict(start_balance=np.inf)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_model(**change)
