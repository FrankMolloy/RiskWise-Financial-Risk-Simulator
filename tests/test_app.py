import unittest
from unittest.mock import patch

import requests

from app import server, solve_monthly, update_dashboard


class AppTests(unittest.TestCase):
    def test_dash_http_routes(self):
        with server.test_client() as client:
            for route in ['/', '/_dash-layout', '/_dash-dependencies']:
                with self.subTest(route=route):
                    self.assertEqual(client.get(route).status_code, 200)

    def dashboard(self, **changes):
        inputs = dict(scenario='balanced', scenario_b='aggressive', compare=[],
                      sim_quality='fast', years=2, monthly=100, goal=5000,
                      use_real_data=[], asset='spy.us', fees_pct=.3,
                      crash_on=[], crash_sev=30, sens_on=[])
        return update_dashboard(**(inputs | changes))

    def test_dashboard_generates_charts_and_statistics(self):
        result = self.dashboard()
        self.assertEqual(len(result), 12)
        self.assertEqual(len(result[0].data), 4)
        self.assertIn('%', result[5])

    def test_comparison_stress_and_sensitivity(self):
        result = self.dashboard(compare=['on'], crash_on=['on'], sens_on=['on'])
        self.assertEqual(len(result[0].data), 7)
        self.assertEqual(len(result[1].data), 2)

    @patch('app.fetch_stooq_close_prices', side_effect=requests.Timeout)
    def test_market_failure_keeps_dashboard_working(self, fetch):
        result = self.dashboard(use_real_data=['on'])
        self.assertIn('Using presets', result[2])
        self.assertIn('Using presets', result[9].children[-1].children)

    def test_solver_reports_unreachable(self):
        message, stored = solve_monthly(1, 'balanced', 1, 1e12, 75, [], 'spy.us', .3, [], 30)
        self.assertIn('search limit', message)
        self.assertEqual(message, stored)

    def test_solver_disabled_without_goal(self):
        message, _ = solve_monthly(1, 'balanced', 1, 0, 75, [], 'spy.us', .3, [], 30)
        self.assertIn('above 0', message)

    @patch('app.fetch_stooq_close_prices', side_effect=requests.Timeout)
    def test_solver_discloses_calibration_fallback(self, fetch):
        message, _ = solve_monthly(1, 'balanced', 1, 10000, 75, ['on'], 'spy.us', .3, [], 30)
        self.assertIn('using preset assumptions', message)

    def test_solver_waits_for_click(self):
        message, stored = solve_monthly(0, 'balanced', 1, 10000, 75, [], 'spy.us', .3, [], 30)
        self.assertIn('click Solve', message)
        self.assertEqual(stored, '')
