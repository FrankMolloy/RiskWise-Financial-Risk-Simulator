import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

import market_data


class MarketDataTests(unittest.TestCase):
    def setUp(self):
        market_data._CACHE.clear()

    def test_log_return_annualisation_with_sample_std(self):
        log_returns = np.array([.01, -.02, .03])
        prices = pd.Series(100 * np.exp(np.r_[0, np.cumsum(log_returns)]))
        mu, vol = market_data.estimate_annual_return_vol(prices)
        self.assertAlmostEqual(mu, log_returns.mean() * 252)
        self.assertAlmostEqual(vol, log_returns.std(ddof=1) * np.sqrt(252))

    def test_invalid_prices(self):
        for prices in [[1, 2], [1, 0, 3], [1, -1, 3], [1, np.nan, 3]]:
            with self.subTest(prices=prices), self.assertRaises(ValueError):
                market_data.estimate_annual_return_vol(pd.Series(prices))

    @patch('market_data.requests.get')
    def test_fetch_parses_sorts_and_caches_without_network(self, get):
        dates = pd.date_range('2020-01-01', periods=201)
        csv = pd.DataFrame({'Date': dates, 'Close': np.arange(201) + 100})
        get.return_value = Mock(text=csv.iloc[::-1].to_csv(index=False))
        first = market_data.fetch_stooq_close_prices(' SPY.US ')
        second = market_data.fetch_stooq_close_prices('spy.us')
        self.assertTrue(first.index.is_monotonic_increasing)
        pd.testing.assert_series_equal(first, second)
        get.assert_called_once_with('https://stooq.com/q/d/l/?s=spy.us&i=d', timeout=10)

    @patch('market_data.requests.get')
    def test_invalid_csv(self, get):
        get.return_value = Mock(text='No data')
        with self.assertRaises(ValueError):
            market_data.fetch_stooq_close_prices('spy.us')

    @patch('market_data.requests.get')
    def test_insufficient_history_not_cached(self, get):
        get.return_value = Mock(text='Date,Close\n2020-01-01,100\n')
        with self.assertRaises(ValueError):
            market_data.fetch_stooq_close_prices('spy.us')
        self.assertFalse(market_data._CACHE)

    @patch('market_data.time.time')
    @patch('market_data.requests.get')
    def test_cache_expiry_refetches_prices(self, get, clock):
        csv = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=200),
                            'Close': np.arange(200) + 100}).to_csv(index=False)
        get.return_value = Mock(text=csv)
        clock.side_effect = [100, 3701]
        market_data.fetch_stooq_close_prices('spy.us')
        market_data.fetch_stooq_close_prices('spy.us')
        self.assertEqual(get.call_count, 2)

    @patch('market_data.requests.get', side_effect=requests.Timeout)
    def test_timeout_not_cached(self, get):
        with self.assertRaises(requests.Timeout):
            market_data.fetch_stooq_close_prices('spy.us')
        self.assertFalse(market_data._CACHE)
