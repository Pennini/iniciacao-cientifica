import unittest

import numpy as np
import pandas as pd

from src.dataset import RepositorioDados


class TestRepositorioDados(unittest.TestCase):
    def setUp(self):
        self.repo = RepositorioDados()

    def test_validate_fracs(self):
        self.repo._validate_fracs(0.7, 0.2)
        with self.assertRaises(ValueError):
            self.repo._validate_fracs(1.0, 0.0)
        with self.assertRaises(ValueError):
            self.repo._validate_fracs(0.8, 0.3)

    def test_calculate_features_gera_colunas_esperadas(self):
        n = 5000  # ~17 dias em frequência de 5 minutos
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=n, freq="5min"),
                "close": np.linspace(100.0, 110.0, n),
            }
        )
        out = self.repo.calculate_features(
            df,
            timestamp_col="timestamp",
            lags=3,
            use_mean_features=True,
        )

        self.assertIn("Vol", out.columns)
        self.assertIn("Vol_lag_1", out.columns)
        self.assertIn("Vol_lag_2", out.columns)
        self.assertIn("Vol_lag_3", out.columns)
        self.assertIn("Vol_week_mean", out.columns)
        self.assertIn("Vol_month_mean", out.columns)
        self.assertFalse(out.empty)


if __name__ == "__main__":
    unittest.main()
