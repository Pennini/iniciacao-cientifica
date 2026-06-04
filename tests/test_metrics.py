import unittest

import numpy as np

from src.metrics import calcular_metricas_regressao


class TestMetrics(unittest.TestCase):
    def test_metricas_retorna_campos_esperados(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.5, 2.5, 3.5])

        metricas = calcular_metricas_regressao(y_true, y_pred)

        for campo in ("MSE", "MAE", "RMSE", "MAPE", "sMAPE"):
            self.assertIn(campo, metricas)
            self.assertGreaterEqual(metricas[campo], 0.0)


if __name__ == "__main__":
    unittest.main()

