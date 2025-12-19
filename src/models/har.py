import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from pathlib import Path
import sys

# Seed para reproducibilidade
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
)
from config import (
    HAR_PRED_FILE,
    HAR_METRICS_FILE,
    FEATURES
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class HarModel():
    def __init__(self) -> None:
        self.model = LinearRegression()

    def carregar_dados(self):
        self.X_train = pd.read_csv(X_TRAIN_FILE)[FEATURES]
        self.y_train = pd.read_csv(Y_TRAIN_FILE).values.ravel()
        self.X_test = pd.read_csv(X_TEST_FILE)[FEATURES]
        self.y_test = pd.read_csv(Y_TEST_FILE).values.ravel()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
        return mse, mae, rmse, mape

    def visualize_predictions(self, y_true, y_pred):
        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label="True Values", color="blue")
        plt.plot(y_pred, label="Predicted Values", color="red", linestyle="--")
        plt.title("HAR: True vs Predicted Values")
        plt.xlabel("Samples")
        plt.ylabel("Target Value")
        plt.legend()
        plt.grid()
        plt.show()
    
    def salvar_resultados(self, y_true, y_pred):
        df_results = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
        df_results.to_csv(HAR_PRED_FILE, index=False)
        print(f"Previsões salvas em: {HAR_PRED_FILE}")

        mse, mae, rmse, mape = self.evaluate(y_true, y_pred)

        metrics = {
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
        }

        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics.to_csv(HAR_METRICS_FILE, index=False)
        print(f"Métricas salvas em: {HAR_METRICS_FILE}")

        return True
    
    def evaluate_and_visualize(self, y_train, y_pred_train, y_test, y_pred_test):
        train_mse, train_mae, train_rmse, train_mape = self.evaluate(y_train, y_pred_train)
        test_mse, test_mae, test_rmse, test_mape = self.evaluate(y_test, y_pred_test)

        print("\n=== AVALIAÇÃO FINAL - LinearRegression (HAR) ===")
        print("TREINO:")
        print(f"  MSE: {train_mse:.6e}")
        print(f"  MAE: {train_mae:.6e}")
        print(f"  RMSE: {train_rmse:.6e}")
        print(f"  MAPE:  {train_mape:.6e}")

        print("\nTESTE:")
        print(f"  MSE: {test_mse:.6e}")
        print(f"  MAE: {test_mae:.6e}")
        print(f"  RMSE: {test_rmse:.6e}")
        print(f"  MAPE:  {test_mape:.6e}")
        print("=" * 40)

        print("\nVisualizando previsões no conjunto de teste...")

        self.visualize_predictions(y_test, y_pred_test)
    
    def main(self, verbose=True):
        self.carregar_dados()
        self.train(self.X_train, self.y_train)
        y_pred_train = self.predict(self.X_train)
        y_pred_test = self.predict(self.X_test)
        
        if verbose:
            self.evaluate_and_visualize(self.y_train, y_pred_train, self.y_test, y_pred_test)
        self.salvar_resultados(self.y_test, y_pred_test)

        return True

if __name__ == "__main__":
    har_model = HarModel()
    har_model.main()