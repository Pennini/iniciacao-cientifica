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

try:
    from ..config import (
        X_TRAIN_FILE,
        X_TEST_FILE,
        Y_TRAIN_FILE,
        Y_TEST_FILE,
        HAR_PRED_FILE,
        HAR_METRICS_FILE,
        FEATURES,
    )
except ImportError:  # Compatibilidade para execução direta
    from config import (  # type: ignore
        X_TRAIN_FILE,
        X_TEST_FILE,
        Y_TRAIN_FILE,
        Y_TEST_FILE,
        HAR_PRED_FILE,
        HAR_METRICS_FILE,
        FEATURES,
    )
from sklearn.metrics import mean_squared_error, mean_absolute_error


class HarModel:
    def __init__(self) -> None:
        self.model = LinearRegression()

    def executar(self, X_train, y_train, X_test, y_test, test_dates):
        self.train(X_train, y_train)

        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)

        self.evaluate_and_visualize(
            y_test, y_pred_test, test_dates=test_dates
        )

        return y_pred_test

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

    def visualize_predictions(self, dates, y_true, y_pred):
        # Configuração do estilo para paper
        plt.style.use("seaborn-v0_8-paper")

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

        # Plotar dados com cores profissionais
        ax.plot(
            dates, y_true, label="Observado", color="blue", linewidth=1.5, alpha=0.8
        )
        ax.plot(
            dates,
            y_pred,
            label="Predito",
            color="red",
            linewidth=1.5,
            linestyle="--",
            alpha=0.9,
        )

        # Labels e título
        ax.set_xlabel("Data", fontsize=11, fontweight="normal")
        ax.set_ylabel("Volatilidade Realizada", fontsize=11, fontweight="normal")

        # Legenda
        ax.legend(
            loc="best",
            frameon=True,
            framealpha=0.95,
            edgecolor="gray",
            fontsize=10,
            fancybox=False,
        )

        # Remover spines superiores e direitas
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Melhorar os ticks
        ax.tick_params(axis="both", which="major", labelsize=9)

        # Formatar eixo Y para notação científica padronizada
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        # Ajustar espaçamento
        plt.tight_layout()
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

    def evaluate_and_visualize(
        self, y_test, y_pred_test, test_dates=None
    ):
        test_mse, test_mae, test_rmse, test_mape = self.evaluate(y_test, y_pred_test)

        print("\n=== AVALIAÇÃO FINAL - LinearRegression (HAR) ===")
        print("\nTESTE:")
        print(f"  MSE: {test_mse:.6e}")
        print(f"  MAE: {test_mae:.6e}")
        print(f"  RMSE: {test_rmse:.6e}")
        print(f"  MAPE: {test_mape:.6e}")
        print("=" * 40)

        if test_dates is not None:
            print("\nVisualizando previsões no conjunto de teste...")
            self.visualize_predictions(test_dates, y_test, y_pred_test)

        metrics = {
            "MSE": test_mse,
            "RMSE": test_rmse,
            "MAE": test_mae,
            "MAPE": test_mape
        }

        return metrics, y_test, y_pred_test

    def main(self, verbose=True):
        self.carregar_dados()
        self.train(self.X_train, self.y_train)
        y_pred_train = self.predict(self.X_train)
        y_pred_test = self.predict(self.X_test)

        if verbose:
            self.evaluate_and_visualize(
                self.y_test, y_pred_test
            )
        self.salvar_resultados(self.y_test, y_pred_test)

        return True


if __name__ == "__main__":
    har_model = HarModel()
    har_model.main()
