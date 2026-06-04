from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calcular_metricas_regressao(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100)
    smape = float(
        np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9))
        * 100
    )

    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
    }


def salvar_metricas_csv(path: Path, metricas: Dict[str, float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metricas]).to_csv(path, index=False)
    return path


def salvar_previsoes_csv(
    path: Path,
    y_true,
    y_pred,
    timestamps: Optional[pd.Series] = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"y_true": np.asarray(y_true).reshape(-1), "y_pred": np.asarray(y_pred).reshape(-1)})
    if timestamps is not None:
        df.insert(0, "timestamp", pd.to_datetime(timestamps))
    df.to_csv(path, index=False)
    return path

