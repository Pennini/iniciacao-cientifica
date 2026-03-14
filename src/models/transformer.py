import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
)

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_metrics(eval_pred):
    pred, labels = eval_pred
    pred = pred[0] if isinstance(pred, tuple) else pred

    # 1-step ahead
    pred_1 = pred[:, 0, 0]
    label_1 = labels[:, 0, 0]

    mse = mean_squared_error(label_1, pred_1)
    mae = mean_absolute_error(label_1, pred_1)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((label_1 - pred_1) / (np.abs(label_1) + 1e-9))) * 100

    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE": mape}

# ============================================================
# Desnormalização
# ============================================================

def inverse_transform_targets(tsp, values, target_column):

    scaler = next(iter(tsp.target_scaler_dict.values()))

    mean = scaler.mean_[0]
    std  = scaler.scale_[0]

    return values * std + mean


# ============================================================
# Avaliação + Plot
# ============================================================

def evaluate_and_visualize(
    model,
    test_dataset,
    tsp,
    test_df,
    timestamp_col,
    target_column,
    context_length,
    model_name="Modelo"
):

    # ============================
    # Predict
    # ============================

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="metric_temp",
            per_device_eval_batch_size=32,
            label_names=["future_values"]
        )
    )

    outputs = trainer.predict(test_dataset)

    preds = outputs.predictions
    labels = outputs.label_ids

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.array(preds)
    labels = np.array(labels)

    # ============================
    # Padroniza shape
    # ============================

    if preds.ndim == 2:
        preds = preds[..., None]
        labels = labels[..., None]

    # ============================
    # Desnormalização
    # ============================

    preds_real = inverse_transform_targets(tsp, preds, target_column)
    labels_real = inverse_transform_targets(tsp, labels, target_column)

    # ============================
    # 1-step
    # ============================

    preds_1 = preds_real[:, 0, 0]
    labels_1 = labels_real[:, 0, 0]

    # ============================
    # Datas
    # ============================

    dates = test_df[timestamp_col].values
    forecast_dates = dates[context_length : context_length + len(preds_1)]

    # ============================
    # Métricas
    # ============================

    metrics = {
        "MSE": mean_squared_error(labels_1, preds_1),
        "MAE": mean_absolute_error(labels_1, preds_1),
        "RMSE": np.sqrt(mean_squared_error(labels_1, preds_1)),
        "MAPE": np.mean(
            np.abs((labels_1 - preds_1) /
            (np.abs(labels_1) + 1e-8))
        ) * 100
    }

    print(f"\n=== {model_name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6e}")

    # ============================
    # Plot
    # ============================

    plt.figure(figsize=(12, 6))
    plt.plot(forecast_dates, labels_1, label="True", color='blue')
    plt.plot(forecast_dates, preds_1, "--", label="Predicted", color='red')
    plt.title(f"{model_name} – True vs Predicted")
    plt.xlabel("Time")
    plt.ylabel(str(target_column))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return forecast_dates, metrics, labels_1, preds_1


def train_and_evaluate_patchtst(
    train_dataset,
) -> Dict[str, Any]:
    """
    Função geral para treinar, avaliar e visualizar um modelo PatchTST.

    Parâmetros:
    -----------
    train_dataset : ForecastDFDataset
        Dataset de treinamento
    valid_dataset : ForecastDFDataset
        Dataset de validação
    test_dataset : ForecastDFDataset
        Dataset de teste
    test_df : pd.DataFrame
        DataFrame de teste com coluna de timestamp
    tsp : TimeSeriesPreprocessor
        Preprocessador para desnormalização
    model_config : dict
        Configuração do modelo contendo:
        - context_length (int)
        - patch_length (int)
        - num_input_channels (int)
        - prediction_length (int)
        - d_model (int)
        - num_attention_heads (int)
        - num_hidden_layers (int)
        - ffn_dim (int)
        - dropout (float)
        - E outros parâmetros do PatchTSTConfig
    training_config : dict
        Configuração de treinamento contendo:
        - learning_rate (float)
        - num_epochs (int)
        - batch_size (int)
        - num_workers (int)
        - early_stopping_patience (int, opcional)
        - early_stopping_threshold (float, opcional)
        - E outros parâmetros do TrainingArguments
    device : torch.device
        Dispositivo para treino (cuda ou cpu)
    dtype : torch.dtype
        Tipo de dados (float32, float16, bfloat16)
    context_length : int
        Tamanho da janela temporal de entrada
    timestamp_column : str
        Nome da coluna com timestamps
    model_name : str
        Nome do modelo (para visualizações)
    output_dir : str
        Diretório para salvar checkpoints
    callbacks : list, opcional
        Lista de callbacks customizados (além do early stopping)
    verbose : bool
        Se True, imprime progresso detalhado

    Retorna:
    --------
    dict com chaves:
        - 'model': Modelo treinado
        - 'trainer': Trainer do HF
        - 'train_result': Resultado do treinamento
        - 'eval_result': Resultado da validação
        - 'test_metrics': Métricas no conjunto de teste
        - 'forecast_dates': Datas das previsões
        - 'predictions': Previsões desnormalizadas
        - 'labels': Rótulos desnormalizados
    """
    pass
