import numpy as np

from transformers import (
    Trainer,
    TrainingArguments    
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
    mape = np.mean(
        np.abs((label_1 - pred_1) / (np.abs(label_1) + 1e-9))
    ) * 100

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

def evaluate_and_visualize(
    model,
    test_dataset,
    tsp,
    test_df,
    model_name="Modelo"
):
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="metric_temp",
            per_device_eval_batch_size=32,
            label_names=["future_values"]
        )
    )

    outputs = trainer.predict(test_dataset)

    preds = outputs.predictions[0] if isinstance(outputs.predictions, tuple) else outputs.predictions
    labels = outputs.label_ids

    # Desnormalização
    C = len(tsp.target_columns)
    scaler = next(iter(tsp.target_scaler_dict.values()))
    
    pred_original = scaler.inverse_transform(preds.reshape(-1, C)).reshape(preds.shape)
    labels_original = scaler.inverse_transform(labels.reshape(-1, C)).reshape(labels.shape)

    # 1-step ahead
    preds_1step = pred_original[:, 0, 0]
    labels_1step = labels_original[:, 0, 0]

    # Datas corretas
    test_dates = test_df[TIMESTAMP_COLUMN].values
    forecast_dates = test_dates[CONTEXT_LENGTH : CONTEXT_LENGTH + len(preds_1step)]

    # Métricas finais (fora do Trainer)
    metrics = {
        "MSE": mean_squared_error(labels_1step, preds_1step),
        "MAE": mean_absolute_error(labels_1step, preds_1step),
        "RMSE": np.sqrt(mean_squared_error(labels_1step, preds_1step)),
        "MAPE": np.mean(np.abs((labels_1step - preds_1step) / (np.abs(labels_1step) + 1e-9))) * 100
    }

    print(f"=== AVALIAÇÃO FINAL – {model_name} ===\n")
    for k, v in metrics.items():
        print(f"{k}: {v:.6e}")

    plt.figure(figsize=(12, 6))
    plt.plot(forecast_dates, labels_1step, label="True", color="blue")
    plt.plot(forecast_dates, preds_1step, label="Predicted", linestyle="--", color="red")
    plt.title(f"{model_name} – True vs Predicted Volatility")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return forecast_dates, metrics