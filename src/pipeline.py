from pathlib import Path

import pandas as pd
import torch

from .config import (
    BATCH_SIZE,
    CONTEXT_LENGTH,
    EPOCHS,
    FEATURES,
    FORECAST_HORIZON,
    HAR_METRICS_FILE,
    HAR_PRED_FILE,
    INTERIM_DATA_DIR,
    LEARNING_RATE,
    LOGS_DIR,
    NUM_WORKERS,
    PATCH_LENGTH,
    PATCHTST_METRICS_FILE,
    PATCHTST_PRED_FILE,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    TRAIN_FRAC,
    VALID_FRAC,
    X_TEST_FILE,
    X_TRAIN_FILE,
    X_VAL_FILE,
    Y_TEST_FILE,
    Y_TRAIN_FILE,
    Y_VAL_FILE,
)
from .dataset import RepositorioDados
from .metrics import calcular_metricas_regressao, salvar_metricas_csv, salvar_previsoes_csv
from .models.har import HarModel
from .models.patchtst import avaliar_patchtst, criar_config_patchtst, treinar_patchtst


def _salvar_split_har(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
    for split_df, x_path, y_path in (
        (train_df, X_TRAIN_FILE, Y_TRAIN_FILE),
        (valid_df, X_VAL_FILE, Y_VAL_FILE),
        (test_df, X_TEST_FILE, Y_TEST_FILE),
    ):
        split_df[FEATURES].to_csv(x_path, index=False)
        split_df[TARGET_COLUMN].to_csv(y_path, index=False)


def preparar_dados(
    context_length: int = CONTEXT_LENGTH,
    forecast_horizon: int = FORECAST_HORIZON,
    use_mean_features: bool = True,
    lags: int = 3,
):
    repo = RepositorioDados()
    tsp, train_ds, valid_ds, test_ds, train_df, valid_df, test_df = repo.executar(
        timestamp_col=TIMESTAMP_COLUMN,
        train_frac=TRAIN_FRAC,
        valid_frac=VALID_FRAC,
        context_length=context_length,
        target_col=TARGET_COLUMN,
        id_columns=[],
        forecast_horizon=forecast_horizon,
        use_mean_features=use_mean_features,
        lags=lags,
    )

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(INTERIM_DATA_DIR / "train.csv", index=False)
    valid_df.to_csv(INTERIM_DATA_DIR / "valid.csv", index=False)
    test_df.to_csv(INTERIM_DATA_DIR / "test.csv", index=False)
    _salvar_split_har(train_df, valid_df, test_df)
    return {
        "tsp": tsp,
        "train_ds": train_ds,
        "valid_ds": valid_ds,
        "test_ds": test_ds,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
    }


def treinar_har():
    if not X_TRAIN_FILE.exists() or not Y_TRAIN_FILE.exists():
        preparar_dados()

    X_train = pd.read_csv(X_TRAIN_FILE)[FEATURES]
    y_train = pd.read_csv(Y_TRAIN_FILE)[TARGET_COLUMN].values.ravel()
    X_test = pd.read_csv(X_TEST_FILE)[FEATURES]
    y_test = pd.read_csv(Y_TEST_FILE)[TARGET_COLUMN].values.ravel()
    test_df = pd.read_csv(INTERIM_DATA_DIR / "test.csv")
    test_dates = pd.to_datetime(test_df[TIMESTAMP_COLUMN].values)

    modelo = HarModel()
    modelo.train(X_train, y_train)
    y_pred = modelo.predict(X_test)
    metricas = calcular_metricas_regressao(y_test, y_pred)

    salvar_previsoes_csv(HAR_PRED_FILE, y_test, y_pred, timestamps=test_dates)
    salvar_metricas_csv(HAR_METRICS_FILE, metricas)
    return metricas


def treinar_e_avaliar_patchtst(
    use_mean_features: bool = True,
    lags: int = 3,
    context_length: int = CONTEXT_LENGTH,
    forecast_horizon: int = FORECAST_HORIZON,
):
    dados = preparar_dados(
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        use_mean_features=use_mean_features,
        lags=lags,
    )

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    config = criar_config_patchtst(
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        num_input_channels=len(TARGET_COLUMN),
        patch_length=PATCH_LENGTH,
    )
    output_dir = str(Path("models") / "patchtst" / f"context{context_length}_lags{lags}_use{use_mean_features}")
    logging_dir = str(LOGS_DIR / "patchtst")

    treino = treinar_patchtst(
        train_ds=dados["train_ds"],
        valid_ds=dados["valid_ds"],
        config=config,
        output_dir=output_dir,
        logging_dir=logging_dir,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        dtype=dtype,
    )

    _, _, labels, preds = avaliar_patchtst(
        model=treino["model"],
        test_ds=dados["test_ds"],
        tsp=dados["tsp"],
        test_df=dados["test_df"],
        timestamp_col=TIMESTAMP_COLUMN,
        target_column=TARGET_COLUMN,
        context_length=context_length,
        model_name="PatchTST Scratch",
    )

    metricas = calcular_metricas_regressao(labels, preds)
    timestamps = pd.to_datetime(
        dados["test_df"][TIMESTAMP_COLUMN].iloc[context_length : context_length + len(preds)]
    )
    salvar_previsoes_csv(PATCHTST_PRED_FILE, labels, preds, timestamps=timestamps)
    salvar_metricas_csv(PATCHTST_METRICS_FILE, metricas)
    treino["trainer"].save_model(output_dir)
    return metricas


def comparar_modelos():
    if not HAR_METRICS_FILE.exists():
        treinar_har()
    if not PATCHTST_METRICS_FILE.exists():
        treinar_e_avaliar_patchtst()

    har = pd.read_csv(HAR_METRICS_FILE).assign(modelo="HAR")
    patch = pd.read_csv(PATCHTST_METRICS_FILE).assign(modelo="PatchTST")
    colunas = ["modelo", "MSE", "MAE", "RMSE", "MAPE", "sMAPE"]
    return pd.concat([har, patch], ignore_index=True)[colunas]

