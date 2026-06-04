from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
)

from ..config import (
    D_MODEL,
    DROPOUT,
    FFN_DIM,
    NUM_ATTENTION_HEADS,
    NUM_HIDDEN_LAYERS,
    SCALING,
)
from .transformer import compute_metrics, evaluate_and_visualize


def criar_config_patchtst(
    context_length: int,
    forecast_horizon: int,
    num_input_channels: int,
    patch_length: int,
    d_model: int = D_MODEL,
    num_attention_heads: int = NUM_ATTENTION_HEADS,
    num_hidden_layers: int = NUM_HIDDEN_LAYERS,
    ffn_dim: int = FFN_DIM,
    dropout: float = DROPOUT,
    scaling=SCALING,
) -> PatchTSTConfig:
    return PatchTSTConfig(
        do_mask_input=False,
        context_length=context_length,
        patch_length=patch_length,
        num_input_channels=num_input_channels,
        patch_stride=patch_length,
        prediction_length=forecast_horizon,
        d_model=d_model,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
        head_dropout=dropout,
        pooling_type=None,
        channel_attention=True,
        scaling=scaling,
        loss="mse",
        pre_norm=True,
        norm_type="batchnorm",
    )


def carregar_patchtst_pre_treinado(
    model_name_or_path: str,
    num_input_channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> PatchTSTForPrediction:
    config = PatchTSTConfig.from_pretrained(model_name_or_path)
    config.num_input_channels = num_input_channels
    return PatchTSTForPrediction.from_pretrained(
        model_name_or_path,
        config=config,
        local_files_only=Path(model_name_or_path).exists(),
    ).to(device).to(dtype)


def treinar_patchtst(
    train_ds,
    valid_ds,
    config: PatchTSTConfig,
    output_dir: str,
    logging_dir: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
    dtype: torch.dtype,
    early_stopping_patience: int = 7,
) -> Dict[str, Any]:
    model = PatchTSTForPrediction(config=config).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ).to(dtype)

    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        logging_dir=logging_dir,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_names=["future_values"],
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=0.0,
            )
        ],
    )
    trainer.train()
    return {"model": model, "trainer": trainer}


def avaliar_patchtst(
    model,
    test_ds,
    tsp,
    test_df,
    timestamp_col: str,
    target_column,
    context_length: int,
    model_name: Optional[str] = None,
):
    return evaluate_and_visualize(
        model=model,
        test_dataset=test_ds,
        tsp=tsp,
        test_df=test_df,
        timestamp_col=timestamp_col,
        target_column=target_column,
        context_length=context_length,
        model_name=model_name or "PatchTST",
    )

