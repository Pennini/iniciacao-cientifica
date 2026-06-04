# Copilot Instructions for `iniciacao-cientifica`

## Build, test, and lint commands

This repository is currently **not** set up with an automated build/lint/test pipeline (no `pytest`, `ruff`, `flake8`, `pylint`, `mypy`, `tox`, `nox`, or CI workflow files found).

Use these commands as the practical baseline:

```bash
pip install -r requirements.txt
python src\config.py
python src\models\har.py
```

Single-test command: **not available yet** (no automated test suite is configured in the repository).

## High-level architecture

The project is organized as a volatility forecasting workflow with two model families (HAR and PatchTST), where notebooks orchestrate experiments and `src/` provides reusable pieces:

1. `src\config.py` is the central contract for paths, dataset splits, feature names, and model hyperparameters.
2. `src\dataset.py` (`RepositorioDados`) builds the data pipeline:
   - reads raw BTC 5m data (`data\raw\BTCUSDT_5m.txt`, `;`-separated),
   - computes realized variance (`Vol`) plus lag/rolling features,
   - performs chronological train/validation/test splitting with context overlap,
   - normalizes with `TimeSeriesPreprocessor`,
   - creates `ForecastDFDataset` windows for transformer training/evaluation.
3. `src\models\har.py` trains/evaluates a linear HAR baseline (`LinearRegression`) and saves predictions/metrics to `data\processed\`.
4. `src\models\transformer.py` provides PatchTST metric/evaluation helpers around Hugging Face `Trainer` (typically called from notebooks).
5. `notebooks\` are the main experiment entrypoints; they add `..\src` to `sys.path`, call `RepositorioDados`, run PatchTST experiments, and compare against HAR.

## Key codebase conventions

- **Notebook-first orchestration**: end-to-end training/evaluation logic is mostly driven from notebooks; `src/` modules are helper building blocks.
- **Centralized constants in `config.py`**: paths, feature columns, split ratios, context/horizon, and output filenames should be consumed from `config.py` rather than duplicated.
- **Strict feature-name coupling**: HAR expects columns exactly matching `FEATURES` (`Vol_lag_1`, `Vol_week_mean`, `Vol_month_mean`) when loading CSVs.
- **Time-series split behavior**: validation/test sets intentionally start at `train_end - context_length` / `valid_end - context_length` to preserve context windows for forecasting.
- **Portuguese domain naming**: classes/methods and logs use Portuguese (`RepositorioDados`, `executar`, `carregar_dados`, etc.); keep naming/style consistent in new code.
- **Data artifact layout**: raw/interim/processed outputs follow `data\raw`, `data\interim`, `data\processed`; model artifacts go in `models\`, runtime outputs in `logs\`.
