import typer

from .pipeline import comparar_modelos, preparar_dados, treinar_e_avaliar_patchtst, treinar_har

app = typer.Typer(add_completion=False, help="CLI do pipeline de volatilidade (HAR e PatchTST).")


@app.command("prepare")
def cmd_prepare(
    context_length: int = 512,
    forecast_horizon: int = 96,
    use_mean_features: bool = True,
    lags: int = 3,
):
    preparar_dados(
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        use_mean_features=use_mean_features,
        lags=lags,
    )
    typer.echo("Dados preparados com sucesso.")


@app.command("train-har")
def cmd_train_har():
    metricas = treinar_har()
    typer.echo(f"HAR treinado. RMSE={metricas['RMSE']:.6f} | MAE={metricas['MAE']:.6f}")


@app.command("train-patchtst")
def cmd_train_patchtst(
    use_mean_features: bool = True,
    lags: int = 3,
    context_length: int = 512,
    forecast_horizon: int = 96,
):
    metricas = treinar_e_avaliar_patchtst(
        use_mean_features=use_mean_features,
        lags=lags,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
    )
    typer.echo(f"PatchTST treinado. RMSE={metricas['RMSE']:.6f} | MAE={metricas['MAE']:.6f}")


@app.command("compare")
def cmd_compare():
    comparativo = comparar_modelos()
    typer.echo(comparativo.to_string(index=False))


if __name__ == "__main__":
    app()

