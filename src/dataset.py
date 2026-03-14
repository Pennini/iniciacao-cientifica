from pathlib import Path

import pandas as pd
import numpy as np
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.dataset import ForecastDFDataset

from config import BTC_DATA_FILE

# Constantes
WINDOW_WEEKLY = 7
WINDOW_MONTHLY = 30


class RepositorioDados:
    def executar(
        self,
        timestamp_col,
        train_frac,
        valid_frac,
        context_length,
        target_col,
        id_columns,
        forecast_horizon,
        use_mean_features=True,
        lags=1,
        file_path: str = BTC_DATA_FILE,
    ):
        """Orquestra o pipeline completo de preparação de dados."""
        try:
            self._validate_fracs(train_frac, valid_frac)

            df = self.load_and_prepare_data(file_path, timestamp_col)
            data, features = self.build_features(
                df,
                timestamp_col,
                use_mean_features,
                lags,
            )

            train_df, valid_df, test_df, tsp = self.split_and_preprocess(
                data,
                timestamp_col,
                target_col,
                id_columns,
                features,
                train_frac,
                valid_frac,
                context_length,
            )

            train_ds = self.make_dataset(
                train_df,
                tsp,
                id_columns,
                timestamp_col,
                target_col,
                context_length,
                forecast_horizon,
            )
            valid_ds = self.make_dataset(
                valid_df,
                tsp,
                id_columns,
                timestamp_col,
                target_col,
                context_length,
                forecast_horizon,
            )
            test_ds = self.make_dataset(
                test_df,
                tsp,
                id_columns,
                timestamp_col,
                target_col,
                context_length,
                forecast_horizon,
            )

            return tsp, train_ds, valid_ds, test_ds, train_df, valid_df, test_df

        except Exception as e:
            print(f"Erro ao processar dados: {e}")
            raise

    def load_and_prepare_data(self, file_path: str, timestamp_col) -> pd.DataFrame:
        """
        Carrega e realiza limpeza básica nos dados.

        Args:
            file_path: Caminho do arquivo CSV

        Returns:
            DataFrame com dados preparados

        Raises:
            FileNotFoundError: Se arquivo não existir
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        print(f"Carregando dados de {file_path}")
        df = pd.read_csv(file_path, sep=";")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit="ms", errors="raise")

        return df

    def calculate_features(
        self,
        df: pd.DataFrame,
        usa_features: bool = True,
        timestamp_col="timestamp",
        use_mean_features: bool = True,
        lags: int = 1,
    ) -> pd.DataFrame:
        """
        Calcula features de volatilidade para o modelo.

        Args:
            df: DataFrame com dados OHLC

        Returns:
            DataFrame com features de volatilidade
        """
        print("Calculando features de volatilidade")

        # Calcula retornos logarítmicos
        if "close" not in df.columns:
            raise ValueError("Coluna 'close' não encontrada no DataFrame de entrada")

        df = df.sort_values(timestamp_col).copy()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["squared_log_returns"] = df["log_returns"] ** 2

        # Agrupa por dia
        df_indexed = df.set_index(timestamp_col)
        daily_realized_variance = df_indexed.groupby(df_indexed.index.date)[
            "squared_log_returns"
        ].sum()

        # Cria features de volatilidade
        df_vol = daily_realized_variance.to_frame(name="Vol")

        for lag in range(1, lags + 1):
            df_vol[f"Vol_lag_{lag}"] = df_vol["Vol"].shift(lag)

        if usa_features:
            if use_mean_features:
                df_vol["Vol_week_mean"] = (
                    df_vol["Vol"]
                    .rolling(window=WINDOW_WEEKLY, min_periods=1)
                    .mean()
                    .shift(1)
                )

                df_vol["Vol_month_mean"] = (
                    df_vol["Vol"]
                    .rolling(window=WINDOW_MONTHLY, min_periods=1)
                    .mean()
                    .shift(1)
                )

        df_vol = df_vol.dropna()
        df_vol.index = pd.to_datetime(df_vol.index)
        df_vol.reset_index(inplace=True)
        df_vol.rename(columns={"index": timestamp_col}, inplace=True)

        return df_vol

    def build_features(self, df, timestamp_col, use_mean_features, lags):

        dados = self.calculate_features(
            df,
            usa_features=use_mean_features,
            lags=lags,
            timestamp_col=timestamp_col,
        )

        features = [c for c in dados.columns if c.startswith("Vol_lag_")]

        if use_mean_features:
            for c in ["Vol_week_mean", "Vol_month_mean"]:
                if c in dados.columns:
                    features.append(c)

        return dados, features

    def split_and_preprocess(
        self,
        data,
        timestamp_col,
        target_col,
        id_columns,
        features,
        train_frac,
        valid_frac,
        context_length,
    ):

        data = data.sort_values(timestamp_col)

        n = len(data)
        train_end = int(n * train_frac)
        valid_end = int(n * (train_frac + valid_frac))

        if train_end <= context_length:
            raise ValueError(
                "Context_length maior ou igual ao tamanho do treino; reduza o contexto ou aumente o train_frac"
            )

        if valid_end <= train_end:
            raise ValueError(
                "valid_frac muito pequeno; aumente a validação ou ajuste as frações"
            )

        train_df = data.iloc[:train_end]
        valid_df = data.iloc[max(train_end - context_length, 0) : valid_end]
        test_df = data.iloc[max(valid_end - context_length, 0) :]

        tsp = TimeSeriesPreprocessor(
            timestamp_column=timestamp_col,
            target_column=target_col,
            feature_columns=features,
            id_columns=id_columns,
            scaling="std",
        )

        tsp.train(train_df)

        return train_df, valid_df, test_df, tsp

    def make_dataset(
        self,
        df,
        tsp,
        id_columns,
        timestamp_col,
        target_col,
        context_length,
        horizon,
    ):

        return ForecastDFDataset(
            tsp.preprocess(df),
            id_columns=id_columns,
            timestamp_column=timestamp_col,
            target_columns=target_col,
            context_length=context_length,
            prediction_length=horizon,
        )

    def resumo_periodo(self, df, date_col="timestamp", nome="Dataset"):
        """
        Exibe um resumo claro do período temporal de um DataFrame.

        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame contendo a coluna de datas.
        date_col : str
            Nome da coluna de datas.
        nome : str
            Nome do conjunto (ex: 'Treino', 'Validação', 'Teste').

        Retorno
        -------
        dict com informações do período (útil para logs)
        """
        try:
            datas = pd.to_datetime(df[date_col])
        except KeyError:
            datas = pd.to_datetime(df.index)

        inicio = datas.min()
        fim = datas.max()

        delta = fim - inicio
        anos = delta.days // 365
        dias_restantes = delta.days % 365

        print(
            f"📅 {nome}\n"
            f"   Início : {inicio:%d/%m/%Y}\n"
            f"   Fim    : {fim:%d/%m/%Y}\n"
            f"   Duração: {anos} anos e {dias_restantes} dias "
            f"({delta.days} dias no total)\n"
        )

        infos = {
            "anos": anos,
            "dias_restantes": dias_restantes,
            "total_dias": delta.days,
            "inicio": inicio,
            "fim": fim,
        }

        return infos

    def save_datasets(self, *kwargs) -> None:
        """Salva datasets em arquivos CSV."""
        print("Salvando datasets")

        for data, (path, idx) in kwargs:
            data.to_csv(path, index=idx)

        print("Datasets salvos com sucesso")

    @staticmethod
    def _validate_fracs(train_frac: float, valid_frac: float) -> None:
        if not (0 < train_frac < 1):
            raise ValueError("train_frac deve estar entre 0 e 1")
        if not (0 <= valid_frac < 1):
            raise ValueError("valid_frac deve estar entre 0 e 1")
        if train_frac + valid_frac >= 1:
            raise ValueError("A soma de train_frac e valid_frac deve ser menor que 1")


if __name__ == "__main__":
    rep = RepositorioDados()
    rep.executar()
