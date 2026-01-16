from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tsfm_public.toolkit.util import select_by_index
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.dataset import ForecastDFDataset

from config import BTC_DATA_FILE

# Constantes
WINDOW_WEEKLY = 5
WINDOW_MONTHLY = 22

class RepositorioDados:
    def executar(self, timestamp_col, train_frac, valid_frac, context_length, features, target, id_cols, forecast_horizon) -> None:
        """
        Orquestra o pipeline completo de preparação de dados.
        """
        try:
            df = self.load_and_prepare_data(BTC_DATA_FILE, timestamp_col)
            df_vol = self.calculate_features(df)

            train_df, valid_df, test_df = self.train_valid_test_split(
                df_vol,
                timestamp_col=timestamp_col,
                train_frac=train_frac,
                valid_frac=valid_frac,
                context_length=context_length
            )

            tsp, train_ds, valid_ds, test_ds = self.normalize_data(
                timestamp_col=timestamp_col,
                target_col=target,
                feature_cols=features,
                id_cols=id_cols,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
                train_df=train_df,
                valid_df=valid_df,
                test_df=test_df
            )

            return tsp, train_ds, valid_ds, test_ds
            
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
        df = pd.read_csv(file_path, sep=';')
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms', errors='raise')
        
        return df


    def calculate_features(self, df: pd.DataFrame, usa_features: bool = True, timestamp_col='timestamp') -> pd.DataFrame:
        """
        Calcula features de volatilidade para o modelo.
        
        Args:
            df: DataFrame com dados OHLC
            
        Returns:
            DataFrame com features de volatilidade
        """
        print("Calculando features de volatilidade")
        
        # Calcula retornos logarítmicos
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['squared_log_returns'] = df['log_returns'] ** 2
        
        # Agrupa por dia
        df_indexed = df.set_index(timestamp_col)
        daily_realized_variance = df_indexed.groupby(df_indexed.index.date)['squared_log_returns'].sum()
        
        # Cria features de volatilidade
        df_vol = daily_realized_variance.to_frame(name='Vol')

        if usa_features:
            df_vol['Vol_lag_1'] = df_vol['Vol'].shift(1)

            df_vol['Vol_week_mean'] = df_vol['Vol'].rolling(
                window=WINDOW_WEEKLY, 
                min_periods=1
            ).mean().shift(1)

            df_vol['Vol_month_mean'] = df_vol['Vol'].rolling(
                window=WINDOW_MONTHLY, 
                min_periods=1
            ).mean().shift(1)
        
        df_vol = df_vol.dropna()
        df_vol.index = pd.to_datetime(df_vol.index)
        df_vol.reset_index(inplace=True)
        df_vol.rename(columns={'index': timestamp_col}, inplace=True)

        return df_vol

    def train_valid_test_split(self, dados, timestamp_col, train_frac, valid_frac, context_length):
        dados.sort_values(by=timestamp_col, inplace=True)

        n = len(dados)
        train_end = int(n * train_frac)
        valid_end = int(n * (train_frac + valid_frac))

        train_df = select_by_index(dados, start_index=None, end_index=train_end)
        valid_df = select_by_index(dados, start_index=train_end - context_length, end_index=valid_end)
        test_df = select_by_index(dados, start_index=valid_end - context_length, end_index=None)

        return train_df, valid_df, test_df

    def normalize_data(self, timestamp_col, target_col, feature_cols, id_cols, context_length, forecast_horizon,
                    train_df, valid_df, test_df):
        tsp = TimeSeriesPreprocessor(
            timestamp_column=timestamp_col,
            target_column=target_col,
            feature_columns=feature_cols,
            id_columns=id_cols,
            scaling='std'  # standardscaler: (x - mean)/std
        )
        tsp.train(train_df)  # Aprende mean/std no treino

        def make_ds(df):
            """Cria dataset com janelas deslizantes"""
            return ForecastDFDataset(
                tsp.preprocess(df),  # Normaliza usando parâmetros do treino
                id_columns=id_cols,
                target_columns=target_col,
                context_length=context_length,      # Quantos dias de histórico usar
                prediction_length=forecast_horizon, # Quantos dias prever
            )

        train_ds, valid_ds, test_ds = map(make_ds, [train_df, valid_df, test_df])
        print(f"Treino: {len(train_ds)} amostras | Val: {len(valid_ds)} | Teste: {len(test_ds)}")

        return tsp, train_ds, valid_ds, test_ds

    def resumo_periodo(
        self,
        df,
        date_col="timestamp",
        nome="Dataset"
    ):
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
            "fim": fim
        }

        return infos


    def save_datasets(
        self,
        *kwargs
    ) -> None:
        """
        Salva datasets em arquivos CSV.
        
        Args:
            X_train, X_test, y_train, y_test: Dados de treino/teste
        """
        print("Salvando datasets")

        for data, (path, idx) in kwargs:
            data.to_csv(path, index=idx)
        
        print("Datasets salvos com sucesso")

if __name__ == "__main__":
    rep = RepositorioDados()
    rep.executar()