from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    BTC_DATA_FILE,
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
)

# Constantes
WINDOW_WEEKLY = 5
WINDOW_MONTHLY = 22
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Feautures para o modelo
FEATURES = ['Vol_lag_1', 'Vol_week_mean', 'Vol_month_mean']


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
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
    df['tempo'] = pd.to_datetime(df['tempo'], format='mixed')
    
    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df_indexed = df.set_index('tempo')
    daily_realized_variance = df_indexed.groupby(df_indexed.index.date)['squared_log_returns'].sum()
    
    # Cria features de volatilidade
    df_vol = pd.DataFrame({'Vol': daily_realized_variance})
    df_vol['Vol_lag_1'] = df_vol['Vol'].shift(1)
    df_vol['Vol_week_mean'] = df_vol['Vol'].rolling(
        window=WINDOW_WEEKLY, 
        min_periods=1
    ).mean().shift(1)
    df_vol['Vol_month_mean'] = df_vol['Vol'].rolling(
        window=WINDOW_MONTHLY, 
        min_periods=1
    ).mean().shift(1)
    
    return df_vol


def prepare_train_test_split(df_vol: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Realiza split treino/teste com limpeza de dados.
    
    Args:
        df_vol: DataFrame com features
        
    Returns:
        Tupla com (X_train, X_test, y_train, y_test)
    """
    print("Preparando split treino/teste")
    
    df_clean = df_vol.dropna().copy()
    print(f"Removidas {len(df_vol) - len(df_clean)} linhas com valores faltantes")
    
    X = df_clean[FEATURES]
    y = df_clean['Vol']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def save_datasets(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series
) -> None:
    """
    Salva datasets em arquivos CSV.
    
    Args:
        X_train, X_test, y_train, y_test: Dados de treino/teste
    """
    print("Salvando datasets")

    X_train.index.name = 'tempo'
    X_test.index.name = 'tempo'

    X_train.to_csv(X_TRAIN_FILE, index=True)
    X_test.to_csv(X_TEST_FILE, index=True)
    y_train.to_csv(Y_TRAIN_FILE, index=False)
    y_test.to_csv(Y_TEST_FILE, index=False)
    
    print("Datasets salvos com sucesso")


def train_test_split_data() -> None:
    """
    Orquestra o pipeline completo de preparação de dados.
    """
    try:
        df = load_and_prepare_data(BTC_DATA_FILE)
        df_vol = calculate_features(df)
        X_train, X_test, y_train, y_test = prepare_train_test_split(df_vol)
        save_datasets(X_train, X_test, y_train, y_test)
        
        print("Pipeline concluído com sucesso")
        
    except Exception as e:
        print(f"Erro ao processar dados: {e}")
        raise


if __name__ == "__main__":
    train_test_split_data()