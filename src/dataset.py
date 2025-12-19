from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    BTC_DATA_FILE,
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
    DF_VOL_FILE,
    FEATURES
)

# Constantes
WINDOW_WEEKLY = 5
WINDOW_MONTHLY = 22
TEST_SIZE = 0.3
RANDOM_STATE = 42


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
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    
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
    df_indexed = df.set_index('timestamp')
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

def normalize_features(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Normaliza features.
    Args:
        df_vol: DataFrame com features
    Returns:
        DataFrame com features normalizadas
    """
    print("Normalizando features")
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    
    return pd.DataFrame(X_train_scaled, columns=FEATURES, index=X_train.index), pd.DataFrame(X_test_scaled, columns=FEATURES, index=X_test.index), pd.Series(y_train_scaled), pd.Series(y_test_scaled)

def prepare_train_test_split(df_vol: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Realiza split treino/teste com limpeza de dados.
    
    Args:
        df_vol: DataFrame com features
        
    Returns:
        Tupla com (X_train, X_test, y_train, y_test)
    """
    print("Preparando split treino/teste")
    
    df_clean = df_vol.copy()
    
    X = df_clean[FEATURES]
    y = df_clean['Vol']

    print(f"X: {len(X)} registros antes da limpeza")
    print(f"y: {len(y)} registros antes da limpeza")

    print(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )

    X_train.index.name = 'timestamp'
    X_test.index.name = 'timestamp'

    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def save_datasets(
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


def train_test_split_data() -> None:
    """
    Orquestra o pipeline completo de preparação de dados.
    """
    try:
        df = load_and_prepare_data(BTC_DATA_FILE)
        df_vol = calculate_features(df)
        X_train, X_test, y_train, y_test = prepare_train_test_split(df_vol)

        datasets = [
            (X_train, (X_TRAIN_FILE, True)),
            (X_test, (X_TEST_FILE, True)),
            (y_train, (Y_TRAIN_FILE, False)),
            (y_test, (Y_TEST_FILE, False)),
            (df_vol, (DF_VOL_FILE, True))
        ]

        save_datasets(*datasets)
        
        print("Pipeline concluído com sucesso")
        
    except Exception as e:
        print(f"Erro ao processar dados: {e}")
        raise


if __name__ == "__main__":
    train_test_split_data()