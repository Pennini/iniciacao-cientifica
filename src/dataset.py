from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from config import (
    BTC_DATA_FILE,
    X_TRAIN_FILE,
    X_VAL_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_VAL_FILE,
    Y_TEST_FILE,
    DF_VOL_FILE,
    FEATURES
)

# Constantes
WINDOW_WEEKLY = 5
WINDOW_MONTHLY = 22
TRAIN_SIZE = 0.7
VALID_SIZE = 0.1

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
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='raise')
    
    return df


def calculate_features(df: pd.DataFrame, usa_features: bool = True) -> pd.DataFrame:
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
    df_vol.index.name = 'date'
    df_vol.index = pd.to_datetime(df_vol.index)

    return df_vol

def temporal_train_test_split(
    df,
    date_col,
    features,
    target,
    train_size=TRAIN_SIZE,
    val_size=VALID_SIZE,
    use_val=True
):
    """
    Split temporal por proporção.
    """

    df = df.sort_values(date_col)

    n = len(df)
    train_end = int(n * train_size)

    if use_val:
        val_end = int(n * (train_size + val_size))

        train = df.iloc[:train_end]
        val   = df.iloc[train_end:val_end]
        test  = df.iloc[val_end:]

        X_train, y_train = train[features], train[target]
        X_val, y_val     = val[features], val[target]
        X_test, y_test   = test[features], test[target]

        return X_train, X_val, X_test, y_train, y_val, y_test

    else:
        train = df.iloc[:train_end]
        test  = df.iloc[train_end:]

        X_train, y_train = train[features], train[target]
        X_test, y_test   = test[features], test[target]

        return X_train, X_test, y_train, y_test


def temporal_split_by_date(
    df,
    date_col,
    features,
    target,
    train_end,
    val_end=None
):
    """
    Split temporal por data.
    """

    df = df.sort_values(date_col)

    if val_end is not None:
        train = df[df[date_col] <= train_end]
        val   = df[(df[date_col] > train_end) & (df[date_col] <= val_end)]
        test  = df[df[date_col] > val_end]

        X_train, y_train = train[features], train[target]
        X_val, y_val     = val[features], val[target]
        X_test, y_test   = test[features], test[target]

        return X_train, X_val, X_test, y_train, y_val, y_test

    else:
        train = df[df[date_col] <= train_end]
        test  = df[df[date_col] > train_end]

        X_train, y_train = train[features], train[target]
        X_test, y_test   = test[features], test[target]

        return X_train, X_test, y_train, y_test

def normalize_time_series(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    X_test=None,
    y_test=None,
    method="standard",
    normalize_y=False
):
    """
    Normalização segura para séries temporais (fit apenas no treino).
    """

    # Escolha do scaler
    if method == "standard":
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
    elif method == "minmax":
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
    elif method == "robust":
        scaler_X = RobustScaler()
        scaler_y = RobustScaler()
    else:
        raise ValueError("method deve ser: 'standard', 'minmax' ou 'robust'")

    # Fit SOMENTE no treino
    scaler_X.fit(X_train)

    X_train_scaled = scaler_X.transform(X_train)
    X_val_scaled   = scaler_X.transform(X_val) if X_val is not None else None
    X_test_scaled  = scaler_X.transform(X_test) if X_test is not None else None

    if normalize_y:
        y_train = y_train.values.reshape(-1, 1)
        scaler_y.fit(y_train)

        y_train_scaled = scaler_y.transform(y_train).ravel()
        y_val_scaled   = (
            scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
            if y_val is not None else None
        )
        y_test_scaled  = (
            scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
            if y_test is not None else None
        )
    else:
        scaler_y = None
        y_train_scaled = y_train
        y_val_scaled   = y_val
        y_test_scaled  = y_test

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train_scaled,
        "y_val": y_val_scaled,
        "y_test": y_test_scaled,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y
    }


def resumo_periodo(
    df,
    date_col="date",
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


def prepare_data_features(use_val=True) -> None:
    """
    Orquestra o pipeline completo de preparação de dados.
    """
    try:
        df = load_and_prepare_data(BTC_DATA_FILE)
        df_vol = calculate_features(df)
        
        if use_val:
            X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_test_split(
                df_vol,
                date_col='date',
                features=FEATURES,
                target='Vol',
                train_size=TRAIN_SIZE,
                val_size=VALID_SIZE,
                use_val=True
            )
        else:
            X_train, X_test, y_train, y_test = temporal_train_test_split(
                df_vol,
                date_col='date',
                features=FEATURES,
                target='Vol',
                train_size=TRAIN_SIZE,
                use_val=False
            )

        datasets = [
            (X_train, (X_TRAIN_FILE, True)),
            (X_test, (X_TEST_FILE, True)),
            (y_train, (Y_TRAIN_FILE, False)),
            (y_test, (Y_TEST_FILE, False)),
            (df_vol, (DF_VOL_FILE, True))
        ]

        if use_val:
            datasets += [(X_val, (X_VAL_FILE, True)), (y_val, (Y_VAL_FILE, False))]

        for data, (nome, _) in datasets:
            nome_arrumado = nome.name
            if "X" in nome_arrumado:
                resumo_periodo(data, nome=nome_arrumado)

        save_datasets(*datasets)
        
        print("Pipeline concluído com sucesso")
        
    except Exception as e:
        print(f"Erro ao processar dados: {e}")
        raise

def prepare_data_vol_diaria(use_val=True):
    df = load_and_prepare_data(BTC_DATA_FILE)
    df_vol = calculate_features(df, usa_features=False)

    if use_val:
        X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_test_split(
            df_vol,
            date_col='date',
            features=['Vol'],
            target='Vol',
            train_size=TRAIN_SIZE,
            val_size=VALID_SIZE,
            use_val=True
        )
    else:
        X_train, X_test, y_train, y_test = temporal_train_test_split(
            df_vol,
            date_col='date',
            features=['Vol'],
            target='Vol',
            train_size=TRAIN_SIZE,
            use_val=False
        )
    
    datasets = [
        (X_train, (X_TRAIN_FILE, True)),
        (X_test, (X_TEST_FILE, True)),
        (y_train, (Y_TRAIN_FILE, False)),
        (y_test, (Y_TEST_FILE, False)),
        (df_vol, (DF_VOL_FILE, True))
    ]

    if use_val:
        datasets += [(X_val, (X_VAL_FILE, True)), (y_val, (Y_VAL_FILE, False))]
    
    for idx in range(len(datasets)):
        data, (nome, tem) = datasets[idx]
        nome_arrumado = nome.name.split(".")[0] + "_vol_diaria"
        datasets[idx] = (data, (Path(nome.parent, nome_arrumado), tem))
        if "X" in nome_arrumado:
            resumo_periodo(data, nome=nome_arrumado)
    
    save_datasets(*datasets)

    print("Pipeline de dados apenas com volatilidade diária concluído com sucesso")

if __name__ == "__main__":
    prepare_data_features()
    prepare_data_vol_diaria()