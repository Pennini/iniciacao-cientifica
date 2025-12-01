import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import PatchTSTConfig, PatchTSTForPrediction
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import sys

# Seed para reproducibilidade
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
)

from config.config import RESULTS_DATA_DIR

# ==================== 1. Carregar dados ====================
print("Carregando dados...")
X_train = pd.read_csv(str(X_TRAIN_FILE)).values.tolist()
X_test = pd.read_csv(str(X_TEST_FILE)).values.tolist()
y_train = pd.read_csv(str(Y_TRAIN_FILE)).values.flatten().tolist()
y_test = pd.read_csv(str(Y_TEST_FILE)).values.flatten().tolist()

# Converter para tensores PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# ==================== 2. Preparar sequências ====================
def create_sequences(X, y, window_size=10):
    """
    Cria sequências temporais para o modelo PatchTST

    Args:
        X: Features (n_samples, n_features)
        y: Target (n_samples,)
        window_size: Tamanho da janela de contexto

    Returns:
        X_seq: Lista de tensores (n_sequences, window_size, n_features)
        y_seq: Tensores de targets (n_sequences,)
    """
    X_seq, y_seq = [], []

    for i in range(len(X) - window_size):
        X_seq.append(torch.tensor(X[i : i + window_size], dtype=torch.float32))
        y_seq.append(torch.tensor(y[i + window_size], dtype=torch.float32))

    return torch.stack(X_seq), torch.stack(y_seq)


WINDOW_SIZE = 10
X_train_tensor, y_train_tensor = create_sequences(X_train, y_train, WINDOW_SIZE)
X_test_tensor, y_test_tensor = create_sequences(X_test, y_test, WINDOW_SIZE)

print(f"\nX_train_tensor shape: {X_train_tensor.shape}")
print(f"y_train_tensor shape: {y_train_tensor.shape}")
print(f"X_test_tensor shape: {X_test_tensor.shape}")
print(f"y_test_tensor shape: {y_test_tensor.shape}")

# ==================== 3. Preparar targets com dimensão correta ====================
y_train_tensor = y_train_tensor.unsqueeze(1)
y_test_tensor = y_test_tensor.unsqueeze(1)

# ==================== 4. Criar DataLoaders ====================
BATCH_SIZE = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(RANDOM_STATE),
)
BATCH_SIZE = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==================== 5. Configuração do modelo ====================
num_features = X_train_tensor.shape[2]  # 3 features

config = PatchTSTConfig(
    prediction_length=1,  # Prever 1 passo à frente
    context_length=WINDOW_SIZE,  # Usar histórico de WINDOW_SIZE
    num_input_channels=num_features,  # 3 features (Vol_lag_1, Vol_week_mean, Vol_month_mean)
    patch_length=2,  # Tamanho do patch
    d_model=128,  # Dimensão do modelo
    num_heads=4,  # Número de cabeças de atenção
    dropout=0.1,  # Dropout para regularização
)

model = PatchTSTForPrediction(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"\nModelo: {model.__class__.__name__}")
print(f"Device: {device}")
print(f"Total de parâmetros: {sum(p.numel() for p in model.parameters()):,}")

# Garantir diretórios necessários
ensure_directories()
# Diretório para salvar/carregar modelos em data/models
data_models_dir = RESULTS_DATA_DIR.parent / "models"
data_models_dir.mkdir(parents=True, exist_ok=True)
saved_model_path = data_models_dir / "patch_model.pth"

if saved_model_path.exists():
    print(
        f"Modelo pré-treinado encontrado em {saved_model_path}, carregando e pulando treinamento."
    )
    model.load_state_dict(torch.load(str(saved_model_path), map_location=device))
    trained = True
else:
    trained = False

# ==================== 6. Função de perda e otimizador ====================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==================== 7. Treinamento ====================
EPOCHS = 10
best_val_loss = float("inf")

print(f"\nIniciando treinamento por {EPOCHS} épocas...")

if not trained:
    for epoch in range(EPOCHS):
        # Treino
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(past_values=batch_X)
            loss = criterion(outputs.prediction_outputs, batch_y.unsqueeze(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

    # Validação
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(past_values=batch_X)
            loss = criterion(outputs.prediction_outputs, batch_y.unsqueeze(-1))
            val_loss += loss.item()

    val_loss /= len(test_loader)

    print(
        f"Época {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
    )

    # Salvar melhor modelo
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_path = data_models_dir / "patch_model.pth"
        torch.save(model.state_dict(), str(model_path))
else:
    print("Pulando treinamento — modelo carregado do disco.")

# ==================== 8. Avaliação Final ====================
print("\n" + "=" * 50)
print("AVALIAÇÃO FINAL")
print("=" * 50)

model.eval()
y_pred_train = []
y_true_train = []

with torch.no_grad():
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        outputs = model(past_values=batch_X)
        # Extrair scalars de tensores (sem usar NumPy)
        preds_cpu = outputs.prediction_outputs.cpu()
        for p in preds_cpu:
            arr = p.squeeze()
            # Se arr tiver múltiplos elementos (vários canais), escolher o primeiro canal (Vol)
            if arr.numel() > 1:
                val = float(arr.reshape(-1)[0].item())
            else:
                val = float(arr.item())
            y_pred_train.append(val)

        for by in batch_y.cpu():
            y_true_train.append(float(by.squeeze().item()))

y_pred_test = []
y_true_test = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(past_values=batch_X)
        preds_cpu = outputs.prediction_outputs.cpu()
        for p in preds_cpu:
            arr = p.squeeze()
            if arr.numel() > 1:
                val = float(arr.reshape(-1)[0].item())
            else:
                val = float(arr.item())
            y_pred_test.append(val)

        for by in batch_y.cpu():
            y_true_test.append(float(by.squeeze().item()))


# Calcular métricas manualmente (sem NumPy)
def mse(y_true, y_pred):
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)


def mae(y_true, y_pred):
    return sum(abs(y_true[i] - y_pred[i]) for i in range(len(y_true))) / len(y_true)


def r2(y_true, y_pred):
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    ss_tot = sum(
        (y_true[i] - sum(y_true) / len(y_true)) ** 2 for i in range(len(y_true))
    )
    return 1 - (ss_res / ss_tot)


train_mse_val = mse(y_true_train, y_pred_train)
train_mae_val = mae(y_true_train, y_pred_train)
train_r2_val = r2(y_true_train, y_pred_train)

test_mse_val = mse(y_true_test, y_pred_test)
test_mae_val = mae(y_true_test, y_pred_test)
test_r2_val = r2(y_true_test, y_pred_test)

print(f"\nTREINAMENTO:")
print(f"  MSE:  {train_mse_val:.6e}")
print(f"  MAE:  {train_mae_val:.6e}")
print(f"  R²:   {train_r2_val:.6e}")

print(f"\nTESTES:")
print(f"  MSE:  {test_mse_val:.6e}")
print(f"  MAE:  {test_mae_val:.6e}")
print(f"  R²:   {test_r2_val:.6e}")

# ==================== 9. Salvar previsões no DataFrame de teste ====================

# Cria DataFrame com valores reais e previstos
df_test_results = pd.DataFrame(
    {
        "y_true": y_true_test,
        "y_pred": y_pred_test,
    }
)

pred_file = RESULTS_DATA_DIR / "predictions_test.csv"
df_test_results.to_csv(str(pred_file), index=False)

print(f"Previsões salvas em: {pred_file}")
