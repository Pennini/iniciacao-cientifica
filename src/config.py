from pathlib import Path


PROJ_ROOT = Path(__file__).parent.parent.resolve()

DATA_DIR = PROJ_ROOT / "data"
MODEL_DIR = PROJ_ROOT / "models"
PATCHTST_DIR = MODEL_DIR / "patchtst"
LOGS_DIR = PROJ_ROOT / "logs"
SRC_DIR = PROJ_ROOT / "src"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

BTC_DATA_FILE = RAW_DATA_DIR / "BTCUSDT_5m.txt"
X_TRAIN_FILE = INTERIM_DATA_DIR / "X_train.csv"
X_VAL_FILE = INTERIM_DATA_DIR / "X_val.csv"
X_TEST_FILE = INTERIM_DATA_DIR / "X_test.csv"
Y_TRAIN_FILE = INTERIM_DATA_DIR / "y_train.csv"
Y_VAL_FILE = INTERIM_DATA_DIR / "y_val.csv"
Y_TEST_FILE = INTERIM_DATA_DIR / "y_test.csv"
DF_VOL_FILE = INTERIM_DATA_DIR / "df_vol.csv"

MODELS_SCR_DIR = SRC_DIR / "models"
HAR_MODELING_FILE = MODELS_SCR_DIR / "har.py"
PATCHTST_MODELING_FILE = MODELS_SCR_DIR / "transformer.py"
HAR_MODEL_FILE = MODEL_DIR / "har_model.pkl"
PATCHTST_MODEL_FILE = MODEL_DIR / "patchtst_model.pth"

HAR_PRED_FILE = PROCESSED_DATA_DIR / "har_predictions.csv"
PATCHTST_PRED_FILE = PROCESSED_DATA_DIR / "patchtst_predictions.csv"

HAR_METRICS_FILE = PROCESSED_DATA_DIR / "har_metrics.csv"
PATCHTST_METRICS_FILE = PROCESSED_DATA_DIR / "patchtst_metrics.csv"

TIMESTAMP_COLUMN = 'timestamp'
TARGET_COLUMN = ['Vol']
FEATURES = ["Vol_lag_1", "Vol_week_mean", "Vol_month_mean"]
ID_COLUMNS = []

CONTEXT_LENGTH_IBM = 512
FORECAST_HORIZON_IBM = 96

# PatchTST "Scratch" (notebook "8. Ajuste Scratch") — valores que reproduzem o gráfico do notebook
CONTEXT_LENGTH = 256
FORECAST_HORIZON = 1
D_MODEL = 64
NUM_ATTENTION_HEADS = 16
NUM_HIDDEN_LAYERS = 2
FFN_DIM = 128
DROPOUT = 0.05
SCALING = "std"  # default do PatchTSTConfig; o notebook NÃO passa scaling, então cai no "std"

TRAIN_FRAC, VALID_FRAC = 0.7, 0.1

# Hyperparâmetros do modelo
PATCH_LENGTH = 1
BATCH_SIZE = 32
NUM_WORKERS = 0
EPOCHS = 50
LEARNING_RATE = 5e-4

def ensure_directories():
    for dir in [
        DATA_DIR,
        MODEL_DIR,
        PATCHTST_DIR,
        LOGS_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
    ]:
        dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()