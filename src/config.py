from pathlib import Path


PROJ_ROOT = Path(__file__).parent.parent.resolve()

DATA_DIR = PROJ_ROOT / "data"
MODEL_DIR = PROJ_ROOT / "models"
LOGS_DIR = PROJ_ROOT / "logs"
SRC_DIR = PROJ_ROOT / "src"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

BTC_DATA_FILE = RAW_DATA_DIR / "BTCUSDT_5m.txt"
X_TRAIN_FILE = INTERIM_DATA_DIR / "X_train.csv"
X_TEST_FILE = INTERIM_DATA_DIR / "X_test.csv"
Y_TRAIN_FILE = INTERIM_DATA_DIR / "y_train.csv"
Y_TEST_FILE = INTERIM_DATA_DIR / "y_test.csv"

MODELS_SCR_DIR = SRC_DIR / "models"
HAR_MODELING_FILE = MODELS_SCR_DIR / "har.py"
PATCHTST_MODELING_FILE = MODELS_SCR_DIR / "transformer.py"
HAR_MODEL_FILE = MODEL_DIR / "har_model.pkl"
PATCHTST_MODEL_FILE = MODEL_DIR / "patchtst_model.pth"

HAR_PRED_FILE = PROCESSED_DATA_DIR / "har_predictions.csv"
PATCHTST_PRED_FILE = PROCESSED_DATA_DIR / "patchtst_predictions.csv"

HAR_METRICS_FILE = PROCESSED_DATA_DIR / "har_metrics.csv"
PATCHTST_METRICS_FILE = PROCESSED_DATA_DIR / "patchtst_metrics.csv"

def ensure_directories():
    for dir in [DATA_DIR, MODEL_DIR, LOGS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()