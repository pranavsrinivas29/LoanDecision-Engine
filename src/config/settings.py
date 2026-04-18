from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

APP_ENV = os.getenv("APP_ENV", "development")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-mini")

DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts")))
KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", str(PROJECT_ROOT / "knowledge_base")))
LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs")))
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ARTIFACTS_DIR)))

CHROMA_DIR = ARTIFACTS_DIR / "chroma_db"

MODEL_PATH = MODEL_DIR / "best_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

RAW_DATA_PATH = RAW_DATA_DIR / "loan_approval_dataset.csv"

TARGET_COL = "loan_status"
ID_COLUMNS = ["loan_id"]
CATEGORICAL_TEXT_COLUMNS = ["education", "self_employed"]

TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))
LOCAL_EXPLANATION_TOP_N = int(os.getenv("LOCAL_EXPLANATION_TOP_N", "5"))
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{(DATA_DIR / 'loan_agent.db').as_posix()}"
)

# backward compatibility for old imports
if DATABASE_URL.startswith("sqlite:///"):
    DB_PATH = Path(DATABASE_URL.replace("sqlite:///", ""))
else:
    DB_PATH = DATA_DIR / "loan_agent.db"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"file:{(PROJECT_ROOT / 'mlruns').as_posix()}"
)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "loan_approval_training")

MODEL_INFO_PATH = ARTIFACTS_DIR / "active_model_metadata.json"
MODEL_COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"

DRIFT_DIR = ARTIFACTS_DIR / "drift"
REFERENCE_STATS_PATH = DRIFT_DIR / "reference_stats.json"
LATEST_DRIFT_REPORT_PATH = DRIFT_DIR / "latest_drift_report.json"

MODEL_PATH = MODEL_DIR / "best_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
MODEL_INFO_PATH = ARTIFACTS_DIR / "active_model_metadata.json"
MODEL_COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"

DRIFT_MONITOR_WINDOW_SIZE = int(os.getenv("DRIFT_MONITOR_WINDOW_SIZE", "100"))
ENABLE_AUTO_RETRAIN_ON_DRIFT = os.getenv("ENABLE_AUTO_RETRAIN_ON_DRIFT", "true").lower() == "true"
MIN_RECORDS_FOR_DRIFT_CHECK = int(os.getenv("MIN_RECORDS_FOR_DRIFT_CHECK", "50"))