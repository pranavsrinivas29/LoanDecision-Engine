from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"

CHROMA_DIR = ARTIFACTS_DIR / "chroma_db"

MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

RAW_DATA_PATH = RAW_DATA_DIR / "loan_approval_dataset.csv"

TARGET_COL = "loan_status"

ID_COLUMNS = ["loan_id"]

CATEGORICAL_TEXT_COLUMNS = ["education", "self_employed"]

OPENAI_MODEL_NAME = "gpt-4.1-mini"

TOP_K_RETRIEVAL = 3
LOCAL_EXPLANATION_TOP_N = 5
PREDICTION_THRESHOLD = 0.5


DB_PATH = PROJECT_ROOT / "artifacts" / "loan_agent.db"

# ---------------- MLflow ----------------
MLFLOW_TRACKING_URI = f"file:{PROJECT_ROOT / 'mlruns'}"
MLFLOW_EXPERIMENT_NAME = "loan_approval_training"
MODEL_INFO_PATH = ARTIFACTS_DIR / "active_model_metadata.json"
MODEL_COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"