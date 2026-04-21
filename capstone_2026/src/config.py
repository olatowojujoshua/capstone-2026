from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


LOCAL_DATA_ROOT = PROJECT_ROOT.parents[1] / "local_capstone_2026" / "data"

DATA_DIR = LOCAL_DATA_ROOT
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

MODELS_DIR = PROJECT_ROOT / "models"

RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


TIME_BUCKET_MINUTES = 15