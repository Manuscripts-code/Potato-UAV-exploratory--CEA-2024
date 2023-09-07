import os
from pathlib import Path

# PATHS
BASE_DIR = Path(__file__).parent.parent.absolute()
LOGS_DIR = Path(BASE_DIR, "logs")
DATA_DIR = Path(BASE_DIR, "data")
TOML_DIR = Path(BASE_DIR, "configs", "specific")

MUTISPECTRAL_DIR = Path(DATA_DIR, "multispectral_images")
SHAPEFILES_DIR = Path(DATA_DIR, "shapefiles")

SAVE_DIR = Path(BASE_DIR, "saved")
SAVE_MERGED_DIR = Path(SAVE_DIR, "merged")

# MAKE DIRS
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# DATA CONFIGS
CACHING = os.getenv("CACHING", "false") == "true"
SAVE_COORDS = os.getenv("SAVE_COORDS", "false") == "true"
TOML_ENV_NAME = "DATA_TOML_NAME"
TOML_DEFAULT_FILE_NAME = "testing.toml"

# TOML CONFIG ROOT KEYS
GENERAL_CFG_NAME = "general"
MULTISPECTRAL_CFG_NAME = "multispectral"
SAMPLER_CFG_NAME = "sampler"
FORMATTER_CFG_NAME = "formatter"
MODEL_CFG_NAME = "model"
OPTIMIZER_CFG_NAME = "optimizer"
EVALUATOR_CFG_NAME = "evaluator"
REGISTRY_CFG_NAME = "registry"

# MULTISPECTRAL LOADER CONFIG
DATES = "dates"
TREATMENTS = "treatments"
COLUMNS_SLO = ["Blok", "Rastlina", "Sorta"]
COLUMNS_ENG = ["blocks", "plants", "varieties"]

# MLFLOW ARTIFACTS SAVE VARS
MLFLOW_TRAIN = "train"
MLFLOW_TEST = "test"
MLFLOW_RESULTS = "results"
MLFLOW_CONFIGS = "configs"
MLFLOW_MODEL = "model"
