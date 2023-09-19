import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# PATHS
BASE_DIR = Path(__file__).parent.parent.absolute()
LOGS_DIR = Path(BASE_DIR, "logs")
DATA_DIR = Path(BASE_DIR, "data")
TOML_DIR = Path(BASE_DIR, "configs", "specific")

MUTISPECTRAL_DIR = Path(DATA_DIR, "multispectral_images")
SHAPEFILES_DIR = Path(DATA_DIR, "shapefiles")
MEASUREMENTS_DIR = Path(DATA_DIR, "measurements")

SAVE_DIR = Path(BASE_DIR, "saved")
SAVE_MERGED_DIR = Path(SAVE_DIR, "merged")

# MAKE DIRS
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# DATA CONFIGS
CACHING = os.getenv("CACHING", "false") == "true"
SAVE_COORDS = os.getenv("SAVE_COORDS", "false") == "true"
TOML_ENV_NAME = "DATA_TOML_NAME"
TOML_DEFAULT_FILE_NAME = "test_classification.toml"
REGISTER_MODEL = os.getenv("REGISTER_MODEL", "false") == "true"
USE_REDUCED_DATASET = os.getenv("USE_REDUCED_DATASET", "false") == "true"

# TOML CONFIG ROOT KEYS
GENERAL_CFG_NAME = "general"
MULTISPECTRAL_CFG_NAME = "multispectral"
SAMPLER_CFG_NAME = "sampler"
FORMATTER_CFG_NAME = "formatter"
MODEL_CFG_NAME = "model"
OPTIMIZER_CFG_NAME = "optimizer"
EVALUATOR_CFG_NAME = "evaluator"
REGISTRY_CFG_NAME = "registry"

# MULTISPECTRAL AND MEASUREMENTS LOADER CONFIG
DATE_ENG = "dates"
TREATMENT_ENG = "treatments"
BLOCK_ENG = "blocks"
PLANT_ENG = "plants"
VARIETY_ENG = "varieties"

DATE_SLO = "Datum"
TREATMENT_SLO = "Poskus"
BLOCK_SLO = "Blok"
PLANT_SLO = "Rastlina"
VARIETY_SLO = "Sorta"

# MLFLOW ARTIFACTS SAVE VARS
MLFLOW_TRAIN = "train"
MLFLOW_TEST = "test"
MLFLOW_RESULTS = "results"
MLFLOW_CONFIGS = "configs"
MLFLOW_MODEL = "model"

# COMMAND LINE CONFIGS
CMD_TRAIN_AND_REGISTER = "train"
CMD_DEPLOY_AND_TEST = "test"
CMD_EXECUTE_ALL = "all"

# MATERIALIZER CONFIGS
MATERIALIZER_DATA_JSON = "structured_data.json"
MATERIALIZER_DESCRIBE_DATA_CSV = "describe_data.csv"
MATERIALIZER_DESCRIBE_META_CSV = "describe_meta.csv"
MATERIALIZER_DESCRIBE_TARGET_CSV = "describe_target.csv"

# DATABASE CONFIGS
DB_NAME = os.getenv("DB_NAME", "database.db")
DB_PATH = Path(SAVE_DIR, DB_NAME)
DB_ECHO = os.getenv("DB_ECHO", "false") == "true"

# MISCELLANEOUS
DATE_FORMAT = "%Y_%m_%d"
