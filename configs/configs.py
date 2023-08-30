import os
from pathlib import Path

from utils.utils import read_toml

# PATHS
BASE_DIR = Path(__file__).parent.parent.absolute()
LOGS_DIR = Path(BASE_DIR, "logs")
DATA_DIR = Path(BASE_DIR, "data")
TOML_DIR = Path(BASE_DIR, "configs", "specific")

MUTISPECTRAL_DIR = Path(DATA_DIR, "multispectral_images")
SHAPEFILES_DIR = Path(DATA_DIR, "shapefiles")

SAVE_DIR = Path(BASE_DIR, "saved")
SAVE_MERGED_DIR = Path(SAVE_DIR, "merged")

# DATA CONFIGS
CACHING = os.getenv("CACHING", "false") == "true"
SAVE_COORDS = os.getenv("SAVE_COORDS", "false") == "true"
CONFIGS_TOML = read_toml(TOML_DIR / os.getenv("DATA_TOML_NAME", "testing.toml"))

# TOML CONFIG ROOT KEYS
GENERAL_CFG_NAME = "general"
MULTISPECTRAL_CFG_NAME = "multispectral"
SAMPLER_CFG_NAME = "sampler"
FORMATTER_CFG_NAME = "formatter"
MODEL_CFG_NAME = "model"
OPTIMIZER_CFG_NAME = "optimizer"

# MULTISPECTRAL LOADER CONFIG
DATES = "dates"
TREATMENTS = "treatments"
COLUMNS_SLO = ["Blok", "Rastlina", "Sorta"]
COLUMNS_ENG = ["blocks", "plants", "varieties"]
