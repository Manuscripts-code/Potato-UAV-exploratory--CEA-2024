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
SAVE_COORDS = os.getenv("SAVE_COORDS", "false") == "true"
NUM_CLOSEST_POINTS = int(os.getenv("NUM_CLOSEST_POINTS", 10))
CONFIGS_TOML = read_toml(TOML_DIR / os.getenv("DATA_TOML_NAME", "testing.toml"))
