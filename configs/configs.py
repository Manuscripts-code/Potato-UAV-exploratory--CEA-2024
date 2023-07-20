from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
LOGS_DIR = Path(BASE_DIR, "logs")
DATA_DIR = Path(BASE_DIR, "data")

MUTISPECTRAL_DIR = Path(DATA_DIR, "multispectral_images")
SHAPEFILES_DIR = Path(DATA_DIR, "shapefiles")
