from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
LOGS_DIR = Path(BASE_DIR, "logs")
DATA_DIR = Path(BASE_DIR, "data")
MUTISPECTRAL_DATA_DIR = Path(DATA_DIR, "multispectral_images")
