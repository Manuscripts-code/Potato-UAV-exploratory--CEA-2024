from configs import configs, specific_paths
from utils.utils import read_toml


class MultispectralLoader:
    def __init__(self):
        self.rasters_paths = specific_paths.PATHS_MULTISPECTRAL_IMAGES
        self.shapefiles_paths = specific_paths.PATHS_SHAPEFILES


if __name__ == "__main__":
    s = read_toml("./configs/specific/testing.toml")
    pass
