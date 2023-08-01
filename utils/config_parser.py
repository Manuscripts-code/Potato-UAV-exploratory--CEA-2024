from dataclasses import dataclass

from configs import configs, global_enums, specific_paths

TYPE_DICT2 = dict[str, dict[str, str]]
TYPE_DICT3 = dict[str, dict[str, dict[str, str]]]


@dataclass
class MultispectralConfig:
    dates: list[str]
    treatments: list[str]
    channels: list[str]
    location_type: str
    rasters_paths: TYPE_DICT3
    shapefiles_paths: TYPE_DICT2

    def parse_specific_paths(self):
        return self.rasters_paths.copy(), self.shapefiles_paths.copy()

    def parse_toml_config(self):
        return self.dates, self.treatments, self.channels, self.location_type


class ConfigParser:
    def __init__(self):
        self.rasters_paths = specific_paths.PATHS_MULTISPECTRAL_IMAGES
        self.shapefiles_paths = specific_paths.PATHS_SHAPEFILES
        self.multispectral_enum = global_enums.MultispectralEnum
        self.cfg = configs.CONFIGS_TOML[str(self.multispectral_enum.ROOT)]

    def get_multispectral_configs(self):
        try:
            dates = self.cfg[str(self.multispectral_enum.DATES)]
            treatments = self.cfg[str(self.multispectral_enum.TREATMENTS)]
            channels = self.cfg[str(self.multispectral_enum.CHANNELS)]
            location_type = self.cfg[str(self.multispectral_enum.LOCATION_TYPE)]
            multispectral_config = MultispectralConfig(
                dates, treatments, channels, location_type, self.rasters_paths, self.shapefiles_paths
            )
        except KeyError:
            raise KeyError("Missing key in toml config file.")
        return multispectral_config
