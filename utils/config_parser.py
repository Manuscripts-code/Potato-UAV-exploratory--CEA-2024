from pydantic import BaseModel

from configs import configs, global_enums, specific_paths

TYPE_DICT2 = dict[str, dict[str, str]]
TYPE_DICT3 = dict[str, dict[str, dict[str, str]]]


class GeneralConfig(BaseModel):
    num_closest_points: int


class MultispectralConfig(BaseModel):
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
        self.general_enum = global_enums.GeneralConfigEnum
        self.multispectral_enum = global_enums.MultispectralConfigEnum
        self.toml_cfg = configs.CONFIGS_TOML

    def get_general_configs(self):
        cfg = self.toml_cfg[str(self.general_enum.ROOT)]
        try:
            num_closest_points = cfg[str(self.general_enum.NUM_CLOSEST_POINTS)]
            general_config = GeneralConfig(num_closest_points=num_closest_points)
        except KeyError:
            raise KeyError("Missing key in toml config file.")
        return general_config

    def get_multispectral_configs(self):
        cfg = self.toml_cfg[str(self.multispectral_enum.ROOT)]
        try:
            dates = cfg[str(self.multispectral_enum.DATES)]
            treatments = cfg[str(self.multispectral_enum.TREATMENTS)]
            channels = cfg[str(self.multispectral_enum.CHANNELS)]
            location_type = cfg[str(self.multispectral_enum.LOCATION_TYPE)]
            multispectral_config = MultispectralConfig(
                dates=dates,
                treatments=treatments,
                channels=channels,
                location_type=location_type,
                rasters_paths=self.rasters_paths,
                shapefiles_paths=self.shapefiles_paths,
            )
        except KeyError:
            raise KeyError("Missing key in toml config file.")
        return multispectral_config
