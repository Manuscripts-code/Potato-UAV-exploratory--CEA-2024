from pydantic import BaseModel, Field

from configs import configs, specific_paths

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


class SamplerConfig(BaseModel):
    random_state: int = Field(-1, ge=-1)
    splitter: str
    shuffle: bool
    split_size_val: float = Field(0.2, ge=0, le=1)
    split_size_test: float = Field(0.2, ge=0, le=1)


class FormatterConfig(BaseModel):
    formatter: str
    classes: list[str]


class ConfigParser:
    def __init__(self):
        self.rasters_paths = specific_paths.PATHS_MULTISPECTRAL_IMAGES
        self.shapefiles_paths = specific_paths.PATHS_SHAPEFILES
        self.toml_cfg = configs.CONFIGS_TOML

    def general(self) -> GeneralConfig:
        cfg = self.toml_cfg[configs.GENERAL_CFG_NAME]
        try:
            general_config = GeneralConfig(**cfg)
        except KeyError:
            raise KeyError("Missing key in toml config file.")
        return general_config

    def multispectral(self) -> MultispectralConfig:
        cfg = self.toml_cfg[configs.MULTISPECTRAL_CFG_NAME]
        try:
            multispectral_config = MultispectralConfig(
                rasters_paths=self.rasters_paths,
                shapefiles_paths=self.shapefiles_paths,
                **cfg,
            )
        except KeyError:
            raise KeyError("Missing key in toml config file.")
        return multispectral_config

    def sampler(self) -> SamplerConfig:
        cfg = self.toml_cfg[configs.SAMPLER_CFG_NAME]
        try:
            sampler_config = SamplerConfig(**cfg)
        except KeyError:
            raise KeyError("Missing key in toml config file.")
        return sampler_config

    def formatter(self) -> FormatterConfig:
        cfg = self.toml_cfg[configs.FORMATTER_CFG_NAME]
        try:
            formatter_config = FormatterConfig(**cfg)
        except KeyError:
            raise KeyError("Missing key in toml config file.")
        return formatter_config
