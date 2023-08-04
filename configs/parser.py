from functools import partial

from pydantic import BaseModel, Field, ValidationError

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
    splitter: str
    shuffle: bool = True
    random_state: int = Field(-1, ge=-1)
    split_size_val: float = Field(0.2, ge=0, le=1)
    split_size_test: float = Field(0.2, ge=0, le=1)


class FormatterConfig(BaseModel):
    formatter: str
    Labels_to_encode: list[str]


class ConfigParser:
    def __init__(self):
        self.rasters_paths = specific_paths.PATHS_MULTISPECTRAL_IMAGES
        self.shapefiles_paths = specific_paths.PATHS_SHAPEFILES
        self.toml_cfg = configs.CONFIGS_TOML

    def _parse_config(self, config_name: str, config_class: type[BaseModel]) -> BaseModel:
        specific_cfg = self.toml_cfg[config_name]
        try:
            config = config_class(**specific_cfg)
        except ValidationError as err:
            print(f"Toml configuration error. \nProblem in: {str(err.model)} \n{err}")
            raise
        return config

    def general(self) -> GeneralConfig:
        return self._parse_config(configs.GENERAL_CFG_NAME, GeneralConfig)

    def multispectral(self) -> MultispectralConfig:
        config = partial(
            MultispectralConfig,
            rasters_paths=self.rasters_paths,
            shapefiles_paths=self.shapefiles_paths,
        )
        return self._parse_config(configs.MULTISPECTRAL_CFG_NAME, config)

    def sampler(self) -> SamplerConfig:
        return self._parse_config(configs.SAMPLER_CFG_NAME, SamplerConfig)

    def formatter(self) -> FormatterConfig:
        return self._parse_config(configs.FORMATTER_CFG_NAME, FormatterConfig)
