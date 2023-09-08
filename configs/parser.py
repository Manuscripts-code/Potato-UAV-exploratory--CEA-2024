import os
from functools import partial

from pydantic import BaseModel, Field, ValidationError

from configs import configs, specific_paths
from utils.utils import read_toml

TYPE_DICT2 = dict[str, dict[str, str]]
TYPE_DICT3 = dict[str, dict[str, dict[str, str]]]


class GeneralConfig(BaseModel):
    raster_num_closest_points: int
    dates: list[str]
    treatments: list[str]


class MultispectralConfig(BaseModel):
    channels: list[str]
    location_type: str
    rasters_paths: TYPE_DICT3
    shapefiles_paths: TYPE_DICT2

    def parse_specific_paths(self):
        return self.rasters_paths.copy(), self.shapefiles_paths.copy()


class SamplerConfig(BaseModel):
    splitter: str
    split_size_val: float = Field(0.2, ge=0, le=1)
    split_size_test: float = Field(0.2, ge=0, le=1)
    shuffle: bool = True
    random_state: int = Field(-1, ge=-1)
    stratify: bool = True

    def params(self):
        dict_ = self.dict()
        dict_.pop("splitter", None)
        return dict_


class FormatterConfig(BaseModel):
    formatter: str
    regression_label: str = None
    classification_labels: list[str] = None


class ModelConfig(BaseModel):
    pipeline: list[str]
    unions: dict[str, list[str]] = {}


class TunedParametersConfig(BaseModel):
    optimize_int: dict[str, list[int]] = {}
    optimize_float: dict[str, list[float]] = {}
    optimize_category: dict[str, list] = {}


class ValidatorConfig(BaseModel):
    validator: str
    n_splits: int = None
    n_repeats: int = None
    random_state: int = None

    def params(self):
        dict_ = self.dict()
        dict_.pop("validator", None)
        return dict_


class OptimizerConfig(BaseModel):
    tuned_parameters: TunedParametersConfig
    validator: ValidatorConfig
    n_trials: int
    n_jobs: int
    scoring_metric: str
    scoring_mode: str
    timeout: int = 1000


class RegistryConfig(BaseModel):
    model_name: str
    description: str
    metadata: list[str]
    timeout: int = 100

    @property
    def metadata_dict(self):
        return dict(param.split("=") for param in self.metadata)


class EvaluatorConfig(BaseModel):
    logger: str


class ConfigParser:
    def __init__(self):
        self.rasters_paths = specific_paths.PATHS_MULTISPECTRAL_IMAGES
        self.shapefiles_paths = specific_paths.PATHS_SHAPEFILES
        self.toml_cfg_path = configs.TOML_DIR / os.getenv(
            configs.TOML_ENV_NAME,
            configs.TOML_DEFAULT_FILE_NAME,
        )
        self.toml_cfg = read_toml(self.toml_cfg_path)

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

    def model(self) -> ModelConfig:
        return self._parse_config(configs.MODEL_CFG_NAME, ModelConfig)

    def optimizer(self) -> OptimizerConfig:
        return self._parse_config(configs.OPTIMIZER_CFG_NAME, OptimizerConfig)

    def evaluator(self) -> EvaluatorConfig:
        return self._parse_config(configs.EVALUATOR_CFG_NAME, EvaluatorConfig)

    def registry(self) -> RegistryConfig:
        return self._parse_config(configs.REGISTRY_CFG_NAME, RegistryConfig)
