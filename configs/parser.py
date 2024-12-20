import logging
import os
from functools import partial
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from rich import print

from configs import configs, paths
from utils.utils import read_toml

TYPE_DICT2 = dict[str, dict[str, str]]
TYPE_DICT3 = dict[str, dict[str, dict[str, str]]]
TYPE_DICT_TUPL = dict[str, tuple[str, str]]


class GeneralConfig(BaseModel):
    raster_num_closest_points: int
    dates: list[str]
    treatments: list[str]
    varieties: list[str] | None

    def without_varieties(self):
        return self.copy(update={"varieties": None})


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


class FeaturesConfig(BaseModel):
    features_engineer: str = None
    feateng_steps: int = 1
    verbose: int = 0
    n_jobs: int = 1

    def params(self):
        dict_ = self.dict()
        dict_.pop("features_engineer", None)
        return dict_


class BalancerConfig(BaseModel):
    use: bool = False


class FormatterConfig(BaseModel):
    formatter: str
    measurements_paths: TYPE_DICT_TUPL
    regression_label: str = None
    classification_label: str = None
    classification_labels: list[str] = None
    date_as_feature: bool = False
    average_duplicates: bool = False
    stratify_by_meta: bool = False


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
    scoring_metric: str
    scoring_mode: str
    n_jobs: int = 1
    timeout: int = None


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
        self.rasters_paths = paths.PATHS_MULTISPECTRAL_IMAGES
        self.shapefiles_paths = paths.PATHS_SHAPEFILES
        self.toml_cfg_path = configs.TOML_DIR / os.getenv(
            configs.TOML_ENV_NAME,
            configs.TOML_DEFAULT_FILE_NAME,
        )
        self.toml_cfg = self._prepare_toml(self.toml_cfg_path)

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

    def features(self) -> FeaturesConfig:
        return self._parse_config(configs.FEATURES_CFG_NAME, FeaturesConfig)

    def balancer(self) -> BalancerConfig:
        return self._parse_config(configs.BALANCER_CFG_NAME, BalancerConfig)

    def formatter(self) -> FormatterConfig:
        config = partial(
            FormatterConfig,
            measurements_paths=paths.PATHS_MEASUREMENTS,
        )
        return self._parse_config(configs.FORMATTER_CFG_NAME, config)

    def model(self) -> ModelConfig:
        return self._parse_config(configs.MODEL_CFG_NAME, ModelConfig)

    def optimizer(self) -> OptimizerConfig:
        return self._parse_config(configs.OPTIMIZER_CFG_NAME, OptimizerConfig)

    def evaluator(self) -> EvaluatorConfig:
        return self._parse_config(configs.EVALUATOR_CFG_NAME, EvaluatorConfig)

    def registry(self) -> RegistryConfig:
        return self._parse_config(configs.REGISTRY_CFG_NAME, RegistryConfig)

    def _parse_config(self, config_name: str, config_class: type[BaseModel]) -> BaseModel:
        try:
            specific_cfg = self.toml_cfg[config_name]
        except KeyError as err:
            print(f"Warning: Config name not found in toml file: {err}")
            return config_class()
        try:
            config = config_class(**specific_cfg)
        except ValidationError as err:
            print(f"Error: Toml configuration problem: {str(err.model)} \n{err}")
            raise
        return config

    def _prepare_toml(self, toml_cfg_path: Path) -> dict:
        toml_cfg = read_toml(toml_cfg_path)
        toml_base_cfg = read_toml(toml_cfg_path.parent / configs.BASE_CFG_NAME)

        # rewrite base config with specific config
        for key, value in toml_base_cfg.items():
            if key not in toml_cfg:
                continue
            for key2, value2 in value.items():
                if key2 not in toml_cfg[key]:
                    continue
                toml_base_cfg[key][key2] = toml_cfg[key][key2]
                logging.info(
                    f"Rewriting base toml: {key} / {key2} = '{value2}'  -->  '{toml_cfg[key][key2]}'"
                )

        return toml_base_cfg
