from sklearn.pipeline import Pipeline
from zenml import step
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step
from zenml.model_registries.base_model_registry import ModelRegistryModelMetadata

from configs.parser import RegistryConfig


def model_register(best_model: Pipeline, registry_cfg: RegistryConfig) -> None:
    register = mlflow_register_model_step.with_options(
        parameters=dict(
            name=registry_cfg.model_name,
            description=registry_cfg.description,
            metadata=ModelRegistryModelMetadata(**registry_cfg.metadata_dict),
        )
    )
    register(best_model)
