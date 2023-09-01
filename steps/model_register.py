from zenml import step
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step

from configs.parser import RegistryConfig


def model_register(registry_cfg: RegistryConfig) -> step:
    register = mlflow_register_model_step.with_options(
        parameters=dict(
            name=registry_cfg.model_name,
            # version=registry_cfg.version,
            # experiment_name=registry_cfg.experiment_name,
            # run_name=registry_cfg.run_name,
            description=registry_cfg.description,
        )
    )
    return register
