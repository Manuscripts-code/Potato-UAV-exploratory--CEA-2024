from zenml import pipeline
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_registry_deployer_step
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step
from zenml.logger import get_logger

from configs import configs
from configs.parser import ConfigParser
from steps import (
    data_formatter,
    data_loader,
    data_sampler,
    model_deployer,
    model_predictor,
    model_service,
)

logger = get_logger(__name__)


@pipeline(enable_cache=configs.CACHING)
def deployment_inference_pipeline() -> None:
    cfg_parser = ConfigParser()
    logger.info(f"Using toml file: {cfg_parser.toml_cfg_path}")

    data = data_loader(cfg_parser.general(), cfg_parser.multispectral())
    data = data_formatter(data, cfg_parser.formatter())
    data_train, data_val, data_test = data_sampler(data, cfg_parser.sampler())

    # deployed_model = model_deployer(cfg_parser.registry())

    deployed_model = mlflow_model_registry_deployer_step.with_options(
        parameters=dict(
            registry_model_name=cfg_parser.registry().model_name,
            registry_model_version=4,
            timeout=300,
            # or you can use the model stage if you have set it in the MLflow registry
            # registered_model_stage="None" # "Staging", "Production", "Archived"
        )
    )
    model_service.after(deployed_model)
    deployed_model()

    model_deployment_service = model_service()
    model_predictor(model_deployment_service, data_test)


if __name__ == "__main__":
    deployment_inference_pipeline()
