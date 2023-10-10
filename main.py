import os

import click
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

from configs import configs
from pipelines.test import deployment_inference_pipeline
from pipelines.train import train_and_register_model_pipeline
from utils.utils import set_random_seed


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice(
        [configs.CMD_TRAIN_AND_REGISTER, configs.CMD_DEPLOY_AND_TEST, configs.CMD_EXECUTE_ALL]
    ),
    default=configs.CMD_EXECUTE_ALL,
    help="Optionally you can choose to only run specific pipelines.",
)
@click.option(
    "--toml-config-file",
    default=configs.TOML_DEFAULT_FILE_NAME,
    type=str,
    help="Select among possible toml configs located in 'configs/specific/*.toml'",
)
def main(config: str, toml_config_file: str):
    set_random_seed(configs.RANDOM_SEED)
    # Set the TOML config file as an environment variable (parsed in the pipelines)
    os.environ[configs.TOML_ENV_NAME] = toml_config_file

    do_train_and_register = config == configs.CMD_TRAIN_AND_REGISTER or config == configs.CMD_EXECUTE_ALL
    do_deploy_and_test = config == configs.CMD_DEPLOY_AND_TEST or config == configs.CMD_EXECUTE_ALL

    if do_train_and_register:
        train_and_register_model_pipeline()

    if do_deploy_and_test:
        deployment_inference_pipeline()

    print(
        "\nYou can run:\n "
        f"[italic green]    mlflow ui --backend-store-uri {get_tracking_uri()} [/italic green]\n"
        "--> to inspect your experiment runs within the MLflow UI.\n"
        f"[italic green]    optuna-dashboard sqlite:///{configs.DB_PATH} [/italic green]\n"
        "--> to inspect your optuna study run within the Optuna dashboard.\n"
    )


if __name__ == "__main__":
    main()
