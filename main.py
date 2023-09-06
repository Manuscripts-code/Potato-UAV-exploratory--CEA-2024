import os

import click
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

from configs import configs
from pipelines.test import deployment_inference_pipeline
from pipelines.train import train_and_register_model_pipeline


@click.command()
@click.option(
    "--toml-config-file",
    default=configs.TOML_DEFAULT_FILE_NAME,
    type=str,
    help="Select among possible toml configs located in 'configs/specific/*.toml'",
)
def main(toml_config_file: str):
    os.environ[configs.TOML_ENV_NAME] = toml_config_file
    train_and_register_model_pipeline()
    deployment_inference_pipeline()

    print(
        "\nYou can run:\n "
        f"[italic green]    mlflow ui --backend-store-uri {get_tracking_uri()}"
        "[/italic green]\n ...to inspect your experiment runs within the MLflow"
        " UI.\nYou can find your runs tracked within the "
        "`mlflow_example_pipeline` experiment. There you'll also be able to "
        "compare two or more runs.\n\n"
    )


if __name__ == "__main__":
    main()
