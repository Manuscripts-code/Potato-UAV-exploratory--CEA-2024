import os

from zenml import pipeline

from configs import configs
from configs.parser import ConfigParser
from data_manager.loaders import StructuredData
from steps import data_formatter, data_loader
from utils.plot_utils import save_features_plot
from utils.utils import ensure_dir

data_loader = data_loader.with_options(enable_cache=True)
data_formatter = data_formatter.with_options(enable_cache=True)


@pipeline(enable_cache=True)  # type: ignore
def load_data() -> StructuredData:
    cfg_parser = ConfigParser()
    data = data_loader(cfg_parser.general().without_varieties(), cfg_parser.multispectral())
    data = data_formatter(data, cfg_parser.general(), cfg_parser.formatter())
    return data


def save_features_plot_by_dataset(config_name: str):
    # Set the TOML config file as an environment variable (parsed in the pipelines)
    os.environ[configs.TOML_ENV_NAME] = str(configs.TOML_DIR / f"clf/{config_name}.toml")
    # Run the pipeline only the first time to load the data
    load_data()

    last_run = load_data.model.last_successful_run
    data = last_run.steps["data_formatter"]
    data = data.outputs["data"].load()

    data_ = data.data
    y_data_encoded = data.target.value
    classes = ["".join(name) for name in data.target.encoding.to_dict().values()]
    mapping = {"class 1": "Healthy", "class 2": "Infected"}  # in case of alternaria
    classes = [mapping.get(cls, cls) for cls in classes]

    save_features_plot(
        data_,
        y_data_encoded,
        classes,
        save_path=ensure_dir(configs.SAVE_RESULTS_DIR / "features_plots") / f"{config_name}.pdf",
    )


if __name__ == "__main__":
    # Define config names
    configs_names = ["alternaria_b", "varieties_8_clf"]
    for config in configs_names:
        save_features_plot_by_dataset(config)
