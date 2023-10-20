import itertools
import os

from zenml import pipeline

from configs import configs
from configs.parser import ConfigParser
from data_manager.loaders import StructuredData
from steps import data_features, data_formatter, data_loader, model_creator
from utils.plot_utils import show_umap

data_loader = data_loader.with_options(enable_cache=True)
data_formatter = data_formatter.with_options(enable_cache=True)
data_features = data_features.with_options(enable_cache=True)
model_creator = model_creator.with_options(enable_cache=True)


@pipeline(enable_cache=True)
def load_data() -> StructuredData:
    cfg_parser = ConfigParser()
    data = data_loader(cfg_parser.general().without_varieties(), cfg_parser.multispectral())
    data = data_formatter(data, cfg_parser.general(), cfg_parser.formatter())
    data, _, _ = data_features(data, features_cfg=cfg_parser.features())
    return data


if __name__ == "__main__":
    # configs
    dates = ["2022_06_15", "2022_07_11", "2022_07_20"]
    treatments = ["eko"]
    varieties = [
        # "Carolus",
        # "Alouette",
        # "Twister",
        # "Otolia",
        "KIS_Tamar",
        "KIS_Blegos",
        # "KIS_Kokra",
        # "Levante",
    ]
    # umap
    n_neighbors = 30
    min_dist = 1
    metric = "euclidean"

    ########################################################################

    # Set the TOML config file as an environment variable (parsed in the pipelines)
    os.environ[configs.TOML_ENV_NAME] = str(configs.TOML_DIR / "clf/umap_varieties.toml")

    # Run the pipeline only the first time to load the data
    if not load_data.model.last_successful_run:
        load_data()

    last_run = load_data.model.last_successful_run
    data = last_run.steps["data_features"]
    data = data.outputs["data_train"].load()

    # filer based on parameters
    indices = data.meta.index[
        data.meta[configs.VARIETY_ENG].isin(varieties)
        & data.meta[configs.TREATMENT_ENG].isin(treatments)
        & data.meta[configs.DATE_ENG].isin(dates)
    ].to_list()
    data_copy = data[indices].reset_index()

    show_umap(
        data=data_copy.data,
        y_data_encoded=data_copy.target.value,
        classes=list(itertools.chain(*data_copy.target.encoding.tolist())),
        save_path=None,
        supervised=False,
        **dict(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
        ),
    )
