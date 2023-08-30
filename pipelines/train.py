from zenml import pipeline

from configs import configs
from configs.parser import ConfigParser
from steps import data_formatter, data_loader, data_sampler, model_creator, model_optimizer


@pipeline(enable_cache=configs.CACHING)
def train_and_register_model_pipeline() -> None:
    cfg_parser = ConfigParser()

    data = data_loader(cfg_parser.general(), cfg_parser.multispectral())
    data = data_formatter(data, cfg_parser.formatter())
    data_train, data_val, data_test = data_sampler(data, cfg_parser.sampler())

    model = model_creator(cfg_parser.model())
    model_best = model_optimizer(model, data_train, data_val, cfg_parser.optimizer())


if __name__ == "__main__":
    train_and_register_model_pipeline()
