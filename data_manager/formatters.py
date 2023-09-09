from abc import ABC, abstractmethod

import pandas as pd

from configs import configs
from configs.parser import FormatterConfig, GeneralConfig
from data_manager.structure import StructuredData, Target


class Formatter(ABC):
    @abstractmethod
    def __init__(self, general_cfg: GeneralConfig, formatter_cfg: FormatterConfig):
        pass

    @abstractmethod
    def format(self, data: StructuredData) -> StructuredData:
        pass


class ClassificationFormatter(Formatter):
    def __init__(self, general_cfg: GeneralConfig, formatter_cfg: FormatterConfig):
        self.formatter_cfg = formatter_cfg
        self.classification_labels = formatter_cfg.classification_labels

    def format(self, data: StructuredData) -> StructuredData:
        # create one column labels from multiple columns
        label = data.meta[self.classification_labels].apply(tuple, axis=1)
        # encode to numbers
        encoded, encoding = pd.factorize(label)
        encoded = pd.Series(encoded, name="encoded")
        encoding = pd.Series(encoding, name="encoding")
        data.target = Target(label=label, encoded=encoded, encoding=encoding)
        return data


class RegressionFormatter(Formatter):
    def __init__(self, general_cfg: GeneralConfig, formatter_cfg: FormatterConfig):
        self.general_cfg = general_cfg
        self.formatter_cfg = formatter_cfg

    def format(self, data: StructuredData) -> StructuredData:
        regression_label = self.formatter_cfg.regression_label
        measurements_paths = self.formatter_cfg.measurements_paths
        target_column_label = measurements_paths[regression_label][0]
        file_path = measurements_paths[regression_label][1]
        df = pd.read_excel(file_path)
        # keep only rows where the date and treatment are defined in the config
        df = df[
            df[configs.DATE_SLO].isin(self.general_cfg.dates)
            & df[configs.TREATMENT_SLO].isin(self.general_cfg.treatments)
        ].reset_index(drop=True)
        pass
        return data
