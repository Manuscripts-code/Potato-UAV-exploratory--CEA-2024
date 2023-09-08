from abc import ABC, abstractmethod

import pandas as pd

from configs.parser import FormatterConfig
from data_manager.structure import StructuredData, Target


class Formatter(ABC):
    def __init__(self, formatter_cfg: FormatterConfig):
        self.formatter_cfg = formatter_cfg

    @abstractmethod
    def format(self, data: StructuredData) -> StructuredData:
        pass


class ClassificationFormatter(Formatter):
    def __init__(self, formatter_cfg: FormatterConfig):
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
    def __init__(self, formatter_cfg: FormatterConfig):
        self.formatter_cfg = formatter_cfg
        self.regression_label = formatter_cfg.regression_label

    def format(self, data: StructuredData) -> StructuredData:
        pass
