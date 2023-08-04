import pandas as pd

from data_manager.loaders import StructuredData


class ClassificationFormatter:
    def __init__(self, labels_to_encode):
        self.labels_to_encode = labels_to_encode

    def format(self, data: StructuredData) -> StructuredData:
        # create one column labels from multiple columns
        data.label = data.meta[self.labels_to_encode].apply(tuple, axis=1)
        # encode to numbers
        codes, uniques = pd.factorize(data.label)
        data.target = pd.Series(codes)
        data.label_target_relation = pd.Series(uniques)
        return data
