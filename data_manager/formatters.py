import logging
from abc import ABC, abstractmethod

import pandas as pd

from configs import configs
from configs.parser import FormatterConfig, GeneralConfig
from data_structures.schemas import ClassificationTarget, RegressionTarget, StructuredData


class Formatter(ABC):
    def __init__(self, general_cfg: GeneralConfig, formatter_cfg: FormatterConfig):
        self.general_cfg = general_cfg
        self.formatter_cfg = formatter_cfg

        self.columns_slo = [
            configs.DATE_SLO,
            configs.TREATMENT_SLO,
            configs.BLOCK_SLO,
            configs.PLANT_SLO,
            configs.VARIETY_SLO,
        ]
        self.columns_eng = [
            configs.DATE_ENG,
            configs.TREATMENT_ENG,
            configs.BLOCK_ENG,
            configs.PLANT_ENG,
            configs.VARIETY_ENG,
        ]
        self.columns = {
            old_name: new_name for old_name, new_name in zip(self.columns_slo, self.columns_eng)
        }

    @abstractmethod
    def format(self, data: StructuredData) -> StructuredData:
        pass

    def _filter_data(self, data: StructuredData) -> StructuredData:
        # keep only rows where the date, treatment and varieties are defined in the toml config
        # note: date and treatment filtered before, so only varieties are filtered here
        indices = data.meta.index[
            data.meta[configs.VARIETY_ENG].isin(self.general_cfg.varieties)
            & data.meta[configs.TREATMENT_ENG].isin(self.general_cfg.treatments)
            & data.meta[configs.DATE_ENG].isin(self.general_cfg.dates)
        ].to_list()
        return data[indices].reset_index()

    def _modify_data(self, data: StructuredData) -> StructuredData:
        if self.formatter_cfg.date_as_feature:
            # ! the increasing dates have higher values, consequently sorting in ascending order
            encoded, encoding = pd.factorize(data.meta[configs.DATE_ENG].sort_values())
            data.data[configs.DATE_FEATURE_ENCODING] = encoded

        if self.formatter_cfg.stratify_by_meta:
            # apply stratify by date, treatment, block and variety
            stratify = data.meta[
                [configs.DATE_ENG, configs.TREATMENT_ENG, configs.BLOCK_ENG, configs.VARIETY_ENG]
            ].apply(tuple, axis=1)
            encoded, encoding = pd.factorize(stratify)
            encoded = pd.Series(encoded)
            # Calculate the size of each group and sample the same number of values from each group
            group_sizes = encoded.groupby(encoded).size()
            stratified = encoded.groupby(encoded).apply(lambda x: x.sample(n=group_sizes.min()))
            data = data[stratified.reset_index()["level_1"]].reset_index()

        return data

    def _sanity_check(self, df1: pd.DataFrame, df2: pd.DataFrame, df1_name: str, df2_name: str):
        if len(df1) != len(df2):
            logging.error(
                f"Number of rows in {df1_name} and {df2_name} do not match.\n"
                f"{df1_name}: {len(df1)}, {df2_name}: {len(df2)}."
            )
            raise ValueError(f"Number of rows in {df1_name} and {df2_name} do not match.")

    def _merge_measurements_with_meta(
        self, meta: pd.DataFrame, measurements: pd.DataFrame, target_column_label: str
    ) -> pd.DataFrame:
        measurements = measurements[self.columns_eng + [target_column_label]]

        if self.formatter_cfg.average_duplicates:
            measurements = measurements.groupby(self.columns_eng).mean().reset_index()

        return pd.merge(meta, measurements, on=self.columns_eng, how="inner", validate="one_to_one")


class ClassificationFormatter(Formatter):
    def format(self, data: StructuredData) -> StructuredData:
        # keep only varieties defined in the toml config
        data = self._filter_data(data)
        # add other features to data.data
        data = self._modify_data(data)

        # create one column labels from multiple columns
        label = data.meta[self.formatter_cfg.classification_labels].apply(tuple, axis=1)
        # encode to numbers
        encoded, encoding = pd.factorize(label)
        encoded = pd.Series(encoded, name="encoded")
        encoding = pd.Series(encoding, name="encoding")
        data.target = ClassificationTarget(label=label, value=encoded, encoding=encoding)
        self._sanity_check(data.data, data.target.value, "Data", "Target")
        self._sanity_check(data.meta, data.target.value, "Metadata", "Target")
        return data


class RegressionFromExcelFormatter(Formatter):
    def format(self, data: StructuredData) -> StructuredData:
        regression_label = self.formatter_cfg.regression_label
        measurements_paths = self.formatter_cfg.measurements_paths
        target_column_label = measurements_paths[regression_label][0]
        file_path = measurements_paths[regression_label][1]

        measurements = pd.read_excel(file_path)
        # change date format to match the one in the config
        measurements[configs.DATE_SLO] = measurements[configs.DATE_SLO].dt.strftime(configs.DATE_FORMAT)
        measurements.rename(columns=self.columns, inplace=True, errors="raise")

        # keep only varieties defined in the toml config
        data = self._filter_data(data)
        # add other features to data.data
        data = self._modify_data(data)
        # match rows from measurements with rows from metadata
        merged_df = self._merge_measurements_with_meta(data.meta, measurements, target_column_label)
        self._sanity_check(data.meta, merged_df, "Metadata", "Measurements")
        data.target = RegressionTarget(name=target_column_label, value=merged_df[target_column_label])
        return data


class ClassificationFromExcelFormatter(Formatter):
    def format(self, data: StructuredData) -> StructuredData:
        classification_label = self.formatter_cfg.classification_label
        measurements_paths = self.formatter_cfg.measurements_paths
        target_column_label = measurements_paths[classification_label][0]
        file_path = measurements_paths[classification_label][1]

        measurements = pd.read_excel(file_path)
        # change date format to match the one in the config
        measurements[configs.DATE_SLO] = measurements[configs.DATE_SLO].dt.strftime(configs.DATE_FORMAT)
        measurements.rename(columns=self.columns, inplace=True, errors="raise")

        # keep only varieties defined in the toml config
        data = self._filter_data(data)
        # add other features to data.data
        data = self._modify_data(data)
        # match rows from measurements with rows from metadata
        merged_df = self._merge_measurements_with_meta(data.meta, measurements, target_column_label)
        self._sanity_check(data.meta, merged_df, "Metadata", "Measurements")
        label = merged_df[target_column_label]
        # encode to numbers
        encoded, encoding = pd.factorize(label)
        encoded = pd.Series(encoded, name="encoded")
        encoding = pd.Series(encoding, name="encoding")
        data.target = ClassificationTarget(label=label, value=encoded, encoding=encoding)
        return data
