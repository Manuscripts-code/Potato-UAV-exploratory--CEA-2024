from dataclasses import dataclass
from itertools import product

import pandas as pd

from configs.global_enums import MultispectralEnum
from data_manager.geotiffs import MultiGeotiffRaster
from data_manager.mergers import MultiRasterPointsMerger, RasterPointsMerger
from data_manager.shapefiles import PointsShapefile
from utils.config_parser import MultispectralConfig
from utils.utils import ensure_dir


@dataclass
class StructuredData:
    data: pd.DataFrame
    meta: pd.DataFrame
    target: pd.DataFrame = None


class MultispectralLoader:
    def __init__(
        self,
        multispectral_config: MultispectralConfig,
        *,
        save_dir="saved",
        save_coords=False,
        num_closest_points=1,
    ):
        self.save_dir = ensure_dir(save_dir)
        self.save_coords = save_coords
        self.num_closest_points = num_closest_points

        self.rasters_paths, self.shapefiles_paths = multispectral_config.parse_specific_paths()
        (
            self.dates,
            self.treatments,
            self.channels,
            self.location_type,
        ) = multispectral_config.parse_toml_config()

        self._mergers = None
        self._multi_merger = None
        self._structured_data = None

    def __str__(self):
        return f"<MultispectralLoader object with {len(self._mergers)} mergers>"

    def load(self):
        self.load_mergers()
        self.run_merges()
        self.final_merge()
        return self

    def load_mergers(self):
        self._mergers = []
        for date, treatment in product(self.dates, self.treatments):
            merger = self._create_merger(date, treatment, self.channels, self.location_type)
            self._mergers.append(merger)

    def run_merges(self):
        if self._mergers is None:
            raise ValueError("Mergers are not loaded.")
        self._multi_merger = MultiRasterPointsMerger(self._mergers)
        self._multi_merger.run_merges()

    def final_merge(self):
        columns_meta = MultispectralEnum.COLUMNS_SLO.value + [
            MultispectralEnum.TREATMENTS.value,
            MultispectralEnum.DATES.value,
        ]
        columns_data = self.channels

        df_meta_merged = pd.DataFrame(columns=columns_meta)
        df_data_merged = pd.DataFrame(columns=columns_data)

        for merged_df in self.merged_dfs:
            df_data, treatment, date = self._extract_data(merged_df)
            df_meta = self._extract_meta(merged_df, treatment, date, columns_meta)
            df_data_merged = pd.concat([df_data_merged, df_data], axis=0)
            df_meta_merged = pd.concat([df_meta_merged, df_meta], axis=0)

        columns = {
            old_name: new_name
            for old_name, new_name in zip(
                MultispectralEnum.COLUMNS_SLO.value, MultispectralEnum.COLUMNS_ENG.value
            )
        }
        df_meta_merged.rename(columns=columns, inplace=True, errors="raise")
        df_data_merged.reset_index(drop=True, inplace=True)
        df_meta_merged.reset_index(drop=True, inplace=True)
        self._structured_data = StructuredData(data=df_data_merged, meta=df_meta_merged)

    def _extract_data(self, merged_df):
        df_data = merged_df.loc[:, merged_df.columns.isin(self.data_column_names)]
        treatment, date = df_data.columns[0].split("__")[:2]
        channels = [column.split("__")[2] for column in df_data.columns]
        df_data.columns = channels
        return df_data, treatment, date

    def _extract_meta(self, merged_df, treatment, date, columns_meta):
        merged_df[[MultispectralEnum.TREATMENTS.value, MultispectralEnum.DATES.value]] = [
            treatment,
            date,
        ]
        df_meta = merged_df.loc[:, merged_df.columns.isin(columns_meta)]
        return df_meta

    def _create_merger(self, date, treatment, channels, location_type):
        base_path = self.rasters_paths[treatment][date]
        paths = {channel: base_path[channel] for channel in channels}
        raster = MultiGeotiffRaster.from_paths(paths)
        raster.set_name("".join([treatment, "__", date]))
        path_shape = self.shapefiles_paths[treatment][location_type]
        shapefile = PointsShapefile.from_path(path_shape)
        return RasterPointsMerger(
            raster,
            shapefile,
            save_dir=self.save_dir,
            save_coords=self.save_coords,
            num_closest_points=self.num_closest_points,
        )

    @property
    def mergers(self):
        return self._multi_merger.mergers

    @property
    def merged_dfs(self):
        return self._multi_merger.merged_dfs

    @property
    def data_column_names(self):
        return self._multi_merger.data_column_names

    @property
    def structured_data(self):
        return self._structured_data


if __name__ == "__main__":
    from utils.config_parser import ConfigParser

    config_parser = ConfigParser()
    multispectral_config = config_parser.get_multispectral_configs()

    loader = MultispectralLoader(multispectral_config).load()
    pass
