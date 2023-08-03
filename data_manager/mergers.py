import numpy as np
import pandas as pd
from rich.progress import track
from scipy.spatial import distance

from data_manager.geotiffs import GeotiffRaster, MultiGeotiffRaster
from data_manager.shapefiles import PointsShapefile
from utils.utils import ensure_dir


class RasterPointsMerger:
    def __init__(
        self,
        rasters: MultiGeotiffRaster,
        shapefile: PointsShapefile,
        *,
        save_dir="saved",
        save_coords=False,
        num_closest_points=1,
    ):
        self._rasters = rasters
        self._shapefile = shapefile

        self.shapefile_X = self._shapefile.X
        self.shapefile_Y = self._shapefile.Y

        self._merged_df = None
        self.save_dir = ensure_dir(save_dir)
        self.save_coords = save_coords
        self.num_closest_points = num_closest_points

    def run_merge(self):
        # # TEMP!
        # shapefile_df = self._shapefile.to_pandas()
        # self._merged_df = shapefile_df.iloc[0:4].copy().reset_index(drop=True)
        self._merged_df = self._shapefile.to_pandas()
        # #
        for raster in self._rasters:
            reflectance_list = self._extract_reflectances(
                raster,
                self._merged_df,
                n_closest=self.num_closest_points,
                save_coords=self.save_coords,
            )
            self._merged_df[raster.name] = reflectance_list
        return self._merged_df

    def _extract_reflectances(
        self,
        raster: GeotiffRaster,
        shapefile_df: pd.DataFrame,
        n_closest=1,
        save_coords=False,
    ):
        raster_df = raster.to_pandas()
        coordinates = raster_df[[raster.X, raster.Y]].to_numpy()
        reflectances = raster_df[[raster.DATA_COLUMN_NAME]].to_numpy()

        reflectance_list = []
        coordinates_list = []

        for _, row in track(
            shapefile_df.iterrows(), description=f"Extracting reflectances for: {raster.name}"
        ):
            row_coord = row[[self.shapefile_X, self.shapefile_Y]].to_numpy().astype(float)
            row_coord = np.expand_dims(row_coord, axis=0)
            n_closest_indices = self._get_closest_distance_indices(
                coordinates,
                row_coord,
                n_closest=n_closest,
            )
            reflectance_list.append(reflectances[n_closest_indices].mean())

            if save_coords:
                coordinates_list.append(coordinates[n_closest_indices])

        if save_coords:
            self._save_coords(coordinates_list, save_name=raster.name)

        return reflectance_list

    def _get_closest_distance_indices(self, arr1, arr2, metric="euclidean", n_closest=1):
        dist = distance.cdist(arr1, arr2, metric)
        indices = np.argpartition(dist, kth=n_closest, axis=0)[0:n_closest]
        # closest_values = np.sort(dist[indices], axis=0)
        return np.sort(indices.flatten())

    def _save_coords(self, coordinates_list, save_name=""):
        coordinates = np.concatenate(coordinates_list, axis=0)
        coordinates_df = pd.DataFrame(coordinates, columns=[self.shapefile_X, self.shapefile_Y])
        coordinates_df.to_csv(self.save_dir / f"{save_name}_coordinates_closest.csv", index=False)

    def merged_df(self):
        return self._merged_df

    @property
    def rasters_name(self):
        return self._rasters.name

    @property
    def rasters_paths(self):
        return self._rasters.paths

    @property
    def rasters_channels(self):
        return self._rasters.channels

    @property
    def shapefile_name(self):
        return self._shapefile.name

    @property
    def shapefile_path(self):
        return self._shapefile.path


class MultiRasterPointsMerger:
    def __init__(self, merger: list[RasterPointsMerger] = None):
        self._mergers = [] if merger is None else merger
        self._merged_dfs = None
        self._data_column_names = None

    def __str__(self):
        return f"<MultiRasterPointsMerger object with {len(self._mergers)} mergers>"

    def __len__(self):
        return len(self._mergers)

    def __getitem__(self, index):
        return self._mergers[index]

    def add_mergers(self, mergers: list[RasterPointsMerger]):
        self._mergers.extend(mergers)
        return self

    def run_merges(self):
        self._data_column_names = []
        self._merged_dfs = []
        for merger in self._mergers:
            merged_df = merger.run_merge()
            merged_df, new_names = self._change_column_names(
                merged_df, merger.rasters_name, merger.rasters_channels
            )
            self._data_column_names.extend(new_names)
            self._merged_dfs.append(merged_df)

    def _change_column_names(self, merged_df, rasters_name, rasters_channels):
        new_names = [f"{rasters_name}__{channel}" for channel in rasters_channels]
        columns = {old_name: new_name for old_name, new_name in zip(rasters_channels, new_names)}
        merged_df.rename(columns=columns, inplace=True, errors="raise")
        return merged_df, new_names

    @property
    def mergers(self):
        return self._mergers

    @property
    def merged_dfs(self):
        return self._merged_dfs

    @property
    def data_column_names(self):
        return self._data_column_names


if __name__ == "__main__":
    from configs import specific_paths

    date = "2022_06_15"
    base_path = specific_paths.PATHS_MULTISPECTRAL_IMAGES["eko"][date]
    paths = {
        "blue": base_path["blue"],
        "green": base_path["green"],
    }
    raster = MultiGeotiffRaster.from_paths(paths)
    raster.set_name(date)

    path_shape = specific_paths.PATHS_SHAPEFILES["eko"]["measured"]
    shapefile = PointsShapefile.from_path(path_shape)

    merger1 = RasterPointsMerger(raster, shapefile)

    ###############

    date = "2022_07_11"
    base_path = specific_paths.PATHS_MULTISPECTRAL_IMAGES["eko"][date]
    paths = {
        "blue": base_path["blue"],
        "green": base_path["green"],
    }
    raster = MultiGeotiffRaster.from_paths(paths)
    raster.set_name(date)

    merger2 = RasterPointsMerger(raster, shapefile)

    ###############

    mergers = MultiRasterPointsMerger().add_mergers([merger1, merger2])
    mergers.run_merges()
