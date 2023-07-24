from functools import lru_cache
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.spatial import distance

from configs import configs
from data_manager.geotiff import GeotiffRaster
from data_manager.shapefile import ShapefilePoints
from utils.utils import ensure_dir


class Merger:
    def __init__(self, raster: GeotiffRaster, shapefile: ShapefilePoints):
        self._raster = raster
        self._shapefile = shapefile

        self.shapefile_X = self._shapefile.X
        self.shapefile_Y = self._shapefile.Y

        self.merged_df = None
        self.save_dir = ensure_dir(configs.SAVE_MERGED_DIR)

    @lru_cache(maxsize=None)
    def merged_data(self):
        # TEMP!
        shapefile_df = self._shapefile.to_pandas()
        self.merged_df = shapefile_df.iloc[0:50].copy().reset_index(drop=True)
        #

        reflectance_list = self._extract_reflectances(
            self._raster,
            self.merged_df,
            n_closest=10,
            save_coords=True,
        )
        self.merged_df[self._raster.DATA_COLUMN_NAME] = reflectance_list

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

        for _, row in shapefile_df.iterrows():
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
            self._save_coords(coordinates_list)

        return reflectance_list

    def _get_closest_distance_indices(self, arr1, arr2, metric="euclidean", n_closest=1):
        dist = distance.cdist(arr1, arr2, metric)
        indices = np.argpartition(dist, kth=n_closest, axis=0)[0:n_closest]
        # closest_values = np.sort(dist[indices], axis=0)
        return indices.flatten()

    def _save_coords(self, coordinates_list):
        coordinates = np.concatenate(coordinates_list, axis=0)
        coordinates_df = pd.DataFrame(coordinates, columns=[self.shapefile_X, self.shapefile_Y])
        coordinates_df.to_csv(self.save_dir / "coordinates_closest.csv", index=False)


if __name__ == "__main__":
    file = configs.SHAPEFILES_DIR / "oznake.shp"
    shapefile = ShapefilePoints(file)

    file = (
        configs.MUTISPECTRAL_DIR
        / "2022_06_15__eko_ecobreed/Ecobreed_krompir_EKO_15_06_2022_transparent_reflectance_blue_modified.tif"
    )
    raster = GeotiffRaster(file)

    merger = Merger(raster, shapefile)
    merger.merged_data()
