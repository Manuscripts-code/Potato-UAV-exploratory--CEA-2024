from functools import lru_cache
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.spatial import distance

from data_manager.geotiff import GeotiffRaster
from data_manager.shapefile import ShapefilePoints


class Dataframes(NamedTuple):
    raster: pd.DataFrame
    shapefile: pd.DataFrame


class Merger:
    def __init__(self, raster: GeotiffRaster, shapefile: ShapefilePoints):
        self._raster = raster
        self._shapefile = shapefile

    @lru_cache(maxsize=None)
    def to_pandas(self):
        raster_df = self._raster.to_pandas()
        shapefile_df = self._shapefile.to_pandas()
        return Dataframes(raster_df, shapefile_df)

    @lru_cache(maxsize=None)
    def merged_data(self):
        raster_df, shapefile_df = self.to_pandas()

        shapefile_row = shapefile_df[[self._shapefile.X, self._shapefile.Y]].iloc[0].to_numpy()
        shapefile_row = np.expand_dims(shapefile_row, axis=0)
        coordinates = raster_df[[self._shapefile.X, self._shapefile.Y]].to_numpy()
        dist = distance.cdist(coordinates, shapefile_row)

        a = np.argsort(dist, axis=0)[0:5]
        b = np.argpartition(dist, 5, axis=0)[0:5]
        print(dist[a])
        print(np.sort(dist[b], axis=0))


if __name__ == "__main__":
    from configs import configs

    file = configs.SHAPEFILES_DIR / "oznake.shp"
    shapefile = ShapefilePoints(file)

    file = (
        configs.MUTISPECTRAL_DIR
        / "2022_06_15__eko_ecobreed/Ecobreed_krompir_EKO_15_06_2022_transparent_reflectance_blue_modified.tif"
    )
    raster = GeotiffRaster(file)

    merger = Merger(raster, shapefile)
    merger.merged_data()
