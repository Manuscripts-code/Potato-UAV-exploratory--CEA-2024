from functools import lru_cache
from pathlib import Path

import numpy as np
from rioxarray._io import DataArray, open_rasterio


class GeotiffRaster:
    X = "x"
    Y = "y"
    DATA_COLUMN_NAME = "reflectance"

    def __init__(self, raster: DataArray):
        self._raster = raster

    def __str__(self):
        return f"<GeotiffRaster(shape={self._raster.shape})>"

    def __len__(self):
        return len(self._raster)

    @staticmethod
    def _init_geotiff_raster(file_path):
        raster = open_rasterio(file_path)
        raster = raster.squeeze().drop("spatial_ref").drop("band")
        raster.name = GeotiffRaster.DATA_COLUMN_NAME
        return raster

    @classmethod
    def from_path(cls, file_path):
        raster = cls._init_geotiff_raster(file_path)
        return cls(raster)

    @lru_cache(maxsize=None)
    def to_numpy(self, set_nodata_to_zero=True):
        array = self._raster.to_numpy()
        if set_nodata_to_zero:
            array[np.where(array == self.nodata)] = 0
        return array

    @lru_cache(maxsize=None)
    def to_pandas(self):
        df = self._raster.to_dataframe().reset_index()
        df = df[df[self.DATA_COLUMN_NAME] != self.nodata]
        return df.reset_index(drop=True)

    @property
    def nodata(self):
        return self._raster._FillValue

    @property
    def file(self):
        return self._raster

    @property
    def path(self):
        return Path(self._raster.encoding["source"])

    @property
    def name(self):
        return self.path.stem

    @property
    def shape(self):
        return self._raster.shape


class GeotiffRasterMulti:
    def __init__(self, rasters: list[GeotiffRaster] = None):
        self._rasters = [] if rasters is None else rasters

    def __str__(self):
        return f"<GeotiffRasterMulti object with {len(self._rasters)} rasters>"

    def __len__(self):
        return len(self._rasters)

    def __getitem__(self, index):
        return self._rasters[index]

    def add_raster(self, raster: GeotiffRaster):
        self._rasters.append(raster)

    @classmethod
    def from_paths(cls, file_paths: list[str]):
        rasters = [GeotiffRaster.from_path(path) for path in file_paths]
        return cls(rasters)


if __name__ == "__main__":
    from configs import specific_paths

    base_path = specific_paths.PATHS_MULTISPECTRAL_IMAGES["eko"]["2022_06_15"]
    blue = base_path["blue"]
    green = base_path["green"]

    raster = GeotiffRasterMulti.from_paths([blue, green])
    print(raster)
    for channel in raster:
        print(channel)
