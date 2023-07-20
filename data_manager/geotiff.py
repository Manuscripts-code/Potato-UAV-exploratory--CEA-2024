from functools import lru_cache

import numpy as np
from rioxarray._io import open_rasterio


class GeotiffRaster:
    DATA_COLUMN_NAME = "reflectance"

    def __init__(self, file_path):
        self._raster = self._init_geotiff_raster(file_path)

    def __repr__(self):
        return repr(self._raster)

    def __str__(self):
        return str(self._raster)

    def __len__(self):
        return len(self._raster)

    def _init_geotiff_raster(self, file_path):
        raster = open_rasterio(file_path)
        raster = raster.squeeze().drop("spatial_ref").drop("band")
        raster.name = self.DATA_COLUMN_NAME
        return raster

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
    def file_path(self):
        return self._raster.encoding["source"]

    @property
    def shape(self):
        return self._raster.shape


if __name__ == "__main__":
    from configs import configs

    file = (
        configs.MUTISPECTRAL_DIR
        / "2022_06_15__eko_ecobreed/Ecobreed_krompir_EKO_15_06_2022_transparent_reflectance_blue_modified.tif"
    )

    raster = GeotiffRaster(file)
    print(raster)
