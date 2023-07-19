import numpy as np
from rioxarray._io import open_rasterio


class GeotiffRaster:
    def __init__(self, file_path):
        self._raster = self.init_geotiff_raster(file_path)

    def init_geotiff_raster(self, file_path):
        raster = open_rasterio(file_path)
        raster = raster.squeeze().drop("spatial_ref").drop("band")
        raster.name = "reflectance"
        return raster

    def to_numpy(self, set_nodata_to_zero=True):
        array = self._raster.to_numpy()
        if set_nodata_to_zero:
            array[np.where(array == self._raster._FillValue)] = 0
        return array

    def to_dataframe(self):
        return self._raster.to_dataframe().reset_index()
