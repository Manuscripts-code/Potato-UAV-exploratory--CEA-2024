import matplotlib.pyplot as plt
from rioxarray._io import open_rasterio

from configs import configs
from data_manager.geotiffs import GeotiffRaster

if __name__ == "__main__":
    file = (
        configs.MUTISPECTRAL_DATA_DIR
        / "2022_06_15__eko_ecobreed/Ecobreed_krompir_EKO_15_06_2022_transparent_reflectance_blue_modified.tif"
    )

    raster = GeotiffRaster(file)
    pass
