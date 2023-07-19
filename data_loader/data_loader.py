import matplotlib.pyplot as plt
import numpy as np
from rioxarray._io import open_rasterio

from configs import configs

if __name__ == "__main__":
    file = (
        configs.MUTISPECTRAL_DATA_DIR
        / "2022_06_15__eko_ecobreed/Ecobreed_krompir_EKO_15_06_2022_transparent_reflectance_blue_modified.tif"
    )

    # raster = rs.open(file)
    raster = open_rasterio(file)
    array = raster.to_numpy()  # type: ignore
    array[np.where(array < 0)] = 0

    plt.imshow(array[0])
    plt.show()
