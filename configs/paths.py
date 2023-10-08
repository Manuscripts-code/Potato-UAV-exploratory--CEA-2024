import json
from pathlib import Path
from types import SimpleNamespace

from configs import configs

PATHS_MULTISPECTRAL_IMAGES = {
    "eko": {
        "2022_06_15": {
            "blue": str(
                configs.MUTISPECTRAL_DIR
                / "2022_06_15__eko_ecobreed/Ecobreed_krompir_EKO_15_06_2022_transparent_reflectance_blue_modified.tif"
            ),
            "green": str(
                configs.MUTISPECTRAL_DIR
                / "2022_06_15__eko_ecobreed/Ecobreed_krompir_EKO_15_06_2022_transparent_reflectance_green_modified.tif"
            ),
            "nir": str(
                configs.MUTISPECTRAL_DIR
                / "2022_06_15__eko_ecobreed/Ecobreed_krompir_EKO_15_06_2022_transparent_reflectance_nir_modified.tif"
            ),
            "red_edge": str(
                configs.MUTISPECTRAL_DIR
                / "2022_06_15__eko_ecobreed/Ecobreed_krompir_EKO_15_06_2022_transparent_reflectance_red edge_modified.tif"
            ),
            "red": str(
                configs.MUTISPECTRAL_DIR
                / "2022_06_15__eko_ecobreed/Ecobreed_krompir_EKO_15_06_2022_transparent_reflectance_red_modified.tif"
            ),
        },
        "2022_07_11": {
            "blue": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_11__eko_ecobreed/Ecobreed_krompir_eko_11_07_2022_transparent_reflectance_blue_modified.tif"
            ),
            "green": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_11__eko_ecobreed/Ecobreed_krompir_eko_11_07_2022_transparent_reflectance_green_modified.tif"
            ),
            "nir": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_11__eko_ecobreed/Ecobreed_krompir_eko_11_07_2022_transparent_reflectance_nir_modified.tif"
            ),
            "red_edge": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_11__eko_ecobreed/Ecobreed_krompir_eko_11_07_2022_transparent_reflectance_red edge_modified.tif"
            ),
            "red": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_11__eko_ecobreed/Ecobreed_krompir_eko_11_07_2022_transparent_reflectance_red_modified.tif"
            ),
        },
        "2022_07_20": {
            "blue": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_20__eko_ecobreed/Ecobreed_krompir_eko_20_07_2022_transparent_reflectance_blue_modified.tif"
            ),
            "green": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_20__eko_ecobreed/Ecobreed_krompir_eko_20_07_2022_transparent_reflectance_green_modified.tif"
            ),
            "nir": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_20__eko_ecobreed/Ecobreed_krompir_eko_20_07_2022_transparent_reflectance_nir_modified.tif"
            ),
            "red_edge": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_20__eko_ecobreed/Ecobreed_krompir_eko_20_07_2022_transparent_reflectance_red edge_modified.tif"
            ),
            "red": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_20__eko_ecobreed/Ecobreed_krompir_eko_20_07_2022_transparent_reflectance_red_modified.tif"
            ),
        },
    },
    "konv": {
        "2022_06_15": {
            "blue": str(
                configs.MUTISPECTRAL_DIR
                / "2022_06_15__konv_ecobreed/Ecobreed_krompir_KONV_15_06_2022_transparent_reflectance_blue_modified.tif"
            ),
            "green": str(
                configs.MUTISPECTRAL_DIR
                / "2022_06_15__konv_ecobreed/Ecobreed_krompir_KONV_15_06_2022_transparent_reflectance_green_modified.tif"
            ),
            "nir": str(
                configs.MUTISPECTRAL_DIR
                / "2022_06_15__konv_ecobreed/Ecobreed_krompir_KONV_15_06_2022_transparent_reflectance_nir_modified.tif"
            ),
            "red_edge": str(
                configs.MUTISPECTRAL_DIR
                / "2022_06_15__konv_ecobreed/Ecobreed_krompir_KONV_15_06_2022_transparent_reflectance_red edge_modified.tif"
            ),
            "red": str(
                configs.MUTISPECTRAL_DIR
                / "2022_06_15__konv_ecobreed/Ecobreed_krompir_KONV_15_06_2022_transparent_reflectance_red_modified.tif"
            ),
        },
        "2022_07_11": {
            "blue": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_11__konv_ecobreed/Ecobreed_krompir_konv_11_07_2022_transparent_reflectance_blue_modified.tif"
            ),
            "green": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_11__konv_ecobreed/Ecobreed_krompir_konv_11_07_2022_transparent_reflectance_green_modified.tif"
            ),
            "nir": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_11__konv_ecobreed/Ecobreed_krompir_konv_11_07_2022_transparent_reflectance_nir_modified.tif"
            ),
            "red_edge": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_11__konv_ecobreed/Ecobreed_krompir_konv_11_07_2022_transparent_reflectance_red edge_modified.tif"
            ),
            "red": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_11__konv_ecobreed/Ecobreed_krompir_konv_11_07_2022_transparent_reflectance_red_modified.tif"
            ),
        },
        "2022_07_20": {
            "blue": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_20__konv_ecobreed/Ecobreed_krompir_konv_20_07_2022_transparent_reflectance_blue_modified.tif"
            ),
            "green": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_20__konv_ecobreed/Ecobreed_krompir_konv_20_07_2022_transparent_reflectance_green_modified.tif"
            ),
            "nir": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_20__konv_ecobreed/Ecobreed_krompir_konv_20_07_2022_transparent_reflectance_nir_modified.tif"
            ),
            "red_edge": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_20__konv_ecobreed/Ecobreed_krompir_konv_20_07_2022_transparent_reflectance_red edge_modified.tif"
            ),
            "red": str(
                configs.MUTISPECTRAL_DIR
                / "2022_07_20__konv_ecobreed/Ecobreed_krompir_konv_20_07_2022_transparent_reflectance_red_modified.tif"
            ),
        },
    },
}

PATHS_SHAPEFILES = {
    "eko": {
        "all": str(configs.SHAPEFILES_DIR / "potato_locations_eko.shp"),
        "measured": str(configs.SHAPEFILES_DIR / "potato_measured_locations_eko.shp"),
    },
    "konv": {
        "all": str(configs.SHAPEFILES_DIR / "potato_locations_konv.shp"),
        "measured": str(configs.SHAPEFILES_DIR / "potato_measured_locations_konv.shp"),
    },
}

PATHS_MEASUREMENTS = {
    "SPAD": (
        "SPAD_mean",
        str(configs.MEASUREMENTS_DIR / "SPAD_Ecobreed_krompir_2022.xlsx"),
    ),
    "Alternaria": (
        "Procent",
        str(configs.MEASUREMENTS_DIR / "Alternaria_ocenjevanje1_Ecobreed_krompir_2022.xlsx"),
    ),
}
