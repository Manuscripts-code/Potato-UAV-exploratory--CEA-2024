# Ecobreed-potato

## 🔍 Introduction

place abstract here

## 📚 Getting started


### ⚙️ Environment setup

Setup tested on Ubuntu Linux machine with python 3.10.

1) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install packages into virtual environment:
```bash
pip3 install -r requirements.txt
```

3) Go to root of the repository and run the following to create .env file:
```bash
echo "PYTHONPATH=$(pwd)" > .env
```

4) Initialize zenML tool, by running the following:

```bash
# Initialize ZenML
zenml init

# Start the ZenServer to enable dashboard access
zenml up

# Register required ZenML stack components
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml model-registry register mlflow_registry --flavor=mlflow

# Register a new stack with the new stack components
zenml stack register ecobreed -a default\
                                      -o default\
                                      -d mlflow_deployer\
                                      -e mlflow_tracker\
                                      -r mlflow_registry\
                                      --set
```

5) (Required only after system restart) Run to re-initialize:
```bash
# Disconnect first if you closed PC
zenml disconnect

# Start the ZenServer to enable dashboard access
zenml up
```

### 📅 Dataset

Download data from [Zenodo](add-link) and decompress (unzip) to `data` directory.
After successful extraction, the directory structure should resemble the following:

```
📁 data
├── 📜.gitkeep
├── 📁 multispectral_images
│   ├── 📁 2022_07_20__konv_ecobreed
│   │   ├── 🖼️ Ecobreed_krompir_konv_20_07_2022_transparent_reflectance_nir_modified.tif
│   │   ├── 🖼️ Ecobreed_krompir_konv_20_07_2022_transparent_reflectance_red_modified.tif
|   |   └── 🖼️ ...
│   ├── 📁 2022_07_20__eko_ecobreed
│   │   ├── 🖼️ Ecobreed_krompir_eko_20_07_2022_transparent_reflectance_nir_modified.tif
│   │   ├── 🖼️ Ecobreed_krompir_eko_20_07_2022_transparent_reflectance_red_modified.tif
|   |   └── 🖼️ ...
│   └── 📁 ...
├── 📁 shapefiles
│   ├── 📄 potato_locations_eko.shp
│   ├── 📄 potato_locations_konv.shp
│   ├── 📄 potato_measured_locations_eko.shp
│   ├── 📄 potato_measured_locations_konv.shp
│   └── 📄 ...
└── 📁 measurements
    ├── 📊 Alternaria_ocenjevanje1_Ecobreed_krompir_2022.xlsx
    ├── 📊 LICOR_Ecobreed_krompir_2022.xlsx
    ├── 📊 SPAD_Ecobreed_krompir_2022.xlsx
    └── 📊 Varieties_grouped_Ecobreed_krompir_2022.xlsx
```

The `data` directory contains three types of data: multispectral images, shapefiles and measurements:

- **multispectral_images**: This directory contains raster images captured by a multispectral camera. The images are organized into subdirectories by date and treatment type (conventional-konv or ecological-eko). Each image file, such as `Ecobreed_krompir_konv_20_07_2022_transparent_reflectance_nir_modified.tif`, represents a specific spectral band i.e., near-infrared of the multispectral image.

- **shapefiles**: This directory contains shapefiles (.shp), which are used to store the geographic coordinates of potato plants. Files like `potato_locations_eko.shp` and `potato_locations_konv.shp` contain the locations of all potato plants for ecological and conventional treatments, respectively. Meanwhile, files like `potato_measured_locations_eko.shp` and `potato_measured_locations_konv.shp` contain the locations of potato plants where physiological measurements were taken.

- **measurements**: This directory contains Excel (.xlsx) files with various ground measurements taken from the potato plants. Each file represents a different type of measurement, such as `Alternaria_ocenjevanje1_Ecobreed_krompir_2022.xlsx` for Alternaria scoring, `LICOR_Ecobreed_krompir_2022.xlsx` for LICOR measurements, `SPAD_Ecobreed_krompir_2022.xlsx` for SPAD measurements, and `Varieties_grouped_Ecobreed_krompir_2022.xlsx` for grouped variety data.

# Run the script
python3 main.py






## Bugs:
(findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial, Liberation Sans, Bitstream Vera Sans, sans-serif)

$ sudo apt install msttcorefonts -qq
$ rm ~/.cache/matplotlib -rf

## Issues

This project was initially developed by Janez Lapajne. If you have any questions or encounter any issues, post an issue on Github.