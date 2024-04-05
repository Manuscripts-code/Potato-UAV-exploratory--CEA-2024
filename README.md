# Ecobreed-potato

### 🔍 Introduction

place abstract here



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

4) Initialize ZenML tool, by running the following:

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


### 📚 How to use

For optimal control and customization, it is recommended to execute the experiments via the `main.py` python script, which offers several command-line options:

- `--config` or `-c`: This option allows to specify which pipeline to run. The available choices are `train`, `test`, and `all`. If not specified, the script will execute `all` pipelines by default.

  Example usage: `python3 main.py --config train`

- `--toml-config-file` or `-t`: This option allows to specify a TOML configuration file located in the 'configs/specific/**/*.toml' directory. If not specified, the script will use the default TOML configuration file.

  Example usage: `python3 main.py --toml-config-file clf/alternaria_b.toml`

- `--results` or `-r`: This is a flag option. If set, the script will produce results, i.e., calculate metrics and save artifacts.

  Example usage: `python3 main.py --results`

The options could be combined as needed. For example, to run the train pipeline with a specific TOML configuration file and produce results, the following could be used:

```bash
python3 main.py --config train --toml-config-file reg/E.toml --results
```

Alternatively, specific lines could be uncommented in the `run.sh` script and the script then executed to perform batch tasks processing sequentially.

By including the `--results` flag, some results will be automatically generated. For additional results, plots, and classification metrics, utilization of scripts and notebooks found in the `notebooks` directory is required.


### 🌟 Troubleshooting:


- **Error** findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial, Liberation Sans, Bitstream Vera Sans, sans-serif

This can be resolved by installing the `msttcorefonts` package and clearing the Matplotlib cache. The following should resolve the issue:

```bash
sudo apt install msttcorefonts -qq
rm -rf ~/.cache/matplotlib
```


### 📬 Contact

This project was initially developed by Janez Lapajne. If you have any questions or encounter any other problem, feel free to post an issue on Github.