# Ecobreed-potato

## Introduction

place abstract here

## Getting started


### Environment setup

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

### Dataset

# Run the script
python3 main.py






## Bugs:
(findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial, Liberation Sans, Bitstream Vera Sans, sans-serif)

$ sudo apt install msttcorefonts -qq
$ rm ~/.cache/matplotlib -rf