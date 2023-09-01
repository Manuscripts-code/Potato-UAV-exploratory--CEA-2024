# ecobreed-potato

```sh
How to install:

- python 3.10
- pip install -r requirements.txt
- install gdal: 

follow: https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/
in essence: install binary from here: https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
and run: python -m pip install GDAL-3.4.3-cp310-cp310-win_amd64.whl
```

ZenML:

# Initialize ZenML

zenml init

# Start the ZenServer to enable dashboard access

zenml up

# Register required ZenML stack

zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml model-registry register mlflow_registry --flavor=mlflow

# Register a new stack with the new stack components

zenml stack register ecobreed -a default -o default -d mlflow_deployer -e mlflow_tracker -r mlflow_registry --set

# To clean up, simply spin down the ZenML server

zenml down
zenml clean

```sh
## RUN MLFLOW USI
```

```sh
mlflow ui --backend-store-uri file:///C:\\Users\\janezla\\AppData\\Roaming\\zenml\\local_stores\\508e55dd-c0c3-4bb9-8e2b-25e97c57bf21\\mlruns   
```
