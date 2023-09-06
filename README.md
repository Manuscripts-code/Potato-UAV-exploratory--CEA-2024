# ecobreed-potato

```sh
How to install:

- python 3.10
- python3 -m venv env
- source env/bin/activate
- pip install -r requirements.txt

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
