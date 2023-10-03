# ecobreed-potato

```sh
How to install:

- python 3.10
- python3 -m venv env
- source env/bin/activate
- pip3 install -r requirements.txt

ZenML:

# Disconnect first if you closed PC
zenml disconnect

# Initialize ZenML
zenml init

# Start the ZenServer to enable dashboard access
zenml up

# Register required ZenML stack
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

# Run the script
python3 main.py

# To clean up, simply spin down the ZenML server

zenml down
zenml clean


# Visualization error:
(findfont: Generic family 'sans-serif' not found because none of the following families were found: Arial, Liberation Sans, Bitstream Vera Sans, sans-serif)

$ sudo apt install msttcorefonts -qq
$ rm ~/.cache/matplotlib -rf