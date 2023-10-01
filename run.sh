#!/bin/bash

python3 main.py --toml-config-file clf/DummyClassifier.toml;
python3 main.py --toml-config-file clf/XGBClassifier.toml;
python3 main.py --toml-config-file clf/Feat_XGBClassifier.toml;

python3 main.py --toml-config-file reg/DummyRegressor.toml;
python3 main.py --toml-config-file reg/XGBRegressor.toml;
python3 main.py --toml-config-file reg/Feat_XGBRegressor.toml;

python3 main.py --toml-config-file test_classification.toml;
python3 main.py --toml-config-file test_regression.toml;