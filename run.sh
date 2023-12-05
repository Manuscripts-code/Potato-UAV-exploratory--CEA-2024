#!/bin/bash

# python3 main.py --toml-config-file clf/alternaria.toml --results;
# python3 main.py --toml-config-file clf/alternaria_b.toml --results;

python3 main.py --toml-config-file reg/E.toml;
python3 main.py --toml-config-file reg/ETR.toml;
python3 main.py --toml-config-file reg/gsw.toml;
python3 main.py --toml-config-file reg/PhiPS2.toml;
python3 main.py --toml-config-file reg/SPAD.toml;
