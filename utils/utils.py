import json
import pickle
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import numpy as np
import spectral as sp
import toml


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
    return dirname


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_toml(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return toml.load(handle)


def write_txt(content, fname):
    fname = Path(fname)
    with fname.open("w") as handle:
        handle.write(content)


def write_pickle(content, fname):
    fname = Path(fname)
    with open(fname, "wb") as f:
        pickle.dump(content, f)


def read_pickle(fname):
    fname = Path(fname)
    with open(fname, "rb") as f:
        return pickle.load(f)


def save_image(file_name, image, metadata=None):
    sp.envi.save_image(file_name, image, dtype=np.float32, metadata=metadata, force=True)


def init_object(options_dict, class_name, *args, **kwargs):
    if class_name not in options_dict:
        raise ValueError(f"Key {class_name} not present. Possible options: {options_dict.keys()}")
    return options_dict[class_name](*args, **kwargs)
