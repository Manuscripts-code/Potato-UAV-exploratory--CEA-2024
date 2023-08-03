from enum import Enum


class GeneralConfigEnum(Enum):
    ROOT = "general"
    NUM_CLOSEST_POINTS = "num_closest_points"

    def __str__(self):
        return self.value


class MultispectralConfigEnum(Enum):
    # settings coupled to /configs/specific/*.toml files and original labels in shapefiles
    ROOT = "multispectral"
    DATES = "imagings_dates"
    TREATMENTS = "treatments"
    CHANNELS = "channels"
    LOCATION_TYPE = "location_type"
    COLUMNS_SLO = ["Blok", "Rastlina", "Sorta"]
    COLUMNS_ENG = ["blocks", "plants", "varieties"]

    def __str__(self):
        return self.value


class SamplerConfigEnum(Enum):
    ROOT = "sampler"
    RANDOM_STATE = "random_state"
    SPLITTER = "splitter"
    SHUFFLE = "shuffle"
    SPLIT_SIZE_VAL = "split_size_val"
    SPLIT_SIZE_TEST = "split_size_test"

    def __str__(self):
        return self.value


class FormatterConfigEnum(Enum):
    ROOT = "formatter"
    FORMATTER = "formatter"
    CLASSES = "classes"

    def __str__(self):
        return self.value
