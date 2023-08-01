from enum import Enum


class MultispectralEnum(Enum):
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
