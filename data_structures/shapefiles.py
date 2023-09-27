from pathlib import Path

import geopandas as gpd
import pandas as pd


class PointsShapefile:
    X = "x"
    Y = "y"

    def __init__(self, shapefile: gpd.GeoDataFrame, *, name="", path=""):
        self._shapefile = shapefile
        self._name = name
        self._path = Path(path)

    def __str__(self):
        return f"<PointsShapefile(shape={self._shapefile.shape})>"

    @staticmethod
    def _make_new_coordinates_columns(shapefile):
        if shapefile.geometry.iloc[0].geom_type == "Point":
            shapefile[PointsShapefile.X] = shapefile.geometry.x
            shapefile[PointsShapefile.Y] = shapefile.geometry.y
        elif shapefile.geometry.iloc[0].geom_type == "MultiPoint":
            shapefile[PointsShapefile.X] = shapefile.geometry.centroid.x
            shapefile[PointsShapefile.Y] = shapefile.geometry.centroid.y
        return shapefile

    @staticmethod
    def _init_shapefile_points(file_path):
        shapefile = gpd.read_file(file_path)
        shapefile = PointsShapefile._make_new_coordinates_columns(shapefile)
        return shapefile

    @classmethod
    def from_path(cls, file_path):
        shapefile = cls._init_shapefile_points(file_path)
        kwargs = {"name": Path(file_path).stem, "path": str(file_path)}
        return cls(shapefile, **kwargs)

    def to_pandas(self):
        return pd.DataFrame(self._shapefile)

    @property
    def file(self):
        return self._shapefile

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name


if __name__ == "__main__":
    from configs import paths

    file = paths.PATHS_SHAPEFILES["eko"]["measured"]
    shapefile = PointsShapefile.from_path(file)
    print(shapefile)
