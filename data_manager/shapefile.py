from functools import lru_cache

import geopandas as gpd
import pandas as pd


class ShapefilePoints:
    X = "x"
    Y = "y"

    def __init__(self, file_path: str):
        self._shapefile = self._init_shapefile_points(file_path)

    def _init_shapefile_points(self, file_path):
        shapefile = gpd.read_file(file_path)
        shapefile = self._make_new_coordinates_columns(shapefile)
        return shapefile

    def _make_new_coordinates_columns(self, shapefile):
        x = lambda row: row.geometry.x
        y = lambda row: row.geometry.y
        shapefile[self.X] = shapefile.apply(x, axis=1)
        shapefile[self.Y] = shapefile.apply(y, axis=1)
        return shapefile

    @lru_cache(maxsize=None)
    def to_pandas(self):
        return pd.DataFrame(self._shapefile)


if __name__ == "__main__":
    from configs import configs

    file = configs.SHAPEFILES_DIR / "oznake.shp"
    shapefile = ShapefilePoints(file)
    print(shapefile)
