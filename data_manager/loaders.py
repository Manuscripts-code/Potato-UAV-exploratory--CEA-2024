from itertools import product

from configs import configs, specific_paths
from data_manager.geotiffs import MultiGeotiffRaster
from data_manager.mergers import MultiRasterPointsMerger, RasterPointsMerger
from data_manager.shapefiles import PointsShapefile


class MultispectralLoader:
    ROOT = "multispectral"
    DATES = "imagings_dates"
    TREATMENTS = "treatments"
    CHANNELS = "channels"
    LOCATION_TYPE = "location_type"

    def __init__(self):
        self.rasters_paths = specific_paths.PATHS_MULTISPECTRAL_IMAGES
        self.shapefiles_paths = specific_paths.PATHS_SHAPEFILES
        self.cfg = configs.CONFIGS_TOML[self.ROOT]

        self._mergers = None
        self._multi_merger = None

    def __str__(self):
        return f"<MultispectralLoader object with {len(self._mergers)} mergers>"

    def load(self):
        self.load_mergers()
        self.run_merges()
        self.final_merge()
        return self

    def load_mergers(self):
        dates, treatments, channels, location_type = self._get_configs()
        self._mergers = []
        for date, treatment in product(dates, treatments):
            merger = self._create_merger(date, treatment, channels, location_type)
            self._mergers.append(merger)
        return self

    def run_merges(self):
        if self._mergers is None:
            raise ValueError("Mergers are not loaded.")
        self._multi_merger = MultiRasterPointsMerger(self._mergers)
        self._multi_merger.run_merges()
        return self

    def final_merge(self):
        # TODO: implement
        pass
        return self

    def _get_configs(self):
        try:
            dates = self.cfg[self.DATES]
            treatments = self.cfg[self.TREATMENTS]
            channels = self.cfg[self.CHANNELS]
            location_type = self.cfg[self.LOCATION_TYPE]
        except KeyError:
            raise KeyError("Missing key in config file.")
        return dates, treatments, channels, location_type

    def _create_merger(self, date, treatment, channels, location_type):
        base_path = self.rasters_paths[treatment][date]
        paths = {channel: base_path[channel] for channel in channels}
        raster = MultiGeotiffRaster.from_paths(paths)
        raster.set_name(date)
        path_shape = self.shapefiles_paths[treatment][location_type]
        shapefile = PointsShapefile.from_path(path_shape)
        return RasterPointsMerger(raster, shapefile)

    @property
    def mergers(self):
        return self._multi_merger.mergers

    @property
    def merged_dfs(self):
        return self._multi_merger.merged_dfs


if __name__ == "__main__":
    loader = MultispectralLoader().load()
    pass
