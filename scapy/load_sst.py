import xarray as xr
import os

current_dir = os.path.split(os.path.realpath(__file__))[0]


def load_sst():
    return xr.open_dataset(current_dir + "/example/data/HadISST_sst_2x2.nc")
