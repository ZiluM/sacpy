import xarray as xr
import os

current_dir = os.path.split(os.path.realpath(__file__))[0]

def load_sst():
    """ load example sst data
    """
    return xr.open_dataset(current_dir + "/data/example/HadISST_sst_5x5.nc")


def load_10mwind():
    """ load 10 m wind
    """

    return xr.open_dataset(current_dir + "/data/example/NCEP_wind10m_5x5.nc")
