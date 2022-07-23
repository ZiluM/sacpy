import xarray as xr
import numpy as np
from scipy import signal
import scipy.stats as sts
from .LinReg import LinReg, MultLinReg


def _deTrend(array: np.ndarray):
    """ Remove linear trend
    Args:
        array (np.ndarray): shape = (time,*number)
            Array of linear trends that need to be removed
    Returns:
        array (np.ndarray): shape = (time,*number)
    """
    array = np.array(array)
    # Remove the missing measured value
    flag = np.isnan(array).sum(axis=0) != 0
    flags = np.repeat(flag[np.newaxis, ...], axis=0, repeats=array.shape[0])
    # Set the missing value to 0
    array[flags] = 0
    # Remove linear trend
    res = signal.detrend(array, axis=0)
    # Set the missing value to Nan
    res[flags] = np.NAN
    return res


def get_anom(DaArray: xr.DataArray, method=0):
    """ Get climate data anomaly
    Args:
        DaArray (xr.DataArray): shape = (time, *number) original Dataarray
        method (int, optional): method of getting anomaly.
            1 is Minus the multi-year average of the corresponding month
            2 is Remove the linear trend of the corresponding month
             Defaults to 0.

    Returns:
        anom (xr.DataArray): climate data anomaly
    """
    if type(DaArray) != xr.DataArray:
        raise ValueError("'xr.DataArray' input is required, not the %s" % (type(DaArray)))
    if method == 0:
        anom = DaArray.groupby("time.month") - DaArray.groupby("time.month").mean()
    if method == 1:
        anom = DaArray.groupby("time.month").map(_deTrend)

    return anom
