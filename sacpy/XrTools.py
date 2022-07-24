import xarray as xr
import numpy as np
from scipy import signal


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
            0 is Minus the multi-year average of the corresponding month
            1 is Remove the linear trend of the corresponding month
             Defaults to 0.

    Returns:
        anom (xr.DataArray): climate data anomaly
    """
    if type(DaArray) != xr.DataArray:
        raise ValueError("'xr.DataArray' input is required, not the %s" % (type(DaArray)))
    if not "time" in list(DaArray.coords.keys()):
        raise ValueError("DaArray must have coords 'time' !")
    if method == 0:
        anom = DaArray.groupby("time.month") - DaArray.groupby("time.month").mean()
    if method == 1:
        anom = DaArray.groupby("time.month").map(_deTrend)

    return anom


def spec_moth_dat(DaArray: xr.DataArray, months: list):
    """ get specific month data 
    Args:
        DaArray (xr.DataArray): shape = (time, *number) original Dataarray
        months (list): get data from specific month

    Raises:
        ValueError: xr.DataArray' input is required, not the %s

    Returns:
        xr.DataArray: data in specific month
    """
    if type(DaArray) != xr.DataArray:
        raise ValueError("'xr.DataArray' input is required, not the %s" % (type(DaArray)))
    if not "time" in list(DaArray.coords.keys()):
        raise ValueError("DaArray must have coords 'time' !")
    time = DaArray.time
    time_label = time.dt.month == months[0]
    for i in range(len(months)):
        time_label = time_label | (time.dt.month == months[i])
    return DaArray[time_label]


def spec_moth_yrmean(DaArray: xr.DataArray, months: list):
    """ get specific month data and average them in each year

    Args:
        DaArray (xr.DataArray): shape = (time, *number) original Dataarray
        months (list): get data from specific month

    Returns:
        xr.DataArray: data in specific month and average them in each year
    """
    # get data in specifc month
    smd = spec_moth_dat(DaArray, months)
    # get time
    time = smd.time
    # get month[0]
    month_start = time.dt.month[0].item()
    # get month[-1]
    month_end = time.dt.month[-1].item()
    # get cycle data (cycle is period in months)
    ed_idx = -(len(months) - months.index(month_end) - 1)
    ed_idx = None if ed_idx == 0 else ed_idx
    st_idx = len(months) - months.index(month_start)
    smd_cyc = smd[st_idx:ed_idx]
    # get year list
    year_list = smd_cyc[::len(months)].time.dt.year
    # sum
    smd_sum = 0
    for i in range(len(months)):
        smd_i = smd_cyc[i::len(months)]
        smd_i['time'] = year_list
        smd_sum = smd_sum + smd_i
    # get mean
    smd_m = smd_sum / len(months)
    # return
    return smd_m
