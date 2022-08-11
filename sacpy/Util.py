import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def convert_lon(lon: np.ndarray):
    """ Convert longitude to the range of [-180, 180]

    Args:
        lon (np.ndarray): lon in [0,360]

    Returns:
        lon (np.ndarray): lon in [-180 , 180]
    """
    return (lon + 180) % 360 - 180


def reverse_lon(lon: np.ndarray):
    """ Convert longitude to the range of [0, 360]

    Args:
        lon (np.ndarray): lon in [-180,180]

    Returns:
        lon (np.ndarray): lon in [0, 360]
    """
    lon1 = np.copy(lon)
    lon1[lon < 0] += 360
    return lon1


def wrap_lon_to_180(da, lon='lon'):
    '''
    This code copy from https://github.com/Yefee/xMCA
    Wrap longitude coordinates of DatArray to -180..179
    
    Parameters
    ----------
    da : DatArray
        object with longitude coordinates
    lon : string
        name of the longitude ('lon', 'longitude', ...)

    Returns
    -------
    wrapped : Dataset
        Another dataset array wrapped around.
    '''

    # wrap 0..359 to -180..179
    da = da.assign_coords(lon=(((da[lon] + 180) % 360) - 180))

    # sort the data
    return da.sortby(lon)


def sig_scatter(ax, x: np.ndarray, y: np.ndarray, p: np.ndarray, threshold: float = 0.05, **kwargs):
    """ significance plot 1D

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): axes to plot
        x (np.ndarray): 1D
        y (np.ndarray): 1D
        p (np.ndarray): 1D
        threshold (float, optional): threshold p value to plot. Defaults to 0.05.
    """
    # copy
    x1, y1, p1 = list(map(np.copy, [x, y, p]))
    x1[p1 >= threshold] = np.NAN
    y1[p1 >= threshold] = np.NAN
    # scatter
    sc = ax.scatter(x1, y1, **kwargs)
    return sc


def _correct_type(data, dtype=np.float64):
    """ 

    Args:
        data (_type_): _description_
        dtype (_type_, optional): _description_. Defaults to np.float64.

    Returns:
        _type_: _description_
    """
    if not isinstance(data, np.ndarray):
        data1 = np.array(data, dtype=dtype)
    else:
        data1 = data
    return data1