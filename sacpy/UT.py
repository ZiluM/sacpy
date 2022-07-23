import numpy as np


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
