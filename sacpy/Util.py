import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import copy


def gen_dataarray(data, coords,dims):
    return xr.DataArray(data=data, coords=coords, dims=dims)

def rewapper(data,origin_dataarray,drop_dims=None):
    coords = copy.deepcopy(origin_dataarray.coords)
    dims = list(origin_dataarray.coords.dims)
    if drop_dims is not None:
        if type(drop_dims) is str:
            drop_dims = [drop_dims]
        for dim in drop_dims:
            del coords[dim]
            # del coords.dims[dim]
            dims.remove(dim)
        # coords.dims = tuple(dims)

    # new_coords = xr.DataArray.coords(coords=coords, dims=dims)



    print(coords.dims)

    return gen_dataarray(data,coords=coords,dims=dims)


def convert_lon(lon: np.ndarray):
    """ Convert longitude to the range of [-180, 180] (from xarray)

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


def sig_scatter(ax, x: np.ndarray, y: np.ndarray, p: np.ndarray,
                threshold: float = 0.05, **kwargs):
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


def _correct_type0(data, dtype=np.float64):
    """ 

    Args:
        data (any): data to correct
        dtype (any, optional): data format. Defaults to np.float64.

    Returns:
        format data
    """
    if not isinstance(data, np.ndarray):
        if isinstance(data, xr.DataArray):
            data1 = data.to_numpy()
            # data = data.drop_duplicates()
            coords = data.coords
            dims = data.dims
        else:
            try:
                data1 = np.array(data)
                dims = None
                coords = None
            except:
                raise TypeError(f"Can't convert {type(data)} to {np.ndarray}'")
    else:
        data1 = data
        dims = None
        coords = None
    return data1, dims, coords


def _correct_type(data, dtype=np.float64):
    """ 

    Args:
        data (any): data to correct
        dtype (any, optional): data format. Defaults to np.float64.

    Returns:
        format data
    """
    if not isinstance(data, np.ndarray):
        if isinstance(data, xr.DataArray):
            data1 = data.to_numpy()
        else:
            try:
                data1 = np.array(data)
            except:
                raise TypeError(f"Can't convert {type(data)} to {np.ndarray}'")
    else:
        data1 = data
    return data1


def gradient_array(data: np.ndarray, dim: int, method: int = 0):
    """ gradient array in dim

    Args:
        data (np.ndarray): data need to calculate gradient
        dim (int) 
        method (int, optional): 0 is Central difference;
                        1 is front difference or half grid central difference;
                        . Defaults to 0.

    Returns:
        gradient array
    """
    data1 = _correct_type(data)
    data1_sp = data1.swapaxes(0, dim)
    if method == 0:
        grd_dat = data1_sp[2:] - data1_sp[:-2]
    elif method == 1:
        grd_dat = data1_sp[1:] - data1_sp[:-1]
    grd_dat = grd_dat.swapaxes(0, dim)
    return grd_dat


def gradient_da(da, dim, method=0, delta=None):

    if not isinstance(da, xr.DataArray):
        raise TypeError("'xr.DataArray' input is required, not the %s" % (type(da)))
    if not dim in list(da.coords.keys()):
        raise TypeError(f"DaArray don't have coords {dim} !")
    dim_coord = da[dim]
    if method == 0:
        dim_coord1 = dim_coord[1:]
        dim_coord2 = dim_coord[:-1]
    elif method == 1:
        dim_coord1 = dim_coord[2:]
        dim_coord2 = dim_coord[:-2]
    m_coord = (dim_coord1.to_numpy() + dim_coord2.to_numpy()) / 2
    # have finished
    if delta is None:
        # delta_dim = {"lon":111e3,"lat":111e3,"level":5}
        if dim == "lat":
            delta = 1.11e5
        elif dim == "level":
            delta = 5
        elif dim == "lon":
            delta = 1.11e5 * np.cos(np.deg2rad(da["lat"]))
        else:
            delta = 1
    da1 = da.loc[{dim: dim_coord1}].assign_coords({dim: m_coord})
    da2 = da.loc[{dim: dim_coord2}].assign_coords({dim: m_coord})
    diff = (da1 - da2) / delta
    return diff


def shp_mask(shp, Dataarray: xr.DataArray, lat='lat', lon='lon'):
    """
    Args:
        shp (geopandas.geodataframe.GeoDataFrame): shp geodata
        Dataarray (xr.DataArray): data 
        lat (str): y-axis name
        lon (str): x-axis name

    Returns:
        masked data
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "Can't import geopandas, please install through conda or pip")
    try:
        import rioxarray
    except ImportError:
        raise ImportError(
            "Can't import rioxarray, please install through conda or pip")
    try:
        from shapely.geometry import mapping
    except ImportError:
        raise ImportError(
            "Can't import shapely, please install through conda or pip")

    Dataarray = Dataarray.copy()
    Dataarray.rio.write_crs("epsg:4326", inplace=True)
    Dataarray.rio.set_spatial_dims(x_dim=lat, y_dim=lon, inplace=True)
    cliped = Dataarray.rio.clip(shp.geometry.apply(mapping),
                                shp.crs,
                                drop=False)
    return cliped
