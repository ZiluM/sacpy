try:
    from cartopy.mpl.geoaxes import GeoAxesSubplot
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
except:
    raise ImportError("Can't import cartopy, please check your envs")
from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
from .Util import _correct_type
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['image.cmap'] = 'RdBu_r'
# set font to TNR
# plt.rc('font', family='Times New Roman')
# plt.set_cmap("RdBu_r")


def get_levels(data, percentile: int = 98, num_level: int = 13, zero_sym: bool = True) -> np.ndarray:
    """ get levels of data

    Args:
        data (np.ndarray): data need to get levels
        percentile (int): max value or min value percentile
        num_level (int): colorbar levels
        zero_sym (bool): levels is zero symmetric or not

    Returns:
        np.ndarray: levels for contourf or contourf
    """
    #　set default values
    if percentile is None:
        percentile = 98
    if num_level is None:
        num_level = 13
    # if zero_sym is None:
    #     zero_sym = True
    # get min and max
    _max = np.nanpercentile(data, percentile)
    _min = np.nanpercentile(data, 100 - percentile)
    if zero_sym is None:
        if _min * _max < 0:
            zero_sym = True
        else:
            zero_sym = False
    # zero_sym :
    if zero_sym:
        max_real = np.max(np.abs([_max, _min]))
        levels = np.linspace(-max_real, max_real, num_level)
    else:
        levels = np.linspace(_min, _max, num_level)
    return levels


def _store_range(self, x, y):
    """ store data range for x and y of data

    Args:
        x (np.ndarray): x 
        y (np.ndarray): y
    """
    x = _correct_type(x)
    y = _correct_type(y)
    if x is not None:
        # save to xrange
        if not hasattr(self, "xrange"):
            self.xrange = [x]
        else:
            self.xrange.append(x)
    if y is not None:
        # save to yrange
        if not hasattr(self, "yrange"):
            self.yrange = [y]
        else:
            self.xrange.append(y)


def _get_extend(self):
    """ get self.xrange's xmin ,xmax and yrange's ymin and ymax
    Returns:
        xmin, xmax, ymin, ymax (float)
    """
    if hasattr(self, "xrange") and hasattr(self, "yrange"):
        xmin0, ymin0 = 360., 90.
        xmax0, ymax0 = -180, -90
        for xra, yra in zip(self.xrange, self.yrange):
            xmin1 = xra.min()
            ymin1 = yra.min()
            xmax1 = xra.max()
            ymax1 = yra.max()
            if xmin1 <= xmin0:
                xmin0 = xmin1
            if xmax1 >= xmax0:
                xmax0 = xmax1
            if ymin0 >= ymin1:
                ymin0 = ymin1
            if ymax0 <= ymax1:
                ymax0 = ymax1
        xmin, ymin, xmax, ymax = xmin0, ymin0, xmax0, ymax0
    else:
        xmin, ymin, xmax, ymax = None, None, None, None
    return xmin, xmax, ymin, ymax


def _get_extra_param(kwargs):
    """ get extra parameter for get levels

    Args:
        kwargs (dict)

    Returns:
        percentile, num_level, zero_sym (int,float,bool)
    """
    percentile = kwargs.get("percentile")
    if percentile is not None:
        del kwargs['percentile']
    num_level = kwargs.get("num_level")
    if num_level is not None:
        del kwargs['num_level']
    zero_sym = kwargs.get("zero_sym")
    if zero_sym is not None:
        del kwargs['zero_sym']
    return percentile, num_level, zero_sym


def _anal_args(args, clas="contourf"):
    """ analysis args for draw function

    Args:
        args (list)
        clas (str, optional): class of the function. Defaults to "contourf".

    Returns:
        np.ndarray: need value
    """
    if clas == "contourf":
        if len(args) == 1:
            levels0 = None
            Z = args[0]
            x, y = np.arange(Z.shape[0]), np.arange(Z.shape[1])
        elif len(args) == 2:
            levels0 = args[1]
            Z = args[0]
            x, y = np.arange(Z.shape[0]), np.arange(Z.shape[1])
        elif len(args) == 3:
            levels0 = None
            Z = args[2]
            x, y = args[0], args[1]
        elif len(args) == 4:
            levels0 = args[3]
            Z = args[2]
            x, y = args[0], args[1]
        else:
            raise ValueError("More parameters are transported")
        x = _correct_type(x)
        y = _correct_type(y)
        Z = _correct_type(Z)
        return Z, levels0, x, y
    elif clas == "pcolormesh":
        if len(args) == 1:
            Z = args[0]
            x, y = np.arange(Z.shape[0]), np.arange(Z.shape[1])
        elif len(args) == 2:
            raise ValueError("More parameters (2) are transported")
        elif len(args) == 3:
            Z = args[2]
            x, y = args[0], args[1]
        else:
            raise ValueError("More parameters are transported")
        x = _correct_type(x)
        y = _correct_type(y)
        Z = _correct_type(Z)
        return Z, x, y
    elif clas == "quiver":
        if len(args) == 1:
            raise ValueError("Need two arrays to depict wind")
        elif len(args) == 2:
            U, V = args[0], args[1]
            x, y = np.arange(U.shape[0]), np.arange(U.shape[1])
            C = None
        elif len(args) == 3:
            U, V = args[0], args[1]
            C = args[2]
            x, y = np.arange(U.shape[0]), np.arange(U.shape[1])
        elif len(args) == 4:
            x, y, U, V = args
            C = None
        elif len(args) == 5:
            x, y, U, V, C = args
        x = _correct_type(x)
        y = _correct_type(y)
        U = _correct_type(U)
        V = _correct_type(V)
        return x, y, U, V, C


def _contourf(self, *args, **kwargs):
    """ new contourf for GeoAxesSubplot

    Returns:
        mpl.contour.QuadContourSet
    """
    levels1 = kwargs.get("levels")
    extend = kwargs.get("extend")
    if extend is None:
        extend = "both"
        kwargs["extend"] = extend
    percentile, num_level, zero_sym = _get_extra_param(kwargs)
    Z, levels0, x, y = _anal_args(args)
    self._store_range(x, y)
    if levels0 is None and levels1 is None:
        levels = get_levels(Z, percentile, num_level, zero_sym)
        kwargs['levels'] = levels
    cmap = kwargs.get("cmap")
    if cmap is None:
        cmap = "RdBu_r"
        kwargs["cmap"] = cmap
    transform = kwargs.get("transform")
    if transform is None:
        transform = ccrs.PlateCarree()
        kwargs["transform"] = transform
    m = self.contourf(*args, **kwargs)
    return m


def _draw_ticks(self, extend, stepx=30, stepy=10, smallx=5, smally=2.5, bigx=10, bigy=10):
    """ draw map ticks

    Args:
        extend (list): extend of map [x1, x2, y1, y2]
        stepx (int, optional): big x step. Defaults to 30.
        stepy (int, optional): big y step. Defaults to 10.
        smallx (int, optional): x small step. Defaults to 5.
        smally (int, optional): y small step. Defaults to 5.
        bigx/bigy (int, optional): Resolution in X and Y directions
    """
    [x1, x2, y1, y2] = extend
    # if stepx is None:
    #     stepx

    # autoly calculate stepx,stepy,smallx,smally,bigx,bigy
    # cling to 10 times
    xs = x1 // bigx * bigx if x1 % bigx == 0 else (x1 // bigx + 1) * bigx
    xe = x2 // bigx * bigx
    ys = y1 // bigy * bigy if y1 % bigy == 0 else (y1 // bigy + 1) * bigy
    ye = y2 // bigy * bigy
    # get xticks
    xticks = np.arange(xs, xe + 1, stepx)
    yticks = np.arange(ys, ye + 1, stepy)
    # set ticks
    self.set_xticks(xticks, crs=ccrs.PlateCarree())
    self.set_yticks(yticks, crs=ccrs.PlateCarree())
    self.yaxis.set_major_formatter(LatitudeFormatter())
    self.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    self.xaxis.set_minor_locator(MultipleLocator(smallx))
    self.yaxis.set_minor_locator(MultipleLocator(smally))


def _initialize_map(self, same_size=True, coastlines=True, **kwargs):
    """ initial a map
    """
    # draw coastines
    if coastlines:
        self.coastlines()
    # Flatten map
    if same_size:
        self.set_aspect("auto")
    extend = self._get_extends()
    self.draw_ticks(extend, **kwargs)


def _contour(self, *args, **kwargs):
    """ new contour for GeoAxesSubplot

    Returns:
        mpl.contour.QuadContourSet
    """
    # get mpl.contour.QuadContourSet or not
    if not isinstance(args[0], (mpl.contour.QuadContourSet)):
        Z, levels0, x, y = _anal_args(args)
        self._store_range(x, y)
    # add transform
    transform = kwargs.get("transform")
    if transform is None:
        transform = ccrs.PlateCarree()
        kwargs['transform'] = transform
    n = self.contour(*args, **kwargs)
    return n


def _sig_ctrf(self, x, y, pvalue, thrshd=0.05, marker="..", color=None):
    """ Significance test dot

    Args:
        x (np.ndarray): x
        y (np.ndarray): y
        pvalue (np.ndarray): p value
        thrshd (float, optional): threshold of pvalue. Defaults to 0.05.
        marker (str, optional): mark of Significance test dot. Defaults to "..".
    """
    res = self.contourf(x,
                        y,
                        pvalue,
                        levels=[0, thrshd, 1],
                        zorder=1,
                        hatches=[marker, None],
                        colors="None",
                        transform=ccrs.PlateCarree())
    # set color
    if color is not None:
        for collection in res.collections:
            collection.set_linewidth(0.)
            collection.set_edgecolor(color)
    return res


def _pcolormesh(self, *args, **kwargs):
    """ new pcolormesh
    """
    Z, x, y = _anal_args(args, clas="pcolormesh")
    # store x ,y for ticks
    self._store_range(x, y)
    percentile, num_level, zero_sym = _get_extra_param(kwargs)
    vmin, vmax = kwargs.get("vmin"), kwargs.get("vmax")
    if vmin is None and vmax is None:
        levels = get_levels(Z, percentile, num_level, zero_sym)
        kwargs['vmin'] = levels[0]
        kwargs['vmax'] = levels[-1]
    cmap = kwargs.get("cmap")
    if cmap is None:
        cmap = "RdBu_r"
        kwargs["cmap"] = cmap
    transform = kwargs.get("transform")
    if transform is None:
        transform = ccrs.PlateCarree()
        kwargs["transform"] = transform
    m = self.pcolormesh(*args, **kwargs)
    return m


def _quiver(self, *args, **kwargs):
    """ new quiver
    """
    # add projcetion
    transform = kwargs.get("transform")
    if transform is None:
        transform = ccrs.PlateCarree()
        kwargs["transform"] = transform
    stepx = kwargs.get("stepx")
    if stepx is None:
        stepx = 1
    else:
        del kwargs['stepx']
    stepy = kwargs.get("stepy")
    if stepy is None:
        stepy = 1
    else:
        del kwargs['stepy']
    x, y, U, V, C = _anal_args(args, clas="quiver")
    self._store_range(x, y)
    if C is None:
        args1 = [x[::stepx], y[::stepy], U[::stepy, ::stepx], V[::stepy, ::stepx]]
    else:
        args1 = [x[::stepx], y[::stepy], U[::stepy, ::stepx], V[::stepy, ::stepx], C]
    m = self.quiver(*args1, **kwargs)
    return m


def _xr_splot(self, ax=None, label=0, kw1={}, kw2={}):
    if label == 0:
        lon = "lon"
        lat = "lat"
    else:
        lon = "longitude"
        lat = "latitude"
    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=self[lon].mean().item()))
    m = ax.scontourf(self[lon], self[lat], self.to_numpy(), **kw1)
    ax.init_map(**kw2)
    plt.colorbar(m)
    return m, ax


GeoAxesSubplot.scontourf = _contourf
GeoAxesSubplot.scontour = _contour
GeoAxesSubplot.draw_ticks = _draw_ticks
GeoAxesSubplot.init_map = _initialize_map
GeoAxesSubplot.sig_plot = _sig_ctrf
GeoAxesSubplot.spcolormesh = _pcolormesh
GeoAxesSubplot._store_range = _store_range
GeoAxesSubplot._get_extends = _get_extend
GeoAxesSubplot.squiver = _quiver
xr.DataArray.splot = _xr_splot
