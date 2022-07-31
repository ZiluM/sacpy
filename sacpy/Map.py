from cartopy.mpl.geoaxes import GeoAxesSubplot
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import MultipleLocator
import numpy as np


def get_levels(data, percentile: int, num_level: int, zero_sym: bool) -> np.ndarray:
    if percentile is None:
        percentile = 98
    if num_level is None:
        num_level = 13
    if zero_sym is None:
        zero_sym = True
    _max = np.nanpercentile(data, percentile)
    _min = np.nanpercentile(data, 100 - percentile)
    if zero_sym:
        max_real = np.max(np.abs([_max, _min]))
        levels = np.linspace(-max_real, max_real, num_level)
    else:
        levels = np.linspace(_min, _max, num_level)
    return levels


def _store_range(self, x, y):
    if not hasattr(self, "xrange"):
        self.xrange = [x]
    else:
        self.xrange.append(x)
    if not hasattr(self, "yrange"):
        self.yrange = [y]
    else:
        self.append(y)


def _anal_args(args):
    if len(args) == 1:
        levels0 = None
        Z = args[0]
    elif len(args) == 2:
        levels0 = args[1]
        Z = args[0]
    elif len(args) == 3:
        levels0 = None
        Z = args[2]
    elif len(args) == 4:
        levels0 = args[3]
        Z = args[2]
    else:
        raise ValueError("More parameters are transported")
    return Z, levels0


def _contourf(self, *args, **kwargs):
    levels1 = kwargs.get("levels")
    extend = kwargs.get("extend")
    if extend is None:
        extend = "both"
        kwargs["extend"] = extend
    percentile = kwargs.get("percentile")
    if percentile is not None:
        del kwargs['percentile']
    num_level = kwargs.get("num_level")
    if num_level is not None:
        del kwargs['num_level']
    zero_sym = kwargs.get("zero_sym")
    if zero_sym is not None:
        del kwargs['zero_sym']
    Z, levels0 = _anal_args(args)
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


def _draw_ticks(self, extend, stepx=30, stepy=10, xsmall=5, ysmall=5):
    print(extend)
    x1, x2, y1, y2 = extend
    xs = x1 // 10 * 10 if x1 % 10 == 0 else (x1 // 10 + 1) * 10
    xe = x2 // 10 * 10
    ys = y1 // 10 * 10 if y1 % 10 == 0 else (y1 // 10 + 1) * 10
    ye = y2 // 10 * 10
    print(xs, xe, ys, ye)
    xticks = np.arange(xs, xe + 1, stepx)
    yticks = np.arange(ys, ye + 1, stepy)
    self.set_xticks(xticks, crs=ccrs.PlateCarree())
    self.set_yticks(yticks, crs=ccrs.PlateCarree())
    self.yaxis.set_major_formatter(LatitudeFormatter())
    self.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    self.xaxis.set_minor_locator(MultipleLocator(xsmall))
    self.yaxis.set_minor_locator(MultipleLocator(ysmall))


def _initialize_map(self, *params, **kwargs):
    self.coastlines(*params, **kwargs)
    self.set_aspect("auto")


def _contour(self, *args, **kwargs):
    transform = kwargs.get("transform")
    if transform is None:
        transform = ccrs.PlateCarree()
        kwargs['transform'] = transform
    n = self.contour(*args, **kwargs)
    return n


def _sig_ctrf(self, x, y, pvalue, thrshd=0.05, marker=".."):
    res = self.contourf(x, y, pvalue, levels=[0, thrshd, 1], zorder=1,
                        hatches=[marker, None], colors="None",
                        transform=ccrs.PlateCarree())
    return res


def _pcolormesh(self, *args, **kwargs):
    Z, levels0 = _anal_args(args)
    levels1 = kwargs.get("levels")
    percentile = kwargs.get("percentile")
    if percentile is not None:
        del kwargs['percentile']
    num_level = kwargs.get("num_level")
    if num_level is not None:
        del kwargs['num_level']
    zero_sym = kwargs.get("zero_sym")
    if zero_sym is not None:
        del kwargs['zero_sym']
    Z, levels0 = _anal_args(args)
    if levels0 is None and levels1 is None:
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


GeoAxesSubplot.ctrf = _contourf
GeoAxesSubplot.ctr = _contour
GeoAxesSubplot.draw_ticks = _draw_ticks
GeoAxesSubplot.init_map = _initialize_map
GeoAxesSubplot.sig_ctrf = _sig_ctrf
GeoAxesSubplot.pclms = _pcolormesh
