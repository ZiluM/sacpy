import xarray as xr
import matplotlib.pyplot as plt
from . import Map
import cartopy.crs as ccrs

# xr.DataArray.scontourf=Map._scontourf()


def _scontourf(self, ax=None, *args, **kwargs):
    data = self.to_numpy()
    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    m = ax.scontourf(self.lon,self.lat,data,*args,**kwargs)
    ax.init_map()
    return m


xr.DataArray.scontourf = _scontourf
