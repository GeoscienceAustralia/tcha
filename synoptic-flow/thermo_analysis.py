"""
Analysis of climatological thermodynamics in the Southern Hemisphere

Requires pre-calculated file of winds at 850 & 250 hPa, including variance
and covariance. This is created by Lin's TC model code (ref)

Dependencies:
- xarray
- pandas
- pyproj
- cartopy
- seaborn

"""

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from calendar import month_name, month_abbr
import pyproj
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
import cartopy.util as cutil
import seaborn as sns

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

DATA_DIR = "/g/data/w85/data/tc"
geodesic = pyproj.Geod(ellps="WGS84")
proj = ccrs.PlateCarree(central_longitude=180)
trans = ccrs.PlateCarree()

LONLOCATOR = MultipleLocator(30)
LATLOCATOR = MultipleLocator(10)

def savefig(filename, *args, **kwargs):
    """
    Add a timestamp to each figure when saving

    :param str filename: Path to store the figure at
    :param args: Additional arguments to pass to `plt.savefig`
    :param kwargs: Additional keyword arguments to pass to `plt.savefig`
    """
    fig = plt.gcf()
    plt.text(0.99, 0.01, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
            transform=fig.transFigure, ha='right', va='bottom',
            fontsize='xx-small')
    plt.savefig(filename, *args, **kwargs)

def cyclic_wrapper(x, dim="longitude"):
    """
    Use cartopy.util.add_cyclic_point with an xarray Dataset to
    add a cyclic or wrap-around pixel to the `lon` dimension. This can be useful
    for plotting with `cartopy`

    So add_cyclic_point() works on 'dim' via xarray Dataset.map()

    :param x: `xr.Dataset` to process
    :param str dim: Dimension of the dataset to wrap on (default "longitude")
    """
    wrap_data, wrap_lon = cutil.add_cyclic_point(
        x.values,
        coord=x.coords[dim].data,
        axis=x.dims.index(dim)
    )
    return xr.DataArray(
        wrap_data,
        coords={dim: wrap_lon, **x.drop_vars(dim).coords},
        dims=x.dims
    )

BASEDIR = "C:/Workspace/tropical_cyclone_risk/data/era5"
windfile = os.path.join(BASEDIR, "thermo_era5_198101_202112.nc")
ds = xr.open_dataset(windfile)
# This shifts to a central longitude of 180E, rather than 0E
ds = ds.roll(lon=-180, roll_coords=True)
ds['lon'] = np.where(ds['lon'] < 0, ds['lon'] + 360, ds['lon'])

vmax = ds['vmax']     # Potential intensity
chi = ds['chi']       # Entropy deficit
rhmid = ds['rh_mid']  # Mid-level humidity (600 hPa)

vmax = cyclic_wrapper(vmax, "lon")
chi = cyclic_wrapper(chi, "lon")
rhmid = cyclic_wrapper(rhmid, "lon")

landmask = xr.open_dataset("C:/WorkSpace/tropical_cyclone_risk/land/GL.nc")
mask = landmask['basin']
mask = cyclic_wrapper(mask, "lon")


# Seasonal mean values in each basin
vmaxsh_djf = vmax.sel(time=vmax.time.dt.season=="DJF").sel(lat=slice(-5, -15)).mean(dim='lat', skipna=True)
vmaxsh_seasonal = vmaxsh_djf.groupby(vmaxsh_djf.time.dt.year).mean("time")
chish_djf = chi.sel(time=chi.time.dt.season=="DJF").sel(lat=slice(-5, -15)).mean(dim='lat')
chish_seasonal = chish_djf.groupby(chish_djf.time.dt.year).mean("time")
rhmidsh_djf = rhmid.sel(time=rhmid.time.dt.season=="DJF").sel(lat=slice(-5, -15)).mean(dim='lat')
rhmidsh_seasonal = rhmidsh_djf.groupby(rhmidsh_djf.time.dt.year).mean("time")

vmaxnh_jja = vmax.sel(time=vmax.time.dt.season=="JJA").sel(lat=slice(15, 5)).mean(dim='lat', skipna=True)
vmaxnh_seasonal = vmaxnh_jja.groupby(vmaxnh_jja.time.dt.year).mean("time")
chinh_jja = chi.sel(time=chi.time.dt.season=="JJA").sel(lat=slice(15, 5)).mean(dim='lat')
chinh_seasonal = chinh_jja.groupby(chinh_jja.time.dt.year).mean("time")
rhmidnh_jja = rhmid.sel(time=rhmid.time.dt.season=="JJA").sel(lat=slice(15, 5)).mean(dim='lat')
rhmidnh_seasonal = rhmidnh_jja.groupby(rhmidnh_jja.time.dt.year).mean("time")

# Plot timeseries of seasonal vorticity:
fig, ax = plt.subplots(3, 1, figsize=(12, 8))
ax[0].plot(vmaxsh_seasonal.year, vmaxsh_seasonal.sel(lon=slice(160, 180)).mean(axis=1, skipna=True), label='SWP',)
ax[0].plot(vmaxsh_seasonal.year, vmaxsh_seasonal.sel(lon=slice(90, 110)).mean(axis=1, skipna=True), label='EIO', linestyle='--')
ax[0].plot(vmaxnh_seasonal.year, vmaxnh_seasonal.sel(lon=slice(120, 140)).mean(axis=1, skipna=True), label='NWP', linestyle='--')
ax[0].axhline(vmaxsh_seasonal.sel(lon=slice(160, 180)).mean(), linestyle='--', alpha=0.5)
ax[0].axhline(vmaxsh_seasonal.sel(lon=slice(90, 110)).mean(), linestyle='--', color='orange', alpha=0.5)
ax[0].axhline(vmaxnh_seasonal.sel(lon=slice(120, 140)).mean(), linestyle='--', color='green', alpha=0.5)
ax[0].set_ylabel(r"Potential intensity [m/s]")
ax[0].text(0.05, 0.95, r"$v_{max}$", transform=ax[0].transAxes, fontweight='bold', va='top')
ax[0].grid(True)
ax[0].legend(loc=1)

ax[1].plot(chish_seasonal.year, chish_seasonal.sel(lon=slice(160, 180)).mean(axis=1), label='SWP',)
ax[1].plot(chish_seasonal.year, chish_seasonal.sel(lon=slice(90, 110)).mean(axis=1), label='EIO', linestyle='--')
ax[1].plot(chinh_seasonal.year, chinh_seasonal.sel(lon=slice(120, 140)).mean(axis=1), label='NWP', linestyle='--')
ax[1].axhline(chish_seasonal.sel(lon=slice(160, 180)).mean(), linestyle='--', alpha=0.5)
ax[1].axhline(chish_seasonal.sel(lon=slice(90, 110)).mean(), linestyle='--', color='orange', alpha=0.5)
ax[1].axhline(chinh_seasonal.sel(lon=slice(120, 140)).mean(), linestyle='--', color='green', alpha=0.5)
ax[1].set_ylabel(r"Entropy deficit")
ax[1].text(0.05, 0.95, r"$\chi$", transform=ax[1].transAxes, fontweight='bold', va='top')
ax[1].grid(True)
ax[1].legend(loc=1)

ax[2].plot(rhmidsh_seasonal.year, rhmidsh_seasonal.sel(lon=slice(160, 180)).mean(axis=1), label='SWP',)
ax[2].plot(rhmidsh_seasonal.year, rhmidsh_seasonal.sel(lon=slice(90, 110)).mean(axis=1), label='EIO', linestyle='--')
ax[2].plot(rhmidnh_seasonal.year, rhmidnh_seasonal.sel(lon=slice(120, 140)).mean(axis=1), label='NWP', linestyle='--')
ax[2].axhline(rhmidsh_seasonal.sel(lon=slice(160, 180)).mean(), linestyle='--', alpha=0.5)
ax[2].axhline(rhmidsh_seasonal.sel(lon=slice(90, 110)).mean(), linestyle='--', color='orange', alpha=0.5)
ax[2].axhline(rhmidnh_seasonal.sel(lon=slice(120, 140)).mean(), linestyle='--', color='green', alpha=0.5)
ax[2].set_ylabel(r"Mid-level humidity")
ax[2].text(0.05, 0.95, r"$RH_{600}$", transform=ax[2].transAxes, fontweight='bold', va='top')
ax[2].grid(True)
ax[2].legend(loc=1)
fig.tight_layout()
savefig(os.path.join(BASEDIR, "thermo_timeseries.png"))