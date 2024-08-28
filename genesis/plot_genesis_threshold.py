import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
import xarray as xr

from datetime import datetime
from calendar import month_name, month_abbr
import pyproj
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

import cartopy.crs as ccrs
import cartopy.util as cutil
import cartopy.feature as cfeature
import seaborn as sns

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import metpy.constants as mpconst
import metpy.calc as mpcalc

from pathlib import Path

d = Path().resolve().parent
sys.path.append(str(d))
import utils


proj = ccrs.PlateCarree(central_longitude=180)
trans = ccrs.PlateCarree()

LONLOCATOR = mticker.MultipleLocator(30)
LATLOCATOR = mticker.MultipleLocator(15)
DATEFORMATTER = mdates.DateFormatter("%Y")
CBFORMATTER = mticker.ScalarFormatter(useMathText=True)
CBFORMATTER.set_powerlimits((-2, 2))
EXTENT = (30, 330, 50, -50)
MONTHS = [6, 7, 8]

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
        x.values, coord=x.coords[dim].data, axis=x.dims.index(dim)
    )
    return xr.DataArray(
        wrap_data, coords={dim: wrap_lon, **x.drop_vars(dim).coords}, dims=x.dims
    )


def savefig(filename, *args, **kwargs):
    """
    Add a timestamp to each figure when saving

    :param str filename: Path to store the figure at
    :param args: Additional arguments to pass to `plt.savefig`
    :param kwargs: Additional keyword arguments to pass to `plt.savefig`
    """
    fig = plt.gcf()
    plt.text(
        0.99,
        0.01,
        f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
        transform=fig.transFigure,
        ha="right",
        va="bottom",
        fontsize="xx-small",
    )
    plt.savefig(filename, *args, **kwargs)


basedir = "/scratch/w85/cxa547/tcpi"
logging.info("Load IBTRACS data")
df = utils.load_ibtracs_df()
df = df[(df.SEASON > 1980) & (df.SEASON < 2024)]
groupdf = df.groupby("DISTURBANCE_ID")
recs = []
for name, group in groupdf:
    # Filter the group for records where MAX_WIND_SPD exceeds the threshold
    exceeding_records = group[group["MAX_WIND_SPD"] >= 34]

    # If there are any records exceeding the threshold, select the first one
    if not exceeding_records.empty:
        rec = exceeding_records.iloc[0]
        recs.append(rec)

gpdf = pd.DataFrame(recs)

logging.info("Load vorticity")
flist = glob.glob(os.path.join(basedir, "abv.*.nc"))
ds = xr.open_mfdataset(flist)

# Take every 4th grid point (so a 1x1 degree grid)
djfds = ds.isel(longitude=slice(0, 1440, 4), latitude=slice(0, 721, 4)).\
            sel(time=ds["time.month"].isin(MONTHS))

djfdsmean = djfds.mean(dim="time")

djfdsmean = djfdsmean.roll(longitude=-180, roll_coords=True)
djfdsmean["longitude"] = np.where(
    djfdsmean["longitude"] < 0, djfdsmean["longitude"] + 360, djfdsmean["longitude"]
)

R = mpconst.earth_avg_radius
omega = mpconst.earth_avg_angular_vel

# Apply 2 passes of a 9-point smoother:
beta = cyclic_wrapper(djfdsmean["beta"], "longitude")
eta = cyclic_wrapper(djfdsmean["eta"], "longitude")
beta = mpcalc.smooth_n_point(beta, 9, 2)
eta = mpcalc.smooth_n_point(eta, 9, 2)

# Lower bounded value of beta for calculating xi:
beta_fl = xr.where(beta < 5e-12, 5e-12, beta)
xi = np.abs(eta) / (beta_fl * (R / (2 * omega)))

logging.info("Load MPI data")
# This is already on a 1.0 degree grid, so we do
# not need to subsample (using slice), nor apply a nine-point smoother.
flist = glob.glob(os.path.join(basedir, "pcmin.*.nc"))
ds = xr.open_mfdataset(flist)
djfds = ds.sel(time=ds["time.month"].isin(MONTHS))

djfdsmean = djfds.mean(dim="time")

djfdsmean = djfdsmean.roll(longitude=-180, roll_coords=True)
djfdsmean["longitude"] = np.where(
    djfdsmean["longitude"] < 0, djfdsmean["longitude"] + 360, djfdsmean["longitude"]
)
vmax = cyclic_wrapper(djfdsmean["vmax"], "longitude")

# Load wind shear data:
flist = os.path.join(basedir, "shear.1981-2023.nc")
ds = xr.open_dataset(flist)
djfds = ds.isel(longitude=slice(0, 1440, 4), latitude=slice(0, 721, 4)).sel(
    time=ds["time.month"].isin(MONTHS)
)

djfdsmean = djfds.mean(dim="time")

djfdsmean = djfdsmean.roll(longitude=-180, roll_coords=True)
djfdsmean["longitude"] = np.where(
    djfdsmean["longitude"] < 0, djfdsmean["longitude"] + 360, djfdsmean["longitude"]
)
shear = cyclic_wrapper(djfdsmean["shear"], "longitude")
shear = mpcalc.smooth_n_point(shear, 9, 2)


logging.info("Load relative humidity data")
flist = os.path.join(basedir, "humidity.1981-2023.nc")
ds = xr.open_dataset(flist)
djfds = ds.isel(longitude=slice(0, 1440, 4), latitude=slice(0, 721, 4)).sel(
    time=ds["time.month"].isin(MONTHS)
)

djfdsmean = djfds.mean(dim="time")

djfdsmean = djfdsmean.roll(longitude=-180, roll_coords=True)
djfdsmean["longitude"] = np.where(
    djfdsmean["longitude"] < 0, djfdsmean["longitude"] + 360, djfdsmean["longitude"]
)
rh = cyclic_wrapper(djfdsmean["r"], "longitude")
rh = mpcalc.smooth_n_point(rh, 9, 2)

cmap = sns.blend_palette(["#a05806", "#FFFFFF", "#1e76cf"], n_colors=20, as_cmap=True)

logging.info("Calculate normalised thresholds")
nu = (vmax / 40) - 1
mu = (xi.metpy.dequantify() / 2e-5) - 1
rho = (rh / 40) - 1
sigma = 1 - (shear / 20)

nu = xr.where(nu < 0, 0, nu)
mu = xr.where(mu < 0, 0, mu)
rho = xr.where(rho < 0, 0, rho)
sigma = xr.where(sigma < 0, 0, sigma)
tcgp = nu * mu * sigma * rho

fig, ax = plt.subplots(
    1, 1, figsize=(12, 6), subplot_kw={"projection": proj}, sharex=True
)

# This figure replicates Tory et al. 2018, Fig 2a, albeit with DJF, not JFM, and ERA5, not ERA-Interim
cs = ax.contourf(
    beta["longitude"],
    beta["latitude"],
    beta,
    levels=np.linspace(0, 8e-11, 17),
    transform=trans,
    extend="both",
    cmap=cmap,
)
ax.contour(
    beta["longitude"],
    beta["latitude"],
    beta,
    levels=[5e-12],
    transform=trans,
    colors="w",
    linewidths=1,
)
ax.contour(
    beta["longitude"],
    beta["latitude"],
    beta,
    levels=[0.23e-10],
    transform=trans,
    colors="k",
    linewidths=1,
)

cb = plt.colorbar(
    cs,
    ax=ax,
    orientation="horizontal",
    pad=0.025,
    label=r"$\beta^*_{700}$ [$m^{-1} s^{-1}$]",
    aspect=40,
)

cb.ax.xaxis.set_major_formatter(CBFORMATTER)
ax.coastlines()
gl = ax.gridlines(draw_labels=True, linestyle=":")
gl.xlocator = LONLOCATOR
gl.ylocator = LATLOCATOR
gl.bottom_labels = False
gl.right_labels = False
ax.set_extent(EXTENT, crs=trans)
savefig(os.path.join(basedir, "avrt_grad.JJA.png"))


logging.info("Plot absolute value of absolute vorticity")
fig, ax = plt.subplots(
    1,
    1,
    figsize=(12, 6),
    subplot_kw={"projection": proj},
)

cs = ax.contourf(
    eta["longitude"],
    eta["latitude"],
    np.abs(eta),
    levels=np.linspace(0, 8e-5, 17),
    transform=trans,
    extend="both",
    cmap=cmap,
)

cb = plt.colorbar(
    cs,
    ax=ax,
    orientation="horizontal",
    pad=0.025,
    label=r"$|\eta|_{850}$ [$s^{-1}$]",
    aspect=40,
)
cb.ax.xaxis.set_major_formatter(CBFORMATTER)
ax.coastlines()
gl = ax.gridlines(draw_labels=True, linestyle=":")
gl.xlocator = LONLOCATOR
gl.ylocator = LATLOCATOR
gl.bottom_labels = False
gl.right_labels = False
ax.set_extent(EXTENT, crs=trans)
savefig(os.path.join(basedir, "avrt.JJA.png"))

logging.info("Plot ratio of absolute vorticity to meridional gradient of vorticity")
fig, ax = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={"projection": proj})
cs = ax.contourf(
    xi["longitude"],
    xi["latitude"],
    xi,
    levels=np.linspace(0, 1e-4, 21),
    transform=trans,
    extend="both",
    cmap=cmap,
)
ax.contour(
    xi["longitude"],
    xi["latitude"],
    xi,
    levels=[2e-5],
    colors="r",
    linewidths=1,
    transform=trans,
)

cb = plt.colorbar(
    cs, ax=ax, orientation="horizontal", pad=0.025, label=r"$\xi$ [$s^{-1}$]", aspect=40
)
cb.ax.xaxis.set_major_formatter(CBFORMATTER)

ax.coastlines()
gl = ax.gridlines(draw_labels=True, linestyle=":")
gl.xlocator = LONLOCATOR
gl.ylocator = LATLOCATOR
gl.bottom_labels = False
gl.right_labels = False
ax.set_extent(EXTENT, crs=trans)
savefig(os.path.join(basedir, "xi.JJA.png"))

logging.info("Plot relative humidity")
fig, ax = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={"projection": proj})
cs = ax.contourf(
    rh["longitude"],
    rh["latitude"],
    rh,
    levels=np.linspace(10, 90, 17),
    transform=trans,
    extend="both",
    cmap=cmap,
)
ax.contour(
    rh["longitude"],
    rh["latitude"],
    rh,
    levels=[40],
    colors="b",
    linewidths=1,
    transform=trans,
)

cb = plt.colorbar(
    cs, ax=ax, orientation="horizontal", pad=0.025, label=r"$RH_{700}$ [%]", aspect=40
)
cb.ax.xaxis.set_major_formatter(CBFORMATTER)

ax.coastlines()
gl = ax.gridlines(draw_labels=True, linestyle=":")
gl.xlocator = LONLOCATOR
gl.ylocator = LATLOCATOR
gl.bottom_labels = False
gl.right_labels = False
ax.set_extent(EXTENT, crs=trans)
savefig(os.path.join(basedir, "RH.JJA.png"))

logging.info("Plot potential intensity")
fig, ax = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={"projection": proj})
cs = ax.contourf(
    vmax["longitude"],
    vmax["latitude"],
    vmax,
    levels=np.linspace(20, 90, 15),
    transform=trans,
    extend="both",
    cmap='viridis_r',
)
ax.contour(
    vmax["longitude"],
    vmax["latitude"],
    vmax,
    levels=[40],
    colors="r",
    linewidths=1,
    transform=trans,
)

cb = plt.colorbar(
    cs, ax=ax, orientation="horizontal", pad=0.025, label=r"$V_{m}$ [$ms^{-1}$]", aspect=40
)
cb.ax.xaxis.set_major_formatter(CBFORMATTER)

ax.coastlines()
gl = ax.gridlines(draw_labels=True, linestyle=":")
gl.xlocator = LONLOCATOR
gl.ylocator = LATLOCATOR
gl.bottom_labels = False
gl.right_labels = False
ax.set_extent(EXTENT, crs=trans)
savefig(os.path.join(basedir, "VMAX.JJA.png"))

logging.info("Plot wind shear")
fig, ax = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={"projection": proj})
cs = ax.contourf(
    shear["longitude"],
    shear["latitude"],
    shear,
    levels=np.linspace(10, 50, 9),
    transform=trans,
    extend="both",
    cmap='viridis_r',
)
ax.contour(
    shear["longitude"],
    shear["latitude"],
    shear,
    levels=[20],
    colors="r",
    linewidths=1,
    transform=trans,
)

cb = plt.colorbar(
    cs, ax=ax, orientation="horizontal", pad=0.025, label=r"$V_{sh}$ [$ms^{-1}$]", aspect=40
)
cb.ax.xaxis.set_major_formatter(CBFORMATTER)

ax.coastlines()
gl = ax.gridlines(draw_labels=True, linestyle=":")
gl.xlocator = LONLOCATOR
gl.ylocator = LATLOCATOR
gl.bottom_labels = False
gl.right_labels = False
ax.set_extent(EXTENT, crs=trans)
savefig(os.path.join(basedir, "VSH.JJA.png"))

logging.info("Plot TCGP")
fig, ax = plt.subplots(
    1, 1, figsize=(12, 6), subplot_kw={"projection": proj}, sharex=True
)
cs = ax.contourf(
    tcgp["longitude"],
    tcgp["latitude"],
    tcgp,
    levels=np.insert(np.linspace(0.5, 5, 19), 0, 0.05),
    transform=trans,
    extend="max",
    cmap="viridis_r",
)
cb = plt.colorbar(
    cs, ax=ax, orientation="horizontal", pad=0.025, label=r"TCGP", aspect=40
)
cb.ax.xaxis.set_major_formatter(CBFORMATTER)

lw = 0.5
ax.contour(
    vmax["longitude"],
    vmax["latitude"],
    vmax,
    levels=[40],
    colors="red",
    linewidths=lw,
    transform=trans,
)
ax.contour(
    xi["longitude"],
    xi["latitude"],
    xi,
    levels=[2e-5],
    colors="orange",
    linewidths=lw,
    transform=trans,
)
ax.contour(
    rh["longitude"],
    rh["latitude"],
    rh,
    levels=[40],
    colors="b",
    linewidths=lw,
    transform=trans,
)
ax.contour(
    shear["longitude"],
    shear["latitude"],
    shear,
    levels=[20],
    colors="green",
    linewidths=lw,
    transform=trans,
)
seasgen = gpdf[gpdf.TM.dt.month.isin(MONTHS)]
ax.scatter(seasgen.LON, seasgen.LAT, s=2, c='k', alpha=0.5, transform=trans)

ax.coastlines()
gl = ax.gridlines(draw_labels=True, linestyle=":")
gl.xlocator = LONLOCATOR
gl.ylocator = LATLOCATOR
gl.bottom_labels = False
gl.right_labels = False
ax.set_extent(EXTENT, crs=trans)
savefig(os.path.join(basedir, "TCGP.JJA.png"))
