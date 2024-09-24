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
MONTHS = [1, 2, 12]
CMAP = sns.blend_palette(["#a05806", "#FFFFFF", "#1e76cf"],
                         n_colors=20, as_cmap=True)

BBOX=dict(boxstyle="square",
          ec=(1., 0.5, 0.5),
          fc=(1., 0.8, 0.8),
          )


basedir = "/scratch/w85/cxa547/tcpi"
logging.info("Load IBTRACS data for observed genesis locations")
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
# DataFrame of genesis locations:
gpdf = pd.DataFrame(recs)

logging.info("Load TCGP and components")
fname = os.path.join(basedir, "tcgp.1981-2023.nc")
ds = xr.open_dataset(fname)
# Create long-term mean dataset:
ltmds = ds.groupby(ds.time.dt.month).mean(dim="time")

djfds = ds.sel(time=ds["time.month"].isin(MONTHS)).mean(dim="time")


for month in range(1, 13):
    logging.info(f"Plot ratio of absolute vorticity to meridional gradient of vorticity for {month_name[month]}")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), subplot_kw={"projection": proj})
    cs = ax.contourf(
        ltmds["longitude"],
        ltmds["latitude"],
        ltmds['xi'][month-1, :, :],
        levels=np.linspace(0, 1e-4, 21),
        transform=trans,
        extend="both",
        cmap=CMAP,
    )
    ax.contour(
        ltmds["longitude"],
        ltmds["latitude"],
        ltmds['xi'][month-1, :, :],
        levels=[2e-5],
        colors="r",
        linewidths=1,
        transform=trans,
    )

    cb = plt.colorbar(
        cs, ax=ax, orientation="horizontal",
        pad=0.025, label=r"$\xi$ [$s^{-1}$]",
        aspect=40
    )
    cb.ax.xaxis.set_major_formatter(CBFORMATTER)

    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle=":")
    gl.xlocator = LONLOCATOR
    gl.ylocator = LATLOCATOR
    gl.bottom_labels = False
    gl.right_labels = False
    ax.set_extent(EXTENT, crs=trans)
    ax.text(50, -40, month_name[month], transform=trans, bbox=BBOX)
    utils.savefig(os.path.join(basedir, f"xi.{month:02d}.pdf"))

for month in range(1, 13):
    logging.info(f"Plot ratio of absolute vorticity to meridional gradient of vorticity for {month_name[month]}")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), subplot_kw={"projection": proj})
    zz = 1. / (1 + np.power(ltmds['xi'][month-1, :, :], -1/0.15))
    cs = ax.contourf(
        ltmds["longitude"],
        ltmds["latitude"],
        zz,
        levels=np.linspace(0, 1, 21),
        transform=trans,
        extend="both",
        cmap=CMAP,
    )
    ax.contour(
        ltmds["longitude"],
        ltmds["latitude"],
        zz,
        levels=[0.5],
        colors="r",
        linewidths=1,
        transform=trans,
    )

    cb = plt.colorbar(
        cs, ax=ax, orientation="horizontal",
        pad=0.025, label=r"$Z$ ",
        aspect=40
    )
    cb.ax.xaxis.set_major_formatter(CBFORMATTER)

    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle=":")
    gl.xlocator = LONLOCATOR
    gl.ylocator = LATLOCATOR
    gl.bottom_labels = False
    gl.right_labels = False
    ax.set_extent(EXTENT, crs=trans)
    ax.text(50, -40, month_name[month], transform=trans, bbox=BBOX)
    utils.savefig(os.path.join(basedir, f"Z.{month:02d}.pdf"))

for month in range(1, 13):
    logging.info(f"Plot relative humidity for {month_name[month]}")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), subplot_kw={"projection": proj})
    cs = ax.contourf(
        ltmds["longitude"],
        ltmds["latitude"],
        ltmds['rh'][month-1, :, :],
        levels=np.linspace(10, 90, 17),
        transform=trans,
        extend="both",
        cmap=CMAP,
    )
    ax.contour(
        ltmds["longitude"],
        ltmds["latitude"],
        ltmds['rh'][month-1, :, :],
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
    ax.text(50, -40, month_name[month], transform=trans, bbox=BBOX)
    utils.savefig(os.path.join(basedir, f"RH.{month:02d}.pdf"))

for month in range(1, 13):
    logging.info(f"Plot potential intensity for {month_name[month]}")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), subplot_kw={"projection": proj})
    cs = ax.contourf(
        ltmds["longitude"],
        ltmds["latitude"],
        ltmds['vmax'][month-1, :, :],
        levels=np.linspace(20, 90, 15),
        transform=trans,
        extend="both",
        cmap='viridis_r',
    )
    ax.contour(
        ltmds["longitude"],
        ltmds["latitude"],
        ltmds['vmax'][month-1, :, :],
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
    ax.text(50, -40, month_name[month], transform=trans, bbox=BBOX)
    utils.savefig(os.path.join(basedir, f"VMAX.{month:02d}.pdf"))

for month in range(1, 13):
    logging.info(f"Plot wind shear for {month_name[month]}")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), subplot_kw={"projection": proj})
    cs = ax.contourf(
        ltmds["longitude"],
        ltmds["latitude"],
        ltmds['shear'][month-1, :, :],
        levels=np.linspace(10, 50, 9),
        transform=trans,
        extend="both",
        cmap='viridis_r',
    )
    ax.contour(
        ltmds["longitude"],
        ltmds["latitude"],
        ltmds['shear'][month-1, :, :],
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
    ax.text(50, -40, month_name[month], transform=trans, bbox=BBOX)
    utils.savefig(os.path.join(basedir, f"VSH.{month:02d}.pdf"))

for month in range(1, 13):
    logging.info(f"Plot TCGP for {month_name[month]}")
    fig, ax = plt.subplots(
        1, 1, figsize=(12, 5), subplot_kw={"projection": proj}, sharex=True
    )
    cs = ax.contourf(
        ltmds["longitude"],
        ltmds["latitude"],
        ltmds['tcgp'][month-1, :, :],
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
        ltmds["longitude"],
        ltmds["latitude"],
        ltmds['vmax'][month-1, :, :],
        levels=[40],
        colors="red",
        linewidths=lw,
        transform=trans,
    )
    ax.contour(
        ltmds["longitude"],
        ltmds["latitude"],
        ltmds['xi'][month-1, :, :],
        levels=[2e-5],
        colors="orange",
        linewidths=lw,
        transform=trans,
    )
    ax.contour(
        ltmds["longitude"],
        ltmds["latitude"],
        ltmds['rh'][month-1, :, :],
        levels=[40],
        colors="b",
        linewidths=lw,
        transform=trans,
    )
    ax.contour(
        ltmds["longitude"],
        ltmds["latitude"],
        ltmds['shear'][month-1, :, :],
        levels=[20],
        colors="green",
        linewidths=lw,
        transform=trans,
    )
    seasgen = gpdf[gpdf.TM.dt.month==month]
    ax.scatter(seasgen.LON, seasgen.LAT, s=2, c='k', alpha=0.5, transform=trans)

    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle=":")
    gl.xlocator = LONLOCATOR
    gl.ylocator = LATLOCATOR
    gl.bottom_labels = False
    gl.right_labels = False
    ax.set_extent(EXTENT, crs=trans)
    ax.text(50, -40, month_name[month], transform=trans, bbox=BBOX)
    utils.savefig(os.path.join(basedir, f"TCGP.{month:02d}.pdf"))

