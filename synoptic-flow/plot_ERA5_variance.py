import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from calendar import month_name
import pyproj
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

BASEDIR = "/scratch/w85/cxa547/tcr/data/era5"
windfile = os.path.join(BASEDIR, "env_wnd_era5_198101_202112.nc")

# Open the dataset and calculate weighted monthly averages
ds = xr.open_dataset(windfile)
month_length= ds.time.dt.days_in_month
weights = (
    month_length.groupby("time.month") / month_length.groupby("time.month").sum()
)
dsw = (ds * weights).groupby("time.month").sum(dim="time")
extents = (120, 180, -30, 0)
dlevs = np.arange(-20, 20.1, 2.5) # For diverging variables
levs = np.arange(-10, 10.1, 1)
clevs = np.arange(-10, 10.1, .5)

for month in dsw.month:
    fig, ax = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()},
                           figsize=(12, 8), sharex=True, sharey=True)
    dds = dsw.sel(month=month)
    ua850_cov = np.sqrt(dds['ua850_Var'])/dds['ua850_Mean']
    cs = ax[0, 0].contourf(dsw.lon, dsw.lat, ua850_cov,
                      levels=levs, extend='both', cmap='RdBu')
    ax[0, 0].contour(dsw.lon, dsw.lat, ua850_cov,
                     levels=clevs, linewidths=0.5)
    ax[0, 0].set_title(r"$u_{850}$")
    va850_cov = np.sqrt(dds['va850_Var'])/dds['va850_Mean']
    ax[0, 1].contourf(dsw.lon, dsw.lat, va850_cov,
                      levels=levs, extend='both', cmap='RdBu')
    ax[0, 1].contour(dsw.lon, dsw.lat, va850_cov,
                     levels=clevs, linewidths=0.5)
    ax[0, 1].set_title(r"$v_{850}$")
    
    ua250_cov = np.sqrt(dds['ua250_Var'])/dds['ua250_Mean']
    ax[1, 0].contourf(dsw.lon, dsw.lat, ua250_cov,
                      levels=levs, extend='both', cmap='RdBu')
    ax[1, 0].contour(dsw.lon, dsw.lat, ua250_cov,
                     levels=clevs, linewidths=0.5)
    ax[1, 0].set_title(r"$u_{250}$")

    va250_cov = np.sqrt(dds['va250_Var'])/dds['va250_Mean']
    ax[1, 1].contourf(dsw.lon, dsw.lat, va250_cov,
                      levels=levs, extend='both', cmap='RdBu')
    ax[1, 1].contour(dsw.lon, dsw.lat, va250_cov,
                     levels=clevs, linewidths=0.5)
    ax[1, 1].set_title(r"$v_{250}$")
    
    for a in ax.flatten():
        a.coastlines(color='0.5', linewidth=1.5)
        gl = a.gridlines(draw_labels=True, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style={'size': 'x-small'}
        gl.ylabel_style={'size': 'x-small'}
        gl.top_labels = False
        gl.right_labels = False
        a.set_extent(extents)

    fig.subplots_adjust(bottom=0.25, top=0.9, )
    cbar_ax = fig.add_axes([0.2, 0.2, 0.6, 0.02])
    cbar=fig.colorbar(cs, cax=cbar_ax,orientation='horizontal')
    fig.suptitle(f"CoV - {month_name[month.values]}")
    figname = os.path.join(BASEDIR, f"{month_name[month.values]}.CoV.png")
    plt.savefig(figname, bbox_inches='tight')
    plt.close(fig)

for month in dsw.month:
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 8))
    dds = dsw.sel(month=month)

    ax.barbs(dsw.lon[::2], dsw.lat[::2], 
             dds['ua850_Mean'][::2, ::2],
             dds['va850_Mean'][::2, ::2],
             color='k', flip_barb=True, length=5, linewidth=1,
            fill_empty=True)
    ax.barbs(dsw.lon[::2], dsw.lat[::2],
             dds['ua250_Mean'][::2, ::2],
             dds['va250_Mean'][::2, ::2],
             color='r', flip_barb=True, length=5, linewidth=1)
    ax.coastlines(color='0.5', linewidth=1.5)
    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style={'size': 'x-small'}
    gl.ylabel_style={'size': 'x-small'}
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent(extents)
    fig.suptitle(f"{month_name[month.values]}")
    figname = os.path.join(BASEDIR, f"{month_name[month.values]}.meanflow.png")
    plt.savefig(figname, bbox_inches='tight')
    plt.close(fig)