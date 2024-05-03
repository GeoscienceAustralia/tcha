import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

projection = ccrs.PlateCarree(central_longitude=180)
transform = ccrs.PlateCarree()

DATEFMT = "%Y-%m-%d %H:%M"
OUTPUTDIR = r"..\data\intensity"
path = os.path.dirname(os.getcwd())
sys.path.append(path)
from utils import load_ibtracs_df, savefig

udf = load_ibtracs_df(basins=["SI", "SP"])
ax = sns.histplot(udf.groupby("DISTURBANCE_ID")['MAX_WIND_SPD'].max(),
             bins=np.arange(10, 150.1, 5),
             stat='probability', kde=True)
ax.set_xlabel("Maximum intensity [kts]")
fig = plt.gcf()
plt.text(0.01, 0.0, "Source: IBTrACS doi:10.25921/82ty-9e16",
        transform=fig.transFigure, ha='left', va='bottom',
        fontsize='xx-small')
savefig(os.path.join(OUTPUTDIR, "maximum_intensity.jpg"),
        dpi=600, bbox_inches='tight')
plt.close(plt.gcf())

lmiidx = udf.groupby(["DISTURBANCE_ID"])["MAX_WIND_SPD"].idxmax().dropna()
lmidf = udf.loc[lmiidx]

lmidf.to_csv(os.path.join(OUTPUTDIR, "lmidata.csv"), index=False, date_format=DATEFMT)

bins = np.arange(-40, 0.1, 2.5)
ax = sns.histplot(lmidf['LAT'], bins=bins, stat='density')
ax.set_xlabel("Latitude")
fig = plt.gcf()
plt.text(0.01, 0.0, "Source: IBTrACS doi:10.25921/82ty-9e16",
        transform=fig.transFigure, ha='left', va='bottom',
        fontsize='xx-small')
savefig(os.path.join(OUTPUTDIR, "latitude_maximum_intensity.jpg"),
        dpi=600, bbox_inches='tight')

xbins = np.arange(40, 220.1, 2.0)
ybins = np.arange(-40, 0.1, 2.0)

from copy import copy

palette = copy(plt.get_cmap("viridis_r"))
palette.set_under("white", 1.0)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), subplot_kw={"projection": projection})

tdhist, xedges, yedges = np.histogram2d(udf['LON'], udf['LAT'], bins=[xbins, ybins])
lmihist, _, _ = np.histogram2d(lmidf['LON'], lmidf['LAT'], bins=[xbins, ybins])
nseasons = udf.iloc[-1].TM.year - udf.iloc[0].TM.year + 1

cb = axes[0].pcolormesh(xedges, yedges, tdhist.T/nseasons, transform=transform, cmap=palette)
fig.colorbar(cb, ax=axes[0], orientation="horizontal", label="TCs/year", aspect=50)

cb = axes[1].pcolormesh(xedges, yedges, lmihist.T/nseasons, transform=transform, cmap=palette)
fig.colorbar(cb, ax=axes[1], orientation="horizontal", label="Max intensity TCs/year", aspect=50)

for ax in axes:
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle=':')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style={'size': 'x-small'}
    gl.ylabel_style={'size': 'x-small'}

    # TODO:
    # Fix longitude grid interval

plt.text(0.01, 0.0, "Source: IBTrACS doi:10.25921/82ty-9e16",
        transform=fig.transFigure, ha='left', va='bottom',
        fontsize='xx-small')
savefig(os.path.join(OUTPUTDIR, "trackdensity.jpg"),
        dpi=600, bbox_inches='tight')