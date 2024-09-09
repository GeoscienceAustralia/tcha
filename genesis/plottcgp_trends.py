import os
import sys
import logging
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from pathlib import Path

d = Path().resolve().parent
sys.path.append(str(d))
import utils

proj = ccrs.PlateCarree(central_longitude=180)
trans = ccrs.PlateCarree()

LONLOCATOR = MultipleLocator(30)
LATLOCATOR = MultipleLocator(10)
DATEFORMATTER = mdates.DateFormatter("%Y")

df = utils.load_ibtracs_df()
df = df[(df.SEASON >= 1980) & (df.SEASON < 2024)]
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


basedir = "/scratch/w85/cxa547/tcpi"
logging.info("Load TCGP and components")
fname = os.path.join(basedir, "tcgp.1981-2023.nc")
ds = xr.open_dataset(fname)

domains = {
    "IO": {
        "name": "Indian Ocean",
        "lonmin": 110.,
        "lonmax": 130.,
        "latmax": -10.,
        "latmin": -20.,
        "color": "b"
    },
    "CS": {
        "name": "Coral Sea",
        "lonmin": 145.,
        "lonmax": 165.,
        "latmax": -10.,
        "latmin": -20.,
        "color": "r"
    },
    "SWP": {
        "name": "SW Pacific",
        "lonmin": 160.,
        "lonmax": 180.,
        "latmax": -5.,
        "latmin": -15.,
        "color": "g"
    }
}

variables = {
    'vmax':{
        'levels':np.arange(30, 100, 5),
        'cbar_name': r"$V_m$ [m/s]"
    },
    'xi':{
        'levels': np.arange(280, 310, 1),
        'cbar_name': r"$\xi$ [s$^{-1}$]"
    },
    'rh':{
        'levels': np.arange(10, 91, 5),
        'cbar_name': r"$RH_{700}$ [%]"
    },
    'shear':{
        'levels': np.arange(0., 20.1, 2),
        'cbar_name': r"$V_{sh}$ [m/s]"
    },
    'tcgp':{
        'levels': np.arange(0., 20.1, 2),
        'cbar_name': "TCGP"
    }
}

for d in domains.keys():
    lonslice = slice(domains[d]['lonmin'], domains[d]['lonmax'])
    latslice = slice(domains[d]['latmax'], domains[d]['latmin'])
    subregion = ds.sel(latitude=latslice, longitude=lonslice)

    # Resample to quarters starting in December
    dsres = subregion.resample(time='QS-DEC').mean(dim="time")

    # Group by the year of the January month
    djf_grouped = dsres.sel(time=dsres.time.dt.season=="DJF")

    # Calculate the mean over DJF months
    domains[d]['data'] = djf_grouped.mean(dim=['latitude', 'longitude'])


fig, ax = plt.subplots(6, 1, figsize=(12, 10))
axes = ax.flatten()
axes[-1].set_frame_on(False)

axes[-1].xaxis.set_tick_params(labelbottom=False)
axes[-1].yaxis.set_tick_params(labelleft=False)
axes[-1].set_xticks([])
axes[-1].set_yticks([])
# Assign the first panel to be a Cartopy GeoAxes
geo_ax = plt.subplot(6, 1, 6, projection=ccrs.PlateCarree())
geo_ax.coastlines()
geo_ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k')
geo_ax.set_extent((80, 180, 0, -30), crs=trans)
gl = geo_ax.gridlines(draw_labels=True)
gl.top_labels=False
gl.right_labels=False

for d in domains.keys():
    for i, v in enumerate(['vmax', 'xi', 'rh', 'shear', 'tcgp']):
        axes[i].plot(domains[d]['data']['time'].dt.year, domains[d]['data'][v],
                     color=domains[d]['color'], label=d)
        geo_ax.plot([domains[d]['lonmin'],
                     domains[d]['lonmax'],
                     domains[d]['lonmax'],
                     domains[d]['lonmin'],
                     domains[d]['lonmin']],
                    [domains[d]['latmin'],
                     domains[d]['latmin'],
                     domains[d]['latmax'],
                     domains[d]['latmax'],
                     domains[d]['latmin']],
            color=domains[d]['color'],
            linewidth=2,
            transform=ccrs.PlateCarree())
axes[0].legend()
axes[0].grid(); axes[0].set_ylabel(r"$V_m$ [m/s]")
axes[1].grid(); axes[1].set_ylabel(r"$\xi$ [s$^{-1}$]")
axes[2].grid(); axes[2].set_ylabel(r"$RH_{700}$ [%]")
axes[3].grid(); axes[3].set_ylabel(r"$V_{sh}$ [m/s]")
axes[4].grid(); axes[4].set_ylabel("TCGP")


shift = -0.3
width = 0.25
countax = axes[4].twinx()
for d in domains.keys():
    ntcs = gpdf.loc[
    (gpdf.LON>domains[d]['lonmin']) &
    (gpdf.LON<domains[d]['lonmax']) &
    (gpdf.LAT>domains[d]['latmin']) &
    (gpdf.LAT<domains[d]['latmax']) &
    (gpdf.TM.dt.month.isin([12, 1, 2]))]['SEASON'].value_counts(sort=False)
    countax.bar(ntcs.index+shift, ntcs.values, width=width, align='center', color=domains[d]['color'], alpha=0.5)
    shift += 0.3

fig.suptitle(r"DJF mean $V_m$, $\xi$, $RH_{700}$, $V_{sh}, TCGP$")
utils.savefig(os.path.join(basedir, "TCGP_trends.pdf"), bbox_inches='tight')
