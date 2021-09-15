"""
tc_frequency.py - plot the annual frequency of TCs based on best available data
from BoM.

Source: http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv

Objective Tropical Cyclone Reanalysis:
Source: http://www.bom.gov.au/cyclone/history/database/OTCR_alldata_final_external.csv

NOTE:: A number of minor edits are required to ensure the data is correctly
read. The original source file contains some carriage return characters in the
"COMMENTS" field which throws out the normal `pandas.read_csv` function. If it's
possible to programmatically remove those issues, then we may look to fully
automate this script to read directly from the URL.

"""
from os.path import join as pjoin
from datetime import datetime

import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from cartopy import crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

from shapely.geometry import box as sbox
from shapely.geometry import LineString


mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['savefig.dpi'] = 600

states = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')

def season(year, month):
    """
    Determine the southern hemisphere TC season based on the year and month value.
    If the month is earlier than June, we assign the season to be the preceding year.

    :params int year: Year
    :params int month: Month

    """
    s = year
    if month < 6:
        s = year - 1
    return int(s)

def filter_tracks_domain(df, minlon=90, maxlon=180, minlat=-40, maxlat=0,
                         idcode='num', latname='lat', lonname='lon'):
    """
    Takes a `DataFrame` and filters on the basis of whether the track interscts
    the given domain, which is specified by the minimum and maximum longitude and
    latitude.

    NOTE: This assumes the tracks and bounding box are in the same geographic
    coordinate system (i.e. generally a latitude-longitude coordinate system).
    It will NOT support different projections (e.g. UTM data for the bounds and
    geographic for the tracks).

    NOTE: This doesn't work if there is only one point for the track.

    :param df: :class:`pandas.DataFrame` that holds the TCLV data
    :param float minlon: minimum longitude of the bounding box
    :param float minlat: minimum latitude of the bounding box
    :param float maxlon: maximum longitude of the bounding box
    :param float maxlat: maximum latitude of the bounding box
    """

    domain = sbox(minlon, minlat, maxlon, maxlat, ccw=False)
    tracks = df.groupby(idcode)
    tempfilter = tracks.filter(lambda x: len(x) > 1)
    filterdf = tempfilter.groupby(idcode).filter(lambda x: LineString(zip(x[lonname], x[latname])).intersects(domain))
    return filterdf

# Start with the default TC best track database:
inputPath = r"X:\georisk\HaRIA_B_Wind\data\raw\from_bom\tc"
dataFile = pjoin(inputPath, r"IDCKMSTM0S - 20210722.csv")
outputPath = r"X:\georisk\HaRIA_B_Wind\projects\tcha\data\derived\tcfrequency"
usecols = [0, 1, 2, 7, 8, 16, 49, 53]
colnames = ['NAME', 'DISTURBANCE_ID', 'TM', 'LAT', 'LON',
            'CENTRAL_PRES', 'MAX_WIND_SPD', 'MAX_WIND_GUST']
dtypes = [str, str, str, float, float, float, float, float]
df = pd.read_csv(dataFile, skiprows=4, usecols=usecols,
                 dtype=dict(zip(colnames, dtypes)), na_values=[' '])

df = filter_tracks_domain(df, 90, 160, -40, 0, 'DISTURBANCE_ID', 'LAT', 'LON')

df['TM'] = pd.to_datetime(df.TM, format="%Y-%m-%d %H:%M", errors='coerce')
df['year'] = pd.DatetimeIndex(df['TM']).year
df['month'] = pd.DatetimeIndex(df['TM']).month
df['season'] = df[['year', 'month']].apply(lambda x: season(*x), axis=1)
df['LON'] = df['LON'] % 360

df = df[df.season>=1980]

tdf = df.groupby('DISTURBANCE_ID').agg({
        'TM':min,
        'LON':'first',
        'LAT':'first',
        'CENTRAL_PRES':np.min,
        'MAX_WIND_SPD':np.max,
        'MAX_WIND_GUST':np.max
        })

geometry = df.groupby('DISTURBANCE_ID').apply(lambda x: LineString(zip(x['LON'], x['LAT'])))
trackdf = gpd.GeoDataFrame(tdf, geometry=geometry)
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection':ccrs.PlateCarree()})
trackdf.plot(ax=ax, transform=ccrs.PlateCarree(), color='0.5', linewidth=0.5, label="Historical TC tracks")

# Add BoM Area of Responsibility
tcaor = gpd.read_file(r"X:\georisk\HaRIA_B_Wind\data\raw\boundaries\bom\IDM00005.shp")
tcaor.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='r', alpha=0.5, zorder=1000, label="BoM AOR")

# Add proposed hazard assessment domain:
domain = sbox(90, -36, 160, -5, ccw=False)
ax.add_geometries([domain], crs=ccrs.PlateCarree(), edgecolor='green', facecolor='none', label="TCHA domain")
ax.legend()
ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
ax.coastlines(resolution='10m')
gl = ax.gridlines(draw_labels=True, linestyle=":")
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.top_labels = False
gl.right_labels = False
ax.set_extent((80, 170, -40, 0), crs=ccrs.PlateCarree())

plt.text(0.0, -0.1, "Source: http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "TC_tracks.IDCKMSTM0S.1980-2020.png"), bbox_inches='tight')
