import os
import fnmatch
import logging
from os.path import join as pjoin
from datetime import datetime, timedelta

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import patheffects

from shapely.geometry import LineString
from shapely.geometry import box as sbox

import numpy as np
import pandas as pd
import geopandas as gpd

from metutils import convert
from files import flStartLog

mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['savefig.dpi'] = 600

LOGGER = flStartLog('jtwc_frequency.log', logLevel="DEBUG", verbose=True, datestamp=True)

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

def convertLatLon(strval: str) -> float:
    """
    Convert a string representing lat/lon values from '140S to -14.0, etc.

    :param str strval: string containing the latitude or longitude.

    :returns: Latitude/longitude as a float value.

    """
    hemi = strval[-1].upper()
    fval = float(strval[:-1]) / 10.
    if (hemi == 'S') | (hemi == 'W'): 
        fval *= -1
    if (hemi == 'E') | (hemi == 'W'):
        fval = fval % 360
    return fval

def season(year: int, month: int) -> int:
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



COLTYPES = ['|S2', 'i', datetime, 'i', '|S4', 'i', 'f', 'f', 'f', 'f',
            '|S4', 'f', '|S3', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
            '|S1', 'f', '|S3', 'f', 'f', '|S10', '|S1', 'f',
            '|S3', 'f', 'f', 'f', 'f']
COLUNITS = ['', '', '', '', '', '', '', '', 'kts', 'hPa',
            '', 'nm', '', 'nm', 'nm', 'nm', 'nm', 'hPa', 'nm', 'nm', 'kts', 'nm',
            '', '', '', 'degrees', 'kts', '', '', '',
            '', '', '', '', '']

DATEFORMAT = "%Y%m%d%H"
CONVERTERS = {
    'Number': lambda s: s.strip(' ,'),
    "Datetime": lambda s: datetime.strptime(s.strip(' ,'), DATEFORMAT),
    "Latitude": lambda s: float(convertLatLon(s.strip(' ,'))),
    "Longitude": lambda s: float(convertLatLon(s.strip(' ,'))),
    "Windspeed": lambda s: float(s.strip(' ,') or 0),
    "Pressure": lambda s: float(s.strip(' ,') or 0),
    "Status": lambda s: s.strip(' ,'),
    "RAD": lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[11], 'km'),
    "WINDCODE": lambda s: s.strip(' ,'),
    "RAD1": lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[13], 'km'),
    "RAD2": lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[14], 'km'),
    "RAD3": lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[15], 'km'),
    "RAD4": lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[16], 'km'),
    "Poci": lambda s: float(s.strip(' ,') or 0),
    "Roci": lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[18], 'km'),
    "RMAX": lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[19], 'km')
}


def loadFile(filename: str) -> pd.DataFrame:
    """
    Attempt to load a JTWC B-DECK format file. Older format files have fewer
    fields, so only require a subset of column names. If the full list of column
    names is used, a `ValueError` is raised when applying the converters.

    :param filename: Path to the file to load

    :returns: `pd.DataFrame` of the track contained in the file.
    """

    try:
        COLNAMES = ['BASIN','Number', 'Datetime', 'TECHNUM', 'TECH', 'TAU', 'Latitude', 'Longitude', 'Windspeed', 'Pressure',
            'Status', 'RAD', 'WINDCODE', 'RAD1', 'RAD2', 'RAD3', 'RAD4', 'Poci', 'Roci', 'rMax', 'GUSTS', 'EYE',
            'SUBREGION', 'MAXSEAS', 'INITIALS', 'DIR', 'SPEED', 'STORMNAME', 'DEPTH', 'SEAS',
            'SEASCODE', 'SEAS1', 'SEAS2', 'SEAS3', 'SEAS4']
        df = pd.read_csv(filename, names=COLNAMES, delimiter=",",
                         parse_dates=[2], infer_datetime_format=True,
                         skipinitialspace=True, converters=CONVERTERS,
                         error_bad_lines=False)
    except ValueError:
        LOGGER.debug("Older format file")
        try:
            COLNAMES = ['BASIN','Number', 'Datetime','TECHNUM', 'TECH', 'TAU',
                        'Latitude', 'Longitude', 'Windspeed','Pressure', '']
            df = pd.read_csv(filename, names=COLNAMES, delimiter=",",
                             parse_dates=[2], infer_datetime_format=True,
                             skipinitialspace=True, converters=CONVERTERS,
                             error_bad_lines=False)
        except ValueError:
            LOGGER.debug("Intermediate format")
            COLNAMES = ['BASIN','Number', 'Datetime','TECHNUM', 'TECH', 'TAU',
                        'Latitude', 'Longitude', 'Windspeed','Pressure', ]
            df = pd.read_csv(filename, names=COLNAMES, delimiter=",",
                             parse_dates=[2], infer_datetime_format=True,
                             skipinitialspace=True, converters=CONVERTERS,
                             usecols=range(10), error_bad_lines=False)

    return df

inputPath = "X:/georisk/HaRIA_B_Wind/data/raw/from_jtwc/bsh/"
outputPath = r"X:\georisk\HaRIA_B_Wind\projects\tcha\data\derived\tcfrequency\JTWC"

LOGGER.info(f"Input folder is: {inputPath}")
alltracks = []
LOGGER.debug(f"there are {len(os.listdir(inputPath))} files in the input folder")
for file in os.listdir(inputPath):
    if fnmatch.fnmatch(file, "bsh*.txt") or fnmatch.fnmatch(file, "bsh*.dat"):
        LOGGER.debug(f"Loading {file}")
        tempdf = loadFile(pjoin(inputPath, file))
        tempdf['eventid'] = os.path.splitext(file)[0]
        alltracks.append(tempdf)

df = pd.concat(alltracks)
df = filter_tracks_domain(df, 90, 160, -40, 0,
                          'eventid', 'Latitude', 'Longitude')

df['year'] = pd.DatetimeIndex(df['Datetime']).year
df['month'] = pd.DatetimeIndex(df['Datetime']).month
df['season'] = df[['year', 'month']].apply(lambda x: season(*x), axis=1)

tdf = df.groupby('eventid').agg({
        'Datetime':min,
        'Longitude':'first',
        'Latitude':'first',
        'Pressure':np.min,
        'Windspeed':np.max
        })

geometry = df.groupby('eventid').apply(lambda x: LineString(zip(x['Longitude'], x['Latitude'])))
trackdf = gpd.GeoDataFrame(tdf, geometry=geometry)
# Calculate the number of unique values in each season:
sc = df.groupby(['season']).nunique()

xc = df.groupby(['eventid',]).agg({
    'Pressure': np.min,
    'Windspeed': np.max,
    'season': 'max',
    'eventid':'first'})
ns = xc[xc['Windspeed'] > 64].groupby('season').nunique()['eventid']
idx = sc.index >= 1970
idx2 = sc.index >= 1985
nsidx = ns.index >= 1970
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.bar(sc.index[idx], sc.eventid[idx], label="All TCs")
ax.bar(ns.index[nsidx], ns.values[nsidx], color='orange', label="Severe TCs")
ax.axhline(np.mean(sc.eventid[idx]), color='0.5', path_effects=[patheffects.withStroke(linewidth=3, foreground='white')], label="Mean frequency (1970-2020)")
ax.axhline(np.mean(sc.eventid[idx2]), color='r', path_effects=[patheffects.withStroke(linewidth=3, foreground='white')], label="Mean frequency (1985-2020)")
ax.grid(True)
ax.set_yticks(np.arange(0, 21, 2))
ax.set_xlabel("Season")
ax.set_ylabel("Count")
ax.legend(fontsize='small')
plt.text(0.0, -0.1, "Source: https://www.metoc.navy.mil/jtwc/jtwc.html \n(accessed 2021-09-14)",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "TC_frequency.png"), bbox_inches='tight')

# Add regression lines - one for all years >= 1970, another for all years >= 1985
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.bar(sc.index[idx], sc.eventid[idx], label="All TCs")
ax.bar(ns.index[nsidx], ns.values[nsidx], color='orange', label="Severe TCs")
sns.regplot(x=sc.index[idx], y=sc.eventid[idx], ax=ax, color='0.5', scatter=False, label='1970-2020 trend')
sns.regplot(x=sc.index[idx2], y=sc.eventid[idx2], ax=ax, color='r', scatter=False, label='1985-2020 trend')

ax.grid(True)
ax.set_yticks(np.arange(0, 21, 2))
ax.set_xlabel("Season")
ax.set_ylabel("Count")
ax.legend(fontsize='small')
plt.text(0.0, -0.1, "Source: https://www.metoc.navy.mil/jtwc/jtwc.html \n(accessed 2021-09-14)",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "TC_frequency_reg.png"), bbox_inches='tight')

ns.to_csv(pjoin(outputPath, "severe_tcs.csv"))
sc.to_csv(pjoin(outputPath, "all_tcs.csv"))
