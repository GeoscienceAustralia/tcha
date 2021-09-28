"""
Read the objective TC reanalysis data from CSV and extract the lifetime maximum
intensity of each separate TC. 

Source:
http://www.bom.gov.au/cyclone/history/database/OTCR_alldata_final_external.csv

NOTES:
The original data downloaded includes a comments field. For some records, there
is a carriage return in the comment, which leads to blank records in the data.
Please check the data and remove these carriage returns (or the blank lines)
prior to running the script.

"""
from os.path import join as pjoin
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.geometry import box as sbox
from scipy.stats import kendalltau


import seaborn as sns

# From TCRM codebase
from Utilities.loadData import getSpeedBearing
from metutils import convert


DATEFMT = "%Y-%m-%d %H:%M"

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

def filter_tracks_domain(df, minlon=90, maxlon=180, minlat=-40, maxlat=0):
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
    tracks = df.groupby('num')
    tempfilter = tracks.filter(lambda x: len(x) > 1)
    tempfilter.head()
    filterdf = tempfilter.groupby('num').filter(lambda x: LineString(zip(x['lon'], x['lat'])).intersects(domain))
    return filterdf


inputPath = "X:/georisk/HaRIA_B_Wind/data/raw/from_bom/tc"
outputPath = "X:/georisk/HaRIA_B_Wind/projects/tcha/data/derived/lmi/


dataFile = pjoin(inputPath, r"IDCKMSTM0S - 20210722.csv")
source = "http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv"

usecols = [0, 1, 2, 7, 8, 11, 12, 13]
usecols = [0, 1, 2, 7, 8, 16, 49, 53]
colnames = ['NAME', 'DISTURBANCE_ID', 'TM', 'LAT', 'LON',
            'CENTRAL_PRES', 'MAX_WIND_SPD', 'MAX_WIND_GUST']
dtypes = [str, str, str, float, float, float, float, float]
df = pd.read_csv(dataFile, skiprows=4, usecols=usecols,
                 dtype=dict(zip(colnames, dtypes)),
                 na_values=[' '])
colrenames = {'DISTURBANCE_ID': 'num',
              'TM': 'datetime',
              'LON': 'lon', 'LAT': 'lat',
              'MAX_WIND_SPD':'vmax',
              'CENTRAL_PRES': 'pmin',}
df.rename(colrenames, axis=1, inplace=True)
df['vmax'] = convert(df.vmax.values, "mps", "kts")

df['datetime'] = pd.to_datetime(df.datetime, format="%Y-%m-%d %H:%M", errors='coerce')
df['year'] = pd.DatetimeIndex(df['datetime']).year
df['month'] = pd.DatetimeIndex(df['datetime']).month
df['season'] = df[['year', 'month']].apply(lambda x: season(*x), axis=1)

df = df[df.vmax.notnull()]
obstc = filter_tracks_domain(df, 90, 160, -35, -5)

obstc['deltaT'] = obstc.datetime.diff().dt.total_seconds().div(3600, fill_value=0)
idx = obstc.num.values
varidx = np.ones(len(idx))
varidx[1:][idx[1:] == idx[:-1]] = 0
speed, bearing = getSpeedBearing(varidx, obstc.lon.values, obstc.lat.values, obstc.deltaT.values)
speed[varidx == 1] = 0
obstc['speed'] = speed

lmidf = obstc.loc[obstc.groupby(["num"])["vmax"].idxmax()]

lmidf['lmidt'] = pd.to_datetime(obstc.loc[obstc.groupby(["num"])["vmax"].idxmax()]['datetime'])
lmidf['lmidtyear'] = pd.DatetimeIndex(lmidf['lmidt']).year
lmidf['lmidtmonth'] = lmidf['lmidt'].apply(lambda x: datetime.strftime(x, "%B"))
lmidf['startdt'] = pd.to_datetime(obstc.loc[obstc.index.to_series().groupby(obstc['num']).first().reset_index(name='idx')['idx']]['datetime']).values
lmidf['lmitelapsed'] = (lmidf.lmidt - lmidf.startdt).dt.total_seconds()/3600.
lmidf['initlat'] = obstc.loc[obstc.index.to_series().groupby(obstc['num']).first().reset_index(name='idx')['idx']]['lat'].values
lmidf['initlon'] = obstc.loc[obstc.index.to_series().groupby(obstc['num']).first().reset_index(name='idx')['idx']]['lon'].values
lmidf['lmilat'] = obstc.loc[obstc.groupby(["num"])["vmax"].idxmax()]['lat']
lmidf['lmilon'] = obstc.loc[obstc.groupby(["num"])["vmax"].idxmax()]['lon']
lmidf.to_csv(pjoin(outputPath, "lmi.20210928.csv"), index=False, date_format=DATEFMT)

# Plot distribution of LMI:
fig, ax = plt.subplots(figsize=(10, 8))
sns.histplot(lmidf.vmax, ax=ax, kde=True)
ax.set_xlabel("Maximum wind speed [kts]")
plt.text(0.0, -0.1, f"Source: {source}",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "lmivmax_dist.png"), bbox_inches='tight')

# Plot distribution of latitude of LMI
fig, ax = plt.subplots(figsize=(10, 8))
sns.histplot(lmidf.lmilat, ax=ax, kde=True)
ax.set_xlabel(r"Latitude of LMI [$^{\circ}$S]")
plt.text(0.0, -0.1, f"Source: {source}",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "lmilat_dist.png"), bbox_inches='tight')

# Plot scatter plot of latitude of LMI vs year:
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x='lmidtyear', y='lmilat', data=lmidf, ax=ax)
ax.grid(True)
ax.set_xlabel("Season")
ax.set_ylabel(r"Latitude of LMI [$^{\circ}$S]")
plt.text(-0.1, -0.1, f"Source: {source}",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "lmilat_timeseries.png"), bbox_inches='tight')

# Plot scatter plot of vmax vs year:
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x='lmidtyear', y='vmax', data=lmidf, ax=ax)
ax.grid(True)
ax.set_xlabel("Season")
ax.set_ylabel(r"Maximum wind speed [kts]")
plt.text(-0.1, -0.1, f"Source: {source}",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "lmivmax_timeseries.png"), bbox_inches='tight')

# Plot a scatter plot of LMI vs time to LMI:
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.jointplot(
    x='lmitelapsed', y='vmax', data=lmidf, joint_kws={'alpha':0.5}
    ).plot_joint(sns.kdeplot, zorder=1000, n_levels=6)

ax.set_axis_labels("Time to LMI [hrs]", 'LMI [kts]')
plt.text(-0.1, -0.1, f"Source: {source}",
         transform=ax.ax_joint.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.1, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.ax_joint.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "lmivmax_timeelapsed.png"), bbox_inches='tight')

g = sns.FacetGrid(lmidf, col="month", col_wrap=4, ylim=(-30, -5))
g.map(sns.regplot, "season", "lmilat")
g.set_axis_labels("Season", r"Latitude of LMI [$^{\circ}$S]")
for col_key,ax in g.axes_dict.items():
    ax.set_title(f"{datetime.strftime(datetime(2000, col_key, 1), '%B')}")
g.fig.text(0.01, 0.01, f"Source: {source}", transform=g.fig.transFigure,
           fontsize='xx-small', ha='left',)
plt.text(0.99, 0.01, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=g.fig.transFigure, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "llmi_monthlytrend.png"), bbox_inches='tight')


# This calculates the trend and significance of the trend for each month
mdf = lmidf.groupby('month').apply(
    lambda x: pd.Series(kendalltau(x['lmidt'], x['lmilat']),['tau', 'pval']))

mdf.to_csv(pjoin(outputPath, "llmi_monthlytrend.csv"), float_format="%.4f")

tbl = lmidf.groupby('season').agg(
    {'num': len,
     'vmax': np.mean,
     'lmilat': np.mean,
     'lmitelapsed': np.mean}
    )
tbl.to_csv(pjoin(outputPath, "seasonal_meanlmi.csv"), float_format="%.4f")

