from os.path import join as pjoin
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import patheffects
from shapely.geometry import box as sbox
from shapely.geometry import LineString

from  statsmodels.discrete.discrete_model import NegativeBinomial as NB
from  statsmodels.discrete.discrete_model import Poisson

from scipy.stats import nbinom, poisson

sns.set_style('whitegrid')


mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['savefig.dpi'] = 600
locator = mdates.AutoDateLocator(minticks=10, maxticks=20)
formatter = mdates.ConciseDateFormatter(locator)

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
    filterdf = tempfilter.groupby(idcode).filter(
        lambda x: LineString(zip(x[lonname], x[latname])).intersects(domain)
        )
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
                 dtype=dict(zip(colnames, dtypes)),
                 na_values=[' '])

df = filter_tracks_domain(df, 90, 160, -40, 0,
                          'DISTURBANCE_ID', 'LAT', 'LON')


df['TM'] = pd.to_datetime(df.TM, format="%Y-%m-%d %H:%M", errors='coerce')
df['year'] = pd.DatetimeIndex(df['TM']).year
df['month'] = pd.DatetimeIndex(df['TM']).month
df['season'] = df[['year', 'month']].apply(lambda x: season(*x), axis=1)

# Determine season based on the coded disturbance identifier:
# This is of the form "AU201617_<ID>". The first four digits represent the first
# year of the season, the last two the second year of the season
# (which runs November - April)

new = df['DISTURBANCE_ID'].str.split("_", expand=True)
df['ID'] = new[1]
df['IDSEAS'] = new[0].str[:6].str.strip('AU').astype(int)

# Calculate the number of unique values in each season:
sc = df.groupby(['IDSEAS']).nunique()

# Determine the number of severe TCs. 
# take the number of TCs with maximum wind speed > 32 m/s
xc = df.groupby(['DISTURBANCE_ID',]).agg({
    'CENTRAL_PRES': np.min,
    'MAX_WIND_GUST': np.max,
    'MAX_WIND_SPD': np.max,
    'ID':np.max, 'IDSEAS': 'max'})
ns = xc[xc['MAX_WIND_SPD'] > 32].groupby('IDSEAS').nunique()['ID']

idx = sc.index >= 1981


fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

sns.histplot(sc['ID'][idx], bins=np.arange(0, 21), discrete=True, ax=ax, stat='probability')
x = sc.ID[idx]
mu = sc.ID[idx].mean()
sigma = sc.ID[idx].var()
likelihoods = {}

p = mu / sigma
r = p * mu / (1 - p)

likelihoods['nbinom'] = -np.log(x.map(lambda val: nbinom.pmf(val, r, p)).prod())
likelihoods['poisson'] = -np.log(x.map(lambda val: poisson.pmf(val, mu)).prod())
best_fit = min(likelihoods, key=lambda x: likelihoods[x])
print("Best fit:", best_fit)
print(f"NLL: {likelihoods[best_fit]:.2f}")

bins = np.arange(0, 30, 1)
ax.plot(bins, nbinom.pmf(bins, r, p), label="Negative binomial", path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
#ax.plot(bins, poisson.pmf(bins, mu), label="Poisson", path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
ax.legend(fontsize='small')
ax.set_xlabel("Annual TC count")
ax.set_title(f"{x.index.min()} - {x.index.max()}")
plt.text(0.0, -0.1, "Source: http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "TC_frequency_distribution.png"), bbox_inches='tight')
