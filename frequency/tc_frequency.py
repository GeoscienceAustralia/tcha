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
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import patheffects
from shapely.geometry import box as sbox
from shapely.geometry import LineString

from sklearn.linear_model import LinearRegression

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

def regression_trend(numbers, start_year, end_year):
    """
    Calculate the trend for linear regression of TC numbers for a range of
    years.

    :param numbers: `pandas.DataFrame` that contains the annual number of TCs.
    :param int start_year: First year to calculate regression for
    :param int end_year: Last year to calculate regression for

    :returns: `pandas.DataFrame` containing the slope, intercept and R-squared
    value for each regression.
    """
    years = pd.to_datetime([datetime(y, 1, 1) for y in range(start_year, end_year+1)])
    results = pd.DataFrame(columns=['slope', 'intercept', 'rsq', 'mean', 'var'],
                           index=years)
    for year in years:
        idx = numbers.index >= year.year
        x = numbers.index[idx].values.reshape(-1, 1)
        y = numbers.ID[idx].values
        model = LinearRegression()
        model.fit(x, y)
        slope = model.coef_
        intercept = model.intercept_
        r_sq = model.score(x, y)
        mean = y.mean()
        var =y.var()
        results.loc[year] = [slope[0], intercept, r_sq, mean, var]

    return results

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
dataFile = pjoin(inputPath, r"IDCKMSTM0S - 20240502.csv")
outputPath = r"..\data\frequency"
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
ssc = df.groupby(['IDSEAS']).nunique()
# This just ensures we capture the seasons with a zero count
sc = pd.Series(index=range(ssc.index.min(), ssc.index.max()+1), dtype=int, data=0)
sc.loc[ssc.index] = np.array(ssc.ID.values, dtype='int32')

# Determine the number of severe TCs.
# take the number of TCs with maximum wind speed > 32 m/s
xc = df.groupby(['DISTURBANCE_ID',]).agg({
    'CENTRAL_PRES': 'min',
    'MAX_WIND_GUST': 'max',
    'MAX_WIND_SPD': 'max',
    'ID': 'max', 'IDSEAS': 'max'})
nns = xc[xc['MAX_WIND_SPD'] > 32].groupby('IDSEAS').nunique()['ID']
ns = pd.Series(index=range(nns.index.min(), nns.index.max()+1), dtype=int, data=0)
ns.loc[nns.index] = np.array(nns.values, dtype='int32')

idx = sc.index >= 1970
idx2 = sc.index >= 1980
nsidx = ns.index >= 1970
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.bar(ssc.index[idx], ssc.ID[idx], label="All TCs")
ax.bar(ns.index[nsidx], ns.values[nsidx], color='orange', label="Severe TCs")
ax.axhline(np.mean(ssc.ID[idx]), color='0.5', path_effects=[patheffects.withStroke(linewidth=3, foreground='white')], label="Mean frequency (1970-2020)")
ax.axhline(np.mean(ssc.ID[idx2]), color='r', path_effects=[patheffects.withStroke(linewidth=3, foreground='white')], label="Mean frequency (1985-2020)")
ax.grid(True)
ax.set_yticks(np.arange(0, 21, 2))
ax.set_xlabel("Season")
ax.set_ylabel("Count")
ax.legend(fontsize='small')
plt.text(0.0, -0.1, "Source: http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "TC_frequency.png"), bbox_inches='tight')

# Add regression lines - one for all years >= 1970, another for all years >= 1985
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.bar(ssc.index[idx], ssc.ID[idx], label="All TCs")
ax.bar(ns.index[nsidx], ns.values[nsidx], color='orange', label="Severe TCs")
sns.regplot(x=ssc.index[idx], y=ssc.ID[idx], ax=ax, color='0.5', scatter=False, label='1970-2023 trend')
sns.regplot(x=ssc.index[idx2], y=ssc.ID[idx2], ax=ax, color='r', scatter=False, label='1980-2023 trend')

ax.grid(True)
ax.set_yticks(np.arange(0, 21, 2))
ax.set_xlabel("Season")
ax.set_ylabel("Count")
ax.legend(fontsize='small')
plt.text(0.0, -0.1, "Source: http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "TC_frequency_reg.png"), bbox_inches='tight')
xlim = ax.get_xlim()


ns.to_csv(pjoin(outputPath, "severe_tcs.csv"))
ssc.to_csv(pjoin(outputPath, "all_tcs.csv"))

# Calculate the trends for a range of years:
# Use the IDCKMSTM0S.csv data for this bit
rdf = regression_trend(ssc, 1950, 2000)
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.patch.set_facecolor('white')

ax[0].plot(rdf.index, rdf.slope*10)
ax[0].set_ylabel("Trend [TCs/decade]")
ax[0].grid(True)
ax[1].plot(rdf.index, rdf.rsq)
ax[1].set_ylabel(r"$R^2$")
ax[1].grid(True)
ax[1].xaxis.set_major_locator(locator)
ax[1].xaxis.set_major_formatter(formatter)
plt.text(0.0, -0.1, "Source: http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv",
         transform=ax[1].transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax[1].transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "TC_trends.png"), bbox_inches='tight')

# Calculate and plot the fraction of observations with central pressure, maximum
# wind speed and maximum wind gust by season.
fracdf = df.groupby('IDSEAS').apply(lambda x: x.notnull().mean(), include_groups=False)
fracdf.to_csv(pjoin(outputPath, "fraction_complete.csv"))
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
sns.lineplot(data=fracdf[['CENTRAL_PRES', 'MAX_WIND_SPD', 'MAX_WIND_GUST']][idx], ax=ax)
ax.set_xlabel("Season")
ax.set_ylabel("Fraction complete")
ax.legend(fontsize='small')
ax.grid(True)
plt.text(0.0, -0.1, "Source: http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "TC_fraction_complete.png"), bbox_inches='tight')


"""
Following is an analysis of the Objective TC Reanalysis dataset, created by Joe
Courtney (BoM) in 2017. The data is available online, but requires minor quality
controls for some comments that include carriage return characters.

http://www.bom.gov.au/cyclone/history/database/OTCR_alldata_final_external.csv


"""

dataFile = pjoin(inputPath, r"Objective Tropical Cyclone Reanalysis - QC.csv")
usecols = [0, 1, 2, 7, 8, 11, 12]
colnames = ['NAME', 'DISTURBANCE_ID', 'TM', 'LAT', 'LON',
            'adj. ADT Vm (kn)', 'CP(CKZ(Lok R34,LokPOCI, adj. Vm),hPa)']
dtypes = [str, str, str, float, float, float, float]
otcrdf = pd.read_csv(dataFile, usecols=usecols,
                     dtype=dict(zip(colnames, dtypes)), na_values=[' '], nrows=13743)
otcrdf = filter_tracks_domain(otcrdf, 90, 160, -40, 0, 'DISTURBANCE_ID', 'LAT', 'LON')
colrenames = {'adj. ADT Vm (kn)':'MAX_WIND_SPD',
              'TM': 'datetime',
              'CP(CKZ(Lok R34,LokPOCI, adj. Vm),hPa)': 'CENTRAL_PRES'}
otcrdf.rename(colrenames, axis=1, inplace=True)
otcrdf['datetime'] = pd.to_datetime(otcrdf.datetime, format="%Y-%m-%d %H:%M", errors='coerce')
otcrdf['year'] = pd.DatetimeIndex(otcrdf['datetime']).year
otcrdf['month'] = pd.DatetimeIndex(otcrdf['datetime']).month
otcrdf['season'] = otcrdf[['year', 'month']].apply(lambda x: season(*x), axis=1)

new = otcrdf['DISTURBANCE_ID'].str.split("_", expand=True)
otcrdf['ID'] = new[1]
otcrdf['IDSEAS'] = new[0].str[:6].str.strip('AU').astype(int)
# Calculate the number of unique values in each season:
ssc = otcrdf.groupby(['IDSEAS']).nunique()
otcrsc = pd.Series(index=range(ssc.index.min(), ssc.index.max()+1), dtype=int, data=0)
otcrsc.loc[ssc.index] = np.array(ssc.ID.values, dtype='int32')

# Determine the number of severe TCs.
# take the number of TCs with maximum wind speed > 63 kts
# NOTE: The OTCR data uses knots, not metres/second!
otcrxc = otcrdf.groupby(['DISTURBANCE_ID',]).agg({
    'CENTRAL_PRES': 'min',
    'MAX_WIND_SPD': 'max',
    'ID': 'max', 'IDSEAS': 'max'})

nns = otcrxc[otcrxc['MAX_WIND_SPD'] > 63].groupby('IDSEAS').nunique()['ID']
otcrns = pd.Series(index=range(nns.index.min(), nns.index.max()+1), dtype=int, data=0)
otcrns.loc[nns.index] = np.array(nns.values, dtype='int32')

idx = otcrsc.index >= 1980
idx2 = otcrsc.index >= 1985
nsidx = otcrns.index >= 1980
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.bar(ssc.index[idx], ssc.ID[idx], label="All TCs")
ax.bar(otcrns.index[nsidx], otcrns.values[nsidx], color='orange', label="Severe TCs")
ax.grid(True)
ax.set_yticks(np.arange(0, 21, 2))
ax.set_xlim(xlim)
ax.set_xlabel("Season")
ax.set_ylabel("Count")
ax.legend(fontsize='small')
plt.text(0.0, -0.1, "Source: http://www.bom.gov.au/cyclone/history/database/OTCR_alldata_final_external.csv",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "TC_frequency_otcr.png"), bbox_inches='tight')

# Add regression lines - one for all years >= 1970, another for all years >= 1985
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.bar(ssc.index[idx], ssc.ID[idx], label="All TCs")
ax.bar(otcrns.index[nsidx], otcrns.values[nsidx], color='orange', label="Severe TCs")
sns.regplot(x=ssc.index[idx], y=ssc.ID[idx], ax=ax, color='0.5', scatter=False, label='1970-2020 trend')
sns.regplot(x=ssc.index[idx2], y=ssc.ID[idx2], ax=ax, color='r', scatter=False, label='1985-2020 trend')
ax.grid(True)

ax.set_yticks(np.arange(0, 21, 2))
ax.set_xlim(xlim)
ax.set_xlabel("Season")
ax.set_ylabel("Count")
ax.legend(fontsize='small')
plt.text(0.0, -0.1, "Source: http://www.bom.gov.au/cyclone/history/database/OTCR_alldata_final_external.csv",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "TC_frequency_reg_otcr.png"), bbox_inches='tight')


otcrns.to_csv(pjoin(outputPath, "severe_tcs_otcr.csv"))
otcrsc.to_csv(pjoin(outputPath, "all_tcs_otcr.csv"))

# Calculate and plot the fraction of observations with central pressure, maximum
# wind speed and maximum wind gust by season.
fracdf = otcrdf.groupby('IDSEAS').apply(lambda x: x.notnull().mean(), include_groups=False)
fracdf.to_csv(pjoin(outputPath, "fraction_complete.otcr.csv"))
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
sns.lineplot(data=fracdf[['CENTRAL_PRES', 'MAX_WIND_SPD']][idx], ax=ax)
ax.set_xlabel("Season")
ax.set_ylabel("Fraction complete")
ax.legend(fontsize='small')
ax.grid(True)
plt.text(0.0, -0.1,
         "Source: http://www.bom.gov.au/cyclone/history/database/OTCR_alldata_final_external.csv",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "TC_fraction_complete.otcr.png"), bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
fig.patch.set_facecolor('white')

ax.plot(rdf.index, rdf['mean'], label="Mean frequency")
ax.set_ylabel("Mean [TCs/year] / Variance")
ax.grid(True)
ax.plot(rdf.index, rdf['var'], label='Variance')
ax.legend(fontsize='small')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.text(0.0, -0.1, "Source: http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(outputPath, "TC_trends_mean_var.png"), bbox_inches='tight')
rdf.to_csv(pjoin(outputPath, "regression_trend.csv"))
