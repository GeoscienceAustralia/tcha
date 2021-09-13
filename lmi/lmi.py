#!/usr/bin/env python
# coding: utf-8

# # Lifetime maximum intensity of simulated TCs
# 
# * Calculate lifetime maximum intensity (LMI) for TC's 
# * Flag any that achieve LMI at landfall
# * Plot normalised intensity (defined as intensity / LMI) versus time
# * Compare to observed LMI & normalised intensity
# 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
from os import walk
from os.path import join as pjoin

import cftime
from datetime import datetime
datefmt = "%Y-%m-%d %H:%M"

import matplotlib.pyplot as plt
import geopandas as gpd
from Utilities import track
from shapely.geometry import LineString, Point
import shapely.geometry as sg
from shapely.geometry import box as sbox
import numpy as np
import pandas as pd

# From TCRM codebase
from Utilities.loadData import maxWindSpeed, getSpeedBearing

import scipy.stats as stats
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context('talk')

palette = [(1.000, 1.000, 1.000), (0.000, 0.627, 0.235), (0.412, 0.627, 0.235), 
           (0.663, 0.780, 0.282), (0.957, 0.812, 0.000), (0.925, 0.643, 0.016), 
           (0.835, 0.314, 0.118), (0.780, 0.086, 0.118)]
cmap = sns.blend_palette(palette, as_cmap=True)

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

def isLeft(line, point):
    """
    Test whether a point is to the left of a (directed) line segment. 
    
    :param line: :class:`Shapely.geometry.LineString` of the line feature being tested
    :param point: :class:`Shapely.geometry.Point` being tested
    """
    start = Point(line.coords[0])
    end = Point(line.coords[1])

    det = (end.x - start.x) * (point.y - start.y) - (end.y - start.y) * (point.x - start.x)

    if det > 0: return True
    if det <= 0: return False

def isLandfall(gate, tracks):
    crossings = tracks.crosses(gate.geometry)
    landfall = []
    for t in tracks[crossings].itertuples():
        if isLeft(gate.geometry, Point(t.geometry.coords[0])):
            landfall.append(True)
        else:
            landfall.append(False)

    return tracks[crossings][landfall]

def readTracks(trackFile):
    tracks = track.ncReadTrackData(trackFile)
    trackgdf = []
    for t in tracks:
        segments = []
        for n in range(len(t.data) - 1):
            segment = LineString([[t.Longitude[n], t.Latitude[n]],
                                  [t.Longitude[n+1], t.Latitude[n+1]]])
            
            segments.append(segment)
        gdf = gpd.GeoDataFrame.from_records(t.data[:-1])
        gdf['geometry'] = segments
        gdf['category'] = pd.cut(gdf['CentralPressure'], 
                                bins=[0, 930, 955, 970, 985, 990, 1020], 
                                labels=[5,4,3,2,1,0])
        # Calculate pressure difference and normalised intensity
        gdf['pdiff'] = gdf.EnvPressure - gdf.CentralPressure
        gdf['ni'] = gdf.pdiff / gdf.pdiff.max()
        trackgdf.append(gdf)
    trackgdf = pd.concat(trackgdf)
    return trackgdf

def calculateMaxWind(df, dtname='ISO_TIME'):
    """
    Calculate a maximum gust wind speed based on the central pressure deficit and the 
    wind-pressure relation defined in Holland (2008). This uses the function defined in 
    the TCRM code base, and simply passes the correct variables from the data frame
    to the function
    
    This returns a `DataFrame` with an additional column (`vmax`), which represents an estimated
    0.2 second maximum gust wind speed.
    
    'CycloneNumber', 'Datetime', 'TimeElapsed', 'Longitude', 'Latitude',
       'Speed', 'Bearing', 'CentralPressure', 'EnvPressure', 'rMax',
       'geometry', 'category', 'pdiff', 'ni'
    """
    idx = df.CycloneNumber.values
    varidx = np.ones(len(idx))
    varidx[1:][idx[1:]==idx[:-1]] = 0
    
    #dt = (df[dtname] - df[dtname].shift()).fillna(pd.Timedelta(seconds=0)).apply(lambda x: x / np.timedelta64(1,'h')).astype('int64') % (24*60)
    df['vmax'] = maxWindSpeed(varidx, np.ones(len(df.index)), df.Longitude.values, df.Latitude.values,
                              df.CentralPressure.values, df.EnvPressure.values, gustfactor=1.223)
    return df


# In[2]:


best = pd.read_csv('C:/Workspace/data/tc/Objective Tropical Cyclone Reanalysis.csv', 
                       skiprows=[1],
                       usecols=[0, 1,2,7,8,13, 19,52],
                       na_values=[' '],
                       parse_dates=['TM'], infer_datetime_format=True)
best.rename(columns={'DISTURBANCE_ID':'num', 'TM':'datetime', 'LAT': 'lat', 'LON':'lon', 'POCI (Lok, hPa)':'poci',
                     'CENTRAL_PRES':'pmin', 'MAX_WIND_SPD':'vmax'}, inplace=True)
best = best[best.vmax.notnull()]
obstc = filter_tracks_domain(best, 90, 160, -35, -5)


# In[3]:


obstc['deltaT'] = obstc.datetime.diff().dt.total_seconds().div(3600, fill_value=0)


# In[4]:


idx = obstc.num.values
varidx = np.ones(len(idx))
varidx[1:][idx[1:]==idx[:-1]] = 0
speed, bearing = getSpeedBearing(varidx, obstc.lon.values, obstc.lat.values, obstc.deltaT.values)
speed[varidx==1] = 0
obstc['speed'] = speed


# In[5]:


lmidf = obstc.loc[obstc.groupby(["num"])["vmax"].idxmax()] 


# In[6]:


lmidf


# In[7]:


fig, ax = plt.subplots(1, 1)
sns.histplot(lmidf.vmax*1.28*3.6, stat='probability', ax=ax, kde=True)


# In[8]:


lmidf['lmidt'] = pd.to_datetime(obstc.loc[obstc.groupby(["num"])["vmax"].idxmax()]['datetime'])
lmidf['startdt'] = pd.to_datetime(obstc.loc[obstc.index.to_series().groupby(obstc['num']).first().reset_index(name='idx')['idx']]['datetime']).values
lmidf['lmitelapsed'] = (lmidf.lmidt - lmidf.startdt).dt.total_seconds()/3600.
lmidf['initlat'] = obstc.loc[obstc.index.to_series().groupby(obstc['num']).first().reset_index(name='idx')['idx']]['lat'].values
lmidf['initlon'] = obstc.loc[obstc.index.to_series().groupby(obstc['num']).first().reset_index(name='idx')['idx']]['lon'].values
lmidf['lmilat'] = obstc.loc[obstc.groupby(["num"])["vmax"].idxmax()]['lat']
lmidf['lmilon'] = obstc.loc[obstc.groupby(["num"])["vmax"].idxmax()]['lon']


# In[9]:


fig, ax = plt.subplots(1, 1)
sns.distplot(lmidf['lmitelapsed'] ,bins=np.arange(0, 241, 12), ax=ax)
ax.set_xticks(np.arange(0, 241, 48))
ax.set_xlim((0, 240))


# In[10]:


sns.distplot(lmidf.vmax, fit=stats.lognorm)


# In[11]:


ax = sns.jointplot('lmitelapsed', 'vmax', lmidf, joint_kws={'alpha':0.5}).plot_joint(sns.kdeplot, zorder=1000, n_levels=6)
ax.set_axis_labels("Time to LMI (hrs)", 'LMI (m/s)')


# In[12]:


stats.pearsonr(lmidf.lmitelapsed, lmidf.vmax)


# There is a weak relationship between the lifetime maximum intensity and the time taken to achieve LMI (Pearson $r=0.41$). A small number of TCs take a long time (>168 hours/7 days) to achieve LMI. A more detailed quality control of the IBTrACS data would reveal whether these are TCs fo the full period recorded, or achieve TC intensity at some intervening time. The most intense events take at least 120 hours to reach LMI.
# 
# _Not sure if this needs to be discussed_

# In[11]:


#lmidf.to_csv("C:/WorkSpace/data/OTCR.lmi.csv", index=False, date_format="%Y-%m-%d %H:%M")


# In[13]:


filelist = []
datapath = "C:/WorkSpace/data/tcha/tracks"
for (dirpath, dirnames, filenames) in walk(datapath):
    filelist.extend([fn for fn in filenames if fn.endswith('nc')])
    break
nfiles = len(filelist)
print(f"There are {nfiles} track files")


# In[14]:


tracks = []
for i in range(100):
    t = readTracks(pjoin(datapath,filelist[i]))
    t.CycloneNumber = t.CycloneNumber.apply(lambda x: "{0:03d}-{1:05d}".format(i, x))
    tracks.append(t)

tracks = pd.concat(tracks, ignore_index=True)


# In[16]:


simlmidf = tracks.loc[tracks.groupby(["CycloneNumber"])["ni"].idxmax()] 


# In[17]:


simlmidf


# In[18]:


simlmidf['lmidt'] = tracks.loc[tracks.groupby(["CycloneNumber"])["ni"].idxmax()]['Datetime']
simlmidf['startdt'] = tracks.loc[tracks.index.to_series().groupby(tracks['CycloneNumber']).first().reset_index(name='idx')['idx']]['Datetime'].values
simlmidf['lmitelapsed'] = (simlmidf.lmidt - simlmidf.startdt).dt.total_seconds()/3600.
simlmidf['initlat'] = tracks.loc[tracks.index.to_series().groupby(tracks['CycloneNumber']).first().reset_index(name='idx')['idx']]['Latitude'].values
simlmidf['initlon'] = tracks.loc[tracks.index.to_series().groupby(tracks['CycloneNumber']).first().reset_index(name='idx')['idx']]['Longitude'].values
simlmidf['lmilat'] = tracks.loc[tracks.groupby(["CycloneNumber"])["ni"].idxmax()]['Latitude']
simlmidf['lmilon'] = tracks.loc[tracks.groupby(["CycloneNumber"])["ni"].idxmax()]['Longitude']
simlmidf.drop(labels=['geometry'], axis=1, inplace=True)

simlmidf['lmidt'] = simlmidf.lmidt.apply(lambda x: x.strftime(datefmt))
simlmidf['startdt'] = simlmidf.startdt.apply(lambda x: x.strftime(datefmt))
simlmidf['Datetime'] = simlmidf.Datetime.apply(lambda x: x.strftime(datefmt))
simlmidf = calculateMaxWind(simlmidf, dtname='Datetime')


# In[31]:


simlmidf.to_csv("C:/WorkSpace/data/TCHA.lmi.csv", index=False, float_format="%.2f")#date_format="%Y-%m-%d %H:%M")


# In[19]:


fix, ax = plt.subplots(1, 1, figsize=(12,8))
ax.scatter(simlmidf.loc[simlmidf.vmax > 17]['lmitelapsed'], 0.88*simlmidf.loc[simlmidf.vmax > 17]['vmax'], color='r', alpha=0.25, label='Simulated')
ax.scatter(lmidf['lmitelapsed'].values, lmidf['vmax'].values, marker='+', color='k', alpha=0.5, label='Observed')
ax.set_xlabel("Time of maximum intensity (h)")
ax.set_ylabel("Maximum intensity (m/s)")
ax.legend(loc=2)


# In[23]:


fix, ax = plt.subplots(1, 1, figsize=(12,8))
ax.scatter(simlmidf.loc[(simlmidf.lmitelapsed > 12) & (simlmidf.vmax > 17)]['lmitelapsed'], 
           0.88*simlmidf.loc[(simlmidf.lmitelapsed > 12) & (simlmidf.vmax > 17)]['vmax'], 
           color='r', alpha=0.05, label='Simulated')
ax.scatter(lmidf['lmitelapsed'].values, lmidf['vmax'].values, marker='+', color='k', alpha=0.5, label='Observed')
ax.set_xlabel("Time of maximum intensity (h)")
ax.set_ylabel("Maximum intensity (m/s)")
ax.set_xticks(np.arange(0, 480, 48))
ax.set_xlim((0, 432))
ax.legend(loc=2)


# In[21]:


fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.distplot(simlmidf.loc[simlmidf.vmax > 17]['lmitelapsed'], bins=np.arange(0, 241, 12), ax=ax)
ax.set_xticks(np.arange(0, 241, 48))
ax.set_xlim((0, 240))
ax.set_xlabel('Time since genesis (h)')


# In[79]:


import cartopy.crs as ccrs
import cartopy.feature as feature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.ticker as mticker
from matplotlib import colors as mcolors

longrid=np.arange(90, 166, 2.5)
latgrid = np.arange(-25, -0, 2.5)

fig, ax = plt.subplots(1, 2, figsize=(16,9), sharex=True)
bbox_props ={'boxstyle':'round', 'facecolor':'white',
             'alpha':0.75, 'edgecolor':'k'}

ax[0] = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
ax[0].coastlines('50m')
ax[0].set_extent([90, 160, -25, -5])

n, xx, yy = np.histogram2d(lmidf.initlon, lmidf.initlat, [longrid, latgrid])
wn, xx, yy = np.histogram2d(lmidf.initlon, lmidf.initlat, [longrid, latgrid], 
                            weights=lmidf.lmitelapsed,
                            normed=False)

xg, yg = np.meshgrid(xx[:-1], yy[:-1])
cs1 = ax[0].pcolormesh(xg, yg, np.nan_to_num(wn/n).T, cmap=cmap, vmin=0, vmax=240)
gl = ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.MultipleLocator(10)
gl.ylocator = mticker.MultipleLocator(5)
ax[0].grid(True)
ax[0].text(95, -27.5, "Observed", bbox=bbox_props)


ax[1] = plt.subplot(2, 1, 2, projection=ccrs.PlateCarree())
ax[1].coastlines('50m')
ax[1].set_extent([90, 160, -25, -5])

n, xx, yy = np.histogram2d(simlmidf.initlon, simlmidf.initlat, [longrid, latgrid])
wn, xx, yy = np.histogram2d(simlmidf.loc[simlmidf.vmax > 17].initlon, 
                            simlmidf.loc[simlmidf.vmax > 17].initlat, 
                            [longrid, latgrid], 
                            weights=simlmidf.loc[simlmidf.vmax > 17].lmitelapsed,
                            normed=False)
cs2 = ax[1].pcolormesh( xg, yg, np.nan_to_num(wn/n).T, cmap=cmap, vmin=0, vmax=240)

gl = ax[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.MultipleLocator(10)
gl.ylocator = mticker.MultipleLocator(5)
ax[1].grid(True)
ax[1].text(95, -27.5, "Simulated", bbox=bbox_props)

fig.colorbar(cs2, ax=ax, orientation='vertical', 
             shrink=0.75, aspect=20, pad=0.05, extend='max', 
             ticks=range(0,241, 48), label="Time to LMI (hrs)")
ax[0].set_title("Mean time to LMI, based on genesis location")
fig.tight_layout()
None
plt.savefig("C:/Workspace/data/tc/Time_to_LMI_by_genesis.png", bbox_inches='tight')


# In[80]:


fig, ax = plt.subplots(1, 2, figsize=(16,9), sharex=True)

ax[0] = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
ax[0].coastlines('50m')
ax[0].set_extent([90, 160, -25, -5])
n, xx, yy = np.histogram2d(lmidf.lmilon, lmidf.lmilat, [longrid, latgrid])
wn, xx, yy = np.histogram2d(lmidf.lmilon, lmidf.lmilat, [longrid, latgrid], 
                            weights=lmidf.lmitelapsed,
                            normed=False)

xg, yg = np.meshgrid(xx[:-1], yy[:-1])
cs1 = ax[0].pcolormesh(xg, yg, np.nan_to_num(wn/n).T, cmap=cmap, vmin=0, vmax=360)
gl = ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.MultipleLocator(10)
gl.ylocator = mticker.MultipleLocator(5)
ax[0].grid(True)
ax[0].text(95, -27.5, "Observed", bbox=bbox_props)

ax[1] = plt.subplot(2, 1, 2, projection=ccrs.PlateCarree())
ax[1].coastlines('50m')
ax[1].set_extent([90, 160, -25, -5])
n, xx, yy = np.histogram2d(simlmidf.lmilon, simlmidf.lmilat, [longrid, latgrid])
wn, xx, yy = np.histogram2d(simlmidf.loc[simlmidf.vmax > 17].lmilon, 
                            simlmidf.loc[simlmidf.vmax > 17].lmilat, 
                            [longrid, latgrid], 
                            weights=simlmidf.loc[simlmidf.vmax > 17].lmitelapsed,
                            normed=False)
cs2 = ax[1].pcolormesh( xg, yg, np.nan_to_num(wn/n).T, cmap=cmap, vmin=0, vmax=360)

gl = ax[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.MultipleLocator(10)
gl.ylocator = mticker.MultipleLocator(5)
ax[1].grid(True)
ax[1].text(95, -27.5, "Simulated", bbox=bbox_props)
norm = mcolors.Normalize(vmin=0, vmax=240)
cs1.set_norm(norm)
cs2.set_norm(norm)
fig.colorbar(cs1, ax=ax, orientation='vertical', shrink=0.75, aspect=20, extend='max', pad=0.05, 
             label='Time to LMI (hrs)', ticks=range(0, 241, 48))

ax[0].set_title("Mean time to LMI, based on location of LMI")
fig.tight_layout()
plt.savefig("C:/Workspace/data/tc/Time_to_LMI_by_location.png", bbox_inches='tight')


# Here we present the mean time taken to achive lifetime maximum intensity (LMI), based on where LMI is achieved. A 2.5 by 2.5 degree grid is used due to the smaller number of observed LIM values (n=377). The observed time to LMI shows little clear pattern, though the areas off the western coast of Australia do tend to be slightly higher. The simulated tracks display a strong tendency to achieve LMI at higher latitudes (> 17.5$^{\circ}$S), and take longer to achieve LMI in these areas. 
# 
# Comparing geographical areas, observed time to LMI off the northwest coastline is around 96-144 hours, while simulated time to LMI is only around 48-72 hours. On the east coast, there appears less difference between observed and simulated time to LMI, ranging between 48 and 120 hours on average.

# In[81]:


fig, ax = plt.subplots(1, 2, figsize=(16,9), sharex=True)
bbox_props ={'boxstyle':'round', 'facecolor':'white',
             'alpha':0.75, 'edgecolor':'k'}

ax[0] = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
ax[0].coastlines('50m')

n, xx, yy = np.histogram2d(lmidf.lmilon, lmidf.lmilat, [longrid, latgrid])
wn, xx, yy = np.histogram2d(lmidf.lmilon, lmidf.lmilat, [longrid, latgrid], 
                            weights=lmidf.vmax,
                            normed=False)

xg, yg = np.meshgrid(xx[:-1], yy[:-1])

cs1 = ax[0].pcolormesh(xg, yg, np.nan_to_num(wn/n).T, cmap=cmap, vmin=0, vmax=100)
gl = ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabels_bottom = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.MultipleLocator(10)
gl.ylocator = mticker.MultipleLocator(5)
ax[0].grid(True)
ax[0].set_extent([90,160, -25, -5])
ax[0].text(95, -27.5, "Observed", bbox=bbox_props)
#plt.colorbar(cs1, ax=ax1, orientation='horizontal', shrink=0.5, aspect=30, extend='max')

ax[1] = plt.subplot(2, 1, 2, projection=ccrs.PlateCarree())
ax[1].coastlines('50m')
n, xx, yy = np.histogram2d(simlmidf.loc[simlmidf.vmax > 17].lmilon, 
                           simlmidf.loc[simlmidf.vmax > 17].lmilat, [longrid, latgrid])
wn, xx, yy = np.histogram2d(simlmidf.loc[simlmidf.vmax > 17].lmilon, 
                            simlmidf.loc[simlmidf.vmax > 17].lmilat, 
                            [longrid, latgrid], 
                            weights=simlmidf.loc[simlmidf.vmax > 17].vmax,
                            normed=False)
cs2 = ax[1].pcolormesh( xg, yg, np.nan_to_num(wn/n).T, cmap=cmap, vmin=0, vmax=100)

gl = ax[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.MultipleLocator(10)
gl.ylocator = mticker.MultipleLocator(5)
ax[1].grid(True)
ax[1].set_extent([90,160, -25, -5])
#ax[1].set_ylim((-30, -5))
ax[1].text(95, -27.5, "Simulated", bbox=bbox_props)
norm = mcolors.Normalize(vmin=0, vmax=100)
cs1.set_norm(norm)
cs2.set_norm(norm)
fig.colorbar(cs1, ax=ax, orientation='vertical', shrink=0.75, aspect=20, extend='max', pad=0.05,
             label="Lifetime maximum intensity (m/s)")
fig.tight_layout()
None
plt.savefig("C:/Workspace/data/tc/Location_of_LMI.png", bbox_inches='tight')


# Little discernable pattern in the observed location of LMI. The mean LMI is generally evenly distributed throughout the domain, though lower LMI values can be seen over Cape York (145$^{\circ}$E) and south of Indonesia at low latitudes. For the simulated events, there is again a clear trend towards higher LMI at higher latitudes in the Indian Ocean, with highest LMI simulated at 20-25$^{\circ}$S. There is also indications that LMI increases towards the south in the Coral Sea to teh east of Australia.

# In[ ]:




