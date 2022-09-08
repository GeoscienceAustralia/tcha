#!/usr/bin/env python
# coding: utf-8

from os.path import join as pjoin
from datetime import datetime

import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.geometry import box as sbox
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

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

def countCrossings(gates, tracks, sim, dtfield="ISO_TIME"):
    gates['sim'] = sim
    startyear = tracks[dtfield].dt.year.min()
    endyear = tracks[dtfield].dt.year.max()
    nyears = endyear - startyear + 1
    for i, gate in enumerate(gates.itertuples(index=False)):
        ncrossings = 0
        l = isLandfall(gate, tracks)
        ncrossings = len(l)
        #crossings = tracks.crosses(gate.geometry)
        #ncrossings = np.sum(tracks.crosses(gate.geometry))
        if ncrossings > 0:
            gates.loc[i, 'count'] = ncrossings
            gates.loc[i, 'meanlfintensity'] = l['pmin'].mean()
            gates.loc[i, 'minlfintensity'] = l['pmin'].min()
            cathist, bins = np.histogram(l['category'].values,
                                         bins=[0, 1, 2, 3, 4, 5, 6])
            gates.loc[i, ['cat1', 'cat2', 'cat3', 'cat4', 'cat5']] = cathist[1:]
        else:
            gates.loc[i, 'meanlfintensity'] = np.nan
            gates.loc[i, 'minlfintensity'] = np.nan
            gates.loc[i, ['cat1', 'cat2', 'cat3', 'cat4', 'cat5']] = 0
    gates['prob'] = gates['count']/gates['count'].sum()
    gates['rate'] = gates['count']/nyears
    return gates

def countLandfall(tcdf, gates):
    trackgdf = []
    for k, t in tcdf.groupby('num'):
        segments = []
        for n in range(len(t.num) - 1):
                segment = LineString([[t.lon.iloc[n], t.lat.iloc[n]],
                                      [t.lon.iloc[n+1], t.lat.iloc[n+1]]])
                segments.append(segment)
        gdf = gpd.GeoDataFrame.from_records(t[:-1])
        gdf['geometry'] = segments
        gdf['category'] = pd.cut(gdf['pmin'], 
                                 bins=[0, 930, 955, 970, 985, 990, 1020], 
                                 labels=[5,4,3,2,1,0])
        trackgdf.append(gdf)

    trackgdf = pd.concat(trackgdf)
    gatedf = gates.copy()
    gatedf = countCrossings(gatedf, trackgdf, 0)
    return gatedf


# Start with reading in the gates into a GeoDataFrame, and adding some
# additional attributes. This GeoDataFrame will be duplicated for each
# simulation, then aggregated for the summary statistics.

gates = gpd.read_file("C:/WorkSpace/tcha/data/gates.shp")
gates['sim'] = 0
gates['count'] = 0
gates['meanlfintensity'] = np.nan
gates['minlfintensity'] = np.nan
gates['cat1'] = 0
gates['cat2'] = 0
gates['cat3'] = 0
gates['cat4'] = 0
gates['cat5'] = 0

source = "doi.org/10.25921/82ty-9e16"
source_path = "C:/Workspace/data/tc/"
output_path = "C:/WorkSpace/data/tcha"
obstc80 = pd.read_csv(pjoin(source_path, 'ibtracs.since1980.list.v04r00.csv'),
                       skiprows=[1],
                       usecols=[0,6,8,9,11,113],
                       na_values=[' '],
                       parse_dates=[1])
obstc80.rename(columns={'SID':'num', 'LAT': 'lat', 'LON':'lon', 'WMO_PRES':'pmin', 'BOM_POCI':'poci'}, inplace=True)
obstc80 = filter_tracks_domain(obstc80)

obstc = pd.read_csv(pjoin(source_path, 'ibtracs.ALL.list.v04r00.csv'),
                       skiprows=[1],
                       usecols=[0,6,8,9,11,113],
                       na_values=[' '],
                       parse_dates=[1])
obstc.rename(columns={'SID':'num', 'LAT': 'lat', 'LON':'lon', 'WMO_PRES':'pmin', 'BOM_POCI':'poci'}, inplace=True)
obstc = filter_tracks_domain(obstc)

gatedf80 = countLandfall(obstc80, gates)
gatedfAll = countLandfall(obstc, gates)

width=0.4
fig, ax = plt.subplots(3,1,figsize=(12,16),sharex=True)
cat12 = np.add(gates['cat1'], gatedf80['cat2']).tolist()
cat123 = np.add(cat12, gatedf80['cat3']).tolist()
cat1234 = np.add(cat123, gatedf80['cat4']).tolist()
ax[0].bar(gatedf80['gate'],gatedf80['cat1'], color='b', label="Cat 1")
ax[0].bar(gatedf80['gate'], gatedf80['cat2'], bottom=gatedf80['cat1'], color='g', label='Cat 2')
ax[0].bar(gatedf80['gate'], gatedf80['cat3'], bottom=cat12, color='y', label='Cat 3')
ax[0].bar(gatedf80['gate'], gatedf80['cat4'], bottom=cat123, color='orange', label='Cat 4')
ax[0].bar(gatedf80['gate'], gatedf80['cat5'], bottom=cat1234, color='r', label='Cat 5')

ax[0].legend()
ax[0].set_ylabel("Number of TCs")
ax[1].plot(gatedf80['gate'], gatedf80['minlfintensity'], label='Minimum landfall intensity')
ax[1].plot(gatedf80['gate'], gatedf80['meanlfintensity'], color='r', label='Mean landfall intensity')
#ax[1].fill_between(gates['gate'], gates['meanlfintensity_q10'],
#                   gates['meanlfintensity_q90'],color='r', alpha=0.25)

ax[1].legend(loc=2)
ax[1].set_ylim((900, 1020))
ax[1].set_ylabel("Pressure (hPa)")
ax[2].bar(gatedf80['gate'], gatedf80['prob'])
ax[2].set_xlim((0,48))
ax[2].set_xticks(np.arange(0,49,2))
ax[2].set_yticks(np.arange(0,0.11,.02))
ax[2].set_xticklabels(gatedf80['label'][::2], rotation='vertical')
ax[2].set_ylabel("Mean probability of landfall")
plt.text(0.99, 0.01, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=fig.transFigure, ha='right', fontsize='xx-small',)
plt.text(0.01, 0.01, f"Source: {source}", transform=fig.transFigure, fontsize='xx-small', ha='left',)
plt.savefig(pjoin(output_path, "obs_landfall_intensity.png"), bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
ax.plot(gatedf80.gate, gatedf80['prob'], label="Post-1980")
ax.plot(gatedfAll.gate, gatedfAll['prob'], label="Full record")
ax.set_xlim((0, 48))
ax.set_xticks(np.arange(0, 49,2))
ax.set_yticks(np.arange(0, 0.11, .01))
ax.set_ylabel('Probability of landfall')
ax.set_xticklabels(gatedf80['label'][::2], rotation='vertical')
plt.legend()
plt.text(0.99, 0.01, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=fig.transFigure, ha='right', fontsize='xx-small',)
plt.text(0.01, 0.01, f"Source: {source}", transform=fig.transFigure, fontsize='xx-small', ha='left',)
plt.savefig(pjoin(output_path, "obs_landfall_prob.png"), bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
ax.plot(gatedf80.gate, gatedf80['rate'], label="Post-1980")
ax.plot(gatedfAll.gate, gatedfAll['rate'], label="Full record")

ax.set_xlim((0,48))
ax.set_xticks(np.arange(0, 49, 2))
ax.set_yticks(np.arange(0, 0.5, .05))
ax.set_ylabel("Annual landfall rate [TCs/yr]")
ax.set_xticklabels(gatedf80['label'][::2], rotation='vertical')
plt.legend()
plt.text(0.99, -0.05, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=fig.transFigure, ha='right', fontsize='xx-small',)
plt.text(0.01, -0.05, f"Source: {source}", transform=fig.transFigure, fontsize='xx-small', ha='left',)
plt.savefig(pjoin(output_path, "obs_landfall_rate.png"), bbox_inches='tight')

gatedf80.to_file(pjoin(output_path, "obs_landfall.shp"))
gatedf80_nogeom = pd.DataFrame(gatedf80.drop(columns='geometry'))
gatedf80_nogeom.to_csv(pjoin(output_path, "obs_landfall.csv"), index=False)
gatedfAll.to_file(pjoin(output_path, "obs_landfall_full.shp"))
gatedfAll_nogeom = pd.DataFrame(gatedfAll.drop(columns='geometry'))
gatedfAll_nogeom.to_csv(pjoin(output_path, "obs_landfall_full.csv"), index=False)