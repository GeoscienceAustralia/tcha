"""
Read in the SPEArTC database and extract events that pass within a defined
distance of a specific location.

Save data and plot a basic map showing the location and tracks of events that
pass that locationn

"""

import os
import pandas as pd
import geopandas as gpd
from datetime import datetime
from shapely.geometry import LineString

dataPath = "C:/WorkSpace/data/tc"
dataFile = os.path.join(dataPath, "SPEArTC_December_2020.csv")
usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
colnames = ['serial_num', 'season', 'num', 'basin', 'sub_basin', 'name',
            'ISO_Time', 'nature', 'lat', 'lon', 'vmax', 'pmin', 'centre']

df = pd.read_csv(dataFile, skiprows=47, na_values=[' '], usecols=usecols,
                 names=colnames, parse_dates=[6])

df['lon'] = df['lon'] % 360
df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
linedf = df.groupby(['serial_num'], as_index=False)['geometry'].apply(lambda x: LineString(x.tolist()) if x.size > 1 else x.tolist())
linedf = gpd.GeoDataFrame(linedf, geometry='geometry')

stationFile = "C:/WorkSpace/tcrm/input/stationlist.shp"
stndf = gpd.read_file(stationFile)
# Buffer our selected station. In this case, it's Eroro, PNG
# Buffer distance is in degrees
stn = stndf.iloc[13870]['geometry'].buffer(2.0)

tcs = linedf[linedf.intersects(stn)]

outdf = df[df['serial_num'].isin(tcs['serial_num'])].drop(columns='geometry')
outdf.to_csv(os.path.join(dataPath, 'SPEArTC_Kimbe.csv'), index=False)

# Create a GeoDataFrame
trackgdf = []
for k, t in outdf.groupby('serial_num'):
    segments = []
    for n in range(len(t.num) - 1):
        segment = LineString([[t.lon.iloc[n], t.lat.iloc[n]],
                              [t.lon.iloc[n+1], t.lat.iloc[n+1]]])
        segments.append(segment)

    tgdf = gpd.GeoDataFrame.from_records(t[:-1])
    tgdf['geometry'] = segments
    tgdf['category'] = pd.cut(tgdf['vmax'],
                              bins=[0, 34, 48, 63, 86, 108, 500],
                              labels=[0, 1, 2, 3, 4, 5])
    trackgdf.append(tgdf)

trackgdf = pd.concat(trackgdf)
trackgdf['category'] = trackgdf['category'].cat.codes
trackgdf = trackgdf.set_crs('EPSG:4326')
trackgdf.to_file(os.path.join(dataPath, 'SPEArTC_Kimbe.json'), driver='GeoJSON')

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

figsize=(10, 8)
proj = ccrs.PlateCarree(central_longitude=180)
proj_proj4 = proj.proj4_init
dataproj = ccrs.PlateCarree(central_longitude=0)
dataproj_proj4 = dataproj.proj4_init
trackgdf = trackgdf.set_crs(proj_proj4, allow_override=True)

borders = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none')

fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection':proj})
trackgdf.plot(ax=ax, column='category', linewidth=1, legend=True,
              categorical=True, cmap='viridis',
              legend_kwds={'title':'TC intensity', 'fontsize':'x-small'},
              transform=dataproj)
plt.plot(*stn.exterior.xy, zorder=1000, color='r', transform=dataproj)
stnpt = stndf.iloc[13870]['geometry']
plt.plot(stnpt.x, stnpt.y, color='r', marker='*', transform=dataproj)

ax.coastlines(resolution='10m')
ax.add_feature(borders, edgecolor='k', linewidth=0.5)
gl = ax.gridlines(draw_labels=True, linestyle='--', crs=proj,)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 'x-small'}
gl.ylabel_style = {'size': 'x-small'}
ax.set_extent((140, 180, -25, 0), crs=dataproj)
plt.text(-0.05, -0.1, "Source: http://apdrc.soest.hawaii.edu/projects/speartc/",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.tight_layout()
plt.savefig(os.path.join(dataPath, "SPEArTC_Kimbe.png"), bbox_inches='tight')
