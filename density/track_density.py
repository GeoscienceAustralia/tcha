import sys
from os.path import join as pjoin
from itertools import product
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from shapely.geometry import LineString, Point, Polygon, box as sbox

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import seaborn as sns
colorseq=['#FFFFFF', '#ceebfd', '#87CEFA', '#4969E1', '#228B22', 
          '#90EE90', '#FFDD66', '#FFCC00', '#FF9933', 
          '#FF6600', '#FF0000', '#B30000', '#73264d']
cmap = sns.blend_palette(colorseq, as_cmap=True)

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
    :param float minlon: minimum longitude of the bounding box (default=90)
    :param float minlat: minimum latitude of the bounding box (default=-40)
    :param float maxlon: maximum longitude of the bounding box (default=180)
    :param float maxlat: maximum latitude of the bounding box (default=0)
    :param str idcode: Name of the 
    """

    domain = sbox(minlon, minlat, maxlon, maxlat, ccw=False)
    tracks = df.groupby(idcode)
    tempfilter = tracks.filter(lambda x: len(x) > 1)
    filterdf = tempfilter.groupby(idcode).filter(lambda x: LineString(zip(x[lonname], x[latname])).intersects(domain))
    return filterdf

def createGrid(xmin, xmax, ymin, ymax, wide, length):
    cols = list(np.arange(xmin, xmax + wide, wide))
    rows = list(np.arange(ymin, ymax + length, length))
    polygons = []
    for x, y in product(cols[:-1], rows[:-1]):
        polygons.append(
            Polygon([(x, y), (x + wide, y), 
                     (x + wide, y + length), 
                     (x, y + length)]))
    gridid = np.arange(len(polygons))
    grid = gpd.GeoDataFrame({'gridid': gridid,
                             'geometry': polygons})
    return grid


storm_id_field = "DISTURBANCE_ID"
grid_id_field = "gridid"

minlon = 90
maxlon = 160
minlat = -40
maxlat = 0
dx = 0.5
dy = 0.5

lon = np.arange(minlon, maxlon, dx)
lat = np.arange(minlat, maxlat, dy)
xx, yy = np.meshgrid(lon, lat)

dfgrid = createGrid(minlon, maxlon, minlat, maxlat, dx, dy)
dims = (int((maxlon - minlon)/dx), int((maxlat-minlat)/dy))


# Start with the default TC best track database:
inputPath = r"X:\georisk\HaRIA_B_Wind\data\raw\from_bom\tc"
dataFile = pjoin(inputPath, r"IDCKMSTM0S - 20210722.csv")
outputPath = r"X:\georisk\HaRIA_B_Wind\projects\tcha\data\derived\density"
usecols = [0, 1, 2, 7, 8, 16, 49, 53]
colnames = ['NAME', 'DISTURBANCE_ID', 'TM', 'LAT', 'LON',
            'CENTRAL_PRES', 'MAX_WIND_SPD', 'MAX_WIND_GUST']
dtypes = [str, str, str, float, float, float, float, float]
df = pd.read_csv(dataFile, skiprows=4, usecols=usecols,
                 dtype=dict(zip(colnames, dtypes)),
                 na_values=[' '])

df = filter_tracks_domain(df, 90, 160, -40, 0,
                          'DISTURBANCE_ID', 'LAT', 'LON')

tracks = []
for k, t in df.groupby(storm_id_field):
    segments = []
    for n in range(len(t[storm_id_field]) - 1):
        segment = LineString([[t.LON.iloc[n], t.LAT.iloc[n]], [t.LON.iloc[n+1], t.LAT.iloc[n+1]]])
        segments.append(segment)
    gdf = gpd.GeoDataFrame.from_records(t[:-1])
    gdf['geometry'] = segments
    tracks.append(gdf)
    
dfstorm = pd.concat(tracks)

dfjoin = gpd.sjoin(dfgrid, dfstorm) #Spatial join

df2 = dfjoin.groupby(grid_id_field)[storm_id_field].nunique() #Count unique storms for each grid
dfcount = dfgrid.merge(df2, how='left', left_on=grid_id_field, right_index=True) #Merge the resulting geoseries with original grid dataframe
dfcount[storm_id_field] = dfcount[storm_id_field].fillna(0) / 40 #Replace NA with 0 for grids with no storms
dfcount.rename(columns = {storm_id_field:'storm_count'}, inplace = True)
dfcount['storm_count'] = dfcount['storm_count'].fillna(0)

grarray = dfcount['storm_count'].values.reshape(dims)

da = xr.DataArray(grarray, coords=[lon, lat], dims=['lon', 'lat'],
                  attrs=dict(long_name='Mean annual TC frequency',
                             units='1/year'))
ds = xr.Dataset({'density': da},
                attrs=dict(
                    description="Mean annual TC frequency",
                    source="http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv",
                    history=f"{datetime.now():%Y-%m-%d %H:%M}: {sys.argv[0]}"
                ))

ds.to_netcdf(pjoin(outputPath, "mean_track_density.nc"))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.figure.set_size_inches(10,6)
cb = plt.contourf(xx, yy, grarray.T, cmap=cmap, transform=ccrs.PlateCarree(), levels=np.arange(0.05, 1.1, 0.05), extend='both')
plt.colorbar(cb, label="TCs/year", shrink=0.9)
ax.coastlines()
gl = ax.gridlines(draw_labels=True, linestyle=":")
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 'small'}
gl.ylabel_style = {'size': 'small'}

plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}", transform=ax.transAxes, ha='right', fontsize='xx-small',)
plt.text(0.0, -0.1, "Source: http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv", transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.savefig(pjoin(outputPath, "mean_track_density.png"), bbox_inches='tight')