"""
This script calculates and plots the mean annual TC frequency in a grid across the simulation domain.

Runs both the main BoM best track and the Objective TC Reanalysis dataset (Courtney et. al. 2018). 

Data are stored in a netcdf file (both mean annual frequency and total TC count per grid cell)

"""

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
    """
    Create a uniform grid across a specified extent, returning a 
    `gpd.GeoDataFrame` of the grid to facilitate a spatial join process.

    :param float xmin: minimum longitude of the grid
    :param float xmax: maximum longitude of the grid
    :param float ymin: minimum latitude of the grid
    :param float ymax: maximum latitude of the grid
    :param float wide: longitudinal extent of each grid cell
    :param float length: latitudinal extent of each grid cell

    :returns: `gpd.GeoDataFrame` of collection of polygons representing the
    grid.
    """
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

def gridDensity(tracks: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, 
                grid_id_field: str, storm_id_field: str):
    """
    Calculate the count of events passing across each grid cell
    """
    dfjoin = gpd.sjoin(grid, tracks)
    df2 = dfjoin.groupby(grid_id_field)[storm_id_field].nunique()
    dfcount = grid.merge(df2, how='left', left_on=grid_id_field, right_index=True)
    dfcount[storm_id_field] = dfcount[storm_id_field].fillna(0)
    dfcount.rename(columns = {storm_id_field:'storm_count'}, inplace = True)
    dfcount['storm_count'] = dfcount['storm_count'].fillna(0)
    return dfcount

def plot_density(dataArray: xr.DataArray, source: str, outputFile: str):
    """
    Plot track density and save figure to file
    """
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.figure.set_size_inches(10,6)
    cb = plt.contourf(xx, yy, dataArray.T, cmap=cmap,
                      transform=ccrs.PlateCarree(),
                      levels=np.arange(0.05, .51, 0.025),
                      extend='both')
    plt.colorbar(cb, label="TCs/year", shrink=0.9)
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 'small'}
    gl.ylabel_style = {'size': 'small'}

    plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=ax.transAxes, ha='right', fontsize='xx-small',)
    plt.text(0.0, -0.1, f"Source: {source}", transform=ax.transAxes, fontsize='xx-small', ha='left',)
    plt.savefig(outputFile, bbox_inches='tight')
    plt.close()

def addGeometry(trackdf: pd.DataFrame, storm_id_field: str, lonname='LON', latname='LAT') -> gpd.GeoDataFrame:
    """
    Add `LineString` geometry to each separate track
    """
    tracks = []
    for k, t in df.groupby(storm_id_field):
        segments = []
        for n in range(len(t[storm_id_field]) - 1):
            segment = LineString([[t[lonname].iloc[n], t[latname].iloc[n]], 
                                  [t[lonname].iloc[n+1], t[latname].iloc[n+1]]])
            segments.append(segment)
        gdf = gpd.GeoDataFrame.from_records(t[:-1])
        gdf['geometry'] = segments
        tracks.append(gdf)
    
    outgdf = pd.concat(tracks)
    return outgdf

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

df['TM']= pd.to_datetime(df.TM, format="%Y-%m-%d %H:%M", errors='coerce')
df['year'] = pd.DatetimeIndex(df['TM']).year
df['month'] = pd.DatetimeIndex(df['TM']).month
df['season'] = df[['year', 'month']].apply(lambda x: season(*x), axis=1)
df = df[df.season >= 1981]
nseasons = df.season.max() - df.season.min() + 1

dfstorm = addGeometry(df, storm_id_field, 'LON', 'LAT')
dfcount = gridDensity(dfstorm, dfgrid, grid_id_field, storm_id_field)
dfcount['density'] = dfcount['storm_count'] / nseasons

tdarray = dfcount['density'].values.reshape(dims) # td = tropical cyclone *density*
tcarray = dfcount['storm_count'].values.reshape(dims) # tc = tropical cyclone *count*

da = xr.DataArray(tdarray, coords=[lon, lat], dims=['lon', 'lat'],
                  attrs=dict(long_name='Mean annual TC frequency',
                             units='TCs/year'))
dc = xr.DataArray(tcarray, coords=[lon, lat], dims=['lon', 'lat'],
                  attrs=dict(long_name='Total TC count',
                             units='TCs'))
ds = xr.Dataset({'density': da,
                 'count': dc},
                attrs=dict(
                    description="Mean annual TC frequency",
                    source="http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv",
                    history=f"{datetime.now():%Y-%m-%d %H:%M}: {sys.argv[0]}"
                ))

ds.to_netcdf(pjoin(outputPath, "mean_track_density.nc"))
plot_density(da, "http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv", pjoin(outputPath, "mean_track_density.png"))


dataFile = pjoin(inputPath, r"Objective Tropical Cyclone Reanalysis - QC.csv")
source="http://www.bom.gov.au/cyclone/history/database/OTCR_alldata_final_external.csv"
usecols = [0, 1, 2, 7, 8, 11, 12]
colnames = ['NAME', 'DISTURBANCE_ID', 'TM', 'LAT', 'LON',
            'adj. ADT Vm (kn)', 'CP(CKZ(Lok R34,LokPOCI, adj. Vm),hPa)']
dtypes = [str, str, str, float, float, float, float]
df = pd.read_csv(dataFile, usecols=usecols,
                     dtype=dict(zip(colnames, dtypes)), na_values=[' '], nrows=13743)
df['TM']= pd.to_datetime(df.TM, format="%Y-%m-%d %H:%M", errors='coerce')
df['year'] = pd.DatetimeIndex(df['TM']).year
df['month'] = pd.DatetimeIndex(df['TM']).month
df['season'] = df[['year', 'month']].apply(lambda x: season(*x), axis=1)
df = df[df.season >= 1981]
nseasons = df.season.max() - df.season.min() + 1
dfstorm = addGeometry(df, storm_id_field, 'LON', 'LAT')
dfcount = gridDensity(dfstorm, dfgrid, grid_id_field, storm_id_field)
dfcount['density'] = dfcount['storm_count'] / nseasons

tdarray = dfcount['density'].values.reshape(dims) # td = tropical cyclone *density*
tcarray = dfcount['storm_count'].values.reshape(dims) # tc = tropical cyclone *count*

da = xr.DataArray(tdarray, coords=[lon, lat], dims=['lon', 'lat'],
                  attrs=dict(long_name='Mean annual TC frequency',
                             units='TCs/year'))
dc = xr.DataArray(tcarray, coords=[lon, lat], dims=['lon', 'lat'],
                  attrs=dict(long_name='Total TC count',
                             units='TCs'))
ds = xr.Dataset({'density': da,
                 'count': dc},
                attrs=dict(
                    description="Mean annual TC frequency",
                    source=source,
                    history=f"{datetime.now():%Y-%m-%d %H:%M}: {sys.argv[0]}"
                ))

ds.to_netcdf(pjoin(outputPath, "mean_track_density.OTCR.nc"))
plot_density(da, source, pjoin(outputPath, "mean_track_density.OTCR.png"))
