import os
from os import walk
from os.path import join as pjoin
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.geometry import box as sbox
import numpy as np
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
import seaborn as sns
from os.path import expanduser
import geopandas as gpd
from shapely.geometry import LineString
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from shapely.geometry import box as sbox
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from itertools import product
from shapely.geometry import LineString, Point, Polygon, box as sbox
from datetime import datetime
import xarray as xr
import sys
from os.path import join as pjoin

colorseq=['#FFFFFF', '#ceebfd', '#87CEFA', '#4969E1', '#228B22',
          '#90EE90', '#FFDD66', '#FFCC00', '#FF9933',
          '#FF6600', '#FF0000', '#B30000', '#73264d']
cmap = sns.blend_palette(colorseq, as_cmap=True)

sns.set_style('whitegrid')


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


def gridDensity(tracks: gpd.GeoDataFrame, grid: gpd.GeoDataFrame,
                grid_id_field: str, storm_id_field: str):
    """
    Calculate the count of events passing across each grid cell
    """
    dfjoin = gpd.sjoin(grid, tracks)
    df2 = dfjoin.groupby(grid_id_field).agg({storm_id_field: 'nunique'})
    dfcount = grid.merge(df2, how='left', left_on=grid_id_field, right_index=True)
    dfcount[storm_id_field] = dfcount[storm_id_field].fillna(0)
    dfcount.rename(columns={storm_id_field: 'storm_count'}, inplace=True)
    dfcount['storm_count'] = dfcount['storm_count'].fillna(0)
    return dfcount


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


def gridDensityBootstrap(tracks: gpd.GeoDataFrame, grid: gpd.GeoDataFrame,
                         grid_id_field: str, storm_id_field: str):
    """
    Use a jackknife bootstrap approach to calculate a mean track density
    """
    seasons = tracks.season
    # Remember we are leaving out one season each time...
    nseasons = seasons.max() - seasons.min() - 2
    outframes = []
    for season in seasons.unique():
        dfcount = gridDensity(tracks[tracks.season != season], grid, grid_id_field, storm_id_field)
        retval = dfcount['storm_count'] / nseasons
        outframes.append(retval)

    outdf = pd.concat(outframes, axis=1).apply(['mean', 'std'], axis=1)
    return outdf


def plot_density(dataArray: xr.DataArray, source: str, outputFile: str, xx, yy):
    """
    Plot track density and save figure to file
    """
    plt.close()
    plt.figure()

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.figure.set_size_inches(20, 12)
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
             transform=ax.transAxes, ha='right', fontsize='small', )
    plt.text(0.0, -0.1, f"Source: {source}", transform=ax.transAxes, fontsize='small', ha='left', )
    plt.savefig(outputFile, bbox_inches='tight')
    plt.close()


def plot_density_difference(dataArray: xr.DataArray, source: str, outputFile: str):
    """
    Plot difference in track density and save figure to file
    """
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.figure.set_size_inches(20, 12)
    cb = plt.contourf(xx, yy, dataArray.T, cmap='bwr',
                      transform=ccrs.PlateCarree(),
                      levels=np.arange(-0.2, .21, 0.02),
                      extend='both')
    plt.colorbar(cb, label="Difference TCs/year", shrink=0.9)
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 'small'}
    gl.ylabel_style = {'size': 'small'}
    plt.title("Difference in mean track density (1981-2020 - 1951-2020)")
    plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=ax.transAxes, ha='right', fontsize='xx-small', )
    plt.text(0.0, -0.1, f"Source: {source}", transform=ax.transAxes, fontsize='xx-small', ha='left', )
    #     plt.savefig(outputFile, bbox_inches='tight')
    #     plt.close()
    plt.show()


def addGeometry(trackdf: pd.DataFrame, storm_id_field: str, lonname='LON', latname='LAT') -> gpd.GeoDataFrame:
    """
    Add `LineString` geometry to each separate track
    """
    tracks = []
    for k, t in trackdf.groupby(storm_id_field):
        segments = []
        for n in range(len(t[storm_id_field]) - 1):
            segment = LineString([[t[lonname].iloc[n], t[latname].iloc[n]],
                                  [t[lonname].iloc[n + 1], t[latname].iloc[n + 1]]])
            segments.append(segment)
        gdf = gpd.GeoDataFrame.from_records(t[:-1])
        gdf['geometry'] = segments
        tracks.append(gdf)

    outgdf = pd.concat(tracks)
    return outgdf


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


def countCrossings(gates, tracks, sim):
    gates['sim'] = sim
    for i, gate in enumerate(gates.itertuples(index=False)):
        ncrossings = 0
        l = isLandfall(gate, tracks)
        ncrossings = len(l)
        # crossings = tracks.crosses(gate.geometry)
        # ncrossings = np.sum(tracks.crosses(gate.geometry))
        if ncrossings > 0:
            gates['count'].iloc[i] = ncrossings
    return gates


def plot_tracks(df, latfield, longfield, idfield, title, source, filepath):
    singulars = []
    for group in df.groupby('DISTURBANCE_ID'):
        if len(group[1]) == 1:
            singulars.append(group[0])

    df = df[~df.DISTURBANCE_ID.isin(singulars)]
    mask = (pd.isnull(df.lons_sim)) | (pd.isnull(df.lats_sim))
    badids = pd.unique(df[mask].DISTURBANCE_ID)
    df = df[~df.DISTURBANCE_ID.isin(badids)]

    len(df)

    tdf = df.groupby(idfield).agg({
        longfield: 'first',
        latfield: 'first',
    })

    geometry = df.groupby(idfield).apply(lambda x: LineString(zip(x[longfield], x[latfield])))
    trackdf = gpd.GeoDataFrame(tdf, geometry=geometry)
    fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    trackdf.plot(ax=ax, transform=ccrs.PlateCarree(), color='0.5', linewidth=0.5)

    plt.title(title, fontsize=20)
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
    plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=ax.transAxes, ha='right', fontsize='small', )
    plt.text(0.0, -0.1, f"Source: {source}", transform=ax.transAxes, fontsize='small', ha='left', )
    plt.savefig(filepath)
