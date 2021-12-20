import numpy as np
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Polygon, box as sbox
from matplotlib import pyplot as plt
import xarray as xr
import seaborn as sns
from track_utils import createGrid, countCrossings, filter_tracks_domain, addGeometry, gridDensity, plot_density
import sys
sys.path.insert(0, sys.path.insert(0, os.path.expanduser('~/tcrm')))

BASE_DIR = os.path.expanduser("/scratch/w85/kr4383")
DATA_DIR = os.path.join(BASE_DIR, "tracks")
OUT_DIR = os.path.join(BASE_DIR, "plots")

if __name__ == '__main__':

    colorseq=['#FFFFFF', '#ceebfd', '#87CEFA', '#4969E1', '#228B22',
              '#90EE90', '#FFDD66', '#FFCC00', '#FF9933',
              '#FF6600', '#FF0000', '#B30000', '#73264d']
    cmap = sns.blend_palette(colorseq, as_cmap=True)

    files = [os.path.join(DATA_DIR, fn) for fn in os.listdir(DATA_DIR)]
    arrs = [np.load(fp) for fp in files]
    lens = np.zeros(len(arrs))
    lens[1:] = np.cumsum([arr.shape[-1] for arr in arrs])[:-1]
    latitudes = np.concatenate([arr[0].transpose().flatten() for arr in arrs])
    longitudes = np.concatenate([arr[1].transpose().flatten() for arr in arrs])

    idxs = np.concatenate([
        (np.arange(arrs[i].shape[-1])[:, None] * np.ones(arrs[i].shape[1])[None, :]).flatten() + lens[i]
        for i in range(len(arrs))
    ]).astype(int)

    df = pd.DataFrame()
    df["eventid"] = idxs
    df["latitude"] = latitudes
    df["longitude"] = longitudes
    mask = ~pd.isnull(df.latitude)
    df = df[mask]

    df = df.iloc[:10_000]

    storm_id_field = "eventid"
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

    dims = (int((maxlon - minlon)/dx), int((maxlat-minlat)/dy))
    dfgrid = createGrid(minlon, maxlon, minlat, maxlat, dx, dy)

    dfstorm = addGeometry(df, storm_id_field, 'longitude', 'latitude')
    dfcount = gridDensity(dfstorm, dfgrid, grid_id_field, storm_id_field)

    nseasons = 10_000

    dfcount['density'] = dfcount['storm_count'] / nseasons
    tdarray = dfcount['density'].values.reshape(dims) # td = tropical cyclone *density*
    tcarray = dfcount['storm_count'].values.reshape(dims) # tc = tropical cyclone *count*
    da_obs = xr.DataArray(
        tdarray, coords=[lon, lat], dims=['longitude', 'latitude'],
        attrs=dict(long_name='Mean annual TC frequency', units='TCs/year')
    )

    plt.figure(figsize=(10, 10))

    plot_density(
        da_obs, "http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv", os.path.join(OUT_DIR, "10_000_track_density.png"),
        xx, yy
    )
    gates = gpd.read_file(os.path.join("..", "data", "gates.shp"))
    gates['sim'] = 0
    gates['count'] = 0

    trackgdf = []
    for k, t in df.groupby('eventid'):
        segments = []
        for n in range(len(t.eventid) - 1):
            segment = LineString(
                [[t.longitude.iloc[n], t.latitude.iloc[n]], [t.longitude.iloc[n + 1], t.latitude.iloc[n + 1]]])
            segments.append(segment)

        gdf = gpd.GeoDataFrame.from_records(t[:-1])
        gdf['geometry'] = segments

        trackgdf.append(gdf)

    trackgdf = pd.concat(trackgdf)
    gatedf = gates.copy()
    gatedf = countCrossings(gatedf, trackgdf, 0)

    dataFile = os.path.join(BASE_DIR, "IDCKMSTM0S.csv")
    usecols = [0, 1, 2, 7, 8, 16, 49, 53]
    colnames = ['NAME', 'DISTURBANCE_ID', 'TM', 'LAT', 'LON',
                'CENTRAL_PRES', 'MAX_WIND_SPD', 'MAX_WIND_GUST']
    dtypes = [str, str, str, float, float, float, float, float]

    bom_df = pd.read_csv(dataFile, skiprows=4, usecols=usecols, dtype=dict(zip(colnames, dtypes)), na_values=[' '])
    bom_df.TM = pd.to_datetime(bom_df.TM)
    bom_df['TM'] = pd.to_datetime(bom_df.TM, format="%Y-%m-%d %H:%M", errors='coerce')
    bom_df = bom_df[~pd.isnull(bom_df['TM'])]
    bom_df['TM'] = pd.to_datetime(bom_df.TM, format="%Y-%m-%d %H:%M", errors='coerce')
    bom_df = bom_df[~pd.isnull(bom_df.TM)]
    bom_df['season'] = pd.DatetimeIndex(bom_df['TM']).year - (pd.DatetimeIndex(bom_df['TM']).month < 6)
    bom_df = bom_df[bom_df.season >= 1981]

    bom_gates = gpd.read_file(os.path.join("..", "data", "gates.shp"))

    bom_gates['sim'] = 0
    bom_gates['count'] = 0

    bom_trackgdf = []
    for k, t in bom_df.groupby('DISTURBANCE_ID'):
        segments = []
        for n in range(len(t.DISTURBANCE_ID) - 1):
            segment = LineString([[t.LON.iloc[n], t.LAT.iloc[n]], [t.LON.iloc[n + 1], t.LAT.iloc[n + 1]]])
            segments.append(segment)

        bom_gdf = gpd.GeoDataFrame.from_records(t[:-1])
        bom_gdf['geometry'] = segments

        bom_trackgdf.append(bom_gdf)

    bom_trackgdf = pd.concat(bom_trackgdf)
    bom_gatedf = gates.copy()
    bom_gatedf = countCrossings(bom_gatedf, bom_trackgdf, 0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
    ax.plot(gatedf.gate, gatedf['count'] / 10_000, label="Simulated TC Tracks")
    ax.plot(bom_gatedf.gate, bom_gatedf['count'] / 40, label="BOM Best Track")
    plt.legend()
    ax.set_xlim((0,48))
    ax.set_xticks(np.arange(0,48,2))
    ax.set_xticklabels(gatedf['label'][::2][:-1], rotation='vertical')

    plt.savefig(os.path.join(OUT_DIR, "10_000_yr_landfall.png"))
