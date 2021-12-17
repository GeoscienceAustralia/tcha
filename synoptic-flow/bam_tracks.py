import numpy as np
import scipy.signal as sps
import os
import pandas as pd
from matplotlib import pyplot as plt
import pyproj
from lmfit import Model
from sklearn.linear_model import LinearRegression
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from scipy.interpolate import interp2d
from vincenty import vincenty
import xarray as xr
from calendar import monthrange
import geopy
import geopandas as gpd
from shapely.geometry import LineString, Point
import time
from track_utils import plot_tracks, createGrid, plot_density, gridDensity, addGeometry, countCrossings
from geopy.distance import geodesic as gdg
from datetime import datetime


# this script requires the results of 'era5_dlm.py', bom best track, OCTR tracks
# and gates.shp stored in the DATA_DIR folder

geodesic = pyproj.Geod(ellps='WGS84')
WIDTH = 4
DATA_DIR = os.path.expanduser("~/geoscience/data")
OUT_DIR = os.path.expanduser("~/geoscience/data/plots")
NOISE = False


def fit_plot(vel, steering, result, residuals, mask, data, component, source):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))

    sns.distplot(residuals,
                 hist_kws={'ec': 'b', 'width': 0.05},
                 kde_kws={'label': 'Residuals', 'linestyle': '--'},
                 ax=ax0, norm_hist=True)
    pp = sm.ProbPlot(residuals, stats.norm, fit=True)
    pp.qqplot('Residuals', 'Normal', line='45', ax=ax1, color='gray', alpha=0.5)
    fig.tight_layout()
    x = np.sort(residuals)

    ax0.legend(loc=0)

    fp = stats.norm.fit(x)
    ax0.plot(x, stats.norm.pdf(x, fp[0], fp[1]), label='Normal', color='r')
    print(stats.normaltest(residuals))
    ax0.legend()
    plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=ax1.transAxes, ha='right', fontsize='xx-small', )
    plt.text(0.0, -0.1, f"Source: {source}", transform=ax0.transAxes, fontsize='xx-small', ha='left', )
    plt.savefig(os.path.join(OUT_DIR, f"BoM_BAM_{component}_residuals.png"))

    fct = sns.relplot(result.eval(x=steering[mask]), vel[mask])
    fct.ax.set_xlabel(f'{component} predicted')
    plt.text(1.0, -0.2, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=fct.ax.transAxes, ha='right', fontsize='xx-small', )
    plt.text(0.0, -0.2, f"Source: {source}", transform=fct.ax.transAxes, fontsize='xx-small', ha='left', )
    plt.savefig(os.path.join(OUT_DIR, f"{data}_BAM_{component}_predicted.png"))


def load_bom_df():
    dataFile = os.path.join(DATA_DIR, "IDCKMSTM0S.csv")
    usecols = [0, 1, 2, 7, 8, 16, 49, 53]
    colnames = ['NAME', 'DISTURBANCE_ID', 'TM', 'LAT', 'LON',
                'CENTRAL_PRES', 'MAX_WIND_SPD', 'MAX_WIND_GUST']
    dtypes = [str, str, str, float, float, float, float, float]

    df = pd.read_csv(dataFile, skiprows=4, usecols=usecols, dtype=dict(zip(colnames, dtypes)), na_values=[' '])
    df['TM'] = pd.to_datetime(df.TM, format="%Y-%m-%d %H:%M", errors='coerce')
    df = df[~pd.isnull(df.TM)]
    df['season'] = pd.DatetimeIndex(df['TM']).year - (pd.DatetimeIndex(df['TM']).month < 6)
    df = df[df.season >= 1981]
    df.reset_index(inplace=True)

    # distance =
    fwd_azimuth, _, distances = geodesic.inv(
        df.LON[:-1], df.LAT[:-1],
        df.LON[1:], df.LAT[1:],
    )

    df['new_index'] = np.arange(len(df))
    idxs = df.groupby(['DISTURBANCE_ID']).agg({'new_index': np.max}).values.flatten()
    df.drop('new_index', axis=1, inplace=True)

    dt = np.diff(df.TM).astype(np.float) / 3_600_000_000_000
    u = np.zeros_like(df.LAT)
    v = np.zeros_like(df.LAT)
    v[:-1] = np.cos(fwd_azimuth * np.pi / 180) * distances / (dt * 1000)
    u[:-1] = np.sin(fwd_azimuth * np.pi / 180) * distances / (dt * 1000)

    v[idxs] = 0
    u[idxs] = 0
    df['u'] = u
    df['v'] = v

    dt = np.diff(df.TM).astype(np.float) / 3_600_000_000_000
    dt_ = np.zeros(len(df))
    dt_[:-1] = dt
    df['dt'] = dt_

    df = df[df.u != 0].copy()

    return df


def load_otcr_df():
    dataFile = os.path.join(DATA_DIR, "OTCR_alldata_final_external.csv")
    usecols = [0, 1, 2, 7, 8, 11, 12]
    colnames = ['NAME', 'DISTURBANCE_ID', 'TM', 'LAT', 'LON',
                'adj. ADT Vm (kn)', 'CP(CKZ(Lok R34,LokPOCI, adj. Vm),hPa)']
    dtypes = [str, str, str, float, float, float, float]

    df = pd.read_csv(dataFile, usecols=usecols, dtype=dict(zip(colnames, dtypes)), na_values=[' '], nrows=13743)

    df['TM']= pd.to_datetime(df.TM, format="%d/%m/%Y %H:%M", errors='coerce')
    df = df[~pd.isnull(df.TM)]
    df['season'] = pd.DatetimeIndex(df['TM']).year - (pd.DatetimeIndex(df['TM']).month < 6)
    df = df[df.season >= 1981]
    df.reset_index(inplace=True)

    # distance =
    fwd_azimuth, _, distances = geodesic.inv(
        df.LON[:-1], df.LAT[:-1],
        df.LON[1:], df.LAT[1:],
    )

    df['new_index'] = np.arange(len(df))
    idxs = df.groupby(['DISTURBANCE_ID']).agg({'new_index': np.max}).values.flatten()
    df.drop('new_index', axis=1, inplace=True)

    dt = np.diff(df.TM).astype(np.float) / 3_600_000_000_000
    u = np.zeros_like(df.LAT)
    v = np.zeros_like(df.LAT)
    v[:-1] = np.cos(fwd_azimuth * np.pi / 180) * distances / (dt * 1000)
    u[:-1] = np.sin(fwd_azimuth * np.pi / 180) * distances / (dt * 1000)

    v[idxs] = 0
    u[idxs] = 0
    df['u'] = u
    df['v'] = v

    dt = np.diff(df.TM).astype(np.float) / 3_600_000_000_000
    dt_ = np.zeros(len(df))
    dt_[:-1] = dt
    df['dt'] = dt_

    df = df[df.u != 0].copy()

    return df


def load_steering(uds, vds, df):
    u_steering = []
    v_steering = []
    var = "__xarray_dataarray_variable__"

    for row in df.itertuples():
        lat_cntr = 0.25 * np.round(row.LAT * 4)
        lon_cntr = 0.25 * np.round(row.LON * 4)

        lat_slice = slice(lat_cntr + WIDTH, lat_cntr - WIDTH)
        long_slice = slice(lon_cntr - WIDTH, lon_cntr + WIDTH)

        u_dlm = 3.6 * uds[var].sel(time=row.TM, longitude=long_slice, latitude=lat_slice)
        v_dlm = 3.6 * vds[var].sel(time=row.TM, longitude=long_slice, latitude=lat_slice)

        u_steering.append(u_dlm.mean())
        v_steering.append(v_dlm.mean())

    u_steering = np.array(u_steering)
    v_steering = np.array(v_steering)

    return u_steering, v_steering


def lin_model(x, alpha, beta):
    return alpha * x + beta


def simulation(df, uds, vds, uresult, vresult, u_std, v_std):

    var = "__xarray_dataarray_variable__"
    lats = [None]
    lons = [None]
    ids = set()

    auto = 0.5
    compl = np.sqrt(1 - auto ** 2)

    for row in list(df.itertuples())[:]:

        if row.DISTURBANCE_ID not in ids:
            # using forward difference
            # discard last position of previous TC track
            lats[-1] = row.LAT
            lons[-1] = row.LON
            ids.add(row.DISTURBANCE_ID)
            u_noise = np.random.normal(scale=u_std)
            v_noise = np.random.normal(scale=v_std)

        if np.isnan(lats[-1]):
            # TC went out of domain
            lats.append(np.nan)
            lons.append(np.nan)
            continue

        timestamp = row.TM

        month = timestamp.month
        year = timestamp.year

        lat_cntr = 0.25 * np.round(lats[-1] * 4)
        lon_cntr = 0.25 * np.round(lons[-1] * 4)
        lat_slice = slice(lat_cntr + WIDTH, lat_cntr - WIDTH)
        long_slice = slice(lon_cntr - WIDTH, lon_cntr + WIDTH)

        try:
            # calculate the DML
            u_dlm = 3.6 * uds[var].sel(time=timestamp, longitude=long_slice, latitude=lat_slice)
            v_dlm = 3.6 * vds[var].sel(time=timestamp, longitude=long_slice, latitude=lat_slice)

            # calculate TC velocity and time step
            u_noise = auto * u_noise + compl * np.random.normal(scale=u_std)
            v_noise = auto * v_noise + compl * np.random.normal(scale=v_std)

            u = uresult.values['alpha'] * u_dlm.mean() + uresult.values['beta'] + u_noise * NOISE
            v = vresult.values['alpha'] * v_dlm.mean() + vresult.values['beta'] + v_noise * NOISE

            if (u >= 0) and (v >= 0):
                bearing = np.arctan(u / v) * 180 / np.pi
            elif (u >= 0) and (v < 0):
                bearing = 180 + np.arctan(u / v) * 180 / np.pi
            elif (u < 0) and (v < 0):
                bearing = 180 + np.arctan(u / v) * 180 / np.pi
            else:
                bearing = 360 + np.arctan(u / v) * 180 / np.pi

            distance = np.sqrt(u ** 2 + v ** 2) * row.dt

            origin = geopy.Point(lats[-1], lons[-1])
            destination = gdg(kilometers=distance).destination(origin, bearing)
            lats.append(destination.latitude)
            lons.append(destination.longitude)

        except IndexError as e:
            print(e)
            lats.append(np.nan)
            lons.append(np.nan)

        except ValueError as e:
            print(e)
            lats.append(np.nan)
            lons.append(np.nan)

    df = df.iloc[:len(lats)].copy()
    lats = lats[:len(df)]
    lons = lons[:len(df)]

    df['lats_sim'] = np.array(lats)
    df['lons_sim'] = np.array(lons)

    return df


def run(name, source):

    # load data

    if name == 'BoM':
        df = load_bom_df()
    elif name == 'OTCR':
        df = load_otcr_df()

    upath = os.path.join(DATA_DIR, "era5dlm/u_dlm_{}.netcdf")
    vpath = os.path.join(DATA_DIR, "era5dlm/v_dlm_{}.netcdf")

    udss = [xr.open_dataset((upath.format(year))) for year in range(1981, 2022)]
    uds = xr.concat(udss, dim='time')

    vdss = [xr.open_dataset((vpath.format(year))) for year in range(1981, 2022)]
    vds = xr.concat(vdss, dim='time')

    u_steering, v_steering = load_steering(uds, vds, df)

    ## fit the BAM model

    mask = ~np.isnan(u_steering)

    rmod = Model(lin_model)
    params = rmod.make_params(alpha=1., beta=-0.001, )
    uresult = rmod.fit(df.u[mask], x=u_steering[mask], params=params)
    print(uresult.values)
    pred = uresult.eval(x=u_steering[mask])
    u_residuals = pred - df.u[mask]

    rmod = Model(lin_model)
    params = rmod.make_params(alpha=1., beta=-0.001, )
    vresult = rmod.fit(df.v[mask], x=v_steering[mask], params=params)
    print(vresult.values)
    pred = vresult.eval(x=v_steering[mask])
    v_residuals = pred - df.v[mask]

    u_rsme = np.sqrt(np.mean(u_residuals ** 2))
    v_rsme = np.sqrt(np.mean(v_residuals ** 2))

    u_std = np.std(u_residuals)
    v_std = np.std(v_residuals)

    print("u resid std:", np.std(u_residuals))
    print("v resid std:", np.std(v_residuals))

    fit_plot(df.u, u_steering, uresult, u_residuals, mask, name, "u", source)
    fit_plot(df.v, v_steering, vresult, v_residuals, mask, name, "v", source)

    # run the track simulation and plot the tracks

    df = simulation(df, uds, vds, uresult, vresult, u_std, v_std)
    plot_tracks(
        df, "LAT", "LON", "DISTURBANCE_ID", f"{name} Best Tracks", source, os.path.join(OUT_DIR, f"{name}_tracks.png")
    )
    plot_tracks(
        df, "lats_sim", "lons_sim", "DISTURBANCE_ID", "BAM Tracks", source, os.path.join(OUT_DIR, f"{name}_bam_tracks.png")
    )

    ## plot the track density

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

    dims = (int((maxlon - minlon) / dx), int((maxlat - minlat) / dy))
    dfgrid = createGrid(minlon, maxlon, minlat, maxlat, dx, dy)

    nseasons = df.season.max() - df.season.min() + 1

    dfstorm = addGeometry(df, storm_id_field, 'LON', 'LAT')
    dfcount = gridDensity(dfstorm, dfgrid, grid_id_field, storm_id_field)
    dfcount['density'] = dfcount['storm_count'] / nseasons
    tdarray = dfcount['density'].values.reshape(dims)  # td = tropical cyclone *density*
    tcarray = dfcount['storm_count'].values.reshape(dims)  # tc = tropical cyclone *count*
    da_obs = xr.DataArray(
        tdarray, coords=[lon, lat], dims=['lon', 'lat'],
        attrs=dict(long_name='Mean annual TC frequency', units='TCs/year')
    )

    plot_density(da_obs, source, os.path.join(OUT_DIR, f"{name}_mean_track_density.bootstrap.png"), xx, yy)

    dfstorm = addGeometry(df[~pd.isnull(df.lons_sim)].copy(), storm_id_field, 'lons_sim', 'lats_sim')
    dfcount = gridDensity(dfstorm, dfgrid, grid_id_field, storm_id_field)
    dfcount['density'] = dfcount['storm_count'] / nseasons
    tdarray = dfcount['density'].values.reshape(dims)  # td = tropical cyclone *density*
    da_bam = xr.DataArray(
        tdarray, coords=[lon, lat], dims=['lon', 'lat'],
        attrs=dict(long_name='Mean annual TC frequency', units='TCs/year')
    )

    plot_density(da_bam, source, os.path.join(OUT_DIR, f"{name}_BAM_mean_track_density.bootstrap.png"), xx, yy)

    ## landfall

    gates = gpd.read_file("../data/gates.shp")
    gates['sim'] = 0
    gates['count'] = 0

    trackgdf = []
    for k, t in df.groupby('DISTURBANCE_ID'):
        segments = []
        for n in range(len(t.DISTURBANCE_ID) - 1):
            segment = LineString([[t.LON.iloc[n], t.LAT.iloc[n]], [t.LON.iloc[n + 1], t.LAT.iloc[n + 1]]])
            segments.append(segment)

        gdf = gpd.GeoDataFrame.from_records(t[:-1])
        gdf['geometry'] = segments

        trackgdf.append(gdf)

    gates = gpd.read_file("../data/gates.shp")
    gates['sim'] = 0
    gates['count'] = 0

    trackgdf = []
    for k, t in df.groupby('DISTURBANCE_ID'):
        segments = []
        for n in range(len(t.DISTURBANCE_ID) - 1):
            segment = LineString([[t.LON.iloc[n], t.LAT.iloc[n]], [t.LON.iloc[n + 1], t.LAT.iloc[n + 1]]])
            segments.append(segment)

        gdf = gpd.GeoDataFrame.from_records(t[:-1])
        gdf['geometry'] = segments

        trackgdf.append(gdf)

    trackgdf = pd.concat(trackgdf)
    gatedf = gates.copy()
    gatedf = countCrossings(gatedf, trackgdf, 0)

    gates = gpd.read_file("../data/gates.shp")
    gates['sim'] = 0
    gates['count'] = 0

    trackgdf = []
    for k, t in df[~pd.isnull(df.lats_sim)].groupby('DISTURBANCE_ID'):
        segments = []
        for n in range(len(t.DISTURBANCE_ID) - 1):
            segment = LineString(
                [[t.lons_sim.iloc[n], t.lats_sim.iloc[n]], [t.lons_sim.iloc[n + 1], t.lats_sim.iloc[n + 1]]])
            segments.append(segment)

        gdf = gpd.GeoDataFrame.from_records(t[:-1])
        gdf['geometry'] = segments
        trackgdf.append(gdf)

    trackgdf = pd.concat(trackgdf)
    gatedf_sim = gates.copy()
    gatedf_sim = countCrossings(gatedf_sim, trackgdf, 0)

    freqs = []

    for season in np.unique(df.season):
        trackgdf = []
        for k, t in df[df.season == season].groupby('DISTURBANCE_ID'):
            segments = []
            for n in range(len(t.DISTURBANCE_ID) - 1):
                segment = LineString([[t.LON.iloc[n], t.LAT.iloc[n]], [t.LON.iloc[n + 1], t.LAT.iloc[n + 1]]])
                segments.append(segment)

            gdf = gpd.GeoDataFrame.from_records(t[:-1])
            gdf['geometry'] = segments
            trackgdf.append(gdf)
        trackgdf = pd.concat(trackgdf)
        gatedf_sim_tmp = gates.copy()
        gatedf_sim_tmp = countCrossings(gatedf_sim_tmp, trackgdf, 0)
        freqs.append(gatedf_sim_tmp['count'])

    freqs = np.array([f.values for f in freqs])

    perrs = []
    merrs = []
    for i in range(49):
        arr = freqs[:, i]
        sample = np.random.choice(arr, size=(10000, 40))
        means = sample.mean(axis=1)
        perrs.append(np.percentile(means, 90))
        merrs.append(np.percentile(means, 10))

    obs_perrs = np.array(perrs)
    obs_merrs = np.array(merrs)

    freqs = []

    for season in np.unique(df.season):
        trackgdf = []
        for k, t in df[(df.season == season) & (~pd.isnull(df.lats_sim))].groupby('DISTURBANCE_ID'):
            segments = []
            for n in range(len(t.DISTURBANCE_ID) - 1):
                segment = LineString(
                    [[t.lons_sim.iloc[n], t.lats_sim.iloc[n]], [t.lons_sim.iloc[n + 1], t.lats_sim.iloc[n + 1]]])
                segments.append(segment)

            gdf = gpd.GeoDataFrame.from_records(t[:-1])
            gdf['geometry'] = segments
            trackgdf.append(gdf)
        trackgdf = pd.concat(trackgdf)
        gatedf_sim_tmp = gates.copy()
        gatedf_sim_tmp = countCrossings(gatedf_sim_tmp, trackgdf, 0)
        freqs.append(gatedf_sim_tmp['count'])

    freqs = np.array([f.values for f in freqs])

    perrs = []
    merrs = []
    for i in range(49):
        arr = freqs[:, i]
        sample = np.random.choice(arr, size=(10000, 40))
        means = sample.mean(axis=1)
        perrs.append(np.percentile(means, 90))
        merrs.append(np.percentile(means, 10))

    bam_perrs = np.array(perrs)
    bam_merrs = np.array(merrs)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharex=True)

    ax.plot(gatedf_sim.gate, gatedf_sim['count'] / 40, label="BAM Model")
    ax.plot(gatedf.gate, gatedf['count'] / 40, label="Observed")

    ax.set_xlim((0, 48))
    ax.set_xticks(np.arange(0, 48, 2))
    ax.set_xticklabels(gatedf_sim['label'][::2][:-1], rotation='vertical')
    plt.legend()

    ax.fill_between(gatedf_sim.gate, obs_merrs, obs_perrs, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    ax.fill_between(gatedf_sim.gate, bam_merrs, bam_perrs, alpha=0.5)

    plt.text(1.0, -0.3, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=ax.transAxes, ha='right', fontsize='x-small', )
    plt.text(0.0, -0.3, f"Source: {source}", transform=ax.transAxes, fontsize='x-small', ha='left', )
    plt.savefig(os.path.join(OUT_DIR, f"{name}_landfall.png"))


if __name__ == "__main__":

    # run("BoM", "http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv")
    run("OTCR", "http://www.bom.gov.au/cyclone/history/database/OTCR_alldata_final_external.csv")
