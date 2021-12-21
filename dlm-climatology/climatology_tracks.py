import pandas as pd
import xarray as xr
import numpy as np
import os
from spatial_correlation import smooth
import time
import logging
from calendar import monthrange
import sys
import scipy.stats as stats
from geopy.distance import geodesic as gdg
import geopy
from mpi4py import MPI


sys.path.insert(0, sys.path.insert(0, os.path.expanduser('~/tcrm')))
from StatInterface.SamplingOrigin import SamplingOrigin

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

path = "/scratch/w85/kr4383/era5dlm"
logging.basicConfig(filename='climatology_tc_tracks.log', level=logging.DEBUG)


def destination(lat1, lon1, dist, bearing):
    lon2 = np.zeros_like(lon1)
    lat2 = np.zeros_like(lat1)

    for i in range(len(lat1)):
        origin = geopy.Point(lat1[i], lon1[i])
        dest = gdg(kilometers=dist[i]).destination(origin, bearing[i])
        lon2[i] = dest.longitude
        lat2[i] = dest.latitude

    return lat2, lon2


def get_climatology(month):

    ufile = os.path.join(path, "u_dlm_{}_{}.netcdf".format(month, "{}"))
    vfile = os.path.join(path, "v_dlm_{}_{}.netcdf".format(month, "{}"))

    udlms = [xr.open_dataset(ufile.format(year), chunks='auto') for year in range(1981, 1983)]
    vdlms = [xr.open_dataset(vfile.format(year), chunks='auto') for year in range(1981, 1983)]

    udlm = xr.concat(udlms, dim='time')
    vdlm = xr.concat(vdlms, dim='time')

    u_mean = udlm.mean(dim='time').compute().u.data
    u_std = udlm.std(dim='time').compute().u.data
    v_mean = vdlm.mean(dim='time').compute().v.data
    v_std = vdlm.std(dim='time').compute().v.data

    return np.array([u_mean, u_std, v_mean, v_std])


def perturbation(t, N, T, phase):
    n = np.arange(1, N + 1)[None, :]

    n_inv_23 = n ** (-3 / 2)
    a = np.sqrt(2) / np.linalg.norm(n_inv_23)

    return a * (n_inv_23 * np.sin(2 * np.pi * (phase + n * t[:, None] / T))).sum(axis=1)


def tc_velocity(climatologies, t, N, T, phase, idxs1, idxs2, weights1, weights2):
    pert = perturbation(t, N, T, phase)

    u1 = climatologies[0].take(idxs1) + climatologies[1].take(idxs1)
    v1 = climatologies[2].take(idxs1) + climatologies[3].take(idxs1)
    u2 = climatologies[0].take(idxs2) + climatologies[1].take(idxs2)
    v2 = climatologies[2].take(idxs2) + climatologies[3].take(idxs2)

    u = (u1 * weights1 + u2 * weights2) / (weights1 + weights2)
    v = (v1 * weights1 + v2 * weights2) / (weights1 + weights2)

    u = -4.5205 + 0.8978 * u * 3.6
    v = -1.2542 + 0.7877 * v * 3.6

    return u, v


N = 15
T = 15 * 24  # 15 days
phase = np.random.random(N)

u_phase = np.random.random(N)
v_phase = np.random.random(N)

dataFile = os.path.join("/scratch/w85/kr4383/IDCKMSTM0S.csv")
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
genesis_points = df.groupby('DISTURBANCE_ID').aggregate({'TM': 'first', 'LAT': 'first', 'LON': 'first'})
tm = pd.DatetimeIndex(genesis_points.TM)
genesis_points['TM'] = (tm.dayofyear * 24.0 + tm.hour)
genesis_points = np.row_stack([genesis_points[c] for c in genesis_points.columns])
pde = stats.gaussian_kde(genesis_points)

pressure = np.array(
    [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850]
)

months = np.arange(1, 13)
rank = comm.Get_rank()
rank_months = months[(months % comm.size) == rank]

repeats = 10_000  # simulate a total of 10_000 years

month_rates = {
    1: 2.3415, 2: 2.0976, 3: 2.0, 12: 1.439, 4: 1.3659, 11: 0.5122,
    5: 0.3659, 10: 0.122, 7: 0.0488, 6: 0.0488, 8: 0.0244, 9: 0.0244
}

print("Starting simulation.")

year = 2001  # arbitrary choice of non leap year

climatologies = np.array([smooth(get_climatology(month)) for month in range(1, 13)]).swapaxes(0, 1)
# udlm_2, vdlm_2 = load_dlm(year + (month // 12), (month % 12) + 1)

t0 = time.time()

# sufficient repeats that the sum should be equal to the mean * number of repeats

revisit = []

days_in_month = monthrange(year, month)[1]
longitudes = []

num_events = int(sum(month_rates.values()) * repeats)

durations = (np.round(stats.lognorm.rvs(0.5491, 0., 153.27, size=num_events))).astype(int)

samples = pde.resample(num_events)
mask = (samples[2, :] >= 170) | (samples[2, :] <= 80)

while mask.any():
    samples[:, mask] = pde.resample(mask.sum())
    mask = (samples[2, :] >= 170) | (samples[2, :] <= 80)

coords = np.empty((2, durations.max(), num_events)) * np.nan

latitudes = coords[0, ...]
latitudes[0, :] = samples[1, :]

longitudes = coords[1, ...]
longitudes[0, :] = samples[2, :]

tm = samples[0, :].astype(int) % (365 * 24)
timestamps = pd.to_datetime('01-01-2001') + tm * pd.Timedelta(hours=1)

for step in range(durations.max() - 1):

    mask = (longitudes[step] <= 170) & (longitudes[step] >= 80)
    mask &= (latitudes[step] >= -40) & (latitudes[step] <= 0)
    # mask &= timestamps <= udlm.coords['time'].data[-1]
    mask &= step <= durations

    long_idxs = np.round(4 * longitudes[step][mask]) - (4 * 80)
    lat_idxs = np.round(4 * latitudes[step][mask])

    idxs = (lat_idxs * climatologies.shape[-1] + long_idxs).astype(int)

    time_pd = pd.DatetimeIndex(timestamps)
    t = time_pd.dayofyear * 24 + time_pd.hour

    c = (time_pd.day < 15)  # if less than 30 days use current and previous month
    month = time_pd.month
    idxs1 = idxs + (month - 1) * climatologies[0, 0].size
    idxs2 = idxs + c * ((month - 2) % 12) * climatologies[0, 0].size
    idxs2 += (~c) * (month % 12) * climatologies[0, 0].size

    weights2 = np.abs(time_pd.day - 15)
    weights1 = 30 - weights1

    u, v = tc_velocity(climatologies, t[mask], N, T, phase, idxs1, idxs2, weights1, weights2)

    dist = np.sqrt(u ** 2 + v ** 2)  # km travelled in one hour

    mask1 = (u >= 0) & (v >= 0)
    mask2 = (u >= 0) & (v < 0)
    mask3 = (u < 0) & (v < 0)

    bearing = 360 + np.arctan(u / v) * 180 / np.pi
    bearing[mask1] = (np.arctan(u / v) * 180 / np.pi)[mask1]
    bearing[mask2] = (180 + np.arctan(u / v) * 180 / np.pi)[mask2]
    bearing[mask3] = (180 + np.arctan(u / v) * 180 / np.pi)[mask3]

    dest = destination(latitudes[step, mask], longitudes[step, mask], dist, bearing)
    latitudes[step + 1, mask] = dest[0]
    longitudes[step + 1, mask] = dest[1]

    timestamps += np.timedelta64(1, 'h')

t1 = time.time()

print(f"Finished simulating tracks for {month}/{year}. Time taken: {t1 - t0}s")
logging.info(f"Finished simulating tracks for {month}/{year}. Time taken: {t1 - t0}s")
np.save(f"/scratch/w85/kr4383/climatology_tracks/climatology_tracks_{month}_{year}.npy", coords)
