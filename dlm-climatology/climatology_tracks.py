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

sys.path.insert(0, sys.path.insert(0, os.path.expanduser('~/tcrm')))
from StatInterface.SamplingOrigin import SamplingOrigin


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

    udlms = [xr.open_dataset(ufile.format(year)).u.data for year in range(1981, 2021)]
    vdlms = [xr.open_dataset(vfile.format(year)).v.data for year in range(1980, 2021)]

    udlm = np.concatenate(udlms, axis=0)
    vdlm = np.concatenate(vdlms, axis=0)

    return np.array([udlm.mean(axis=0), np.std(udlm, axis=0), vdlm.mean(axis=0), np.std(vdlm, axis=0)])


def perturbation(t, N, T, phase):
    n = np.arange(1, N + 1)[None, :]

    n_inv_23 = n ** (-3 / 2)
    a = np.sqrt(2) / np.linalg.norm(n_inv_23)

    return a * (n_inv_23 * np.sin(2 * np.pi * (phase + n * t / T))).sum(axis=1)


def tc_velocity(climatology, month, t, N, T, phase, idxs):
    pert = perturbation(t, N, T, phase)

    u = climatology[month + 1, 0].take(idxs) + climatology[month + 1, 1].take(idxs) * pert
    v = climatology[month + 1, 2].take(idxs) + climatology[month + 1, 3].take(idxs) * pert

    u = -4.5205 + 0.8978 * u * 3.6
    v = -1.2542 + 0.7877 * v * 3.6

    return u, v


N = 15
T = 15 * 24  # 15 days
phase = np.random.random(N)

climatology = np.array([get_climatology(month) for month in range(1, 13)])
climatology = smooth(climatology.reshape((-1,) + climatology.shape[-2:])).reshape(climatology.shape)

u_mean = climatology[:, 0]
u_std = climatology[:, 1]
v_mean = climatology[:, 2]
v_std = climatology[:, 3]

u_phase = np.random.random(N)
v_phase = np.random.random(N)

genesis_sampler = SamplingOrigin(
    kdeOrigin="/g/data/fj6/TCRM/TCHA18/process/originPDF.nc"
)
pressure = np.array(
    [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850]
)

years = np.arange(1981, 2021)
rank = comm.Get_rank()
rank_years = years[(years % comm.size) == rank]

repeats = 10_000  # simulate a total of 10_000 years

month_rates = {
    1: 2.3415, 2: 2.0976, 3: 2.0, 12: 1.439, 4: 1.3659, 11: 0.5122,
    5: 0.3659, 10: 0.122, 7: 0.0488, 6: 0.0488, 8: 0.0244, 9: 0.0244
}

rows = []
print("Starting simulation.")

year = 2001  # arbitrary choice of non leap year

for month in range(1, 13):
    # udlm_2, vdlm_2 = load_dlm(year + (month // 12), (month % 12) + 1)

    t0 = time.time()

    # sufficient repeats that the sum should be equal to the mean * number of repeats

    logging.info(f"Simulating tracks for {month}")
    print(f"Simulating tracks for {month}")

    revisit = []

    days_in_month = monthrange(year, month)[1]
    longitudes = []

    num_events = int(np.round(month_rates[month] * repeats))
    durations = (np.round(stats.lognorm.rvs(0.5491, 0., 153.27, size=num_events))).astype(int)

    year_month = np.datetime64(f'{year}-{month:02d}')
    days = np.random.randint(1, days_in_month + 1, size=num_events) * np.timedelta64(1, 'D')
    hours = np.random.randint(0, 24, size=num_events) * np.timedelta64(1, 'h')
    timestamps = year_month + hours + days

    origin = genesis_sampler.generateSamples(num_events)
    mask = (origin[:, 0] >= 170) | (origin[:, 0] <= 80)
    while mask.any():
        origin[mask, :] = genesis_sampler.generateSamples(mask.sum())
        mask = (origin[:, 0] >= 170) | (origin[:, 0] <= 80)

    coords = np.empty((2, durations.max(), num_events)) * np.nan

    latitudes = coords[0, ...]
    latitudes[0, :] = origin[:, 1]

    longitudes = coords[1, ...]
    longitudes[0, :] = origin[:, 0]

    for step in range(durations.max() - 1):

        mask = (longitudes[step] <= 170) & (longitudes[step] >= 80)
        mask &= (latitudes[step] >= -40) & (latitudes[step] <= 0)
        # mask &= timestamps <= udlm.coords['time'].data[-1]
        mask &= step <= durations

        long_idxs = np.round(4 * longitudes[step][mask]) - (4 * 80)
        lat_idxs = np.round(4 * latitudes[step][mask])

        idxs = (lat_idxs * climatology.shape[-1] + long_idxs).astype(int)

        time_pd = pd.DatetimeIndex(timestamps)
        t = time_pd.dayofyear * 24 + time_pd.hour

        u, v = tc_velocity(climatology, month, t, N, T, phase, idxs)

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
