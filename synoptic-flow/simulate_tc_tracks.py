print("Importing.")
import numpy as np
import os
import xarray as xr
import pandas as pd
from calendar import monthrange
import time
import scipy.stats as stats
from mpi4py import MPI
import logging
import geopy
from geopy.distance import geodesic as gdg
import sys

print("Imports dones. Setting up logs.")
sys.path.insert(0, sys.path.insert(0, os.path.expanduser('~/tcrm')))

from StatInterface.SamplingOrigin import SamplingOrigin

logging.basicConfig(filename='simulate_tc_tracks.log', level=logging.DEBUG)


def destination(lat1, lon1, dist, bearing):
    lat1 = lat1 * np.pi / 180
    lon1 = lon1 * np.pi / 180
    bearing = bearing * (np.pi / 180)
    # bearing = np.pi - bearing

    rad = dist / 6371  # distance in radians
    lat2 = np.arcsin(np.sin(lat1) * np.cos(rad) + np.cos(lat1) * np.sin(rad) * np.cos(bearing))
    lon2 = lon1 - np.arcsin(np.sin(-bearing) * np.sin(rad) / np.cos(lat2)) + np.pi
    lon2 = (lon2 % (2 * np.pi)) - np.pi

    return lat2 * 180 / np.pi, lon2 * 180 / np.pi


def load_dlm(year, month):

    ufile = f"/scratch/w85/kr4383/era5dlm/u_dlm_{month}_{year}.netcdf"
    vfile = f"/scratch/w85/kr4383/era5dlm/v_dlm_{month}_{year}.netcdf"

    udlm = xr.open_dataset(ufile)
    vdlm = xr.open_dataset(vfile)

    ufile = f"/scratch/w85/kr4383/era5dlm/u_dlm_{(month % 12) + 1}_{year + (month // 12)}.netcdf"
    vfile = f"/scratch/w85/kr4383/era5dlm/v_dlm_{(month % 12) + 1}_{year + (month // 12)}.netcdf"

    udlm1 = xr.open_dataset(ufile)
    vdlm1 = xr.open_dataset(vfile)

    udlm = xr.concat([udlm, udlm1], dim='time')
    vdlm = xr.concat([vdlm, vdlm1], dim='time')

    return udlm, vdlm


def tc_velocity(udlm, vdlm, long_idxs, lat_idxs, time_idxs):

    sz2 = udlm.u.data.shape[2]
    sz1 = sz2 * udlm.u.data.shape[1]

    idxs = time_idxs * sz1 + lat_idxs * sz2 + long_idxs

    u = udlm.u.data.flatten().take(idxs.astype(int)).mean(axis=1) * 3.6  # convert to km/hr
    v = vdlm.v.data.flatten().take(idxs.astype(int)).mean(axis=1) * 3.6

    u = -4.5205 + 0.8978 * u
    v = -1.2542 + 0.7877 * v

    return u, v


def timestep(latitude, longitude, u, v, dt):
    if (u >= 0) and (v >= 0):
        bearing = np.arctan(u / v) * 180 / np.pi
    elif (u >= 0) and (v < 0):
        bearing = 180 + np.arctan(u / v) * 180 / np.pi
    elif (u < 0) and (v < 0):
        bearing = 180 + np.arctan(u / v) * 180 / np.pi
    else:
        bearing = 360 + np.arctan(u / v) * 180 / np.pi

    distance = np.sqrt(u ** 2 + v ** 2) * dt

    origin = geopy.Point(latitude, longitude)
    destination = gdg(kilometers=distance).destination(origin, bearing)
    return destination


comm = MPI.COMM_WORLD

print("Loading genesis distribution")
genesis_sampler = SamplingOrigin(
    kdeOrigin="/g/data/fj6/TCRM/TCHA18/process/originPDF.nc"
)
pressure = np.array(
    [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850]
)

years = np.arange(1981, 2021)
rank = comm.Get_rank()
rank_years = years[(years % comm.size) == rank]

repeats = 10_000 / 40  # repeat the catalogue of 40s years until a total of 10_000 years have been simulated

month_rates = {
    1: 2.3415, 2: 2.0976, 3: 2.0, 12: 1.439, 4: 1.3659, 11: 0.5122,
    5: 0.3659, 10: 0.122, 7: 0.0488, 6: 0.0488, 8: 0.0244, 9: 0.0244
}

lat_offset = np.arange(-25, 25)
long_offset = np.arange(-25, 25)
lat_offset, long_offset = np.meshgrid(lat_offset, long_offset)

lat_offset = lat_offset.flatten()[None, :]
long_offset = long_offset.flatten()[None, :]
time_offset = np.zeros((1, 2500))

rows = []
print("Starting simulation.")
for year in rank_years[:1]:
    for month in range(1, 2):
        udlm, vdlm = load_dlm(year, month)
        # udlm_2, vdlm_2 = load_dlm(year + (month // 12), (month % 12) + 1)

        t0 = time.time()

        # sufficient repeats that the sum should be equal to the mean * number of repeats

        logging.info(f"Simulating tracks for {month}/{year}")
        print(f"Simulating tracks for {month}/{year}")

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

        coords = np.empty((2, durations.max(), num_events))

        latitudes = coords[0, ...]
        latitudes[0, :] = origin[:, 1]

        longitudes = coords[1, ...]
        longitudes[0, :] = origin[:, 0]

        longitude_index = pd.Series(np.arange(len(udlm.coords['longitude'].data)), udlm.coords['longitude'].data)
        latitude_index = pd.Series(np.arange(len(udlm.coords['latitude'].data)), udlm.coords['latitude'].data)
        time_index = pd.Series(np.arange(len(udlm.coords['time'].data)), udlm.coords['time'].data)

        for step in range(durations.max() - 1):

            mask = (longitudes[step] <= 170) & (longitudes[step] >= 80)
            mask &= (latitudes[step] >= -40) & (latitudes[step] <= 0)
            mask &= timestamps <= udlm.coords['time'].data[-1]

            long_idxs = long_offset + longitude_index.loc[np.round(4 * longitudes[step][mask]) / 4].values[:, None]
            lat_idxs = lat_offset + latitude_index.loc[np.round(4 * latitudes[step][mask]) / 4].values[:, None]
            time_idxs = time_offset + time_index.loc[timestamps[mask]].values[:, None]

            u, v = tc_velocity(udlm, vdlm, long_idxs, lat_idxs, time_idxs)

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
            latitudes[step + 1][step + 1 > durations] = np.nan
            longitudes[step + 1][step + 1 > durations] = np.nan

        t1 = time.time()

        logging.info(f"Finished simulating tracks for {month}/{year}. Time taken: {t1 - t0}s")
        print(f"Finished simulating tracks for {month}/{year}. Time taken: {t1 - t0}s")

        np.save(f"/scratch/w85/kr4383/tracks/tracks_{month}_{year}.npy", coords)
