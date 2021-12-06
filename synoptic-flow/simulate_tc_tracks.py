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

    return udlm, vdlm


def tc_velocity(udlm_1, vdlm_1, udlm_2, vdlm_2, latitude, longitude, t):
    lat_cntr = 0.25 * np.round(latitude * 4)
    lon_cntr = 0.25 * np.round(longitude * 4)
    lat_slice = slice(lat_cntr + 6.25, lat_cntr - 6.25)
    long_slice = slice(lon_cntr - 6.25, lon_cntr + 6.25)

    try:
        u = -4.5205 + 0.8978 * udlm_1.sel(time=t, longitude=long_slice, latitude=lat_slice).mean()
        v = -1.2542 + 0.7877 * vdlm_1.sel(time=t, longitude=long_slice, latitude=lat_slice).mean()

    except KeyError:
        u = -4.5205 + 0.8978 * udlm_2.sel(time=t, longitude=long_slice, latitude=lat_slice).mean()
        v = -1.2542 + 0.7877 * vdlm_2.sel(time=t, longitude=long_slice, latitude=lat_slice).mean()

    return u.u.data, v.v.data


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

rows = []
print("Starting simulation.")
for year in rank_years[:1]:
    for month in range(1, 2):
        udlm_1, vdlm_1 = load_dlm(year, month)
        udlm_2, vdlm_2 = load_dlm(year + (month // 12), (month % 12) + 1)

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

        coords = np.zeros((2, durations.max(), num_events))

        latitudes = coords[0, ...]
        latitudes[0, :] = origin[:, 1]

        longitudes = coords[1, ...]
        longitudes[0, :] = origin[:, 0]

        for step in range(durations.max() - 1):

            u = np.zeros(num_events)
            v = np.zeros(num_events)
            for i in range(num_events):
                u[i], v[i] = tc_velocity(
                    udlm_1, vdlm_1, udlm_2, vdlm_2, latitudes[step, i],
                    longitudes[step, i], timestamps[i] + np.timedelta64(step, 'h')
                )

            dist = np.sqrt(u ** 2 + v ** 2)  # km travelled in one hour

            mask1 = (u >= 0) & (v >= 0)
            mask2 = (u >= 0) & (v < 0)
            mask3 = (u < 0) & (v < 0)

            bearing = 360 + np.arctan(u / v) * 180 / np.pi
            bearing[mask1] = (np.arctan(u / v) * 180 / np.pi)[mask1]
            bearing[mask2] = (180 + np.arctan(u / v) * 180 / np.pi)[mask2]
            bearing[mask3] = (180 + np.arctan(u / v) * 180 / np.pi)[mask3]

            dest = destination(latitudes[step], longitudes[step], dist, bearing)
            latitudes[step + 1, :] = dest[0]
            longitudes[step + 1, :] = dest[1]

            timestamps += np.timedelta64(1, 'h')
            latitudes[step][step > durations] = np.nan
            longitudes[step][step > durations] = np.nan

        t1 = time.time()

        logging.info(f"Finished simulating tracks for {month}/{year}. Time taken: {t1 - t0}s")
        print(f"Finished simulating tracks for {month}/{year}. Time taken: {t1 - t0}s")

        np.save(f"/scratch/w85/kr4383/tracks/tracks_{month}_{year}.npy")


df = pd.DataFrame(rows, columns=["uid", "latitude", "longitude"])
df.to_csv(os.path.expanduser('/scratch/w85/kr4383/simulated_tc_tracks.csv'))
