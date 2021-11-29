import numpy as np
import os
import xarray as xr
import pandas as pd
from calendar import monthrange
import time
import scipy.stats as stats
from mpi4py import MPI
import geopy
from geopy.distance import geodesic as gdg
import sys

sys.path.insert(0, "")

from StatInterface.SamplingOrigin import SamplingOrigin


def load_dlm(year, month):
    days = monthrange(year, month)[1]

    lat_slice = slice(0, -40)
    long_slice = slice(80, 170)
    pslice = slice(300, 850)

    ufile = f"{prefix}/u/{year}/u_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
    vfile = f"{prefix}/v/{year}/v_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"

    uds = xr.open_dataset(ufile, chunks='auto')
    uenv = uds.u.sel(level=pslice, longitude=long_slice, latitude=lat_slice).compute()
    udlm = np.trapz(uenv.data, pressure, axis=1) / 550

    vds = xr.open_dataset(vfile, chunks='auto')
    venv = vds.v.sel(level=pslice, longitude=long_slice, latitude=lat_slice).compute()
    vdlm = np.trapz(venv.data, pressure, axis=1) / 550

    udlm = xr.DataArray(
        udlm,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": uenv.coords["time"].data,
            "latitude": uenv.coords["latitude"].data,
            "longitude": uenv.coords["longitude"].data},
    )

    vdlm = xr.DataArray(
        vdlm,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": uenv.coords["time"].data,
            "latitude": uenv.coords["latitude"].data,
            "longitude": uenv.coords["longitude"].data},
    )

    return udlm, vdlm


def tc_velocity(udlm, vdlm, latitude, longitude):
    lat_cntr = 0.25 * np.round(latitude * 4)
    lon_cntr = 0.25 * np.round(longitude * 4)
    lat_slice = slice(lat_cntr + 6.25, lat_cntr - 6.25)
    long_slice = slice(lon_cntr - 6.25, lon_cntr + 6.25)

    u = udlm.sel(time=timestamp, longitude=long_slice, latitude=lat_slice).mean()
    v = vdlm.sel(time=timestamp, longitude=long_slice, latitude=lat_slice).mean()

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

prefix = "/g/data/rt52/era5/pressure-levels/reanalysis"

genesis_sampler = SamplingOrigin(
    kdeOrigin="/g/data/fj6/TCRM/TCHA18/process/originPDF.nc"
)
pressure = np.array(
    [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850]
)

years = np.arange(1981, 2021)
rank = comm.Get_rank()
rank_years = years[(years % 16) == rank]

repeats = 10_000 / 40  # repeat the catalogue of 40s years until a total of 10_000 years have been simulated

month_rates = {
    1: 2.3415, 2: 2.0976, 3: 2.0, 12: 1.439, 4: 1.3659, 11: 0.5122,
    5: 0.3659, 10: 0.122, 7: 0.0488, 6: 0.0488, 8: 0.0244, 9: 0.0
}


for year in rank_years:
    print(f"Loading data for {month}/{year}")
    udlm, vdlm = load_dlm(year, 1)
    print(f"Finished loading data for {month}/{year}. Time taken: {t1 - t0}s")
    for month in range(1, 13):

        t0 = time.time()
        # sufficient repeats that the sum should be equal to the mean * number of repeats
        num_events = int(np.round(month_rates[month] * repeats))

        t1 = time.time()

        print(f"Simulating tracks for {month}/{year}")
        revisit = []

        days = monthrange(year, month)[1]
        latitudes = [[] for _ in range(num_events)]
        longitudes = [[] for _ in range(num_events)]
        for idx in range(num_events):
            uid = f"{idx}-{month}-{year}"
            # uniform random sample day of month and hour
            day = np.random.randint(1, days + 1)
            hour = np.random.randint(0, 24)
            timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour)

            duration = int(np.round(stats.lognorm.rvs(0.5491, 0., 153.27)))  # duration of cyclone in hours

            origin = genesis_sampler.generateSamples(1)
            latitudes[idx].append(origin[1])
            longitudes[idx].append(origin[0])

            for step in range(duration):

                if timestamp.month != month:
                    revisit.append((idx, timestamp, duration - step))
                    break

                # calculate TC velocity and time step
                u, v = tc_velocity(udlm, vdlm, latitudes[idx][-1], longitudes[idx][-1])
                dest = timestep(latitudes[idx][-1], longitudes[idx][-1], u, v, dt=1)
                latitudes[idx].append(dest.latitude)
                longitudes[idx].append(dest.longitude)

        print(f"Finished simulating tracks for {month}/{year}. Time taken: {time.time() - t1}s")
        # in case TC track exceeds 1 month
        udlm, vdlm = load_dlm(year, month + 1)
        for idx, timestamp, duration in revisit:
            uid = f"{idx}-{month}-{year}"

            for step in range(duration):

                if timestamp.month != month + 1:
                    break

                # calculate TC velocity and time step
                u, v = tc_velocity(udlm, vdlm, latitudes[idx][-1], longitudes[idx][-1])
                dest = timestep(latitudes[idx][-1], longitudes[idx][-1], u, v, dt=1)
                latitudes[idx].append(dest.latitude)
                longitudes[idx].append(dest.longitude)

