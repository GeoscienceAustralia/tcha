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
logging.basicConfig(filename='climatology.log', level=logging.DEBUG)


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

    u_mean = udlm.mean(dim='time').compute()
    u_std = udlm.std(dim='time').compute()
    v_mean = vdlm.mean(dim='time').compute()
    v_std = vdlm.std(dim='time').compute()

    ds = xr.merge({'u_mean': u_mean, 'u_std': u_std, 'v_mean': v_mean, 'v_std': v_std})
    return ds


def perturbation(t, N, T, phase):
    n = np.arange(1, N + 1)[None, :]

    n_inv_23 = n ** (-3 / 2)
    a = np.sqrt(2) / np.linalg.norm(n_inv_23)

    return a * (n_inv_23 * np.sin(2 * np.pi * (phase + n * t[:, None] / T))).sum(axis=1)


def tc_velocity(climatology, t, N, T, phase, idxs):
    pert = perturbation(t, N, T, phase)

    u = climatology[0].take(idxs) + climatology[1].take(idxs) * pert
    v = climatology[2].take(idxs) + climatology[3].take(idxs) * pert

    u = -4.5205 + 0.8978 * u * 3.6
    v = -1.2542 + 0.7877 * v * 3.6

    return u, v


N = 15
T = 15 * 24  # 15 days
phase = np.random.random(N)

u_phase = np.random.random(N)
v_phase = np.random.random(N)

genesis_sampler = SamplingOrigin(
    kdeOrigin="/g/data/fj6/TCRM/TCHA18/process/originPDF.nc"
)
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

for month in rank_months:

    logging.info(f"Simulating tracks for {month}")
    print(f"Simulating tracks for {month}")

    climatology = get_climatology(month)

    climatology.to_netcdf(f"/scratch/w85/kr4383/climatology/climatology_{month}.netcdf")
