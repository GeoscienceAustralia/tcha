"""
Extract required ERA5 data from NCI replication dataset

Requirements:
- access to NCI project rt52
@author: Craig Arthur

"""
import os
import numpy as np
import glob
import xarray as xr
from mpi4py import MPI

import dask.array as da
NCPUS = int(os.environ.get('NCPUS'))

start_year = 1981
end_year = 2023

years = range(start_year, end_year+1)

# single-level variables
base_dir = "/scratch/w85/cxa547/tcr/data/era5-new"
os.makedirs(base_dir, exist_ok = True)

mon2d = ['sst', 'sp']
mon3d = ['q', 't']
day3d = ['u', 'v']


pressure_levels= [70, 100, 125, 150, 175, 200,
                  225, 250, 300, 350, 400, 450,
                  500, 550, 600, 650, 700, 750,
                  775, 800, 825, 850, 875, 900,
                  925, 950, 975,1000,]

# The pressure levels to integrate for DLM integration
# I define two here - a deep (`dlmd`) and shallow (`dlms`)
# representing the two steering layers for weak and intense storms
# in Velden and Leslie (1991) - see their Figure 2 and Table 2.
dlmd = np.array(
        [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850]
    )
dlms = np.array(
        [500, 550, 600, 650, 700, 750, 775, 800, 825, 850]
    )

def calcdlm(ds, var, prs):
    """
    Calculate mass-weighted vertically-averaged deep layer mean flow
    Use a simple trapezoidal approximation for the weighted integration.

    :param ds: `xr.Dataset` containing wind speed at multiple levels
    :param var: variable name from the dataset
    :param prs: `np.array` of pressure values to use in the calculation
                of the deep-layer mean flow
    """
    # trapezoidal integration coefficients
    coeff = np.zeros(len(prs))
    minprs = dlmprs.min()
    maxprs = dlmprs.max()
    coeff[1:] += 0.5 * np.diff(prs)
    coeff[:-1] += 0.5 * np.diff(prs)
    coeff = coeff.reshape((1, -1, 1, 1))
    uenv = ds[var].sel(level=slice(minprs, maxprs))
    udlm = (coeff * uenv).sum(axis=1) / (maxprs - minprs)
    return udlm

def process(year):
    dest_dir = f"{base_dir}/{year:0d}"
    os.makedirs(dest_dir, exist_ok = True)

    src_dir = "/g/data/rt52/era5/single-levels/monthly-averaged"
    for var in mon2d:
        print(f"Extracting {var} monthly 2-d data for {year}")
        destfn = os.path.join(dest_dir, f"era5_{var}_monthly_{year:0d}.nc")
        if os.path.exists(destfn):
            print(f"Already extracted {var} monthly 2-d data for {year}")
            continue
        fnlist = sorted(glob.glob(os.path.join(src_dir, var, f"{year:0d}", f"{var}_era5_*_sfc_*.nc")))
        ds = xr.open_mfdataset(fnlist, combine='nested', concat_dim='time', chunks={'latitude': 721, 'longitude': 1440, 'time': -1})
        outds = ds.isel(longitude=slice(0, 1440, 4), latitude=slice(0, 721, 4))
        outds.to_netcdf(destfn)

    src_dir = "/g/data/rt52/era5/pressure-levels/monthly-averaged"
    for var in mon3d:
        print(f"Extracting {var} monthly 3-d data for {year}")
        destfn = os.path.join(dest_dir, f"era5_{var}_monthly_{year:0d}.nc")
        if os.path.exists(destfn):
            print(f"Already extracted {var} monthly 3-d data for {year}")
            continue
        fnlist = sorted(glob.glob(os.path.join(src_dir, var, f"{year:0d}", f"{var}_era5_*_pl_*.nc")))
        ds = xr.open_mfdataset(fnlist, combine='nested', concat_dim='time', chunks={'latitude': 721, 'longitude': 1440, 'time': -1}, parallel=True)
        ntimes = len(ds.time)
        outds = ds.isel(longitude=slice(0, 1440, 4), latitude=slice(0, 721, 4))
        outds = outds.sel(level=pressure_levels)
        outds.to_netcdf(destfn)

    src_dir = "/g/data/rt52/era5/pressure-levels/reanalysis"
    for var in day3d:
        print(f"Extracting {var} daily 3-d data for {year}")
        destfn = os.path.join(dest_dir, f"era5_{var}_daily_{year:0d}.nc")
        if os.path.exists(destfn):
            print(f"Already extracted {var} daily 3-d data for {year}")
            continue
        fnlist = sorted(glob.glob(os.path.join(src_dir, var, f"{year:0d}", f"{var}_era5_*_pl_*.nc")))
        ds = xr.open_mfdataset(fnlist, combine='nested', concat_dim='time', chunks={'latitude': 721, 'longitude': 1440, 'time': -1}, parallel=True)
        ntimes = len(ds.time)
        outds = ds.isel(time=slice(0, ntimes, 6), longitude=slice(0, 1440, 4), latitude=slice(0, 721, 4))
        outds = outds.sel(level=[850, 250])
        outds.to_netcdf(destfn)

        # Calculate deep-layer mean flow for two different depths
        dlmdd = calcdlm(ds.isel(time=slice(0, ntimes, 6), longitude=slice(0, 1440, 4), latitude=slice(0, 721, 4)), var, dlmd)
        dlmds = calcdlm(ds.isel(time=slice(0, ntimes, 6), longitude=slice(0, 1440, 4), latitude=slice(0, 721, 4)), var, dlms)
        destfn = os.path.join(dest_dir, f"era5_{var}dlm_daily_{year:0d}.nc")
        dlm = xr.Dataset(data_vars={f"{var}s":dlmds, f"{var}d":dlmd})
        dlm.to_netcdf(destfn)


comm = MPI.COMM_WORLD
years = np.arange(1981, 2023+1)
rank = comm.Get_rank()
rank_years = years[(years % comm.size) == rank]
for year in rank_years:
    process(year)
