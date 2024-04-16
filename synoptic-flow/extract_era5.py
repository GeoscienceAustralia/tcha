"""
Extract required ERA5 data for the Lin TC model from NCI's ERA5 replication
dataset. Additionally, it extracts a deep layer mean (DLM) flow, based on
the definitions in Velden and Leslie (1991).

References:
Lin, J., R. Rousseau-Rizzi, C.-Y. Lee, and A. Sobel, 2023: An Open-Source,
Physics-Based, Tropical Cyclone Downscaling Model With Intensity-Dependent
Steering. Journal of Advances in Modeling Earth Systems, 15, e2023MS003686,
https://doi.org/10.1029/2023MS003686.

Velden, C. S., and L. M. Leslie, 1991: The Basic Relationship between
Tropical Cyclone Intensity and the Depth of the Environmental Steering Layer
in the Australian Region. Weather and Forecasting, 6 (10).

Hersbach, H., and Coauthors, 2020: The ERA5 global reanalysis. Quarterly
Journal of the Royal Meteorological Society, 146, 1999-2049,
https://doi.org/10.1002/qj.3803.

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

NCPUS = int(os.environ.get("NCPUS"))

start_year = 1981
end_year = 2023

years = range(start_year, end_year + 1)

# single-level variables
base_dir = "/scratch/w85/cxa547/tcr/data/era5"
os.makedirs(base_dir, exist_ok=True)

mon2d = ["sst", "sp"]
mon3d = ["q", "t"]
day3d = ["u", "v"]


pressure_levels = [
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]

# The pressure levels to integrate for DLM integration
# I define two here - a deep (`dlmd`) and shallow (`dlms`)
# representing the two steering layers for weak and intense storms
# in Velden and Leslie (1991) - see their Figure 2 and Table 2.
dlmd = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850])
dlms = np.array([500, 550, 600, 650, 700, 750, 775, 800, 825, 850])


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
    minprs = prs.min()
    maxprs = prs.max()
    coeff[1:] += 0.5 * np.diff(prs)
    coeff[:-1] += 0.5 * np.diff(prs)
    coeff = coeff.reshape((1, -1, 1, 1))
    uenv = ds[var].sel(level=slice(minprs, maxprs))
    udlm = (coeff * uenv).sum(axis=1) / (maxprs - minprs)
    return udlm


def process(year):
    dest_dir = f"{base_dir}/{year:0d}"
    os.makedirs(dest_dir, exist_ok=True)

    src_dir = "/g/data/rt52/era5/single-levels/monthly-averaged"
    for var in mon2d:
        print(f"Extracting {var} monthly 2-d data for {year}")
        destfn = os.path.join(dest_dir, f"era5_{var}_monthly_{year:0d}.nc")
        if os.path.exists(destfn):
            print(f"Already extracted {var} monthly 2-d data for {year}")
            continue
        fnlist = sorted(
            glob.glob(
                os.path.join(
                    src_dir, var, f"{year:0d}", f"{var}_era5_*_sfc_*.nc"
                    )
            )
        )
        ds = xr.open_mfdataset(
            fnlist,
            combine="nested",
            concat_dim="time",
            chunks={"latitude": 721, "longitude": 1440, "time": -1},
        )
        outds = ds.isel(longitude=slice(0, 1440, 4),
                        latitude=slice(0, 721, 4))
        outds.to_netcdf(destfn)

    src_dir = "/g/data/rt52/era5/pressure-levels/monthly-averaged"
    for var in mon3d:
        print(f"Extracting {var} monthly 3-d data for {year}")
        destfn = os.path.join(dest_dir, f"era5_{var}_monthly_{year:0d}.nc")
        if os.path.exists(destfn):
            print(f"Already extracted {var} monthly 3-d data for {year}")
            continue
        fnlist = sorted(
            glob.glob(os.path.join(
                src_dir, var, f"{year:0d}", f"{var}_era5_*_pl_*.nc")
                      )
        )
        ds = xr.open_mfdataset(
            fnlist,
            combine="nested",
            concat_dim="time",
            chunks={"latitude": 721, "longitude": 1440, "time": -1},
            parallel=True,
        )
        ntimes = len(ds.time)
        outds = ds.isel(longitude=slice(0, 1440, 4),
                        latitude=slice(0, 721, 4))
        outds = outds.sel(level=pressure_levels)
        outds.to_netcdf(destfn)

    src_dir = "/g/data/rt52/era5/pressure-levels/reanalysis"
    for var in day3d:
        print(f"Extracting {var} daily 3-d data for {year}")
        destfn = os.path.join(dest_dir, f"era5_{var}_daily_{year:0d}.nc")
        if os.path.exists(destfn):
            print(f"Already extracted {var} daily 3-d data for {year}")
            continue
        fnlist = sorted(
            glob.glob(os.path.join(
                src_dir, var, f"{year:0d}", f"{var}_era5_*_pl_*.nc")
                      )
        )
        ds = xr.open_mfdataset(
            fnlist,
            combine="nested",
            concat_dim="time",
            chunks={"latitude": 721, "longitude": 1440, "time": -1},
            parallel=True,
        )
        ntimes = len(ds.time)
        outds = ds.isel(
            time=slice(0, ntimes, 6),
            longitude=slice(0, 1440, 4),
            latitude=slice(0, 721, 4),
        )
        outds = outds.sel(level=[850, 250])
        outds.attrs = ds.attrs
        outds.to_netcdf(destfn)

    for var in day3d:
        # Calculate deep-layer mean flow for two different depths
        print(f"Extracting {var} daily DLM data for {year}")
        destfn = os.path.join(dest_dir, f"era5_{var}dlm_daily_{year:0d}.nc")
        if os.path.exists(destfn):
            print(f"Already extracted {var} daily DLM data for {year}")
            continue

        dlmdd = calcdlm(
            ds.isel(
                time=slice(0, ntimes, 6),
                longitude=slice(0, 1440, 4),
                latitude=slice(0, 721, 4),
            ),
            var,
            dlmd,
        )
        dlmds = calcdlm(
            ds.isel(
                time=slice(0, ntimes, 6),
                longitude=slice(0, 1440, 4),
                latitude=slice(0, 721, 4),
            ),
            var,
            dlms,
        )

        dvar = xr.concat([dlmds, dlmdd], dim="level")
        dvar = dvar.transpose("time", "level", "latitude", "longitude")
        dlm = xr.Dataset(data_vars={var: dvar})
        dlm.attrs = ds.attrs
        dlm.to_netcdf(destfn)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
rank_years = years[(years % comm.size) == rank]
for year in rank_years:
    process(year)
