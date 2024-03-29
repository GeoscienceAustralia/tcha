print("Importing.")
import numpy as np
import os
import xarray as xr
from calendar import monthrange
import time
import logging
from mpi4py import MPI


print("Imports dones. Setting up logs.")


logging.basicConfig(filename='era5_dlm.log', level=logging.DEBUG)


def load_dlm(year, month):
    """
    Loads the pressure level u and v ERA5 files and calculate the DLM.

    This using dask to lazily load the minimum data needed.
    """

    days = monthrange(year, month)[1]
     # Australian region
    lat_slice = slice(0, -40)
    long_slice = slice(80, 170)
    pslice = slice(300, 850)

    # the pressure levels to integrate
    pressure = np.array(
        [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850]
    )

    # trapezoidal integration coefficients
    coeff = np.zeros(len(pressure))
    coeff[1:] += 0.5 * np.diff(pressure)
    coeff[:-1] += 0.5 * np.diff(pressure)
    coeff = coeff.reshape((1, -1, 1, 1))

    prefix = "/g/data/rt52/era5/pressure-levels/reanalysis"
    ufile = f"{prefix}/u/{year}/u_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
    vfile = f"{prefix}/v/{year}/v_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"

    # load data and calculate DLM lazily
    uds = xr.open_dataset(ufile, chunks={'time': 24})  # xr.open_dataset(ufile, chunks='auto')

    uenv = uds.u.sel(level=pslice, longitude=long_slice, latitude=lat_slice)
    # udlm = np.trapz(uenv.data, pressure, axis=1) / 550
    udlm = (coeff * uenv).sum(axis=1).compute(scheduler='single-threaded') / 550

    vds = xr.open_dataset(vfile, chunks={'time': 24})
    venv = vds.v.sel(level=pslice, longitude=long_slice, latitude=lat_slice)  #.compute(scheduler='single-threaded')
    # vdlm = np.trapz(venv.data, pressure, axis=1) / 550
    vdlm = (coeff * venv).sum(axis=1).compute(scheduler='single-threaded') / 550

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


comm = MPI.COMM_WORLD
years = np.arange(1981, 2024)
rank = comm.Get_rank()
rank_years = years[(years % comm.size) == rank]
print("Starting simulation.")

for year in rank_years:
    for month in range(1, 13):

        if os.path.exists(f"/scratch/w85/cxa547/era5dlm/v_dlm_{month}_{year}.nc"):
            print(f"Already processed {month}/{year}")
            continue
        t0 = time.time()
        logging.info(f"Loading data for {month}/{year}")
        print(f"Loading data for {month}/{year}")

        udlm, vdlm = load_dlm(year, month)
        t1 = time.time()

        logging.info(f"Finished loading data for {month}/{year}. Time taken: {time.time() - t0}s")
        print(f"Finished loading data for {month}/{year}. Time taken: {time.time() - t0}s")

        logging.info(f"Saving data for {month}/{year}")
        print(f"Saving data for {month}/{year}")

        udlm.to_netcdf(f"/scratch/w85/cxa547/era5dlm/u_dlm_{month}_{year}.nc")
        vdlm.to_netcdf(f"/scratch/w85/cxa547/era5dlm/v_dlm_{month}_{year}.nc")

        logging.info(f"Finished saving data for {month}/{year}. Time taken: {time.time() - t1}s")
        print(f"Finished saving data for {month}/{year}. Time taken: {time.time() - t1}s")

