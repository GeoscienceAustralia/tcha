import xarray as xr
from calendar import monthrange
import numpy as np
import pandas as pd
import os
import pyproj
from tqdm import tqdm
from era5_extract import load_otcr_df
from scipy.stats import linregress
from math import ceil
from mpi4py import MPI


DATA_DIR = os.path.expanduser("~")


def load_data(time, lat, lon):
    """
    Loads the pressure level u and v ERA5 files and calculate the DLM.

    This using dask to lazily load the minimum data needed.
    """
    time = time.round('D') - pd.Timedelta(hours=12)
    year = time.year
    month = time.month
    lat_slice = slice(lat - 2.5, lat + 2.5)
    long_slice = slice(lon - 2.5, lon + 2.5)

    prefix = "/g/data/gb6/BRAN/BRAN2020/daily"

    try:
        fp = f"{prefix}/ocean_temp_{year}_{month:02d}.nc"
        ds = xr.open_dataset(fp, chunks={'Time': 24})
        y = ds.sel(xt_ocean=long_slice, yt_ocean=lat_slice).mean(dim=["xt_ocean", "yt_ocean"]).compute()
        y = y.sel(Time=time)
        temp = y.temp.data.copy()
        depth = y.st_ocean.data.copy()
        out = np.concatenate([depth, temp])

    except Exception as e:
        print(e)
        out = np.zeros(2 * 51)

    return out


if __name__ == "__main__":

    df = load_otcr_df()
    df.TM = df.TM - pd.Timedelta(days=1)
    # df = df.iloc[:70].copy()

    temp = []

    comm = MPI.COMM_WORLD
    years = np.arange(1981, 2021)
    rank = comm.Get_rank()
    numcores = comm.size
    step = (len(df) // numcores)
    start = rank * step

    if rank == (numcores - 1):
        stop = len(df)
    else:
        stop = min((rank + 1) * step, len(df))

    for i in tqdm(range(start, stop)):
        row = df.iloc[i]
        temp.append(load_data(row.TM, row.LAT, row.LON))

    temp = np.array(temp)
    np.save(os.path.join(DATA_DIR, f"tc_intensity_temp_profile_{rank}.npy"), temp)

    arr = np.load(os.path.join(DATA_DIR, f"tc_intensity_temp_profile_{rank}.npy"))
    mask1 = ~np.isnan(arr)
    mask2 = ~np.isnan(temp)

    comm.Barrier()
    print(rank, "Array saved correctly:", np.allclose(arr[mask1], temp[mask2]), arr.shape, temp.shape)
    comm.Barrier()
    if rank == 0:
        print("Combing arrays:")
        combined = np.concatenate(
            [
                np.load(os.path.join(DATA_DIR, f"tc_intensity_temp_profile_{i}.npy"))
                for i in range(numcores)    
            ],
            axis=0
        )
        np.save(os.path.join(DATA_DIR, f"tc_intensity_temp_profile.npy"), combined)
        print("Combined length correct:", len(combined) == len(df))
    comm.Barrier()
    combined = np.load(os.path.join(DATA_DIR, f"tc_intensity_temp_profile.npy"))
    combined = combined[start:stop]
    mask1 = ~np.isnan(arr)
    mask2 = ~np.isnan(combined)
    print(rank, "Combined array saved correctly:", np.allclose(arr[mask1], combined[mask2]))
