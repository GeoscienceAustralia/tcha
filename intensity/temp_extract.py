import xarray as xr
from calendar import monthrange
import numpy as np
import pandas as pd
import os
import pyproj
from tqdm import tqdm
from era5_extract import load_otcr_df
from scipy.stats import linregress

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
        out = np.concatenate(depth, temp)

    except Exception as e:
        out = np.zeroes(2 * 51)

    return out


if __name__ == "__main__":
    df = load_otcr_df()
    df.TM = df.TM - pd.Timedelta(days=1)

    temp = []

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        temp.append(load_data(row.TM, row.LAT, row.LON))
        # print(bran[-1])

    bran = np.array(temp)
    np.save(os.path.join(DATA_DIR, "tc_intensity_temp_profile.npy"), temp)

    arr = np.load(os.path.join(DATA_DIR, "tc_intensity_temp_profile.npy"))
    print("Array saved correctly:", np.allclose(arr, bran))
