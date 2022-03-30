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
    
    # TODO: get time interpolation working

    vars = ["sst", "sp", "2d", "2t"]
    prefix = "/g/data/gb6/BRAN/BRAN2020/daily"
    
    try:
        fp = f"{prefix}/ocean_mld_{year}_{month:02d}.nc"
        ds = xr.open_dataset(fp, chunks={'Time': 24})
        y = ds.sel(xt_ocean=long_slice, yt_ocean=lat_slice).mean(dim=["xt_ocean", "yt_ocean"]).compute()
        y = y.sel(Time=time) #interp
        hm = y['mld'].data

        # fp = f"{prefix}/ocean_temp_{year}_{month:02d}.nc"
        # ds = xr.open_dataset(fp, chunks={'Time': 24})
        # y = ds.sel(xt_ocean=long_slice, yt_ocean=lat_slice).mean(dim=["xt_ocean", "yt_ocean"]).compute()
        # y = y.sel(Time=time)
        # temp = y.temp.data.copy()
        # depth = y.st_ocean.data.copy()
        # temp = temp[depth < 1000]
        # depth = depth[depth < 1000]

        # # calculate mixed layer depth averaged temperature 
        # mask = depth < hm
        # ml_tmp = np.trapz(temp[mask], depth[mask]) / (depth[mask][-1] - depth[0])

        # # calculate temperature lapse rate and jump across mixed layer boundary
        # mask = (depth > hm)
        # m, b, *_ = linregress(depth[mask], temp[mask])
        # dsst = ml_tmp - (m * hm + b)
        # gm = -100 * m
        # # print(hm, dsst, gm)

    except Exception as e:
        print(e)
        print(time)
        print()
        hm = np.nan
        dsst = np.nan
        gm = np.nan

    return np.array([hm])



if __name__ == "__main__":
    df = load_otcr_df()
    df.TM = df.TM - pd.Timedelta(days=1)

    bran = []

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        bran.append(load_data(row.TM, row.LAT, row.LON))
        # print(bran[-1])

    bran = np.array(bran)
    np.save(os.path.join(DATA_DIR, "tc_intensity_bran2020.npy"), bran)

    arr = np.load(os.path.join(DATA_DIR, "tc_intensity_bran2020.npy"))
    print("Array saved correctly:", np.allclose(arr, bran))