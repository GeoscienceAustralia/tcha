import xarray as xr
from calendar import monthrange
import numpy as np
import pandas as pd
import os
import pyproj
from tqdm import tqdm
from era5_extract import load_otcr_df


DATA_DIR = os.path.expanduser("~")
# pressure levels
# array([   1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,
#         125,  150,  175,  200,  225,  250,  300,  350,  400,  450,  500,
#         550,  600,  650,  700,  750,  775,  800,  825,  850,  875,  900,
#         925,  950,  975, 1000], dtype=int32)

def load_data(time, lat, lon):
    """
    Loads the pressure level rh ERA5 files and
    computes an aqverage over 5 x 5 degree box for the specified time (rounded to the nearest hour) and location.

    This using dask to lazily load the minimum data needed.
    """
    time = time.round('H')
    time = time - pd.Timedelta(hours=24)
    year = time.year
    month = time.month
    days = monthrange(year, month)[1]
    lat_slice = slice(lat + 2.5, lat - 2.5)
    long_slice = slice(lon - 2.5, lon + 2.5)

    vars = ["sst", "sp", "2d", "2t"]
    prefix = "/g/data/rt52/era5/pressure-levels/reanalysis"
    out = []
    
    fp = f"{prefix}/r/{year}/r_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
    ds = xr.open_dataset(fp, chunks={'time': 24})

    try:
        y = ds.sel(time=time, longitude=long_slice, latitude=lat_slice).mean(['latitude', 'longitude']).compute(scheduler='single-threaded')
        out = y.r.data
    except KeyError as e:
        print("Bad time:", time)
        out = np.zeros(len(ds.level))
        
    return out


if __name__ == "__main__":
    df = load_otcr_df()

    rh = []

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        rh.append(load_data(row.TM, row.LAT, row.LON))
        # print(bran[-1])

    rh = np.array(rh)
    np.save(os.path.join(DATA_DIR, "tc_intensity_rh.npy"), rh)

    arr = np.load(os.path.join(DATA_DIR, "tc_intensity_rh.npy"))
    print("Array saved correctly:", np.allclose(arr, rh))