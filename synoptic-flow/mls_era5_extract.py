import numpy as np
import os
import xarray as xr
import pandas as pd
from calendar import monthrange
import time
# import dask


prefix = "/g/data/rt52/era5/single-levels/reanalysis"
df = pd.read_csv(os.path.expanduser("~/jtwc_clean.csv"))

dt = pd.Timedelta(1, units='hours')
df.Datetime = pd.to_datetime(df.Datetime)

t0 = time.time()

# out = np.empty((len(df), 6, 6, 51, 51))
out = np.empty((len(df), 51, 51))
prev_eventid = None
prev_month = None

for i, row in enumerate(list(df.itertuples())[:]):
    timestamp = row.Datetime

    month = timestamp.month
    year = timestamp.year
    days = monthrange(year, month)[1]

    lat_cntr = 0.25 * np.round(row.Latitude * 4)
    lon_cntr = 0.25 * np.round(row.Longitude * 4)

    lat_slice = slice(lat_cntr + 6.25, lat_cntr - 6.25)
    long_slice = slice(lon_cntr - 6.25, lon_cntr + 6.25)
    time_slice = slice(timestamp, timestamp + np.timedelta64(5, 'h'))

    mslfile = f"{prefix}/msl/{year}/msl_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{days}.nc"

    mslds = xr.open_dataset(mslfile, chunks='auto')
    arr = mslds.u.sel(time=timestamp, longitude=long_slice, latitude=lat_slice).compute()
    try:
        out[i] = arr
    except ValueError:
        offset = 51 - arr.shape[-1]
        out[i, :, :-offset] = arr
        out[i, :, -offset:] = np.nan

print(time.time() - t0, 's')
np.save(os.path.expanduser("~/era5_msl_dump"), out)

