import numpy as np
import os
import xarray as xr
import pandas as pd
from calendar import monthrange
import time
# import dask


prefix = "/g/data/rt52/era5/pressure-levels/reanalysis"
df = pd.read_csv(os.path.expanduser("~/jtwc_clean.csv"))

dt = pd.Timedelta(1, units='hours')
df.Datetime = pd.to_datetime(df.Datetime)

t0 = time.time()

out = np.empty((len(df), 6, 6, 51, 51))
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

    ufile = f"{prefix}/u/{year}/u_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
    vfile = f"{prefix}/v/{year}/v_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"

    uds = xr.open_dataset(ufile, chunks='auto')
    out[i, 0, ...] = uds.u.sel(time=time_slice, level=200, longitude=long_slice, latitude=lat_slice).compute()
    out[i, 1, ...] = uds.u.sel(time=time_slice, level=500, longitude=long_slice, latitude=lat_slice).compute()
    out[i, 2, ...] = uds.u.sel(time=time_slice, level=850, longitude=long_slice, latitude=lat_slice).compute()

    vds = xr.open_dataset(vfile, chunks='auto')
    out[i, 3, ...] = vds.v.sel(time=time_slice, level=200, longitude=long_slice, latitude=lat_slice).compute()
    out[i, 4, ...] = vds.v.sel(time=time_slice, level=500, longitude=long_slice, latitude=lat_slice).compute()
    out[i, 5, ...] = vds.v.sel(time=time_slice, level=850, longitude=long_slice, latitude=lat_slice).compute()


print(time.time() - t0, 's')
np.save(os.path.expanduser("~/era5_time_series_dump"), out)

