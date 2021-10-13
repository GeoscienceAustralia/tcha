import numpy as np
import os
import xarray as xr
import pandas as pd
from calendar import monthrange
import time
# import dask


prefix = "/g/data/rt52/era5/pressure-levels/reanalysis"
df = pd.read_csv("jtwc_clean.csv")

dt = pd.Timedelta(1, units='hours')
df.Datetime = pd.to_datetime(df.Datetime)

t0 = time.time()

out = []
prev_eventid = None
prev_month = None

for row in list(df.itertuples()):
    timestamp = row.Datetime

    month = timestamp.month
    year = timestamp.year
    days = monthrange(year, month)[1]

    lat_min = np.floor(row.Latitude * 4) / 4
    lon_min = np.floor(row.Longitude * 4) / 4
    lat_slice = slice(lat_min + 0.25, lat_min)
    long_slice = slice(lon_min, lon_min + 0.25)

    ufile = f"{prefix}/u/{year}/u_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
    vfile = f"{prefix}/v/{year}/v_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"

    uds = xr.open_dataset(ufile, chunks='auto')
    uds_850 = uds.u.sel(time=timestamp, level=850, longitude=long_slice, latitude=lat_slice).compute()
    uds_250 = uds.u.sel(time=timestamp, level=250, longitude=long_slice, latitude=lat_slice).compute()

    vds = xr.open_dataset(vfile, chunks='auto')
    vds_850 = vds.v.sel(time=timestamp, level=850, longitude=long_slice, latitude=lat_slice).compute()
    vds_250 = vds.v.sel(time=timestamp, level=250, longitude=long_slice, latitude=lat_slice).compute()

    uds_interp_850 = uds_850.interp(latitude=row.Latitude, longitude=row.Longitude)
    vds_interp_850 = vds_850.interp(latitude=row.Latitude, longitude=row.Longitude)

    uds_interp_250 = uds_250.interp(latitude=row.Latitude, longitude=row.Longitude)
    vds_interp_250 = vds_250.interp(latitude=row.Latitude, longitude=row.Longitude)
    out.append([uds_interp_250, vds_interp_250, uds_interp_850, vds_interp_850])


print(time.time() - t0, 's')
out = np.array(out)
df['u_250'] = out[:, 0]
df['v_250'] = out[:, 1]
df['u_850'] = out[:, 2]
df['v_850'] = out[:, 3]
df.to_csv("jtwc_era5.csv")
