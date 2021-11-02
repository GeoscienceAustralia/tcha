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

uout = np.empty((10, 161, 361))
vout = np.empty((10, 161, 361))

prev_eventid = None
prev_month = None

pressure = np.array(
    [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850]
)

times = np.sort(df.Datetime)[:10]

for i, timestamp in enumerate(np.sort(df.Datetime)[:10]):
    timestamp = pd.to_datetime(timestamp)
    month = timestamp.month
    year = timestamp.year
    days = monthrange(year, month)[1]

    lat_slice = slice(0, -40)
    long_slice = slice(80, 170)
    pslice = pslice = slice(300, 850)

    ufile = f"{prefix}/u/{year}/u_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
    vfile = f"{prefix}/v/{year}/v_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"

    uds = xr.open_dataset(ufile, chunks='auto')
    uenv = uds.u.sel(time=timestamp, level=pslice, longitude=long_slice, latitude=lat_slice).compute()
    udlm = np.trapz(uenv.data * pressure[:, None, None], pressure, axis=0) / np.trapz(pressure, pressure)
    uout[i] = udlm

    vds = xr.open_dataset(vfile, chunks='auto')
    venv = vds.v.sel(time=timestamp, level=pslice, longitude=long_slice, latitude=lat_slice).compute()
    vdlm = np.trapz(venv.data * pressure[:, None, None], pressure, axis=0) / np.trapz(pressure, pressure)
    vout[i] = vdlm


uout = xr.DataArray(
    uout,
    dims=["time", "latitude", "longitude"],
    coords={"time": times, "latitude": uenv.coords["latitude"].data, "longitude": uenv.coords["longitude"].data},
)