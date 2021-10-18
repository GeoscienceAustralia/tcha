import numpy as np
import os
import xarray as xr
import pandas as pd
from calendar import monthrange
import time
import geopy
from geopy.distance import geodesic


prefix = "/g/data/rt52/era5/pressure-levels/reanalysis"
df = pd.read_csv(os.path.expanduser("~/jtwc_clean.csv"))

dt = pd.Timedelta(1, units='hours')
df.Datetime = pd.to_datetime(df.Datetime)

t0 = time.time()

out = []
prev_eventid = None
prev_month = None

lats = [None]
lons = [None]
ids = set()

for row in list(df.itertuples())[:10]:

    if row.eventid not in ids:
        # using forward difference
        # discard last position of previous TC track
        lats[-1] = row.Latitude
        lons[-1] = row.Longitude
        ids.add(row.eventid)

    if np.isnan(lats[-1]):
        # TC went out of domain
        lats.append(np.nan)
        lons.append(np.nan)
        continue

    timestamp = row.Datetime

    month = timestamp.month
    year = timestamp.year
    days = monthrange(year, month)[1]

    lat_slice = slice(lats[-1] + 0.5, lats[-1] - 0.5)
    long_slice = slice(lons[-1] - 0.5, lons[-1] + 0.5)

    ufile = f"{prefix}/u/{year}/u_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
    vfile = f"{prefix}/v/{year}/v_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"

    uds = xr.open_dataset(ufile, chunks='auto')
    uds_850 = uds.u.sel(time=timestamp, level=850, longitude=long_slice, latitude=lat_slice).compute()
    uds_250 = uds.u.sel(time=timestamp, level=250, longitude=long_slice, latitude=lat_slice).compute()

    vds = xr.open_dataset(vfile, chunks='auto')
    vds_850 = vds.v.sel(time=timestamp, level=850, longitude=long_slice, latitude=lat_slice).compute()
    vds_250 = vds.v.sel(time=timestamp, level=250, longitude=long_slice, latitude=lat_slice).compute()

    try:
        uds_interp_850 = uds_850.interp(latitude=row.Latitude, longitude=row.Longitude)
        vds_interp_850 = vds_850.interp(latitude=row.Latitude, longitude=row.Longitude)

        uds_interp_250 = uds_250.interp(latitude=row.Latitude, longitude=row.Longitude)
        vds_interp_250 = vds_250.interp(latitude=row.Latitude, longitude=row.Longitude)

        u = -3.0575 + 0.4897 * uds_interp_850 + 0.6752 * uds_interp_250 + np.random.normal(loc=0, size=1, scale=10.85)[0]
        v = -5.1207 + 0.3257 * vds_interp_850 + 0.1502 * vds_interp_250 + np.random.normal(loc=0, size=1, scale=7.232)[0]

        dt = 6  # hours
        bearing = np.atan(u / v)
        distance = np.sqrt(u ** 2 + v ** 2) * dt

        origin = geopy.Point(lats[-1], lons[-1])
        destination = geodesic(kilometers=distance).destination(origin, bearing)
        lats.append(destination.latitude)
        lons.append(destination.longitude)

    except IndexError:
        lats.append(np.nan)
        lons.append(np.nan)


print(time.time() - t0, 's')
out = np.array(out)

df = df.iloc[:out.shape[0]].copy()
df['u_250'] = out[:, 0]
df['v_250'] = out[:, 1]
df['u_850'] = out[:, 2]
df['v_850'] = out[:, 3]
df.to_csv("coarse_tc_tracks.csv")
