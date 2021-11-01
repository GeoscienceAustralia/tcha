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

mat = np.array([
    [0.33497787,  1.6584962 ,  1.36592335, -0.2392938 ,  0.03324952, 0.42346533],
    [0.0097799 , -0.19549655, -0.10247046,  0.25220198,  1.58043784, 1.31665305]
])

beta_drift = np.array([-4.17982995, -0.62322132])

df.Datetime = pd.to_datetime(df.Datetime)
dts = df.groupby('eventid').agg({'Datetime': max}) - df.groupby('eventid').agg({'Datetime': min})
dts.Datetime = (dts.Datetime.astype('timedelta64[h]') // 6) + 1

for row in list(df.itertuples())[:]:

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

    lat_cntr = lats[-1]
    lon_cntr = lons[-1]
    lat_slice = slice(lat_cntr + 6.25, lat_cntr - 6.25)
    long_slice = slice(lon_cntr - 6.25, lon_cntr + 6.25)

    ufile = f"{prefix}/u/{year}/u_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
    vfile = f"{prefix}/v/{year}/v_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"

    uds = xr.open_dataset(ufile, chunks='auto')
    vds = xr.open_dataset(vfile, chunks='auto')

    try:
        arr = np.empty(6)
        arr[0] = uds.u.sel(time=timestamp, level=200, longitude=long_slice, latitude=lat_slice).compute().mean()
        arr[1] = uds.u.sel(time=timestamp, level=500, longitude=long_slice, latitude=lat_slice).compute().mean()
        arr[2] = uds.u.sel(time=timestamp, level=850, longitude=long_slice, latitude=lat_slice).compute().mean()

        arr[3] = vds.v.sel(time=timestamp, level=200, longitude=long_slice, latitude=lat_slice).compute().mean()
        arr[4] = vds.v.sel(time=timestamp, level=500, longitude=long_slice, latitude=lat_slice).compute().mean()
        arr[5] = vds.v.sel(time=timestamp, level=850, longitude=long_slice, latitude=lat_slice).compute().mean()

        vel = mat.dot(arr) + beta_drift
        u = vel[0]
        v = vel[1]

        dt = 6  # hours
        bearing = np.arctan(u / v) * 180 / np.pi
        distance = np.sqrt(u ** 2 + v ** 2) * dt

        origin = geopy.Point(lats[-1], lons[-1])
        destination = geodesic(kilometers=distance).destination(origin, bearing)
        lats.append(destination.latitude)
        lons.append(destination.longitude)

    except IndexError:
        lats.append(np.nan)
        lons.append(np.nan)


print(time.time() - t0, 's')

df = df.iloc[:len(lats)].copy()
lats = lats[:len(df)]
lons = lons[:len(df)]

df['lats_sim'] = np.array(lats)
df['lons_sim'] = np.array(lons)
df.to_csv(os.path.expanduser("~/coarse_tc_tracks.csv"))
