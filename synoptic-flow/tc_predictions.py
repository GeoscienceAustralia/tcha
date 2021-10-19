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

    # don't generate tracks
    # perform a one step prediction
    lat = row.Latitude
    lon = row.Longitude

    for _ in range(6):

        month = timestamp.month
        year = timestamp.year
        days = monthrange(year, month)[1]

        ufile = f"{prefix}/u/{year}/u_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
        vfile = f"{prefix}/v/{year}/v_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"

        uds = xr.open_dataset(ufile, chunks='auto')
        vds = xr.open_dataset(vfile, chunks='auto')

        lat_slice = slice(lat + 0.5, lat - 0.5)
        long_slice = slice(lon - 0.5, lon + 0.5)

        uds_850 = uds.u.sel(time=timestamp, level=850, longitude=long_slice, latitude=lat_slice).compute()
        uds_250 = uds.u.sel(time=timestamp, level=250, longitude=long_slice, latitude=lat_slice).compute()

        vds_850 = vds.v.sel(time=timestamp, level=850, longitude=long_slice, latitude=lat_slice).compute()
        vds_250 = vds.v.sel(time=timestamp, level=250, longitude=long_slice, latitude=lat_slice).compute()

        try:
            uds_interp_850 = uds_850.interp(latitude=lat, longitude=lon)
            vds_interp_850 = vds_850.interp(latitude=lat, longitude=lon)

            uds_interp_250 = uds_250.interp(latitude=lat, longitude=lon)
            vds_interp_250 = vds_250.interp(latitude=lat, longitude=lon)

            u = -3.0575 + 0.4897 * uds_interp_850 + 0.6752 * uds_interp_250
            v = -5.1207 + 0.3257 * vds_interp_850 + 0.1502 * vds_interp_250

            dt = 1  # hours
            bearing = np.arctan(u / v) * 180 / np.pi
            distance = np.sqrt(u ** 2 + v ** 2) * dt

            origin = geopy.Point(lats[-1], lons[-1])
            destination = geodesic(kilometers=distance).destination(origin, bearing)
            lat = destination.latitude
            lon = destination.longitude

        except IndexError:
            lat = np.nan
            lon = np.nan

        timestamp += np.timedelta64(1, 'h')

    lats.append(lat)
    lons.append(lon)


print(time.time() - t0, 's')

df = df.iloc[:len(lats)].copy()
lats = lats[:len(df)]
lons = lons[:len(df)]

df['lats_sim'] = np.array(lats)
df['lons_sim'] = np.array(lons)
df.to_csv(os.path.expanduser("~/tc_predictions.csv"))