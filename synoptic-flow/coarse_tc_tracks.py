import numpy as np
import os
import xarray as xr
import pandas as pd
from calendar import monthrange
import time
import geopy
from geopy.distance import geodesic
from vincenty import vincenty


def delete_vortex(arr, mask):
    rows, cols = np.where(~mask)

    index_df = pd.DataFrame(np.column_stack(np.where(~mask)), columns=["rows", "cols"])

    minrow = index_df.groupby('cols').min().loc[cols].rows.values.flatten() - 1
    maxrow = index_df.groupby('cols').max().loc[cols].rows.values.flatten() + 1

    mincol = index_df.groupby('rows').min().loc[rows].cols.values.flatten() - 1
    maxcol = index_df.groupby('rows').max().loc[rows].cols.values.flatten() + 1

    dx2 = maxcol - cols
    dx1 = cols - mincol

    dy2 = maxrow - rows
    dy1 = rows - minrow
    # linearly interpolate in x direction and then y and average - works better than bilinear for large empty region
    interpolated = (dx1 * arr[rows, maxcol] + dx2 * arr[rows, mincol]) / (dx1 + dx2)
    interpolated += (dy1 * arr[maxrow, cols] + dy2 * arr[minrow, cols]) / (dy1 + dy2)
    return 0.5 * interpolated

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

pressure = np.array(
    [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850]
)

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

    lat_cntr = 0.25 * np.round(lats[-1] * 4)
    lon_cntr = 0.25 * np.round(lons[-1] * 4)
    lat_slice = slice(lat_cntr + 6.25, lat_cntr - 6.25)
    long_slice = slice(lon_cntr - 6.25, lon_cntr + 6.25)
    pslice = slice(300, 850)

    ufile = f"{prefix}/u/{year}/u_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
    vfile = f"{prefix}/v/{year}/v_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"

    uds = xr.open_dataset(ufile, chunks='auto')
    vds = xr.open_dataset(vfile, chunks='auto')

    try:
        # calculate the DML
        u_env = uds.u.sel(time=timestamp, level=pslice, longitude=long_slice, latitude=lat_slice).compute()
        v_env = vds.v.sel(time=timestamp, level=pslice, longitude=long_slice, latitude=lat_slice).compute()
        u_dlm = 3.6 * np.trapz(u_env.data * pressure[:, None, None], pressure, axis=0) / np.trapz(pressure, pressure)
        v_dlm = 3.6 * np.trapz(v_env.data * pressure[:, None, None], pressure, axis=0) / np.trapz(pressure, pressure)

        # find the vortex
        long_mesh, lat_mesh,  = np.meshgrid(u_env.coords['longitude'], u_env.coords['latitude'])
        dists = [
            geodesic((lats[-1], lons[-1]), (lat_mesh.flatten()[i], long_mesh.flatten()[i])).km
            for i in range(len(long_mesh.flatten()))
        ]
        dists = np.array(dists).reshape(long_mesh.shape)
        mask = dists > row.rMax

        # delete the vortex
        u_dlm = delete_vortex(u_dlm, mask)
        v_dlm = delete_vortex(v_dlm, mask)

        # calculate TC velocity and time step
        u = 0.95 * u_dlm.mean() - 3.987
        v = 0.81 * v_dlm - 1.66

        dt = 6 # hours
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
