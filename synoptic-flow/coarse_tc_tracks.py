import numpy as np
import os
import xarray as xr
import pandas as pd
from calendar import monthrange
import time
import geopy
from geopy.distance import geodesic


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
df = pd.read_csv(os.path.expanduser("~/geoscience/data/jtwc_clean.csv"))

dt = pd.Timedelta(1, units='hours')
df.Datetime = pd.to_datetime(df.Datetime)

t0 = time.time()

out = []
prev_eventid = None

lats = [None]
lons = [None]
ids = set()

uds = xr.open_dataset(os.path.expanduser("~/geoscience/data/u_dlm.netcdf"))
vds = xr.open_dataset(os.path.expanduser("~/geoscience/data/v_dlm.netcdf"))
var = list(uds.data_vars.keys())[0]

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

    try:
        # calculate the DML
        u_dlm = 3.6 * uds[var].sel(time=timestamp, longitude=long_slice, latitude=lat_slice)
        v_dlm = 3.6 * vds[var].sel(time=timestamp, longitude=long_slice, latitude=lat_slice)

        # find the vortex
        c = np.pi / 180
        long_mesh, lat_mesh = np.meshgrid(u_dlm.coords['longitude'], u_dlm.coords['latitude'])
        hav = np.sin(0.5 * c * (lats[-1] - lat_mesh)) ** 2
        hav += np.cos(c * lat_mesh) * np.cos(c * lats[-1]) * np.sin(0.5 * c * (lons[-1] - long_mesh)) ** 2
        dists = 2 * 6378.137 * np.arcsin(np.sqrt(hav))

        dists = np.array(dists).reshape(long_mesh.shape)
        mask = dists > row.rMax

        u_dlm = u_dlm.data
        v_dlm = v_dlm.data

        if len(u_dlm.shape) == 3:
            u_dlm = u_dlm[0]
            v_dlm = v_dlm[0]
        # delete the vortex
        u_dlm[~mask] = delete_vortex(u_dlm, mask)
        v_dlm[~mask] = delete_vortex(v_dlm, mask)

        # calculate TC velocity and time step
        u = 0.95 * u_dlm.mean() - 3.987
        v = 0.81 * v_dlm.mean() - 1.66

        dt = 6  # hours
        if (u >= 0) and (v >= 0):
            bearing = np.arctan(u / v) * 180 / np.pi
        elif (u >= 0) and (v < 0):
            bearing = 180 + np.arctan(u / v) * 180 / np.pi
        elif (u < 0) and (v < 0):
            bearing = 180 + np.arctan(u / v) * 180 / np.pi
        else:
            bearing = 360 + np.arctan(u / v) * 180 / np.pi
        distance = np.sqrt(u ** 2 + v ** 2) * dt

        origin = geopy.Point(lats[-1], lons[-1])
        destination = geodesic(kilometers=distance).destination(origin, bearing)
        lats.append(destination.latitude)
        lons.append(destination.longitude)

    except IndexError as e:
        print(e)
        lats.append(np.nan)
        lons.append(np.nan)

    except ValueError as e:
        print(e)
        lats.append(np.nan)
        lons.append(np.nan)

print(time.time() - t0, 's')

df = df.iloc[:len(lats)].copy()
lats = lats[:len(df)]
lons = lons[:len(df)]

df['lats_sim'] = np.array(lats)
df['lons_sim'] = np.array(lons)
df.to_csv(os.path.expanduser("~/geoscience/data/coarse_tc_tracks.csv"))
