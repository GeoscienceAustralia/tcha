import numpy as np
import os
import xarray as xr
import pandas as pd
from calendar import monthrange
import time
import geopy
from geopy.distance import geodesic
import dask
from dask.distributed import Client


if __name__ == '__main__':
    dask.config.set({'temporary_directory': '/scratch/w85/kr4383'})
    dask.config.set({'local-directory': '/scratch/w85/kr4383'})
    dask.config.set(shuffle='disk')
    dask.config.set(
        {'distributed.worker.memory.target': False,
         'distributed.worker.memory.spill': False,
         'distributed.worker.memory.pause': False,
         'distributed.worker.memory.terminate': False, }
    )
    # client = Client(set_as_default=True)
    # print(client)

    prefix = "/g/data/rt52/era5/pressure-levels/reanalysis"
    df = pd.read_csv(os.path.expanduser("~/jtwc_clean.csv"))

    dt = pd.Timedelta(1, units='hours')
    df.Datetime = pd.to_datetime(df.Datetime)

    t0 = time.time()

    ufiles = [
        f"{prefix}/u/{t.year}/u_era5_oper_pl_{t.year}{t.month:02d}01-{t.year}{t.month:02d}{monthrange(t.year, t.month)[1]}.nc"
        for t in df.Datetime
    ]
    vfiles = [
        f"{prefix}/v/{t.year}/v_era5_oper_pl_{t.year}{t.month:02d}01-{t.year}{t.month:02d}{monthrange(t.year, t.month)[1]}.nc"
        for t in df.Datetime
    ]

    uds = dict((ufile, xr.open_dataset(ufile, chunks='auto')) for ufile in set(ufiles))
    vds = dict((vfile, xr.open_dataset(vfile, chunks='auto')) for vfile in set(vfiles))

    out = []
    prev_eventid = None
    prev_month = None

    lats = []
    lons = []
    ids = set()

    for i, row in enumerate(list(df.itertuples())[:10]):

        if row.eventid not in ids:
            # using forward difference
            # discard last position of previous TC track
            ids.add(row.eventid)

        timestamp = row.Datetime

        # don't generate tracks
        # perform a one step prediction
        lat = row.Latitude
        lon = row.Longitude

        for _ in range(6):
            print("Start:", lat)

            ufile = f"{prefix}/u/{timestamp.year}/u_era5_oper_pl_{timestamp.year}{timestamp.month:02d}01-"
            ufile += f"{timestamp.year}{timestamp.month:02d}{monthrange(timestamp.year, timestamp.month)[1]}.nc"
            vfile = f"{prefix}/v/{timestamp.year}/v_era5_oper_pl_{timestamp.year}{timestamp.month:02d}01-"
            vfile += f"{timestamp.year}{timestamp.month:02d}{monthrange(timestamp.year, timestamp.month)[1]}.nc"

            uds = xr.open_dataset(ufile, chunks='auto')
            vds = xr.open_dataset(vfile, chunks='auto')

            lat_slice = slice(lat + 0.5, lat - 0.5)
            long_slice = slice(lon - 0.5, lon + 0.5)

            time_slice = slice(timestamp, timestamp + np.timedelta64(6, 'h'))

            try:
                uds_850 = uds.u.sel(time=timestamp, level=850, longitude=long_slice, latitude=lat_slice).compute()
                uds_250 = uds.u.sel(time=timestamp, level=250, longitude=long_slice, latitude=lat_slice).compute()
                vds_850 = vds.v.sel(time=timestamp, level=850, longitude=long_slice, latitude=lat_slice).compute()
                vds_250 = vds.v.sel(time=timestamp, level=250, longitude=long_slice, latitude=lat_slice).compute()

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
                print("End:", lat)

            except Exception as e:
                # print(e)
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
