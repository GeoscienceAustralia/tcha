import numpy as np
import os
import xarray as xr
import pandas as pd
from calendar import monthrange
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD

prefix = "/g/data/rt52/era5/pressure-levels/reanalysis"

dataFile = os.path.expanduser("~/OTCR_alldata_final_external.csv")
source="http://www.bom.gov.au/cyclone/history/database/OTCR_alldata_final_external.csv"
usecols = [0, 1, 2, 7, 8, 11, 12]
colnames = ['NAME', 'DISTURBANCE_ID', 'TM', 'LAT', 'LON',
            'adj. ADT Vm (kn)', 'CP(CKZ(Lok R34,LokPOCI, adj. Vm),hPa)']
dtypes = [str, str, str, float, float, float, float]

df = pd.read_csv(dataFile, usecols=usecols, dtype=dict(zip(colnames, dtypes)), na_values=[' '], nrows=13743)
df['TM']= pd.to_datetime(df.TM, format="%d/%m/%Y %H:%M", errors='coerce')
df = df[~pd.isnull(df.TM)]
df['season'] = pd.DatetimeIndex(df['TM']).year - (pd.DatetimeIndex(df['TM']).month < 6)
df = df[df.season >= 1981]

dataFile = os.path.expanduser("~/IDCKMSTM0S.csv")
usecols = [0, 1, 2, 7, 8, 16, 49, 53]
colnames = ['NAME', 'DISTURBANCE_ID', 'TM', 'LAT', 'LON',
            'CENTRAL_PRES', 'MAX_WIND_SPD', 'MAX_WIND_GUST']
dtypes = [str, str, str, float, float, float, float, float]

bomdf = pd.read_csv(dataFile, skiprows=4, usecols=usecols, dtype=dict(zip(colnames, dtypes)), na_values=[' '])
bomdf['TM']= pd.to_datetime(bomdf.TM, format="%Y-%m-%d %H:%M", errors='coerce')
bomdf = bomdf[~pd.isnull(bomdf.TM)]
bomdf['season'] = pd.DatetimeIndex(bomdf['TM']).year - (pd.DatetimeIndex(bomdf['TM']).month < 6)
bomdf = bomdf[bomdf.season >= 1981]

times = np.sort(pd.unique(np.concatenate([df.TM.values, bomdf.TM.values])))[:]

t0 = time.time()

prev_eventid = None
prev_month = None

pressure = np.array(
    [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850]
)

years = np.sort(np.unique(pd.DatetimeIndex(times).year))
rank = comm.Get_rank()
rank_years = years[(years % 16) == rank]

for year in rank_years:
    print("Starting:", year)
    mask = pd.DatetimeIndex(times).year == year
    year_times = times[mask]
    uout = np.empty((len(year_times), 161, 361))
    vout = np.empty((len(year_times), 161, 361))
    for i, timestamp in enumerate(year_times):

        timestamp = pd.to_datetime(timestamp)
        month = timestamp.month
        year = timestamp.year
        days = monthrange(year, month)[1]

        lat_slice = slice(0, -40)
        long_slice = slice(80, 170)
        pslice = pslice = slice(300, 850)

        ufile = f"{prefix}/u/{year}/u_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
        vfile = f"{prefix}/v/{year}/v_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
        try:
            uds = xr.open_dataset(ufile, chunks='auto')
            uenv = uds.u.sel(time=timestamp, level=pslice, longitude=long_slice, latitude=lat_slice).compute()
            udlm = np.trapz(uenv.data, pressure, axis=0) / 550
            uout[i] = udlm

            vds = xr.open_dataset(vfile, chunks='auto')
            venv = vds.v.sel(time=timestamp, level=pslice, longitude=long_slice, latitude=lat_slice).compute()
            vdlm = np.trapz(venv.data, pressure, axis=0) / 550
            vout[i] = vdlm
        except KeyError as e:
            try:
                t0 = timestamp - pd.Timedelta(minutes=timestamp.minute)
                t1 = t0 + pd.Timedelta(hours=1)
                dt0 = (timestamp - t0).seconds
                dt1 = (t1 - timestamp).seconds

                uenv = uds.u.sel(time=t0, level=pslice, longitude=long_slice, latitude=lat_slice).compute()
                udlm_0 = np.trapz(uenv.data, pressure, axis=0) / 550
                uenv = uds.u.sel(time=t1, level=pslice, longitude=long_slice, latitude=lat_slice).compute()
                udlm_1 = np.trapz(uenv.data, pressure, axis=0) / 550

                uout[i] = udlm_0 * dt1 + udlm_1 * dt0
                uout[i] /= dt0 + dt1

                venv = vds.v.sel(time=t0, level=pslice, longitude=long_slice, latitude=lat_slice).compute()
                vdlm_0 = np.trapz(venv.data, pressure, axis=0) / 550
                venv = vds.v.sel(time=t1, level=pslice, longitude=long_slice, latitude=lat_slice).compute()
                vdlm_1 = np.trapz(venv.data, pressure, axis=0) / 550

                vout[i] = vdlm_0 * dt1 + vdlm_1 * dt0
                vout[i] /= dt0 + dt1
            except KeyError as e:
                print(e)

    uout = xr.DataArray(
        uout,
        dims=["time", "latitude", "longitude"],
        coords={"time": year_times, "latitude": uenv.coords["latitude"].data, "longitude": uenv.coords["longitude"].data},
    )
    uout.to_netcdf(os.path.expanduser(f"/scratch/w85/kr4383/era5dlm/u_dlm_{year}.netcdf"))

    vout = xr.DataArray(
        vout,
        dims=["time", "latitude", "longitude"],
        coords={"time": year_times, "latitude": venv.coords["latitude"].data, "longitude": venv.coords["longitude"].data},
    )
    vout.to_netcdf(os.path.expanduser(f"/scratch/w85/kr4383/era5dlm/v_dlm_{year}.netcdf"))
    print("Finished: ", year)