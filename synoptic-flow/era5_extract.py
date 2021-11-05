import numpy as np
import os
import xarray as xr
import pandas as pd
from calendar import monthrange
import time
# import dask


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

times = np.sort(pd.unique(np.concatenate([df.TM.values, bomdf.TM.values])))[:100]
year_month = pd.DatetimeIndex(times).year * 100 + pd.DatetimeIndex(times).month
data = {'time': times, 'ym': year_month.values}
timedf = pd.DataFrame(data)

t0 = time.time()

prev_eventid = None
prev_month = None

pressure = np.array(
    [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850]
)


uout = np.empty((len(times), 161, 361))
vout = np.empty((len(times), 161, 361))

groups = timedf.groupby('ym').groups

for idxs in groups.values():
    year = timedf.iloc[idxs[0]][0].year
    month = timedf.iloc[idxs[0]][0].month
    month_times = timedf.iloc[idxs].time.values
    out_idxs = np.where(timedf.time.isin(month_times).values)[0]

    days = monthrange(year, month)[1]

    lat_slice = slice(0, -40)
    long_slice = slice(80, 170)
    pslice = pslice = slice(300, 850)

    ufile = f"{prefix}/u/{year}/u_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
    vfile = f"{prefix}/v/{year}/v_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"

    uds = xr.open_dataset(ufile, chunks='auto')
    uenv = uds.u.sel(time=month_times, level=pslice, longitude=long_slice, latitude=lat_slice).compute()
    udlm = np.trapz(uenv.data * pressure[:, None, None], pressure, axis=0) / np.trapz(pressure, pressure)
    uout[out_idxs] = udlm

    vds = xr.open_dataset(vfile, chunks='auto')
    venv = vds.v.sel(time=month_times, level=pslice, longitude=long_slice, latitude=lat_slice).compute()
    vdlm = np.trapz(venv.data * pressure[:, None, None], pressure, axis=0) / np.trapz(pressure, pressure)
    vout[out_idxs] = vdlm
#
# for i, timestamp in enumerate(times):
#     if i % 10 == 0:
#         print(f"Processed {i} out of {len(times)}")
#     timestamp = pd.to_datetime(timestamp)
#     month = timestamp.month
#     year = timestamp.year
#     days = monthrange(year, month)[1]
#
#     lat_slice = slice(0, -40)
#     long_slice = slice(80, 170)
#     pslice = pslice = slice(300, 850)
#
#     ufile = f"{prefix}/u/{year}/u_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
#     vfile = f"{prefix}/v/{year}/v_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{days}.nc"
#
#     uds = xr.open_dataset(ufile, chunks='auto')
#     uenv = uds.u.sel(time=timestamp, level=pslice, longitude=long_slice, latitude=lat_slice).compute()
#     udlm = np.trapz(uenv.data * pressure[:, None, None], pressure, axis=0) / np.trapz(pressure, pressure)
#     uout[i] = udlm
#
#     vds = xr.open_dataset(vfile, chunks='auto')
#     venv = vds.v.sel(time=timestamp, level=pslice, longitude=long_slice, latitude=lat_slice).compute()
#     vdlm = np.trapz(venv.data * pressure[:, None, None], pressure, axis=0) / np.trapz(pressure, pressure)
#     vout[i] = vdlm


uout = xr.DataArray(
    uout,
    dims=["time", "latitude", "longitude"],
    coords={"time": times, "latitude": uenv.coords["latitude"].data, "longitude": uenv.coords["longitude"].data},
)
uout.to_netcdf(os.path.expanduser("~/u_dlm.netcdf"))

vout = xr.DataArray(
    vout,
    dims=["time", "latitude", "longitude"],
    coords={"time": times, "latitude": venv.coords["latitude"].data, "longitude": venv.coords["longitude"].data},
)
vout.to_netcdf(os.path.expanduser("~/v_dlm.netcdf"))
