import xarray as xr
from calendar import monthrange
import numpy as np
import pandas as pd
import os
import pyproj


geodesic = pyproj.Geod(ellps='WGS84')
DATA_DIR = os.path.expanduser("~/geoscience/data")


def load_otcr_df():
    """
    Helper function to load the OTCR database
    """
    dataFile = os.path.join(DATA_DIR, "OTCR_alldata_final_external.csv")
    usecols = [0, 1, 2, 7, 8, 11, 12]
    colnames = ['NAME', 'DISTURBANCE_ID', 'TM', 'LAT', 'LON',
                'adj. ADT Vm (kn)', 'CP(CKZ(Lok R34,LokPOCI, adj. Vm),hPa)']
    dtypes = [str, str, str, float, float, float, float]

    df = pd.read_csv(dataFile, usecols=usecols, dtype=dict(zip(colnames, dtypes)), na_values=[' '], nrows=13743)

    df['TM']= pd.to_datetime(df.TM, format="%d/%m/%Y %H:%M", errors='coerce')
    df = df[~pd.isnull(df.TM)]
    df['season'] = pd.DatetimeIndex(df['TM']).year - (pd.DatetimeIndex(df['TM']).month < 6)
    df = df[df.season >= 1981]
    df.reset_index(inplace=True)

    # calculate translation velocity
    fwd_azimuth, _, distances = geodesic.inv(
        df.LON[:-1], df.LAT[:-1],
        df.LON[1:], df.LAT[1:],
    )

    df['new_index'] = np.arange(len(df))
    idxs = df.groupby(['DISTURBANCE_ID']).agg({'new_index': np.max}).values.flatten()
    df.drop('new_index', axis=1, inplace=True)

    dt = np.diff(df.TM).astype(np.float) / 3_600_000_000_000
    vfm = np.zeros_like(df.LAT)
    vfm[:-1] = distances / (dt * 1000)
    vfm[idxs] = vfm[idxs - 1]
    df['vfm'] = vfm

    return df


def load_data(time, lat, lon):
    """
    Loads the pressure level u and v ERA5 files and calculate the DLM.

    This using dask to lazily load the minimum data needed.
    """
    year = time.month
    month = time.month
    days = monthrange(year, month)[1]
    lat_slice = slice(lat + 2.5, lat - 2.5)
    long_slice = slice(lon - 2.5, lon + 2.5)

    vars = ["sst", "sp", "2d", "2t"]
    prefix = "/g/data/rt52/era5/single-levels/reanalysis"
    out = []
    for var in vars:
        fp = f"{prefix}/{var}/{year}/{var}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{days}.nc"
        ds = xr.open_dataset(fp, chunks={'time': 24})
        y = ds.sel(time=time, longitude=long_slice, latitude=lat_slice).mean().compute(scheduler='single-threaded')
        out.append(y[var].data)

    return np.array(out)


if __name__ == "__main__":
    df = load_otcr_df()

    era5 = []

    for row in df.itertuples():
        era5.append(load_data(row.TM, row.LAT, row.LON))
