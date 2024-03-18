import numpy as np
import scipy.signal as sps
import os
import sys
import glob
import pandas as pd
import pyproj
from lmfit import Model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.stats as stats
from scipy.interpolate import interp2d
sys.path.append("/scratch/w85/cxa547/python/lib/python3.10/site-packages")
from vincenty import vincenty
import dask.array as da
import xarray as xr
from calendar import monthrange
import geopy
import geopandas as gpd
from shapely.geometry import LineString, Point
import time
from geopy.distance import geodesic as gdg
from datetime import datetime
from metpy.calc import lat_lon_grid_deltas
from windspharm.xarray import VectorWind

from pde import CartesianGrid, solve_poisson_equation, ScalarField



# this script requires the results of 'era5_dlm.py', bom best track,
# and OCTR tracks are stored in the DATA_DIR folder

geodesic = pyproj.Geod(ellps='WGS84')
WIDTH = 6.25
DATA_DIR = "/g/data/w85/data/tc"
DLM_DIR = "/scratch/w85/cxa547/tcr/data/era5"
OUT_DIR = "/scratch/w85/cxa547/bam"
NOISE = False


def load_bom_df(season):
    """
    Helper function to load the BoM best track database
    """

    dataFile = os.path.join(DATA_DIR, "IDCKMSTM0S.csv")
    usecols = [0, 1, 2, 7, 8, 16, 49, 53]
    colnames = ['NAME', 'DISTURBANCE_ID', 'TM', 'LAT', 'LON',
                'CENTRAL_PRES', 'MAX_WIND_SPD', 'MAX_WIND_GUST']
    dtypes = [str, str, str, float, float, float, float, float]

    df = pd.read_csv(dataFile, skiprows=4, usecols=usecols, dtype=dict(zip(colnames, dtypes)), na_values=[' '])
    df['TM'] = pd.to_datetime(df.TM, format="%Y-%m-%d %H:%M", errors='coerce')
    df = df[~pd.isnull(df.TM)]
    df['season'] = pd.DatetimeIndex(df['TM']).year - (pd.DatetimeIndex(df['TM']).month < 6)
    df = df[df.season == season]
    df.reset_index(inplace=True)

    # distance =
    fwd_azimuth, _, distances = geodesic.inv(
        df.LON[:-1], df.LAT[:-1],
        df.LON[1:], df.LAT[1:],
    )

    df['new_index'] = np.arange(len(df))
    idxs = df.groupby(['DISTURBANCE_ID']).agg({'new_index': 'max'}).values.flatten()
    df.drop('new_index', axis=1, inplace=True)

    dt = np.diff(df.TM).astype(float) / 3_600_000_000_000
    u = np.zeros_like(df.LAT)
    v = np.zeros_like(df.LAT)
    v[:-1] = np.cos(fwd_azimuth * np.pi / 180) * distances / (dt * 1000) / 3.6
    u[:-1] = np.sin(fwd_azimuth * np.pi / 180) * distances / (dt * 1000) / 3.6

    v[idxs] = 0
    u[idxs] = 0
    df['u'] = u
    df['v'] = v

    dt = np.diff(df.TM).astype(float) / 3_600_000_000_000
    dt_ = np.zeros(len(df))
    dt_[:-1] = dt
    df['dt'] = dt_

    df = df[df.u != 0].copy()
    print(f"Number of records: {len(df)}")
    return df

def solvePoisson(scalar):
    """
    Solve the 2-d Poisson equation on a `ScalarField`, with defined
    boundary conditions
    """
    bcs = [{'value':0,}, {'value':0}]
    sol = solve_poisson_equation(scalar, bc=bcs)
    uu, vv = np.gradient(sol.data)
    return uu, vv


def extract_steering(uda, vda, cLon, cLat, dx, dy, width=4):
    """
    Calculate the steering flow for a region centred at `cLon`, `cLat`

    :param uda: `xr.DataArray` of u-component of wind
    :param vda: `xr.DataArray` of v-component of wind
    :param cLon: Longitude of TC location
    :param cLat: Latitude of TC location
    :param dx: zonal grid spacing
    :param dy: meridional grid spacing
    :param width: optional width of the box around the TC to calculate
                  environmental flow for.
    """
    # Take a copy of the datasets:
    uenv = uda.copy()
    venv = vda.copy()
    levels = uda.levels
    # Calculate virticity and divergence of the flow:
    w = VectorWind(uda, vda, legfunc='computed')
    vrt, div = w.vrtdiv()
    vrtz = xr.zeros_like(vrt)
    divz = xr.zeros_like(div)
    lat_slice = slice(cLat + width, cLat - width)
    lon_slice = slice(cLon - width, cLon + width)

    # Replace all areas *outside* the box with zeros:
    # This effectively sets the boundary conditions for the Laplace
    # equation to zero at the edge of the TC vortex
    vrtz.loc[:, lat_slice, lon_slice] = vrt.loc[:, lat_slice, lon_slice]
    divz.loc[:, lat_slice, lon_slice] = div.loc[:, lat_slice, lon_slice]

    # Create the scalar fields of vorticity and divergence for the `pde` library
    gr = CartesianGrid([[divz.latitude.min(), divz.latitude.max()],
                        [divz.longitude.min(), divz.longitude.max()]],
                       [len(divz.latitude), len(divz.longitude)])

    for lev in range(len(levels)):
        # Calculate the vector gradient of the solutions to the Laplace equation
        # (2-d version of the Poisson)
        fpsi = ScalarField(gr, data=vrtz[lev])
        fchi = ScalarField(gr, data=divz[lev])
        upsi, vpsi = solvePoisson(fpsi)
        uchi, vchi = solvePoisson(fchi)

        # Remove these local nondivergent and irrotational components from the
        # flow to give the environmental flow.
        uenv[lev, :, :] = (uda[0] - upsi*dx - uchi*dx)
        venv[lev, :, :] = (vda[0] - vpsi*dy - vchi*dy)

    return uenv, venv

def load_steering(uda, vda, df, width=4.0):

    griddx, griddy = lat_lon_grid_deltas(uda.longitude, uda.latitude)
    griddx = np.hstack((griddx.magnitude, griddx.magnitude[:, -1].reshape(-1, 1)))
    griddy = np.abs(np.vstack((griddy.magnitude, griddy.magnitude[-1, :].reshape(1, -1))))
    output = []
    for row in df.itertuples():
        dt = row.TM.to_numpy()
        print(f"Calculating environmental flow at {dt}")
        lat_cntr = 0.25 * np.round(row.LAT * 4)
        lon_cntr = 0.25 * np.round(row.LON * 4)
        lat_slice = slice(lat_cntr + width, lat_cntr - width)
        long_slice = slice(lon_cntr - width, lon_cntr + width)

        # Select data
        us = uda.sel(time=dt, method='nearest')
        vs = vda.sel(time=dt, method='nearest')
        uenv, venv = extract_steering(us, vs, lon_cntr, lat_cntr, griddx, griddy)

        u_steering = uenv.sel(latitude=lat_slice, longitude=long_slice).mean(axis=(1, 2)).values
        v_steering = venv.sel(latitude=lat_slice, longitude=long_slice).mean(axis=(1, 2)).values
        res = np.hstack((row.index, u_steering, v_steering))
        output.append(res)
    return output


if __name__ == "__main__":

    print("Loading ")
    season = int(sys.argv[1])
    df = load_bom_df(season)
    upath = sorted(glob.glob(os.path.join(DLM_DIR, f"**/era5_u_daily_*.nc")))
    vpath = sorted(glob.glob(os.path.join(DLM_DIR, f"**/era5_v_daily_*.nc")))

    udss = xr.open_mfdataset(upath, combine='nested', concat_dim='time',
                             chunks={'latitude': 721, 'longitude': 1440, 'time': -1},
                             parallel=True)
    vdss = xr.open_mfdataset(vpath, combine='nested', concat_dim='time',
                             chunks={'latitude': 721, 'longitude': 1440, 'time': -1},
                             parallel=True)

    uda = udss['u']
    vda = vdss['v']

    print("extract steering current from environmental flow:")
    results = load_steering(uda, vda, df)

    cols = ['index', 'u850', 'u250', 'v850', 'v250']
    vdf = pd.DataFrame(data=np.array([r for r in results]).squeeze(), columns=cols)
    vdf['index'] = vdf['index'].astype(int)
    outdf = df.merge(vdf, left_on='index', right_on='index', how='inner')
    outdf.to_csv(os.path.join(OUT_DIR, f"tcenvflow_serial.{season}.csv"))
    """

    run("BoM", "http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv")
    #run("OTCR", "http://www.bom.gov.au/cyclone/history/database/OTCR_alldata_final_external.csv")
    """