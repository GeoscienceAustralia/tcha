"""
envflow.py - calculate the environmental steering flow for TCs

Using ERA5 reanalysis of 850 and 250 hPa sub-daily (6 hours) winds,
calculate the environmental flow in the vicinity of each TC
observation.

The steering wind is calculated using "vortex surgery" to remove the
TC vortex from the reanalysis data to determine the background flow.
The details of this process are described in Galarneau and Davis (2013).
We use the same approach to extract the environmental wind at 850 hPa
and 250 hPa for each TC observation from ERA5 reanalysis (Hersbach et al.,
2020) for all storms in the Australian region.

The output is the TC track info, with additional fields representing the
eastward (`u`) and northward (`v`) components of the storm translation speed,
plus the same for the 850 and 250 hPa winds around the storm. Initially set
to +/- 4 degrees of the storm centre (as at March 2024).

We add a cyclic point at 180E to ensure there's no strange anomalies from a 
missing value at this longitude.

Dependencies:
pde - py-pde provides methods and classes for sovling partial differential
    equations. This is used to solve the inversion of the laplacian to
    calculate streamflow and velocity potential from the divergence and
    vorticity fields. https://py-pde.readthedocs.io/en/latest/

Author: Craig Arthur

NOTE: check all paths. The `pde` library is installed in a user scratch
directory; output paths are also in a user scratch directory.
"""


import os
import sys
import glob
import time
sys.path.append("/scratch/w85/cxa547/python/lib/python3.10/site-packages")

from mpi4py import MPI
from pde import CartesianGrid, solve_poisson_equation, ScalarField
from windspharm.xarray import VectorWind
from metpy.calc import lat_lon_grid_deltas
from datetime import datetime
import xarray as xr
import dask.array as da
import numpy as np

import cartopy.util as cutil

import pandas as pd
import pyproj

# This script requires the results of 'extract_era5.py'
# BoM best track and IBTrACS are stored in the DATA_DIR folder

geodesic = pyproj.Geod(ellps="WGS84")
WIDTH = 6.25
DATA_DIR = "/g/data/w85/data/tc"
DLM_DIR = "/scratch/w85/cxa547/tcr/data/era5"
OUT_DIR = "/scratch/w85/cxa547/envflow/cyclic"
BASINS = ['SP','SI']

def cyclic_wrapper(x, dim="longitude"):
    """
    Use cartopy.util.add_cyclic_point with an xarray Dataset to 
    add a cyclic or wrap-around pixel to the `lon` dimension. This can be useful
    for plotting with `cartopy`
    
    So add_cyclic_point() works on 'dim' via xarray Dataset.map()
    
    :param x: `xr.Dataset` to process
    :param str dim: Dimension of the dataset to wrap on (default "longitude")
    """
    wrap_data, wrap_lon = cutil.add_cyclic_point(
        x.values, 
        coord=x.coords[dim].data,
        axis=x.dims.index(dim)
    )
    return xr.DataArray(
        wrap_data, 
        coords={dim: wrap_lon, **x.drop_vars(dim).coords}, 
        dims=x.dims
    )

def load_ibtracs_df(season, basins=['SI', 'SP']):
    """
    Helper function to load the IBTrACS database.
    Column names are mapped to the same as the BoM dataset to minimise
    the changes elsewhere in the code

    :param int season: Season to filter data by
    :param list basins: select only those TCs from the given basins

    """
    dataFile = os.path.join(DATA_DIR, "ibtracs.since1980.list.v04r00.csv")
    df = pd.read_csv(
        dataFile,
        skiprows=[1],
        usecols=[0, 1, 3, 5, 6, 8, 9, 10, 11, 13, 23],
        keep_default_na=False,
        na_values=[" "],
        parse_dates=[1],
        date_format="%Y-%m-%d %H:%M:%S",
    )
    df.rename(
        columns={
            "SID": "DISTURBANCE_ID",
            "ISO_TIME": "TM",
            #"WMO_WIND": "MAX_WIND_SPD",
            "WMO_PRES": "CENTRAL_PRES",
            "USA_WIND": "MAX_WIND_SPD",
        },
        inplace=True,
    )

    df["TM"] = pd.to_datetime(
        df.TM, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df[~pd.isnull(df.TM)]

    # Filter to every 6 hours (to match sub-daily ERA data)
    df["hour"] = df["TM"].dt.hour
    df = df[df["hour"].isin([0, 6, 12, 18])]
    df.drop(columns=["hour"], inplace=True)
    df['SEASON'] = df['SEASON'].astype(int)
    df = df[df.SEASON == season]

    # IBTrACS includes spur tracks (bits of tracks that are
    # different to the official) - these need to be dropped.
    df = df[df.TRACK_TYPE == "main"]
    df = df[df["BASIN"].isin(basins)]

    df.reset_index(inplace=True)
    fwd_azimuth, _, distances = geodesic.inv(
        df.LON[:-1],
        df.LAT[:-1],
        df.LON[1:],
        df.LAT[1:],
    )

    df["new_index"] = np.arange(len(df))
    idxs = df.groupby(["DISTURBANCE_ID"]).agg(
        {"new_index": "max"}).values.flatten()
    df.drop("new_index", axis=1, inplace=True)
    # Convert max wind speed to m/s for consistency
    df["MAX_WIND_SPD"] = (df["MAX_WIND_SPD"] * 0.5144)

    dt = np.diff(df.TM).astype(float) / 3_600_000_000_000
    u = np.zeros_like(df.LAT)
    v = np.zeros_like(df.LAT)
    v[:-1] = np.cos(fwd_azimuth * np.pi / 180) * distances / (dt * 1000) / 3.6
    u[:-1] = np.sin(fwd_azimuth * np.pi / 180) * distances / (dt * 1000) / 3.6

    v[idxs] = 0
    u[idxs] = 0
    df["u"] = u
    df["v"] = v

    dt = np.diff(df.TM).astype(float) / 3_600_000_000_000
    dt_ = np.zeros(len(df))
    dt_[:-1] = dt
    df["dt"] = dt_

    df = df[df.u != 0].copy()
    print(f"Number of records: {len(df)}")
    return df


def load_bom_df(season):
    """
    Helper function to load the BoM best track database
    """

    dataFile = os.path.join(DATA_DIR, "IDCKMSTM0S.csv")
    usecols = [0, 1, 2, 7, 8, 16, 49, 53]
    colnames = [
        "NAME",
        "DISTURBANCE_ID",
        "TM",
        "LAT",
        "LON",
        "CENTRAL_PRES",
        "MAX_WIND_SPD",
        "MAX_WIND_GUST",
    ]
    dtypes = [str, str, str, float, float, float, float, float]

    df = pd.read_csv(
        dataFile,
        skiprows=4,
        usecols=usecols,
        dtype=dict(zip(colnames, dtypes)),
        na_values=[" "],
    )
    df["TM"] = pd.to_datetime(df.TM, format="%Y-%m-%d %H:%M", errors="coerce")
    df = df[~pd.isnull(df.TM)]
    df["season"] = pd.DatetimeIndex(df["TM"]).year - (
        pd.DatetimeIndex(df["TM"]).month < 6
    )
    df = df[df.season == season]
    df.reset_index(inplace=True)

    # distance =
    fwd_azimuth, _, distances = geodesic.inv(
        df.LON[:-1],
        df.LAT[:-1],
        df.LON[1:],
        df.LAT[1:],
    )

    df["new_index"] = np.arange(len(df))
    idxs = df.groupby(["DISTURBANCE_ID"]).agg(
        {"new_index": "max"}).values.flatten()
    df.drop("new_index", axis=1, inplace=True)

    dt = np.diff(df.TM).astype(float) / 3_600_000_000_000
    u = np.zeros_like(df.LAT)
    v = np.zeros_like(df.LAT)
    v[:-1] = np.cos(fwd_azimuth * np.pi / 180) * distances / (dt * 1000) / 3.6
    u[:-1] = np.sin(fwd_azimuth * np.pi / 180) * distances / (dt * 1000) / 3.6

    v[idxs] = 0
    u[idxs] = 0
    df["u"] = u
    df["v"] = v

    dt = np.diff(df.TM).astype(float) / 3_600_000_000_000
    dt_ = np.zeros(len(df))
    dt_[:-1] = dt
    df["dt"] = dt_

    df = df[df.u != 0].copy()
    print(f"Number of records: {len(df)}")
    return df


def solvePoisson(scalar):
    """
    Solve the 2-d Poisson equation on a `ScalarField`, with defined
    boundary conditions (0 at the edge of the domain).
    The scalar field is either the vorticity or divergence, and the
    returned values are either irrotational flow or non-divergent
    flow respectively
    """
    bcs = [{"value": 0},
           {"value": 0},]
    sol = solve_poisson_equation(scalar, bc=bcs)
    uu, vv = np.gradient(sol.data)
    return uu, vv


def extract_steering(uda, vda, cLon, cLat, dx, dy, width=4):
    """
    Calculate the steering flow for a region centred at `cLon`, `cLat`.

    This function performs the inversion of the streamfunction and
    velocity potential associated with the vortex (Lin et al, 2020) and
    returns the mean u- and v-components of the environmental wind.

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
    levels = uda.level
    # Calculate virticity and divergence of the flow:
    w = VectorWind(uda, vda, legfunc="computed")
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
    gr = CartesianGrid(
        [
            [divz.latitude.min(), divz.latitude.max()],
            [divz.longitude.min(), divz.longitude.max()],
        ],
        [len(divz.latitude), len(divz.longitude)],
    )

    for lev in range(len(levels)):
        # Calculate the vector gradient of the solutions to the Laplace equation
        # (2-d version of the Poisson)
        fpsi = ScalarField(gr, data=vrtz[lev])
        fchi = ScalarField(gr, data=divz[lev])
        upsi, vpsi = solvePoisson(fpsi)
        uchi, vchi = solvePoisson(fchi)

        # Remove these local nondivergent and irrotational components from the
        # flow to give the environmental flow.
        uenv[lev, :, :] = uda[lev] - upsi * dx - uchi * dx
        venv[lev, :, :] = vda[lev] - vpsi * dy - vchi * dy

    return uenv, venv


def load_steering(uda, vda, df, width=4.0):
    """
    Load the steering flow data for a given set of TC records

    :param uda: `xr.DataArray` of eastward component of wind
    :param vda: `xr.DataArray` of northward component of wind
    :param df: `pd.DataFrame` of TC observations. This must include at a
               minimum the time, latitude and longitude of the TC positions,
               named as 'TM', 'LAT', 'LON' respectively.
    :param width: Size of the box over which to calculate mean steering flow.
                  Default is 4 degrees. Loosely based on the 400 km used in
                  Lin et al. (2023) and Galarneau and Davis (2013).
    """
    griddx, griddy = lat_lon_grid_deltas(
        np.hstack((uda.longitude.values, 180)),
        uda.latitude.values
        )
    griddx = np.hstack(
        (griddx.magnitude, griddx.magnitude[:, -1].reshape(-1, 1)))
    griddy = np.abs(
        np.vstack(
            (griddy.magnitude, griddy.magnitude[-1, :].reshape(1, -1)))
    )
    output = []
    for row in df.itertuples():
        dt = row.TM.to_numpy()
        print(f"Calculating environmental flow at {dt}")
        lat_cntr = 0.25 * np.round(row.LAT * 4)
        lon_cntr = 0.25 * np.round(row.LON * 4)
        lat_slice = slice(lat_cntr + width, lat_cntr - width)
        long_slice = slice(lon_cntr - width, lon_cntr + width)

        # Select data
        us = uda.sel(time=dt, method="nearest")
        vs = vda.sel(time=dt, method="nearest")
        us = cyclic_wrapper(us, "longitude")
        vs = cyclic_wrapper(vs, "longitude")
        uenv, venv = extract_steering(
            us, vs, lon_cntr, lat_cntr, griddx, griddy, width)

        u_steering = (
            uenv.sel(latitude=lat_slice, longitude=long_slice).mean(
                axis=(1, 2)).values
        )
        v_steering = (
            venv.sel(latitude=lat_slice, longitude=long_slice).mean(
                axis=(1, 2)).values
        )
        res = np.hstack((row.index, u_steering, v_steering))
        output.append(res)
    return output


def process(season):
    """
    Run the calculation for a given season

    :param int season: TC season to calculate environmental flow for

    :returns: None. Data are saved to a file with the season in the filename.
    """
    df = load_ibtracs_df(season, BASINS)
    print("extract steering current from environmental flow:")
    results = load_steering(uda, vda, df)

    cols = ["index", "u850", "u250", "v850", "v250"]
    vdf = pd.DataFrame(data=np.array(
        [r for r in results]).squeeze(), columns=cols)
    vdf["index"] = vdf["index"].astype(int)
    outdf = df.merge(vdf, left_on="index", right_on="index", how="inner")
    print(f"Finished processing {season}")
    outdf.to_csv(os.path.join(OUT_DIR, f"tcenvflow_serial.{season}.csv"))


# Main code:

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
upath = sorted(glob.glob(os.path.join(DLM_DIR, f"**/era5_u_daily_*.nc")))
vpath = sorted(glob.glob(os.path.join(DLM_DIR, f"**/era5_v_daily_*.nc")))

print("Load the files as a multifile dataset")
udss = xr.open_mfdataset(
    upath,
    combine="nested",
    concat_dim="time",
    chunks={"latitude": 721, "longitude": 1440, "time": -1},
    parallel=True,
)
vdss = xr.open_mfdataset(
    vpath,
    combine="nested",
    concat_dim="time",
    chunks={"latitude": 721, "longitude": 1440, "time": -1},
    parallel=True,
)

# This shifts to a central longitude of 180E, rather than 0E
udss = udss.roll(longitude=-180, roll_coords=True)
udss['longitude'] = np.where(udss['longitude'] < 0, udss['longitude'] + 360, udss['longitude'])

vdss = vdss.roll(longitude=-180, roll_coords=True)
vdss['longitude'] = np.where(vdss['longitude'] < 0, vdss['longitude'] + 360, vdss['longitude'])


uda = udss["u"]
vda = vdss["v"]

# Scatter across available processors:
years = np.arange(1981, 2023)
rank_years = years[(years % comm.size) == rank]
for year in rank_years:
    process(year)
