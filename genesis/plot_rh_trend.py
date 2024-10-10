import os
import sys
import glob
import logging
import argparse
import datetime
from calendar import monthrange
from configparser import ConfigParser
from os.path import join as pjoin, realpath, isdir, dirname, splitext
import dask
import numpy as np
import xarray as xr
import pandas as pd
import metpy.calc as mpcalc
import metpy.constants as mpconst
import matplotlib.pyplot as plt
from pathlib import Path
from git import Repo

d = Path().resolve().parent
sys.path.append(str(d))
import utils
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

DOMAINS = {
    "IO": {
        "name": "Indian Ocean",
        "lonmin": 110.,
        "lonmax": 130.,
        "latmax": -10.,
        "latmin": -20.,
        "color": "b"
    },
    "CS": {
        "name": "Coral Sea",
        "lonmin": 145.,
        "lonmax": 165.,
        "latmax": -10.,
        "latmin": -20.,
        "color": "r"
    },
    "SWP": {
        "name": "SW Pacific",
        "lonmin": 160.,
        "lonmax": 180.,
        "latmax": -5.,
        "latmin": -15.,
        "color": "g"
    }
}


LOGGER = logging.getLogger()
repo = Repo('', search_parent_directories=True)
COMMIT = str(repo.commit('HEAD'))

BASEPATH = "/g/data/rt52/era5"
OUTPATH = "/scratch/w85/cxa547/tcpi"

def humidityfilelist(basepath):
    """Return a list of ERA5 files that contain humidity data"""
    rfiles = glob.glob(f"{basepath}/pressure-levels/monthly-averaged/r/*/r_*.nc")
    return rfiles

def calculateRH(ds, level):
    """Calculate relative humidity"""
    LOGGER.info("Calculating relative humidity")
    rds = ds.isel(longitude=slice(0, 1440, 4), latitude=slice(0, 721, 4))
    rds = rds.roll(longitude=-180, roll_coords=True)
    rds["longitude"] = np.where(
        rds["longitude"] < 0, rds["longitude"] + 360, rds["longitude"]
    )
    rh = rds.sel(level=level)['r']
    rh = mpcalc.smooth_n_point(rh, 9, 2)
    return rh.metpy.dequantify()


def process():

    startYear = 1950
    endYear = 2020
    timeslice = slice(datetime.datetime(startYear, 1, 1),
                      datetime.datetime(endYear, 12, 31))
    # Humidity
    rfiles = humidityfilelist(BASEPATH)
    rds = xr.open_mfdataset(rfiles, chunks={"longitude": 240, "latitude": 240}, parallel=True)
    rds = rds.sel(time=timeslice)
    rh = calculateRH(rds, 700)
    LOGGER.info("Plotting trend of mid-level relative humidity")
    for d in DOMAINS.keys():
        lonslice = slice(DOMAINS[d]['lonmin'], DOMAINS[d]['lonmax'])
        latslice = slice(DOMAINS[d]['latmax'], DOMAINS[d]['latmin'])
        subregion = rh.sel(latitude=latslice, longitude=lonslice)

        # Resample to quarters starting in December
        dsres = subregion.resample(time='QS-DEC').mean(dim="time")

        # Group by the year of the January month
        djf_grouped = dsres.sel(time=dsres.time.dt.season=="DJF")

        # Calculate the mean over DJF months
        DOMAINS[d]['data'] = djf_grouped.mean(dim=['latitude', 'longitude'])

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for d in DOMAINS.keys():
        ax.plot(DOMAINS[d]['data']['time'].dt.year,
                DOMAINS[d]['data'],
                color=DOMAINS[d]['color'], label=d)
        ax.axhline(np.mean(DOMAINS[d]['data'].values),
                   color=DOMAINS[d]['color'],
                   ls="--", alpha=0.5)
    ax.legend()
    ax.grid()
    ax.set_ylabel(r"$RH_{700}$ [%]")
    utils.savefig(os.path.join(OUTPATH, "RH_trends.png"), bbox_inches='tight')
    utils.savefig(os.path.join(OUTPATH, "RH_trends.pdf"), bbox_inches='tight')

    LOGGER.info("Plot annual cycle of mid-level relative humidity")
    startYear = 1981
    endYear = 2020
    timeslice = slice(datetime.datetime(startYear, 1, 1),
                      datetime.datetime(endYear, 12, 31))
    rh = rh.sel(time=timeslice)
    for d in DOMAINS.keys():
        lonslice = slice(DOMAINS[d]['lonmin'], DOMAINS[d]['lonmax'])
        latslice = slice(DOMAINS[d]['latmax'], DOMAINS[d]['latmin'])
        subregion = rh.sel(latitude=latslice, longitude=lonslice).mean(dim=['latitude', 'longitude'])

        DOMAINS[d]['data'] = subregion.groupby(subregion.time.dt.month).mean(dim='time')
        DOMAINS[d]['q10'] = subregion.groupby(subregion.time.dt.month).quantile(0.1, dim='time')
        DOMAINS[d]['q90'] = subregion.groupby(subregion.time.dt.month).quantile(0.9, dim='time')

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for d in DOMAINS.keys():
        ax.plot(DOMAINS[d]['data']['month'],
                DOMAINS[d]['data'],
                color=DOMAINS[d]['color'], label=d)

        ax.fill_between(DOMAINS[d]['data']['month'],
                        DOMAINS[d]['q10'],
                        DOMAINS[d]['q90'],
                        color=DOMAINS[d]['color'],
                        alpha=0.25)
    ax.legend()
    ax.grid()
    ax.set_xlim((1, 12))
    ax.set_ylabel(r"$RH_{700}$ [%]")
    ax.set_title(rf"{startYear}-{endYear} monthly mean $RH_{{700}}$")
    utils.savefig(os.path.join(OUTPATH, "RH_annual_cycle.png"), bbox_inches='tight')
    utils.savefig(os.path.join(OUTPATH, "RH_annual_cycle.pdf"), bbox_inches='tight')

    LOGGER.info("Complete")


process()