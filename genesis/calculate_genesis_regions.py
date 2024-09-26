"""
Calculate a TC genesis parameter based on combination of dynamic and
therodynamic variables

The TC genesis parameter is based on Tory et al. (2018), while we also
calculate the Z parameter defined in Hsieh et al (2020).

Data is from ERA5. Data are on a 1x1 degree grid. Wind fields and
relative humidity are assed through two iterations of a 9-point smoother (see
`metpy.calc.smooth_n_point` for the weighting of points).

The potential intensity has been precalculated from ERA5 based on Gilford
(2021).

Tory, K. J., H. Ye, and R. A. Dare, 2018: Understanding the geographic
distribution of tropical cyclone formation for applications in climate
models. Climate Dynamics, 50, 2489-2512,
https://doi.org/10.1007/s00382-017-3752-4.

Hsieh, T-L., Vecchi, G. A., Yang, W. Held, I. M., and Garner, S. T., 2020:
Large-scale control on the frequency of tropical cyclones and seeds: a
consistent relationship across a hierarchyof global atmospheric models.
Climate Dynamics, 55, 3177-3196, https://doi.org/10.1007/s00382-020-05446-5.


"""


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
from pathlib import Path
from git import Repo

d = Path().resolve().parent
sys.path.append(str(d))
import utils


LOGGER = logging.getLogger()
repo = Repo('', search_parent_directories=True)
COMMIT = str(repo.commit('HEAD'))

def filedatestr(year, month):
    """
    Return a string of the file date for a given year and month.

    """
    startdate = datetime.datetime(year, month, 1)
    enddate = datetime.datetime(year, month, monthrange(year, month)[1])

    return f"{startdate.strftime('%Y%m%d')}-{enddate.strftime('%Y%m%d')}"


def windfilelist(basepath):
    """Return a list of ERA5 files that contain wind data"""
    ufiles = glob.glob(f"{basepath}/pressure-levels/monthly-averaged/u/*/u_*.nc")
    vfiles = glob.glob(f"{basepath}/pressure-levels/monthly-averaged/v/*/v_*.nc")
    return [*ufiles, *vfiles]

def humidityfilelist(basepath):
    """Return a list of ERA5 files that contain humidity data"""
    rfiles = glob.glob(f"{basepath}/pressure-levels/monthly-averaged/r/*/r_*.nc")
    return rfiles

def mpifilelist(basepath):
    """
    Return a list of files that contain potential intensity data

    The PI data is diagnosed from ERA5 data, so is not stored in the same
    location as the other variables.

    """
    return [os.path.join(basepath, f"tcpi.1981-2023.nc")]

def calculateEta(ds, level):
    """
    Calculate absolute vorticity for a specified pressure level.

    :param ds: `xr.Dataset` that contains `u` and `v` wind components
    :param level: float value of the pressure value to use

    :returns: `metpy.units`-aware `xr.DataArray` of absolute vorticity.
    """
    uda = ds.sel(level=level)['u']
    vda = ds.sel(level=level)['v']
    avrt = mpcalc.absolute_vorticity(uda, vda)
    return avrt

def calculateXi(ds, level):
    """
    Calculate the xi term in equation 12 from Tory et al. 2018.

    :param ds: `xr.Dataset` that contains `u` and `v` wind components
    :param level: float value of the pressure value to use.

    NOTE: This will require wind data at 850 hPa as well. See the reference
    for details.

    :returns: `xr.DataArray` of normalised absolute vorticity parameter.
    """
    LOGGER.info("Calculating vorticity parameter")
    R = mpconst.earth_avg_radius
    omega = mpconst.earth_avg_angular_vel

    uda = ds.sel(level=level)['u']
    vda = ds.sel(level=level)['v']
    avrt = mpcalc.absolute_vorticity(uda, vda)
    dedy, _ = mpcalc.gradient(avrt, axes=['latitude', 'longitude'])
    dedy = dedy.metpy.dequantify()
    beta = xr.where(dedy < 5e-12, 5e-12, dedy)
    avrt850 = calculateEta(ds, level=850.)
    xi = np.abs(avrt850) / (beta * (R/(2*omega)))
    xi = utils.cyclic_wrapper(xi, "longitude")
    xi = mpcalc.smooth_n_point(xi, 9, 2)
    return xi.metpy.dequantify()

def calculateZ(ds, level):
    """
    Calculate the Z parameter in Hsieh et al. 2020.

    :param ds: `xr.Dataset` that contains `u` and `v` wind components
    :param level: float value of the pressure value to use.

    :returns: `xr.DataArray` of Z term
    """
    LOGGER.info("Calculating Z parameter")
    uda = ds.sel(level=level)['u']
    vda = ds.sel(level=level)['v']
    avrt = mpcalc.absolute_vorticity(uda, vda)
    dedy, _ = mpcalc.gradient(avrt, axes=['latitude', 'longitude'])
    U = 20.
    Z = np.abs(avrt) / np.sqrt(np.abs(dedy) * U)
    Z = utils.cyclic_wrapper(Z, "longitude")
    Z = mpcalc.smooth_n_point(Z, 9, 2)
    Z = Z.drop('level')
    return Z.metpy.dequantify()

def calculateShear(ds, upper, lower):
    """
    Calculate magnitude of vertical wind shear
    """
    LOGGER.info("Calculating wind shear")
    uu = ds.sel(level=upper)['u']
    ul = ds.sel(level=lower)['u']
    vu = ds.sel(level=upper)['v']
    vl = ds.sel(level=lower)['v']

    shear = np.sqrt((uu - ul)**2 + (vu - vl)**2)
    shear = utils.cyclic_wrapper(shear, "longitude")
    shear = mpcalc.smooth_n_point(shear, 9, 2)
    return shear.metpy.dequantify()

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

def calculateTCGP(vmax, xi, rh, shear):
    """
    Calculate TC genesis potential

    This is a first-pass effort, where we normalise all components by the
    thresholds reported in Tory et al. (2018), then mask areas below the
    threshold. The four components are multiplied to give the TCGP

    :param vmax: `xr.DataArray` of maximum potential intensity
    :param xi: `xr.DataArray` of normalised absolute vorticity
    :param rh: `xr.DataArray` of 700-hPa relative humidity
    :param shear: `xr.DataArray` of 200-850 hPa wind shear

    """
    LOGGER.info("Calculating TC genesis parameter")
    nu = xr.where(((vmax / 40) - 1) < 0, 0, (vmax / 40) - 1)
    mu = xr.where(((xi / 2e-5) - 1) < 0, 0, (xi / 2e-5) - 1)
    rho = xr.where(((rh / 40) - 1) < 0, 0, (rh / 40) - 1)
    sigma = xr.where((1 - (shear / 20)) < 0, 0, 1 - (shear / 20))

    tcgp = nu * mu * rho * sigma
    return tcgp.metpy.dequantify()


def calculateTCGPZ(vmax, Z, rh, shear):
    """
    Calculate TC genesis potential using teh vorticity ratio from Hsieh
    et al. (2020). Set the threshold to be 0.5.

    This is a first-pass effort, where we normalise all components by the
    thresholds reported in Tory et al. (2018), then mask areas below the
    threshold. The four components are multiplied to give the TCGP

    :param vmax: `xr.DataArray` of maximum potential intensity
    :param Z: `xr.DataArray` of absolute vorticity ratio
    :param rh: `xr.DataArray` of 700-hPa relative humidity
    :param shear: `xr.DataArray` of 200-850 hPa wind shear

    """
    LOGGER.info("Calculating TC genesis parameter")
    p = 1 / (1 + np.power(Z, (-1/0.15)))
    nu = xr.where(((vmax / 40) - 1) < 0, 0, (vmax / 40) - 1)
    mu = xr.where(((p / 0.5) - 1) < 0, 0, (p / 0.5) - 1)
    rho = xr.where(((rh / 40) - 1) < 0, 0, (rh / 40) - 1)
    sigma = xr.where((1 - (shear / 20)) < 0, 0, 1 - (shear / 20))

    tcgp = nu * mu * rho * sigma
    return tcgp.metpy.dequantify()


def calculateMeans(da, quantile=0.9):
    """
    Calculate monthly mean and quantiles of a DataArray

    :param da: `xr.DataArray` to calculate means and quantiles
    :param quantile: quantile level to calculate [0.0, 1.0]

    """
    LOGGER.info("Calculating monthly long term means and quantiles")
    mongrp = da.groupby('time.month')
    monmean = mongrp.mean(dim="time")
    monquant = mongrp.quantile(quantile, dim="time")
    monmean = monmean.assign_coords(month=("month", range(1, 13)))
    monquant = monquant.assign_coords(month=("month", range(1, 13)))
    return monmean, monquant

def saveTCGP(ds, filepath):
    """Save TCGP data to file"""
    logging.info(f"Saving data to {filepath}")
    try:
        ds.to_netcdf(filepath)
    except:
        logging.exception(f"Cannot save {filepath}")
        raise

def process(basepath, config):

    startYear = config.getint("Input", "StartYear")
    endYear = config.getint("Input", "EndYear")
    timeslice = slice(datetime.datetime(startYear, 1, 1),
                      datetime.datetime(endYear, 12, 31))
    windfiles = windfilelist(basepath)
    ds = xr.open_mfdataset(windfiles, chunks={"time":365*4}, parallel=True) # "longitude": 240, "latitude": 240}, parallel=True)
    ds = ds.sel(time=timeslice)
    ds = ds.isel(longitude=slice(0, 1440, 4), latitude=slice(0, 721, 4))
    ds = ds.roll(longitude=-180, roll_coords=True)
    ds["longitude"] = np.where(
        ds["longitude"] < 0, ds["longitude"] + 360, ds["longitude"]
    )

    RHlevel = config.getfloat("Levels", "RH")
    shear_upper_level = config.getfloat("Levels", "Upper")
    shear_lower_level = config.getfloat("Levels", "Lower")

    xi = calculateXi(ds, level=700.)
    Z = calculateZ(ds, level=850.)
    shear = calculateShear(ds, upper=shear_upper_level, lower=shear_lower_level)
    # Humidity
    rfiles = humidityfilelist(basepath)
    rds = xr.open_mfdataset(rfiles, chunks={"longitude": 240, "latitude": 240}, parallel=True)
    rds = rds.sel(time=timeslice)
    rh = calculateRH(rds, RHlevel)

    # Load potential intensity. This has been calculated separately,
    # so lives in a different directory
    mpipath = config.get("Input", "MPI")
    mpifiles = mpifilelist(mpipath)

    mpids = xr.open_mfdataset(mpifiles, chunks={"longitude": 240, "latitude": 240}, parallel=True)
    mpids = mpids.roll(longitude=-180, roll_coords=True)
    mpids["longitude"] = np.where(
        mpids["longitude"] < 0, mpids["longitude"] + 360, mpids["longitude"]
    )
    vmax = mpids['vmax']

    tcgp_xi = calculateTCGP(vmax, xi, rh, shear)
    tcgp_Z = calculateTCGPZ(vmax, Z, rh, shear)

    outds = xr.Dataset({
        'tcgp': tcgp_xi,
        'tcgpZ': tcgp_Z,
        'shear': shear,
        'rh': rh,
        'xi': xi,
        'vmax': vmax,
        'Z': Z
    }
    )

    # For downstream convenience, change the dates to be the middle of the month
    time = outds['time']
    time15 = pd.to_datetime(time.values).to_period("M").to_timestamp("D") + pd.Timedelta(days=14)
    outds = outds.assign_coords(time=time15)

    outds['tcgp'].attrs['standard_name'] = "TC genesis parameter (T2018)"
    outds['tcgp'].attrs['units'] = ''
    outds['tcgpZ'].attrs['standard_name'] = "TC genesis parameter (H2020)"
    outds['tcgpZ'].attrs['units'] = ''
    outds['shear'].attrs['standard_name'] = "wind shear"
    outds['shear'].attrs['upper_level'] = shear_upper_level
    outds['shear'].attrs['lower_level'] = shear_lower_level
    outds['shear'].attrs['units'] = 'm/s'
    outds['vmax'].attrs['standard_name'] = 'potential intensity'
    outds['vmax'].attrs['units'] = 'm/s'
    outds['rh'].attrs['standard_name'] = "relative humidity"
    outds['rh'].attrs['units'] = '%'
    outds['rh'].attrs['level'] = RHlevel
    outds['xi'].attrs['standard_name'] = "normalised vorticity"
    outds['xi'].attrs['units'] = 's-1'
    outds['Z'].attrs['standard_name'] = "vorticity ratio"
    outds['Z'].attrs['units'] = ''

    outds.attrs['title'] = "Tropical cyclone genesis parameter"
    outds.attrs['description'] = "TC genesis parameter and components"
    curdate = datetime.datetime.now()
    history = (f"{curdate:%Y-%m-%d %H:%M:%S}: {' '.join(sys.argv)}")
    outds.attrs['history'] = history
    outds.attrs['version'] = COMMIT

    outpath = config.get("Output", "Path")
    outfile = os.path.join(outpath, f"tcgp.{startYear:0d}-{endYear:0d}.nc")
    saveTCGP(outds, outfile)

    tcgpmean, tcgpquant = calculateMeans(tcgp_xi, 0.9)
    tcgpZmean, tcgpZquant = calculateMeans(tcgp_Z, 0.9)
    meands = xr.Dataset({
        'tcgpmean': tcgpmean,
        'tcgpquant': tcgpquant,
        'tcgpZmean': tcgpZmean,
        'tcgpZquant': tcgpZquant
    })
    meands['tcgpmean'].attrs['standard_name'] = "Monthly mean TC genesis parameter (T2018)"
    meands['tcgpmean'].attrs['units'] = ''
    meands['tcgpquant'].attrs['standard_name'] = "Quantile TC genesis parameter (T2018)"
    meands['tcgpquant'].attrs['units'] = ''
    meands['tcgpZmean'].attrs['standard_name'] = "Monthly mean TC genesis parameter (H2020)"
    meands['tcgpZmean'].attrs['units'] = ''
    meands['tcgpZquant'].attrs['standard_name'] = "Quantile TC genesis parameter (H2020)"
    meands['tcgpZquant'].attrs['units'] = ''

    meands.attrs['title'] = "LTM Tropical cyclone genesis parameter"
    meands.attrs['description'] = "TC genesis parameter monthly long term mean"
    curdate = datetime.datetime.now()
    history = (f"{curdate:%Y-%m-%d %H:%M:%S}: {' '.join(sys.argv)}")
    meands.attrs['history'] = history
    meands.attrs['version'] = COMMIT
    outfile = os.path.join(outpath, "tcgp.monltm.nc")
    saveTCGP(meands, outfile)

def main():
    """
    Handle command line arguments and call processing functions

    """
    p = argparse.ArgumentParser()

    p.add_argument('-c', '--config_file', help="Configuration file")
    p.add_argument('-v', '--verbose',
                   help="Verbose output",
                   action='store_true')

    args = p.parse_args()

    configFile = args.config_file
    config = ConfigParser()
    config.read(configFile)

    logFile = config.get('Logging', 'LogFile')
    logdir = dirname(realpath(logFile))

    # if log file directory does not exist, create it
    if not isdir(logdir):
        try:
            os.makedirs(logdir)
        except OSError:
            logFile = pjoin(os.getcwd(), 'pcmin.log')

    logLevel = config.get('Logging', 'LogLevel')
    verbose = config.getboolean('Logging', 'Verbose')
    datestamp = config.getboolean('Logging', 'Datestamp')
    if args.verbose:
        verbose = True

    if datestamp:
        base, ext = splitext(logFile)
        curdate = datetime.datetime.now()
        curdatestr = curdate.strftime('%Y%m%d%H%M')
        logfile = f"{base}.{curdatestr}.{ext.lstrip('.')}"

    logging.basicConfig(level=logLevel,
                        format="%(asctime)s: %(funcName)s: %(message)s",
                        filename=logfile, filemode='w',
                        datefmt="%Y-%m-%d %H:%M:%S")

    if verbose:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(getattr(logging, logLevel))
        formatter = logging.Formatter('%(asctime)s: %(funcName)s:  %(message)s',
                                      datefmt='%H:%M:%S', )
        console.setFormatter(formatter)
        LOGGER.addHandler(console)

    LOGGER.info(f"Started {sys.argv[0]} (pid {os.getpid()})")
    LOGGER.info(f"Log file: {logfile} (detail level {logLevel})")
    LOGGER.info(f"Code version: f{COMMIT}")

    basepath = config.get("Input", "Path")
    outpath = config.get("Output", "Path")

    if not os.path.exists(outpath):
        LOGGER.info(f"Making output path: {outpath}")
        os.makedirs(outpath)

    process(basepath, config)
    LOGGER.info("Completed")

if __name__ == "__main__":
    main()