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
    """Return a list of files that contain potential intensity data"""
    return [os.path.join(basepath, f"tcpi.1981-2023.nc")]

def calculateEta(ds, level):
    """
    Calculate absolute vorticity for a
    specified pressure level.
    """
    uda = ds.sel(level=level)['u']
    vda = ds.sel(level=level)['v']
    avrt = mpcalc.absolute_vorticity(uda, vda)
    return avrt

def calculateXi(ds, level):
    """
    Calculate the xi term in equation 12 from Tory et al. 2018.

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
    ds = xr.open_mfdataset(windfiles, chunks={"longitude": 240, "latitude": 240}, parallel=True)
    ds = ds.sel(time=timeslice)
    ds = ds.isel(longitude=slice(0, 1440, 4), latitude=slice(0, 721, 4))
    ds = ds.roll(longitude=-180, roll_coords=True)
    ds["longitude"] = np.where(
        ds["longitude"] < 0, ds["longitude"] + 360, ds["longitude"]
    )

    xi = calculateXi(ds, level=700.)
    shear = calculateShear(ds, upper=200., lower=850.)
    
    # Humidity
    rfiles = humidityfilelist(basepath)
    rds = xr.open_mfdataset(rfiles, chunks={"longitude": 240, "latitude": 240}, parallel=True)
    rds = rds.sel(time=timeslice)
    rh = calculateRH(rds, 700)

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
    
    tcgp = calculateTCGP(vmax, xi, rh, shear)
    outds = xr.Dataset({
        'tcgp': tcgp,
        'shear': shear,
        'rh': rh,
        'xi': xi,
        'vmax': vmax
    }
    )
    
    outds['tcgp'].attrs['standard_name'] = "TC genesis parameter"
    outds['tcgp'].attrs['units'] = ''
    outds['shear'].attrs['standard_name'] = "200-850 hPa wind shear"
    outds['shear'].attrs['units'] = 'm/s'
    outds['vmax'].attrs['standard_name'] = 'potential intensity'
    outds['vmax'].attrs['units'] = 'm/s'
    outds['rh'].attrs['standard_name'] = "relative humidity"
    outds['rh'].attrs['units'] = '%'
    outds['xi'].attrs['standard_name'] = "normalised vorticity"
    outds['xi'].attrs['units'] = 's-1'
    
    outds.attrs['title'] = "Tropical cyclone genesis parameter"
    outds.attrs['description'] = "TC genesis parameter and components"
    curdate = datetime.datetime.now()
    history = (f"{curdate:%Y-%m-%d %H:%M:%S}: {' '.join(sys.argv)}")
    outds.attrs['history'] = history
    outds.attrs['version'] = COMMIT
    
    outpath = config.get("Output", "Path")
    outfile = os.path.join(outpath, "tcgp.1981-2023.nc")
    saveTCGP(outds, outfile)

    tcgpmean, tcgpquant = calculateMeans(tcgp, 0.9)
    meands = xr.Dataset({
        'tcgpmean': tcgpmean,
        'tcgpquant': tcgpquant
    })
    meands['tcgpmean'].attrs['standard_name'] = "Monthly mean TC genesis parameter"
    meands['tcgpmean'].attrs['units'] = ''
    meands['tcgpquant'].attrs['standard_name'] = "Quantile TC genesis parameter"
    meands['tcgpquant'].attrs['units'] = ''

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