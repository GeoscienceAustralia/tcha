import os
import sys
import glob
import logging
import argparse
import datetime
from calendar import monthrange
from configparser import ConfigParser
from os.path import join as pjoin, realpath, isdir, dirname, splitext

import metpy.calc as mpcalc
import metpy.constants as mpconst

import dask
import numpy as np
import xarray as xr
from git import Repo

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

def filelist(basepath, year):
    """
    Generate a list of files that contain the required variables for the
    given year. As we are working with monthly mean data, we can open
    all files for a given year using `xr.open_mfdataset`. This includes all
    variables, so we end up with a dataset that has the required variables
    available.
    """

    sstfiles = glob.glob(f"{basepath}/single-levels/monthly-averaged/sst/{year}/sst_*.nc")
    mslfiles = glob.glob(f"{basepath}/single-levels/monthly-averaged/msl/{year}/msl_*nc")
    tfiles = glob.glob(f"{basepath}/pressure-levels/monthly-averaged/t/{year}/t_*.nc")
    qfiles = glob.glob(f"{basepath}/pressure-levels/monthly-averaged/q/{year}/q_*.nc")
    ufiles = glob.glob(f"{basepath}/pressure-levels/monthly-averaged/u/{year}/u_*.nc")
    vfiles = glob.glob(f"{basepath}/pressure-levels/monthly-averaged/v/{year}/v_*.nc")

    return [*sstfiles, *mslfiles, *tfiles, *qfiles]

def windfilelist(basepath, year):
    """Return a list of ERA5 files that contain wind data"""
    ufiles = glob.glob(f"{basepath}/pressure-levels/monthly-averaged/u/{year}/u_*.nc")
    vfiles = glob.glob(f"{basepath}/pressure-levels/monthly-averaged/v/{year}/v_*.nc")
    return [*ufiles, *vfiles]

def calculateEta(ds, level):
    """
    Calculate absolute vorticity on a global grid and for a
    specified pressure level.
    """
    uda = ds.sel(level=level)['u']
    vda = ds.sel(level=level)['v']
    avrt = mpcalc.absolute_vorticity(uda, vda)
    return avrt.metpy.dequantify()

def calculateGradEta(ds, level):
    """
    Calculate meridional gradient of absolute vorticity
    """
    avrt = calculateEta(ds, level)
    dedy, _ = mpcalc.gradient(avrt, axes=['latitude', 'longitude'])
    return dedy.metpy.dequantify()

def calculateXi(ds, level):
    """
    Calculate the xi term in equation 12 from Tory et al. 2018.

    This needs to be calculated on a global grid, not a subset.
    See https://ajdawson.github.io/windspharm/latest/userguide/overview.html
    """
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
    return xi

def processYear(basepath, year, outpath, config):
    windfiles = windfilelist(basepath, year)
    ds = xr.open_mfdataset(windfiles)
    minLon = config.getfloat('Domain', 'MinLon')
    maxLon = config.getfloat('Domain', 'MaxLon')
    minLat = config.getfloat('Domain', 'MinLat')
    maxLat = config.getfloat('Domain', 'MaxLat')

    # Take a subset of the data based on the domain set in the config file:
    # NOTE: Order of latitude dimension is 90N - 90S, so maxLat *must* be
    # the northern edge of the domain.
    # The vertical level also needs to be reversed for this structure of the
    # ERA5 data (level=slice(None, None, -1))

    eta = calculateEta(ds, level=850.)
    beta = calculateGradEta(ds, level=700.)
    xi = calculateXi(ds, level=700.)

    outds = xr.merge([
        eta.rename('eta'),
        beta.rename('beta'),
        xi.rename('xi')],
        compat='override')


    outds.eta.attrs['standard_name'] ='850 hPa absolute vorticity'
    outds.eta.attrs['units'] = 's^-1'
    outds.beta.attrs['standard_name'] ='700 hPa meridonal gradient of absolute vorticity'
    outds.beta.attrs['units'] = 'm^-1 s^-1'
    outds.xi.attrs['standard_name'] = 'Ratio of absolute vorticity to gradient of absolute vorticity'
    outds.xi.attrs['units'] = 's^-1'
    outputfile = os.path.join(outpath,
                              f"abv.{year}.nc")

    LOGGER.info(f"Saving data to {outputfile}")

    description = ("Meridional gradient of 700 hPa absolute vorticity and "
                   "850 hPa absolute vorticity")
    curdate = datetime.datetime.now()
    history = (f"{curdate:%Y-%m-%d %H:%M:%s}: {' '.join(sys.argv)}")
    outds.attrs['title'] = f"700 hPa eta gradient and 850 eta for {year}"
    outds.attrs['description'] = description
    outds.attrs['history'] = history
    outds.attrs['version'] = COMMIT
    outds.to_netcdf(outputfile)

def main():
    """
    Handle command line arguments and call processing functions

    """
    p = argparse.ArgumentParser()

    p.add_argument('-c', '--config_file', help="Configuration file")
    p.add_argument('-v', '--verbose',
                   help="Verbose output",
                   action='store_true')
    p.add_argument('-y', '--year', help="Year to process (1979-2020)")

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

    if args.year:
        year = int(args.year)
    else:
        year = 2015

    minLon = config.getfloat('Domain', 'MinLon')
    maxLon = config.getfloat('Domain', 'MaxLon')
    minLat = config.getfloat('Domain', 'MinLat')
    maxLat = config.getfloat('Domain', 'MaxLat')

    LOGGER.info(f"Domain: {minLon}-{maxLon}, {minLat}-{maxLat}")

    basepath = config.get("Input", "Path")
    outpath = config.get("Output", "Path")

    if not os.path.exists(outpath):
        LOGGER.info(f"Making output path: {outpath}")
        os.makedirs(outpath)

    startYear = config.getint("Input", "StartYear")
    endYear = config.getint("Input", "EndYear")
    for year in range(startYear, endYear + 1):

        processYear(basepath, year, outpath, config)
    LOGGER.info("Completed")

if __name__=="__main__":
    main()