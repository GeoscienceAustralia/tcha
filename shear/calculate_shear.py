import os
import sys
from os.path import join as pjoin
import logging
import argparse
from time import sleep
from datetime import datetime
from calendar import monthrange
from itertools import product
from configparser import ConfigParser
import xarray as xr
from metpy.units import units
from metpy.calc import geopotential_to_height, bulk_shear, wind_speed
import numpy as np
 

from files import flStartLog, flProgramVersion
from parallel import attemptParallel, disableOnWorkers

global LOGGER
LOGGER = logging.getLogger()

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
    logLevel = config.get('Logging', 'LogLevel', fallback="INFO")
    verbose = config.getboolean('Logging', 'Verbose', fallback=False)
    datestamp = config.getboolean('Logging', 'Datestamp', fallback=False)


    if comm.size > 1 and comm.rank > 0:
        logFile += '-' + str(comm.rank)
        verbose = False
    LOGGER = flStartLog(logFile, logLevel, verbose, datestamp)

    if args.year:
        year = int(args.year)
    else:
        year = 2015

    minLon = config.getfloat('Domain', 'MinLon')
    maxLon = config.getfloat('Domain', 'MaxLon')
    minLat = config.getfloat('Domain', 'MinLat')
    maxLat = config.getfloat('Domain', 'MaxLat')
    domain = (minLon, minLat, maxLon, maxLat)
    LOGGER.info(f"Domain: {minLon}-{maxLon}, {minLat}-{maxLat}")

    inputPath = config.get('Input', 'Path')
    outputPath = config.get('Output', 'Path')

    for month in range(1, 13):
        LOGGER.info(f"Processing {year}-{month}")
        process(year, month, domain, inputPath, outputPath)

def process(year: int, month: int, domain: tuple, inputPath: str, outputPath: str):
    """
    :param year: Year to process
    :param month: Month to process
    :param domain: Spatial extent to process (minLon, minLat, maxLon, maxLat)
    :param inputPath: Base path to input files
    :param outputPath: Output destination

    """
    minlon, minlat, maxlon, maxlat = domain
    datestr = getDatestr(year, month)
    ufile = pjoin(inputPath, 'u', f"{year}", f"u_era5_oper_pl_{datestr}.nc")
    vfile = pjoin(inputPath, 'v', f"{year}", f"v_era5_oper_pl_{datestr}.nc")
    zfile = pjoin(inputPath, 'z', f"{year}", f"z_era5_oper_pl_{datestr}.nc")

    try:
        assert(os.path.isfile(ufile))
    except AssertionError:
        LOGGER.warning(f"Input file is missing: {ufile}")
        LOGGER.warning(f"Skipping month {month}")
        return False

    lonslice = slice(minlon, maxlon)
    latslice = slice(maxlat, minlat)
    LOGGER.debug("Opening input files")
    dsz = xr.open_dataset(zfile).sel(longitude=slice(minlon, maxlon), latitude=slice(maxlat, minlat))
    dsu = xr.open_dataset(ufile).sel(longitude=slice(minlon, maxlon), latitude=slice(maxlat, minlat))
    dsv = xr.open_dataset(vfile).sel(longitude=slice(minlon, maxlon), latitude=slice(maxlat, minlat))
    LOGGER.debug("Calculating height from geopotential")
    prs = dsz.level
    nt, nz, ny, nx = dsz.z.shape
    ngrid = nt * ny * nx
    outarray = np.zeros((nt, ny, nx)) * units.m / units.s

    times = dsz.time

    status = MPI.Status()
    work_tag = 0
    result_tag = 1
    if (comm.rank == 0) and (comm.size > 1):
        w = 0
        p = comm.size - 1
        for d in range(1, comm.size):
            if w < nt:
                LOGGER.debug(f"Sending time {w} to node {d}")
                comm.send(w, dest=d, tag=work_tag)
                w += 1
            else:
                comm.send(None, dest=d, tag=work_tag)
                p = w

        terminated = 0
        while(terminated < p):
            result, tdx = comm.recv(source=MPI.ANY_SOURCE, status=status, tag=MPI.ANY_TAG)
            LOGGER.debug(f"Received data back from {status.source}")
            outarray[tdx, :, :] = result * units.m / units.s
            d = status.source

            if w < nt:
                LOGGER.debug(f"Sending time {times[w].values} to node {d}")
                comm.send(w, dest=d, tag=status.tag)
                w += 1
            else:
                comm.send(None, dest=d, tag=status.tag)
                terminated += 1

    elif (comm.size > 1) and (comm.rank != 0):
        status = MPI.Status()
        W = None
        while(True):
            W = comm.recv(source=0, tag=work_tag, status=status)
            if W is None:
                # Received empty packet - no work required
                LOGGER.debug("No work to be done on this processor: {0}".format(comm.rank))
                break
            LOGGER.debug(f"Processing time {times[W].values} on node {comm.rank}")
            results = calculate(prs, dsu.u[W, :, :, :], dsv.v[W, :, :, :], dsz.z[W, :, :, :], 6000*units.m)
            comm.send((results, W), dest=0, tag=status.tag)
            
    elif (comm.size == 1) and (comm.rank == 0):
        for tdx in range(nt):
            LOGGER.debug(f"Processing time {times[tdx]}")
            breakpoint()
            outarray[tdx, :, :] = calculate(prs,
                                dsu.u[tdx, :, :, :],
                                dsv.v[tdx, :, :, :],
                                dsz.z[tdx, :, :, :],
                                6000*units.m)

    LOGGER.debug("Creating output data array")
    outda = xr.DataArray(
        data=outarray,
        dims = ('time', 'latitude', 'longitude'),
        coords=[dsz.coords['time'],
            dsz.coords['latitude'],
            dsz.coords['longitude']],
        name='s06',
        attrs={
            'long_name': '0-6 km wind shear',
            'standard_name': 'wind_speed_shear',
            'units': "m s**-1"}
        )
    if comm.rank == 0:
        sleep(5)
    comm.Barrier()
    saveOutput(outda, pjoin(outputPath, f"s06_era5_moda_pl_{datestr}.nc"), gattrs=dsz.attrs)

@disableOnWorkers
def saveOutput(da: xr.DataArray, outputPath: str, gattrs: dict):
    """
    Save output data to a netcdf file

    :param da: `xr.DataArray` containing the data
    :param outputPath: Destination directory for the output data
    :param gattrs: :class:`dict` of global attributes to append to the output file
    """
    LOGGER.info(f"Saving data to {outputPath}")
    ds = da.to_dataset()
    basepath = os.path.dirname(sys.argv[0])
    historymsg = f"{datetime.now()}: {sys.argv[0]} ({flProgramVersion(basepath)})"
    if 'history' in gattrs:
        historymsg += gattrs['history']

    LOGGER.debug("Updating history attribute")
    gattrs.update({'history': historymsg})
    ds.assign_attrs(**gattrs)
    ds.to_netcdf(outputPath)
    return

def calculate(prs, u, v, z, depth=6000.*units.m):
    """
    All values are passed to `metpy.calc.bulk_shear`

    :param prs: array of pressure levels
    :param u: array of u values on pressure levels
    :param v: array of v values on pressure levels
    :param z: array of geopotential values on pressure levels
    :param depth: depth of layer to calculate shear

    :returns: magnitude of the 0-6 km vertical wind shear

    NOTE: This returns a units-aware value from metpy.
    """
    LOGGER.debug("Calculating bulk shear")
    ny, nx = u.shape[1:]
    h = geopotential_to_height(z)
    shear = np.zeros((ny, nx)) * units.m / units.s
    for j, i in product(range(ny), range(nx)):
        uu, vv = bulk_shear(prs, u[:, j, i], v[:, j, i], h[:, j, i], depth)
        shear[j, i] = wind_speed(uu, vv)
        LOGGER.debug(f"Mean value: {shear.mean()}")
    return shear.m

def getDatestr(year: int, month: int) -> str:
    """
    Convert a year and month combination into a datestring that matches the
    file format used for ERA5 files

    e.g. for May 2020::

    :param int year: Year
    :param int month: Month

    :returns: Formatted string that contains start and end date for the given
    year and month

    >>> getDatestr(2020, 5)
    '20200501-20200531'

    >>> getDatestr(2020, 2)
    '20200201-20200229'
    """
    startdate = datetime(year, month, 1)
    enddate = datetime(year, month, monthrange(year, month)[1])
    filedatestr = f"{startdate.strftime('%Y%m%d')}-{enddate.strftime('%Y%m%d')}"
    LOGGER.debug(f"{year} {month} = {filedatestr}")
    return filedatestr

"""
basepath = "/g/data/rt52/era5/pressure-levels/monthly-averaged"
outputpath = "/scratch/w85/cxa547/s06"

dsz = xr.open_dataset(os.path.join(basepath, 'z', '2020', 'z_era5_moda_pl_20200101-20200131.nc')).sel(longitude=slice(110, 160), latitude=slice(-5, -45))
h = geopotential_to_height(dsz.z)
dsu = xr.open_dataset(os.path.join(basepath, 'u', '2020', 'u_era5_moda_pl_20200101-20200131.nc')).sel(longitude=slice(110, 160), latitude=slice(-5, -45))
dsv = xr.open_dataset(os.path.join(basepath, 'v', '2020', 'v_era5_moda_pl_20200101-20200131.nc')).sel(longitude=slice(110, 160), latitude=slice(-5, -45))
prs = dsz.level
nt, nz, ny, nx = h.shape
s06 = np.zeros((nt, ny, nx)) * units.m / units.s
for k, j ,i in product(range(nt), range(ny), range(nx)):
    uu, vv = bulk_shear(
                prs,
                dsu.u[k, :, j, i],
                dsv.v[k, :, j, i],
                h[k, :, j, i],
                depth=6000*units.m)
    s06[k, j, i] = np.sqrt(uu**2 + vv**2)

dims = ('time', 'latitude', 'longitude')
coords =[dsz.coords['time'],
         dsz.coords['latitude'],
         dsz.coords['longitude']
        ]
das06 = xr.DataArray(
            data=s06,
            coords=coords,
            dims=dims,
            name='s06',
            attrs={
                'long_name': '0-6 km wind shear',
                'standard_name': 'wind_speed_shear',
                'units': "m s**-1"}
                )
outputfile = pjoin(outputpath, "s06_era5_20200101-20200131.nc")
das06.to_netcdf(outputfile, )
"""

if __name__ == "__main__":
    from parallel import attemptParallel, disableOnWorkers
    global MPI, comm
    MPI = attemptParallel()
    comm = MPI.COMM_WORLD
    import atexit
    atexit.register(MPI.Finalize)
    main()
