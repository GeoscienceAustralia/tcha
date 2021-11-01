# Extract precipitation from ERA5 hourly data along a TC track
# On gadi.nci.org.au:
# $ module use /g/data/dk92/apps/Modules/modulefiles
# $ module avail NCI-data-analysis
# $ module load NCI-data-analysis_<version>
# $ export PYTHONPATH=$PYTHONPATH:$HOME/pylib/python

import os
from os.path import join as pjoin
import logging
from datetime import datetime
from calendar import monthrange
import xarray as xr
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from files import flStartLog
# This should be moved to config file or command-line args:
trackfile = "/g/data/w85/QFES_SWHA/hazard/input/ibtracs.since1980.list.v04r00.csv"
lsrrpath = "/g/data/rt52/era5/single-levels/reanalysis/lsrr"
crrpath = "/g/data/rt52/era5/single-levels/reanalysis/crr"
outputPath = "/scratch/w85/cxa547/precip"

precipcolorseq=['#FFFFFF', '#ceebfd', '#87CEFA', '#4969E1', '#228B22', 
                '#90EE90', '#FFDD66', '#FFCC00', '#FF9933', 
                '#FF6600', '#FF0000', '#B30000', '#73264d']
precipcmap = sns.blend_palette(precipcolorseq, len(precipcolorseq), as_cmap=True)
preciplevels = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0]
cbar_kwargs = {"shrink":0.9, 'ticks': preciplevels, 'label': 'Accumulated precipitation [kg m**-2]'}

erasource = "ECMWF Reanalysis version 5\nhttps://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels"
tcsource = "IBTrACS v4.0\nhttps://www.ncdc.noaa.gov/ibtracs/"

LOGGER = flStartLog("extract_precip.log", "INFO", verbose=True, datestamp=False)

def plot_precip(ds: xr.DataArray, track: pd.DataFrame, outputFile: str):

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.figure.set_size_inches(15, 12)
    try:
        ds.sum(dim='time').plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
                                 levels=preciplevels,
                                 extend='both',
                                 cmap=precipcmap,
                                 cbar_kwargs=cbar_kwargs)
    except TypeError:
        plt.close()
        return
    ax.plot(track['LON'], track['LAT'], color='k')
    ax.set_aspect('equal')
    startdate = pd.to_datetime(ds.time.values[0])
    enddate = pd.to_datetime(ds.time.values[-1])
    titlestr = f"{startdate:%Y-%m-%d %H:%M} - {enddate:%Y-%m-%d %H:%M}"
    ax.set_title(titlestr)
    ax.coastlines(resolution='10m')
    plt.text(-0.1, -0.05, f"Source: {erasource}",
             transform=ax.transAxes,
             fontsize='xx-small', ha='left',)
    plt.text(1.1, -0.05, f"TC data: {tcsource}",
             transform=ax.transAxes,
             fontsize='xx-small', ha='right',)
    plt.text(1.1, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=ax.transAxes,
             fontsize='xx-small', ha='right')

    gl = ax.gridlines(draw_labels=True, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    LOGGER.info(f"Saving plot to {outputFile}")
    plt.savefig(outputFile, bbox_inches='tight')
    plt.close()


def sel_ds(dataArray: xr.DataArray, time: datetime, lat: float, lon: float, margin: float = 5.):
    LOGGER.debug(f"Selecting slice of DataArray for time {time}")
    method = 'nearest'
    return dataArray.sel(
        time=time, method=method).sel(
        latitude=slice(lat + margin, lat - margin),
        longitude=slice(lon - margin, lon + margin)).load()

def extract(sid: str, track: pd.DataFrame, outputPath: str):
    """
    Extract two rainfall related parameters for the full range of dates covered
    by the track::
      lsrr - large scale rain rate
      crr - convective rain rate

    Combined, these provide the total rainfall rate (kg/m^2/s), which is then
    converted to a rainfall total by multiplying by 3600 seconds.

    NOTE::
    This function could readily be made more generic to extract any field, or a
    collection of fields, from ERA5 data. Since `xarray` can open multiple files
    using `xr.open_mfdataset`, one could feasibly create a list that spans times
    and variables, then open that

    :param str sid: Storm identifier
    :param track: `pd.DataFrame` containing details of the track.
    :param str outputPath: Folder to store the resulting data (in netcdf format)

    """

    lsrrfileset = set()
    crrfileset = set()

    for dt in track['ISO_TIME']:
        startdate = datetime(dt.year, dt.month, 1)
        enddate = datetime(dt.year, dt.month, monthrange(dt.year, dt.month)[1])
        datestr = f"{startdate.strftime('%Y%m%d')}-{enddate.strftime('%Y%m%d')}"
        LOGGER.debug(f"Datestring is: {datestr}")
        lsrrfileset.add(pjoin(lsrrpath, f"{dt.year}", f"lsrr_era5_oper_sfc_{datestr}.nc"))
        crrfileset.add(pjoin(crrpath, f"{dt.year}", f"crr_era5_oper_sfc_{datestr}.nc"))

    lsrrfiles = list(lsrrfileset)
    crrfiles = list(crrfileset)

    lsds = xr.open_mfdataset(lsrrfiles)[['lsrr']]
    crds = xr.open_mfdataset(crrfiles)[['crr']]

    ds_list = []
    LOGGER.debug("Iterating through each timestep in the track")
    for idx, row in track.iterrows():
        dscr = sel_ds(crds, row['ISO_TIME'], row['LAT'], row['LON'])
        dsls = sel_ds(lsds, row['ISO_TIME'], row['LAT'], row['LON'])
        ds_list.append((dsls.lsrr + dscr.crr) * 3600)

    LOGGER.debug(f"Concatenating {len(ds_list)} time steps")
    dsfootprint = xr.concat(ds_list, 'time').assign_coords(**{'time':track['ISO_TIME'].values})
    dsfootprint.assign_attrs(**{
            'long_name': 'Total hourly rainfall amount',
            'standard_name': 'precipitation_amount',
            'units': 'kg m**-2'
            })
    outds = dsfootprint.to_dataset(name='pr', promote_attrs=True)
    outds.pr.attrs['units'] = 'kg m**-2'
    outds.pr.attrs['long_name'] = 'Total hourly rainfall amount'
    outds.pr.attrs['standard_name'] = 'precipitation_amount'
    LOGGER.info(f"Saving data for {sid} to {outputPath}")
    outds.to_netcdf(pjoin(outputPath, f"{sid}_precip.nc"))
    plot_precip(outds.pr, track[['LON', 'LAT']], pjoin(outputPath, f"{sid}_precip.png"))

    return

# Read in the track data from file:
try:
    df = pd.read_csv(trackfile, skiprows=[1], header=0)
except:
    LOGGER.exception(f"Cannot load {trackfile}")
    raise

df = df.loc[df['BASIN'].isin(['SP', 'SI'])]
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
df.loc[df['LON'] < 0, 'LON'] += 360.

tracks = df.groupby('SID')

for sid, track in tracks:
    LOGGER.info(f"Extracting data for {sid}")
    dftc = track.set_index('ISO_TIME').resample('H').interpolate('linear').reset_index()
    extract(sid, track, outputPath)

"""
lsrrfileset = set()
crrfileset = set()

for dt in df_tc['ISO_TIME']:
    startdate = datetime(dt.year, dt.month, 1)
    enddate = datetime(dt.year, dt.month, monthrange(dt.year, dt.month)[1])
    datestr = f"{startdate.strftime('%Y%m%d')}-{enddate.strftime('%Y%m%d')}"
    lsrrfileset.add(pjoin(lsrrpath, f"{dt.year}", f"lsrr_era5_oper_sfc_{datestr}.nc"))
    crrfileset.add(pjoin(crrpath, f"{dt.year}", f"crr_era5_oper_sfc_{datestr}.nc"))

lsrrfiles = list(lsrrfileset)
crrfiles = list(crrfileset)

lsds = xr.open_mfdataset(lsrrfiles)[['lsrr']]
crds = xr.open_mfdataset(crrfiles)[['crr']]


ds_list = []
for i, (r, row) in enumerate(df_tc.iterrows()):
    dscr = sel_ds(crds, row['ISO_TIME'], row['LAT'], row['LON'])
    dsls = sel_ds(lsds, row['ISO_TIME'], row['LAT'], row['LON'])
    dsprcp = (dsls.lsrr + dscr.crr) * 3600
    ds_list.append(dsprcp)

dsfootprint = xr.concat(ds_list, 'time').assign_coords(**{'time':df_tc['ISO_TIME'].values})

x = dsfootprint['longitude'].values
y = dsfootprint['latitude'].values
plt.contour(x, y, dsfootprint.sum(axis=0),)

outds = dsfootprint.to_dataset(name='precip')
outds.to_netcdf(pjoin(outputpath, "yasi_precip.nc"))
"""
