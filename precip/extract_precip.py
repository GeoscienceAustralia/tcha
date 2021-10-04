# Extract precipitation from ERA5 hourly data along a TC track
# On gadi.nci.org.au:
# $ module use /g/data/dk92/apps/Modules/modulefiles
# $ module avail NCI-data-analysis
# $ module load NCI-data-analysis_<version>

import os
from os.path import join as pjoin
import logging
from datetime import datetime
from calendar import monthrange
import xarray as xr
import pandas as pd

trackfile = "/g/data/w85/QFES_SWHA/hazard/input/ibtracs.since1980.list.v04r00.csv"
lsrrpath = "/g/data/rt52/era5/single-levels/reanalysis/lsrr"
crrpath = "/g/data/rt52/era5/single-levels/reanalysis/crr"
outputpath = "/scratch/w85/cxa547/precip"


def sel_ds(dataArray, time, lat, lon, margin=3.):
    method = 'nearest'
    return dataArray.sel(
        time=time, method=method).sel(
        latitude=slice(lat + margin, lat - margin),
        longitude=slice(lon - margin, lon + margin)).load()

def extract(sid, track):
    # lsrr - large-scale rain rate [kg m**-2 s**-1]
    # crr - convective rain rate [kg m**-2 s**-1]
    # Will need to multiply by 3600 to get hourly rain rate.
    lsrrfileset = set()
    crrfileset = set()
    for dt in track['ISO_TIME']:
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
    for i, (r, row) in enumerate(track.iterrows()):
        dscr = sel_ds(crds, row['ISO_TIME'], row['LAT'], row['LON'])
        dsls = sel_ds(lsds, row['ISO_TIME'], row['LAT'], row['LON'])
        dsprcp = (dsls.lsrr + dscr.crr) * 3600
        ds_list.append(dsprcp)

    dsfootprint = xr.concat(ds_list, 'time').assign_coords(**{'time':track['ISO_TIME'].values})
    outds = dsfootprint.to_dataset(name='precip')
    outds.to_netcdf(pjoin(outputpath, f"{sid}_precip.nc"))
    return

# Read in the track data from file:
df = pd.read_csv(trackfile, skiprows=[1], header=0)
df = df.loc[df['BASIN'].isin(['SP', 'SI'])]
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
df.loc[df['lon'] < 0, 'lon'] += 360.

tracks = df.groupby('SID')

for sid, track in tracks:
    dftc = track.set_index('ISO_TIME').resample('H').interpolate('linear').reset_index()
    extract(sid, track)

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