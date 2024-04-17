"""
Analysis of climatological vorticity and divergence in the Southern Hemisphere

Requires pre-calculated file of winds at 850 & 250 hPa, including variance
and covariance. This is created by Lin's TC model code (ref)

Dependencies:
- xarray
- pandas
- pyproj
- cartopy
- seaborn
- windspharm

"""

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from calendar import month_name, month_abbr
import pyproj
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
import cartopy.util as cutil
import seaborn as sns

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from windspharm.xarray import VectorWind

DATA_DIR = "/g/data/w85/data/tc"
geodesic = pyproj.Geod(ellps="WGS84")
proj = ccrs.PlateCarree(central_longitude=180)
trans = ccrs.PlateCarree()

LONLOCATOR = MultipleLocator(30)
LATLOCATOR = MultipleLocator(10)

def savefig(filename, *args, **kwargs):
    """
    Add a timestamp to each figure when saving

    :param str filename: Path to store the figure at
    :param args: Additional arguments to pass to `plt.savefig`
    :param kwargs: Additional keyword arguments to pass to `plt.savefig`
    """
    fig = plt.gcf()
    plt.text(0.99, 0.01, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
            transform=fig.transFigure, ha='right', va='bottom',
            fontsize='xx-small')
    plt.savefig(filename, *args, **kwargs)

def load_ibtracs_df(season=None):
    """
    Helper function to load the IBTrACS database.
    Column names are mapped to the same as the BoM dataset to minimise
    the changes elsewhere in the code

    :param int season: (Optional) Season to filter data by

    NOTE: Only returns data for SP and SI basins.
    """
    dataFile = os.path.join(DATA_DIR, "ibtracs.since1980.list.v04r00.csv")
    df = pd.read_csv(
        dataFile,
        skiprows=[1],
        usecols=[0, 1, 3, 5, 6, 8, 9, 11, 13, 23],
        keep_default_na=False,
        na_values=[" "],
        parse_dates=[1],
        date_format="%Y-%m-%d %H:%M:%S",
    )
    df.rename(
        columns={
            "SID": "DISTURBANCE_ID",
            "ISO_TIME": "TM",
            "WMO_WIND": "MAX_WIND_SPD",
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
    # Filter on season, if provided
    if season:
        df = df[df.SEASON == season]

    # IBTrACS includes spur tracks (bits of tracks that are
    # different to the official) - these need to be dropped.
    df = df[df.TRACK_TYPE == "main"]
    # NOTE: Only using SP and SI basins here
    # df = df[df["BASIN"].isin(["SP", "SI"])]

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


def calcTClonPercentiles():
    """
    Calculate percentiles of longitude values in each basin
    
    :returns: `pd.DataFrame` of 10th and 90th percentile of longitude
    values in each TC basin.
    """
    tcdf = load_ibtracs_df()
    tcdf.loc[tcdf['LON']<0, 'LON'] = tcdf['LON'] + 360
    basins = tcdf.BASIN.unique()
    percentiles = pd.DataFrame(columns=['p10', 'p90'], index=basins)
    for basin in basins:
        p10 = tcdf.loc[tcdf.BASIN==basin, 'LON'].quantile(0.1)
        p90 = tcdf.loc[tcdf.BASIN==basin, 'LON'].quantile(0.9)
        percentiles.loc[basin] = [p10, p90]

    return percentiles

BASEDIR = "/scratch/w85/cxa547/tcr/data/era5"
windfile = os.path.join(BASEDIR, "env_wnd_era5_198101_202112.nc")
ds = xr.open_dataset(windfile)
month_length = ds.time.dt.days_in_month

# This shifts to a central longitude of 180E, rather than 0E
ds = ds.roll(lon=-180, roll_coords=True)
ds['lona'] = np.where(ds['lon'] < 0, ds['lon'] + 360, ds['lon'])
ds['lon'] = ds['lona']

# Calculate vorticity/divergence at 850 & 250 hPa
ua850 = ds['ua850_Mean']
va850 = ds['va850_Mean']
w850 = VectorWind(ua850, va850, legfunc="computed")
vrt850, div850 = w850.vrtdiv(truncation=17)
# Calculate gradient of vorticity:
vrt850x, vrt850y = w850.gradient(vrt850)

ua250 = ds['ua250_Mean']
va250 = ds['va250_Mean']
w250 = VectorWind(ua250, va250, legfunc="computed")
vrt250, div250 = w250.vrtdiv(truncation=17)
vrt250x, vrt250y = w250.gradient(vrt250)

# Enable plotting across the dateline:
vrt850c, clon, clat = cutil.add_cyclic(vrt850, vrt850.lon, vrt850.lat)
vrt250c, clon, clat = cutil.add_cyclic(vrt250, vrt250.lon, vrt250.lat)
div850c, clon, clat = cutil.add_cyclic(div850, div850.lon, div850.lat)
div250c, clon, clat = cutil.add_cyclic(div250, div250.lon, div250.lat)

vrt850yc, clon, clat = cutil.add_cyclic(vrt850y, vrt850y.lon, vrt850y.lat)
vrt250yc, clon, clat = cutil.add_cyclic(vrt250y, vrt250y.lon, vrt250y.lat)

# Mean vorticity between 5 & 25S:
vrtsh850 = vrt850.sel(lat=slice(-5, -25)).mean(dim='lat')
vrtsh250 = vrt250.sel(lat=slice(-5, -25)).mean(dim='lat')
vrtnh850 = vrt850.sel(lat=slice(25, 5)).mean(dim='lat')
vrtnh250 = vrt250.sel(lat=slice(25, 5)).mean(dim='lat')

# Mean divergence between 5 & 25S:
divsh850 = div850.sel(lat=slice(-5, -25)).mean(dim='lat')
divsh250 = div250.sel(lat=slice(-5, -25)).mean(dim='lat')
divnh850 = div850.sel(lat=slice(25, 5)).mean(dim='lat')
divnh250 = div250.sel(lat=slice(25, 5)).mean(dim='lat')

# Mean vorticity gradient between 5 & 25S:
vrtsh850y = vrt850y.sel(lat=slice(-5, -25)).mean(dim='lat')
vrtsh250y = vrt250y.sel(lat=slice(-5, -25)).mean(dim='lat')
vrtnh850y = vrt850y.sel(lat=slice(25, 5)).mean(dim='lat')
vrtnh250y = vrt250y.sel(lat=slice(25, 5)).mean(dim='lat')


# Hovmoller diagram:
levels = 10e-6 * np.arange(-2., 2.01, 0.1)
fig, ax = plt.subplots(1, 1, figsize=(10, 12))
cs = ax.contourf(vrtsh850.lon, vrtsh850.time, vrtsh850, levels=levels, extend='both', cmap='RdBu')
ax.set_xlim((60, 240))
ax.grid(linestyle='--', color='0.5')
plt.colorbar(cs, aspect=20, label=r"$\zeta_{850}$ [$s^{-1}$]")
savefig("SH_vorticity.850.monmean.png")

fig, ax = plt.subplots(1, 1, figsize=(10, 12))
cs = ax.contourf(vrtsh250.lon, vrtsh250.time, vrtsh250, levels=levels, extend='both', cmap='RdBu')
ax.set_xlim((60, 240))
ax.grid(linestyle='--', color='0.5')
plt.colorbar(cs, aspect=20, label=r"$\zeta_{250}$ [$s^{-1}$]")
savefig("SH_vorticity.250.monmean.png")

# Time series of mean vorticity in the main TC regions in each basin:
basin_percentiles = calcTClonPercentiles()
fig, ax = plt.subplots(2, 1, figsize=(12,8), sharex=True)
for basin in ['SI', 'SP']:
    lonslice = slice(basin_percentiles.loc[basin, 'p10'], basin_percentiles.loc[basin, 'p90'])
    l, = ax[0].plot(vrtsh250.time, vrtsh250.sel(lon=lonslice).mean(axis=1), label=basin,)
    ax[0].axhline(vrtsh250.sel(lon=lonslice).mean(), linestyle='--', alpha=0.5, color=l.get_color())
    ax[0].set_ylabel(r"$\zeta_{250}$ [$s^{-1}$]")
    l, = ax[1].plot(vrtsh850.time, vrtsh850.sel(lon=lonslice).mean(axis=1), label=basin,)
    ax[1].axhline(vrtsh850.sel(lon=lonslice).mean(), linestyle='--', alpha=0.5, color=l.get_color())
    ax[1].set_ylabel(r"$\zeta_{850}$ [$s^{-1}$]")
ax[0].text(0.05, 0.95, "250 hPa", transform=ax[0].transAxes,
            fontweight='bold', va='top')
ax[1].text(0.05, 0.95, "850 hPa", transform=ax[1].transAxes,
            fontweight='bold', va='top')
ax[0].legend()
ax[0].grid(True)
ax[1].grid(True)
fig.tight_layout()
savefig("SH_vorticity.timeseries.png")

# Time series of mean divergence in the main TC regions in each basin:
fig, ax = plt.subplots(2, 1, figsize=(12,8), sharex=True)
for basin in ['SI', 'SP']:
    lonslice = slice(basin_percentiles.loc[basin, 'p10'], basin_percentiles.loc[basin, 'p90'])
    l, = ax[0].plot(divsh250.time, divsh250.sel(lon=lonslice).mean(axis=1), label=basin,)
    ax[0].axhline(divsh250.sel(lon=lonslice).mean(), linestyle='--', alpha=0.5, color=l.get_color())
    ax[0].set_ylabel(r"$\delta_{250}$ [$s^{-1}$]")
    l, = ax[1].plot(divsh850.time, divsh850.sel(lon=lonslice).mean(axis=1), label=basin,)
    ax[1].axhline(divsh850.sel(lon=lonslice).mean(), linestyle='--', alpha=0.5, color=l.get_color())
    ax[1].set_ylabel(r"$\delta_{850}$ [$s^{-1}$]")
ax[0].text(0.05, 0.95, "250 hPa", transform=ax[0].transAxes,
            fontweight='bold', va='top')
ax[1].text(0.05, 0.95, "850 hPa", transform=ax[1].transAxes,
            fontweight='bold', va='top')
ax[0].legend()
ax[0].grid(True)
ax[1].grid(True)
fig.tight_layout()
savefig("SH_divergence.timeseries.png")

# Seasonal mean vorticity in each basin (850 and 250 hPa)
vrtsh_djf850 = vrtsh850.sel(time=vrtsh850.time.dt.season=="DJF")
vrtsh_seasonal850 = vrtsh_djf850.groupby(vrtsh_djf850.time.dt.year).mean("time")
vrtsh_djf250 = vrtsh250.sel(time=vrtsh250.time.dt.season=="DJF")
vrtsh_seasonal250 = vrtsh_djf250.groupby(vrtsh_djf250.time.dt.year).mean("time")

# For northern hemisphere, we use JJA
vrtnh_djf850 = vrtnh850.sel(time=vrtnh850.time.dt.season=="JJA")
vrtnh_seasonal850 = vrtnh_djf850.groupby(vrtnh_djf850.time.dt.year).mean("time")
vrtnh_djf250 = vrtnh250.sel(time=vrtnh250.time.dt.season=="JJA")
vrtnh_seasonal250 = vrtnh_djf250.groupby(vrtnh_djf250.time.dt.year).mean("time")

# Seasonal mean divergence in each basin (850 and 250 hPa)
divsh_djf850 = divsh850.sel(time=divsh850.time.dt.season=="DJF")
divsh_seasonal850 = divsh_djf850.groupby(divsh_djf850.time.dt.year).mean("time")
divsh_djf250 = divsh250.sel(time=divsh250.time.dt.season=="DJF")
divsh_seasonal250 = divsh_djf250.groupby(divsh_djf250.time.dt.year).mean("time")

# For northern hemisphere, we use JJA
divnh_djf850 = divnh850.sel(time=divnh850.time.dt.season=="JJA")
divnh_seasonal850 = divnh_djf850.groupby(divnh_djf850.time.dt.year).mean("time")
divnh_djf250 = divnh250.sel(time=divnh250.time.dt.season=="JJA")
divnh_seasonal250 = divnh_djf250.groupby(divnh_djf250.time.dt.year).mean("time")

# Seasonal mean vorticity gradient in each basin (850 and 250 hPa)
vrtysh_djf850 = vrtsh850y.sel(time=vrtsh850y.time.dt.season=="DJF")
vrtysh_seasonal850 = vrtysh_djf850.groupby(vrtysh_djf850.time.dt.year).mean("time")
vrtysh_djf250 = vrtsh250y.sel(time=vrtsh250y.time.dt.season=="DJF")
vrtysh_seasonal250 = vrtysh_djf250.groupby(vrtysh_djf250.time.dt.year).mean("time")

# For northern hemisphere, we use JJA
vrtynh_djf850 = vrtnh850y.sel(time=vrtnh850y.time.dt.season=="JJA")
vrtynh_seasonal850 = vrtynh_djf850.groupby(vrtynh_djf850.time.dt.year).mean("time")
vrtynh_djf250 = vrtnh250y.sel(time=vrtnh250y.time.dt.season=="JJA")
vrtynh_seasonal250 = vrtynh_djf250.groupby(vrtynh_djf250.time.dt.year).mean("time")


# Plot timeseries of seasonal vorticity:
fig, ax = plt.subplots(2, 1, figsize=(12,8))
ax[0].plot(vrtsh_seasonal250.year, vrtsh_seasonal250.sel(lon=slice(144.1, 200.4)).mean(axis=1), label='SP',)
ax[0].plot(vrtsh_seasonal250.year, vrtsh_seasonal250.sel(lon=slice(47.8, 120.5)).mean(axis=1), label='SI', linestyle='--')
ax[0].axhline(vrtsh_seasonal250.sel(lon=slice(144.1, 200.4)).mean(), linestyle='--', alpha=0.5)
ax[0].axhline(vrtsh_seasonal250.sel(lon=slice(47.8, 120.5)).mean(), linestyle='--', color='orange', alpha=0.5)
ax[0].set_ylabel(r"$\zeta_{250}$ [$s^{-1}$]")
ax[0].text(0.05, 0.95, "250 hPa", transform=ax[0].transAxes,
            fontweight='bold', va='top')
ax[0].grid(True)
ax[0].legend(loc=1)
ax[1].plot(vrtsh_seasonal850.year, vrtsh_seasonal850.sel(lon=slice(144.1, 200.4)).mean(axis=1), label='SPAC',)
ax[1].plot(vrtsh_seasonal850.year, vrtsh_seasonal850.sel(lon=slice(47.8, 120.5)).mean(axis=1), label='SIND', linestyle='--')
ax[1].axhline(vrtsh_seasonal850.sel(lon=slice(144.1, 200.4)).mean(), linestyle='--', alpha=0.5)
ax[1].axhline(vrtsh_seasonal850.sel(lon=slice(47.8, 120.5)).mean(), linestyle='--', color='orange', alpha=0.5)
ax[1].grid(True)
ax[1].set_ylabel(r"$\zeta_{850}$ [$s^{-1}$]")
ax[1].text(0.05, 0.95, "850 hPa", transform=ax[1].transAxes,
            fontweight='bold', va='top')
fig.suptitle("DJF Mean vorticity")
fig.tight_layout()
savefig("SH_vorticity.timeseries.DJF.png")

# Plot mean vorticity in each longitude band
fig, ax = plt.subplots(2, 1, figsize=(12,8))
for l in range(60, 200, 20):
    lon = slice(l-10, l+10)
    ax[0].plot(vrtsh_seasonal250.year, vrtsh_seasonal250.sel(lon=lon).mean(axis=1), label=l,)
    ax[0].set_ylabel(r"$\zeta_{250}$ [$s^{-1}$]")
    ax[0].text(0.05, 0.95, "250 hPa", transform=ax[0].transAxes,
                fontweight='bold', va='top')
    ax[0].grid(True)
    ax[0].legend(loc=1)
    ax[1].plot(vrtsh_seasonal850.year, vrtsh_seasonal850.sel(lon=lon).mean(axis=1), label=l)
    ax[1].grid(True)
    ax[1].legend(loc=1)

    ax[1].set_ylabel(r"$\zeta_{850}$ [$s^{-1}$]")
    ax[1].text(0.05, 0.95, "850 hPa", transform=ax[1].transAxes,
                fontweight='bold', va='top')
fig.suptitle("DJF Mean vorticity by longitude")
fig.tight_layout()
savefig("SH_vorticity.timeseries.DJF.longitude.png")

# Plot timeseries of seasonal divergence:
fig, ax = plt.subplots(2, 1, figsize=(12,8))
ax[0].plot(divsh_seasonal250.year, divsh_seasonal250.sel(lon=slice(144.1, 200.4)).mean(axis=1), label='SP',)
ax[0].plot(divsh_seasonal250.year, divsh_seasonal250.sel(lon=slice(47.8, 120.5)).mean(axis=1), label='SI', linestyle='--')
ax[0].axhline(divsh_seasonal250.sel(lon=slice(144.1, 200.4)).mean(), linestyle='--', alpha=0.5)
ax[0].axhline(divsh_seasonal250.sel(lon=slice(47.8, 120.5)).mean(), linestyle='--', color='orange', alpha=0.5)
ax[0].set_ylabel(r"$\delta_{250}$ [$s^{-1}$]")
ax[0].text(0.05, 0.95, "250 hPa", transform=ax[0].transAxes,
            fontweight='bold', va='top')
ax[0].grid(True)
ax[0].legend(loc=1)
ax[1].plot(divsh_seasonal850.year, divsh_seasonal850.sel(lon=slice(144.1, 200.4)).mean(axis=1), label='SPAC',)
ax[1].plot(divsh_seasonal850.year, divsh_seasonal850.sel(lon=slice(47.8, 120.5)).mean(axis=1), label='SIND', linestyle='--')
ax[1].axhline(divsh_seasonal850.sel(lon=slice(144.1, 200.4)).mean(), linestyle='--', alpha=0.5)
ax[1].axhline(divsh_seasonal850.sel(lon=slice(47.8, 120.5)).mean(), linestyle='--', color='orange', alpha=0.5)
ax[1].grid(True)
ax[1].set_ylabel(r"$\delta_{850}$ [$s^{-1}$]")
ax[1].text(0.05, 0.95, "850 hPa", transform=ax[1].transAxes,
            fontweight='bold', va='top')
fig.suptitle("DJF Mean divergence")
fig.tight_layout()
savefig("SH_divergence.timeseries.DJF.png")

fig, ax = plt.subplots(2, 1, figsize=(12,8))
for l in range(60, 200, 20):
    lon = slice(l-10, l+10)
    ax[0].plot(divsh_seasonal250.year, divsh_seasonal250.sel(lon=lon).mean(axis=1), label=l,)
    ax[0].set_ylabel(r"$\delta_{250}$ [$s^{-1}$]")
    ax[0].text(0.05, 0.95, "250 hPa", transform=ax[0].transAxes,
                fontweight='bold', va='top')
    ax[0].grid(True)
    ax[0].legend(loc=1)
    ax[1].plot(divsh_seasonal850.year, divsh_seasonal850.sel(lon=lon).mean(axis=1), label=l)
    ax[1].grid(True)
    ax[1].legend(loc=1)

    ax[1].set_ylabel(r"$\delta_{850}$ [$s^{-1}$]")
    ax[1].text(0.05, 0.95, "850 hPa", transform=ax[1].transAxes,
                fontweight='bold', va='top')
fig.suptitle("DJF Mean divergence by longitude")
fig.tight_layout()
savefig("SH_divergence.timeseries.DJF.longitude.png")

# Plot monthly mean (plus std dev.) of vorticity in 20 degree longitude
# bands for 850- and 250-hPa, in southern and northern hemisphere.
# Values for the northern hemisphere are inverted to match SH

# NOTE: NH values west of 100* should be ignored as this region is over land.

lons = np.arange(60, 240, 20)
y250sh = np.zeros(len(lons))
y250shstd = np.zeros(len(lons))
y850sh = np.zeros(len(lons))
y850shstd = np.zeros(len(lons))
y250nh = np.zeros(len(lons))
y250nhstd = np.zeros(len(lons))
y850nh = np.zeros(len(lons))
y850nhstd = np.zeros(len(lons))

for i, l in enumerate(lons):
    lon = slice(l-10, l+10)
    y250sh[i] = vrtsh_seasonal250.sel(lon=lon).mean(axis=(0, 1))
    y250shstd[i] = vrtsh_seasonal250.sel(lon=lon).std(axis=(0, 1))
    y850sh[i] = vrtsh_seasonal850.sel(lon=lon).mean(axis=(0, 1))
    y850shstd[i] = vrtsh_seasonal850.sel(lon=lon).std(axis=(0, 1))

    y250nh[i] = -1*vrtnh_seasonal250.sel(lon=lon).mean(axis=(0, 1))
    y250nhstd[i] = vrtnh_seasonal250.sel(lon=lon).std(axis=(0, 1))
    y850nh[i] =  -1*vrtnh_seasonal850.sel(lon=lon).mean(axis=(0, 1))
    y850nhstd[i] = vrtnh_seasonal850.sel(lon=lon).std(axis=(0, 1))


fig, ax = plt.subplots(2, 1, figsize=(12,8))

ax[0].errorbar(lons, y250sh, y250shstd, marker='o', color='r', label='SH')
ax[1].errorbar(lons, y850sh, y850shstd, marker='o', color='r')
ax[0].errorbar(lons+1, y250nh, y250nhstd, marker='o', color='b', label='NH')
ax[1].errorbar(lons+1, y850nh, y850nhstd, marker='o', color='b')
ax[0].legend()
ax[0].grid()
ax[1].grid()
ax[0].set_ylabel(r"$\zeta_{250}$ [$s^{-1}$]")
ax[0].text(0.05, 0.95, "250 hPa", transform=ax[0].transAxes,
            fontweight='bold', va='top')
ax[1].set_ylabel(r"$\zeta_{850}$ [$s^{-1}$]")
ax[1].text(0.05, 0.95, "850 hPa", transform=ax[1].transAxes,
            fontweight='bold', va='top')
fig.suptitle("Mean vorticity by longitude\n"
             r"5-25$^\circ$ - 1981-2022")
fig.tight_layout()
savefig("SH_vorticity.clim.DJF.longitude.png")

# Plot monthly mean (plus std dev.) of divergence in 20 degree longitude
# bands for 850- and 250-hPa, in southern and northern hemisphere.
# Values for the northern hemisphere are inverted to match SH

# NOTE: NH values west of 100* should be ignored as this region is over land.
lons = np.arange(60, 240, 20)
y250sh = np.zeros(len(lons))
y250shstd = np.zeros(len(lons))
y850sh = np.zeros(len(lons))
y850shstd = np.zeros(len(lons))
y250nh = np.zeros(len(lons))
y250nhstd = np.zeros(len(lons))
y850nh = np.zeros(len(lons))
y850nhstd = np.zeros(len(lons))

for i, l in enumerate(lons):
    lon = slice(l-10, l+10)
    y250sh[i] = divsh_seasonal250.sel(lon=lon).mean(axis=(0, 1))
    y250shstd[i] = divsh_seasonal250.sel(lon=lon).std(axis=(0, 1))
    y850sh[i] = divsh_seasonal850.sel(lon=lon).mean(axis=(0, 1))
    y850shstd[i] = divsh_seasonal850.sel(lon=lon).std(axis=(0, 1))

    y250nh[i] = -1*divnh_seasonal250.sel(lon=lon).mean(axis=(0, 1))
    y250nhstd[i] = divnh_seasonal250.sel(lon=lon).std(axis=(0, 1))
    y850nh[i] =  -1*divnh_seasonal850.sel(lon=lon).mean(axis=(0, 1))
    y850nhstd[i] = divnh_seasonal850.sel(lon=lon).std(axis=(0, 1))


fig, ax = plt.subplots(2, 1, figsize=(12,8))

ax[0].errorbar(lons, y250sh, y250shstd, marker='o', color='r', label='SH')
ax[1].errorbar(lons, y850sh, y850shstd, marker='o', color='r')
ax[0].errorbar(lons+1, y250nh, y250nhstd, marker='o', color='b', label='NH')
ax[1].errorbar(lons+1, y850nh, y850nhstd, marker='o', color='b')
ax[0].legend()
ax[0].grid()
ax[1].grid()
ax[0].set_ylabel(r"$\delta_{250}$ [$s^{-1}$]")
ax[0].text(0.05, 0.95, "250 hPa", transform=ax[0].transAxes,
            fontweight='bold', va='top')
ax[1].set_ylabel(r"$\delta_{850}$ [$s^{-1}$]")
ax[1].text(0.05, 0.95, "850 hPa", transform=ax[1].transAxes,
            fontweight='bold', va='top')
fig.suptitle("Mean divergence by longitude\n"
             r"5-25$^\circ$ - 1981-2022")
fig.tight_layout()
savefig("SH_divergence.clim.DJF.longitude.png")

# Plot monthly mean (plus std dev.) of vorticity gradient in 20 degree longitude
# bands for 850- and 250-hPa, in southern and northern hemisphere.
# Values for the northern hemisphere are inverted to match SH

# NOTE: NH values west of 100* should be ignored as this region is over land.
lons = np.arange(60, 240, 20)
y250sh = np.zeros(len(lons))
y250shstd = np.zeros(len(lons))
y850sh = np.zeros(len(lons))
y850shstd = np.zeros(len(lons))
y250nh = np.zeros(len(lons))
y250nhstd = np.zeros(len(lons))
y850nh = np.zeros(len(lons))
y850nhstd = np.zeros(len(lons))

for i, l in enumerate(lons):
    lon = slice(l-10, l+10)
    y250sh[i] = vrtysh_seasonal250.sel(lon=lon).mean(axis=(0, 1))
    y250shstd[i] = vrtysh_seasonal250.sel(lon=lon).std(axis=(0, 1))
    y850sh[i] = vrtysh_seasonal850.sel(lon=lon).mean(axis=(0, 1))
    y850shstd[i] = vrtysh_seasonal850.sel(lon=lon).std(axis=(0, 1))

    y250nh[i] = -1*vrtynh_seasonal250.sel(lon=lon).mean(axis=(0, 1))
    y250nhstd[i] = vrtynh_seasonal250.sel(lon=lon).std(axis=(0, 1))
    y850nh[i] =  -1*vrtynh_seasonal850.sel(lon=lon).mean(axis=(0, 1))
    y850nhstd[i] = vrtynh_seasonal850.sel(lon=lon).std(axis=(0, 1))


fig, ax = plt.subplots(2, 1, figsize=(12,8))

ax[0].errorbar(lons, y250sh, y250shstd, marker='o', color='r', label='SH')
ax[1].errorbar(lons, y850sh, y850shstd, marker='o', color='r')
ax[0].errorbar(lons+1, y250nh, y250nhstd, marker='o', color='b', label='NH')
ax[1].errorbar(lons+1, y850nh, y850nhstd, marker='o', color='b')
ax[0].legend()
ax[0].grid()
ax[1].grid()
ax[0].set_ylabel(r"$\nabla\zeta_{250}$ [$m^{-1}s^{-1}$]")
ax[0].text(0.05, 0.95, "250 hPa", transform=ax[0].transAxes,
            fontweight='bold', va='top')
ax[1].set_ylabel(r"$\nabla\zeta_{850}$ [$m^{-1}s^{-1}$]")
ax[1].text(0.05, 0.95, "850 hPa", transform=ax[1].transAxes,
            fontweight='bold', va='top')
fig.suptitle("Mean vorticity gradient by longitude\n"
             r"5-25$^\circ$ - 1981-2022")
fig.tight_layout()
savefig("SH_vortgrad.clim.DJF.longitude.png")

# Plot 850- and 250-hPa vorticity, coloured by month, for SI and SP basins
# This is the average in the main TC regions in each basin (80% of all storms in the
# given longitude range)
fig, ax = plt.subplots(2, 1, figsize=(8,8), sharey=True, sharex=True)
x1 = vrtsh850.sel(lon=slice(144.1, 200.4)).mean(axis=1)
y1 = vrtsh250.sel(lon=slice(144.1, 200.4)).mean(axis=1)
z1 = vrtsh850.time.dt.month.values
x2 = vrtsh850.sel(lon=slice(47.8, 120.5)).mean(axis=1)
y2 = vrtsh250.sel(lon=slice(47.8, 120.5)).mean(axis=1)
z2 = vrtsh850.time.dt.month.values

ax[0].scatter(x1, y1, ls='-', marker='o', c=z1, 
              cmap=sns.color_palette("hls", 12, as_cmap=True))
ax[0].grid()
ax[0].set_ylabel(r"$\zeta_{250}$ [$s^{-1}$]")
ax[0].text(0.05, 0.95, "SP", transform=ax[0].transAxes,
            fontweight='bold', va='top')

cs = ax[1].scatter(x2, y2, ls='-', marker='o', c=z2, 
                   cmap=sns.color_palette("hls", 12, as_cmap=True))
ax[1].grid()
ax[1].set_xlabel(r"$\zeta_{850}$ [$s^{-1}$]")
ax[1].set_ylabel(r"$\zeta_{250}$ [$s^{-1}$]")
ax[1].text(0.05, 0.95, "SI", transform=ax[1].transAxes,
            fontweight='bold', va='top')
handles, labels = cs.legend_elements()
labels = [month_abbr[i] for i in range(1, 13)]

plt.subplots_adjust(bottom=0.175,)
plt.legend(handles, labels, title="Month", bbox_to_anchor=(0.5, 0), 
           loc="lower center", ncols=6, bbox_transform=fig.transFigure)
savefig("SH_vorticity.850-250.png")


#########
"""
extents = (90, 180, -30, 0)
vrt850monmean, clon, clat = cutil.add_cyclic(vrt850.groupby("time.month").mean("time"), ua850.lon, ua850.lat)
vrt250monmean, clon, clat = cutil.add_cyclic(vrt250.groupby("time.month").mean("time"), ua250.lon, ua250.lat)

fig, axes = plt.subplots(4, 3, figsize=(12, 10),
                         sharex=True, sharey=True,
                         subplot_kw={
                             'projection': proj
                             })
for i, ax in enumerate(axes.flatten()):
    ax.contourf(clon, clat, vrt850monmean[i, :, :], levels=levels, extend='both', cmap='RdBu', transform=trans)
    ax.coastlines(color='0.5', linewidth=1.5)
    gl = ax.gridlines(draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style={'size': 'x-small'}
    gl.ylabel_style={'size': 'x-small'}
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent(extents, crs=trans)
fig.tight_layout()

extents = (90, 180, -30, 0)
levels = 10e-6 * np.arange(-2., 2.01, 0.1)
fig, axes = plt.subplots(4, 3, figsize=(12, 12), subplot_kw={'projection': proj})
for i, ax in enumerate(axes.flatten()):
    ax.contourf(clon, clat, vrt250monmean[i, :, :], levels=levels, extend='both', cmap='RdBu', transform=trans)
    ax.coastlines(color='0.5', linewidth=1.5)
    gl = ax.gridlines(draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style={'size': 'x-small'}
    gl.ylabel_style={'size': 'x-small'}
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent(extents, crs=trans)
fig.tight_layout()
"""
extents = (90, 220, -25, 25)
levels = 10e-12*np.arange(-3., 3.1, 0.2)
fig, axes = plt.subplots(4, 3, figsize=(16, 12), 
                         subplot_kw={'projection': proj},
                         sharex=True, sharey=True)
for i, ax in enumerate(axes.flatten()):
    cs = ax.contourf(clon, clat, vrt850yc[i, :, :], extend='both',
                     levels=levels, cmap='RdBu', transform=trans)
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = LONLOCATOR
    gl.ylocator = LATLOCATOR
    gl.xlabel_style={'size': 'x-small'}
    gl.ylabel_style={'size': 'x-small'}
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent(extents, crs=trans)
    ax.set_title(month_name[i+1])

fig.subplots_adjust(bottom=0.15, hspace=0.025)
cbarax = fig.add_axes([0.1, 0.1, 0.8, 0.025])
fig.suptitle(r"$\nabla \zeta_{850}$")
plt.colorbar(cs, cax=cbarax, orientation='horizontal',
             label=r"$\nabla \zeta_{850}$")

savefig("vortgrad.850_mean.png")

# Plot maps of low-level vorticity
levels = 10e-6 * np.arange(-2., 2.01, 0.1)
fig, axes = plt.subplots(4, 3, figsize=(16, 12), 
                         subplot_kw={'projection': proj},
                         sharex=True, sharey=True)
for i, ax in enumerate(axes.flatten()):
    cs = ax.contourf(clon, clat, vrt850c[i, :, :], extend='both',
                     levels=levels, cmap='RdBu', transform=trans)
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = LONLOCATOR
    gl.ylocator = LATLOCATOR
    gl.xlabel_style={'size': 'x-small'}
    gl.ylabel_style={'size': 'x-small'}
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent(extents, crs=trans)
    ax.set_title(month_name[i+1])

fig.subplots_adjust(bottom=0.15, hspace=0.025)
cbarax = fig.add_axes([0.1, 0.1, 0.8, 0.025])
fig.suptitle(r"$\zeta_{850}$")
plt.colorbar(cs, cax=cbarax, orientation='horizontal',
             label=r"$\zeta_{850}$")
savefig("vorticity.850_mean.png")


# Plot profiles of mean vorticity
# between 5 & 25, for each longitude band (10 degrees each)

vrtsh850 = vrt850.sel(lat=slice(-5, -25))
vrtsh250 = vrt250.sel(lat=slice(-5, -25))
vrtnh850 = vrt850.sel(lat=slice(25, 5))
vrtnh250 = vrt250.sel(lat=slice(25, 5))

vrtsh850y = vrt850y.sel(lat=slice(-5, -25))
vrtsh250y = vrt250y.sel(lat=slice(-5, -25))
vrtnh850y = vrt850y.sel(lat=slice(25, 5))
vrtnh250y = vrt250y.sel(lat=slice(25, 5))

lons = np.arange(60, 210, 10)
slats = vrtsh850.lat.values
nlats = vrtnh850.lat.values
vrt850ysh = vrt850y.sel(lat=slice(-5, -25))
vrt850ynh = vrt850y.sel(lat=slice(25, 5))

lons = np.arange(60, 240, 10)
slats = vrt850ysh.lat.values
nlats = vrt850ynh.lat.values
y250sh = np.zeros((len(lons), len(slats)))
y250shstd = np.zeros((len(lons), len(slats)))
y850sh = np.zeros((len(lons), len(slats)))
y850shstd = np.zeros((len(lons), len(slats)))
y250nh = np.zeros((len(lons), len(nlats)))
y250nhstd = np.zeros((len(lons), len(nlats)))
y850nh = np.zeros((len(lons), len(nlats)))
y850nhstd = np.zeros((len(lons), len(nlats)))

for i, l in enumerate(lons):
    lonslice = slice(l-5, l+5)
    y850sh[i, :] = vrt850ysh.sel(lon=lonslice, time=vrt850y.time.dt.season=="DJF").mean(dim=["lon", "time"])
    y850shstd[i, :] = vrt850ysh.sel(lon=lonslice, time=vrt850y.time.dt.season=="DJF").mean(dim="lon").std(dim="time")
    y850nh[i, :] = vrt850ysh.sel(lon=lonslice, time=vrt850y.time.dt.season=="JJA").mean(dim=["lon", "time"])
    y850nhstd[i, :] = vrt850ysh.sel(lon=lonslice, time=vrt850y.time.dt.season=="JJA").mean(dim="lon").std(dim="time")
    
# Plot mean meridional gradient of vorticity by latitude, for each longitude:
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for i, l in enumerate(lons):
    ax[0].fill_betweenx(nlats, l + 10e11*y850nh[i,:].T, x2=l, color='k', alpha=0.25)
    ax[1].fill_betweenx(slats, l + 10e11*y850sh[i,:].T, x2=l, color='k', alpha=0.25)

ax[0].hlines(y=5, xmin=basin_percentiles.loc['WP', 'p10'],
             xmax=basin_percentiles.loc['WP', 'p90'], 
             color='k', linewidth=5, 
             transform=ax[0].transData)
rect = plt.Rectangle((30, 5), 60, 20, facecolor="white", alpha=0.85, zorder=1)
ax[0].add_patch(rect)
ax[1].hlines(y=-5, xmin=basin_percentiles.loc['SI', 'p10'], 
             xmax=basin_percentiles.loc['SI', 'p90'], 
             color='r', linewidth=5, 
             transform=ax[1].transData)
ax[1].hlines(y=-5, xmin=basin_percentiles.loc['SP', 'p10'],
             xmax=basin_percentiles.loc['SP', 'p90'],
             color='b', linewidth=5,
             transform=ax[1].transData)

ax[0].text(0.05, 0.95, "NH (JJA)", transform=ax[0].transAxes,
           fontweight='bold', va='top')
ax[1].text(0.05, 0.95, "SH (DJF)", transform=ax[1].transAxes,
           fontweight='bold', va='top')

ax[0].hlines(y=5,
             xmin=basin_percentiles.loc['WP', 'p10'],
             xmax=basin_percentiles.loc['WP', 'p90'],
             linewidth=5, color='k',
             transform=ax[0].transData)

ax[1].hlines(y=-5,
             xmin=basin_percentiles.loc['SP', 'p10'],
             xmax=basin_percentiles.loc['SP', 'p90'],
             linewidth=5, color='r',
             transform=ax[1].transData)

ax[1].hlines(y=-5,
             xmin=basin_percentiles.loc['SI', 'p10'],
             xmax=basin_percentiles.loc['SI', 'p90'],
             linewidth=5, color='b',
             transform=ax[1].transData)

rect = plt.Rectangle((30, 5), 70, 20, facecolor='w', alpha=0.85, zorder=1)
ax[1].set_xticks(lons)
ax[0].add_patch(rect)
ax[0].grid(which='both', linestyle='--')
ax[1].grid(linestyle='--')
fig.suptitle(r"$\zeta_{850} \times 10^{-5}$ [s$^{-1}$]")
ax[1].set_xlim((50, 210))
fig.tight_layout()
savefig("vorticity.850.longitude.profile.png")

# Plot profiles of mean meridional vorticity gradient
# between 5 & 25, for each longitude band (10 degrees each)

lons = np.arange(60, 210, 10)
slats = vrtsh850.lat.values
nlats = vrtnh850.lat.values

y250sh = np.zeros((len(lons), len(slats)))
y250shstd = np.zeros((len(lons), len(slats)))
y850sh = np.zeros((len(lons), len(slats)))
y850shstd = np.zeros((len(lons), len(slats)))
y250nh = np.zeros((len(lons), len(nlats)))
y250nhstd = np.zeros((len(lons), len(nlats)))
y850nh = np.zeros((len(lons), len(nlats)))
y850nhstd = np.zeros((len(lons), len(nlats)))

for i, l in enumerate(lons):
    lon = slice(l-5, l+5)
    y250sh[i, :] = vrtsh250y.sel(lon=lon, time=vrtsh250y.time.dt.season=="DJF").mean(dim=["time", "lon"])
    y250shstd[i, :] = vrtsh250y.sel(lon=lon, time=vrtsh250y.time.dt.season=="DJF").mean(dim="lon").std(dim="time")
    y850sh[i, :] = vrtsh850y.sel(lon=lon, time=vrtsh850y.time.dt.season=="DJF").mean(dim=["time", "lon"])
    y850shstd[i, :] = vrtsh850y.sel(lon=lon, time=vrtsh850y.time.dt.season=="DJF").mean(dim="lon").std(dim="time")

    y250nh[i, :] = vrtnh250y.sel(lon=lon, time=vrtnh250y.time.dt.season=="JJA").mean(dim=["time", "lon"])
    y250nhstd[i, :] = vrtnh250y.sel(lon=lon, time=vrtnh250y.time.dt.season=="JJA").mean(dim="lon").std(dim="time")
    y850nh[i, :] = vrtnh850y.sel(lon=lon, time=vrtnh850y.time.dt.season=="JJA").mean(dim=["time", "lon"])
    y850nhstd[i, :] = vrtnh850y.sel(lon=lon, time=vrtnh850y.time.dt.season=="JJA").mean(dim="lon").std(dim="time")

fig, ax = plt.subplots(2, 1, figsize=(12,6), sharex=True)

for i, l in enumerate(lons):
    ax[0].fill_betweenx(nlats, l + 10e11*y850nh[i, :], x2=l, color='k', alpha=0.5)
    ax[1].fill_betweenx(slats, l + 10e11*y850sh[i, :], x2=l, color='k', alpha=0.5)

ax[0].text(0.05, 0.95, "NH (JJA)", transform=ax[0].transAxes,
            fontweight='bold', va='top')
ax[1].text(0.05, 0.95, "SH (DJF)", transform=ax[1].transAxes,
            fontweight='bold', va='top')

ax[0].hlines(y=5,
             xmin=basin_percentiles.loc['WP', 'p10'],
             xmax=basin_percentiles.loc['WP', 'p90'],
             linewidth=5, color='k',
             transform=ax[0].transData)

ax[1].hlines(y=-5,
             xmin=basin_percentiles.loc['SP', 'p10'],
             xmax=basin_percentiles.loc['SP', 'p90'],
             linewidth=5, color='r',
             transform=ax[1].transData)

ax[1].hlines(y=-5,
             xmin=basin_percentiles.loc['SI', 'p10'],
             xmax=basin_percentiles.loc['SI', 'p90'],
             linewidth=5, color='b',
             transform=ax[1].transData)

rect = plt.Rectangle((30, 5), 70, 20, facecolor='w', alpha=0.85, zorder=1)
ax[1].set_xticks(lons)
ax[0].add_patch(rect)
ax[0].grid(linestyle='--')
ax[1].grid(linestyle='--')
fig.suptitle(r"$\partial \zeta_{850}/\partial y \times 10^{-11}$ [m$^{-1}$s$^{-1}$]")
ax[1].set_xlim((50, 210))
fig.tight_layout()
savefig("vortgrad.850.longitude.profile.png")
