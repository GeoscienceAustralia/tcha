import os
import sys
import numpy as np
import glob
import pyproj
import pandas as pd
import xarray as xr

sys.path.append("/scratch/w85/cxa547/python/lib/python3.10/site-packages")
from windspharm.xarray import VectorWind
import dask.array as da

from pde import CartesianGrid, solve_poisson_equation, ScalarField
from metpy.calc import lat_lon_grid_deltas

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import tcmarkers
from datetime import datetime
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import imageio.v2 as imageio

geodesic = pyproj.Geod(ellps='WGS84')

DATA_DIR = "/g/data/w85/data/tc"
DLM_DIR = "/scratch/w85/cxa547/tcr/data/era5"
OUT_DIR = "/scratch/w85/cxa547/envflow/tcmax"

def load_bom_df():
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
    df = df[df.season >= 1981]
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
    v[:-1] = np.cos(fwd_azimuth * np.pi / 180) * distances / (dt * 1000)
    u[:-1] = np.sin(fwd_azimuth * np.pi / 180) * distances / (dt * 1000)

    v[idxs] = 0
    u[idxs] = 0
    df['u'] = u
    df['v'] = v

    dt = np.diff(df.TM).astype(float) / 3_600_000_000_000
    dt_ = np.zeros(len(df))
    dt_[:-1] = dt
    df['dt'] = dt_

    df = df[df.u != 0].copy()

    return df


def extract_steering(uda, vda, cLon, cLat, dx, dy, width=4):
    """
    Calculate the steering flow for a region centred at `cLon`, `cLat`
    """
    w = VectorWind(uda, vda, legfunc='computed')
    vrt, div = w.vrtdiv()
    vrtz = xr.zeros_like(vrt)
    divz = xr.zeros_like(div)
    lat_slice = slice(cLat + width, cLat - width)
    lon_slice = slice(cLon - width, cLon + width)
    vrtz.loc[:, lat_slice, lon_slice] = vrt.loc[:, lat_slice, lon_slice]
    divz.loc[:, lat_slice, lon_slice] = div.loc[:, lat_slice, lon_slice]

    gr = CartesianGrid([[divz.latitude.min(), divz.latitude.max()],
                        [divz.longitude.min(), divz.longitude.max()]],
                       [len(divz.latitude), len(divz.longitude)])
    fpsi850 = ScalarField(gr, data=vrtz[0])
    fchi850 = ScalarField(gr, data=divz[0])
    fpsi250 = ScalarField(gr, data=vrtz[1])
    fchi250 = ScalarField(gr, data=divz[1])

    upsi850, vpsi850 = np.gradient(solve_poisson_equation(fpsi850, bc=[{'value':0,}, {'value':0}]).data,)
    uchi850, vchi850 = np.gradient(solve_poisson_equation(fchi850, bc=[{'value':0,}, {'value':0}]).data,)
    upsi250, vpsi250 = np.gradient(solve_poisson_equation(fpsi250, bc=[{'value':0,}, {'value':0}]).data,)
    uchi250, vchi250 = np.gradient(solve_poisson_equation(fchi250, bc=[{'value':0,}, {'value':0}]).data,)

    uenv = uda.copy()
    venv = vda.copy()

    uenv[0, :, :] = (uda[0] - upsi850*dx - uchi850*dx)
    venv[0, :, :] = (vda[0] - vpsi850*dy - vchi850*dy)
    uenv[1, :, :] = (uda[1] - upsi250*dx - uchi250*dx)
    venv[1, :, :] = (vda[1] - vpsi250*dy - vchi250*dy)
    return uenv, venv

def plot_dlm_flow(uda, vda, df, width=4):
    griddx, griddy = lat_lon_grid_deltas(uda.longitude, uda.latitude)
    griddx = np.hstack((griddx.magnitude, griddx.magnitude[:, -1].reshape(-1, 1)))
    griddy = np.abs(np.vstack((griddy.magnitude, griddy.magnitude[-1, :].reshape(1, -1))))
    for row in df.itertuples():
        dt = row.TM.to_numpy()
        lat_cntr = 0.25 * np.round(row.LAT * 4)
        lon_cntr = 0.25 * np.round(row.LON * 4)

        lat_slice = slice(lat_cntr + width, lat_cntr - width)
        long_slice = slice(lon_cntr - width, lon_cntr + width)

        # Select data
        us = uda.sel(time=dt, method='nearest')
        vs = vda.sel(time=dt, method='nearest')
        uenv, venv = extract_steering(us, vs, lon_cntr, lat_cntr, griddx, griddy)

        fig, ax = plt.subplots(1, 2, subplot_kw={'projection':ccrs.PlateCarree()},
                               figsize=(6, 8), sharex=True, sharey=True)
        levels = np.arange(0, 40.1, 2)
        ax[0].contourf(us.longitude, us.latitude, np.sqrt(us**2+vs**2), levels=levels, extend="max", cmap="viridis_r")
        ax[0].barbs(us.longitude, us.latitude, us, vs, flip_barb=True, length=5)
        ax[0].set_title("Full DLM flow [m/s]")
        ax[1].contourf(uenv.longitude, uenv.latitude, np.sqrt(uenv**2+venv**2), levels=levels, extend="max", cmap="viridis_r")
        ax[1].barbs(uenv.longitude, uenv.latitude, uenv, venv, flip_barb=True, length=5)
        ax[1].add_patch(mpatches.Rectangle(xy=[lon_cntr - width, lat_cntr - width],
                           width=2*width, height=2*width,
                           facecolor='none', edgecolor='r',
                           linewidth=2,
                           transform=ccrs.PlateCarree()))
        ax[1].set_title("Environmental DLM flow [m/s]")
        for axes in ax.flatten():
            axes.plot(row.LON, row.LAT, marker=tcmarkers.SH_HU, color='r', markeredgecolor='r', markersize=7.5)
            axes.coastlines(color='0.5', linewidth=1.5)
            gl = axes.gridlines(draw_labels=True, linestyle='--')
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style={'size': 'x-small'}
            gl.ylabel_style={'size': 'x-small'}
            gl.top_labels = False
            gl.right_labels = False
            axes.set_extent((lon_cntr - 3*width, lon_cntr + 3*width, lat_cntr-2*width, lat_cntr+2*width))

        fig.suptitle(f"{row.DISTURBANCE_ID} - {row.TM}")
        plotfn = os.path.join(OUT_DIR, f"{row.DISTURBANCE_ID}.dlmflow.png")
        plt.savefig(plotfn, bbox_inches='tight')
        plt.close(fig)


def plot_flow(uda, vda, df, width=4.0):
    griddx, griddy = lat_lon_grid_deltas(uda.longitude, uda.latitude)
    griddx = np.hstack((griddx.magnitude, griddx.magnitude[:, -1].reshape(-1, 1)))
    griddy = np.abs(np.vstack((griddy.magnitude, griddy.magnitude[-1, :].reshape(1, -1))))
    for row in df.itertuples():
        dt = row.TM.to_numpy()
        lat_cntr = 0.25 * np.round(row.LAT * 4)
        lon_cntr = 0.25 * np.round(row.LON * 4)

        lat_slice = slice(lat_cntr + width, lat_cntr - width)
        long_slice = slice(lon_cntr - width, lon_cntr + width)

        # Select data
        us = uda.sel(time=dt, method='nearest')
        vs = vda.sel(time=dt, method='nearest')
        uenv, venv = extract_steering(us, vs, lon_cntr, lat_cntr, griddx, griddy)

        usteer = uenv.sel(longitude=long_slice, latitude=lat_slice).mean(dim=('longitude', 'latitude')).values
        vsteer = venv.sel(longitude=long_slice, latitude=lat_slice).mean(dim=('longitude', 'latitude')).values

        fig, ax = plt.subplots(2, 2, subplot_kw={'projection':ccrs.PlateCarree()},
                               figsize=(12, 8), sharex=True, sharey=True)
        levels = np.arange(0, 40.1, 2)
        ax[0, 0].contourf(us.longitude, us.latitude, np.sqrt(us[0]**2+vs[0]**2), levels=levels, extend="max", cmap="viridis_r")
        ax[0, 0].barbs(us.longitude, us.latitude, us[0], vs[0], flip_barb=True, length=5)
        ax[0, 0].arrow(lon_cntr, lat_cntr, row.u/10, row.v/10, fc='r', ec='r', width=0.2)
        ax[0, 0].set_title("Full flow (850 hPa) [m/s]")

        ax[1, 0].contourf(uenv.longitude, uenv.latitude, np.sqrt(uenv[0]**2+venv[0]**2), levels=levels, extend="max", cmap="viridis_r")
        ax[1, 0].barbs(uenv.longitude, uenv.latitude, uenv[0], venv[0], flip_barb=True, length=5)
        ax[1, 0].arrow(lon_cntr, lat_cntr, row.u/10, row.v/10, fc='w', ec='k', width=0.2)
        ax[1, 0].arrow(lon_cntr, lat_cntr, 3.6*usteer[0]/10, 3.6*vsteer[0]/10, fc='r', ec='r', width=0.2 )
        ax[1, 0].add_patch(mpatches.Rectangle(xy=[lon_cntr - width, lat_cntr - width],
                           width=2*width, height=2*width,
                           facecolor='none', edgecolor='r',
                           linewidth=2,
                           transform=ccrs.PlateCarree()))
        ax[1, 0].set_title("Environmental flow (850 hPa) [m/s]")

        ax[0, 1].contourf(us.longitude, us.latitude, np.sqrt(us[1]**2+vs[1]**2), levels=levels, extend="max", cmap="viridis_r")
        ax[0, 1].barbs(us.longitude, us.latitude, us[1], vs[1], flip_barb=True, length=5)
        ax[0, 1].arrow(lon_cntr, lat_cntr, row.u/10, row.v/10, fc='r', ec='r', width=0.2)
        ax[0, 1].set_title("Full flow (250 hPa) [m/s]")

        ax[1, 1].contourf(uenv.longitude, uenv.latitude, np.sqrt(uenv[1]**2+venv[1]**2), levels=levels, extend="max", cmap="viridis_r")
        ax[1, 1].barbs(uenv.longitude, uenv.latitude, uenv[1], venv[1], flip_barb=True, length=5)
        ax[1, 1].arrow(lon_cntr, lat_cntr, row.u/10, row.v/10, fc='w', ec='k', width=0.2 )
        ax[1, 1].arrow(lon_cntr, lat_cntr, 3.6*usteer[1]/10, 3.6*vsteer[1]/10, fc='r', ec='r', width=0.2 )
        ax[1, 1].add_patch(mpatches.Rectangle(
                           xy=[lon_cntr - width, lat_cntr - width],
                           width=2*width, height=2*width,
                           facecolor='none', edgecolor='r',
                           linewidth=2,
                           transform=ccrs.PlateCarree()))
        ax[1, 1].set_title("Environmental flow (250 hPa) [m/s]")

        for axes in ax.flatten():
            axes.plot(row.LON, row.LAT, marker=tcmarkers.SH_HU, color='w', markeredgecolor='k', markersize=10)
            axes.coastlines(color='0.75', linewidth=1.0)
            gl = axes.gridlines(draw_labels=True, linestyle='--')
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style={'size': 'x-small'}
            gl.ylabel_style={'size': 'x-small'}
            gl.top_labels = False
            gl.right_labels = False
            axes.set_extent((lon_cntr - 3*width, lon_cntr + 3*width, lat_cntr-2*width, lat_cntr+2*width))

        fig.suptitle(f"{row.DISTURBANCE_ID} - {row.TM}")
        plotfn = os.path.join(OUT_DIR, f"{row.DISTURBANCE_ID}.envflow.png")
        plt.savefig(plotfn, bbox_inches='tight')
        plt.close(fig)


def plot_event_flow(uda, vda, df, event, width=4.0):
    os.makedirs(os.path.join(OUT_DIR, event), exist_ok = True)
    griddx, griddy = lat_lon_grid_deltas(uda.longitude, uda.latitude)
    griddx = np.hstack((griddx.magnitude, griddx.magnitude[:, -1].reshape(-1, 1)))
    griddy = np.abs(np.vstack((griddy.magnitude, griddy.magnitude[-1, :].reshape(1, -1))))
    imglist = []
    for row in df.itertuples():
        dt = row.TM.to_numpy()
        lat_cntr = 0.25 * np.round(row.LAT * 4)
        lon_cntr = 0.25 * np.round(row.LON * 4)

        lat_slice = slice(lat_cntr + width, lat_cntr - width)
        long_slice = slice(lon_cntr - width, lon_cntr + width)

        # Select data
        us = uda.sel(time=dt, method='nearest')
        vs = vda.sel(time=dt, method='nearest')
        uenv, venv = extract_steering(us, vs, lon_cntr, lat_cntr, griddx, griddy)

        usteer = uenv.sel(longitude=long_slice, latitude=lat_slice).mean(dim=('longitude', 'latitude')).values
        vsteer = venv.sel(longitude=long_slice, latitude=lat_slice).mean(dim=('longitude', 'latitude')).values

        fig, ax = plt.subplots(2, 2, subplot_kw={'projection':ccrs.PlateCarree()},
                               figsize=(12, 8), sharex=True, sharey=True)
        levels = np.arange(0, 40.1, 2)
        ax[0, 0].contourf(us.longitude, us.latitude, np.sqrt(us[0]**2+vs[0]**2), levels=levels, extend="max", cmap="viridis_r")
        ax[0, 0].barbs(us.longitude, us.latitude, us[0], vs[0], flip_barb=True, length=5)
        ax[0, 0].arrow(lon_cntr, lat_cntr, row.u/10, row.v/10, fc='w', ec='k', width=0.2)
        ax[0, 0].set_title("Full flow (850 hPa) [m/s]")

        ax[1, 0].contourf(uenv.longitude, uenv.latitude, np.sqrt(uenv[0]**2+venv[0]**2), levels=levels, extend="max", cmap="viridis_r")
        ax[1, 0].barbs(uenv.longitude, uenv.latitude, uenv[0], venv[0], flip_barb=True, length=5)
        ax[1, 0].arrow(lon_cntr, lat_cntr, row.u/10, row.v/10, fc='w', ec='k', width=0.2)
        ax[1, 0].arrow(lon_cntr, lat_cntr, 3.6*usteer[0]/10, 3.6*vsteer[0]/10, fc='r', ec='r', width=0.2 )
        ax[1, 0].add_patch(mpatches.Rectangle(xy=[lon_cntr - width, lat_cntr - width],
                           width=2*width, height=2*width,
                           facecolor='none', edgecolor='r',
                           linewidth=2,
                           transform=ccrs.PlateCarree()))
        ax[1, 0].set_title("Environmental flow (850 hPa) [m/s]")

        ax[0, 1].contourf(us.longitude, us.latitude, np.sqrt(us[1]**2+vs[1]**2), levels=levels, extend="max", cmap="viridis_r")
        ax[0, 1].barbs(us.longitude, us.latitude, us[1], vs[1], flip_barb=True, length=5)
        ax[0, 1].arrow(lon_cntr, lat_cntr, row.u/10, row.v/10, fc='w', ec='k', width=0.2)
        ax[0, 1].set_title("Full flow (250 hPa) [m/s]")

        ax[1, 1].contourf(uenv.longitude, uenv.latitude, np.sqrt(uenv[1]**2+venv[1]**2), levels=levels, extend="max", cmap="viridis_r")
        ax[1, 1].barbs(uenv.longitude, uenv.latitude, uenv[1], venv[1], flip_barb=True, length=5)
        ax[1, 1].arrow(lon_cntr, lat_cntr, row.u/10, row.v/10, fc='w', ec='k', width=0.2 )
        ax[1, 1].arrow(lon_cntr, lat_cntr, 3.6*usteer[1]/10, 3.6*vsteer[1]/10, fc='r', ec='r', width=0.2 )
        ax[1, 1].add_patch(mpatches.Rectangle(
                           xy=[lon_cntr - width, lat_cntr - width],
                           width=2*width, height=2*width,
                           facecolor='none', edgecolor='r',
                           linewidth=2,
                           transform=ccrs.PlateCarree()))
        ax[1, 1].set_title("Environmental flow (250 hPa) [m/s]")

        for axes in ax.flatten():
            axes.plot(row.LON, row.LAT, marker=tcmarkers.SH_HU, color='w', markeredgecolor='k', markersize=10)
            axes.coastlines(color='0.75', linewidth=1.0)
            gl = axes.gridlines(draw_labels=True, linestyle='--')
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style={'size': 'x-small'}
            gl.ylabel_style={'size': 'x-small'}
            gl.top_labels = False
            gl.right_labels = False
            axes.set_extent((lon_cntr - 3*width, lon_cntr + 3*width, lat_cntr-2*width, lat_cntr+2*width))

        fig.suptitle(f"{row.DISTURBANCE_ID} - {row.TM}")
        plotfn = os.path.join(OUT_DIR, event, f"{row.DISTURBANCE_ID}.envflow.{row.TM.strftime('%Y%m%d%H')}.png")
        plt.savefig(plotfn, bbox_inches='tight')
        plt.close(fig)
        #imglist.append(imageio.imread(plotfn))
    #imageio.mimwrite(os.path.join(OUT_DIR, event, f"{row.DISTURBANCE_ID}.envflow.gif"), imglist, fps=5)


df = load_bom_df()
df = df[df.MAX_WIND_SPD.notnull()]
lmidf = df.loc[df.groupby(['DISTURBANCE_ID'])['MAX_WIND_SPD'].idxmax()]

upath = sorted(glob.glob(os.path.join(DLM_DIR, "**/era5_u_daily_*.nc")))
vpath = sorted(glob.glob(os.path.join(DLM_DIR, "**/era5_v_daily_*.nc")))

udss = xr.open_mfdataset(upath, combine='nested', concat_dim='time',
                         chunks={'latitude': 721, 'longitude': 1440, 'time': -1},
                         parallel=True)
vdss = xr.open_mfdataset(vpath, combine='nested', concat_dim='time',
                         chunks={'latitude': 721, 'longitude': 1440, 'time': -1},
                         parallel=True)

uda = udss['u']
vda = vdss['v']

plot_flow(uda, vda, lmidf)


events = ["AU199899_10U", "AU200506_22U", "AU202021_22U"]

for event in events:
    eventdf = df.loc[df['DISTURBANCE_ID']==event]
    plot_event_flow(uda, vda, eventdf, event)