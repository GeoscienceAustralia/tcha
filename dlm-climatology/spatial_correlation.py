import xarray as xr
import numpy as np
import os
import cv2
import pandas as pd


path = os.path.expanduser("~/Downloads")


def load_dlm(year, month):
    ufile = f"{path}/u_dlm_{month}_{year}.netcdf"
    vfile = f"{path}/v_dlm_{month}_{year}.netcdf"

    udlm = xr.open_dataset(ufile, engine='netcdf4')
    vdlm = xr.open_dataset(vfile, engine='netcdf4')

    return udlm, vdlm


def tc_velocity(udlm, vdlm):
    """
    Perform an average over a 12.5 x 12.5 degree box of the deep layer mean flow using
    opencv.

    Note: to do a spatial averaging opencv requires arrays in the format (y, x, time). Due to a bug in opencv
    if the time axis is too large compared to the y, x, and the size of the averaging area, opencv crashes. As a
    workaround the spatial averaging is performed on chunks of up 500 time steps.
    """
    #
    # using opencv
    ksize = 51

    correction = np.ones_like(udlm.u.data)
    correction *= (1 / cv2.blur(correction[0, ...], ksize=(ksize, ksize), borderType=cv2.BORDER_ISOLATED))[None, ...]

    # annoying opencv bug means this must be split
    udlm_transpose = udlm.u.data.transpose()
    u = np.zeros_like(udlm_transpose)
    u[..., :500] = cv2.blur(udlm_transpose[..., :500], ksize=(ksize, ksize), borderType=cv2.BORDER_ISOLATED)
    u[..., 500:] = cv2.blur(udlm_transpose[..., 500:], ksize=(ksize, ksize), borderType=cv2.BORDER_ISOLATED)
    u = u.transpose() * correction

    vdlm_transpose = vdlm.v.data.transpose()
    v = np.zeros_like(vdlm_transpose)
    v[..., :500] = cv2.blur(vdlm_transpose[..., :500], ksize=(ksize, ksize), borderType=cv2.BORDER_ISOLATED)
    v[..., 500:] = cv2.blur(vdlm_transpose[..., 500:], ksize=(ksize, ksize), borderType=cv2.BORDER_ISOLATED)
    v = v.transpose() * correction

    u = -4.5205 + 0.8978 * u * 3.6
    v = -1.2542 + 0.7877 * v * 3.6

    u = xr.DataArray(u, coords=udlm.coords)
    v = xr.DataArray(v, coords=vdlm.coords)

    return u, v


month = 9
year = 1991

udlm, vdlm = load_dlm(year, month)
u, v = tc_velocity(udlm, vdlm)

longitude, latitude = np.meshgrid(u.coords['longitude'].data, u.coords['latitude'].data)
longitude_index = pd.Series(np.arange(len(udlm.coords['longitude'].data)), udlm.coords['longitude'].data)
latitude_index = pd.Series(np.arange(len(udlm.coords['latitude'].data)), udlm.coords['latitude'].data)

# for each time step:
#   for each point: (vectorized so no explicit for loop)
#       - the spatial average of the deep layer mean is calculated
#       - a the beta advection model is then used to calculate the velocity that a hypothetical
#           cyclone would have at this time and location
#       - the deep layer mean is extracted for locations a dt * velocity displacement away from this point

dt = 6

u_next = np.zeros_like(u)
v_next = np.zeros_like(v)

u_next[:] = np.nan
v_next[:] = np.nan

for time_idx, t in enumerate(u.coords['time'].data[:-dt]):
    lat = latitude + v.data[time_idx] * 180 / (np.pi * 6378)
    long = longitude + u.data[time_idx] * 180 / (np.pi * 6378 * np.cos(np.pi * latitude / 180))

    long = np.round(4 * long) / 4
    lat = np.round(4 * lat) / 4

    mask = (-40 <= lat) & (lat <= 0)
    mask &= (80 <= long) & (long <= 170)

    lat_idxs = latitude_index.loc[lat[mask]].values
    long_idxs = longitude_index.loc[long[mask]].values

    sz2 = udlm.u.data.shape[2]
    sz1 = sz2 * udlm.u.data.shape[1]

    idxs = ((time_idx + dt) * sz1 + lat_idxs * sz2 + long_idxs).astype(int)

    u_next[time_idx][mask] = udlm.u.data.take(idxs)
    v_next[time_idx][mask] = vdlm.v.data.take(idxs)

# mask out pairs of (dlm, dlm at next cyclone location) that are out of bounds

u_flat = udlm.u.data.flatten()
v_flat = vdlm.v.data.flatten()
u_next_flat = u_next.flatten()
v_next_flat = v_next.flatten()

mask = (~np.isnan(u_next_flat)) & (~np.isnan(v_next_flat))

u_flat = u_flat[mask]
v_flat = v_flat[mask]
u_next_flat = u_next_flat[mask]
v_next_flat = v_next_flat[mask]

# calculation the correlation between dlm an dlm at next cyclone location

print("Correlation in zonal velocity:", np.corrcoef(u_flat, u_next_flat)[0, 1])
print("Correlation in meridional velocity:", np.corrcoef(v_flat, v_next_flat)[0, 1])
