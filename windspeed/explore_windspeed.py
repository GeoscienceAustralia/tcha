import tensorflow as tf
import sys
import os
import numpy as np
import xarray as xr

sys.path.append(os.path.expanduser('~/geoscience/tcrm'))

from wind.windmodels import WindProfileModel, HollandWindProfile, WindFieldModel, PowellWindProfile
import Utilities.metutils as metutils
from matplotlib import pyplot as plt
from tf_models import TFHollandWindProfile, TFKerpertHolland, fcast


def polarGridAroundEye(lon, lat, margin=2, resolution=0.02):
    R, theta = makeGrid(lon, lat, margin, resolution)
    return R, theta


def meshGrid(lon, lat, margin=2, resolution=0.02):
    xgrid, ygrid = meshLatLon(lon, lat, margin, resolution)
    return xgrid, ygrid


lat = np.arange(-30, -4, 2, dtype=np.float32)
pc = np.arange(900, 991, 5, dtype=np.float32)
pe = np.arange(995, 1016, dtype=np.float32)
rm = np.arange(10, 91, 5, dtype=np.float32)
vfm = np.arange(0, 51, 5, dtype=np.float32)

coords = {
    'pc': pc,
    'lat': lat,
    'penv': pe,
    'rmax': rm,
    'vfm': vfm,
}

lat, pc, pe, rm, vfm = np.meshgrid(lat, pc, pe, rm, vfm)

dP = 100 * (pe - pc).flatten()
rMax = rm.flatten() * 1000
vFm = vfm.flatten() / 3.6

thetaFm = 0
thetaFm = fcast(thetaFm)

f = metutils.coriolis(lat.flatten())
beta = 1.881093 - 0.010917 * np.abs(lat.flatten()) - 0.005567 * 0.000539957 * rMax.flatten()

## run optimization

import time

R = tf.Variable(rm.flatten() * 1000)
lam = tf.Variable(np.zeros_like(f))

u, v = TFKerpertHolland.field(dP, beta, f, rMax, R, lam, vFm, thetaFm, thetaMax=0.)
speed = tf.sqrt(u ** 2 + v ** 2) * 1.268
speed_old = speed.numpy()
speed_orig = speed.numpy().copy()

step = 0.1
t0 = time.time()
for _ in range(20):
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            u, v = TFKerpertHolland.field(dP, beta, f, rMax, R, lam, vFm, thetaFm, thetaMax=0.)
            speed = tf.sqrt(u ** 2 + v ** 2)

            dydx = tape1.gradient(speed, lam)
            lam.assign(lam + step * dydx)
            dydx = tape2.gradient(speed, R)
            R.assign(R + step * dydx)
            # d2ydx2 = tape2.gradient(dydx, lam)
            # lam.assign(lam + dydx * tf.math.divide_no_nan(1.0, tf.abs(d2ydx2)))

    u, v = TFKerpertHolland.field(dP, beta, f, rMax, R, lam, vFm, thetaFm, thetaMax=0.)
    speed = tf.sqrt(u ** 2 + v ** 2).numpy() * 1.268
    print((speed - speed_orig).min(), (speed - speed_orig).max(), np.abs(speed - speed_old).max())
    speed_old = speed

print("Time taken:", time.time() - t0, 's')


## plotting

arr = xr.DataArray(speed.reshape(lat.shape), coords=coords, dims=list(coords.keys()))
DIR = os.path.expanduser("~/geoscience/data/windspeed-plots")

for var in coords.keys():
    axes = plt.subplots(6, 1, figsize=(5, 10), sharex=True)
    for i in range(6):
        selection = {key: np.random.choice(value) for key, value in coords.items() if key != var}

        ax = axes[1].flatten()[i]

        ax.set_title(', '.join(f'{key}={var}' for key, var in selection.items()))
        ax.set_ylabel('windspeed (m/s)')
        ax.plot(coords[var], arr.sel(**selection).data)
        ax.grid()

    axes[1].flatten()[-1].set_xlabel(var)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR, f'{var}.png'))

for var in coords.keys():
    axes = plt.subplots(6, 1, figsize=(5, 10), sharex=True)
    for i in range(6):
        selection = {key: np.random.choice(value) for key, value in coords.items() if key != var}

        use_vals = selection.copy()
        data = []
        for x in coords[var]:
            use_vals[var] = x

            profile = PowellWindProfile(
                use_vals['lat'],
                120,
                use_vals['penv'],
                use_vals['pc'],
                use_vals['rmax'] * 1000,
            )
            data.append(np.abs(profile.velocity(R)).max())

        ax = axes[1].flatten()[i]

        ax.set_title(', '.join(f'{key}={var}' for key, var in selection.items()))
        ax.set_ylabel('windspeed (m/s)')
        ax.plot(coords[var], data)
        ax.grid()

    axes[1].flatten()[-1].set_xlabel(var)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR, f'gradient_wind_{var}.png'))