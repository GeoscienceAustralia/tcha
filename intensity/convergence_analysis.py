from era5_extract import load_otcr_df
import os
import numpy as np
from tc_intensity_analysis import Hurricane
import pandas as pd
import time
from matplotlib import pyplot as plt

"""
Ths scripts analysis the convergence of the TC intensity model with default parameters on a challenging
test case with a LMI > maximum MPI for some simulations.
"""


pres_lvls = np.array([ 1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,
    125,  150,  175,  200,  225,  250,  300,  350,  400,  450,  500,
    550,  600,  650,  700,  750,  775,  800,  825,  850,  875,  900,
    925,  950,  975, 1000], dtype=np.float32)

DATA_DIR = os.path.expanduser("~/geoscience/data")

df = load_otcr_df(DATA_DIR)

era5 = np.load(os.path.join(DATA_DIR, "tc_intensity_era5.npy"))
bran = np.load(os.path.join(DATA_DIR, "tc_intensity_bran2020.npy"))
rh = np.load(os.path.join(DATA_DIR, "tc_intensity_rh.npy"))

for i, var in enumerate(["sst", "sp", "d2", "t2"]): df[var] = era5[:, i]
df["hm"] = bran[:, 0]
df['rh'] = rh[:, 21]  # np.trapz(rh[:, pres_lvls >= 200], pres_lvls[pres_lvls >= 200]) / (800)

df = df[~pd.isnull(df["Vmax (kn)"])]
df = df[df.DISTURBANCE_ID == 'AU200405_10U']

dts = 10.0 * (2.0 ** -np.arange(10))
vms = []

for dt in dts:
    hurricane = Hurricane(dt=dt)
    bailed, out = hurricane.simulate(df)
    vms.append(out.vmax.max())

plt.semilogx(dts, vms)
plt.ylabel('LMI (m/s)')
plt.xlabel('Timestep (s)')
plt.show()
