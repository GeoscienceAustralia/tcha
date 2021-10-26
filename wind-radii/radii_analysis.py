import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from jtwc import load_jtwc_data
import warnings
warnings.filterwarnings("ignore")

matplotlib.use('tkagg')

sns.set_style('whitegrid')
sns.set_context('notebook', rc={"grid.linewidth": 0.5, "grid.linestyle": ':'})

input_path = os.path.expanduser("~/geoscience/data/jtwc")
out_path = os.path.expanduser("~/geoscience/data/out")
df = load_jtwc_data(input_path)

use_cols = [
    'rMax', 'dP', 'Windspeed', 'Latitude', 'translation_speed',
    'RAD1', 'RAD2', 'RAD3', 'RAD4', 'Roci', 'r34'
]

# set empty RAD values to NaNs in the tempdf to exclude them from the correlation calculations
tmp_df = df[use_cols].copy()
tmp_df['RAD1'].values[(df['RAD'].values == 0) | (df['RAD1'].values == 0)] = np.nan
tmp_df['RAD2'].values[(df['RAD'].values == 0) | (df['RAD2'].values == 0)] = np.nan
tmp_df['RAD3'].values[(df['RAD'].values == 0) | (df['RAD3'].values == 0)] = np.nan
tmp_df['RAD4'].values[(df['RAD'].values == 0) | (df['RAD4'].values == 0)] = np.nan

corr = tmp_df.corr()
corr.values[:] = np.round(corr.values, 3)

print(corr)

# dP vs Rmax plot
fct = sns.relplot(df.dP, df.rMax, height=5, aspect=1.5)
fct.ax.set_xlabel(r"$\Delta p$ (hPa)")
fct.ax.set_ylabel(r"$R_{max}$ (km)")
plt.text(0.0, -0.2, "Source: https://www.metoc.navy.mil/jtwc/jtwc.html \n(accessed 2021-09-14)", transform=fct.ax.transAxes, fontsize='x-small', ha='left')
plt.text(1.0, -0.2, f"Created: {datetime.now():%Y-%m-%d %H:%M}", transform=fct.ax.transAxes, fontsize='x-small', ha='right')
plt.savefig(os.path.join(out_path, "RMax - dP vs RMax.png"), bbox_inches='tight')

# latitude vs Rmax plot
fct = sns.relplot(df.Latitude, df.rMax, aspect=1.5)
fct.ax.set_ylabel(r"$R_{max}$ (km)")
plt.text(0.0, -0.2, "Source: https://www.metoc.navy.mil/jtwc/jtwc.html \n(accessed 2021-09-14)",
          transform=fct.ax.transAxes, fontsize='x-small', ha='left',)
plt.text(1.0, -0.2, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=fct.ax.transAxes, fontsize='x-small', ha='right')
plt.savefig(os.path.join(out_path, "RMax - Latitude vs RMax.png"), bbox_inches='tight')

# translation speed vs Rmax plot
mask = df.translation_speed > 0
fct = sns.relplot(df.translation_speed[mask], df.rMax[mask], aspect=1.5)
fct.ax.set_ylabel(r"$R_{max}$ (km)")
fct.ax.set_xlabel(r"Translation speed (km/h)")
plt.text(0.0, -0.2, "Source: https://www.metoc.navy.mil/jtwc/jtwc.html \n(accessed 2021-09-14)",
          transform=fct.ax.transAxes, fontsize='x-small', ha='left',)
plt.text(1.0, -0.2, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=fct.ax.transAxes, fontsize='x-small', ha='right')
plt.savefig(os.path.join(out_path, "RMax - Translation Speed vs RMax.png"), bbox_inches='tight')

# dP vs R34 plot
fct = sns.relplot(df.dP, df.r34, height=5, aspect=1.5)
fct.ax.set_xlabel(r"$\Delta p$ (hPa)")
fct.ax.set_ylabel(r"$R_{34}$ (km)")
plt.text(0.0, -0.2, "Source: https://www.metoc.navy.mil/jtwc/jtwc.html \n(accessed 2021-09-14)", transform=fct.ax.transAxes, fontsize='x-small', ha='left')
plt.text(1.0, -0.2, f"Created: {datetime.now():%Y-%m-%d %H:%M}", transform=fct.ax.transAxes, fontsize='x-small', ha='right')
plt.savefig(os.path.join(out_path, "R34 - dP vs R34.png"), bbox_inches='tight')

# latitude vs R34 plot
fct = sns.relplot(df.Latitude, df.r34, aspect=1.5)
fct.ax.set_ylabel(r"$R_{34}$ (km)")
plt.text(0.0, -0.2, "Source: https://www.metoc.navy.mil/jtwc/jtwc.html \n(accessed 2021-09-14)",
          transform=fct.ax.transAxes, fontsize='x-small', ha='left',)
plt.text(1.0, -0.2, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=fct.ax.transAxes, fontsize='x-small', ha='right')
plt.savefig(os.path.join(out_path, "R34 - Latitude vs R34.png"), bbox_inches='tight')

# translation speed vs R34 plot
mask = df.translation_speed > 0
fct = sns.relplot(df.translation_speed[mask], df.r34[mask], aspect=1.5)
fct.ax.set_ylabel(r"$R_{34}$ (km)")
fct.ax.set_xlabel(r"Translation speed (km/h)")
plt.text(0.0, -0.2, "Source: https://www.metoc.navy.mil/jtwc/jtwc.html \n(accessed 2021-09-14)",
          transform=fct.ax.transAxes, fontsize='x-small', ha='left',)
plt.text(1.0, -0.2, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=fct.ax.transAxes, fontsize='x-small', ha='right')
plt.savefig(os.path.join(out_path, "R34 - Translation Speed vs R34.png"), bbox_inches='tight')

#########
### plot of the model means

dps = np.linspace(0, 100, 100)
rmax = np.exp(4.178 - 0.0192 * dps + 0.0033 * 5)
r34 = np.exp(4.285 + 0.00965 * dps + 0.0269 * 5)

plt.figure()
ax = sns.lineplot(dps, rmax, label=r'$R_{max}$')
sns.lineplot(dps, r34, label='$R_{34}$')
ax.set_xlabel(r'$\Delta p$ (hPa)')
ax.set_ylabel(r'Radius')
plt.text(1.0, -0.2, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='x-small', ha='right')
plt.savefig(os.path.join(out_path, "radii model means.png"), bbox_inches='tight')


########
### joint sampling

ln_rmax_mean = 4.178 - 0.0192 * df.dP + 0.0033 * df.Latitude
ln_r34_mean = 4.285 + 0.00965 * df.dP + 0.0269 * df.Latitude

# constrained sampling
corrs = []

for i in range(100):
    noise_rmax = np.random.normal(loc=0, size=len(df), scale=0.4)
    noise_r34 = np.random.normal(loc=0, size=len(df), scale=0.36)

    ln_rmax = ln_rmax_mean + noise_rmax
    mask = ln_r34_mean + noise_r34 < ln_rmax
    while mask.sum() > 0:
        noise_r34[mask] = np.random.normal(loc=0, size=mask.sum(), scale=0.362)
        mask = ln_r34_mean + noise_r34 < ln_rmax

    corr = np.corrcoef(np.exp(ln_rmax), np.exp(ln_r34_mean + noise_r34))
    corrs.append(corr[0, 1])

print("Constrained sampling correlation coefficient:", np.mean(corrs))

corrs = []

for _ in range(100):
    noise_shared = np.random.normal(loc=0, size=len(df), scale=1)

    ln_rmax = ln_rmax_mean + 0.4 * noise_shared
    ln_r34 = ln_r34_mean + 0.36 * noise_shared
    corr = np.corrcoef(np.exp(ln_rmax), np.exp(ln_r34))

    corrs.append(corr[0, 1])

print("Single sampling correlation coefficient:", np.mean(corrs))
