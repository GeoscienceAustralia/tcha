"""
Calculate correlations between a range of storm parameters.

Author: C. Arthur
2024-06-27
"""

import os
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
BASEDIR="/scratch/w85/cxa547/envflow/SH"


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

filename = os.path.join(BASEDIR, "tcenvflow.pred.SH.csv")
df = pd.read_csv(filename)
df = df[~df['MAX_WIND_SPD'].isna()]
df = df[df['v'] <= 50]

basins = df.BASIN.unique()
cols = ['LAT', 'MAX_WIND_SPD', 'u', 'v', 'du', 'dv',
        'u850', 'v850', 'u250', 'v250', 'su', 'sv', 'ub', 'vb']
df['ub'] = df['u'] - df['upred']
df['vb'] = df['v'] - df['vpred']

for basin in basins:
    print(f"Calculating correlations for {basin}")
    sdf = df[df.BASIN==basin]

    sdf['du'] = sdf.groupby("DISTURBANCE_ID")['u'].diff()
    sdf['du'] = sdf['du'].fillna(0)

    sdf['dv'] = sdf.groupby("DISTURBANCE_ID")['v'].diff()
    sdf['dv'] = sdf['dv'].fillna(0)

    # Shear vector components:
    sdf['su'] = sdf['u850'] - sdf['u250']
    sdf['sv'] = sdf['v850'] - sdf['v250']

    #sns.pairplot(sdf[cols])
    sdf[cols].corr().to_csv(os.path.join(BASEDIR, f"{basin}corr.csv"))

    fig, axes = plt.subplots(2, 1, sharex=True)
    bins = np.arange(-1, 1.01, 0.05)
    rhou = sdf.groupby('DISTURBANCE_ID')['u'].apply(pd.Series.autocorr)
    rhov = sdf.groupby('DISTURBANCE_ID')['v'].apply(pd.Series.autocorr)
    sns.histplot(rhou, bins=bins, ax=axes[0], stat='probability')
    sns.histplot(rhov, bins=bins, ax=axes[1], stat='probability')
    axes[0].set_title(rf"Zonal translation speed $\rho_u$ = {rhou.mean():.2f}")
    axes[1].set_title(rf"Meridional translation speed $\rho_v$ = {rhov.mean():.2f}")
    savefig(os.path.join(BASEDIR, f"{basin}.autocorr.png"),
            bbox_inches='tight')