"""
Calculate intensity change

Calculates 6-hourly and 24 hourly intensity change in observed TC tracks

Uses 10-minute mean intensity change.

"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUTDIR = r"..\data\intensity"
path = os.path.dirname(os.getcwd())
sys.path.append(path)
from utils import load_ibtracs_df, savefig

udf = load_ibtracs_df(basins=["SI", "SP"])

udf = udf.sort_values(by = ["DISTURBANCE_ID", "TM"])
udf["6hrdI"] = udf.groupby("DISTURBANCE_ID")["MAX_WIND_SPD"].diff(periods=1)
udf["24hrdI"] = udf.groupby("DISTURBANCE_ID")["MAX_WIND_SPD"].diff(periods=4)

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
bins = np.arange(-80, 80.1, 5)
sns.histplot(data=udf, x="6hrdI", bins=bins, color="skyblue", edgecolor="black", stat="probability", ax=ax[0])
ax[0].set_title("6-hourly intensity change distribution")
ax[0].set_xlabel("Intensity change [kts]")
ax[0].set_ylabel("Probability")

sns.histplot(data=udf, x="24hrdI", bins=bins, color='salmon', edgecolor='black', stat='probability', ax=ax[1])
ax[1].set_title('24-hourly intensity change distribution')
ax[1].set_xlabel('Intensity change [kts]')
ax[1].set_ylabel('Probability')

plt.text(0.01, 0.0, "Source: IBTrACS doi:10.25921/82ty-9e16",
        transform=fig.transFigure, ha='left', va='bottom',
        fontsize='xx-small')

savefig(os.path.join(OUTPUTDIR, "intensity_change.jpg"),
        dpi=600, bbox_inches='tight')
plt.close(fig)

ax = sns.histplot(data=udf, x="6hrdI", bins=bins, color='skyblue', edgecolor='black', stat='probability')
ax.set_title('24-hourly intensity change distribution')
ax.set_xlabel('Intensity change [kts]')
ax.set_xlim((-40, 40))
ax.set_ylabel('Probability')
plt.text(0.01, 0.0, "Source: IBTrACS doi:10.25921/82ty-9e16",
        transform=fig.transFigure, ha='left', va='bottom',
        fontsize='xx-small')
savefig(os.path.join(OUTPUTDIR, "6hr_intensity_change.jpg"),
        dpi=600, bbox_inches='tight')
plt.close(plt.gcf())

ax = sns.histplot(data=udf, x="24hrdI", bins=bins, color='salmon', edgecolor='black', stat='probability')
ax.set_title('24-hourly intensity change distribution')
ax.set_xlabel('Intensity change [kts]')
ax.set_ylabel('Probability')
plt.text(0.01, 0.0, "Source: IBTrACS doi:10.25921/82ty-9e16",
        transform=fig.transFigure, ha='left', va='bottom',
        fontsize='xx-small')
savefig(os.path.join(OUTPUTDIR, "24hr_intensity_change.jpg"),
        dpi=600, bbox_inches='tight')


hist6hr = pd.cut(udf['6hrdI'], bins=bins, labels=bins[:-1], right=False)
hist24hr = pd.cut(udf['24hrdI'], bins=bins, labels=bins[:-1], right=False)

outdf = pd.DataFrame(columns=["dI", "6hr", "24hr"])
outdf["dI"] = bins[:-1]
outdf["6hr"] = hist6hr.value_counts().sort_index().values
outdf["24hr"] = hist24hr.value_counts().sort_index().values
outdf.to_csv(os.path.join(OUTPUTDIR, "intensity_change.csv"))