import os
import pandas as pd
import numpy as np
import seaborn as sns
BASEDIR="/scratch/w85/cxa547/envflow"
filename = os.path.join(BASEDIR, "tcenvflow.csv")
df = pd.read_csv(filename)
df = df[~df['MAX_WIND_SPD'].isna()]
df = df[df['v'] <= 50]
spdf = df[df.BASIN=='SP']
sidf = df[df.BASIN=='SI']

spdf['du'] = spdf.groupby("DISTURBANCE_ID")['u'].diff()
spdf['du'] = spdf['du'].fillna(0)
sidf['du'] = sidf.groupby("DISTURBANCE_ID")['u'].diff()
sidf['du'] = sidf['du'].fillna(0)
spdf['dv'] = spdf.groupby("DISTURBANCE_ID")['v'].diff()
spdf['dv'] = spdf['dv'].fillna(0)
sidf['dv'] = sidf.groupby("DISTURBANCE_ID")['v'].diff()
sidf['dv'] = sidf['dv'].fillna(0)

spdf['su'] = spdf['u850'] - spdf['u250']
spdf['sv'] = spdf['v850'] - spdf['v250']
sidf['su'] = sidf['u850'] - sidf['u250']
sidf['sv'] = sidf['v850'] - sidf['v250']
cols = ['LAT', 'MAX_WIND_SPD', 'u', 'v', 'du', 'dv', 'u850', 'v850', 'u250', 'v250', 'su', 'sv']
sns.pairplot(spdf[cols])
sns.pairplot(sidf[cols])

spdf[cols].corr().to_csv(os.path.join(BASEDIR, "SPcorr.csv"))
sidf[cols].corr().to_csv(os.path.join(BASEDIR, "SIcorr.csv"))