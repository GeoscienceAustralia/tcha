import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from datetime import datetime
sns.set_palette('viridis', n_colors=12)
proj = ccrs.PlateCarree(central_longitude=180)
trans = ccrs.PlateCarree()

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


BASEDIR = "/scratch/w85/cxa547/envflow/cyclic"
df = pd.read_csv(os.path.join(BASEDIR, "tcenvflow.pred.SH.csv"))
df["TM"] = pd.to_datetime(
        df.TM, format="%Y-%m-%d %H:%M:%S", errors="coerce")
df['ub'] = df['u'] - df['upred']
df['vb'] = df['v'] - df['vpred'] 

df = df[df.TM.dt.month.isin([1, 2, 3, 12])]
xBins = np.arange(40, 241, 2.5)
yBins = np.arange(-40, 0.1, 2.5)

xPoints = xBins + 1.25
yPoints = yBins + 1.25
sum_u = np.zeros((len(yBins), len(xBins)))
sum_v = np.zeros((len(yBins), len(xBins)))
count = np.zeros((len(yBins), len(xBins)))
sum_ub = np.zeros((len(yBins), len(xBins)))
sum_vb = np.zeros((len(yBins), len(xBins)))
countb = np.zeros((len(yBins), len(xBins)))
sum_up = np.zeros((len(yBins), len(xBins)))
sum_vp = np.zeros((len(yBins), len(xBins)))
countp = np.zeros((len(yBins), len(xBins)))

for index, row in df.dropna(subset=['upred', 'vpred']).iterrows():
    if row['LAT'] < yBins.min():
        continue
    if (row['LON'] < xBins.min()) | (row['LON'] > xBins.max()):
        continue
    if (row['upred'] == np.nan) | (row['vpred']== np.nan):
        continue
    lat_index = int((row['LAT'] - yBins.min()) // 2.5)
    lon_index = int((row['LON'] - xBins.min()) // 2.5)
    sum_u[lat_index, lon_index] += row['u']
    sum_v[lat_index, lon_index] += row['v']
    sum_ub[lat_index, lon_index] += row['ub']
    sum_vb[lat_index, lon_index] += row['vb']
    sum_up[lat_index, lon_index] += row['upred']
    sum_vp[lat_index, lon_index] += row['vpred']
    count[lat_index, lon_index] += 1

average_u = sum_u / count
average_v = sum_v / count
average_ub = sum_ub / count
average_vb = sum_vb / count
average_up = sum_up / count
average_vp = sum_vp / count
mask = np.where(count <= 5)

average_u[mask] = np.nan
average_v[mask] = np.nan
average_up[mask] = np.nan
average_vp[mask] = np.nan
average_ub[mask] = np.nan
average_vb[mask] = np.nan

mag_o = np.sqrt(average_u**2 + average_v**2)
mag_b = np.sqrt(average_ub**2 + average_vb**2)

fig, ax = plt.subplots(3, 1, figsize=(12, 12), subplot_kw={'projection': proj}, sharex=True)
q0 = ax[0].quiver(xPoints, yPoints, average_u, average_v, pivot='mid', transform=trans, scale=1, units='xy')
q1 = ax[1].quiver(xPoints, yPoints, average_up, average_vp, pivot='mid', color='r', transform=trans, scale=1, units='xy')
q2 = ax[2].quiver(xPoints, yPoints, average_ub, average_vb, pivot='mid', color='b', transform=trans, scale=1, units='xy')
#ax.contour(xPoints, yPoints, mag_b, transform=trans, color='0.5', levels=np.arange(0, 10, 0.5))
qk = ax[0].quiverkey(q0, 0.1, 0.05, 5, r'5 m/s', labelpos='N',
                   coordinates='axes')

ax[0].coastlines()
ax[0].set_extent((100, 200, -40, 0), crs=trans)
gl = ax[0].gridlines(draw_labels=True, linestyle=":")
gl.top_labels = False
gl.right_labels = False
ax[0].text(0.05, 0.9, "Observed", fontweight='bold',
           transform=ax[0].transAxes, 
           bbox=dict(facecolor='white', edgecolor='black', alpha=0.75))

ax[1].coastlines()
ax[1].set_extent((100, 200, -40, 0), crs=trans)
gl = ax[1].gridlines(draw_labels=True, linestyle=":")
gl.top_labels = False
gl.right_labels = False
ax[1].text(0.05, 0.9, "Predicted", fontweight='bold',
           color='r', transform=ax[1].transAxes, 
           bbox=dict(facecolor='white', edgecolor='black', alpha=0.75))

ax[2].coastlines()
ax[2].set_extent((100, 200, -40, 0), crs=trans)
gl = ax[2].gridlines(draw_labels=True, linestyle=":")
gl.top_labels = False
gl.right_labels = False
ax[2].text(0.05, 0.9, "Beta", fontweight='bold',
           color='b', transform=ax[2].transAxes, 
           bbox=dict(facecolor='white', edgecolor='black', alpha=0.75))

savefig(os.path.join(BASEDIR, "tcenvflow.pred.SH.png"), bbox_inches='tight')


# Plot ratio of estimated beta drift to observed translation speed
fig, ax = plt.subplots(1, 1, figsize=(12, 7), subplot_kw={'projection': proj}, sharex=True)
cf = ax.contourf(xPoints, yPoints, mag_b / mag_o, 
                 levels=[0., 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                 transform=trans, cmap='viridis_r', extend='max')
ax.coastlines()
ax.set_extent((100, 200, -40, 0), crs=trans)
gl = ax.gridlines(draw_labels=True, linestyle=":")
gl.top_labels = False
gl.right_labels = False
cax = fig.add_axes([0.1, 0.1, 0.85, 0.05], )
fig.colorbar(cf, cax=cax, orientation='horizontal', aspect=50, extend='max')
savefig(os.path.join(BASEDIR, "tcenvflow.betaratio.SH.png"), bbox_inches='tight')
