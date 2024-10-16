import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from jtwc import load_jtwc_data
import scipy.stats as stats
from seaborn.utils import _kde_support
import pandas as pd
import statsmodels.api as sm
import statsmodels.nonparametric.api as smnp
from six import string_types
from lmfit import Model, Minimizer, fit_report, conf_interval, printfuncs, report_fit
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

matplotlib.use('tkagg')
matplotlib.rcParams['grid.linestyle'] = ':'
matplotlib.rcParams['grid.linewidth'] = 0.5
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['axes.labelsize'] = "small"

sns.set_style('whitegrid')
sns.set_context('notebook', rc={"grid.linewidth": 0.5, "grid.linestyle": ':'})

###########
###########
# set the output and inot paths
input_path = os.path.expanduser("~/geoscience/data/jtwc")
input_path = r"X:\georisk\HaRIA_B_Wind\data\raw\from_jtwc\bsh"
out_path = os.path.expanduser("~/geoscience/data/out")
out_path = r"X:\georisk\HaRIA_B_Wind\projects\tcha\data\derived\windradii"


def lin_dp(x, alpha, beta):
    dp = x[:, 0]
    return alpha + beta * dp


def lin_lat(x, zeta):
    lat = np.abs(x[:, 1])
    return zeta * lat


def resid(p):
    return p['alpha'] + p['beta'] * X[:, 0] + p['zeta'] * np.abs(X[:, 1]) - y

df = load_jtwc_data(input_path)

#
# fit the linear model
#

mask = ~np.isnan(df.r34.values)
X = np.column_stack((df.dP.values[mask], df.Latitude.values[mask]))
y = np.log(df.rMax.values[mask])
# X = np.column_stack((df.dP.values, df.Latitude.values))
# y = np.log(df.rMax.values)
rmod = Model(lin_dp) + Model(lin_lat)
params = rmod.make_params(alpha=1., beta=-0.001, zeta=.001)

params = rmod.make_params(alpha=1., beta=-0.001, zeta=.001)
result = rmod.fit(y, x=X,  params=params)
residuals = result.eval(x=X) - y
print(result.values)
print("RMSE:", np.sqrt(np.mean(residuals ** 2)))

#
# normal test of residuals in log space
#

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))

ax = sns.distplot(
    residuals, hist_kws={'ec':'b', 'width':0.05},
    kde_kws={'label': 'Residuals', 'linestyle': '--'}, ax=ax0, norm_hist=True
)
pp = sm.ProbPlot(residuals, stats.norm, fit=True)
pp.qqplot('Residuals', 'Normal', line='45', ax=ax1, color='gray', alpha=0.5)
fig.tight_layout()
x = np.linspace(-2, 2, 1000)

ax0.legend(loc=0)

fp = stats.norm.fit(residuals)
ax0.plot(x, stats.norm.pdf(x, fp[0], fp[1]), label='Normal', color='r')
print(stats.normaltest(residuals))
ax0.legend()
plt.text(0.0, -0.15, "Source: https://www.metoc.navy.mil/jtwc/jtwc.html \n(accessed 2024-07-17)",
          transform=ax0.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.15, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax1.transAxes, fontsize='xx-small', ha='right')
plt.savefig(os.path.join(out_path, "Rmax residuals.png"), bbox_inches='tight')

#
# plot model vs observations
#
X_all = np.column_stack((df.dP.values, df.Latitude.values))
pred = result.eval(x=X_all)
noise_var = np.var(residuals)
noise = np.random.normal(loc=0, size=len(pred), scale=np.sqrt(noise_var))
rm = np.exp(pred + noise)

sns.set_context("poster")
sns.set_style("whitegrid")
fig, ax = plt.subplots(1, 1, figsize=(12, 8), sharey=True)
ax.scatter(df.dP, rm, c='b', cmap=sns.light_palette('blue', as_cmap=True), s=40, label='Model', alpha=0.5)
ax.scatter(df.dP, df.rMax, c='k', edgecolor=None, s=50, marker='x', label='Observations')
ax.set_xlim(0, 100)
ax.set_xlabel(r"$\Delta p$ (hPa)")
ax.set_ylabel(r"$R_{max}$ (km)")
ax.set_yticks(np.arange(0, 201, 25))
ax.legend(loc=1)
ax.grid(True)
plt.text(-0.2, -0.15, "Source: https://www.metoc.navy.mil/jtwc/jtwc.html \n(accessed 2024-07-17)",
          transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.15, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(os.path.join(out_path, "RMax - model-obs.png"), bbox_inches='tight')

#
# Distribution plots
#

def bivariate_kde(x, y, bw='scott', gridsize=100, cut=3, clip=None):
    if isinstance(bw, string_types):
        bw_func = getattr(smnp.bandwidths, "bw_" + bw)
        x_bw = bw_func(x)
        y_bw = bw_func(y)
        bw = [x_bw, y_bw]
    elif np.isscalar(bw):
        bw = [bw, bw]

    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    kde = smnp.KDEMultivariate([x, y], "cc", bw)
    x_support = _kde_support(x, kde.bw[0], gridsize, cut, [x.min(), x.max()])
    y_support = _kde_support(y, kde.bw[1], gridsize, cut, [y.min(), y.max()])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde.pdf([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z


def l2score(obs, model):
    return np.linalg.norm(obs - model, np.inf)


# generate some noisy predictions
print("Standard deviation:", np.sqrt(noise_var))

xx, yy, odp_rmax = bivariate_kde(df.dP,  df.rMax, bw='scott')
xx, yy, mdp_rmax = bivariate_kde(df.dP, rm, bw='scott')

xx, yy, olat_rmax = bivariate_kde(df.Latitude, df.rMax, bw='scott')
xx, yy, mlat_rmax = bivariate_kde(df.Latitude, rm, bw='scott')

l2rmdp = l2score(odp_rmax, mdp_rmax)
l2rmlat = l2score(olat_rmax, mlat_rmax)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
levs = np.arange(0.01, 0.11, 0.01)
ax = sns.kdeplot(x=df.dP,  y=df.rMax, cmap='Reds', levels=levs, shade=True, shade_lowest=False)
ax = sns.kdeplot(x=df.dP, y=rm, cmap='Blues', levels=levs)
ax.set_xlim(0, 100)
ax.set_ylim(0, 150)
ax.set_xlabel(r"$\Delta p$ (hPa)")
ax.set_ylabel(r"$R_{max}$ (km)")
ax.grid(True)

red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
ax.text(80, 90, "Observed", color=red)
ax.text(80, 80, "Model", color=blue)
ax.text(80, 70, r"$l_2=${0:.3f}".format(l2rmdp))
plt.text(-0.2, -0.15, "Source: https://www.metoc.navy.mil/jtwc/jtwc.html \n(accessed 2021-09-14)",
          transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.15, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(os.path.join(out_path, "RMax - dP RMax model distribution.png"), bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = sns.kdeplot(x=df.Latitude, y=df.rMax, cmap='Reds', levels=levs, shade=True, shade_lowest=False)
ax = sns.kdeplot(x=df.Latitude, y=rm, cmap='Blues', levels=levs)
ax.set_xlim(-30, 0)
ax.set_ylim(0, 150)

ax.set_xlabel("Latitude")
ax.set_ylabel(r"$R_{max}$ (km)")
ax.grid(True)

ax.text(-5, 90, "Observed", color=red)
ax.text(-5, 80, "Model", color=blue)
ax.text(-5, 70, r"$l_2=${0:.3f}".format(l2rmlat))
plt.text(-0.2, -0.15, "Source: https://www.metoc.navy.mil/jtwc/jtwc.html \n(accessed 2021-09-14)",
          transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.15, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(os.path.join(out_path, "RMax - lat RMax model distribution.png"), bbox_inches='tight')

##### model comparison
exp = np.exp(-0.0022 * df.dP ** 2)
pred_old = 3.543 - 0.00378 * df.dP + 0.813 * exp + 0.00157 * df.Latitude ** 2

log_residuals = (pred_old - np.log(df.rMax))
noise_term = np.var(log_residuals)
noise = np.random.normal(loc=0, size=len(df), scale=np.sqrt(noise_term))
pred_old = np.exp(pred_old + noise)


fig, axes = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
axes[0].scatter(df.dP, pred_old, s=6)
axes[1].scatter(df.dP, rm, s=6)
axes[0].title.set_text("Powell (2005) Model")
axes[1].title.set_text("Linear Model")
axes[0].set_xlabel('$\Delta p$ (hPa)', fontsize=16)
axes[1].set_xlabel('$\Delta p$ (hPa)', fontsize=16)
axes[0].set_ylabel('$R_{max}$ (km)', fontsize=16)
plt.text(-0.2, -0.2, "Source: https://www.metoc.navy.mil/jtwc/jtwc.html \n(accessed 2021-09-14)",
          transform=axes[0].transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.2, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=axes[1].transAxes, fontsize='xx-small', ha='right')
fig.savefig(os.path.join(out_path, "RMax old new comparison.png"), bbox_inches='tight')
