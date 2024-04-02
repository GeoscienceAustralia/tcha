"""
Fit beta-advection model parameters to TC motion

Author: Craig Arthur
2024-03-20
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import lmfit
import seaborn as sns
from datetime import datetime
sns.set_palette('viridis', n_colors=12)


BASEDIR = r"X:\georisk\HaRIA_B_Wind\projects\tcha\data\derived\envflow"
filelist = sorted(glob.glob(os.path.join(BASEDIR, "tcenvflow_serial.*.csv")))
df = pd.concat((pd.read_csv(f) for f in filelist), ignore_index=True)

df.loc[df['LON'] < 0, "LON"] = df['LON'] + 360
df.dropna(subset=['u850', 'u250', 'v850', 'v250'], inplace=True)


# Define the different models that will be used:
def twolevel_model(alpha, xl, xu):
    return alpha*xl + (1-alpha)*xu


# This model has fixed ratio for lower and upper flow:
def fixed_model(xl, xu):
    return 0.8*xl + 0.2*xu


# For fitting linear regression of alpha with intensity:
def lin_model(m, b, x):
    return (m*x + b)


# Set up the models:
#  `rmod` has varying alpha with wind speed
#  `fmod` has fixed alpha = 0.8
alpha = lmfit.Parameter('alpha', value=0.8, min=0, max=1)
p = lmfit.Parameters()
p.add(alpha)
rmod = lmfit.Model(twolevel_model, independent_vars=['xl', 'xu'])
fmod = lmfit.Model(fixed_model, independent_vars=['xl', 'xu'])


def plot_scatter(df, filename):
    """
    Scatter plot of predicted and observed translation speed. In this one,
    alpha is not considered a function of storm intensity

    :param df: `pd.DataFrame` of the TC observations, including
    components of observed translation velocity and lower/upper
    steering flow

    :param filename: filename to save the figure to. Uses the `BASEDIR`
    global variable for the destination path.

    """
    # Set predictors and predictands
    u = df['u']  # Observed u component
    ul = df['u850']  # Low-level u component of environmental flow
    uu = df['u250']  # Upper-level u component of environmental flow
    v = df['v']
    vl = df['v850']
    vu = df['v250']

    # Fit the first model
    uresult = rmod.fit(u, p, xl=ul, xu=uu)
    vresult = rmod.fit(v, p, xl=vl, xu=vu)

    ua = uresult.params['alpha'].value
    va = vresult.params['alpha'].value

    print("Result of fitting all data")
    print(uresult.fit_report())
    print(vresult.fit_report())
    # Plot the results of the first model:
    # This model doesn't filter in any way
    # (intensity, basin, time period, etc.)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ax[0].plot(ua*ul + (1-ua)*uu, u, 'o', alpha=0.5)
    sns.kdeplot(x=ua*ul + (1-ua)*uu, y=u, levels=5, color='w',
                linewidths=1, ax=ax[0])
    ax[0].set_ylim((-15, 15))
    ax[0].set_xlim((-15, 15))
    ax[0].set_xlabel(r"Predicted $v_t$ [m/s]")
    ax[0].set_ylabel(r"Observed $v_t$ [m/s]")
    ax[0].grid()
    eqstr = rf"$u_t = {{{np.round(ua, 2)}}}u_{{850}} + {{{np.round(1-ua, 2)}}}u_{{250}}$"  # noqa
    ax[0].text(0.05, 0.95, eqstr, transform=ax[0].transAxes,
               fontweight='bold', va='top')
    ax[1].plot(va*vl + (1-va)*vu, v, 'o', alpha=0.5)
    sns.kdeplot(x=va*vl + (1-va)*vu, y=v, levels=5, color='w',
                linewidths=1, ax=ax[1])
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")

    ax[1].set_ylim((-15, 15))
    ax[1].set_xlim((-15, 15))
    ax[1].grid()
    ax[1].set_xlabel(r"Predicted $v_t$ [m/s]")
    eqstr = rf"$v_t = {{{np.round(va, 2)}}}v_{{850}} + {{{np.round(1-va,2)}}}v_{{250}}$"  # noqa
    ax[1].text(0.05, 0.95, eqstr, transform=ax[1].transAxes,
               fontweight='bold', va='top')

    plt.text(0.95, 0.05, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=fig.transFigure, ha='right', fontsize='xx-small')
    plt.savefig(os.path.join(BASEDIR, filename),
                bbox_inches='tight')


# Add bands for intensity and longitude:
df['band'] = pd.cut(df['MAX_WIND_SPD'], np.arange(10, 71, 5),
                    labels=np.arange(10, 70, 5))
df['lonband'] = pd.cut(df['LON'], np.arange(30, 250.1, 20),
                       labels=np.arange(40, 250, 20))


def fit_model(df):
    resdf = pd.DataFrame(columns=['i', 'au', 'av', 'aust', 'avst',
                                  'ursq', 'ufrsq', 'vrsq', 'vfrsq'])

    for i, x in enumerate(np.arange(10, 70, 5)):
        ddf = df[df['band'] == x]
        model = lmfit.Model(twolevel_model, independent_vars=['xl', 'xu'])
        u = ddf['u']  # Observed u component
        ul = ddf['u850']  # Low-level u component of environmental flow
        uu = ddf['u250']  # Upper-level u component of environmental flow
        v = ddf['v']
        vl = ddf['v850']
        vu = ddf['v250']
        uresult = rmod.fit(u, p, xl=ul, xu=uu)
        vresult = rmod.fit(v, p, xl=vl, xu=vu)
        ufresult = fmod.fit(u, p, xl=ul, xu=uu)
        vfresult = fmod.fit(v, p, xl=vl, xu=vu)

        resdf.loc[len(resdf.index)] = [
            x,
            uresult.params['alpha'].value,
            vresult.params['alpha'].value,
            uresult.params['alpha'].stderr,
            vresult.params['alpha'].stderr,
            uresult.rsquared,
            ufresult.rsquared,
            vresult.rsquared,
            vfresult.rsquared]
    return resdf


def fit_model_longitude(df):
    resdf = pd.DataFrame(columns=['i', 'au', 'av', 'aust', 'avst',
                                  'ursq', 'ufrsq', 'vrsq', 'vfrsq'])

    for i, x in enumerate(np.arange(40, 250.1, 20)):
        ddf = df[df['lonband'] == x]
        model = lmfit.Model(twolevel_model, independent_vars=['xl', 'xu'])
        u = ddf['u']  # Observed u component
        ul = ddf['u850']  # Low-level u component of environmental flow
        uu = ddf['u250']  # Upper-level u component of environmental flow
        v = ddf['v']
        vl = ddf['v850']
        vu = ddf['v250']
        uresult = rmod.fit(u, p, xl=ul, xu=uu)
        vresult = rmod.fit(v, p, xl=vl, xu=vu)
        ufresult = fmod.fit(u, p, xl=ul, xu=uu)
        vfresult = fmod.fit(v, p, xl=vl, xu=vu)

        resdf.loc[len(resdf.index)] = [
            x,
            uresult.params['alpha'].value,
            vresult.params['alpha'].value,
            uresult.params['alpha'].stderr,
            vresult.params['alpha'].stderr,
            uresult.rsquared,
            ufresult.rsquared,
            vresult.rsquared,
            vfresult.rsquared]
    return resdf


def fit_alpha_intensity(df):
    """
    Fit a simple linear regression to the resulting fits of $\alpha$ at
    different intensity:

    $\alpha(v) = m v + b$
    """

    lm = lmfit.Parameter(name='m', value=.01, max=0)
    lb = lmfit.Parameter(name='b', value=1, min=0)
    lp = lmfit.Parameters()
    lp.add(lm)
    lp.add(lb)

    # Fit the linear $\alpha(v)$ to wind speeds of 50 m/s or less.
    # There are fewer records with maximum wind speed $\ge$ 50 m/s:

    au = df['au'].values
    av = df['av'].values

    lmod = lmfit.Model(lin_model, independent_vars=['x'])
    lresult = lmod.fit((au[:-3]+av[:-3])/2, lp, x=np.arange(10, 55, 5))
    print(lresult.rsquared)
    print(lresult.best_fit.max(), lresult.best_fit.min())
    return lresult


# Plotting:
def plotResults(df, filename):
    x = np.arange(10, 70, 5)
    lresult = fit_alpha_intensity(df)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].errorbar(df['i'], df['au'], yerr=df['aust'],
                     color='b', capsize=5, label='Zonal')
    axes[0].errorbar(df['i'], df['av'], yerr=df['avst'],
                     color='r', capsize=5, label='Meridional')
    axes[0].plot(df['i'], (df['au']+df['av'])/2., color='k', label="Mean")
    axes[0].plot(x[:-3], lresult.best_fit, color='k',
                 linestyle='--', label="Linear fit")
    axes[0].grid(True)
    axes[0].set_ylim((0.0, 1.0))
    axes[0].set_ylabel(r"$\alpha(v)$")
    axes[0].legend(ncols=2)

    axes[1].plot(x, df['ursq'], color='b')
    axes[1].plot(x, df['vrsq'], color='r', label=r"$\alpha (v)$")
    axes[1].plot(x, df['ufrsq'], color='b',
                 linestyle='--', label=r"Constant $\alpha$")
    axes[1].plot(x, df['vfrsq'], color='r', linestyle='--')
    axes[1].grid(True)
    axes[1].set_ylim((-1, 1))
    axes[1].set_ylabel(r"$r^{2}$")
    axes[1].set_xlabel("Intensity [m/s]")
    axes[1].legend(ncols=2)
    plt.text(0.95, 0.05, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=fig.transFigure, ha='right')
    plt.savefig(os.path.join(BASEDIR, filename), bbox_inches='tight')


def plotModel(df, fitdf, filename):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for i, x in enumerate(np.arange(10, 70, 5)):
        ualpha = fitdf.loc[i, 'au']
        valpha = fitdf.loc[i, 'av']
        ddf = df[df['band'] == x]
        u = ddf['u']  # Observed u component
        ul = ddf['u850']  # Low-level u component of environmental flow
        uu = ddf['u250']  # Upper-level u component of environmental flow
        v = ddf['v']
        vl = ddf['v850']
        vu = ddf['v250']

        upred = ualpha*ul + (1 - ualpha)*uu
        vpred = valpha*vl + (1 - valpha)*vu
        ax[0].plot(upred, u, 'o', label=f"{x} m/s")
        ax[1].plot(vpred, v, 'o')

    ax[0].set_ylim((-15, 15))
    ax[0].set_xlim((-15, 15))
    ax[0].set_xlabel(r"Predicted $u_t$ [m/s]")
    ax[0].set_ylabel(r"Observed $u_t$ [m/s]")
    ax[0].grid()
    ax[0].text(0.05, 0.95, r"$u_t$", transform=ax[0].transAxes,
               fontweight='bold', va='top')
    ax[1].set_ylim((-15, 15))
    ax[1].set_xlim((-15, 15))
    ax[1].grid()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_ylabel(r"Observed $v_t$ [m/s]")
    ax[1].set_xlabel(r"Predicted $v_t$ [m/s]")
    ax[1].text(0.05, 0.95, r"$v_t$", transform=ax[1].transAxes,
               fontweight='bold', va='top')
    plt.figlegend(bbox_to_anchor=(.5, 0), loc="lower center",
                  bbox_transform=fig.transFigure, ncol=4,
                  title="Storm intensity [m/s]")
    fig.subplots_adjust(bottom=0.25)
    plt.text(0.95, 0.025, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=fig.transFigure, ha='right', fontsize='xx-small')
    plt.savefig(os.path.join(BASEDIR, filename),
                bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for i, x in enumerate(np.arange(10, 70, 5)):
        ualpha = fitdf.loc[i, 'au']
        valpha = fitdf.loc[i, 'av']
        ddf = df[df['band'] == x]
        u = ddf['u']  # Observed u component
        ul = ddf['u850']  # Low-level u component of environmental flow
        uu = ddf['u250']  # Upper-level u component of environmental flow
        v = ddf['v']
        vl = ddf['v850']
        vu = ddf['v250']

        upred = ualpha*ul + (1 - ualpha)*uu
        vpred = valpha*vl + (1 - valpha)*vu

        magu = np.sqrt(u**2 + v**2)
        magupred = np.sqrt(upred**2 + vpred**2)
        ax.plot(magupred, magu, 'o', label=f"{x} m/s")
    ax.set_ylim((0, 20))
    ax.set_xlim((0, 20))
    ax.set_xlabel(r"Predicted $u_t$ [m/s]")
    ax.set_ylabel(r"Observed $u_t$ [m/s]")
    ax.grid()
    plt.figlegend(bbox_to_anchor=(.5, 0), loc="lower center",
                  bbox_transform=fig.transFigure, ncol=4,
                  title="Storm intensity [m/s]")
    fig.subplots_adjust(bottom=0.25)
    plt.text(0.95, 0.025, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=fig.transFigure, ha='right', fontsize='xx-small')
    plt.savefig(os.path.join(BASEDIR, "mag_"+filename),
                bbox_inches='tight')


def plotModelLongitude(df, fitdf, filename):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for i, x in enumerate(np.arange(40, 240.1, 20)):
        ualpha = fitdf.loc[i, 'au']
        valpha = fitdf.loc[i, 'av']
        ddf = df[df['lonband'] == x]
        u = ddf['u']  # Observed u component
        ul = ddf['u850']  # Low-level u component of environmental flow
        uu = ddf['u250']  # Upper-level u component of environmental flow
        v = ddf['v']
        vl = ddf['v850']
        vu = ddf['v250']

        upred = ualpha*ul + (1 - ualpha)*uu
        vpred = valpha*vl + (1 - valpha)*vu
        ax[0].plot(upred, u, 'o', label=f"{x}E")
        ax[1].plot(vpred, v, 'o')

    ax[0].set_ylim((-15, 15))
    ax[0].set_xlim((-15, 15))
    ax[0].set_xlabel(r"Predicted $u_t$ [m/s]")
    ax[0].set_ylabel(r"Observed $u_t$ [m/s]")
    ax[0].grid()
    ax[0].text(0.05, 0.95, r"$u_t$", transform=ax[0].transAxes,
               fontweight='bold', va='top')
    ax[1].set_ylim((-15, 15))
    ax[1].set_xlim((-15, 15))
    ax[1].grid()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_ylabel(r"Observed $v_t$ [m/s]")
    ax[1].set_xlabel(r"Predicted $v_t$ [m/s]")
    ax[1].text(0.05, 0.95, r"$v_t$", transform=ax[1].transAxes,
               fontweight='bold', va='top')
    plt.figlegend(bbox_to_anchor=(.5, 0), loc="lower center",
                  bbox_transform=fig.transFigure, ncol=4,
                  title=r"Longitude [$^{\circ}$E]")
    fig.subplots_adjust(bottom=0.25)
    plt.text(0.95, 0.025, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=fig.transFigure, ha='right', fontsize='xx-small')
    plt.savefig(os.path.join(BASEDIR, filename),
                bbox_inches='tight')


def plotResultsLongitude(df, filename):
    x = np.arange(40, 240.1, 20)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].errorbar(df['i'], df['au'], yerr=df['aust'],
                     color='b', capsize=5, label='Zonal')
    axes[0].errorbar(df['i'], df['av'], yerr=df['avst'],
                     color='r', capsize=5, label='Meridional')
    axes[0].plot(df['i'], (df['au']+df['av'])/2., color='k', label="Mean")

    axes[0].grid(True)
    axes[0].set_ylim((0.0, 1.0))
    axes[0].set_xlim((40, 240))
    axes[0].xaxis.set_major_locator(MultipleLocator(20))
    axes[0].xaxis.set_minor_locator(MultipleLocator(10))

    axes[0].set_ylabel(r"$\alpha(\lambda)$")
    axes[0].legend(ncols=2)

    axes[1].plot(x, df['ursq'], color='b')
    axes[1].plot(x, df['vrsq'], color='r', label=r"$\alpha (v)$")
    axes[1].plot(x, df['ufrsq'], color='b',
                 linestyle='--', label=r"Constant $\alpha$")
    axes[1].plot(x, df['vfrsq'], color='r', linestyle='--')
    axes[1].grid(True)
    axes[1].set_ylim((-1, 1))
    axes[1].set_ylabel(r"$r^{2}$")
    axes[1].set_xlabel(r"Longitude [$^{\circ}$E]")
    axes[1].legend(ncols=2)
    plt.text(0.95, 0.05, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=fig.transFigure, ha='right',
             fontsize='xx-small')
    plt.savefig(os.path.join(BASEDIR, filename), bbox_inches='tight')


fitdf = fit_model(df)
fitdf_lon = fit_model_longitude(df)
plot_scatter(df, filename="tcenvflow_scatter.png")
plotResults(fitdf, filename="tcenvflow_fit.png")
plotResultsLongitude(fitdf_lon, filename="tcenvflow_fit.longitude.png")
plotModel(df, fitdf, filename="tcenvflow_fullfit.png")
plotModelLongitude(df, fitdf_lon, filename="tcenvflow_fullfit.longitude.png")

df.drop("Unnamed: 0", axis=1).to_csv(
    os.path.join(BASEDIR, "tcenvflow.csv"), index=False)
fitdf.to_csv(
    os.path.join(BASEDIR, "tcenvflow.fitstats.csv"), index=False)
for basin in df['BASIN'].unique():
    print(basin)
    basinfit = fit_model(df[df['BASIN'] == basin])
    basinfit.to_csv(
        os.path.join(BASEDIR, f"tcenvflow.fitstats.{basin}.csv"),
        index=False)
    plot_scatter(df[df['BASIN'] == basin],
                 filename=f"tcenvflow_scatter.{basin}.png")

    plotResults(basinfit, filename=f"tcenvflow_fit.{basin}.png")
    plotModel(df[df['BASIN'] == basin], basinfit,
              filename=f"tcenvflow_fullfit.{basin}.png")
