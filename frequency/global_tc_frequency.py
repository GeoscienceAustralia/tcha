"""
Calculate TC frequency in each major TC basin, based on IBTrACS
data. This uses Monte Carlo-Markov Chain modelling to evaluate the
mean rate of TC occurrence, under the assumption of a Poisson distribution
and no trend in frequency. This latter assumption may not hold in all
basins due to changes in observing practices (e.g. Indian Ocean).

"""

import os
from os.path import join as pjoin
from datetime import datetime
import pandas as pd
import xarray as xr
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


import pyproj

geodesic = pyproj.Geod(ellps="WGS84")


DATA_DIR = r"C:\WorkSpace\data\tc"
OUTPUT_DIR = r"..\data\frequency"
SOURCE = "https://www.ncei.noaa.gov/products/international-best-track-archive"


def load_ibtracs_df():
    """
    Helper function to load the IBTrACS database.
    Column names are mapped to the same as the BoM dataset to minimise
    the changes elsewhere in the code

    """
    dataFile = os.path.join(DATA_DIR, "ibtracs.since1980.list.v04r00.csv")
    df = pd.read_csv(
        dataFile,
        skiprows=[1],
        usecols=[0, 1, 3, 5, 6, 8, 9, 11, 13, 23],
        keep_default_na=False,
        na_values=[" "],
        parse_dates=[1],
        date_format="%Y-%m-%d %H:%M:%S",
    )
    df.rename(
        columns={
            "SID": "DISTURBANCE_ID",
            "ISO_TIME": "TM",
            "WMO_WIND": "MAX_WIND_SPD",
            "WMO_PRES": "CENTRAL_PRES",
            "USA_WIND": "MAX_WIND_SPD",
        },
        inplace=True,
    )

    df["TM"] = pd.to_datetime(
        df.TM, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df[~pd.isnull(df.TM)]

    # Filter to every 6 hours (to match sub-daily ERA data)
    df["hour"] = df["TM"].dt.hour
    df = df[df["hour"].isin([0, 6, 12, 18])]
    df.drop(columns=["hour"], inplace=True)
    df["SEASON"] = df["SEASON"].astype(int)
    # df = df[df.SEASON == season]

    # IBTrACS includes spur tracks (bits of tracks that are
    # different to the official) - these need to be dropped.
    df = df[df.TRACK_TYPE == "main"]

    df.reset_index(inplace=True)
    fwd_azimuth, _, distances = geodesic.inv(
        df.LON[:-1],
        df.LAT[:-1],
        df.LON[1:],
        df.LAT[1:],
    )

    df["new_index"] = np.arange(len(df))
    idxs = df.groupby(["DISTURBANCE_ID"]).agg(
        {"new_index": "max"}).values.flatten()
    df.drop("new_index", axis=1, inplace=True)
    # Convert max wind speed to m/s for consistency
    df["MAX_WIND_SPD"] = df["MAX_WIND_SPD"] * 0.5144

    dt = np.diff(df.TM).astype(float) / 3_600_000_000_000
    u = np.zeros_like(df.LAT)
    v = np.zeros_like(df.LAT)
    v[:-1] = np.cos(fwd_azimuth * np.pi / 180) * distances / (dt * 1000) / 3.6
    u[:-1] = np.sin(fwd_azimuth * np.pi / 180) * distances / (dt * 1000) / 3.6

    v[idxs] = 0
    u[idxs] = 0
    df["u"] = u
    df["v"] = v

    dt = np.diff(df.TM).astype(float) / 3_600_000_000_000
    dt_ = np.zeros(len(df))
    dt_[:-1] = dt
    df["dt"] = dt_

    df = df[df.u != 0].copy()
    print(f"Number of records: {len(df)}")
    return df


def calculateFrequency(df):
    # Calculate the number of unique values in each season:
    ntcs = df.groupby(["SEASON"]).nunique()
    # This just ensures we capture the seasons with a zero count
    stormcount = pd.Series(
        index=range(ntcs.index.min(), ntcs.index.max() + 1), dtype=int, data=0
    )
    stormcount.loc[ntcs.index] = np.array(
        ntcs.DISTURBANCE_ID.values, dtype="int32")
    return stormcount


def fitFrequency(tccount):
    """
    Calculate mean frequency (rate parameter for Poisson distribution)

    :param tccount: `pd.DataFrame` of TC counts for a basin

    """
    basinname = tccount.name[0]
    print(f"Fitting frequency for {basin} storms")
    years = (tccount.index - tccount.index[0]).values
    with pm.Model() as mmodel:
        lambda_ = pm.Normal(
            "lambda", mu=tccount.mean(),
            sigma=np.std(tccount))
        observation = pm.Poisson("obs", lambda_, observed=tccount.values)
        step = pm.Metropolis()
        mtrace = pm.sample(
            20000, tune=10000, step=step,
            return_inferencedata=True, chains=4, cores=1
        )
        mtrace.extend(pm.sample_posterior_predictive(mtrace))
    mtrace.posterior["ymodel"] = mtrace.posterior["lambda"] * xr.DataArray(
        np.ones(len(years))
    )
    _, ax = plt.subplots(figsize=(12, 6))
    az.plot_lm(
        idata=mtrace,
        y=tccount,
        x=years,
        num_samples=100,
        axes=ax,
        y_model="ymodel",
        kind_pp="hdi",
        kind_model="hdi",
        y_model_mean_kwargs={"lw": 2, "color": "g", "ls": "--"},
        y_kwargs={"marker": None, "color": "0.75", "label": "_obs"},
        y_model_fill_kwargs={"alpha": 0.25},
        y_hat_fill_kwargs={"hdi_prob": 0.95},
    )
    ax.bar(years, tccount.values, fc="0.9", ec="k", zorder=0)
    ax.legend(loc=1)
    ax.set_xticks(np.arange(-1, 41, 5))
    ax.set_xticklabels(np.arange(1980, 2022, 5))
    ax.set_xlabel("Season")
    ax.set_ylabel("TC count")
    meanrate = mtrace.posterior["lambda"].mean()
    stdrate = mtrace.posterior["lambda"].std()
    titlestr = rf"{meanrate:.1f} $\pm$ {stdrate:.2f}"
    ax.set_title(f"Mean frequency - {basin} basin ({titlestr})")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    sns.despine()
    plt.text(
        0.0,
        -0.1,
        f"Source: {SOURCE}",
        transform=ax.transAxes,
        fontsize="xx-small",
        ha="left",
    )
    plt.text(
        1.0,
        -0.1,
        f"Created: {datetime.now():%Y-%m-%d %H:%M}",
        transform=ax.transAxes,
        fontsize="xx-small",
        ha="right",
    )
    plt.savefig(
        pjoin(
            OUTPUT_DIR, f"{basinname}.seasonal_frequency_posterior_predictive_mean.png"  # noqa
        ),
        bbox_inches="tight",
    )
    print(f"Saving samples for {basinname}")
    mtrace.posterior_predictive["obs"][0].to_dataframe().to_csv(
        pjoin(OUTPUT_DIR, f"tccount.samples.{basinname}.csv")
    )
    fig, ax = plt.subplots(1, 1)
    xmax = int(
        mtrace.posterior_predictive["obs"].values.flatten().max() / 5 + 1) * 5
    ax.hist(
        mtrace.posterior_predictive["obs"].values.flatten(),
        bins=np.arange(0, xmax, 1),
        density=True,
        alpha=0.75,
        width=0.75,
        label="Sampled",
    )
    ax.hist(
        tccount.values,
        bins=np.arange(0, xmax, 1),
        density=True,
        ec="k",
        fc="white",
        alpha=0.5,
        width=0.5,
        label="Observed",
    )
    ax.set_xlabel("Seasonal TC count")
    ax.legend()
    ax.text(
        1.0,
        -0.1,
        f"Created: {datetime.now():%Y-%m-%d %H:%M}",
        transform=ax.transAxes,
        fontsize="xx-small",
        ha="right",
    )
    plt.savefig(
        pjoin(OUTPUT_DIR, f"{basinname}.seasonal_frequency_pp_sample.png"),
        bbox_inches="tight",
    )


df = load_ibtracs_df()
basins = df["BASIN"].unique()

tcdf = pd.DataFrame(columns=[basins])

for basin in basins:
    dfb = df[df["BASIN"] == basin]
    ntcs = calculateFrequency(dfb)
    tcdf[basin] = ntcs

tccount = tcdf.loc[1981:]  # Only analysing data from 1981 onwards
tccount.to_csv(pjoin(OUTPUT_DIR, "tccount.basins.csv"))
for basin in basins:
    fitFrequency(tccount[basin].squeeze())

print("Global TC frequency")
gtcs = calculateFrequency(df)

tcdf["GL"] = gtcs
fitFrequency(tcdf["GL"].squeeze())
