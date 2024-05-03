import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyproj
geodesic = pyproj.Geod(ellps="WGS84")
DATA_DIR=r"X:\georisk\HaRIA_B_Wind\data\raw\from_noaa\ibtracs\v04r00"

def savefig(filename, *args, **kwargs):
    """
    Add a timestamp to each figure when saving

    :param str filename: Path to store the figure at
    :param args: Additional arguments to pass to `plt.savefig`
    :param kwargs: Additional keyword arguments to pass to `plt.savefig`
    """
    fig = plt.gcf()
    plt.text(0.99, 0.0, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
            transform=fig.transFigure, ha='right', va='bottom',
            fontsize='xx-small')
    plt.savefig(filename, *args, **kwargs)

def load_ibtracs_df(basins=None, season=None):
    """
    Helper function to load the IBTrACS database.
    By default, we use the BOM_WIND

    :param int season: Season to filter data by
    :param list basins: select only those TCs from the given basins

    """
    dataFile = os.path.join(DATA_DIR, "ibtracs.since1980.list.v04r00.csv")
    df = pd.read_csv(
        dataFile,
        skiprows=[1],
        usecols=[0, 1, 3, 5, 6, 8, 9, 10, 11, 13, 23, 95],
        keep_default_na=False,
        na_values=[" "],
        parse_dates=[1],
        date_format="%Y-%m-%d %H:%M:%S",
    )
    df.rename(
        columns={
            "SID": "DISTURBANCE_ID",
            "ISO_TIME": "TM",
            "WMO_PRES": "CENTRAL_PRES",
            "BOM_WIND": "MAX_WIND_SPD"},
        inplace=True,
    )

    df["TM"] = pd.to_datetime(
        df.TM, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df[~pd.isnull(df.TM)]

    # Filter to every 6 hours:
    df["hour"] = df["TM"].dt.hour
    df = df[df["hour"].isin([0, 6, 12, 18])]
    df.drop(columns=["hour"], inplace=True)

    # Move longitudes to [0, 360)
    df.loc[df['LON'] < 0, "LON"] = df['LON'] + 360

    # Filter by season if given:
    df['SEASON'] = df['SEASON'].astype(int)
    if season:
        df = df[df.SEASON == season]

    # IBTrACS includes spur tracks (bits of tracks that are
    # different to the official) - these need to be dropped.
    df = df[df.TRACK_TYPE.isin(["main", "PROVISIONAL"])]

    # Filter by basins if given
    if basins:
        df = df[df["BASIN"].isin(basins)]

    # Fill any missing wind speed reports with data from US records
    # We have to convert from a 1-minute sustained wind to a
    # 10-minute mean, using the conversions described in WMO TD1555 (2010).
    # NOTE:
    # 1) I don't handle the case of New Dehli, which uses a 3-minute mean
    # wind speed.
    # 2) Returns wind speeds in knots (rounded to 2 decimal places)

    df.fillna({"MAX_WIND_SPD": df["WMO_WIND"]}, inplace=True)
    df.fillna({"MAX_WIND_SPD": df["USA_WIND"] / 1.029}, inplace=True)
    df["MAX_WIND_SPD"] = np.round(df["MAX_WIND_SPD"], 2)

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