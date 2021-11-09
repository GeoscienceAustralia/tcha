import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import fnmatch
import logging
from os.path import join as pjoin
from datetime import datetime, timedelta
from shapely.geometry import LineString
from shapely.geometry import box as sbox

from vincenty import vincenty
import numpy as np
import pandas as pd

from Utilities.metutils import convert


def filter_tracks_domain(df, minlon=90, maxlon=180, minlat=-40, maxlat=0,
                         idcode='num', latname='lat', lonname='lon'):
    """
    Takes a `DataFrame` and filters on the basis of whether the track interscts
    the given domain, which is specified by the minimum and maximum longitude and
    latitude.
    NOTE: This assumes the tracks and bounding box are in the same geographic
    coordinate system (i.e. generally a latitude-longitude coordinate system).
    It will NOT support different projections (e.g. UTM data for the bounds and
    geographic for the tracks).
    NOTE: This doesn't work if there is only one point for the track.
    :param df: :class:`pandas.DataFrame` that holds the TCLV data
    :param float minlon: minimum longitude of the bounding box
    :param float minlat: minimum latitude of the bounding box
    :param float maxlon: maximum longitude of the bounding box
    :param float maxlat: maximum latitude of the bounding box
    """

    domain = sbox(minlon, minlat, maxlon, maxlat, ccw=False)
    tracks = df.groupby(idcode)
    tempfilter = tracks.filter(lambda x: len(x) > 1)
    filterdf = tempfilter.groupby(idcode).filter(lambda x: LineString(zip(x[lonname], x[latname])).intersects(domain))
    return filterdf

def convertLatLon(strval: str) -> float:
    """
    Convert a string representing lat/lon values from '140S to -14.0, etc.
    :param str strval: string containing the latitude or longitude.
    :returns: Latitude/longitude as a float value.
    """
    hemi = strval[-1].upper()
    fval = float(strval[:-1]) / 10.
    if (hemi == 'S') | (hemi == 'W'):
        fval *= -1
    if (hemi == 'E') | (hemi == 'W'):
        fval = fval % 360
    return fval

def season(year: int, month: int) -> int:
    """
    Determine the southern hemisphere TC season based on the year and month value.
    If the month is earlier than June, we assign the season to be the preceding year.
    :params int year: Year
    :params int month: Month
    """
    s = year
    if month < 6:
        s = year - 1
    return int(s)



COLTYPES = ['|S2', 'i', datetime, 'i', '|S4', 'i', 'f', 'f', 'f', 'f',
            '|S4', 'f', '|S3', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
            '|S1', 'f', '|S3', 'f', 'f', '|S10', '|S1', 'f',
            '|S3', 'f', 'f', 'f', 'f']
COLUNITS = ['', '', '', '', '', '', '', '', 'kts', 'hPa',
            '', 'nm', '', 'nm', 'nm', 'nm', 'nm', 'hPa', 'nm', 'nm', 'kts', 'nm',
            '', '', '', 'degrees', 'kts', '', '', '',
            '', '', '', '', '']

DATEFORMAT = "%Y%m%d%H"
CONVERTERS = {
    'Number': lambda s: s.strip(' ,'),
    "Datetime": lambda s: datetime.strptime(s.strip(' ,'), DATEFORMAT),
    "Latitude": lambda s: float(convertLatLon(s.strip(' ,'))),
    "Longitude": lambda s: float(convertLatLon(s.strip(' ,'))),
    "Windspeed": lambda s: float(s.strip(' ,') or 0),
    "Pressure": lambda s: float(s.strip(' ,') or 0),
    "Status": lambda s: s.strip(' ,'),
    "RAD": lambda s: float(s.strip(' ,') or 0),
    "WINDCODE": lambda s: s.strip(' ,'),
    "RAD1": lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[13], 'km'),
    "RAD2": lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[14], 'km'),
    "RAD3": lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[15], 'km'),
    "RAD4": lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[16], 'km'),
    "Poci": lambda s: float(s.strip(' ,') or 0),
    "Roci": lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[18], 'km'),
    "rMax": lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[19], 'km'),
    "SPEED": lambda s: float(s.strip(' ,') or 0.0),
}


def loadFile(filename: str) -> pd.DataFrame:
    """
    Attempt to load a JTWC B-DECK format file. Older format files have fewer
    fields, so only require a subset of column names. If the full list of column
    names is used, a `ValueError` is raised when applying the converters.
    :param filename: Path to the file to load
    :returns: `pd.DataFrame` of the track contained in the file.
    """

    try:
        COLNAMES = ['BASIN','Number', 'Datetime', 'TECHNUM', 'TECH', 'TAU', 'Latitude', 'Longitude', 'Windspeed', 'Pressure',
            'Status', 'RAD', 'WINDCODE', 'RAD1', 'RAD2', 'RAD3', 'RAD4', 'Poci', 'Roci', 'rMax', 'GUSTS', 'EYE',
            'SUBREGION', 'MAXSEAS', 'INITIALS', 'DIR', 'SPEED', 'STORMNAME', 'DEPTH', 'SEAS',
            'SEASCODE', 'SEAS1', 'SEAS2', 'SEAS3', 'SEAS4']
        df = pd.read_csv(filename, names=COLNAMES, delimiter=",",
                         parse_dates=[2], infer_datetime_format=True,
                         skipinitialspace=True, converters=CONVERTERS,
                         error_bad_lines=False)
    except ValueError:
        try:
            COLNAMES = ['BASIN','Number', 'Datetime','TECHNUM', 'TECH', 'TAU',
                        'Latitude', 'Longitude', 'Windspeed','Pressure', '']
            df = pd.read_csv(filename, names=COLNAMES, delimiter=",",
                             parse_dates=[2], infer_datetime_format=True,
                             skipinitialspace=True, converters=CONVERTERS,
                             error_bad_lines=False)
        except ValueError:
            COLNAMES = ['BASIN','Number', 'Datetime','TECHNUM', 'TECH', 'TAU',
                        'Latitude', 'Longitude', 'Windspeed','Pressure', ]
            df = pd.read_csv(filename, names=COLNAMES, delimiter=",",
                             parse_dates=[2], infer_datetime_format=True,
                             skipinitialspace=True, converters=CONVERTERS,
                             usecols=range(10), error_bad_lines=False)

    return df


def load_jtwc_data(path):
    alltracks = []
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, "bsh*.txt") or fnmatch.fnmatch(file, "bsh*.dat"):
            tempdf = loadFile(pjoin(path, file))
            tempdf['eventid'] = os.path.splitext(file)[0]
            alltracks.append(tempdf)

    df = pd.concat(alltracks)
    df['dP'] = df.Poci - df.Pressure
    df = filter_tracks_domain(
        df, 90, 160, -40, 0, 'eventid', 'Latitude', 'Longitude'
    )

    # remove bad records
    filter_ = (df['Status'] == 'TS') | (df['Status'] == 'TY')
    filter_ &= (df['rMax'] > 0.0)
    filter_ &= (df['Poci'] > 0.0)
    filter_ &= (df['dP'] > 0.0)
    filter_ &= (df['RAD'] == 34) | (df['RAD'] == 35)
    df = df[filter_].copy()

    # create time variables
    df['year'] = pd.DatetimeIndex(df['Datetime']).year
    df['month'] = pd.DatetimeIndex(df['Datetime']).month
    df['season'] = df[['year', 'month']].apply(lambda x: season(*x), axis=1)

    # calculate r34 as mean of quadrants
    # if any quadrant is missing set mean to zero
    rads = [f"RAD{i}" for i in range(1, 5)]
    mask = (df.RAD > 0) & (df[rads].values.min(axis=1) > 0)

    df['r34'] = df[rads].values.mean(axis=1)
    df['r34'].values[~mask] = np.nan

    # delete duplicates and sort by TC event and chronological order
    # enables time differencing
    df.drop_duplicates(['eventid', 'Datetime'], inplace=True)
    df.sort_values(by=['eventid', 'Datetime'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['new_index'] = np.arange(len(df))
    idxs = df.groupby(['eventid']).agg({'new_index': np.min}).values
    df.drop('new_index', axis=1, inplace=True)

    # translation speed is calculated via backward difference
    # using the vincenty distance which is probably overkill
    dt = np.diff(df.Datetime) / (3600 * np.timedelta64(1, 's'))  # in hours
    coords = df[["Latitude", "Longitude"]].values
    dists = [vincenty(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
    speed = np.zeros(len(df))
    speed[1:] = np.array(dists) / dt

    # translation speed of the last record of each TC is set to 0
    speed[idxs] = 0
    df['translation_speed'] = speed

    return df