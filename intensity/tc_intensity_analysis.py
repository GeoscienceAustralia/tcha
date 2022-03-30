import hurr
from era5_extract import load_otcr_df
import os
import numpy as np
import pandas as pd
import time
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units

print("Done imports")

def pytc_intensity(vm, rm, r0, ts, h_a, alat, ahm, pa, tend, hm=30.0, dsst=0.6, gm=8.0):
    """

    Parameters
    ----------
    vm : maximum wind speed (m/s)
    rm : radius of maximum winds (km)
    r0 : radius of zero wind speed (km)
    ts : sea surface temperature (C)
    h_a : undisturbed humidity of environmental air near the surface (percent)
    alat : latitude (degrees)
    nr : number of radial nodes
    dt : time step (s)
    ahm : undisturbed relative humidity of undisturbed lower and middle troposphere
    pa : undisturbed sea surface pressure (hPa)
    hm : undisturbed ocean mixed layer depth (m)
    dsst : temperature jump at base of mixed layer (C)
    gm : Sub mixed layer ocean thermal stratification (C/100m)

    Returns
    -------

    """
    nrd = 200

    om = 'n'  # ocean mixing on
    vdisp = 's'  # plotting param
    dtg = 3  # plotting param
    rog = 0  # plotting param
    rst = 'n'   # plotting param
    to = -70  # environmental temp at top of tc Celsius
    tshear = 200  # time until shear days
    vext = 0  # wind shear m/s
    tland = 200  # time until land
    surface = 'pln'  # type of land
    hs = 0  # swamp depth
    ut = 7  # speed over ocean m/s
    eddytime = 200  # time until storm reaches eddy
    heddy = 30  # peak ocean eddy m
    rwide = 400  # radius km
    dim = 'y'  # dimensional output
    fmt = 'tab'  # tabs
    ro = 1200  # radius of model boundary km
    cd = 0.8  # drag coefficient
    cd1 = 4  # drag coefficient rate of change w/ windspeed
    cdcap = 3  # max drag
    cecd = 1  # ratio of drag coefficients
    pnu = 0.03  # turbulent mixing length
    taur = 8  # radiative relaxation time scale
    radmax = 1.5  # max rad cooling rate
    tauc = 2  # convective relation time scale
    efrac = 0.5  # fraction of convective entropy detrained into lower
    dpb = 50  # boundary layer depth
    nr = 150 # numer of radial nodes
    dt = 20 # time step in seconds

    return hurr.tc_intensity(
        nrd, tend, vdisp, dtg, rog, rst, vm,
        rm, r0, ts, to, h_a, alat, tshear, vext, tland, surface,
        hs, om, ut, eddytime, heddy, rwide, dim, fmt, nr,
        dt, ro, ahm, pa, cd, cd1, cdcap,
        cecd, pnu, taur, radmax, tauc, efrac, dpb, hm, dsst, gm
    )


if __name__ == "__main__":
    DATA_DIR = os.path.expanduser("~")    
    
    df = load_otcr_df()

    era5 = np.load(os.path.join(DATA_DIR, "tc_intensity_era5.npy"))
    bran = np.load(os.path.join(DATA_DIR, "tc_intensity_bran2020.npy"))

    for i, var in enumerate(["sst", "sp", "d2", "t2"]): df[var] = era5[:, i]
    df["hm"] = bran[:, 0]

    df = df[pd.isnull(df).any(axis=1)].copy()
    df['pc'] = np.zeros(len(df)) * np.nan

    t0  = time.time()
    for name, g in df.groupby('DISTURBANCE_ID'): 

        for j, i in enumerate(g.index[:-1]):
            row = g.loc[i]

            sst, sp, d2, t2, hm, lat = row.sst, row.sp, row.d2, row.t2, row.hm, row.LAT
            vm = row["adj. ADT Vm (kn)"]

            sst = sst - 273.15  # convert K -> C
            sp = sp / 100  # convert Pa -> hPa
            vm = vm * 0.514444  # convert knots to m/s
            t2 = units.Quantity(t2, "K")
            d2 = units.Quantity(d2, "K")

            h_a = relative_humidity_from_dewpoint(t2, d2) * 100 # relative humidity near surface
            tend = g.loc[g.index[j + 1]].TM - row.TM
            tend = tend.days + tend.seconds / (3600 * 24)  # simulation time in days

            # typical values
            # TODO: reuse vm, rm, and r0 from previous simulation step
            rm = 80
            r0 = 350
            ahm = 45  # relative humidity in tropo

            pytc_intensity(vm, rm, r0, sst, h_a, lat, ahm, sp, tend)

        break

    print(len(g) / len(df))
    print("time: ", time.time() - t0, "s")
