import hurr
from era5_extract import load_otcr_df
import os
import numpy as np
import pandas as pd
import time
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units
from matplotlib import pyplot as plt

print("Done imports")


# TODO: fix time step

class Hurricane:

    def __init__(self):
        self.rbs1, self.rts1, self.x1, self.xs1, self.xm1, self.mu1 = [np.zeros(200, dtype=np.float32) for _ in
                                                                       range(6)]
        self.rbs2, self.rts2, self.x2, self.xs2, self.xm2, self.mu2, self.ps2, self.ps3 = [np.zeros(200, dtype=np.float32) for _
                                                                                 in range(8)]
        self.init = 'y'

    def pytc_intensity(self, vm, rm, r0, ts, h_a, alat, ahm, pa, tend, hm=30.0, dsst=0.6, gm=8.0):
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
        nr = 50  # numer of radial nodes
        dt = 20  # time step in seconds

        # normalise
        h_a = 0.01 * h_a
        ahm = 0.01 * ahm
        es = 6.112 * np.exp(17.67 * ts / (243.5 + ts))
        ea = h_a * es
        qs = 0.622 * es / pa
        tsa = ts + 273.15
        toa = to + 273.15
        ef = (ts - to) / tsa

        chi = 2.5e6 * ef * qs * (1. - h_a)
        f = (3.14159 / (12. * 3600.)) * np.sin(3.14159 * abs(alat) / 180.)
        for arr in (self.x1, self.xs1, self.xm1, self.x2, self.xs2):
            arr /= chi

        for arr in (self.rbs1, self.rts1, self.rbs2, self.rts2):
            arr /= np.sqrt(chi) / f

        for arr in (self.mu1, self.mu2):
            arr /= 0.001 * cd * np.sqrt(chi)

        for arr in (self.ps2, self.ps3):
            arr /= 0.5 * 0.001 * cd * 9.81 * chi ** (3 / 2) / f ** 2

        h_a *= 100
        ahm *= 100
        out = np.zeros(3, dtype=np.float32)
        hurr.tc_intensity(
            nrd, tend, vm,
            rm, r0, ts, to, h_a, alat, tshear, vext, tland, surface,
            hs, om, ut, eddytime, heddy, rwide, nr,
            dt, ro, ahm, pa, cd, cd1, cdcap,
            cecd, pnu, taur, radmax, tauc, efrac, dpb, hm, dsst, gm, out,
            self.rbs1, self.rts1, self.x1, self.xs1, self.xm1, self.mu1,
            self.rbs2, self.rts2, self.x2, self.xs2, self.xm2, self.mu2,
            self.ps2, self.ps3, self.init
        )

        self.init = 'n'
        for arr in (self.x1, self.xs1, self.xm1, self.x2, self.xs2):
            arr *= chi

        for arr in (self.rbs1, self.rts1, self.rbs2, self.rts2):
            arr *= np.sqrt(chi) / f

        for arr in (self.mu1, self.mu2):
            arr *= 0.001 * cd * np.sqrt(chi)

        for arr in (self.ps2, self.ps3):
            arr *= 0.5 * 0.001 * cd * 9.81 * chi ** (3 / 2) / f ** 2

        return out


if __name__ == "__main__":
    DATA_DIR = os.path.expanduser("~/geoscience/data")

    df = load_otcr_df(DATA_DIR)

    era5 = np.load(os.path.join(DATA_DIR, "tc_intensity_era5.npy"))
    bran = np.load(os.path.join(DATA_DIR, "tc_intensity_bran2020.npy"))
    rh = np.load(os.path.join(DATA_DIR, "tc_intensity_rh.npy"))

    for i, var in enumerate(["sst", "sp", "d2", "t2"]): df[var] = era5[:, i]
    df["hm"] = bran[:, 0]

    groups = df.groupby('DISTURBANCE_ID')
    df['pc'] = np.zeros(len(df)) * np.nan

    t0 = time.time()
    for name, g in df.groupby('DISTURBANCE_ID'):
        hurricane = Hurricane()
        vm = np.nan
        r0 = 350  # do some sort of sampling
        rm = 80  # do some sort of sampling
        for j, i in enumerate(g.index[:-1]):
            row = g.loc[i]

            sst, sp, d2, t2, hm, lat = row.sst, row.sp, row.d2, row.t2, row.hm, row.LAT

            if np.isnan(vm):
                vm = row["adj. ADT Vm (kn)"] * 0.514444  # convert knots to m/s
                if np.isnan(vm): continue

            sst = sst - 273.15  # convert K -> C
            sp = sp / 100  # convert Pa -> hPa
            t2 = units.Quantity(t2, "K")
            d2 = units.Quantity(d2, "K")

            h_a = relative_humidity_from_dewpoint(t2, d2) * 100  # relative humidity near surface
            tend = g.loc[g.index[j + 1]].TM - row.TM
            tend = tend.days + tend.seconds / (3600 * 24)  # simulation time in days

            # typical values
            ahm = 90  # relative humidity in tropo

            # typical values
            vm_actual = g.loc[g.index[j + 1]]["adj. ADT Vm (kn)"] * 0.514444
            out = hurricane.pytc_intensity(vm, rm, r0, sst, h_a, abs(lat), ahm, sp, tend)
            pmin, vm, rm = out[0], out[1], out[2]
            print("Output:", vm, vm_actual, "\n")
        break

    print("Time: ", (time.time() - t0), "s")
