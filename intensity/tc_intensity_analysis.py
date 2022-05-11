from hurr import tc_intensity
from era5_extract import load_otcr_df
import os
import numpy as np
from numpy import log, exp, sqrt
import pandas as pd
import time
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units
from global_land_mask import globe
from matplotlib import pyplot as plt

import sys

TCRM_PATH = os.path.expanduser("~/geoscience/repos/tcrm")
sys.path.append(TCRM_PATH)
from wind.windmodels import PowellWindProfile
from PressureInterface.pressureProfile import PrsProfile
from TrackGenerator.trackSize import rmax
from TrackGenerator.TrackGenerator import SamplePressure
from Utilities.loadData import getPoci

print("Done imports")

OCEAN_MIXING = 'n'



class Hurricane:

    def __init__(self, dt=1, efrac=0.5, cecd=1.0, ocean_mixing=False, use_shear=False):
        self.rbs1, self.rts1, self.x1, self.xs1, self.xm1, self.mu1 = [
            np.zeros(200, dtype=np.float32) for _ in range(6)
        ]
        self.rbs2, self.rts2, self.x2, self.xs2, self.xm2, self.mu2 = [
            np.zeros(200, dtype=np.float32) for _ in range(6)
        ]
        self.uhmix1, self.uhmix2, self.sst1, self.sst2, self.hmix = [
            np.zeros(200, dtype=np.float32) for _ in range(5)
        ]
        self.init = 'y'
        self.diagnostic = np.zeros(200, dtype=np.float32)

        self.dt = dt
        self.efrac = efrac
        self.cecd = cecd
        self.use_shear = use_shear
        self.ocean_mixing = ocean_mixing

    def simulate(self, g, verbose=False):
        row = g.iloc[0]

        sst, sp, hm, lat, ahm, ut, hm, h_a = self.extract_environment(row)
        vm = row["Vmax (kn)"] * 0.514444  # convert knots to m/s

        dP = find_dP(vm, lat)
        rm = rmax(dP, lat, eps=0)
        f = (3.14159 / (12. * 3600.)) * np.sin(3.14159 * abs(lat) / 180.)
        rm *= 1000 * f
        r0 = 1.2 * sqrt(rm * rm * (1. + 2. * vm / rm))
        r0 /= 1000 * f
        rm /= 1000 * f

        if verbose:
            print("Starting:", vm, rm)
        out_rows = []
        any_bailed = False
        diagnostic_info = []
        for j, i in enumerate(g.index[:-1]):
            row = g.loc[i]
            sst, sp, hm, lat, ahm, ut, hm, h_a, dsst, gm, max_depth, shear = self.extract_environment(row)

            if np.isnan(hm) or globe.is_land(row.LAT, row.LON):
                break

            tend = g.loc[g.index[j + 1]].TM - row.TM
            tend = tend.days + tend.seconds / (3600 * 24)  # simulation time in days
            vm_actual = g.loc[g.index[j + 1]]["Vmax (kn)"] * 0.514444
            pmin_actual = g.loc[g.index[j + 1]]['CENTRAL_PRES']
            lat_next, lon_next = g.loc[g.index[j + 1]].LAT, g.loc[g.index[j + 1]].LON

            if j == 0:
                bailed, out = self.pytc_intensity(
                    vm, rm, r0, sst, h_a, abs(lat), ahm, sp, 2.0, ut, hm=hm, match='y', vobs=vm, gamma=100_000
                )
                pmin, vm, rm = out

            bailed, out = self.pytc_intensity(vm, rm, r0, sst, h_a, abs(lat), ahm, sp, tend, ut, hm=hm, gm=gm, dsst=dsst, shear=shear)
            any_bailed |= bailed

            pmin, vm, rm = out[0], out[1], out[2]
            diagnostic_info.append([vm, rm, sst, float(h_a), ahm, tend])

            out_rows.append(
                [
                    g.loc[g.index[j + 1]].TM, row.DISTURBANCE_ID, lat_next,
                    lon_next, pmin, vm, rm, pmin_actual, vm_actual
                ]
            )

        out_df = pd.DataFrame(
            out_rows,
            columns=["time", "DISTURBANCE_ID", "lat", "lon", "pmin", "vmax", "rmax", "pmin_obs", "vmax_obs"]
        )

        if verbose:
            diagnostic_info = pd.DataFrame(diagnostic_info)
            diagnostic_info.columns = ["vm", "rm", "sst", "ssh", "troph", "tend"]
            print("Final vm:", vm)
            print(diagnostic_info)
            print("\n" * 3)
            print(out_df)

        return any_bailed, out_df

    @staticmethod
    def extract_environment(row):
        sst, sp, d2, t2, hm, lat = row.sst, row.sp, row.d2, row.t2, row.hm, row.LAT
        dsst, gm, max_depth, shear = row.dsst, row.gm, row.max_depth, row.shear
        ahm, ut, hm = row.rh, row.vfm, row.hm

        sst = sst - 273.15  # convert K -> C
        sp = sp / 100  # convert Pa -> hPa
        t2 = units.Quantity(t2, "K")
        d2 = units.Quantity(d2, "K")
        h_a = relative_humidity_from_dewpoint(t2, d2) * 100  # relative humidity near surface
        ut = ut / 3.6  # convert km/h -> m/s

        return sst, sp, hm, lat, ahm, ut, hm, h_a, dsst, gm, max_depth, shear

    def pytc_intensity(
            self, vm, rm, r0, ts, h_a, alat, ahm, pa, tend,
            ut, hm=30.0, dsst=0.6, gm=8.0, match='n', vobs=0, gamma=1000,
            max_depth=np.inf, shear=0
    ):
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
        us :
        hm : undisturbed ocean mixed layer depth (m)
        dsst : temperature jump at base of mixed layer (C)
        gm : Sub mixed layer ocean thermal stratification (C/100m)

        Returns
        -------

        """
        nrd = 200

        om = self.ocean_mixing and (max_depth >= 200) # ocean mixing on
        to = -75  # environmental temp at top of tc Celsius
        tshear = 0  # time until shear days
        vext = shear if self.use_shear else 0.0  # wind shear m/s
        tland = 200  # time until land
        surface = 'pln'  # type of land
        hs = 0  # swamp depth
        ro = 1200  # radius of model boundary km
        cd = 0.8  # drag coefficient
        cd1 = 4  # drag coefficient rate of change w/ windspeed
        cdcap = 3  # max drag
        cecd = self.cecd  # ratio of drag coefficients
        pnu = 0.03  # turbulent mixing length
        taur = 8  # radiative relaxation time scale
        radmax = 1.5  # max rad cooling rate
        tauc = 2  # convective relation time scale
        efrac = self.efrac  # fraction of convective entropy detrained into lower
        dpb = 50  # boundary layer depth
        nr = 75  # numer of radial nodes
        dt = self.dt  # time step in seconds

        # normalise
        h_a *= 0.01
        ahm *= 0.01
        cd *= 0.001
        gm *= 0.01

        es = 6.112 * np.exp(17.67 * ts / (243.5 + ts))
        qs = 0.622 * es / pa
        tsa = ts + 273.15
        toa = to + 273.15
        ef = (ts - to) / tsa

        ric = 1.0

        chi = 2.5e6 * ef * qs * (1. - h_a)
        f = (3.14159 / (12. * 3600.)) * np.sin(3.14159 * abs(alat) / 180.)
        atha = 1000. * log(tsa) + 2.5e6 * qs * h_a / tsa - qs * h_a * 461. * log(h_a)
        theta_e = exp((atha - 287. * log(0.001 * pa)) * 0.001)
        pt = 1000.0 * (toa / theta_e) ** 3.484
        delp = 0.5 * (pa - pt)
        tm = 0.85 * ts + 0.15 * to
        esm = 6.112 * exp(17.67 * tm / (243.5 + tm))
        qsm = 0.622 * esm / (pa - 0.5 * delp)
        #beta = chi / (287. * tsa)
        #
        #
        amixfac = 0.5 * 1000. * ef * gm * (1. + 2.5e6 * 2.5e6 * qsm / (1000. * 461. * tsa * tsa)) / chi
        amix = (2. * ric * chi / (9.8 * 3.3e-4 * gm)) * 1.0e-6 * (287. * tsa / 9.8) ** 2 * (delp / pa) ** 2 * amixfac ** 4
        amix = sqrt(amix)

        for arr in (self.x1, self.xs1, self.xm1, self.x2, self.xs2, self.xm2):
            arr /= chi

        for arr in (self.rbs1, self.rts1, self.rbs2, self.rts2):
            arr /= np.sqrt(chi) / f

        for arr in (self.mu1, self.mu2):
            arr /= cd * np.sqrt(chi)

        self.hmix *= amixfac
        self.uhmix1 *= (amixfac ** 2) / (amix * np.sqrt(gm))
        self.uhmix2 *= (amixfac ** 2) / (amix * np.sqrt(gm))
        #
        h_a /= 0.01
        ahm /= 0.01
        cd /= 0.001
        gm /= 0.01

        out = np.zeros(4, dtype=np.float32)
        tc_intensity(
            nrd, tend, vm,
            rm, r0, ts, to, h_a, alat, tshear, vext, tland, surface,
            hs, om, ut, nr, dt, ro, ahm, pa, cd, cd1, cdcap,
            cecd, pnu, taur, radmax, tauc, efrac, dpb, hm, dsst, gm, out,
            self.rbs1, self.rts1, self.x1, self.xs1, self.xm1, self.mu1,
            self.rbs2, self.rts2, self.x2, self.xs2, self.xm2, self.mu2,
            self.uhmix1, self.uhmix2, self.sst1, self.sst2, self.hmix, self.init,
            match, vobs, gamma, self.diagnostic
        )
        bailed = (out[3] == 1) or np.isnan(self.rbs1).any()

        out = out[:3]
        self.init = 'n'
        h_a *= 0.01
        ahm *= 0.01
        cd *= 0.001
        gm *= 0.01

        for arr in (self.x1, self.xs1, self.xm1, self.x2, self.xs2, self.xm2):
            arr *= chi

        for arr in (self.rbs1, self.rts1, self.rbs2, self.rts2):
            arr *= np.sqrt(chi) / f

        for arr in (self.mu1, self.mu2):
            arr *= cd * np.sqrt(chi)

        self.hmix /= amixfac
        self.uhmix1 /= (amixfac ** 2) / (amix * np.sqrt(gm))
        self.uhmix2 /= (amixfac ** 2) / (amix * np.sqrt(gm))

        return bailed, out


def find_dP(vmax, lat):
    a = 1.881093 - 0.010917 * abs(lat)
    b = -0.005567 * 0.539957 * np.exp(4.22 + 0.0023 * abs(lat))

    vm2 = vmax ** 2
    dP = 10
    for _ in range(10):
        beta = a + b * np.exp(-0.0198 * dP)
        dP = vm2 * (np.exp(1) * 1.15) / (100 * beta)

    return dP


def fit_profile(temp, depth, mld):
    ml_mask = depth <= mld
    dl_mask = (~ml_mask) & (depth <= mld + 200)
    gm, *_ = linregress(depth[dl_mask], temp[dl_mask])

    idx = np.where(ml_mask)[0][-1]
    dsst = temp[idx] - temp[idx + 1]

    max_depth = depth[~np.isnan(temp)].max()

    return gm, dsst, max_depth


if __name__ == "__main__":

    pres_lvls = np.array([ 1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,
        125,  150,  175,  200,  225,  250,  300,  350,  400,  450,  500,
        550,  600,  650,  700,  750,  775,  800,  825,  850,  875,  900,
        925,  950,  975, 1000], dtype=np.float32)

    DATA_DIR = os.path.expanduser("~/geoscience/data")

    df = load_otcr_df(DATA_DIR)

    era5 = np.load(os.path.join(DATA_DIR, "tc_intensity_era5.npy"))
    mld = np.load(os.path.join(DATA_DIR, "tc_intensity_bran2020.npy"))
    rh = np.load(os.path.join(DATA_DIR, "tc_intensity_rh.npy"))
    temp = np.load(os.path.join(DATA_DIR, "tc_intensity_temp_profile.npy"))
    shear = np.load(os.path.join(DATA_DIR, "tc_intensity_windshear.npy"))
    depth = temp_profiles[0, :51]
    temp = temp[:, 51:]

    gms = []
    dssts = []
    max_depths = []
    for i in range(temp.shape[0]):
        gm, dsst, max_depth = fit_profile(temp[i], depth, mld[i])

    mslpFile = os.path.join(TCRM_PATH, "MSLP", "slp.day.ltm.nc")
    mslp = SamplePressure(mslpFile)

    for i, var in enumerate(["sst", "sp", "d2", "t2"]): df[var] = era5[:, i]
    df["hm"] = mld[:, 0]
    df['rh'] = rh[:, 23]  # np.trapz(rh[:, pres_lvls >= 200], pres_lvls[pres_lvls >= 200]) / (800)
    df['gm'] = np.array(gm)
    df['dsst'] = np.array(gm)
    df['max_depth'] = np.array(gm)
    df['shear'] = shear

    df = df[~pd.isnull(df["Vmax (kn)"])]

    groups = df.groupby('DISTURBANCE_ID')

    t0 = time.time()
    out_dfs = []
    land_count = 0
    idx = 0
    verbose = False
    for name, g in list(df.groupby('DISTURBANCE_ID'))[:]:
        if len(g) == 0:
            continue
        # if name == 'AU199697_12U':
        #     print(name)
        #     verbose = False
        # else:
        #     verbose = False
        #     continue
        hurricane = Hurricane(dt=1.0)
        bailed, out = hurricane.simulate(g, verbose=verbose)

        if len(out) > 0:
            final_vm = out.vmax.iloc[-1]
        else:
            final_vm = -1

        if final_vm == 0:
            hurricane = Hurricane(dt=1.0)
            bailed, out = hurricane.simulate(g)
            if bailed:
                print(f"{g.iloc[0].DISTURBANCE_ID}: bailed")
        out_dfs.append(out)

    out_df = pd.concat(out_dfs)

    if OCEAN_MIXING == 'y':
        out_df.to_csv(os.path.join(DATA_DIR, "predicted_intensity_ocean_mixing.csv"))
    else:
        out_df.to_csv(os.path.join(DATA_DIR, "predicted_intensity_dt_1s.csv"))
    print(out_df[out_df.DISTURBANCE_ID == 'AU199697_12U'])
    print(land_count)
    print("Time: ", (time.time() - t0), "s")

