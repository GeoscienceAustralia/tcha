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

class Hurricane:

    def __init__(self):
        self.rbs1, self.rbs2, self.rbs3, self.rts1, self.rts2 = [np.zeros(200, dtype=np.float32) for _ in range(5)]
        self.x1, self.x2, self.x3, self.xm1, self.xm2, self.xm3, self.rts3 = [np.zeros(200, dtype=np.float32) for _ in range(7)]
        self.mu1, self.mu2, self.mu3, self.rb1, self.rb2, self.rt1 = [np.zeros(200, dtype=np.float32) for _ in range(6)]
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
        vdisp = 's'  # plotting param
        dtg = 3  # plotting param
        rog = 150  # plotting param
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
        nr = 50 # numer of radial nodes
        dt = 20 # time step in seconds
        
        out = np.zeros(3)
        hurr.tc_intensity(
            nrd, tend, vdisp, dtg, rog, rst, vm,
            rm, r0, ts, to, h_a, alat, tshear, vext, tland, surface,
            hs, om, ut, eddytime, heddy, rwide, dim, fmt, nr,
            dt, ro, ahm, pa, cd, cd1, cdcap,
            cecd, pnu, taur, radmax, tauc, efrac, dpb, hm, dsst, gm, out,
            self.rbs1, self.rbs2, self.rbs3, self.rts1, self.rts2,
            self.x1, self.x2, self.x3, self.xm1, self.xm2, self.xm3, self.rts3, 
            self.mu1, self.mu2, self.mu3, 
            self.rb1, self.rb2, self.rt1, self.init
        )
        self.init = 'n'
        return out


if __name__ == "__main__":
    DATA_DIR = os.path.expanduser("~")    
    
    df = load_otcr_df()

    era5 = np.load(os.path.join(DATA_DIR, "tc_intensity_era5.npy"))
    bran = np.load(os.path.join(DATA_DIR, "tc_intensity_bran2020.npy"))

    for i, var in enumerate(["sst", "sp", "d2", "t2"]): df[var] = era5[:, i]
    df["hm"] = bran[:, 0]

    groups = df.groupby('DISTURBANCE_ID')
    df['pc'] = np.zeros(len(df)) * np.nan

    t0 = time.time()
    for name, g in df.groupby('DISTURBANCE_ID'): 
        hurricane = Hurricane()
        vm = np.nan
        r0 = 350 # do some sort of sampling
        rm = 80 # do some sort of sampling
        for j, i in enumerate(g.index[:-1]):
            row = g.loc[i]

            sst, sp, d2, t2, hm, lat = row.sst, row.sp, row.d2, row.t2, row.hm, row.LAT
            
            if np.isnan(vm):
                vm = row["adj. ADT Vm (kn)"] * 0.514444 # convert knots to m/s
                if np.isnan(vm): continue

            sst = sst - 273.15  # convert K -> C
            sp = sp / 100  # convert Pa -> hPa
            t2 = units.Quantity(t2, "K")
            d2 = units.Quantity(d2, "K")

            h_a = relative_humidity_from_dewpoint(t2, d2) * 100 # relative humidity near surface
            tend = g.loc[g.index[j + 1]].TM - row.TM
            tend = tend.days + tend.seconds / (3600 * 24)  # simulation time in days

            # typical values
            ahm = 45  # relative humidity in tropo            
            

            # typical values
            vm = 15
            sst = 27
            sp = 1005
            h_a = 80
            rm = 80
            lat = 20
            tend = 5

            vm_actual = g.loc[g.index[j + 1]]["adj. ADT Vm (kn)"] * 0.514444
            out = hurricane.pytc_intensity(vm, rm, r0, sst, h_a, abs(lat), ahm, sp, tend)
            pmin, vm_, rm_ = out[0], out[1], out[2]
            print("Output:", vm_, vm_actual)
            out = hurricane.pytc_intensity(vm, rm, r0, sst, h_a, abs(lat), ahm, sp, tend)
            pmin, vm, rm = out[0], out[1], out[2]
            print("Output:", vm, vm_actual)

            hurricane = Hurricane()
            vm = 15
            rm = 80
            out = hurricane.pytc_intensity(vm, rm, r0, sst, h_a, abs(lat), ahm, sp, 2 * tend)
            pmin, vm, rm = out[0], out[1], out[2]
            print("Output:", vm, vm_actual)
            break
        break

    
    print(len(g) / len(df))
    print("Estimated time for full run: ", (time.time() - t0) * len(df) / len(g), "s")
