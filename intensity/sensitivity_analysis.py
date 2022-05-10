from era5_extract import load_otcr_df
import os
import numpy as np
from tc_intensity_analysis import Hurricane
import pandas as pd
import time


if __name__ == "__main__":

    pres_lvls = np.array([ 1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,
        125,  150,  175,  200,  225,  250,  300,  350,  400,  450,  500,
        550,  600,  650,  700,  750,  775,  800,  825,  850,  875,  900,
        925,  950,  975, 1000], dtype=np.float32)

    DATA_DIR = os.path.expanduser("~/geoscience/data")

    df = load_otcr_df(DATA_DIR)

    era5 = np.load(os.path.join(DATA_DIR, "tc_intensity_era5.npy"))
    bran = np.load(os.path.join(DATA_DIR, "tc_intensity_bran2020.npy"))
    rh = np.load(os.path.join(DATA_DIR, "tc_intensity_rh.npy"))

    for i, var in enumerate(["sst", "sp", "d2", "t2"]): df[var] = era5[:, i]
    df["hm"] = bran[:, 0]
    df['rh'] = rh[:, 21]  # np.trapz(rh[:, pres_lvls >= 200], pres_lvls[pres_lvls >= 200]) / (800)

    df = df[~pd.isnull(df["Vmax (kn)"])]

    for cecd in [0.8, 1.2]:
        print("Running cecd=", cecd)
        groups = df.groupby('DISTURBANCE_ID')

        t0 = time.time()
        out_dfs = []
        land_count = 0
        idx = 0
        verbose = False
        for name, g in list(df.groupby('DISTURBANCE_ID'))[:]:
            if len(g) == 0:
                continue

            hurricane = Hurricane(dt=10.0, cecd=cecd)
            bailed, out = hurricane.simulate(g, verbose=verbose)

            if len(out) > 0:
                final_vm = out.vmax.iloc[-1]
            else:
                final_vm = -1

            if final_vm == 0:
                hurricane = Hurricane(dt=1.0, cecd=cecd)
                bailed, out = hurricane.simulate(g)
                if bailed:
                    print(f"{g.iloc[0].DISTURBANCE_ID}: bailed")
            out_dfs.append(out)

        out_df = pd.concat(out_dfs)
        out_df.to_csv(os.path.join(DATA_DIR, f"predicted_intensity_cecd_{cecd}.csv"))
        print(land_count)
        print("Time: ", (time.time() - t0), "s")

    for efrac in [0.3, 0.7]:
        print("Running efrac=", efrac)
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
            hurricane = Hurricane(dt=10.0, efrac=efrac)
            bailed, out = hurricane.simulate(g, verbose=verbose)

            if len(out) > 0:
                final_vm = out.vmax.iloc[-1]
            else:
                final_vm = -1

            if final_vm == 0:
                hurricane = Hurricane(dt=1.0, efrac=efrac)
                bailed, out = hurricane.simulate(g)
                if bailed:
                    print(f"{g.iloc[0].DISTURBANCE_ID}: bailed")
            out_dfs.append(out)

        out_df = pd.concat(out_dfs)
        out_df.to_csv(os.path.join(DATA_DIR, f"predicted_intensity_efrac_{efrac}.csv"))
        print(land_count)
        print("Time: ", (time.time() - t0), "s")