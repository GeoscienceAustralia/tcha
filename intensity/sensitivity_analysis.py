from era5_extract import load_otcr_df
import os
import numpy as np
from tc_intensity_analysis import Hurricane
import pandas as pd
import time
from functools import partial
import concurrent.futures
from mpi4py import MPI


"""
This script performs a sensitivity analysis of the TC intensity model by running it for the historical record
while varying the parameters.
"""

def run_simulation(param, df):
    """
    Helper function to run the simulation for one parameter set
    """

    print("Processing:", param)
    out_dfs = []
    verbose = False
    for name, g in list(df.groupby('DISTURBANCE_ID'))[:]:
        if len(g) == 0:
            continue

        hurricane = Hurricane(dt=0.25, **param)
        bailed, out = hurricane.simulate(g, verbose=verbose)
        out_dfs.append(out)

    out_df = pd.concat(out_dfs)
    key = list(param.keys())[0]
    out_df.to_csv(os.path.join(DATA_DIR, f"predicted_intensity_{key}_{param[key]}.csv"))
    return param


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

    params = [
        {'cecd': 0.8},
        {'cecd': 1.2},
        {'efrac': 0.5},
        {'efrac': 0.9},
        {'tinit': 1.0},
        {'tinit': 4.0},
    ]

    t0 = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    param = params[rank]
    run_simulation(param, df)
    #time.sleep(2 - rank)
    comm.Barrier()

    if rank == 0:
        print("Time: ", (time.time() - t0), "s")
