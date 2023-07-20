"""
Extract lifetime maximum intensity from TCLV data

* Assumes files already contain normalised intensity - preferably scaled via QDM

"""

import os
import pandas as pd

basepath = "/g/data/w85/QFES_SWHA/hazard/input/tclv/20200622"
outputpath = "/g/data/w85/QFES_SWHA/hazard/input/tclv/lmi"

models = ["ACCESS1-0Q", "ACCESS1-3Q", "CCSM4Q",
          "CNRM-CM5Q", "CSIRO-Mk3-6-0Q", "GFDL-CM3Q",
          "GFDL-ESM2MQ", "HadGEM2Q", "MIROC5Q",
          "MPI-ESM-LRQ", "NorESM1-MQ"]
inputperiods = ["1981-2010", "2021-2040", "2041-2060", "2061-2080", "2081-2100"]
outputperiods = ["1980-2019", "2020-2039", "2040-2059", "2060-2079", "2080-2099"]
rcps = ["RCP45", "RCP85"]

for m in models:
    for rcp in rcps:
        for ip, op in zip(inputperiods, outputperiods):
            iname = os.path.join(basepath, f"{m}_{rcp}_{ip}_bc.csv")
            oname = os.path.join(outputpath, f"{m}.{rcp}.{op}.csv")
            plotname = os.path.join(outputpath, f"{m}.{rcp}.{op}.png")

            print(iname)
            df = pd.read_csv(iname)
            outdf = df.loc[df.groupby("num").ni.idxmax().values].to_csv(oname, index=False)
