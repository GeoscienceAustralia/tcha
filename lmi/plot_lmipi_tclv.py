"""
Extract lifetime maximum intensity from TCLV data

* Assumes files already contain normalised intensity - preferably scaled via QDM

"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

outputpath = "/g/data/w85/QFES_SWHA/hazard/input/tclv/lmipi"

models = ["ACCESS1-0Q", "ACCESS1-3Q", "CCSM4Q",
          "CNRM-CM5Q", "CSIRO-Mk3-6-0Q", "GFDL-CM3Q",
          "GFDL-ESM2MQ", "HadGEM2Q", "MIROC5Q",
          "MPI-ESM-LRQ", "NorESM1-MQ"]
inputperiods = ["2020-2039", "2040-2059", "2060-2079", "2080-2099"]
rcps = ["RCP45", "RCP85"]

for m in models:
    for rcp in rcps:
        for ip in inputperiods:
            oname = os.path.join(outputpath, f"{m}.{rcp}.{op}.lmi.pi.csv")
            plotname = os.path.join(outputpath, f"{m}.{rcp}.{op}.png")

            print(oname)
            df = pd.read_csv(oname)
            fig, ax = plt.subplots(1, 1)
            sns.scatterplot(data=df, x='dailyltmvmax', y='vmax')
            ax.plot(np.arange(0, 101), np.arange(0, 101), linestyle='--', alpha=0.5)
            ax.set_xlabel('Daily LTM PI [m/s]')
            ax.set_ylabel('Maximum wind speed [m/s]')
            plt.title(f"{m} - {rcp} - {ip}")
            plt.savefig(plotname)
            plt.close(fig)