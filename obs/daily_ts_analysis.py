import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose as sdc

sns.set_style('whitegrid')
sns.set_context('paper')

datapath = r"C:\WorkSpace\data\observations\wind"
outputpath = r"C:\WorkSpace\data\observations\1-minute\ts"
filelist = os.listdir(datapath)

for f in filelist:
    print(f)
    datafile = os.path.join(datapath, f)
    df = pd.read_csv(datafile)
    df.index = pd.DatetimeIndex(df.date)
    df = df.asfreq("D", method='bfill')
    data = df.windgust
    if len(data)< 365*3: continue
    result = sdc(data, model='additive', period=365, two_sided=True)
    result.plot(); plt.savefig(os.path.join(outputpath, f.replace("txt", "png")), bbox_inches='tight')
    plt.close()
