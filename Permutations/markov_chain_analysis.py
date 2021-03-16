from itertools import product
import pandas as pd
import xarray as xr
from itertools import groupby
import random
import numpy as np

import numpy as np
import matplotlib as plt
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os
import datetime
import pickle
import csv
import scipy as sp
import scipy.ndimage as ndimage
import scipy.io as sio
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Coe_SONclusts.csv")
df.columns = ["C","Clust"]
vals = df["Clust"].values
clustvals = list(map(int,vals))

#Group by year
chunks = [clustvals[x:x+91] for x in range(0, len(clustvals), 91)]

#Remove persistence in each year
yearly = []
for chunk in chunks:

    #ldf = [i[0] for i in groupby(chunk)]
    ldf = chunk
    ldf = list(map(int,ldf))
    yearly.append(ldf)

numbers = [1,2,3,4,5,6,7]
output = []
output2 = []
for i in range(2,3,1):
    possibilities = list(product(numbers,repeat = i))
    count = [0] * len(possibilities)
    next_thing = np.zeros((len(possibilities), 7))
    for thing in yearly:
        for j in range(len(thing) - (i)):
            a = []
            [a.append(thing[x]) for x in range(j, j + i, 1)]
            temp = possibilities.index(tuple(map(int, a)))
            next_thing[temp, thing[j + i]-1] = next_thing[temp, thing[j + i]-1] + 1
            count[temp] = count[temp] + 1
        df3 = pd.DataFrame()
        # possibilities = [tuple(sorted(l)) for l in possibilities]
        df3["Pattern"] = possibilities
        df3["count"] = count
        for k in range(7):
            df3[str(k+1)] = next_thing[:,k]
    next_thing_monte = np.zeros((1000,len(possibilities),7))
    for ll in range(1000):
        for lm in range(len(possibilities)):
            temp_sample = random.sample(clustvals,count[lm])
            for lk in temp_sample:
                next_thing_monte[ll,lm,lk-1] = next_thing_monte[ll,lm,lk-1] + 1
    twenfiv = np.zeros((len(possibilities),7))
    ninefiv = np.zeros((len(possibilities),7))
    for nn in range(len(possibilities)):
        twenfiv[nn,:] = np.sort(next_thing_monte[:,nn,:],axis=0)[24]
        ninefiv[nn,:] = np.sort(next_thing_monte[:, nn, :], axis=0)[974]
