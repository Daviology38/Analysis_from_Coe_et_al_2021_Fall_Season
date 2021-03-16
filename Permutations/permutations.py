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

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from cartopy import crs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

df = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Coe_clusters_new.csv")
df.columns = ["Day","Clust"]
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
#Now compute the persistence for each month
numbers = [1,2,3,4,5,6,7]
output = []
output2 = []
for i in range(2,5,1):
    possibilities = list(product(numbers,repeat = i))
    poss_new = []
    count = [0] * len(possibilities)
    for thing in yearly:
        range_len = len(thing)
        count_range = 0
        list_range = []
        jj = 0
        while jj <= range_len - i:

            if (len(list_range) == 0):
                count_range = jj
                list_range.append(thing[count_range])
                count_range += 1
                jj = jj + 1
            elif (len(list_range) < i):
                # Check if the value is the same
                temp_val = list_range[(len(list_range) - 1)]
                if (temp_val == thing[count_range]):
                    count_range += 1
                    jj = jj + 1
                else:
                    list_range.append(thing[count_range])
                    count_range += 1
                    jj = jj + 1
            else:
                #Check if the possibility we found is in our list
                if((len(poss_new)) == 0):
                    poss_new.append(tuple(list_range))
                    count[0] += 1
                else:

                    if(tuple(list_range) in poss_new):
                        ind = poss_new.index(tuple(list_range))
                        count[ind] += 1
                    else:
                        #Check if the possibility or any of its permutations are in our list
                        for kks in range(len(list_range)):
                            #Move last value to first position
                            list_range.insert(0,list_range.pop())
                            if( tuple(list_range) in poss_new):
                                ind = poss_new.index(tuple(list_range))
                                count[ind] += 1
                                testing_variable = 1
                                break
                            else:
                                testing_variable = 0
                        if(testing_variable == 0):
                            poss_new.append(tuple(list_range))
                            count[len(poss_new)-1] += 1
                list_range = []
    df3 = pd.DataFrame()
    #possibilities = [tuple(sorted(l)) for l in possibilities]
    df3["Pattern"] = poss_new
    df3["Count"] = count[0:len(poss_new)]
    count_p = [(ci / sum(count[0:len(poss_new)])) * 100 for ci in count[0:len(poss_new)]]
    df3["Percent"] = count_p
    df3.to_csv(str(i) + "_nopersistence.csv")


