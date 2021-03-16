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

df = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Coe_clusters_new.csv")
df.columns = ["C","Clust"]
vals = df["Clust"].values
clustvals = list(map(int,vals))

#Group by month
count = 1
sep = []
oct = []
nov = []
for i in clustvals:
    if(count <= 30):
        sep.append(i)
    elif(count > 30 and count <=61):
        oct.append(i)
    elif(count > 61 and count <= 91):
        nov.append(i)
    count = count + 1
    if(count == 92):
        count = 1

#Group by year
chunks_sep = [sep[x:x+30] for x in range(0, len(sep), 30)]
chunks_oct = [oct[x:x+31] for x in range(0, len(oct), 31)]
chunks_nov = [nov[x:x+30] for x in range(0, len(nov), 30)]

#Remove persistence in each year
yearly_sep = []
yearly_oct = []
yearly_nov = []
for chunk in chunks_sep:
    #ldf = [i[0] for i in groupby(chunk)]
    ldf = chunk
    ldf = list(map(int,ldf))
    yearly_sep.append(ldf)

for chunk in chunks_oct:
    #ldf = [i[0] for i in groupby(chunk)]
    ldf = chunk
    ldf = list(map(int,ldf))
    yearly_oct.append(ldf)

for chunk in chunks_nov:
    #ldf = [i[0] for i in groupby(chunk)]
    ldf = chunk
    ldf = list(map(int,ldf))
    yearly_nov.append(ldf)

persist_sep = np.zeros((7,12))
persist_oct = np.zeros((7,12))
persist_nov = np.zeros((7,12))
#Now compute the persistence for each month
numbers = [1,2,3,4,5,6,7]
output = []
output2 = []
for i in range(2,5,1):
    possibilities = list(product(numbers,repeat = i))
    poss_new = []
    count = [0] * len(possibilities)
    for thing in yearly_sep:
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
    df3.to_csv(str(i) + "sep_patterns_nopersistence.csv")

for thing in yearly_sep:
    range_len = len(thing)
    count_range = 0
    list_range = []
    q = 0
    test_v = 0
    counter = 1
    while q < len(thing):
        temp_v = thing[q]
        if(q == 0):
            test_v = temp_v
            q += 1
        elif(temp_v == test_v and q+1 != len(thing)):
            q += 1
            counter += 1
        elif(temp_v == test_v and q+1 == len(thing)):
            q += 1
            counter += 1
            persist_sep[test_v - 1, counter - 1] += 1
        elif(q + 1 == len(thing)):
            persist_sep[test_v-1,counter-1] += 1
            persist_sep[temp_v-1,0] += 1
            q += 1
        else:
            persist_sep[test_v-1,counter-1] += 1
            test_v = temp_v
            counter = 1
            q += 1

dff_persist = pd.DataFrame(persist_sep)
dff_persist.to_csv("sep_persistence.csv")

numbers = [1,2,3,4,5,6,7]
output = []
output2 = []
for i in range(2,5,1):
    possibilities = list(product(numbers,repeat = i))
    poss_new = []
    count = [0] * len(possibilities)
    for thing in yearly_oct:
        range_len = len(thing)
        count_range = 0
        list_range = []
        for jj in range(range_len - i):
            count_range = jj
            if (len(list_range) == 0):
                list_range.append(thing[count_range])
                count_range += 1
            elif (len(list_range) < i):
                # Check if the value is the same
                temp_val = list_range[(len(list_range) - 1)]
                if (temp_val == thing[count_range]):
                    count_range += 1
                else:
                    list_range.append(thing[count_range])
                    count_range += 1
            else:
                # Check if the possibility we found is in our list
                if ((len(poss_new)) == 0):
                    poss_new.append(tuple(list_range))
                    count[0] += 1
                else:

                    if (tuple(list_range) in poss_new):
                        ind = poss_new.index(tuple(list_range))
                        count[ind] += 1
                    else:
                        # Check if the possibility or any of its permutations are in our list
                        for kks in range(len(list_range)):
                            # Move last value to first position
                            list_range.insert(0, list_range.pop())
                            if (tuple(list_range) in poss_new):
                                ind = poss_new.index(tuple(list_range))
                                count[ind] += 1
                                testing_variable = 1
                                break
                            else:
                                testing_variable = 0
                        if (testing_variable == 0):
                            poss_new.append(tuple(list_range))
                            count[len(poss_new) - 1] += 1
                list_range = []
    df3 = pd.DataFrame()
    # possibilities = [tuple(sorted(l)) for l in possibilities]
    df3["Pattern"] = poss_new
    df3["Count"] = count[0:len(poss_new)]
    count_p = [(ci / sum(count[0:len(poss_new)])) * 100 for ci in count[0:len(poss_new)]]
    df3["Percent"] = count_p
    df3.to_csv(str(i) + "oct_patterns_nopersistence.csv")

for thing in yearly_oct:
    range_len = len(thing)
    count_range = 0
    list_range = []
    q = 0
    test_v = 0
    counter = 1
    while q < len(thing):
        temp_v = thing[q]
        if(q == 0):
            test_v = temp_v
            q += 1
        elif(temp_v == test_v and q+1 != len(thing)):
            q += 1
            counter += 1
        elif(temp_v == test_v and q+1 == len(thing)):
            q += 1
            counter += 1
            persist_oct[test_v - 1, counter - 1] += 1
        elif(q + 1 == len(thing)):
            persist_oct[test_v-1,counter-1] += 1
            persist_oct[temp_v-1,0] += 1
            q += 1
        else:
            persist_oct[test_v-1,counter-1] += 1
            test_v = temp_v
            counter = 1
            q += 1

dff_persist = pd.DataFrame(persist_oct)
dff_persist.to_csv("oct_persistence.csv")

numbers = [1,2,3,4,5,6,7]
output = []
output2 = []
for i in range(2,5,1):
    possibilities = list(product(numbers,repeat = i))
    poss_new = []
    count = [0] * len(possibilities)
    for thing in yearly_nov:
        range_len = len(thing)
        count_range = 0
        list_range = []
        for jj in range(range_len - i):
            count_range = jj
            if (len(list_range) == 0):
                list_range.append(thing[count_range])
                count_range += 1
            elif (len(list_range) < i):
                # Check if the value is the same
                temp_val = list_range[(len(list_range) - 1)]
                if (temp_val == thing[count_range]):
                    count_range += 1
                else:
                    list_range.append(thing[count_range])
                    count_range += 1
            else:
                # Check if the possibility we found is in our list
                if ((len(poss_new)) == 0):
                    poss_new.append(tuple(list_range))
                    count[0] += 1
                else:

                    if (tuple(list_range) in poss_new):
                        ind = poss_new.index(tuple(list_range))
                        count[ind] += 1
                    else:
                        # Check if the possibility or any of its permutations are in our list
                        for kks in range(len(list_range)):
                            # Move last value to first position
                            list_range.insert(0, list_range.pop())
                            if (tuple(list_range) in poss_new):
                                ind = poss_new.index(tuple(list_range))
                                count[ind] += 1
                                testing_variable = 1
                                break
                            else:
                                testing_variable = 0
                        if (testing_variable == 0):
                            poss_new.append(tuple(list_range))
                            count[len(poss_new) - 1] += 1
                list_range = []
    df3 = pd.DataFrame()
    # possibilities = [tuple(sorted(l)) for l in possibilities]
    df3["Pattern"] = poss_new
    df3["Count"] = count[0:len(poss_new)]
    count_p = [(ci / sum(count[0:len(poss_new)])) * 100 for ci in count[0:len(poss_new)]]
    df3["Percent"] = count_p
    df3.to_csv(str(i) + "nov_patterns_nopersistence.csv")

for thing in yearly_nov:
    range_len = len(thing)
    count_range = 0
    list_range = []
    q = 0
    test_v = 0
    counter = 1
    while q < len(thing):
        temp_v = thing[q]
        if(q == 0):
            test_v = temp_v
            q += 1
        elif(temp_v == test_v and q+1 != len(thing)):
            q += 1
            counter += 1
        elif(temp_v == test_v and q+1 == len(thing)):
            q += 1
            counter += 1
            persist_nov[test_v - 1, counter - 1] += 1
        elif(q + 1 == len(thing)):
            persist_nov[test_v-1,counter-1] += 1
            persist_nov[temp_v-1,0] += 1
            q += 1
        else:
            persist_nov[test_v-1,counter-1] += 1
            test_v = temp_v
            counter = 1
            q += 1

dff_persist = pd.DataFrame(persist_nov)
dff_persist.to_csv("nov_persistence.csv")

