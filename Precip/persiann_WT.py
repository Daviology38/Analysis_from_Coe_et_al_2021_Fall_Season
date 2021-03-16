#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:55:33 2019

@author: mariofire
"""

import metpy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import cartopy.feature as cfeature
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

dates = []
#Append the path to the file names
for i in range(1983,2015,1):
    for j in range(9,12,1):
        for k in range(1,32,1):
            if( j == 9 and k == 31):
                pass
            elif( j == 11 and k == 31):
                pass
            else:
                if(j < 10):
                    jnew = "0" + str(j)
                else:
                    jnew = str(j)
                if(k < 10):
                    knew = "0" + str(k)
                else:
                    knew = str(k)
                print(str(i)+str(jnew)+str(knew))
                dates.append('H:/persiann-cdr/files/PERSIANN-CDR_v01r01_' + str(i) + str(jnew) + str(knew) + '.nc')
precip_data = np.zeros((len(dates),121,81))
count = 0
for date in dates:
    ds = xr.open_dataset(date)
    precip_data[count,:,:] = ds.precipitation[:,1080:1201,39:120]
    count = count + 1
lats = ds.lat[39:120].values
lons = ds.lon[1080:1201].values
del ds



df = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Coe_clusters_new.csv")
df.columns = ["C","Clust"]
vals = df["Clust"].values
clustvals = list(map(int,vals))

#Group by year
chunks = [clustvals[x:x+91] for x in range(0, len(clustvals), 91)]

chunks = chunks[4:36]

flat_list = [item for sublist in chunks for item in sublist]

#Group the data by WT and average per WT
WT1 = precip_data[[i for i, val in enumerate(flat_list) if val == 1]]
WT2 = precip_data[[i for i, val in enumerate(flat_list) if val == 2]]
WT3 = precip_data[[i for i, val in enumerate(flat_list) if val == 3]]
WT4 = precip_data[[i for i, val in enumerate(flat_list) if val == 4]]
WT5 = precip_data[[i for i, val in enumerate(flat_list) if val == 5]]
WT6 = precip_data[[i for i, val in enumerate(flat_list) if val == 6]]
WT7 = precip_data[[i for i, val in enumerate(flat_list) if val == 7]]

j = 1
#Plot the images for each of the clusters
while j <= 7:
    WT = precip_data[[i for i, val in enumerate(flat_list) if val == j]]
    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # Get our LAT/LON grid
    LON, LAT = np.meshgrid(lons,lats)

    #ylocs = np.arange(30, 50, 5)
    #xlocs = np.arange(lons[0]-360, 10+lons[-1]-360, 10)

    levels = np.arange(0,20,2)
    cs = ax.contourf(LON,LAT,np.nanmean(WT,axis=0).T,levels, cmap='Greens',extend = 'both',transform=ccrs.PlateCarree())
    #plt.clabel(cs, inline=True, fmt='%1.0f', fontsize=12, colors='black')
    plt.colorbar(cs,shrink=0.86)
    ax.add_feature(cfeature.STATES)
    ax.coastlines('50m',linewidth=0.8)
    ax.gridlines(linestyle='dotted', ylocs=lats[::20], xlocs=lons[::20])


    if(j == 1):
        wtt = 1
    elif(j == 2):
        wtt = 2
    elif( j == 3):
        wtt = 3
    elif(j == 4):
        wtt = 4
    elif(j == 5):
        wtt = 5
    elif( j == 6):
        wtt = 6
    else:
        wtt = 7

    title = 'Cluster ' + str(wtt) +' Precipitation (mm)'
    ax.set_title(title, fontsize=20)

    title1 = str(wtt) + '_precip'


    plt.savefig(title1, bbox_inches='tight')

    plt.gcf().clear()

    j = j + 1

WT1_precip = []
WT2_precip = []
WT3_precip = []
WT4_precip = []
WT5_precip = []
WT6_precip = []
WT7_precip = []

for i in range(0,len(WT1),1):
    WT1_precip.append(np.nanmean(WT1[i,:,:]))

for i in range(0,len(WT2),1):
    WT2_precip.append(np.nanmean(WT2[i,:,:]))

for i in range(0,len(WT3),1):
    WT3_precip.append(np.nanmean(WT3[i,:,:]))

for i in range(0,len(WT4),1):
    WT4_precip.append(np.nanmean(WT4[i,:,:]))

for i in range(0,len(WT5),1):
    WT5_precip.append(np.nanmean(WT5[i,:,:]))

for i in range(0,len(WT6),1):
    WT6_precip.append(np.nanmean(WT6[i,:,:]))

for i in range(0,len(WT7),1):
    WT7_precip.append(np.nanmean(WT7[i,:,:]))

WT_list = np.zeros(7)
WT_list[0] = np.nanmean(WT1_precip)
WT_list[1] = np.nanmean(WT2_precip)
WT_list[2] = np.nanmean(WT3_precip)
WT_list[3] = np.nanmean(WT4_precip)
WT_list[4] = np.nanmean(WT5_precip)
WT_list[5] = np.nanmean(WT6_precip)
WT_list[6] = np.nanmean(WT7_precip)

#load the matlab progression data
x = sio.loadmat('C:/Users/CoeFamily/OneDrive - University of Massachusetts Lowell - UMass Lowell/Autumn WTs/SON_3wtprogression.mat')
eseason = x['Earlyseason']
eseason_f = [item for sublist in eseason for item in sublist]

#get the precip values for each type of data
#1 = 1-6-1
#2 = 1-6-2
#3 = 1-6-5

#Group by year

eseason_f = eseason[:,4:36]
list = [item for sublist in eseason_f for item in sublist]
p161 = np.nanmean(precip_data[[i for i, val in enumerate(list) if val == 1]],axis=0)
p126 = np.nanmean(precip_data[[i for i, val in enumerate(list) if val == 2]],axis=0)
p165 = np.nanmean(precip_data[[i for i, val in enumerate(list) if val == 3]],axis=0)

plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
# Get our LAT/LON grid
LON, LAT = np.meshgrid(lons,lats)

#ylocs = np.arange(30, 50, 5)
#xlocs = np.arange(lons[0]-360, 10+lons[-1]-360, 10)

levels = np.arange(0,20,2)
cs = ax.contourf(LON,LAT,p161.T,levels, cmap='Greens',extend = 'both')
#plt.clabel(cs, inline=True, fmt='%1.0f', fontsize=12, colors='black')
plt.colorbar(cs,shrink=0.86)
ax.add_feature(cfeature.STATES)
ax.coastlines('50m',linewidth=0.8)
ax.gridlines(linestyle='dotted', ylocs=lats[::20], xlocs=lons[::20])
title = 'Progression 1-6-1 Precipitation (mm)'
ax.set_title(title, fontsize=20)

title1 = '161_precip'


plt.savefig(title1, bbox_inches='tight')

plt.gcf().clear()

plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
# Get our LAT/LON grid
LON, LAT = np.meshgrid(lons,lats)

#ylocs = np.arange(30, 50, 5)
#xlocs = np.arange(lons[0]-360, 10+lons[-1]-360, 10)

levels = np.arange(0,20,2)
cs = ax.contourf(LON,LAT,p165.T,levels, cmap='Greens',extend = 'both')
#plt.clabel(cs, inline=True, fmt='%1.0f', fontsize=12, colors='black')
plt.colorbar(cs,shrink=0.86)
ax.add_feature(cfeature.STATES)
ax.coastlines('50m',linewidth=0.8)
ax.gridlines(linestyle='dotted', ylocs=lats[::20], xlocs=lons[::20])
title = 'Progression 1-6-5 Precipitation (mm)'
ax.set_title(title, fontsize=20)

title1 = '165_precip'


plt.savefig(title1, bbox_inches='tight')

plt.gcf().clear()

plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
# Get our LAT/LON grid
LON, LAT = np.meshgrid(lons,lats)

#ylocs = np.arange(30, 50, 5)
#xlocs = np.arange(lons[0]-360, 10+lons[-1]-360, 10)

levels = np.arange(0,20,2)
cs = ax.contourf(LON,LAT,p126.T,levels, cmap='Greens',extend = 'both')
#plt.clabel(cs, inline=True, fmt='%1.0f', fontsize=12, colors='black')
plt.colorbar(cs,shrink=0.86)
ax.add_feature(cfeature.STATES)
ax.coastlines('50m',linewidth=0.8)
ax.gridlines(linestyle='dotted', ylocs=lats[::20], xlocs=lons[::20])
title = 'Progression 1-2-6 Precipitation (mm)'
ax.set_title(title, fontsize=20)

title1 = '126_precip'


plt.savefig(title1, bbox_inches='tight')

plt.gcf().clear()

#load the matlab progression data
x = sio.loadmat('C:/Users/CoeFamily/OneDrive - University of Massachusetts Lowell - UMass Lowell/Autumn WTs/SON_4wtprogression.mat')
eseason = x['Earlyseason']
lseason = x['Lateseason']

eseason_f = eseason[:,4:36]
lseason_f = lseason[:,4:36]
list = [item for sublist in eseason_f for item in sublist]
list2 = [item for sublist in lseason_f for item in sublist]
p1265 = np.nanmean(precip_data[[i for i, val in enumerate(list) if val == 1]],axis=0)
p3475 = np.nanmean(precip_data[[i for i, val in enumerate(list2) if val == 2]],axis=0)


plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
# Get our LAT/LON grid
LON, LAT = np.meshgrid(lons,lats)

#ylocs = np.arange(30, 50, 5)
#xlocs = np.arange(lons[0]-360, 10+lons[-1]-360, 10)

levels = np.arange(0,20,2)
cs = ax.contourf(LON,LAT,p1265.T,levels, cmap='Greens',extend = 'both')
#plt.clabel(cs, inline=True, fmt='%1.0f', fontsize=12, colors='black')
plt.colorbar(cs,shrink=0.86)
ax.add_feature(cfeature.STATES)
ax.coastlines('50m',linewidth=0.8)
ax.gridlines(linestyle='dotted', ylocs=lats[::20], xlocs=lons[::20])
title = 'Progression 1-2-6-5 Precipitation (mm)'
ax.set_title(title, fontsize=20)

title1 = '1265_precip'


plt.savefig(title1, bbox_inches='tight')

plt.gcf().clear()

plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
# Get our LAT/LON grid
LON, LAT = np.meshgrid(lons,lats)

#ylocs = np.arange(30, 50, 5)
#xlocs = np.arange(lons[0]-360, 10+lons[-1]-360, 10)

levels = np.arange(0,20,2)
cs = ax.contourf(LON,LAT,p3475.T,levels, cmap='Greens',extend = 'both')
#plt.clabel(cs, inline=True, fmt='%1.0f', fontsize=12, colors='black')
plt.colorbar(cs,shrink=0.86)
ax.add_feature(cfeature.STATES)
ax.coastlines('50m',linewidth=0.8)
ax.gridlines(linestyle='dotted', ylocs=lats[::20], xlocs=lons[::20])
title = 'Progression 3-4-7-5 Precipitation (mm)'
ax.set_title(title, fontsize=20)

title1 = '3475_precip'


plt.savefig(title1, bbox_inches='tight')

plt.gcf().clear()
