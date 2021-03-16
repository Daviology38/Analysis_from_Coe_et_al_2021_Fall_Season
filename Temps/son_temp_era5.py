#This program loads all of the data, splits it by pattern type and then prints pics for each

#Import a bunch of stuff
import os
os.environ['PROJ_LIB'] = r'c:\Users\CoeFamily\Anaconda3\envs\meteorology\Library\share'
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
from pylab import *
#from sklearn.metrics import mean_squared_error
from math import sqrt
from cartopy import crs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import xarray as xr
import pandas as pd
import matplotlib.gridspec as gridspec

gs1 = gridspec.GridSpec(3, 3)
rc('axes', linewidth=2)
#Check to see if out directory exists and if not, make it
outdir = "era5_son_temp_images"

if not os.path.exists(outdir):
    os.makedirs(outdir)

df = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Coe_clusters_new.csv")
df.columns = ["C","Clust"]
vals = df["Clust"].values
clustvals = list(map(int,vals))

#Put Variables into an array
K = clustvals
#Ask user which CI value to make pics for
#ci = int(raw_input("Enter CI value (Between 1-12)"))
ci = 7

#Plot the images for each of the clusters
i = 1
ci = 7
temps = xr.open_dataset("era5_temps.nc")
temps_m = temps.t2m.mean('time').values - 273.15
fig = plt.figure(figsize=(8, 8))

while i <= ci:
    if(i == 1):
        l = 0
        m = 0
    elif(i == 2):
        l = 0
        m = 1
    elif(i == 3):
        l = 0
        m = 2
    elif(i == 4):
        l = 1
        m = 0
    elif(i == 5):
        l = 1
        m = 1
    elif(i == 6):
        l = 1
        m = 2
    else:
        l = 2
        m = 1
    # WT = precip_data[[j for j, val in enumerate(flat_list) if val == i]]
    # LON2, LAT2 = np.meshgrid(lons2, lats2)

    ax = fig.add_subplot(gs1[l,m], projection=crs.PlateCarree())
    # Get our LAT/LON grid
    lats = Dataset("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Temps/era5_temps.nc").variables['latitude'][:]
    lons = Dataset("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Temps/era5_temps.nc").variables['longitude'][:]
    LON, LAT = np.meshgrid(lons, lats)

    ylocs = np.arange(30, 50, 5)
    xlocs = np.arange(lons[0]-360, 10+lons[-1]-360, 10)

    # Load in the data using xarray
    temps = xr.open_dataset("era5_temps.nc")
    temps_m = temps.t2m.mean('time').values - 273.15
    ind = [ij for ij,val in enumerate(K) if val == i]
    T = temps.t2m[ind,:,:].mean("time") - 273.15
    levels = np.arange(-12,12,2)

    cs = ax.contourf(LON, LAT, T - temps_m, levels, cmap='bwr', extend='both')
    plt.colorbar(cs, shrink=0.86)
    states = NaturalEarthFeature(category = 'cultural',
                     scale = '50m',
                     facecolor = 'none',
                     name = 'admin_1_states_provinces_shp')
    ax.add_feature(states,linewidth=1.,edgecolor="black")
    ax.coastlines('50m',linewidth=0.8)
    ax.gridlines(linestyle='dotted', ylocs=lats[::20], xlocs=lons[::20])


    if(i == 1):
        wtt = 1
    elif(i == 2):
        wtt = 2
    elif( i == 3):
        wtt = 3
    elif(i == 4):
        wtt = 4
    elif(i == 5):
        wtt = 5
    elif( i == 6):
        wtt = 6
    else:
        wtt = 7

    title = 'WT ' + str(wtt) + " Anomaly"
    ax.set_title(title, fontsize=20)

    title1 = outdir + '/' + str(wtt) + '_temps_anomaly'




    i = i + 1
plt.suptitle("WT Temperature Anomaly")
plt.savefig(title1, bbox_inches='tight')