# This program loads all of the data, splits it by pattern type and then prints pics for each

# Import a bunch of stuff
import os

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

# from sklearn.metrics import mean_squared_error
from math import sqrt
from cartopy import crs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import xarray as xr
import pandas as pd
from helper_funcs import make_grid

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# create yellow colormaps
N = 256
green = np.ones((N, 4))
green[:, 0] = np.linspace(55 / 256, 1, N)  # R = 55
green[:, 1] = np.linspace(69 / 256, 1, N)  # G = 69
green[:, 2] = np.linspace(33 / 256, 1, N)  # B = 33
green_cmp = ListedColormap(green)

brown = np.ones((N, 4))
brown[:, 0] = np.linspace(102 / 256, 1, N)  # R = 102
brown[:, 1] = np.linspace(58 / 256, 1, N)  # G = 58
brown[:, 2] = np.linspace(20 / 256, 1, N)  # B = 20
brown_cmp = ListedColormap(brown)

newcolors2 = np.vstack(
    (brown_cmp(np.linspace(0, 1, 128)), green_cmp(np.linspace(1, 0, 128)))
)
double = ListedColormap(
    [
        "#543005",
        "#8c510a",
        "#bf812d",
        "#dfc27d",
        "#f6e8c3",
        "#c7eae5",
        "#80cdc1",
        "#35978f",
        "#01665e",
        "#003c30",
    ]
)

rc("axes", linewidth=2)
# Check to see if out directory exists and if not, make it
outdir = "F:/era5_mam_pics/8"

if not os.path.exists(outdir):
    os.makedirs(outdir)

missing = 1 * 10 ** 15

# Load in the CI results Matlab File
dat_mat = sio.loadmat(
    "C:/Users/wayne/OneDrive - University of Massachusetts Lowell - UMass Lowell/WTs_general/era5_mam_jan21_85/CI_Results.mat"
)
k_mat = dat_mat["K"]
ldf = k_mat[:, 7]

# Put Variables into an array
K = ldf

# Ask user which CI value to make pics for
# ci = int(raw_input("Enter CI value (Between 1-12)"))
ci = 8

# #Plot the images for each of the clusters
temps = xr.open_dataset(
    "C:/Users/wayne/OneDrive - University of Massachusetts Lowell - UMass Lowell/WT_matlab/WTs_general/era5_temps_mam.nc"
)
# Create the daily mean dataset of temperatures
# First take the mean of each day's temps
temps_m = temps.coarsen(time=24).mean()
# Now get the mean of each calendar day of the dataset
temps_dm = np.zeros((92, 81, 121))
for i in range(92):
    indices = list(np.arange(i, 3680, 92))
    temps_dm[i, :, :] = np.mean(temps_m.t2m[indices, :, :], axis=0) - 273.15
# Now remove the daily mean from the actual dataset
T = temps_m.t2m

count1 = 0
for i in range(3680):
    T[i, :, :] = (T[i, :, :] - 273.15) - temps_dm[count1, :, :]
    count1 = count1 + 1
    if count1 == 92:
        count1 = 0

dates = []
# Append the path to the file names
for i in range(1983, 2015, 1):
    for j in range(3, 6, 1):
        for k in range(1, 32, 1):
            if j == 4 and k == 31:
                pass
            else:
                if j < 10:
                    jnew = "0" + str(j)
                else:
                    jnew = str(j)
                if k < 10:
                    knew = "0" + str(k)
                else:
                    knew = str(k)
                print(str(i) + str(jnew) + str(knew))
                dates.append(
                    "F:/persiann-cdr/files/PERSIANN-CDR_v01r01_"
                    + str(i)
                    + str(jnew)
                    + str(knew)
                    + ".nc"
                )
precip_data = np.zeros((len(dates), 121, 81))
count = 0
for date in dates:
    ds = xr.open_dataset(date)
    precip_data[count, :, :] = ds.precipitation[:, 1080:1201, 39:120]
    count = count + 1
lats_precip = ds.lat[39:120].values
lons_precip = ds.lon[1080:1201].values
del ds

dmean_precip = np.zeros((92, 121, 81))
y = 0
for i in range(2944):
    dmean_precip[y, :, :] = dmean_precip[y, :, :] + precip_data[i, :, :]
    y = y + 1
    if y == 92:
        y = 0

dmean_precip = dmean_precip / 32

y = 0
for i in range(2944):
    precip_data[i, :, :] = precip_data[i, :, :] - dmean_precip[y, :, :]
    y = y + 1
    if y == 92:
        y = 0

# Group by year
chunks = [K[x : x + 92] for x in range(0, len(K), 92)]

chunks = chunks[4:36]

flat_list = [item for sublist in chunks for item in sublist]

# Group the data by WT and average per WT
WT1 = precip_data[[i for i, val in enumerate(flat_list) if val == 1]]
WT2 = precip_data[[i for i, val in enumerate(flat_list) if val == 2]]
WT3 = precip_data[[i for i, val in enumerate(flat_list) if val == 3]]
WT4 = precip_data[[i for i, val in enumerate(flat_list) if val == 4]]
WT5 = precip_data[[i for i, val in enumerate(flat_list) if val == 5]]
WT6 = precip_data[[i for i, val in enumerate(flat_list) if val == 6]]
WT7 = precip_data[[i for i, val in enumerate(flat_list) if val == 7]]

j = 1
# Plot the images for each of the clusters

# Load in the data using xarray
h500 = sio.loadmat(
    "C:/Users/wayne/OneDrive - University of Massachusetts Lowell - UMass Lowell/WT_matlab/WTs_general/mam_total_data.mat"
)["h500"]
mslp = sio.loadmat(
    "C:/Users/wayne/OneDrive - University of Massachusetts Lowell - UMass Lowell/WT_matlab/WTs_general/mam_total_data.mat"
)["mslp"]
u850 = sio.loadmat(
    "C:/Users/wayne/OneDrive - University of Massachusetts Lowell - UMass Lowell/WT_matlab/WTs_general/mam_total_data.mat"
)["u850"]
v850 = sio.loadmat(
    "C:/Users/wayne/OneDrive - University of Massachusetts Lowell - UMass Lowell/WT_matlab/WTs_general/mam_total_data.mat"
)["v850"]

i = 1
# Plot the images for each of the clusters
axlist = make_grid(3, 3, cbar=False)
while i <= 8:
    if i != 8:
        ax = axlist[i - 1]
    else:
        ax = axlist[i - 1]

    # Get our LAT/LON grid
    lats = Dataset("F:/ERA5/h500_mam/h500_mam_1979.nc").variables["latitude"][160:241]
    lons = Dataset("F:/ERA5/h500_mam/h500_mam_1979.nc").variables["longitude"][
        1080:1201
    ]
    LON, LAT = np.meshgrid(lons, lats)
    LON_P, LAT_P = np.meshgrid(lons_precip, lats_precip)

    ylocs = np.arange(30, 50, 5)
    xlocs = np.arange(lons[0] - 360, 10 + lons[-1] - 360, 10)

    WT = precip_data[[j for j, val in enumerate(flat_list) if val == i]]
    ind = [jj for jj, val in enumerate(K) if val == i]
    print(ind)
    S = np.nanmean(mslp[ind, :, :], axis=0) / 100
    H = np.nanmean(h500[ind, :, :], axis=0) / 10
    U = np.nanmean(u850[ind, :, :], axis=0)
    V = np.nanmean(v850[ind, :, :], axis=0)
    T2 = np.nanmean(T[ind, :, :], axis=0)
    P = np.nanmean(WT, axis=0)

    # levels = np.arange(-20,20,2)
    levels_p = np.arange(0, 20, 2)
    levels3 = np.arange(-4, 4, 0.5)
    levels = np.arange(480, 620, 6)
    # levels = np.arange(-20,20,2)
    levels1 = np.arange(960, 1028, 4)

    Z_500 = ndimage.gaussian_filter(S[:, :], sigma=3, order=0)
    cd = ax.contour(LON, LAT, Z_500[:, :], levels=levels1, colors="red")
    plt.clabel(cd, inline=True, fmt="%1.0f", fontsize=14, colors="red")
    # cd = ax.contour(LON,LAT,H, levels = levels, colors = 'black')
    # ax.clabel(cd, inline=True, fmt='%1.0f', fontsize=12, colors='black')
    # cs = ax.contourf(LON, LAT, T2, levels3, cmap='bwr', extend='both')
    # plt.colorbar(cs, shrink=0.86)
    cs = ax.contourf(LON_P, LAT_P, P.T, levels3, cmap=double, extend="both")
    # plt.clabel(cs, inline=True, fmt='%1.0f', fontsize=12, colors='black')
    ax.quiver(
        LON[::5, ::5],
        LAT[::5, ::5],
        U[::5, ::5],
        V[::5, ::5],
        scale=None,
        scale_units="inches",
    )

    states = NaturalEarthFeature(
        category="cultural",
        scale="50m",
        facecolor="none",
        name="admin_1_states_provinces_shp",
    )
    ax.add_feature(states, linewidth=1.0, edgecolor="black")
    ax.coastlines("50m", linewidth=0.8)
    ax.gridlines(linestyle="dotted", ylocs=lats[::20], xlocs=lons[::20])

    if i == 1:
        wtt = 1
    elif i == 2:
        wtt = 2
    elif i == 3:
        wtt = 3
    elif i == 4:
        wtt = 4
    elif i == 5:
        wtt = 5
    elif i == 6:
        wtt = 6
    elif i == 7:
        wtt = 7
    elif i == 8:
        wtt = 8
    else:
        wtt = 9

    title = "Cluster " + str(wtt)
    ax.set_title(title, fontsize=15)
    if i == 8:
        axlist[-1].set_visible(False)
        plt.colorbar(cs, ax=axlist[:])
    i = i + 1

# axlist[6].set_visible(False)
plt.suptitle(
    "MSLP, 850 hPa u- v- Wind, and Precipitation Anomaly",
    fontsize=25,
    fontweight="bold",
)
title1 = outdir + "/" + str(wtt) + "mslp_uv_prec"
plt.savefig(title1, bbox_inches="tight")
