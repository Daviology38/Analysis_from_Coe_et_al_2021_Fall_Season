# This program loads all of the data, splits it by pattern type and then prints pics for each

# Import a bunch of stuff
import os

os.environ["PROJ_LIB"] = r"c:\Users\CoeFamily\Anaconda3\envs\meteorology\Library\share"
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
import matplotlib.colors as colors

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# create yellow colormaps
N = 256
green = np.ones((N, 4))
green[:, 0] = np.linspace(0 / 256, 1, N)  # R = 0
green[:, 1] = np.linspace(60 / 256, 1, N)  # G = 60
green[:, 2] = np.linspace(48 / 256, 1, N)  # B = 48
green_cmp = ListedColormap(green)

brown = np.ones((N, 4))
brown[:, 0] = np.linspace(84 / 256, 1, N)  # R = 84
brown[:, 1] = np.linspace(48 / 256, 1, N)  # G = 48
brown[:, 2] = np.linspace(5 / 256, 1, N)  # B = 5
brown_cmp = ListedColormap(brown)

newcolors2 = np.vstack(
    (brown_cmp(np.linspace(0, 1, 128)), green_cmp(np.linspace(1, 0, 128)))
)
# double = ListedColormap(newcolors2, name='double')
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
norm = colors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
rc("axes", linewidth=2)
# Check to see if out directory exists and if not, make it
outdir = "F:/son_sequence_pics"

if not os.path.exists(outdir):
    os.makedirs(outdir)

# Load in the data using xarray
h500 = sio.loadmat(
    "C:/Users/wayne/OneDrive - University of Massachusetts Lowell - UMass Lowell/Autumn WTs/sequence_1265.mat"
)["h500_1265"]
mslp = sio.loadmat(
    "C:/Users/wayne/OneDrive - University of Massachusetts Lowell - UMass Lowell/Autumn WTs/sequence_1265.mat"
)["mslp_1265"]
u850 = sio.loadmat(
    "C:/Users/wayne/OneDrive - University of Massachusetts Lowell - UMass Lowell/Autumn WTs/sequence_1265.mat"
)["u850_1265"]
v850 = sio.loadmat(
    "C:/Users/wayne/OneDrive - University of Massachusetts Lowell - UMass Lowell/Autumn WTs/sequence_1265.mat"
)["v850_1265"]
t = sio.loadmat(
    "C:/Users/wayne/OneDrive - University of Massachusetts Lowell - UMass Lowell/Autumn WTs/sequence_1265.mat"
)["t_1265"]
p = sio.loadmat(
    "C:/Users/wayne/OneDrive - University of Massachusetts Lowell - UMass Lowell/Autumn WTs/sequence_1265.mat"
)["p_1265"]

h500 = h500[[0, 3, 1, 2], :]
u850 = u850[[0, 3, 1, 2], :]
v850 = v850[[0, 3, 1, 2], :]
mslp = mslp[[0, 3, 1, 2], :]
t = t[[0, 3, 1, 2], :]
p = p[[0, 3, 1, 2], :]

i = 0
count = 0
# Plot the images for each of the clusters
axlist = make_grid(2, 4, cbar=False)
while i <= 7:
    ax = axlist[i]
    # Get our LAT/LON grid
    lats = Dataset("F:/ERA5/h500_son/h5001979.nc").variables["latitude"][160:241]
    lons = Dataset("F:/ERA5/h500_son/h5001979.nc").variables["longitude"][1080:1201]
    LON, LAT = np.meshgrid(lons, lats)

    lonsp = Dataset("F:/persiann-cdr/files/PERSIANN-CDR_v01r01_19830101.nc").variables[
        "lon"
    ][1080:1201]
    latsp = Dataset("F:/persiann-cdr/files/PERSIANN-CDR_v01r01_19830101.nc").variables[
        "lat"
    ][39:120]
    LONP, LATP = np.meshgrid(lonsp, latsp)

    ylocs = np.arange(30, 50, 5)
    xlocs = np.arange(lons[0] - 360, 10 + lons[-1] - 360, 10)

    if i <= 3:

        S = mslp[i, :, :] / 100
        H = h500[i, :, :] / 10
        U = u850[i, :, :]
        V = v850[i, :, :]
        T = t[i, :, :]
        P = p[i, :, :]

        levels = np.arange(504, 618, 6)
        levels_p = np.arange(0, 20, 2)
        levels3 = np.arange(-8, 8, 0.5)
        # levels = np.arange(480,620,5)
        # levels = np.arange(-20,20,2)
        levels1 = np.arange(960, 1028, 4)

        Z_500 = ndimage.gaussian_filter(S[:, :], sigma=3, order=0)
        cd = ax.contour(LON, LAT, Z_500[:, :], levels=levels1, colors="red")
        plt.clabel(cd, inline=True, fmt="%1.0f", fontsize=14, colors="red")
        # cd = ax.contour(LON,LAT,H[:,:], levels = levels, colors = 'black')
        # ax.clabel(cd, inline=True, fmt='%1.0f', fontsize=12, colors='black')
        # cs = ax.contourf(LON, LAT, T2, levels3, cmap='bwr', extend='both')
        # plt.colorbar(cs, shrink=0.86)
        cs1 = ax.contourf(LONP, LATP, P, levels3, cmap=double, extend="both")
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
        if i == 0:
            title = "1"
        elif i == 1:
            title = "2"
        elif i == 2:
            title = "6"
        else:
            title = "5"

        ax.set_title(title, fontsize=16)

    if i > 3:
        j = i - 4
        S = mslp[j, :, :] / 100
        H = h500[j, :, :] / 10
        U = u850[j, :, :]
        V = v850[j, :, :]
        T = t[j, :, :]

        levels = np.arange(504, 618, 6)
        levels_p = np.arange(0, 20, 2)
        levels3 = np.arange(-4, 4, 0.5)
        # levels = np.arange(480,620,5)
        # levels = np.arange(-20,20,2)
        levels1 = np.arange(960, 1028, 4)

        # Z_500 = ndimage.gaussian_filter(S[:, :], sigma=3, order=0)
        # cd = ax.contour(LON, LAT, Z_500[:, :], levels=levels1, colors='blue')
        # plt.clabel(cd, inline=True, fmt='%1.0f', fontsize=14, colors='blue')
        cd = ax.contour(LON, LAT, H[:, :], levels=levels, colors="black")
        ax.clabel(cd, inline=True, fmt="%1.0f", fontsize=14, colors="black")
        cs = ax.contourf(LON, LAT, T.T, levels3, cmap="bwr", extend="both")
        if i == 7:
            plt.colorbar(cs, ax=axlist[4:], orientation="horizontal", shrink=0.50)
            plt.colorbar(cs1, ax=axlist[0:4], orientation="horizontal", shrink=0.50)
        # cs = ax.contourf(LON_P, LAT_P, P.T, levels3, cmap='bwr', extend='both')
        # plt.clabel(cs, inline=True, fmt='%1.0f', fontsize=12, colors='black')
        # ax.quiver(LON[::5, ::5],LAT[::5, ::5],U[::5,::5],V[::5,::5],scale=None, scale_units='inches')

        states = NaturalEarthFeature(
            category="cultural",
            scale="50m",
            facecolor="none",
            name="admin_1_states_provinces_shp",
        )
        ax.add_feature(states, linewidth=1.0, edgecolor="black")
        ax.coastlines("50m", linewidth=0.8)
        ax.gridlines(linestyle="dotted", ylocs=lats[::20], xlocs=lons[::20])

        if i == 4:
            title = "1"
        elif i == 5:
            title = "2"
        elif i == 6:
            title = "6"
        else:
            title = "5"
        ax.set_title(title, fontsize=15)

    i = i + 1

# plt.suptitle("850 hPa Wind, MSLP, and 500 hPa Height", fontsize=15, fontweight='bold')
title1 = outdir + "/sequence1-2-6-5"
plt.suptitle("Sequence E1 (1-2-6-5)", fontsize=20)
plt.savefig(title1, bbox_inches="tight")
