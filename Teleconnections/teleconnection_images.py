import pandas as pd
import numpy as np
import xarray as xr
from cartopy import crs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec
import os
from netCDF4 import Dataset
import matplotlib.pyplot as plt

# Create 3X3 sub plots
gs = gridspec.GridSpec(3, 3)

# Import the data
dff = pd.read_csv("teleconnections_total.csv")

df = pd.read_csv("clust_vals.csv")
df.columns = ["C", "Clust"]
vals = df["Clust"].values
clustvals = list(map(int, vals))
clustvals = [x + 1 for x in clustvals]
clustvals = [
    2
    if x == 1
    else 5
    if x == 2
    else 7
    if x == 3
    else 1
    if x == 4
    else 4
    if x == 5
    else 3
    if x == 6
    else 6
    for x in clustvals
]
ldf = clustvals

# Make the directory to hold the images

# Check to see if out directory exists and if not, make it
outdir = "F:/son_teleconnection_images"

if not os.path.exists(outdir):
    os.makedirs(outdir)

# Load in the data
# Get our LAT/LON grid
lats = Dataset("F:/ERA5/h500_son/h5001979.nc").variables["latitude"][160:241]
lons = Dataset("F:/ERA5/h500_son/h5001979.nc").variables["longitude"][1080:1201]
LON, LAT = np.meshgrid(lons, lats)

# Initialize our arrays to hold the variables
h500 = np.zeros((3640, 81, 121))
mslp = np.zeros((3640, 81, 121))
u850 = np.zeros((3640, 81, 121))
v850 = np.zeros((3640, 81, 121))
h500m = np.zeros((81, 121))
mslpm = np.zeros((81, 121))
u850m = np.zeros((81, 121))
v850m = np.zeros((81, 121))

# Initialize our counting variable
count = 0

# Fill our arrays with data
for i in range(1979, 2019):
    # put together strings for the filenames based on the year
    start = "F:/ERA5/h500_son/h500"
    start2 = "F:/ERA5/u850_son/u850"
    start3 = "F:/ERA5/v850_son/v850"
    start4 = "F:/ERA5/mslp_son/mslp"
    next = ".nc"
    filename = start + str(i) + next
    filename2 = start2 + str(i) + next
    filename3 = start3 + str(i) + next
    filename4 = start4 + str(i) + next

    # now we fill them. Take 12z data for each day for each year
    j = 11
    while j <= len(Dataset(filename).variables["z"][:, 0, 0]):
        # Put each data array into a temp array
        tmp = Dataset(filename).variables["z"][j, 160:241, 1080:1201]
        tmph = tmp
        tmp = Dataset(filename2).variables["u"][j, 160:241, 1080:1201]
        tmpu = tmp
        tmp = Dataset(filename3).variables["v"][j, 160:241, 1080:1201]
        tmpv = tmp
        tmp = Dataset(filename4).variables["msl"][j, 160:241, 1080:1201]
        tmpm = tmp
        for a in range(121):
            for b in range(81):
                # Fill the arrays with the data from the temp array
                # Fill the mean arrays as well with the sum of the data
                h500[count, b, a] = tmph[b, a] / 98.1
                mslp[count, b, a] = tmpm[b, a] / 100
                u850[count, b, a] = tmpu[b, a]
                v850[count, b, a] = tmpv[b, a]
                h500m[b, a] = h500m[b, a] + tmph[b, a] / 98.1
                mslpm[b, a] = mslpm[b, a] + tmpm[b, a] / 100
                u850m[b, a] = u850m[b, a] + tmpu[b, a]
                v850m[b, a] = v850m[b, a] + tmpv[b, a]
        count = count + 1
        j = j + 24

# Now take the mean arrays and divide by the count to get the seasonal mean

h500m = h500m / count
mslpm = mslpm / count
u850m = u850m / count
v850m = v850m / count

# Now find the data for each WT
wt_list = []

# for i in range(1,8,1):
#     wt_list.append(np. nanmean(h500[[ k for k,ind in enumerate(ldf) if ind ==i],:,:],axis=0) - h500m)
#     wt_list.append(np.nanmean(mslp[[ k for k,ind in enumerate(ldf) if ind ==i],:,:],axis=0) - mslpm)
#
# #Now do the same for each teleconnection
# ao_type = dff.AO.values
# nao_type = dff.NAO.values
# pna_type = dff.PNA.values
# nino_type = dff.NINO.values
#
# teleconnection_list = []
#
# teleconnection_list.append(np.nanmean(h500[ao_type=="Positive"],axis=0) - h500m)
# teleconnection_list.append(np.nanmean(h500[ao_type=="Neutral"],axis=0) - h500m)
# teleconnection_list.append(np.nanmean(h500[ao_type=="Negative"],axis=0) - h500m)
#
# teleconnection_list.append(np.nanmean(h500[nao_type=="Positive"],axis=0) - h500m)
# teleconnection_list.append(np.nanmean(h500[nao_type=="Neutral"],axis=0) - h500m)
# teleconnection_list.append(np.nanmean(h500[nao_type=="Negative"],axis=0) - h500m)
#
# teleconnection_list.append(np.nanmean(h500[pna_type=="Positive"],axis=0) - h500m)
# teleconnection_list.append(np.nanmean(h500[pna_type=="Neutral"],axis=0) - h500m)
# teleconnection_list.append(np.nanmean(h500[pna_type=="Negative"],axis=0) - h500m)
#
# teleconnection_list.append(np.nanmean(h500[nino_type=="Positive"],axis=0) - h500m)
# teleconnection_list.append(np.nanmean(h500[nino_type=="Neutral"],axis=0) - h500m)
# teleconnection_list.append(np.nanmean(h500[nino_type=="Negative"],axis=0) - h500m)
#
# teleconnection_list.append(np.nanmean(mslp[ao_type=="Positive"],axis=0) - mslpm)
# teleconnection_list.append(np.nanmean(mslp[ao_type=="Neutral"],axis=0) - mslpm)
# teleconnection_list.append(np.nanmean(mslp[ao_type=="Negative"],axis=0) - mslpm)
#
# teleconnection_list.append(np.nanmean(mslp[nao_type=="Positive"],axis=0) - mslpm)
# teleconnection_list.append(np.nanmean(mslp[nao_type=="Neutral"],axis=0) - mslpm)
# teleconnection_list.append(np.nanmean(mslp[nao_type=="Negative"],axis=0) - mslpm)
#
# teleconnection_list.append(np.nanmean(mslp[pna_type=="Positive"],axis=0) - mslpm)
# teleconnection_list.append(np.nanmean(mslp[pna_type=="Neutral"],axis=0) - mslpm)
# teleconnection_list.append(np.nanmean(mslp[pna_type=="Negative"],axis=0) - mslpm)
#
# teleconnection_list.append(np.nanmean(mslp[nino_type=="Positive"],axis=0) - mslpm)
# teleconnection_list.append(np.nanmean(mslp[nino_type=="Neutral"],axis=0) - mslpm)
# teleconnection_list.append(np.nanmean(mslp[nino_type=="Negative"],axis=0) - mslpm)
#
# #Now plot the anomaly of each WT to each teleconnection
# names = ["+AO","~AO","-AO","+NAO","~NAO","-NAO","+PNA","~PNA","-PNA","+NINO","~NINO","-NINO"]
# for j in range(12):
#
#     #set up the axes
#     fig= plt.figure()
#     ax1 = fig.add_subplot(gs[0,0],projection=crs.PlateCarree())
#     ax2 = fig.add_subplot(gs[0,1],projection=crs.PlateCarree())
#     ax3 = fig.add_subplot(gs[0,2],projection=crs.PlateCarree())
#     ax4 = fig.add_subplot(gs[1,0],projection=crs.PlateCarree())
#     ax5 = fig.add_subplot(gs[1,1],projection=crs.PlateCarree())
#     ax6 = fig.add_subplot(gs[1,2],projection=crs.PlateCarree())
#     ax7 = fig.add_subplot(gs[2,1],projection=crs.PlateCarree())
#
#
#     # Get our LAT/LON grid
#     lats = Dataset("F:/ERA5/h500_son/h5001979.nc").variables['latitude'][160:241]
#     lons = Dataset("F:/ERA5/h500_son/h5001979.nc").variables['longitude'][1080:1201]
#     LON, LAT = np.meshgrid(lons, lats)
#
#     ylocs = np.arange(30, 50, 5)
#     xlocs = np.arange(lons[0] - 360, 10 + lons[-1] - 360, 10)
#     levels = np.arange(-20,20,4)
#     cd = ax1.contour(LON, LAT, wt_list[0] - teleconnection_list[j], levels=levels, colors='black')
#     plt.clabel(cd, inline=True, fmt='%1.0f', fontsize=12, colors='black')
#     cs = ax1.contourf(LON,LAT,wt_list[1] - teleconnection_list[j+12],levels, cmap='bwr',extend = 'both')
#     # plt.colorbar(cs,shrink=0.86,cax=ax1)
#     states = NaturalEarthFeature(category='cultural',
#                                  scale='50m',
#                                  facecolor='none',
#                                  name='admin_1_states_provinces_shp')
#     ax1.add_feature(states, linewidth=1., edgecolor="black")
#     ax1.coastlines('50m', linewidth=0.8)
#     ax1.gridlines(linestyle='dotted', ylocs=lats[::20], xlocs=lons[::20])
#     ax1.set_title("WT1 " + names[j] + " Anomaly", fontsize=20)
#     fig.colorbar(cs, ax=ax1)
#
#     cd = ax2.contour(LON, LAT, wt_list[2] - teleconnection_list[j], levels=levels, colors='black')
#     plt.clabel(cd, inline=True, fmt='%1.0f', fontsize=12, colors='black')
#     cs = ax2.contourf(LON,LAT,wt_list[3] - teleconnection_list[j+12], cmap='bwr',extend = 'both')
#     #plt.colorbar(cs,shrink=0.86,cax=ax2)
#     states = NaturalEarthFeature(category='cultural',
#                                  scale='50m',
#                                  facecolor='none',
#                                  name='admin_1_states_provinces_shp')
#     ax2.add_feature(states, linewidth=1., edgecolor="black")
#     ax2.coastlines('50m', linewidth=0.8)
#     ax2.gridlines(linestyle='dotted', ylocs=lats[::20], xlocs=lons[::20])
#     ax2.set_title("WT2 " + names[j] + " Anomaly", fontsize=20)
#     fig.colorbar(cs, ax=ax2)
#
#     cd = ax3.contour(LON, LAT, wt_list[4] - teleconnection_list[j], levels=levels, colors='black')
#     plt.clabel(cd, inline=True, fmt='%1.0f', fontsize=12, colors='black')
#     cs = ax3.contourf(LON,LAT,wt_list[5] - teleconnection_list[j+12], cmap='bwr',extend = 'both')
#     #plt.colorbar(cs,shrink=0.86,cax=ax3)
#     states = NaturalEarthFeature(category='cultural',
#                                  scale='50m',
#                                  facecolor='none',
#                                  name='admin_1_states_provinces_shp')
#     ax3.add_feature(states, linewidth=1., edgecolor="black")
#     ax3.coastlines('50m', linewidth=0.8)
#     ax3.gridlines(linestyle='dotted', ylocs=lats[::20], xlocs=lons[::20])
#     ax3.set_title("WT3 " + names[j] + " Anomaly", fontsize=20)
#     fig.colorbar(cs, ax=ax3)
#
#     cd = ax4.contour(LON, LAT, wt_list[6] - teleconnection_list[j], levels=levels, colors='black')
#     plt.clabel(cd, inline=True, fmt='%1.0f', fontsize=12, colors='black')
#     cs = ax4.contourf(LON,LAT,wt_list[7] - teleconnection_list[j+12], cmap='bwr',extend = 'both')
#     #plt.colorbar(cs,shrink=0.86,cax=ax4)
#     states = NaturalEarthFeature(category='cultural',
#                                  scale='50m',
#                                  facecolor='none',
#                                  name='admin_1_states_provinces_shp')
#     ax4.add_feature(states, linewidth=1., edgecolor="black")
#     ax4.coastlines('50m', linewidth=0.8)
#     ax4.gridlines(linestyle='dotted', ylocs=lats[::20], xlocs=lons[::20])
#     ax4.set_title("WT4 " + names[j] + " Anomaly", fontsize=20)
#     fig.colorbar(cs, ax=ax4)
#
#     cd = ax5.contour(LON, LAT, wt_list[8] - teleconnection_list[j], levels=levels, colors='black')
#     plt.clabel(cd, inline=True, fmt='%1.0f', fontsize=12, colors='black')
#     cs = ax5.contourf(LON,LAT,wt_list[9] - teleconnection_list[j+12], cmap='bwr',extend = 'both')
#     #plt.colorbar(cs,shrink=0.86,cax=ax5)
#     states = NaturalEarthFeature(category='cultural',
#                                  scale='50m',
#                                  facecolor='none',
#                                  name='admin_1_states_provinces_shp')
#     ax5.add_feature(states, linewidth=1., edgecolor="black")
#     ax5.coastlines('50m', linewidth=0.8)
#     ax5.gridlines(linestyle='dotted', ylocs=lats[::20], xlocs=lons[::20])
#     ax5.set_title("WT5 " + names[j] + " Anomaly", fontsize=20)
#     fig.colorbar(cs, ax=ax5)
#
#     cd = ax6.contour(LON, LAT, wt_list[10] - teleconnection_list[j], levels=levels, colors='black')
#     plt.clabel(cd, inline=True, fmt='%1.0f', fontsize=12, colors='black')
#     cs = ax6.contourf(LON,LAT,wt_list[11] - teleconnection_list[j+12], cmap='bwr',extend = 'both')
#     states = NaturalEarthFeature(category='cultural',
#                                  scale='50m',
#                                  facecolor='none',
#                                  name='admin_1_states_provinces_shp')
#     ax6.add_feature(states, linewidth=1., edgecolor="black")
#     ax6.coastlines('50m', linewidth=0.8)
#     ax6.gridlines(linestyle='dotted', ylocs=lats[::20], xlocs=lons[::20])
#     ax6.set_title("WT6 " + names[j] + " Anomaly", fontsize=20)
#     fig.colorbar(cs, ax=ax6)
#
#     cd = ax7.contour(LON, LAT, wt_list[12] - teleconnection_list[j], levels=levels, colors='black')
#     plt.clabel(cd, inline=True, fmt='%1.0f', fontsize=12, colors='black')
#     cs = ax7.contourf(LON,LAT,wt_list[13] - teleconnection_list[j+12], cmap='bwr',extend = 'both')
#     #plt.colorbar(cs,shrink=0.86,cax=ax7)
#     states = NaturalEarthFeature(category='cultural',
#                                  scale='50m',
#                                  facecolor='none',
#                                  name='admin_1_states_provinces_shp')
#     ax7.add_feature(states, linewidth=1., edgecolor="black")
#     ax7.coastlines('50m', linewidth=0.8)
#     ax7.gridlines(linestyle='dotted', ylocs=lats[::20], xlocs=lons[::20])
#     ax7.set_title("WT7 " + names[j] + " Anomaly", fontsize=20)
#     fig.colorbar(cs, ax=ax7)
#
#     plt.savefig(names[j] + "anomaly_son.png",bbox_inches="tight")
