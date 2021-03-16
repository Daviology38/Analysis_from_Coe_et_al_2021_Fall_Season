#Import a bunch of stuff
import os
os.environ['PROJ_LIB'] = r'c:\Users\CoeFamily\Anaconda3\envs\meteorology\Library\share'
import numpy as np
import matplotlib as plt
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
import datetime
import pickle
import csv
import scipy as sp
import scipy.ndimage as ndimage
import scipy.io as sio
from sklearn.metrics import mean_squared_error
from math import sqrt
import xarray as xy

#Check to see if out directory exists and if not, make it
outdir = "SON_WT/precip_pics"
if not os.path.exists(outdir):
    os.makedirs(outdir)

df = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Coe_clusters_new.csv")
df.columns = ["Day","Clust"]
vals = df["Clust"].values
clustvals = list(map(int,vals))

ci = 7

precip_total = np.zeros((3640,122,82))
#lats [39:121], lons [959:1081]

month = ['09','10','11']
day_tot = [30,31,30]
year = np.arange(1979,2019,1)

count = 0
for y in year:
    for m, d in zip(month,day_tot):
        for i in range(d):
            if d < 10:
                d = "0" + str(d)
            else:
                d = str(d)

            precip_total[count,:,:] = xr.open_dataset("H:/persiann-cdr/files/PERSIANN-CDR_v01r01_" + str(y) + m + d + ".nc").precipitation.values[:,959:1081,39:121]
            lats_precip = xr.open_dataset("H:/persiann-cdr/files/PERSIANN-CDR_v01r01_" + str(y) + m + d + ".nc").precipitation.lat.values[39:121]
            lons_precip = xr.open_dataset("H:/persiann-cdr/files/PERSIANN-CDR_v01r01_" + str(y) + m + d + ".nc").precipitation.lon.values[959:1081]
            count = count + 1

i = 1
#Plot the images for each of the clusters
while i <= ci:
    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=crs.PlateCarree())
    # Get our LAT/LON grid
    lats = Dataset("H:/ERA5/h500_son/h5001979.nc").variables['latitude'][160:241]
    lons = Dataset("H:/ERA5/h500_son/h5001979.nc").variables['longitude'][1080:1201]
    LON, LAT = np.meshgrid(lons, lats)
    LON_P, LAT_P = np.meshgrid(lons_precip, lats_precip)

    ylocs = np.arange(30, 50, 5)
    xlocs = np.arange(lons[0]-360, 10+lons[-1]-360, 10)

    # Load in the data using xarray
    h500 = xr.open_dataset('H:/ERA5/h500_son_dailyavg.nc')
    mslp = xr.open_dataset('H:/ERA5/mslp_son_daily_avg.nc')
    u850 = xr.open_dataset('H:/ERA5/u850_son_daily_avg.nc')
    v850 = xr.open_dataset('H:/ERA5/v850_son_daily_avg.nc')

    ind = [ij for ij,val in enumerate(ldf) if val == i-1]
    S = mslp.msl[ind,:,:].mean("time")
    H = h500.z[ind,:,:].mean("time")
    U = u850.u[ind,:,:].mean("time")
    V = v850.v[ind,:,:].mean("time")
    P = np.mean(precip_total[ind,:,:],axis=0)
    levels = np.arange(480,620,5)
    levels_p = np.arange(0,40,5)
    levels1 = np.arange(960,1028,2)
    Z_500 = ndimage.gaussian_filter(S[:,:].values/100, sigma=3, order=0)
    cd = ax.contour(LON,LAT,Z_500[:,:], levels = levels1, colors='blue')
    plt.clabel(cd, inline=True, fmt='%1.0f', fontsize=12, colors='blue')
    cd = ax.contour(LON,LAT,H[:,:].values/10, levels = levels, colors = 'black')
    ax.clabel(cd, inline=True, fmt='%1.0f', fontsize=10, colors='black')
    ax.quiver(LON[::5, ::5],LAT[::5, ::5],U[::5,::5].values,V[::5,::5].values,scale=None, scale_units='inches')
    cs = ax.contourf(LON_P,LAT_P,P,levels_p, cmap='Greens',extend = 'both')
    #plt.clabel(cs, inline=True, fmt='%1.0f', fontsize=12, colors='black')
    plt.colorbar(cs,shrink=0.86)
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

    title = 'Cluster ' + str(wtt)
    ax.set_title(title, fontsize=20)

    title1 = outdir + '/' + str(wtt)


    plt.savefig(title1, bbox_inches='tight')

    plt.gcf().clear()

    i = i + 1