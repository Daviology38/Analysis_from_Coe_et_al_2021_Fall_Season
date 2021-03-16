# -*- coding: utf-8 -*-
"""
Created on Sat Feb 09 11:46:48 2019

@author: CoeFamily
"""

import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import scipy.ndimage as ndimage
import scipy.io as sio
import math
from sklearn.metrics import mean_squared_error
import pandas as pd



#Function to find the 2d correlation coefficient
def mean2(x):
    y = np.sum(x) / np.size(x)
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)
    
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r



#with open('/media/mariofire/4TBExternal/era5_son/seasonmean_persistence_dates.txt') as f:
#    content = f.readlines()
#p = 0
#dates = np.zeros((65,3))
#while p < 65:
#    x = content[p].split('\t')
#    dates[p][0] = int(x[0])
#    dates[p][1] = int(x[1][0])
#    dates[p][2] = int(x[2][0])
#    if(int(dates[p][1]) == 1):
#        dates[p][1] = int(x[1][0:2])
#    p = p + 1

#Make directory to save data files to if it doesn't already exist
outdir = "era5_son_new/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

#pick number of clusters to analyze
clustnum = 7

#load in the data
#Define our missing data
missing = 1 * 10**15

#open the overall data file
#Load in the Matlab Files
mat_contents = sio.loadmat('C:/Users/CoeFamily/Documents/Research/era5_son_oct2019_run/era5_son_new/era5_sep.mat')
mat_contents3 = sio.loadmat('C:/Users/CoeFamily/Documents/Research/era5_son_oct2019_run/era5_son_new/era5_oct.mat')
mat_contents4 = sio.loadmat('C:/Users/CoeFamily/Documents/Research/era5_son_oct2019_run/era5_son_new/era5_nov.mat')
mat_contents2 = sio.loadmat('H:/era5_son_oct2019_run/era5_son_new/CI_results.mat')
#mat_contents2 = sio.loadmat('C:/Users/CoeFamily/Documents/MATLAB/SON_new/SONdailymean_95/CI_results.mat')

#Put Variables into an array (change the 3 letter month to the month you need)
h500oct = mat_contents3['h500oct']
h500sep = mat_contents['h500sep']
h500nov = mat_contents4['h500nov']
mslpoct = mat_contents3['mslpoct']
mslpsep = mat_contents['mslpsep']
mslpnov = mat_contents4['mslpnov']
uoct = mat_contents3['u850oct']
usep = mat_contents['u850sep']
unov = mat_contents4['u850nov']
voct = mat_contents3['v850oct']
vsep = mat_contents['v850sep']
vnov = mat_contents4['v850nov']
K = mat_contents2['K']
K = K[:,6]

#combine the h500 arrays
h500 = np.zeros((3640,h500oct.shape[1],h500oct.shape[2]))
mslp = np.zeros((3640,h500oct.shape[1],h500oct.shape[2]))
u = np.zeros((3640,h500oct.shape[1],h500oct.shape[2]))
v = np.zeros((3640,h500oct.shape[1],h500oct.shape[2]))

i = 0
y = 0
countsep = 0
countoct = 0
countnov = 0

while i <=3639:
    if y <=29:
        h500[i][:][:] = h500sep[countsep][:][:]
        mslp[i][:][:] = mslpsep[countsep][:][:]
        u[i][:][:] = usep[countsep][:][:]
        v[i][:][:] = vsep[countsep][:][:]
        y = y + 1
        countsep = countsep + 1
    elif (y > 29 and y <=60):
        h500[i][:][:] = h500oct[countoct][:][:]
        mslp[i][:][:] = mslpoct[countoct][:][:]
        u[i][:][:] = uoct[countoct][:][:]
        v[i][:][:] = voct[countoct][:][:]
        y = y + 1
        countoct = countoct + 1
    else:
        h500[i][:][:] = h500nov[countnov][:][:]
        mslp[i][:][:] = mslpnov[countnov][:][:]
        u[i][:][:] = unov[countnov][:][:]
        v[i][:][:] = vnov[countnov][:][:]
        y = y + 1
        countnov = countnov + 1
        
    if y == 91:
        y = 0
        
    i = i + 1    
    
#Create the anomaly fields for each day
#Start by using the seasonal mean of H500

h500a = h500.mean(axis=0)  
mslpa = mslp.mean(axis=0)
ua = u.mean(axis=0)
va = v.mean(axis=0) 

#Next separate the data by WT

i = 0
wt1h = np.zeros((81,121))
wt2h = np.zeros((81,121))
wt3h = np.zeros((81,121))
wt4h = np.zeros((81,121))
wt5h = np.zeros((81,121))
wt6h = np.zeros((81,121))
wt7h = np.zeros((81,121))
wt8h = np.zeros((81,121))
wt9h = np.zeros((81,121))
wt1m = np.zeros((81,121))
wt2m = np.zeros((81,121))
wt3m = np.zeros((81,121))
wt4m = np.zeros((81,121))
wt5m = np.zeros((81,121))
wt6m = np.zeros((81,121))
wt7m = np.zeros((81,121))
wt8m = np.zeros((81,121))
wt9m = np.zeros((81,121))
wt1u = np.zeros((81,121))
wt2u = np.zeros((81,121))
wt3u = np.zeros((81,121))
wt4u = np.zeros((81,121))
wt5u = np.zeros((81,121))
wt6u = np.zeros((81,121))
wt7u = np.zeros((81,121))
wt8u = np.zeros((81,121))
wt9u = np.zeros((81,121))
wt1v = np.zeros((81,121))
wt2v = np.zeros((81,121))
wt3v = np.zeros((81,121))
wt4v = np.zeros((81,121))
wt5v = np.zeros((81,121))
wt6v = np.zeros((81,121))
wt7v = np.zeros((81,121))
wt8v = np.zeros((81,121))
wt9v = np.zeros((81,121))
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
count8 = 0
count9 = 0

while i <= 3639:
    if(K[i] == 1):
        wt1h[:][:] = wt1h[:][:] + h500[i][:][:]
        wt1m[:][:] = wt1m[:][:] + mslp[i][:][:]
        wt1u[:][:] = wt1u[:][:] + u[i][:][:]
        wt1v[:][:] = wt1v[:][:] + v[i][:][:]
        count1 = count1 + 1
    elif(K[i] == 2):
        wt2h[:][:] = wt2h[:][:] + h500[i][:][:]
        wt2m[:][:] = wt2m[:][:] + mslp[i][:][:]
        wt2u[:][:] = wt2u[:][:] + u[i][:][:]
        wt2v[:][:] = wt2v[:][:] + v[i][:][:]
        count2 = count2 + 1
    elif(K[i] == 3):
        wt3h[:][:] = wt3h[:][:] + h500[i][:][:]
        wt3m[:][:] = wt3m[:][:] + mslp[i][:][:]
        wt3u[:][:] = wt3u[:][:] + u[i][:][:]
        wt3v[:][:] = wt3v[:][:] + v[i][:][:]
        count3 = count3 + 1
    elif(K[i] == 4):
        wt4h[:][:] = wt4h[:][:] + h500[i][:][:]
        wt4m[:][:] = wt4m[:][:] + mslp[i][:][:]
        wt4u[:][:] = wt4u[:][:] + u[i][:][:]
        wt4v[:][:] = wt4v[:][:] + v[i][:][:]
        count4 = count4 + 1
    elif(K[i] == 5):
        wt5h[:][:] = wt5h[:][:] + h500[i][:][:]
        wt5m[:][:] = wt5m[:][:] + mslp[i][:][:]
        wt5u[:][:] = wt5u[:][:] + u[i][:][:]
        wt5v[:][:] = wt5v[:][:] + v[i][:][:]
        count5 = count5 + 1
    elif(K[i] == 6):
        wt6h[:][:] = wt6h[:][:] + h500[i][:][:]
        wt6m[:][:] = wt6m[:][:] + mslp[i][:][:]
        wt6u[:][:] = wt6u[:][:] + u[i][:][:]
        wt6v[:][:] = wt6v[:][:] + v[i][:][:]
        count6 = count6 + 1
    else:
        wt7h[:][:] = wt7h[:][:] + h500[i][:][:]
        wt7m[:][:] = wt7m[:][:] + mslp[i][:][:]
        wt7u[:][:] = wt7u[:][:] + u[i][:][:]
        wt7v[:][:] = wt7v[:][:] + v[i][:][:]
        count7 = count7 + 1
#    elif(K[i] == 7):
#        wt7h[:][:] = wt7h[:][:] + h500[i][:][:]
#        wt7m[:][:] = wt7m[:][:] + mslp[i][:][:]
#        wt7u[:][:] = wt7u[:][:] + u[i][:][:]
#        wt7v[:][:] = wt7v[:][:] + v[i][:][:]
#        count7 = count7 + 1
#    elif(K[i] == 8):
#        wt8h[:][:] = wt8h[:][:] + h500[i][:][:]
#        wt8m[:][:] = wt8m[:][:] + mslp[i][:][:]
#        wt8u[:][:] = wt8u[:][:] + u[i][:][:]
#        wt8v[:][:] = wt8v[:][:] + v[i][:][:]
#        count8 = count8 + 1
#    else:
#        wt9h[:][:] = wt9h[:][:] + h500[i][:][:]
#        wt9m[:][:] = wt9m[:][:] + mslp[i][:][:]
#        wt9u[:][:] = wt9u[:][:] + u[i][:][:]
#        wt9v[:][:] = wt9v[:][:] + v[i][:][:]
#        count9 = count9 + 1
    i = i + 1
    
wt1h = wt1h / count1
wt2h = wt2h / count2 
wt3h = wt3h / count3
wt4h = wt4h / count4
wt5h = wt5h / count5
wt6h = wt6h / count6
wt7h = wt7h / count7
#wt8h = wt8h / count8
#wt9h = wt9h / count9
wt1m = wt1m / count1
wt2m = wt2m / count2 
wt3m = wt3m / count3
wt4m = wt4m / count4
wt5m = wt5m / count5
wt6m = wt6m / count6
wt7m = wt7m / count7
#wt8m = wt8m / count8
#wt9m = wt9m / count9
wt1u = wt1u / count1
wt2u = wt2u / count2 
wt3u = wt3u / count3
wt4u = wt4u / count4
wt5u = wt5u / count5
wt6u = wt6u / count6
wt7u = wt7u / count7
#wt8u = wt8u / count8
#wt9u = wt9u / count9
wt1v = wt1v / count1
wt2v = wt2v / count2 
wt3v = wt3v / count3
wt4v = wt4v / count4
wt5v = wt5v / count5
wt6v = wt6v / count6
wt7v = wt7v / count7
#wt8v = wt8v / count8
#wt9v = wt9v / count9

#make the anomaly arrays
i = 0
h500aoverall = np.zeros((3640,81,121))
mslpaoverall = np.zeros((3640,81,121))
uaoverall = np.zeros((3640,81,121))
vaoverall = np.zeros((3640,81,121))


while i <= 3639:
    h500aoverall[i][:][:] = h500[i][:][:] - h500a[:][:]
    mslpaoverall[i][:][:] = mslp[i][:][:] - mslpa[:][:]
    uaoverall[i][:][:] = u[i][:][:] - ua[:][:]
    vaoverall[i][:][:] = v[i][:][:] - va[:][:]
    i = i + 1
dateindices = np.zeros((3640,5))
wt1ah = wt1h[:][:] - h500a[:][:]
wt2ah = wt2h[:][:] - h500a[:][:]
wt3ah = wt3h[:][:] - h500a[:][:]
wt4ah = wt4h[:][:] - h500a[:][:]
wt5ah = wt5h[:][:] - h500a[:][:]
wt6ah = wt6h[:][:] - h500a[:][:]
wt7ah = wt7h[:][:] - h500a[:][:]  
#wt8ah = wt8h[:][:] - h500a[:][:]
#wt9ah = wt9h[:][:] - h500a[:][:]  
wt1am = wt1m[:][:] - mslpa[:][:]
wt2am = wt2m[:][:] - mslpa[:][:]
wt3am = wt3m[:][:] - mslpa[:][:]
wt4am = wt4m[:][:] - mslpa[:][:]
wt5am = wt5m[:][:] - mslpa[:][:]
wt6am = wt6m[:][:] - mslpa[:][:]
wt7am = wt7m[:][:] - mslpa[:][:]  
#wt8am = wt8m[:][:] - mslpa[:][:]
#wt9am = wt9m[:][:] - mslpa[:][:]  
wt1au = wt1u[:][:] - ua[:][:]
wt2au = wt2u[:][:] - ua[:][:]
wt3au = wt3u[:][:] - ua[:][:]
wt4au = wt4u[:][:] - ua[:][:]
wt5au = wt5u[:][:] - ua[:][:]
wt6au = wt6u[:][:] - ua[:][:]
wt7au = wt7u[:][:] - ua[:][:]  
#wt8au = wt8u[:][:] - ua[:][:]
#wt9au = wt9u[:][:] - ua[:][:]  
wt1av = wt1v[:][:] - va[:][:]
wt2av = wt2v[:][:] - va[:][:]
wt3av = wt3v[:][:] - va[:][:]
wt4av = wt4v[:][:] - va[:][:]
wt5av = wt5v[:][:] - va[:][:]
wt6av = wt6v[:][:] - va[:][:]
wt7av = wt7v[:][:] - va[:][:]  
#wt8av = wt8v[:][:] - va[:][:]
#wt9av = wt9v[:][:] - va[:][:]  

#Now compute the correlation, rmse and bias for each day of the SON season
corrsh = np.empty(shape=[3640,9])
rmseh = np.empty(shape=[3640,9])
biash = np.empty(shape=[3640,9])
corrsm = np.empty(shape=[3640,9])
rmsem = np.empty(shape=[3640,9])
biasm = np.empty(shape=[3640,9])
corrsu = np.empty(shape=[3640,9])
rmseu = np.empty(shape=[3640,9])
biasu = np.empty(shape=[3640,9])
corrsv = np.empty(shape=[3640,9])
rmsev = np.empty(shape=[3640,9])
biasv = np.empty(shape=[3640,9])

corrsh2 = np.empty(shape=[3640,9])
rmseh2 = np.empty(shape=[3640,9])
biash2 = np.empty(shape=[3640,9])
corrsm2 = np.empty(shape=[3640,9])
rmsem2 = np.empty(shape=[3640,9])
biasm2 = np.empty(shape=[3640,9])
corrsu2 = np.empty(shape=[3640,9])
rmseu2 = np.empty(shape=[3640,9])
biasu2 = np.empty(shape=[3640,9])
corrsv2 = np.empty(shape=[3640,9])
rmsev2 = np.empty(shape=[3640,9])
biasv2 = np.empty(shape=[3640,9])

i = 0
count1 = 0
count2 = 0 
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
#count8 = 0
#count9 = 0
count11 = 0
count12 = 0 
count13 = 0
count14 = 0
count15 = 0
count16 = 0
count17 = 0
#count18 = 0
#count19 = 0
while i <= 3639:
    if( K[i] == 1):
        corrsh[count1][0] = corr2(wt1ah,h500aoverall[i][:][:])
        rmseh[count1][0] = math.sqrt(mean_squared_error(wt1ah,h500aoverall[i][:][:]))
        biash[count1][0] = (wt1ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm[count1][0] = corr2(wt1am,mslpaoverall[i][:][:])
        rmsem[count1][0] = math.sqrt(mean_squared_error(wt1am,mslpaoverall[i][:][:]))
        biasm[count1][0] = (wt1am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu[count1][0] = corr2(wt1au,uaoverall[i][:][:])
        rmseu[count1][0] = math.sqrt(mean_squared_error(wt1au,uaoverall[i][:][:]))
        biasu[count1][0] = (wt1au[:][:] - uaoverall[i][:][:]).mean()
        corrsv[count1][0] = corr2(wt1av,vaoverall[i][:][:])
        rmsev[count1][0] = math.sqrt(mean_squared_error(wt1av,vaoverall[i][:][:]))
        biasv[count1][0] = (wt1av[:][:] - vaoverall[i][:][:]).mean()
        count1 = count1 + 1
        
        corrsh2[count12][1] = corr2(wt2ah,h500aoverall[i][:][:])
        rmseh2[count12][1] = math.sqrt(mean_squared_error(wt2ah,h500aoverall[i][:][:]))
        biash2[count12][1] = (wt2ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count12][1] = corr2(wt2am,mslpaoverall[i][:][:])
        rmsem2[count12][1] = math.sqrt(mean_squared_error(wt2am,mslpaoverall[i][:][:]))
        biasm2[count12][1] = (wt2am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count12][1] = corr2(wt2au,uaoverall[i][:][:])
        rmseu2[count12][1] = math.sqrt(mean_squared_error(wt2au,uaoverall[i][:][:]))
        biasu2[count12][1] = (wt2au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count12][1] = corr2(wt2av,vaoverall[i][:][:])
        rmsev2[count12][1] = math.sqrt(mean_squared_error(wt2av,vaoverall[i][:][:]))
        biasv2[count12][1] = (wt2av[:][:] - vaoverall[i][:][:]).mean()
        count12 = count12 + 1
        
        corrsh2[count13][2] = corr2(wt3ah,h500aoverall[i][:][:])
        rmseh2[count13][2] = math.sqrt(mean_squared_error(wt3ah,h500aoverall[i][:][:]))
        biash2[count13][2] = (wt3ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count13][2] = corr2(wt3am,mslpaoverall[i][:][:])
        rmsem2[count13][2] = math.sqrt(mean_squared_error(wt3am,mslpaoverall[i][:][:]))
        biasm2[count13][2] = (wt3am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count13][2] = corr2(wt3au,uaoverall[i][:][:])
        rmseu2[count13][2] = math.sqrt(mean_squared_error(wt3au,uaoverall[i][:][:]))
        biasu2[count13][2] = (wt3au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count13][2] = corr2(wt3av,vaoverall[i][:][:])
        rmsev2[count13][2] = math.sqrt(mean_squared_error(wt3av,vaoverall[i][:][:]))
        biasv2[count13][2] = (wt3av[:][:] - vaoverall[i][:][:]).mean()
        count13 = count13 + 1
        
        corrsh2[count14][3] = corr2(wt4ah,h500aoverall[i][:][:])
        rmseh2[count14][3] = math.sqrt(mean_squared_error(wt4ah,h500aoverall[i][:][:]))
        biash2[count14][3] = (wt4ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count14][3] = corr2(wt4am,mslpaoverall[i][:][:])
        rmsem2[count14][3] = math.sqrt(mean_squared_error(wt4am,mslpaoverall[i][:][:]))
        biasm2[count14][3] = (wt4am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count14][3] = corr2(wt4au,uaoverall[i][:][:])
        rmseu2[count14][3] = math.sqrt(mean_squared_error(wt4au,uaoverall[i][:][:]))
        biasu2[count14][3] = (wt4au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count14][3] = corr2(wt4av,vaoverall[i][:][:])
        rmsev2[count14][3] = math.sqrt(mean_squared_error(wt4av,vaoverall[i][:][:]))
        biasv2[count14][3] = (wt4av[:][:] - vaoverall[i][:][:]).mean()
        count14 = count14 + 1
        
        corrsh2[count15][4] = corr2(wt5ah,h500aoverall[i][:][:])
        rmseh2[count15][4] = math.sqrt(mean_squared_error(wt5ah,h500aoverall[i][:][:]))
        biash2[count15][4] = (wt5ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count15][4] = corr2(wt5am,mslpaoverall[i][:][:])
        rmsem2[count15][4] = math.sqrt(mean_squared_error(wt5am,mslpaoverall[i][:][:]))
        biasm2[count15][4] = (wt5am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count15][4] = corr2(wt5au,uaoverall[i][:][:])
        rmseu2[count15][4] = math.sqrt(mean_squared_error(wt5au,uaoverall[i][:][:]))
        biasu2[count15][4] = (wt5au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count15][4] = corr2(wt5av,vaoverall[i][:][:])
        rmsev2[count15][4] = math.sqrt(mean_squared_error(wt5av,vaoverall[i][:][:]))
        biasv2[count15][4] = (wt5av[:][:] - vaoverall[i][:][:]).mean()
        count15 = count15 + 1
        
        corrsh2[count16][5] = corr2(wt6ah,h500aoverall[i][:][:])
        rmseh2[count16][5] = math.sqrt(mean_squared_error(wt6ah,h500aoverall[i][:][:]))
        biash2[count16][5] = (wt6ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count16][5] = corr2(wt6am,mslpaoverall[i][:][:])
        rmsem2[count16][5] = math.sqrt(mean_squared_error(wt6am,mslpaoverall[i][:][:]))
        biasm2[count16][5] = (wt6am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count16][5] = corr2(wt6au,uaoverall[i][:][:])
        rmseu2[count16][5] = math.sqrt(mean_squared_error(wt6au,uaoverall[i][:][:]))
        biasu2[count16][5] = (wt6au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count16][5] = corr2(wt6av,vaoverall[i][:][:])
        rmsev2[count16][5] = math.sqrt(mean_squared_error(wt6av,vaoverall[i][:][:]))
        biasv2[count16][5] = (wt6av[:][:] - vaoverall[i][:][:]).mean()
        count16 = count16 + 1
        
        corrsh2[count17][6] = corr2(wt7ah,h500aoverall[i][:][:])
        rmseh2[count17][6] = math.sqrt(mean_squared_error(wt7ah,h500aoverall[i][:][:]))
        biash2[count17][6] = (wt7ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count17][6] = corr2(wt7am,mslpaoverall[i][:][:])
        rmsem2[count17][6] = math.sqrt(mean_squared_error(wt7am,mslpaoverall[i][:][:]))
        biasm2[count17][6] = (wt7am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count17][6] = corr2(wt7au,uaoverall[i][:][:])
        rmseu2[count17][6] = math.sqrt(mean_squared_error(wt7au,uaoverall[i][:][:]))
        biasu2[count17][6] = (wt7au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count17][6] = corr2(wt7av,vaoverall[i][:][:])
        rmsev2[count17][6] = math.sqrt(mean_squared_error(wt7av,vaoverall[i][:][:]))
        biasv2[count17][6] = (wt7av[:][:] - vaoverall[i][:][:]).mean()
        count17 = count17 + 1
        
#        corrsh2[count18][7] = corr2(wt8ah,h500aoverall[i][:][:])
#        rmseh2[count18][7] = math.sqrt(mean_squared_error(wt8ah,h500aoverall[i][:][:]))
#        biash2[count18][7] = (wt8ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count18][7] = corr2(wt8am,mslpaoverall[i][:][:])
#        rmsem2[count18][7] = math.sqrt(mean_squared_error(wt8am,mslpaoverall[i][:][:]))
#        biasm2[count18][7] = (wt8am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count18][7] = corr2(wt8au,uaoverall[i][:][:])
#        rmseu2[count18][7] = math.sqrt(mean_squared_error(wt8au,uaoverall[i][:][:]))
#        biasu2[count18][7] = (wt8au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count18][7] = corr2(wt8av,vaoverall[i][:][:])
#        rmsev2[count18][7] = math.sqrt(mean_squared_error(wt8av,vaoverall[i][:][:]))
#        biasv2[count18][7] = (wt8av[:][:] - vaoverall[i][:][:]).mean()
#        count18 = count18 + 1
#        
#                
#        corrsh2[count19][8] = corr2(wt9ah,h500aoverall[i][:][:])
#        rmseh2[count19][8] = math.sqrt(mean_squared_error(wt9ah,h500aoverall[i][:][:]))
#        biash2[count19][8] = (wt9ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count19][8] = corr2(wt9am,mslpaoverall[i][:][:])
#        rmsem2[count19][8] = math.sqrt(mean_squared_error(wt9am,mslpaoverall[i][:][:]))
#        biasm2[count19][8] = (wt9am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count19][8] = corr2(wt9au,uaoverall[i][:][:])
#        rmseu2[count19][8] = math.sqrt(mean_squared_error(wt9au,uaoverall[i][:][:]))
#        biasu2[count19][8] = (wt9au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count19][8] = corr2(wt9av,vaoverall[i][:][:])
#        rmsev2[count19][8] = math.sqrt(mean_squared_error(wt9av,vaoverall[i][:][:]))
#        biasv2[count19][8] = (wt9av[:][:] - vaoverall[i][:][:]).mean()
#        count19 = count19 + 1
    elif( K[i] == 2):
        
        corrsh[count2][1] = corr2(wt2ah, h500aoverall[i][:][:])
        rmseh[count2][1] = math.sqrt(mean_squared_error(wt2ah,h500aoverall[i][:][:]))
        biash[count2][1] = (wt2ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm[count2][1] = corr2(wt2am, mslpaoverall[i][:][:])
        rmsem[count2][1] = math.sqrt(mean_squared_error(wt2am,mslpaoverall[i][:][:]))
        biasm[count2][1] = (wt2am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu[count2][1] = corr2(wt2au, uaoverall[i][:][:])
        rmseu[count2][1] = math.sqrt(mean_squared_error(wt2au,uaoverall[i][:][:]))
        biasu[count2][1] = (wt2au[:][:] - uaoverall[i][:][:]).mean()
        corrsv[count2][1] = corr2(wt2av, vaoverall[i][:][:])
        rmsev[count2][1] = math.sqrt(mean_squared_error(wt2av,vaoverall[i][:][:]))
        biasv[count2][1] = (wt2av[:][:] - vaoverall[i][:][:]).mean()
        dateindices[count2,0] = i
        dateindices[count2,1] = corrsh[count2][1]
        dateindices[count2,2] = corrsm[count2][1]
        dateindices[count2,3] = corrsu[count2][1]
        dateindices[count2,4] = corrsv[count2][1]
        count2 = count2 + 1
        
        corrsh2[count11][0] = corr2(wt1ah,h500aoverall[i][:][:])
        rmseh2[count11][0] = math.sqrt(mean_squared_error(wt1ah,h500aoverall[i][:][:]))
        biash2[count11][0] = (wt1ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count11][0] = corr2(wt1am,mslpaoverall[i][:][:])
        rmsem2[count11][0] = math.sqrt(mean_squared_error(wt1am,mslpaoverall[i][:][:]))
        biasm2[count11][0] = (wt1am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count11][0] = corr2(wt1au,uaoverall[i][:][:])
        rmseu2[count11][0] = math.sqrt(mean_squared_error(wt1au,uaoverall[i][:][:]))
        biasu2[count11][0] = (wt1au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count11][0] = corr2(wt1av,vaoverall[i][:][:])
        rmsev2[count11][0] = math.sqrt(mean_squared_error(wt1av,vaoverall[i][:][:]))
        biasv2[count11][0] = (wt1av[:][:] - vaoverall[i][:][:]).mean()
        count11 = count11 + 1
        
        corrsh2[count13][2] = corr2(wt3ah,h500aoverall[i][:][:])
        rmseh2[count13][2] = math.sqrt(mean_squared_error(wt3ah,h500aoverall[i][:][:]))
        biash2[count13][2] = (wt3ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count13][2] = corr2(wt3am,mslpaoverall[i][:][:])
        rmsem2[count13][2] = math.sqrt(mean_squared_error(wt3am,mslpaoverall[i][:][:]))
        biasm2[count13][2] = (wt3am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count13][2] = corr2(wt3au,uaoverall[i][:][:])
        rmseu2[count13][2] = math.sqrt(mean_squared_error(wt3au,uaoverall[i][:][:]))
        biasu2[count13][2] = (wt3au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count13][2] = corr2(wt3av,vaoverall[i][:][:])
        rmsev2[count13][2] = math.sqrt(mean_squared_error(wt3av,vaoverall[i][:][:]))
        biasv2[count13][2] = (wt3av[:][:] - vaoverall[i][:][:]).mean()
        count13 = count13 + 1
        
        corrsh2[count14][3] = corr2(wt4ah,h500aoverall[i][:][:])
        rmseh2[count14][3] = math.sqrt(mean_squared_error(wt4ah,h500aoverall[i][:][:]))
        biash2[count14][3] = (wt4ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count14][3] = corr2(wt4am,mslpaoverall[i][:][:])
        rmsem2[count14][3] = math.sqrt(mean_squared_error(wt4am,mslpaoverall[i][:][:]))
        biasm2[count14][3] = (wt4am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count14][3] = corr2(wt4au,uaoverall[i][:][:])
        rmseu2[count14][3] = math.sqrt(mean_squared_error(wt4au,uaoverall[i][:][:]))
        biasu2[count14][3] = (wt4au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count14][3] = corr2(wt4av,vaoverall[i][:][:])
        rmsev2[count14][3] = math.sqrt(mean_squared_error(wt4av,vaoverall[i][:][:]))
        biasv2[count14][3] = (wt4av[:][:] - vaoverall[i][:][:]).mean()
        count14 = count14 + 1
        
        corrsh2[count15][4] = corr2(wt5ah,h500aoverall[i][:][:])
        rmseh2[count15][4] = math.sqrt(mean_squared_error(wt5ah,h500aoverall[i][:][:]))
        biash2[count15][4] = (wt5ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count15][4] = corr2(wt5am,mslpaoverall[i][:][:])
        rmsem2[count15][4] = math.sqrt(mean_squared_error(wt5am,mslpaoverall[i][:][:]))
        biasm2[count15][4] = (wt5am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count15][4] = corr2(wt5au,uaoverall[i][:][:])
        rmseu2[count15][4] = math.sqrt(mean_squared_error(wt5au,uaoverall[i][:][:]))
        biasu2[count15][4] = (wt5au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count15][4] = corr2(wt5av,vaoverall[i][:][:])
        rmsev2[count15][4] = math.sqrt(mean_squared_error(wt5av,vaoverall[i][:][:]))
        biasv2[count15][4] = (wt5av[:][:] - vaoverall[i][:][:]).mean()
        count15 = count15 + 1
        
        corrsh2[count16][5] = corr2(wt6ah,h500aoverall[i][:][:])
        rmseh2[count16][5] = math.sqrt(mean_squared_error(wt6ah,h500aoverall[i][:][:]))
        biash2[count16][5] = (wt6ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count16][5] = corr2(wt6am,mslpaoverall[i][:][:])
        rmsem2[count16][5] = math.sqrt(mean_squared_error(wt6am,mslpaoverall[i][:][:]))
        biasm2[count16][5] = (wt6am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count16][5] = corr2(wt6au,uaoverall[i][:][:])
        rmseu2[count16][5] = math.sqrt(mean_squared_error(wt6au,uaoverall[i][:][:]))
        biasu2[count16][5] = (wt6au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count16][5] = corr2(wt6av,vaoverall[i][:][:])
        rmsev2[count16][5] = math.sqrt(mean_squared_error(wt6av,vaoverall[i][:][:]))
        biasv2[count16][5] = (wt6av[:][:] - vaoverall[i][:][:]).mean()
        count16 = count16 + 1
        
        corrsh2[count17][6] = corr2(wt7ah,h500aoverall[i][:][:])
        rmseh2[count17][6] = math.sqrt(mean_squared_error(wt7ah,h500aoverall[i][:][:]))
        biash2[count17][6] = (wt7ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count17][6] = corr2(wt7am,mslpaoverall[i][:][:])
        rmsem2[count17][6] = math.sqrt(mean_squared_error(wt7am,mslpaoverall[i][:][:]))
        biasm2[count17][6] = (wt7am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count17][6] = corr2(wt7au,uaoverall[i][:][:])
        rmseu2[count17][6] = math.sqrt(mean_squared_error(wt7au,uaoverall[i][:][:]))
        biasu2[count17][6] = (wt7au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count17][6] = corr2(wt7av,vaoverall[i][:][:])
        rmsev2[count17][6] = math.sqrt(mean_squared_error(wt7av,vaoverall[i][:][:]))
        biasv2[count17][6] = (wt7av[:][:] - vaoverall[i][:][:]).mean()
        count17 = count17 + 1
        
#        corrsh2[count18][7] = corr2(wt8ah,h500aoverall[i][:][:])
#        rmseh2[count18][7] = math.sqrt(mean_squared_error(wt8ah,h500aoverall[i][:][:]))
#        biash2[count18][7] = (wt8ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count18][7] = corr2(wt8am,mslpaoverall[i][:][:])
#        rmsem2[count18][7] = math.sqrt(mean_squared_error(wt8am,mslpaoverall[i][:][:]))
#        biasm2[count18][7] = (wt8am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count18][7] = corr2(wt8au,uaoverall[i][:][:])
#        rmseu2[count18][7] = math.sqrt(mean_squared_error(wt8au,uaoverall[i][:][:]))
#        biasu2[count18][7] = (wt8au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count18][7] = corr2(wt8av,vaoverall[i][:][:])
#        rmsev2[count18][7] = math.sqrt(mean_squared_error(wt8av,vaoverall[i][:][:]))
#        biasv2[count18][7] = (wt8av[:][:] - vaoverall[i][:][:]).mean()
#        count18 = count18 + 1
#        
#                
#        corrsh2[count19][8] = corr2(wt9ah,h500aoverall[i][:][:])
#        rmseh2[count19][8] = math.sqrt(mean_squared_error(wt9ah,h500aoverall[i][:][:]))
#        biash2[count19][8] = (wt9ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count19][8] = corr2(wt9am,mslpaoverall[i][:][:])
#        rmsem2[count19][8] = math.sqrt(mean_squared_error(wt9am,mslpaoverall[i][:][:]))
#        biasm2[count19][8] = (wt9am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count19][8] = corr2(wt9au,uaoverall[i][:][:])
#        rmseu2[count19][8] = math.sqrt(mean_squared_error(wt9au,uaoverall[i][:][:]))
#        biasu2[count19][8] = (wt9au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count19][8] = corr2(wt9av,vaoverall[i][:][:])
#        rmsev2[count19][8] = math.sqrt(mean_squared_error(wt9av,vaoverall[i][:][:]))
#        biasv2[count19][8] = (wt9av[:][:] - vaoverall[i][:][:]).mean()
#        count19 = count19 + 1
    elif( K[i] == 3):
        corrsh[count3][2] = corr2(wt3ah, h500aoverall[i][:][:])
        rmseh[count3][2] = math.sqrt(mean_squared_error(wt3ah,h500aoverall[i][:][:]))
        biash[count3][2] = (wt3ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm[count3][2] = corr2(wt3am, mslpaoverall[i][:][:])
        rmsem[count3][2] = math.sqrt(mean_squared_error(wt3am,mslpaoverall[i][:][:]))
        biasm[count3][2] = (wt3am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu[count3][2] = corr2(wt3au, uaoverall[i][:][:])
        rmseu[count3][2] = math.sqrt(mean_squared_error(wt3au,uaoverall[i][:][:]))
        biasu[count3][2] = (wt3au[:][:] - uaoverall[i][:][:]).mean()
        corrsv[count3][2] = corr2(wt3av, vaoverall[i][:][:])
        rmsev[count3][2] = math.sqrt(mean_squared_error(wt3av,vaoverall[i][:][:]))
        biasv[count3][2] = (wt3av[:][:] - vaoverall[i][:][:]).mean()
        count3 = count3 + 1
        
        corrsh2[count12][1] = corr2(wt2ah,h500aoverall[i][:][:])
        rmseh2[count12][1] = math.sqrt(mean_squared_error(wt2ah,h500aoverall[i][:][:]))
        biash2[count12][1] = (wt2ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count12][1] = corr2(wt2am,mslpaoverall[i][:][:])
        rmsem2[count12][1] = math.sqrt(mean_squared_error(wt2am,mslpaoverall[i][:][:]))
        biasm2[count12][1] = (wt2am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count12][1] = corr2(wt2au,uaoverall[i][:][:])
        rmseu2[count12][1] = math.sqrt(mean_squared_error(wt2au,uaoverall[i][:][:]))
        biasu2[count12][1] = (wt2au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count12][1] = corr2(wt2av,vaoverall[i][:][:])
        rmsev2[count12][1] = math.sqrt(mean_squared_error(wt2av,vaoverall[i][:][:]))
        biasv2[count12][1] = (wt2av[:][:] - vaoverall[i][:][:]).mean()
        count12 = count12 + 1
        
        corrsh2[count11][0] = corr2(wt1ah,h500aoverall[i][:][:])
        rmseh2[count11][0] = math.sqrt(mean_squared_error(wt1ah,h500aoverall[i][:][:]))
        biash2[count11][0] = (wt1ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count11][0] = corr2(wt1am,mslpaoverall[i][:][:])
        rmsem2[count11][0] = math.sqrt(mean_squared_error(wt1am,mslpaoverall[i][:][:]))
        biasm2[count11][0] = (wt1am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count11][0] = corr2(wt1au,uaoverall[i][:][:])
        rmseu2[count11][0] = math.sqrt(mean_squared_error(wt1au,uaoverall[i][:][:]))
        biasu2[count11][0] = (wt1au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count11][0] = corr2(wt1av,vaoverall[i][:][:])
        rmsev2[count11][0] = math.sqrt(mean_squared_error(wt1av,vaoverall[i][:][:]))
        biasv2[count11][0] = (wt1av[:][:] - vaoverall[i][:][:]).mean()
        count11 = count11 + 1
        
        corrsh2[count14][3] = corr2(wt4ah,h500aoverall[i][:][:])
        rmseh2[count14][3] = math.sqrt(mean_squared_error(wt4ah,h500aoverall[i][:][:]))
        biash2[count14][3] = (wt4ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count14][3] = corr2(wt4am,mslpaoverall[i][:][:])
        rmsem2[count14][3] = math.sqrt(mean_squared_error(wt4am,mslpaoverall[i][:][:]))
        biasm2[count14][3] = (wt4am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count14][3] = corr2(wt4au,uaoverall[i][:][:])
        rmseu2[count14][3] = math.sqrt(mean_squared_error(wt4au,uaoverall[i][:][:]))
        biasu2[count14][3] = (wt4au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count14][3] = corr2(wt4av,vaoverall[i][:][:])
        rmsev2[count14][3] = math.sqrt(mean_squared_error(wt4av,vaoverall[i][:][:]))
        biasv2[count14][3] = (wt4av[:][:] - vaoverall[i][:][:]).mean()
        count14 = count14 + 1
        
        corrsh2[count15][4] = corr2(wt5ah,h500aoverall[i][:][:])
        rmseh2[count15][4] = math.sqrt(mean_squared_error(wt5ah,h500aoverall[i][:][:]))
        biash2[count15][4] = (wt5ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count15][4] = corr2(wt5am,mslpaoverall[i][:][:])
        rmsem2[count15][4] = math.sqrt(mean_squared_error(wt5am,mslpaoverall[i][:][:]))
        biasm2[count15][4] = (wt5am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count15][4] = corr2(wt5au,uaoverall[i][:][:])
        rmseu2[count15][4] = math.sqrt(mean_squared_error(wt5au,uaoverall[i][:][:]))
        biasu2[count15][4] = (wt5au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count15][4] = corr2(wt5av,vaoverall[i][:][:])
        rmsev2[count15][4] = math.sqrt(mean_squared_error(wt5av,vaoverall[i][:][:]))
        biasv2[count15][4] = (wt5av[:][:] - vaoverall[i][:][:]).mean()
        count15 = count15 + 1
        
        corrsh2[count16][5] = corr2(wt6ah,h500aoverall[i][:][:])
        rmseh2[count16][5] = math.sqrt(mean_squared_error(wt6ah,h500aoverall[i][:][:]))
        biash2[count16][5] = (wt6ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count16][5] = corr2(wt6am,mslpaoverall[i][:][:])
        rmsem2[count16][5] = math.sqrt(mean_squared_error(wt6am,mslpaoverall[i][:][:]))
        biasm2[count16][5] = (wt6am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count16][5] = corr2(wt6au,uaoverall[i][:][:])
        rmseu2[count16][5] = math.sqrt(mean_squared_error(wt6au,uaoverall[i][:][:]))
        biasu2[count16][5] = (wt6au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count16][5] = corr2(wt6av,vaoverall[i][:][:])
        rmsev2[count16][5] = math.sqrt(mean_squared_error(wt6av,vaoverall[i][:][:]))
        biasv2[count16][5] = (wt6av[:][:] - vaoverall[i][:][:]).mean()
        count16 = count16 + 1
        
        corrsh2[count17][6] = corr2(wt7ah,h500aoverall[i][:][:])
        rmseh2[count17][6] = math.sqrt(mean_squared_error(wt7ah,h500aoverall[i][:][:]))
        biash2[count17][6] = (wt7ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count17][6] = corr2(wt7am,mslpaoverall[i][:][:])
        rmsem2[count17][6] = math.sqrt(mean_squared_error(wt7am,mslpaoverall[i][:][:]))
        biasm2[count17][6] = (wt7am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count17][6] = corr2(wt7au,uaoverall[i][:][:])
        rmseu2[count17][6] = math.sqrt(mean_squared_error(wt7au,uaoverall[i][:][:]))
        biasu2[count17][6] = (wt7au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count17][6] = corr2(wt7av,vaoverall[i][:][:])
        rmsev2[count17][6] = math.sqrt(mean_squared_error(wt7av,vaoverall[i][:][:]))
        biasv2[count17][6] = (wt7av[:][:] - vaoverall[i][:][:]).mean()
        count17 = count17 + 1
        
#        corrsh2[count18][7] = corr2(wt8ah,h500aoverall[i][:][:])
#        rmseh2[count18][7] = math.sqrt(mean_squared_error(wt8ah,h500aoverall[i][:][:]))
#        biash2[count18][7] = (wt8ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count18][7] = corr2(wt8am,mslpaoverall[i][:][:])
#        rmsem2[count18][7] = math.sqrt(mean_squared_error(wt8am,mslpaoverall[i][:][:]))
#        biasm2[count18][7] = (wt8am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count18][7] = corr2(wt8au,uaoverall[i][:][:])
#        rmseu2[count18][7] = math.sqrt(mean_squared_error(wt8au,uaoverall[i][:][:]))
#        biasu2[count18][7] = (wt8au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count18][7] = corr2(wt8av,vaoverall[i][:][:])
#        rmsev2[count18][7] = math.sqrt(mean_squared_error(wt8av,vaoverall[i][:][:]))
#        biasv2[count18][7] = (wt8av[:][:] - vaoverall[i][:][:]).mean()
#        count18 = count18 + 1
#        
#                
#        corrsh2[count19][8] = corr2(wt9ah,h500aoverall[i][:][:])
#        rmseh2[count19][8] = math.sqrt(mean_squared_error(wt9ah,h500aoverall[i][:][:]))
#        biash2[count19][8] = (wt9ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count19][8] = corr2(wt9am,mslpaoverall[i][:][:])
#        rmsem2[count19][8] = math.sqrt(mean_squared_error(wt9am,mslpaoverall[i][:][:]))
#        biasm2[count19][8] = (wt9am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count19][8] = corr2(wt9au,uaoverall[i][:][:])
#        rmseu2[count19][8] = math.sqrt(mean_squared_error(wt9au,uaoverall[i][:][:]))
#        biasu2[count19][8] = (wt9au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count19][8] = corr2(wt9av,vaoverall[i][:][:])
#        rmsev2[count19][8] = math.sqrt(mean_squared_error(wt9av,vaoverall[i][:][:]))
#        biasv2[count19][8] = (wt9av[:][:] - vaoverall[i][:][:]).mean()
#        count19 = count19 + 1
    elif( K[i] == 4):
        corrsh[count4][3] = corr2(wt4ah, h500aoverall[i][:][:])
        rmseh[count4][3] = math.sqrt(mean_squared_error(wt4ah,h500aoverall[i][:][:]))
        biash[count4][3] = (wt4ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm[count4][3] = corr2(wt4am, mslpaoverall[i][:][:])
        rmsem[count4][3] = math.sqrt(mean_squared_error(wt4am,mslpaoverall[i][:][:]))
        biasm[count4][3] = (wt4am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu[count4][3] = corr2(wt4au, uaoverall[i][:][:])
        rmseu[count4][3] = math.sqrt(mean_squared_error(wt4au,uaoverall[i][:][:]))
        biasu[count4][3] = (wt4au[:][:] - uaoverall[i][:][:]).mean()
        corrsv[count4][3] = corr2(wt4av, vaoverall[i][:][:])
        rmsev[count4][3] = math.sqrt(mean_squared_error(wt4av,vaoverall[i][:][:]))
        biasv[count4][3] = (wt4av[:][:] - vaoverall[i][:][:]).mean()
        count4 = count4 + 1
        
        corrsh2[count12][1] = corr2(wt2ah,h500aoverall[i][:][:])
        rmseh2[count12][1] = math.sqrt(mean_squared_error(wt2ah,h500aoverall[i][:][:]))
        biash2[count12][1] = (wt2ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count12][1] = corr2(wt2am,mslpaoverall[i][:][:])
        rmsem2[count12][1] = math.sqrt(mean_squared_error(wt2am,mslpaoverall[i][:][:]))
        biasm2[count12][1] = (wt2am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count12][1] = corr2(wt2au,uaoverall[i][:][:])
        rmseu2[count12][1] = math.sqrt(mean_squared_error(wt2au,uaoverall[i][:][:]))
        biasu2[count12][1] = (wt2au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count12][1] = corr2(wt2av,vaoverall[i][:][:])
        rmsev2[count12][1] = math.sqrt(mean_squared_error(wt2av,vaoverall[i][:][:]))
        biasv2[count12][1] = (wt2av[:][:] - vaoverall[i][:][:]).mean()
        count12 = count12 + 1
        
        corrsh2[count13][2] = corr2(wt3ah,h500aoverall[i][:][:])
        rmseh2[count13][2] = math.sqrt(mean_squared_error(wt3ah,h500aoverall[i][:][:]))
        biash2[count13][2] = (wt3ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count13][2] = corr2(wt3am,mslpaoverall[i][:][:])
        rmsem2[count13][2] = math.sqrt(mean_squared_error(wt3am,mslpaoverall[i][:][:]))
        biasm2[count13][2] = (wt3am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count13][2] = corr2(wt3au,uaoverall[i][:][:])
        rmseu2[count13][2] = math.sqrt(mean_squared_error(wt3au,uaoverall[i][:][:]))
        biasu2[count13][2] = (wt3au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count13][2] = corr2(wt3av,vaoverall[i][:][:])
        rmsev2[count13][2] = math.sqrt(mean_squared_error(wt3av,vaoverall[i][:][:]))
        biasv2[count13][2] = (wt3av[:][:] - vaoverall[i][:][:]).mean()
        count13 = count13 + 1
        
        corrsh2[count11][0] = corr2(wt1ah,h500aoverall[i][:][:])
        rmseh2[count11][0] = math.sqrt(mean_squared_error(wt1ah,h500aoverall[i][:][:]))
        biash2[count11][0] = (wt1ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count11][0] = corr2(wt1am,mslpaoverall[i][:][:])
        rmsem2[count11][0] = math.sqrt(mean_squared_error(wt1am,mslpaoverall[i][:][:]))
        biasm2[count11][0] = (wt1am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count11][0] = corr2(wt1au,uaoverall[i][:][:])
        rmseu2[count11][0] = math.sqrt(mean_squared_error(wt1au,uaoverall[i][:][:]))
        biasu2[count11][0] = (wt1au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count11][0] = corr2(wt1av,vaoverall[i][:][:])
        rmsev2[count11][0] = math.sqrt(mean_squared_error(wt1av,vaoverall[i][:][:]))
        biasv2[count11][0] = (wt1av[:][:] - vaoverall[i][:][:]).mean()
        count11 = count11 + 1
        
        corrsh2[count15][4] = corr2(wt5ah,h500aoverall[i][:][:])
        rmseh2[count15][4] = math.sqrt(mean_squared_error(wt5ah,h500aoverall[i][:][:]))
        biash2[count15][4] = (wt5ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count15][4] = corr2(wt5am,mslpaoverall[i][:][:])
        rmsem2[count15][4] = math.sqrt(mean_squared_error(wt5am,mslpaoverall[i][:][:]))
        biasm2[count15][4] = (wt5am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count15][4] = corr2(wt5au,uaoverall[i][:][:])
        rmseu2[count15][4] = math.sqrt(mean_squared_error(wt5au,uaoverall[i][:][:]))
        biasu2[count15][4] = (wt5au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count15][4] = corr2(wt5av,vaoverall[i][:][:])
        rmsev2[count15][4] = math.sqrt(mean_squared_error(wt5av,vaoverall[i][:][:]))
        biasv2[count15][4] = (wt5av[:][:] - vaoverall[i][:][:]).mean()
        count15 = count15 + 1
        
        corrsh2[count16][5] = corr2(wt6ah,h500aoverall[i][:][:])
        rmseh2[count16][5] = math.sqrt(mean_squared_error(wt6ah,h500aoverall[i][:][:]))
        biash2[count16][5] = (wt6ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count16][5] = corr2(wt6am,mslpaoverall[i][:][:])
        rmsem2[count16][5] = math.sqrt(mean_squared_error(wt6am,mslpaoverall[i][:][:]))
        biasm2[count16][5] = (wt6am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count16][5] = corr2(wt6au,uaoverall[i][:][:])
        rmseu2[count16][5] = math.sqrt(mean_squared_error(wt6au,uaoverall[i][:][:]))
        biasu2[count16][5] = (wt6au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count16][5] = corr2(wt6av,vaoverall[i][:][:])
        rmsev2[count16][5] = math.sqrt(mean_squared_error(wt6av,vaoverall[i][:][:]))
        biasv2[count16][5] = (wt6av[:][:] - vaoverall[i][:][:]).mean()
        count16 = count16 + 1
        
        corrsh2[count17][6] = corr2(wt7ah,h500aoverall[i][:][:])
        rmseh2[count17][6] = math.sqrt(mean_squared_error(wt7ah,h500aoverall[i][:][:]))
        biash2[count17][6] = (wt7ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count17][6] = corr2(wt7am,mslpaoverall[i][:][:])
        rmsem2[count17][6] = math.sqrt(mean_squared_error(wt7am,mslpaoverall[i][:][:]))
        biasm2[count17][6] = (wt7am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count17][6] = corr2(wt7au,uaoverall[i][:][:])
        rmseu2[count17][6] = math.sqrt(mean_squared_error(wt7au,uaoverall[i][:][:]))
        biasu2[count17][6] = (wt7au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count17][6] = corr2(wt7av,vaoverall[i][:][:])
        rmsev2[count17][6] = math.sqrt(mean_squared_error(wt7av,vaoverall[i][:][:]))
        biasv2[count17][6] = (wt7av[:][:] - vaoverall[i][:][:]).mean()
        count17 = count17 + 1
        
#        corrsh2[count18][7] = corr2(wt8ah,h500aoverall[i][:][:])
#        rmseh2[count18][7] = math.sqrt(mean_squared_error(wt8ah,h500aoverall[i][:][:]))
#        biash2[count18][7] = (wt8ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count18][7] = corr2(wt8am,mslpaoverall[i][:][:])
#        rmsem2[count18][7] = math.sqrt(mean_squared_error(wt8am,mslpaoverall[i][:][:]))
#        biasm2[count18][7] = (wt8am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count18][7] = corr2(wt8au,uaoverall[i][:][:])
#        rmseu2[count18][7] = math.sqrt(mean_squared_error(wt8au,uaoverall[i][:][:]))
#        biasu2[count18][7] = (wt8au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count18][7] = corr2(wt8av,vaoverall[i][:][:])
#        rmsev2[count18][7] = math.sqrt(mean_squared_error(wt8av,vaoverall[i][:][:]))
#        biasv2[count18][7] = (wt8av[:][:] - vaoverall[i][:][:]).mean()
#        count18 = count18 + 1
#        
#                
#        corrsh2[count19][8] = corr2(wt9ah,h500aoverall[i][:][:])
#        rmseh2[count19][8] = math.sqrt(mean_squared_error(wt9ah,h500aoverall[i][:][:]))
#        biash2[count19][8] = (wt9ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count19][8] = corr2(wt9am,mslpaoverall[i][:][:])
#        rmsem2[count19][8] = math.sqrt(mean_squared_error(wt9am,mslpaoverall[i][:][:]))
#        biasm2[count19][8] = (wt9am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count19][8] = corr2(wt9au,uaoverall[i][:][:])
#        rmseu2[count19][8] = math.sqrt(mean_squared_error(wt9au,uaoverall[i][:][:]))
#        biasu2[count19][8] = (wt9au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count19][8] = corr2(wt9av,vaoverall[i][:][:])
#        rmsev2[count19][8] = math.sqrt(mean_squared_error(wt9av,vaoverall[i][:][:]))
#        biasv2[count19][8] = (wt9av[:][:] - vaoverall[i][:][:]).mean()
#        count19 = count19 + 1
    elif( K[i] == 5):
        corrsh[count5][4] = corr2(wt5ah, h500aoverall[i][:][:])
        rmseh[count5][4] = math.sqrt(mean_squared_error(wt5ah,h500aoverall[i][:][:]))
        biash[count5][4] = (wt5ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm[count5][4] = corr2(wt5am, mslpaoverall[i][:][:])
        rmsem[count5][4] = math.sqrt(mean_squared_error(wt5am,mslpaoverall[i][:][:]))
        biasm[count5][4] = (wt5am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu[count5][4] = corr2(wt5au, uaoverall[i][:][:])
        rmseu[count5][4] = math.sqrt(mean_squared_error(wt5au,uaoverall[i][:][:]))
        biasu[count5][4] = (wt5au[:][:] - uaoverall[i][:][:]).mean()
        corrsv[count5][4] = corr2(wt5av, vaoverall[i][:][:])
        rmsev[count5][4] = math.sqrt(mean_squared_error(wt5av,vaoverall[i][:][:]))
        biasv[count5][4] = (wt5av[:][:] - vaoverall[i][:][:]).mean()
        count5 = count5 + 1
        
        corrsh2[count12][1] = corr2(wt2ah,h500aoverall[i][:][:])
        rmseh2[count12][1] = math.sqrt(mean_squared_error(wt2ah,h500aoverall[i][:][:]))
        biash2[count12][1] = (wt2ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count12][1] = corr2(wt2am,mslpaoverall[i][:][:])
        rmsem2[count12][1] = math.sqrt(mean_squared_error(wt2am,mslpaoverall[i][:][:]))
        biasm2[count12][1] = (wt2am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count12][1] = corr2(wt2au,uaoverall[i][:][:])
        rmseu2[count12][1] = math.sqrt(mean_squared_error(wt2au,uaoverall[i][:][:]))
        biasu2[count12][1] = (wt2au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count12][1] = corr2(wt2av,vaoverall[i][:][:])
        rmsev2[count12][1] = math.sqrt(mean_squared_error(wt2av,vaoverall[i][:][:]))
        biasv2[count12][1] = (wt2av[:][:] - vaoverall[i][:][:]).mean()
        count12 = count12 + 1
        
        corrsh2[count13][2] = corr2(wt3ah,h500aoverall[i][:][:])
        rmseh2[count13][2] = math.sqrt(mean_squared_error(wt3ah,h500aoverall[i][:][:]))
        biash2[count13][2] = (wt3ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count13][2] = corr2(wt3am,mslpaoverall[i][:][:])
        rmsem2[count13][2] = math.sqrt(mean_squared_error(wt3am,mslpaoverall[i][:][:]))
        biasm2[count13][2] = (wt3am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count13][2] = corr2(wt3au,uaoverall[i][:][:])
        rmseu2[count13][2] = math.sqrt(mean_squared_error(wt3au,uaoverall[i][:][:]))
        biasu2[count13][2] = (wt3au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count13][2] = corr2(wt3av,vaoverall[i][:][:])
        rmsev2[count13][2] = math.sqrt(mean_squared_error(wt3av,vaoverall[i][:][:]))
        biasv2[count13][2] = (wt3av[:][:] - vaoverall[i][:][:]).mean()
        count13 = count13 + 1
        
        corrsh2[count14][3] = corr2(wt4ah,h500aoverall[i][:][:])
        rmseh2[count14][3] = math.sqrt(mean_squared_error(wt4ah,h500aoverall[i][:][:]))
        biash2[count14][3] = (wt4ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count14][3] = corr2(wt4am,mslpaoverall[i][:][:])
        rmsem2[count14][3] = math.sqrt(mean_squared_error(wt4am,mslpaoverall[i][:][:]))
        biasm2[count14][3] = (wt4am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count14][3] = corr2(wt4au,uaoverall[i][:][:])
        rmseu2[count14][3] = math.sqrt(mean_squared_error(wt4au,uaoverall[i][:][:]))
        biasu2[count14][3] = (wt4au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count14][3] = corr2(wt4av,vaoverall[i][:][:])
        rmsev2[count14][3] = math.sqrt(mean_squared_error(wt4av,vaoverall[i][:][:]))
        biasv2[count14][3] = (wt4av[:][:] - vaoverall[i][:][:]).mean()
        count14 = count14 + 1
        
        corrsh2[count11][0] = corr2(wt1ah,h500aoverall[i][:][:])
        rmseh2[count11][0] = math.sqrt(mean_squared_error(wt1ah,h500aoverall[i][:][:]))
        biash2[count11][0] = (wt1ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count11][0] = corr2(wt1am,mslpaoverall[i][:][:])
        rmsem2[count11][0] = math.sqrt(mean_squared_error(wt1am,mslpaoverall[i][:][:]))
        biasm2[count11][0] = (wt1am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count11][0] = corr2(wt1au,uaoverall[i][:][:])
        rmseu2[count11][0] = math.sqrt(mean_squared_error(wt1au,uaoverall[i][:][:]))
        biasu2[count11][0] = (wt1au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count11][0] = corr2(wt1av,vaoverall[i][:][:])
        rmsev2[count11][0] = math.sqrt(mean_squared_error(wt1av,vaoverall[i][:][:]))
        biasv2[count11][0] = (wt1av[:][:] - vaoverall[i][:][:]).mean()
        count11 = count11 + 1
        
        corrsh2[count16][5] = corr2(wt6ah,h500aoverall[i][:][:])
        rmseh2[count16][5] = math.sqrt(mean_squared_error(wt6ah,h500aoverall[i][:][:]))
        biash2[count16][5] = (wt6ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count16][5] = corr2(wt6am,mslpaoverall[i][:][:])
        rmsem2[count16][5] = math.sqrt(mean_squared_error(wt6am,mslpaoverall[i][:][:]))
        biasm2[count16][5] = (wt6am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count16][5] = corr2(wt6au,uaoverall[i][:][:])
        rmseu2[count16][5] = math.sqrt(mean_squared_error(wt6au,uaoverall[i][:][:]))
        biasu2[count16][5] = (wt6au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count16][5] = corr2(wt6av,vaoverall[i][:][:])
        rmsev2[count16][5] = math.sqrt(mean_squared_error(wt6av,vaoverall[i][:][:]))
        biasv2[count16][5] = (wt6av[:][:] - vaoverall[i][:][:]).mean()
        count16 = count16 + 1
        
        corrsh2[count17][6] = corr2(wt7ah,h500aoverall[i][:][:])
        rmseh2[count17][6] = math.sqrt(mean_squared_error(wt7ah,h500aoverall[i][:][:]))
        biash2[count17][6] = (wt7ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count17][6] = corr2(wt7am,mslpaoverall[i][:][:])
        rmsem2[count17][6] = math.sqrt(mean_squared_error(wt7am,mslpaoverall[i][:][:]))
        biasm2[count17][6] = (wt7am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count17][6] = corr2(wt7au,uaoverall[i][:][:])
        rmseu2[count17][6] = math.sqrt(mean_squared_error(wt7au,uaoverall[i][:][:]))
        biasu2[count17][6] = (wt7au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count17][6] = corr2(wt7av,vaoverall[i][:][:])
        rmsev2[count17][6] = math.sqrt(mean_squared_error(wt7av,vaoverall[i][:][:]))
        biasv2[count17][6] = (wt7av[:][:] - vaoverall[i][:][:]).mean()
        count17 = count17 + 1
        
#        corrsh2[count18][7] = corr2(wt8ah,h500aoverall[i][:][:])
#        rmseh2[count18][7] = math.sqrt(mean_squared_error(wt8ah,h500aoverall[i][:][:]))
#        biash2[count18][7] = (wt8ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count18][7] = corr2(wt8am,mslpaoverall[i][:][:])
#        rmsem2[count18][7] = math.sqrt(mean_squared_error(wt8am,mslpaoverall[i][:][:]))
#        biasm2[count18][7] = (wt8am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count18][7] = corr2(wt8au,uaoverall[i][:][:])
#        rmseu2[count18][7] = math.sqrt(mean_squared_error(wt8au,uaoverall[i][:][:]))
#        biasu2[count18][7] = (wt8au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count18][7] = corr2(wt8av,vaoverall[i][:][:])
#        rmsev2[count18][7] = math.sqrt(mean_squared_error(wt8av,vaoverall[i][:][:]))
#        biasv2[count18][7] = (wt8av[:][:] - vaoverall[i][:][:]).mean()
#        count18 = count18 + 1
#        
#                
#        corrsh2[count19][8] = corr2(wt9ah,h500aoverall[i][:][:])
#        rmseh2[count19][8] = math.sqrt(mean_squared_error(wt9ah,h500aoverall[i][:][:]))
#        biash2[count19][8] = (wt9ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count19][8] = corr2(wt9am,mslpaoverall[i][:][:])
#        rmsem2[count19][8] = math.sqrt(mean_squared_error(wt9am,mslpaoverall[i][:][:]))
#        biasm2[count19][8] = (wt9am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count19][8] = corr2(wt9au,uaoverall[i][:][:])
#        rmseu2[count19][8] = math.sqrt(mean_squared_error(wt9au,uaoverall[i][:][:]))
#        biasu2[count19][8] = (wt9au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count19][8] = corr2(wt9av,vaoverall[i][:][:])
#        rmsev2[count19][8] = math.sqrt(mean_squared_error(wt9av,vaoverall[i][:][:]))
#        biasv2[count19][8] = (wt9av[:][:] - vaoverall[i][:][:]).mean()
#        count19 = count19 + 1
    elif( K[i] == 6):
        corrsh[count6][5] = corr2(wt6ah, h500aoverall[i][:][:])
        rmseh[count6][5] = math.sqrt(mean_squared_error(wt6ah,h500aoverall[i][:][:]))
        biash[count6][5] = (wt6ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm[count6][5] = corr2(wt6am, mslpaoverall[i][:][:])
        rmsem[count6][5] = math.sqrt(mean_squared_error(wt6am,mslpaoverall[i][:][:]))
        biasm[count6][5] = (wt6am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu[count6][5] = corr2(wt6au, uaoverall[i][:][:])
        rmseu[count6][5] = math.sqrt(mean_squared_error(wt6au,uaoverall[i][:][:]))
        biasu[count6][5] = (wt6au[:][:] - uaoverall[i][:][:]).mean()
        corrsv[count6][5] = corr2(wt6av, vaoverall[i][:][:])
        rmsev[count6][5] = math.sqrt(mean_squared_error(wt6av,vaoverall[i][:][:]))
        biasv[count6][5] = (wt6av[:][:] - vaoverall[i][:][:]).mean()
        count6 = count6 + 1
        
        corrsh2[count12][1] = corr2(wt2ah,h500aoverall[i][:][:])
        rmseh2[count12][1] = math.sqrt(mean_squared_error(wt2ah,h500aoverall[i][:][:]))
        biash2[count12][1] = (wt2ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count12][1] = corr2(wt2am,mslpaoverall[i][:][:])
        rmsem2[count12][1] = math.sqrt(mean_squared_error(wt2am,mslpaoverall[i][:][:]))
        biasm2[count12][1] = (wt2am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count12][1] = corr2(wt2au,uaoverall[i][:][:])
        rmseu2[count12][1] = math.sqrt(mean_squared_error(wt2au,uaoverall[i][:][:]))
        biasu2[count12][1] = (wt2au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count12][1] = corr2(wt2av,vaoverall[i][:][:])
        rmsev2[count12][1] = math.sqrt(mean_squared_error(wt2av,vaoverall[i][:][:]))
        biasv2[count12][1] = (wt2av[:][:] - vaoverall[i][:][:]).mean()
        count12 = count12 + 1
        
        corrsh2[count13][2] = corr2(wt3ah,h500aoverall[i][:][:])
        rmseh2[count13][2] = math.sqrt(mean_squared_error(wt3ah,h500aoverall[i][:][:]))
        biash2[count13][2] = (wt3ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count13][2] = corr2(wt3am,mslpaoverall[i][:][:])
        rmsem2[count13][2] = math.sqrt(mean_squared_error(wt3am,mslpaoverall[i][:][:]))
        biasm2[count13][2] = (wt3am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count13][2] = corr2(wt3au,uaoverall[i][:][:])
        rmseu2[count13][2] = math.sqrt(mean_squared_error(wt3au,uaoverall[i][:][:]))
        biasu2[count13][2] = (wt3au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count13][2] = corr2(wt3av,vaoverall[i][:][:])
        rmsev2[count13][2] = math.sqrt(mean_squared_error(wt3av,vaoverall[i][:][:]))
        biasv2[count13][2] = (wt3av[:][:] - vaoverall[i][:][:]).mean()
        count13 = count13 + 1
        
        corrsh2[count14][3] = corr2(wt4ah,h500aoverall[i][:][:])
        rmseh2[count14][3] = math.sqrt(mean_squared_error(wt4ah,h500aoverall[i][:][:]))
        biash2[count14][3] = (wt4ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count14][3] = corr2(wt4am,mslpaoverall[i][:][:])
        rmsem2[count14][3] = math.sqrt(mean_squared_error(wt4am,mslpaoverall[i][:][:]))
        biasm2[count14][3] = (wt4am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count14][3] = corr2(wt4au,uaoverall[i][:][:])
        rmseu2[count14][3] = math.sqrt(mean_squared_error(wt4au,uaoverall[i][:][:]))
        biasu2[count14][3] = (wt4au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count14][3] = corr2(wt4av,vaoverall[i][:][:])
        rmsev2[count14][3] = math.sqrt(mean_squared_error(wt4av,vaoverall[i][:][:]))
        biasv2[count14][3] = (wt4av[:][:] - vaoverall[i][:][:]).mean()
        count14 = count14 + 1
        
        corrsh2[count15][4] = corr2(wt5ah,h500aoverall[i][:][:])
        rmseh2[count15][4] = math.sqrt(mean_squared_error(wt5ah,h500aoverall[i][:][:]))
        biash2[count15][4] = (wt5ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count15][4] = corr2(wt5am,mslpaoverall[i][:][:])
        rmsem2[count15][4] = math.sqrt(mean_squared_error(wt5am,mslpaoverall[i][:][:]))
        biasm2[count15][4] = (wt5am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count15][4] = corr2(wt5au,uaoverall[i][:][:])
        rmseu2[count15][4] = math.sqrt(mean_squared_error(wt5au,uaoverall[i][:][:]))
        biasu2[count15][4] = (wt5au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count15][4] = corr2(wt5av,vaoverall[i][:][:])
        rmsev2[count15][4] = math.sqrt(mean_squared_error(wt5av,vaoverall[i][:][:]))
        biasv2[count15][4] = (wt5av[:][:] - vaoverall[i][:][:]).mean()
        count15 = count15 + 1
        
        corrsh2[count11][0] = corr2(wt1ah,h500aoverall[i][:][:])
        rmseh2[count11][0] = math.sqrt(mean_squared_error(wt1ah,h500aoverall[i][:][:]))
        biash2[count11][0] = (wt1ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count11][0] = corr2(wt1am,mslpaoverall[i][:][:])
        rmsem2[count11][0] = math.sqrt(mean_squared_error(wt1am,mslpaoverall[i][:][:]))
        biasm2[count11][0] = (wt1am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count11][0] = corr2(wt1au,uaoverall[i][:][:])
        rmseu2[count11][0] = math.sqrt(mean_squared_error(wt1au,uaoverall[i][:][:]))
        biasu2[count11][0] = (wt1au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count11][0] = corr2(wt1av,vaoverall[i][:][:])
        rmsev2[count11][0] = math.sqrt(mean_squared_error(wt1av,vaoverall[i][:][:]))
        biasv2[count11][0] = (wt1av[:][:] - vaoverall[i][:][:]).mean()
        count11 = count11 + 1
        
        corrsh2[count17][6] = corr2(wt7ah,h500aoverall[i][:][:])
        rmseh2[count17][6] = math.sqrt(mean_squared_error(wt7ah,h500aoverall[i][:][:]))
        biash2[count17][6] = (wt7ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count17][6] = corr2(wt7am,mslpaoverall[i][:][:])
        rmsem2[count17][6] = math.sqrt(mean_squared_error(wt7am,mslpaoverall[i][:][:]))
        biasm2[count17][6] = (wt7am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count17][6] = corr2(wt7au,uaoverall[i][:][:])
        rmseu2[count17][6] = math.sqrt(mean_squared_error(wt7au,uaoverall[i][:][:]))
        biasu2[count17][6] = (wt7au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count17][6] = corr2(wt7av,vaoverall[i][:][:])
        rmsev2[count17][6] = math.sqrt(mean_squared_error(wt7av,vaoverall[i][:][:]))
        biasv2[count17][6] = (wt7av[:][:] - vaoverall[i][:][:]).mean()
        count17 = count17 + 1
                
#        corrsh2[count18][7] = corr2(wt8ah,h500aoverall[i][:][:])
#        rmseh2[count18][7] = math.sqrt(mean_squared_error(wt8ah,h500aoverall[i][:][:]))
#        biash2[count18][7] = (wt8ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count18][7] = corr2(wt8am,mslpaoverall[i][:][:])
#        rmsem2[count18][7] = math.sqrt(mean_squared_error(wt8am,mslpaoverall[i][:][:]))
#        biasm2[count18][7] = (wt8am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count18][7] = corr2(wt8au,uaoverall[i][:][:])
#        rmseu2[count18][7] = math.sqrt(mean_squared_error(wt8au,uaoverall[i][:][:]))
#        biasu2[count18][7] = (wt8au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count18][7] = corr2(wt8av,vaoverall[i][:][:])
#        rmsev2[count18][7] = math.sqrt(mean_squared_error(wt8av,vaoverall[i][:][:]))
#        biasv2[count18][7] = (wt8av[:][:] - vaoverall[i][:][:]).mean()
#        count18 = count18 + 1
#        
#                
#        corrsh2[count19][8] = corr2(wt9ah,h500aoverall[i][:][:])
#        rmseh2[count19][8] = math.sqrt(mean_squared_error(wt9ah,h500aoverall[i][:][:]))
#        biash2[count19][8] = (wt9ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count19][8] = corr2(wt9am,mslpaoverall[i][:][:])
#        rmsem2[count19][8] = math.sqrt(mean_squared_error(wt9am,mslpaoverall[i][:][:]))
#        biasm2[count19][8] = (wt9am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count19][8] = corr2(wt9au,uaoverall[i][:][:])
#        rmseu2[count19][8] = math.sqrt(mean_squared_error(wt9au,uaoverall[i][:][:]))
#        biasu2[count19][8] = (wt9au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count19][8] = corr2(wt9av,vaoverall[i][:][:])
#        rmsev2[count19][8] = math.sqrt(mean_squared_error(wt9av,vaoverall[i][:][:]))
#        biasv2[count19][8] = (wt9av[:][:] - vaoverall[i][:][:]).mean()
#        count19 = count19 + 1
    else:
        corrsh[count7][6] = corr2(wt7ah, h500aoverall[i][:][:])
        rmseh[count7][6] = math.sqrt(mean_squared_error(wt7ah,h500aoverall[i][:][:]))
        biash[count7][6] = (wt7ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm[count7][6] = corr2(wt7am, mslpaoverall[i][:][:])
        rmsem[count7][6] = math.sqrt(mean_squared_error(wt7am,mslpaoverall[i][:][:]))
        biasm[count7][6] = (wt7am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu[count7][6] = corr2(wt7au, uaoverall[i][:][:])
        rmseu[count7][6] = math.sqrt(mean_squared_error(wt7au,uaoverall[i][:][:]))
        biasu[count7][6] = (wt7au[:][:] - uaoverall[i][:][:]).mean()
        corrsv[count7][6] = corr2(wt7av, vaoverall[i][:][:])
        rmsev[count7][6] = math.sqrt(mean_squared_error(wt7av,vaoverall[i][:][:]))
        biasv[count7][6] = (wt7av[:][:] - vaoverall[i][:][:]).mean()
        count7 = count7 + 1
        
        corrsh2[count12][1] = corr2(wt2ah,h500aoverall[i][:][:])
        rmseh2[count12][1] = math.sqrt(mean_squared_error(wt2ah,h500aoverall[i][:][:]))
        biash2[count12][1] = (wt2ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count12][1] = corr2(wt2am,mslpaoverall[i][:][:])
        rmsem2[count12][1] = math.sqrt(mean_squared_error(wt2am,mslpaoverall[i][:][:]))
        biasm2[count12][1] = (wt2am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count12][1] = corr2(wt2au,uaoverall[i][:][:])
        rmseu2[count12][1] = math.sqrt(mean_squared_error(wt2au,uaoverall[i][:][:]))
        biasu2[count12][1] = (wt2au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count12][1] = corr2(wt2av,vaoverall[i][:][:])
        rmsev2[count12][1] = math.sqrt(mean_squared_error(wt2av,vaoverall[i][:][:]))
        biasv2[count12][1] = (wt2av[:][:] - vaoverall[i][:][:]).mean()
        count12 = count12 + 1
        
        corrsh2[count13][2] = corr2(wt3ah,h500aoverall[i][:][:])
        rmseh2[count13][2] = math.sqrt(mean_squared_error(wt3ah,h500aoverall[i][:][:]))
        biash2[count13][2] = (wt3ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count13][2] = corr2(wt3am,mslpaoverall[i][:][:])
        rmsem2[count13][2] = math.sqrt(mean_squared_error(wt3am,mslpaoverall[i][:][:]))
        biasm2[count13][2] = (wt3am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count13][2] = corr2(wt3au,uaoverall[i][:][:])
        rmseu2[count13][2] = math.sqrt(mean_squared_error(wt3au,uaoverall[i][:][:]))
        biasu2[count13][2] = (wt3au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count13][2] = corr2(wt3av,vaoverall[i][:][:])
        rmsev2[count13][2] = math.sqrt(mean_squared_error(wt3av,vaoverall[i][:][:]))
        biasv2[count13][2] = (wt3av[:][:] - vaoverall[i][:][:]).mean()
        count13 = count13 + 1
        
        corrsh2[count14][3] = corr2(wt4ah,h500aoverall[i][:][:])
        rmseh2[count14][3] = math.sqrt(mean_squared_error(wt4ah,h500aoverall[i][:][:]))
        biash2[count14][3] = (wt4ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count14][3] = corr2(wt4am,mslpaoverall[i][:][:])
        rmsem2[count14][3] = math.sqrt(mean_squared_error(wt4am,mslpaoverall[i][:][:]))
        biasm2[count14][3] = (wt4am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count14][3] = corr2(wt4au,uaoverall[i][:][:])
        rmseu2[count14][3] = math.sqrt(mean_squared_error(wt4au,uaoverall[i][:][:]))
        biasu2[count14][3] = (wt4au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count14][3] = corr2(wt4av,vaoverall[i][:][:])
        rmsev2[count14][3] = math.sqrt(mean_squared_error(wt4av,vaoverall[i][:][:]))
        biasv2[count14][3] = (wt4av[:][:] - vaoverall[i][:][:]).mean()
        count14 = count14 + 1
        
        corrsh2[count15][4] = corr2(wt5ah,h500aoverall[i][:][:])
        rmseh2[count15][4] = math.sqrt(mean_squared_error(wt5ah,h500aoverall[i][:][:]))
        biash2[count15][4] = (wt5ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count15][4] = corr2(wt5am,mslpaoverall[i][:][:])
        rmsem2[count15][4] = math.sqrt(mean_squared_error(wt5am,mslpaoverall[i][:][:]))
        biasm2[count15][4] = (wt5am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count15][4] = corr2(wt5au,uaoverall[i][:][:])
        rmseu2[count15][4] = math.sqrt(mean_squared_error(wt5au,uaoverall[i][:][:]))
        biasu2[count15][4] = (wt5au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count15][4] = corr2(wt5av,vaoverall[i][:][:])
        rmsev2[count15][4] = math.sqrt(mean_squared_error(wt5av,vaoverall[i][:][:]))
        biasv2[count15][4] = (wt5av[:][:] - vaoverall[i][:][:]).mean()
        count15 = count15 + 1
        
        corrsh2[count16][5] = corr2(wt6ah,h500aoverall[i][:][:])
        rmseh2[count16][5] = math.sqrt(mean_squared_error(wt6ah,h500aoverall[i][:][:]))
        biash2[count16][5] = (wt6ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count16][5] = corr2(wt6am,mslpaoverall[i][:][:])
        rmsem2[count16][5] = math.sqrt(mean_squared_error(wt6am,mslpaoverall[i][:][:]))
        biasm2[count16][5] = (wt6am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count16][5] = corr2(wt6au,uaoverall[i][:][:])
        rmseu2[count16][5] = math.sqrt(mean_squared_error(wt6au,uaoverall[i][:][:]))
        biasu2[count16][5] = (wt6au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count16][5] = corr2(wt6av,vaoverall[i][:][:])
        rmsev2[count16][5] = math.sqrt(mean_squared_error(wt6av,vaoverall[i][:][:]))
        biasv2[count16][5] = (wt6av[:][:] - vaoverall[i][:][:]).mean()
        count16 = count16 + 1
        
        corrsh2[count11][0] = corr2(wt1ah,h500aoverall[i][:][:])
        rmseh2[count11][0] = math.sqrt(mean_squared_error(wt1ah,h500aoverall[i][:][:]))
        biash2[count11][0] = (wt1ah[:][:] - h500aoverall[i][:][:]).mean()
        corrsm2[count11][0] = corr2(wt1am,mslpaoverall[i][:][:])
        rmsem2[count11][0] = math.sqrt(mean_squared_error(wt1am,mslpaoverall[i][:][:]))
        biasm2[count11][0] = (wt1am[:][:] - mslpaoverall[i][:][:]).mean()
        corrsu2[count11][0] = corr2(wt1au,uaoverall[i][:][:])
        rmseu2[count11][0] = math.sqrt(mean_squared_error(wt1au,uaoverall[i][:][:]))
        biasu2[count11][0] = (wt1au[:][:] - uaoverall[i][:][:]).mean()
        corrsv2[count11][0] = corr2(wt1av,vaoverall[i][:][:])
        rmsev2[count11][0] = math.sqrt(mean_squared_error(wt1av,vaoverall[i][:][:]))
        biasv2[count11][0] = (wt1av[:][:] - vaoverall[i][:][:]).mean()
        count11 = count11 + 1
#    elif( K[i] == 7):
#        corrsh[count7][6] = corr2(wt7ah, h500aoverall[i][:][:])
#        rmseh[count7][6] = math.sqrt(mean_squared_error(wt7ah,h500aoverall[i][:][:]))
#        biash[count7][6] = (wt7ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm[count7][6] = corr2(wt7am, mslpaoverall[i][:][:])
#        rmsem[count7][6] = math.sqrt(mean_squared_error(wt7am,mslpaoverall[i][:][:]))
#        biasm[count7][6] = (wt7am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu[count7][6] = corr2(wt7au, uaoverall[i][:][:])
#        rmseu[count7][6] = math.sqrt(mean_squared_error(wt7au,uaoverall[i][:][:]))
#        biasu[count7][6] = (wt7au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv[count7][6] = corr2(wt7av, vaoverall[i][:][:])
#        rmsev[count7][6] = math.sqrt(mean_squared_error(wt7av,vaoverall[i][:][:]))
#        biasv[count7][6] = (wt7av[:][:] - vaoverall[i][:][:]).mean()
#        count7 = count7 + 1
#        
#        corrsh2[count12][1] = corr2(wt2ah,h500aoverall[i][:][:])
#        rmseh2[count12][1] = math.sqrt(mean_squared_error(wt2ah,h500aoverall[i][:][:]))
#        biash2[count12][1] = (wt2ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count12][1] = corr2(wt2am,mslpaoverall[i][:][:])
#        rmsem2[count12][1] = math.sqrt(mean_squared_error(wt2am,mslpaoverall[i][:][:]))
#        biasm2[count12][1] = (wt2am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count12][1] = corr2(wt2au,uaoverall[i][:][:])
#        rmseu2[count12][1] = math.sqrt(mean_squared_error(wt2au,uaoverall[i][:][:]))
#        biasu2[count12][1] = (wt2au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count12][1] = corr2(wt2av,vaoverall[i][:][:])
#        rmsev2[count12][1] = math.sqrt(mean_squared_error(wt2av,vaoverall[i][:][:]))
#        biasv2[count12][1] = (wt2av[:][:] - vaoverall[i][:][:]).mean()
#        count12 = count12 + 1
#        
#        corrsh2[count13][2] = corr2(wt3ah,h500aoverall[i][:][:])
#        rmseh2[count13][2] = math.sqrt(mean_squared_error(wt3ah,h500aoverall[i][:][:]))
#        biash2[count13][2] = (wt3ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count13][2] = corr2(wt3am,mslpaoverall[i][:][:])
#        rmsem2[count13][2] = math.sqrt(mean_squared_error(wt3am,mslpaoverall[i][:][:]))
#        biasm2[count13][2] = (wt3am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count13][2] = corr2(wt3au,uaoverall[i][:][:])
#        rmseu2[count13][2] = math.sqrt(mean_squared_error(wt3au,uaoverall[i][:][:]))
#        biasu2[count13][2] = (wt3au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count13][2] = corr2(wt3av,vaoverall[i][:][:])
#        rmsev2[count13][2] = math.sqrt(mean_squared_error(wt3av,vaoverall[i][:][:]))
#        biasv2[count13][2] = (wt3av[:][:] - vaoverall[i][:][:]).mean()
#        count13 = count13 + 1
#        
#        corrsh2[count14][3] = corr2(wt4ah,h500aoverall[i][:][:])
#        rmseh2[count14][3] = math.sqrt(mean_squared_error(wt4ah,h500aoverall[i][:][:]))
#        biash2[count14][3] = (wt4ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count14][3] = corr2(wt4am,mslpaoverall[i][:][:])
#        rmsem2[count14][3] = math.sqrt(mean_squared_error(wt4am,mslpaoverall[i][:][:]))
#        biasm2[count14][3] = (wt4am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count14][3] = corr2(wt4au,uaoverall[i][:][:])
#        rmseu2[count14][3] = math.sqrt(mean_squared_error(wt4au,uaoverall[i][:][:]))
#        biasu2[count14][3] = (wt4au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count14][3] = corr2(wt4av,vaoverall[i][:][:])
#        rmsev2[count14][3] = math.sqrt(mean_squared_error(wt4av,vaoverall[i][:][:]))
#        biasv2[count14][3] = (wt4av[:][:] - vaoverall[i][:][:]).mean()
#        count14 = count14 + 1
#        
#        corrsh2[count15][4] = corr2(wt5ah,h500aoverall[i][:][:])
#        rmseh2[count15][4] = math.sqrt(mean_squared_error(wt5ah,h500aoverall[i][:][:]))
#        biash2[count15][4] = (wt5ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count15][4] = corr2(wt5am,mslpaoverall[i][:][:])
#        rmsem2[count15][4] = math.sqrt(mean_squared_error(wt5am,mslpaoverall[i][:][:]))
#        biasm2[count15][4] = (wt5am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count15][4] = corr2(wt5au,uaoverall[i][:][:])
#        rmseu2[count15][4] = math.sqrt(mean_squared_error(wt5au,uaoverall[i][:][:]))
#        biasu2[count15][4] = (wt5au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count15][4] = corr2(wt5av,vaoverall[i][:][:])
#        rmsev2[count15][4] = math.sqrt(mean_squared_error(wt5av,vaoverall[i][:][:]))
#        biasv2[count15][4] = (wt5av[:][:] - vaoverall[i][:][:]).mean()
#        count15 = count15 + 1
#        
#        corrsh2[count16][5] = corr2(wt6ah,h500aoverall[i][:][:])
#        rmseh2[count16][5] = math.sqrt(mean_squared_error(wt6ah,h500aoverall[i][:][:]))
#        biash2[count16][5] = (wt6ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count16][5] = corr2(wt6am,mslpaoverall[i][:][:])
#        rmsem2[count16][5] = math.sqrt(mean_squared_error(wt6am,mslpaoverall[i][:][:]))
#        biasm2[count16][5] = (wt6am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count16][5] = corr2(wt6au,uaoverall[i][:][:])
#        rmseu2[count16][5] = math.sqrt(mean_squared_error(wt6au,uaoverall[i][:][:]))
#        biasu2[count16][5] = (wt6au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count16][5] = corr2(wt6av,vaoverall[i][:][:])
#        rmsev2[count16][5] = math.sqrt(mean_squared_error(wt6av,vaoverall[i][:][:]))
#        biasv2[count16][5] = (wt6av[:][:] - vaoverall[i][:][:]).mean()
#        count16 = count16 + 1
#        
#        corrsh2[count11][0] = corr2(wt1ah,h500aoverall[i][:][:])
#        rmseh2[count11][0] = math.sqrt(mean_squared_error(wt1ah,h500aoverall[i][:][:]))
#        biash2[count11][0] = (wt1ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count11][0] = corr2(wt1am,mslpaoverall[i][:][:])
#        rmsem2[count11][0] = math.sqrt(mean_squared_error(wt1am,mslpaoverall[i][:][:]))
#        biasm2[count11][0] = (wt1am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count11][0] = corr2(wt1au,uaoverall[i][:][:])
#        rmseu2[count11][0] = math.sqrt(mean_squared_error(wt1au,uaoverall[i][:][:]))
#        biasu2[count11][0] = (wt1au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count11][0] = corr2(wt1av,vaoverall[i][:][:])
#        rmsev2[count11][0] = math.sqrt(mean_squared_error(wt1av,vaoverall[i][:][:]))
#        biasv2[count11][0] = (wt1av[:][:] - vaoverall[i][:][:]).mean()
#        count11 = count11 + 1
#                
#        corrsh2[count18][7] = corr2(wt8ah,h500aoverall[i][:][:])
#        rmseh2[count18][7] = math.sqrt(mean_squared_error(wt8ah,h500aoverall[i][:][:]))
#        biash2[count18][7] = (wt8ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count18][7] = corr2(wt8am,mslpaoverall[i][:][:])
#        rmsem2[count18][7] = math.sqrt(mean_squared_error(wt8am,mslpaoverall[i][:][:]))
#        biasm2[count18][7] = (wt8am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count18][7] = corr2(wt8au,uaoverall[i][:][:])
#        rmseu2[count18][7] = math.sqrt(mean_squared_error(wt8au,uaoverall[i][:][:]))
#        biasu2[count18][7] = (wt8au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count18][7] = corr2(wt8av,vaoverall[i][:][:])
#        rmsev2[count18][7] = math.sqrt(mean_squared_error(wt8av,vaoverall[i][:][:]))
#        biasv2[count18][7] = (wt8av[:][:] - vaoverall[i][:][:]).mean()
#        count18 = count18 + 1
#        
#                
#        corrsh2[count19][8] = corr2(wt9ah,h500aoverall[i][:][:])
#        rmseh2[count19][8] = math.sqrt(mean_squared_error(wt9ah,h500aoverall[i][:][:]))
#        biash2[count19][8] = (wt9ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count19][8] = corr2(wt9am,mslpaoverall[i][:][:])
#        rmsem2[count19][8] = math.sqrt(mean_squared_error(wt9am,mslpaoverall[i][:][:]))
#        biasm2[count19][8] = (wt9am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count19][8] = corr2(wt9au,uaoverall[i][:][:])
#        rmseu2[count19][8] = math.sqrt(mean_squared_error(wt9au,uaoverall[i][:][:]))
#        biasu2[count19][8] = (wt9au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count19][8] = corr2(wt9av,vaoverall[i][:][:])
#        rmsev2[count19][8] = math.sqrt(mean_squared_error(wt9av,vaoverall[i][:][:]))
#        biasv2[count19][8] = (wt9av[:][:] - vaoverall[i][:][:]).mean()
#        count19 = count19 + 1
#    elif( K[i] == 8):
#        corrsh[count8][7] = corr2(wt8ah, h500aoverall[i][:][:])
#        rmseh[count8][7] = math.sqrt(mean_squared_error(wt8ah,h500aoverall[i][:][:]))
#        biash[count8][7] = (wt8ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm[count8][7] = corr2(wt8am, mslpaoverall[i][:][:])
#        rmsem[count8][7] = math.sqrt(mean_squared_error(wt8am,mslpaoverall[i][:][:]))
#        biasm[count8][7] = (wt8am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu[count8][7] = corr2(wt8au, uaoverall[i][:][:])
#        rmseu[count8][7] = math.sqrt(mean_squared_error(wt8au,uaoverall[i][:][:]))
#        biasu[count8][7] = (wt8au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv[count8][7] = corr2(wt8av, vaoverall[i][:][:])
#        rmsev[count8][7] = math.sqrt(mean_squared_error(wt8av,vaoverall[i][:][:]))
#        biasv[count8][7] = (wt8av[:][:] - vaoverall[i][:][:]).mean()
#        count8 = count8 + 1
#        
#        corrsh2[count12][1] = corr2(wt2ah,h500aoverall[i][:][:])
#        rmseh2[count12][1] = math.sqrt(mean_squared_error(wt2ah,h500aoverall[i][:][:]))
#        biash2[count12][1] = (wt2ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count12][1] = corr2(wt2am,mslpaoverall[i][:][:])
#        rmsem2[count12][1] = math.sqrt(mean_squared_error(wt2am,mslpaoverall[i][:][:]))
#        biasm2[count12][1] = (wt2am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count12][1] = corr2(wt2au,uaoverall[i][:][:])
#        rmseu2[count12][1] = math.sqrt(mean_squared_error(wt2au,uaoverall[i][:][:]))
#        biasu2[count12][1] = (wt2au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count12][1] = corr2(wt2av,vaoverall[i][:][:])
#        rmsev2[count12][1] = math.sqrt(mean_squared_error(wt2av,vaoverall[i][:][:]))
#        biasv2[count12][1] = (wt2av[:][:] - vaoverall[i][:][:]).mean()
#        count12 = count12 + 1
#        
#        corrsh2[count13][2] = corr2(wt3ah,h500aoverall[i][:][:])
#        rmseh2[count13][2] = math.sqrt(mean_squared_error(wt3ah,h500aoverall[i][:][:]))
#        biash2[count13][2] = (wt3ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count13][2] = corr2(wt3am,mslpaoverall[i][:][:])
#        rmsem2[count13][2] = math.sqrt(mean_squared_error(wt3am,mslpaoverall[i][:][:]))
#        biasm2[count13][2] = (wt3am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count13][2] = corr2(wt3au,uaoverall[i][:][:])
#        rmseu2[count13][2] = math.sqrt(mean_squared_error(wt3au,uaoverall[i][:][:]))
#        biasu2[count13][2] = (wt3au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count13][2] = corr2(wt3av,vaoverall[i][:][:])
#        rmsev2[count13][2] = math.sqrt(mean_squared_error(wt3av,vaoverall[i][:][:]))
#        biasv2[count13][2] = (wt3av[:][:] - vaoverall[i][:][:]).mean()
#        count13 = count13 + 1
#        
#        corrsh2[count14][3] = corr2(wt4ah,h500aoverall[i][:][:])
#        rmseh2[count14][3] = math.sqrt(mean_squared_error(wt4ah,h500aoverall[i][:][:]))
#        biash2[count14][3] = (wt4ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count14][3] = corr2(wt4am,mslpaoverall[i][:][:])
#        rmsem2[count14][3] = math.sqrt(mean_squared_error(wt4am,mslpaoverall[i][:][:]))
#        biasm2[count14][3] = (wt4am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count14][3] = corr2(wt4au,uaoverall[i][:][:])
#        rmseu2[count14][3] = math.sqrt(mean_squared_error(wt4au,uaoverall[i][:][:]))
#        biasu2[count14][3] = (wt4au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count14][3] = corr2(wt4av,vaoverall[i][:][:])
#        rmsev2[count14][3] = math.sqrt(mean_squared_error(wt4av,vaoverall[i][:][:]))
#        biasv2[count14][3] = (wt4av[:][:] - vaoverall[i][:][:]).mean()
#        count14 = count14 + 1
#        
#        corrsh2[count15][4] = corr2(wt5ah,h500aoverall[i][:][:])
#        rmseh2[count15][4] = math.sqrt(mean_squared_error(wt5ah,h500aoverall[i][:][:]))
#        biash2[count15][4] = (wt5ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count15][4] = corr2(wt5am,mslpaoverall[i][:][:])
#        rmsem2[count15][4] = math.sqrt(mean_squared_error(wt5am,mslpaoverall[i][:][:]))
#        biasm2[count15][4] = (wt5am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count15][4] = corr2(wt5au,uaoverall[i][:][:])
#        rmseu2[count15][4] = math.sqrt(mean_squared_error(wt5au,uaoverall[i][:][:]))
#        biasu2[count15][4] = (wt5au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count15][4] = corr2(wt5av,vaoverall[i][:][:])
#        rmsev2[count15][4] = math.sqrt(mean_squared_error(wt5av,vaoverall[i][:][:]))
#        biasv2[count15][4] = (wt5av[:][:] - vaoverall[i][:][:]).mean()
#        count15 = count15 + 1
#        
#        corrsh2[count16][5] = corr2(wt6ah,h500aoverall[i][:][:])
#        rmseh2[count16][5] = math.sqrt(mean_squared_error(wt6ah,h500aoverall[i][:][:]))
#        biash2[count16][5] = (wt6ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count16][5] = corr2(wt6am,mslpaoverall[i][:][:])
#        rmsem2[count16][5] = math.sqrt(mean_squared_error(wt6am,mslpaoverall[i][:][:]))
#        biasm2[count16][5] = (wt6am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count16][5] = corr2(wt6au,uaoverall[i][:][:])
#        rmseu2[count16][5] = math.sqrt(mean_squared_error(wt6au,uaoverall[i][:][:]))
#        biasu2[count16][5] = (wt6au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count16][5] = corr2(wt6av,vaoverall[i][:][:])
#        rmsev2[count16][5] = math.sqrt(mean_squared_error(wt6av,vaoverall[i][:][:]))
#        biasv2[count16][5] = (wt6av[:][:] - vaoverall[i][:][:]).mean()
#        count16 = count16 + 1
#        
#        corrsh2[count17][6] = corr2(wt7ah,h500aoverall[i][:][:])
#        rmseh2[count17][6] = math.sqrt(mean_squared_error(wt7ah,h500aoverall[i][:][:]))
#        biash2[count17][6] = (wt7ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count17][6] = corr2(wt7am,mslpaoverall[i][:][:])
#        rmsem2[count17][6] = math.sqrt(mean_squared_error(wt7am,mslpaoverall[i][:][:]))
#        biasm2[count17][6] = (wt7am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count17][6] = corr2(wt7au,uaoverall[i][:][:])
#        rmseu2[count17][6] = math.sqrt(mean_squared_error(wt7au,uaoverall[i][:][:]))
#        biasu2[count17][6] = (wt7au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count17][6] = corr2(wt7av,vaoverall[i][:][:])
#        rmsev2[count17][6] = math.sqrt(mean_squared_error(wt7av,vaoverall[i][:][:]))
#        biasv2[count17][6] = (wt7av[:][:] - vaoverall[i][:][:]).mean()
#        count17 = count17 + 1
#                
#        corrsh2[count11][0] = corr2(wt1ah,h500aoverall[i][:][:])
#        rmseh2[count11][0] = math.sqrt(mean_squared_error(wt1ah,h500aoverall[i][:][:]))
#        biash2[count11][0] = (wt1ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count11][0] = corr2(wt1am,mslpaoverall[i][:][:])
#        rmsem2[count11][0] = math.sqrt(mean_squared_error(wt1am,mslpaoverall[i][:][:]))
#        biasm2[count11][0] = (wt1am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count11][0] = corr2(wt1au,uaoverall[i][:][:])
#        rmseu2[count11][0] = math.sqrt(mean_squared_error(wt1au,uaoverall[i][:][:]))
#        biasu2[count11][0] = (wt1au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count11][0] = corr2(wt1av,vaoverall[i][:][:])
#        rmsev2[count11][0] = math.sqrt(mean_squared_error(wt1av,vaoverall[i][:][:]))
#        biasv2[count11][0] = (wt1av[:][:] - vaoverall[i][:][:]).mean()
#        count11 = count11 + 1
#        
#                
#        corrsh2[count19][8] = corr2(wt9ah,h500aoverall[i][:][:])
#        rmseh2[count19][8] = math.sqrt(mean_squared_error(wt9ah,h500aoverall[i][:][:]))
#        biash2[count19][8] = (wt9ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count19][8] = corr2(wt9am,mslpaoverall[i][:][:])
#        rmsem2[count19][8] = math.sqrt(mean_squared_error(wt9am,mslpaoverall[i][:][:]))
#        biasm2[count19][8] = (wt9am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count19][8] = corr2(wt9au,uaoverall[i][:][:])
#        rmseu2[count19][8] = math.sqrt(mean_squared_error(wt9au,uaoverall[i][:][:]))
#        biasu2[count19][8] = (wt9au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count19][8] = corr2(wt9av,vaoverall[i][:][:])
#        rmsev2[count19][8] = math.sqrt(mean_squared_error(wt9av,vaoverall[i][:][:]))
#        biasv2[count19][8] = (wt9av[:][:] - vaoverall[i][:][:]).mean()
#        count19 = count19 + 1
#    else:
#        corrsh[count9][8] = corr2(wt9ah, h500aoverall[i][:][:])
#        rmseh[count9][8] = math.sqrt(mean_squared_error(wt9ah,h500aoverall[i][:][:]))
#        biash[count9][8] = (wt9ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm[count9][8] = corr2(wt9am, mslpaoverall[i][:][:])
#        rmsem[count9][8] = math.sqrt(mean_squared_error(wt9am,mslpaoverall[i][:][:]))
#        biasm[count9][8] = (wt9am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu[count9][8] = corr2(wt9au, uaoverall[i][:][:])
#        rmseu[count9][8] = math.sqrt(mean_squared_error(wt9au,uaoverall[i][:][:]))
#        biasu[count9][8] = (wt9au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv[count9][8] = corr2(wt9av, vaoverall[i][:][:])
#        rmsev[count9][8] = math.sqrt(mean_squared_error(wt9av,vaoverall[i][:][:]))
#        biasv[count9][8] = (wt9av[:][:] - vaoverall[i][:][:]).mean()
#        count9 = count9 + 1
#        
#        corrsh2[count12][1] = corr2(wt2ah,h500aoverall[i][:][:])
#        rmseh2[count12][1] = math.sqrt(mean_squared_error(wt2ah,h500aoverall[i][:][:]))
#        biash2[count12][1] = (wt2ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count12][1] = corr2(wt2am,mslpaoverall[i][:][:])
#        rmsem2[count12][1] = math.sqrt(mean_squared_error(wt2am,mslpaoverall[i][:][:]))
#        biasm2[count12][1] = (wt2am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count12][1] = corr2(wt2au,uaoverall[i][:][:])
#        rmseu2[count12][1] = math.sqrt(mean_squared_error(wt2au,uaoverall[i][:][:]))
#        biasu2[count12][1] = (wt2au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count12][1] = corr2(wt2av,vaoverall[i][:][:])
#        rmsev2[count12][1] = math.sqrt(mean_squared_error(wt2av,vaoverall[i][:][:]))
#        biasv2[count12][1] = (wt2av[:][:] - vaoverall[i][:][:]).mean()
#        count12 = count12 + 1
#        
#        corrsh2[count13][2] = corr2(wt3ah,h500aoverall[i][:][:])
#        rmseh2[count13][2] = math.sqrt(mean_squared_error(wt3ah,h500aoverall[i][:][:]))
#        biash2[count13][2] = (wt3ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count13][2] = corr2(wt3am,mslpaoverall[i][:][:])
#        rmsem2[count13][2] = math.sqrt(mean_squared_error(wt3am,mslpaoverall[i][:][:]))
#        biasm2[count13][2] = (wt3am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count13][2] = corr2(wt3au,uaoverall[i][:][:])
#        rmseu2[count13][2] = math.sqrt(mean_squared_error(wt3au,uaoverall[i][:][:]))
#        biasu2[count13][2] = (wt3au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count13][2] = corr2(wt3av,vaoverall[i][:][:])
#        rmsev2[count13][2] = math.sqrt(mean_squared_error(wt3av,vaoverall[i][:][:]))
#        biasv2[count13][2] = (wt3av[:][:] - vaoverall[i][:][:]).mean()
#        count13 = count13 + 1
#        
#        corrsh2[count14][3] = corr2(wt4ah,h500aoverall[i][:][:])
#        rmseh2[count14][3] = math.sqrt(mean_squared_error(wt4ah,h500aoverall[i][:][:]))
#        biash2[count14][3] = (wt4ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count14][3] = corr2(wt4am,mslpaoverall[i][:][:])
#        rmsem2[count14][3] = math.sqrt(mean_squared_error(wt4am,mslpaoverall[i][:][:]))
#        biasm2[count14][3] = (wt4am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count14][3] = corr2(wt4au,uaoverall[i][:][:])
#        rmseu2[count14][3] = math.sqrt(mean_squared_error(wt4au,uaoverall[i][:][:]))
#        biasu2[count14][3] = (wt4au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count14][3] = corr2(wt4av,vaoverall[i][:][:])
#        rmsev2[count14][3] = math.sqrt(mean_squared_error(wt4av,vaoverall[i][:][:]))
#        biasv2[count14][3] = (wt4av[:][:] - vaoverall[i][:][:]).mean()
#        count14 = count14 + 1
#        
#        corrsh2[count15][4] = corr2(wt5ah,h500aoverall[i][:][:])
#        rmseh2[count15][4] = math.sqrt(mean_squared_error(wt5ah,h500aoverall[i][:][:]))
#        biash2[count15][4] = (wt5ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count15][4] = corr2(wt5am,mslpaoverall[i][:][:])
#        rmsem2[count15][4] = math.sqrt(mean_squared_error(wt5am,mslpaoverall[i][:][:]))
#        biasm2[count15][4] = (wt5am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count15][4] = corr2(wt5au,uaoverall[i][:][:])
#        rmseu2[count15][4] = math.sqrt(mean_squared_error(wt5au,uaoverall[i][:][:]))
#        biasu2[count15][4] = (wt5au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count15][4] = corr2(wt5av,vaoverall[i][:][:])
#        rmsev2[count15][4] = math.sqrt(mean_squared_error(wt5av,vaoverall[i][:][:]))
#        biasv2[count15][4] = (wt5av[:][:] - vaoverall[i][:][:]).mean()
#        count15 = count15 + 1
#        
#        corrsh2[count16][5] = corr2(wt6ah,h500aoverall[i][:][:])
#        rmseh2[count16][5] = math.sqrt(mean_squared_error(wt6ah,h500aoverall[i][:][:]))
#        biash2[count16][5] = (wt6ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count16][5] = corr2(wt6am,mslpaoverall[i][:][:])
#        rmsem2[count16][5] = math.sqrt(mean_squared_error(wt6am,mslpaoverall[i][:][:]))
#        biasm2[count16][5] = (wt6am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count16][5] = corr2(wt6au,uaoverall[i][:][:])
#        rmseu2[count16][5] = math.sqrt(mean_squared_error(wt6au,uaoverall[i][:][:]))
#        biasu2[count16][5] = (wt6au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count16][5] = corr2(wt6av,vaoverall[i][:][:])
#        rmsev2[count16][5] = math.sqrt(mean_squared_error(wt6av,vaoverall[i][:][:]))
#        biasv2[count16][5] = (wt6av[:][:] - vaoverall[i][:][:]).mean()
#        count16 = count16 + 1
#        
#        corrsh2[count17][6] = corr2(wt7ah,h500aoverall[i][:][:])
#        rmseh2[count17][6] = math.sqrt(mean_squared_error(wt7ah,h500aoverall[i][:][:]))
#        biash2[count17][6] = (wt7ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count17][6] = corr2(wt7am,mslpaoverall[i][:][:])
#        rmsem2[count17][6] = math.sqrt(mean_squared_error(wt7am,mslpaoverall[i][:][:]))
#        biasm2[count17][6] = (wt7am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count17][6] = corr2(wt7au,uaoverall[i][:][:])
#        rmseu2[count17][6] = math.sqrt(mean_squared_error(wt7au,uaoverall[i][:][:]))
#        biasu2[count17][6] = (wt7au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count17][6] = corr2(wt7av,vaoverall[i][:][:])
#        rmsev2[count17][6] = math.sqrt(mean_squared_error(wt7av,vaoverall[i][:][:]))
#        biasv2[count17][6] = (wt7av[:][:] - vaoverall[i][:][:]).mean()
#        count17 = count17 + 1
#                
#        corrsh2[count18][7] = corr2(wt8ah,h500aoverall[i][:][:])
#        rmseh2[count18][7] = math.sqrt(mean_squared_error(wt8ah,h500aoverall[i][:][:]))
#        biash2[count18][7] = (wt8ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count18][7] = corr2(wt8am,mslpaoverall[i][:][:])
#        rmsem2[count18][7] = math.sqrt(mean_squared_error(wt8am,mslpaoverall[i][:][:]))
#        biasm2[count18][7] = (wt8am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count18][7] = corr2(wt8au,uaoverall[i][:][:])
#        rmseu2[count18][7] = math.sqrt(mean_squared_error(wt8au,uaoverall[i][:][:]))
#        biasu2[count18][7] = (wt8au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count18][7] = corr2(wt8av,vaoverall[i][:][:])
#        rmsev2[count18][7] = math.sqrt(mean_squared_error(wt8av,vaoverall[i][:][:]))
#        biasv2[count18][7] = (wt8av[:][:] - vaoverall[i][:][:]).mean()
#        count18 = count18 + 1
#        
#                
#        corrsh2[count11][0] = corr2(wt1ah,h500aoverall[i][:][:])
#        rmseh2[count11][0] = math.sqrt(mean_squared_error(wt1ah,h500aoverall[i][:][:]))
#        biash2[count11][0] = (wt1ah[:][:] - h500aoverall[i][:][:]).mean()
#        corrsm2[count11][0] = corr2(wt1am,mslpaoverall[i][:][:])
#        rmsem2[count11][0] = math.sqrt(mean_squared_error(wt1am,mslpaoverall[i][:][:]))
#        biasm2[count11][0] = (wt1am[:][:] - mslpaoverall[i][:][:]).mean()
#        corrsu2[count11][0] = corr2(wt1au,uaoverall[i][:][:])
#        rmseu2[count11][0] = math.sqrt(mean_squared_error(wt1au,uaoverall[i][:][:]))
#        biasu2[count11][0] = (wt1au[:][:] - uaoverall[i][:][:]).mean()
#        corrsv2[count11][0] = corr2(wt1av,vaoverall[i][:][:])
#        rmsev2[count11][0] = math.sqrt(mean_squared_error(wt1av,vaoverall[i][:][:]))
#        biasv2[count11][0] = (wt1av[:][:] - vaoverall[i][:][:]).mean()
#        count11 = count11 + 1
    i = i + 1
corrtotal = (corrsh + corrsm + corrsu + corrsv )/4
rmsetotal = (rmseh + rmsem + rmseu + rmsev) /4
biastotal = (biash + biasm + biasu + biasv) /4
i = i + 1
countoverall = np.zeros(9)
countoverall[0] = int(count1);
countoverall[1] = int(count2);
countoverall[2] = int(count3);
countoverall[3] = int(count4);
countoverall[4] = int(count5);
countoverall[5] = int(count6);
countoverall[6] = int(count7);
#countoverall[7] = int(count8);
#countoverall[8] = int(count9);
countoverall2 = np.zeros(9)
countoverall2[0] = int(count11);
countoverall2[1] = int(count12);
countoverall2[2] = int(count13);
countoverall2[3] = int(count14);
countoverall2[4] = int(count15);
countoverall2[5] = int(count16);
countoverall2[6] = int(count17);
# =============================================================================
# countoverall2[7] = int(count18);
# countoverall2[8] = int(count19);
# =============================================================================


#Find the Correlations for the individual dates above
corrsh3 = np.empty(shape=[3640,9])
rmseh3 = np.empty(shape=[3640,9])
biash3 = np.empty(shape=[3640,9])
corrsm3 = np.empty(shape=[3640,9])
rmsem3 = np.empty(shape=[3640,9])
biasm3 = np.empty(shape=[3640,9])
corrsu3 = np.empty(shape=[3640,9])
rmseu3 = np.empty(shape=[3640,9])
biasu3 = np.empty(shape=[3640,9])
corrsv3 = np.empty(shape=[3640,9])
rmsev3 = np.empty(shape=[3640,9])
biasv3 = np.empty(shape=[3640,9])

#countoverall3 = np.zeros(9)
#i = 0
#ct1 = 0
#ct2 = 0
#ct3 = 0
#ct4 = 0
#ct5 = 0
#ct6 = 0
#ct7 = 0
## =============================================================================
## ct8 = 0
## ct9 = 0
## =============================================================================
#
#while i <= 64:
#    clust = int(dates[i][2])
#    days = int(dates[i][1])
#    w = 0
#    startday = int(dates[i][0])
#    while w <= days:
#        if(clust == 1):
#            corrsh3[ct1][0] = corr2(wt1ah,h500aoverall[startday-1][:][:])
#            rmseh3[ct1][0] = math.sqrt(mean_squared_error(wt1ah,h500aoverall[startday-1][:][:]))
#            biash3[ct1][0] = (wt1ah[:][:] - h500aoverall[startday-1][:][:]).mean()
#            ct1 = ct1 + 1
#        elif(clust == 2):
#            corrsh3[ct2][1] = corr2(wt2ah,h500aoverall[startday-1][:][:])
#            rmseh3[ct2][1] = math.sqrt(mean_squared_error(wt2ah,h500aoverall[startday-1][:][:]))
#            biash3[ct2][1] = (wt2ah[:][:] - h500aoverall[startday-1][:][:]).mean()
#            ct2 = ct2 + 1
#        elif(clust == 3):
#            corrsh3[ct3][2] = corr2(wt3ah,h500aoverall[startday-1][:][:])
#            rmseh3[ct3][2] = math.sqrt(mean_squared_error(wt3ah,h500aoverall[startday-1][:][:]))
#            biash3[ct3][2] = (wt3ah[:][:] - h500aoverall[startday-1][:][:]).mean()
#            ct3 = ct3 + 1
#        elif(clust == 4):
#            corrsh3[ct4][3] = corr2(wt4ah,h500aoverall[startday-1][:][:])
#            rmseh3[ct4][3] = math.sqrt(mean_squared_error(wt4ah,h500aoverall[startday-1][:][:]))
#            biash3[ct4][3] = (wt4ah[:][:] - h500aoverall[startday-1][:][:]).mean()
#            ct4 = ct4 + 1
#        elif(clust == 5):
#            corrsh3[ct5][4] = corr2(wt5ah,h500aoverall[startday-1][:][:])
#            rmseh3[ct5][4] = math.sqrt(mean_squared_error(wt5ah,h500aoverall[startday-1][:][:]))
#            biash3[ct5][4] = (wt5ah[:][:] - h500aoverall[startday-1][:][:]).mean()
#            ct5 = ct5 + 1
#        elif(clust == 6):
#            corrsh3[ct6][5] = corr2(wt6ah,h500aoverall[startday-1][:][:])
#            rmseh3[ct6][5] = math.sqrt(mean_squared_error(wt6ah,h500aoverall[startday-1][:][:]))
#            biash3[ct6][5] = (wt6ah[:][:] - h500aoverall[startday-1][:][:]).mean()
#            ct6 = ct6 + 1
#        else:
#            corrsh3[ct7][6] = corr2(wt7ah,h500aoverall[startday-1][:][:])
#            rmseh3[ct7][6] = math.sqrt(mean_squared_error(wt7ah,h500aoverall[startday-1][:][:]))
#            biash3[ct7][6] = (wt7ah[:][:] - h500aoverall[startday-1][:][:]).mean()
#            ct7 = ct7 + 1
#        startday = startday + 1
#        w = w + 1
#    i = i + 1
#
#countoverall3[0] = ct1
#countoverall3[1] = ct2
#countoverall3[2] = ct3
#countoverall3[3] = ct4
#countoverall3[4] = ct5
#countoverall3[5] = ct6
#countoverall3[6] = ct7
#create the histogram for correlation coefficient
#binscorr = np.arange(-1.,1.,.1)
#i = 0
#fig = plt.figure(figsize=(16,10))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#while i <= clustnum -1:
#    tmp = corrsh[0:int(countoverall[i]),i]
#    tmp2 = corrsh2[0:int(countoverall2[i]),i]
#    hist, bins = np.histogram(tmp, bins = binscorr)
#    hist2, bins2 = np.histogram(tmp2, bins = binscorr)
#    width = 0.7 * (bins[1]-bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    fig.add_subplot(3,3,i+1)    
#    plt.bar(center, (hist/countoverall[i])*100, align='center', width=width)
#    plt.bar(center, (hist2/(3640-countoverall[i])) * 100, align='center', width=width, color='tab:gray')
#    plt.ylim(0,100,10)
#    plt.xlim(-1,1,.1)
#    if( i !=clustnum):
#        name = "WT" + str(i+1)
#    else:
#        name = "WT" + str(i)
#    plt.title(name)
#    i = i + 1
#fig.suptitle('Correlation Coefficients for 500 hPa Heights')
#plt.savefig(outdir+'correlationh500.png', bbox_inches='tight')
#
#
##create the histogram for rmse
#binsrmse = np.arange(0,250,20)
#i = 0
#fig = plt.figure(figsize=(16,10))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#while i <= clustnum-1:
#    tmp = rmseh[0:int(int(countoverall[i])),i]
#    tmp2 = rmseh2[0:int(countoverall2[i]),i]
#    hist, bins = np.histogram(tmp, bins = binsrmse)
#    hist2, bins2 = np.histogram(tmp2, bins = binsrmse)
#    width = 0.7 * (bins[1]-bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    fig.add_subplot(3,3,i+1)    
#    plt.bar(center, (hist/countoverall[i])*100, align='center', width=width)
#    plt.bar(center, (hist2/(3640-countoverall[i])) * 100, align='center', width=width, color='tab:gray')
#    plt.ylim(0,100,10)
#    plt.xlim(0,250,25)
#    if( i !=clustnum):
#        name = "WT" + str(i+1)
#    else:
#        name = "WT" + str(i)
#    plt.title(name)
#    i = i + 1
#fig.suptitle('RMSE for 500 hPa heights')
#plt.savefig(outdir+'rmseh500.png', bbox_inches='tight')
#
##create the histogram for bias
#binsbias = np.arange(-200,200,20)
#i = 0
#fig = plt.figure(figsize=(16,10))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#while i <= clustnum-1:
#    tmp = biash[0:int(countoverall[i]),i]
#    tmp2 = biash2[0:int(countoverall2[i]),i]
#    hist, bins = np.histogram(tmp, bins = binsbias)
#    hist2, bins2 = np.histogram(tmp2, bins = binsbias)
#    width = 0.7 * (bins[1]-bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    fig.add_subplot(3,3,i+1)    
#    plt.bar(center, (hist/countoverall[i])*100, align='center', width=width)
#    plt.bar(center, (hist2/(3640-countoverall[i])) * 100, align='center', width=width, color='tab:gray')
#    plt.ylim(0,100,10)
#    plt.xlim(-200,200,50)
#    if( i !=clustnum):
#        name = "WT" + str(i+1)
#    else:
#        name = "WT" + str(i)
#    plt.title(name)
#    i = i + 1
#fig.suptitle('Bias for 500 hPa heights')
#plt.savefig(outdir+'biash500.png', bbox_inches='tight')
#
##Create for MSLP
##Create the histogram for correlation coefficient
#binscorr = np.arange(-1.,1.,.1)
#i = 0
#fig = plt.figure(figsize=(16,10))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#while i <= clustnum -1:
#    tmp = corrsm[0:int(countoverall[i]),i]
#    tmp2 = corrsm2[0:int(countoverall2[i]),i]
#    hist, bins = np.histogram(tmp, bins = binscorr)
#    hist2, bins2 = np.histogram(tmp2, bins = binscorr)
#    width = 0.7 * (bins[1]-bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    fig.add_subplot(3,3,i+1)    
#    plt.bar(center, (hist/countoverall[i])*100, align='center', width=width)
#    plt.bar(center, (hist2/(3640-countoverall[i])) * 100, align='center', width=width, color='tab:gray')
#    plt.ylim(0,100,10)
#    plt.xlim(-1,1,.1)
#    if( i !=clustnum):
#        name = "WT" + str(i+1)
#    else:
#        name = "WT" + str(i)
#    plt.title(name)
#    i = i + 1
#fig.suptitle('Correlation Coefficients for MSLP')
#plt.savefig(outdir+'correlationmslp.png', bbox_inches='tight')
#
#
##create the histogram for rmse
#binsrmse = np.arange(0,800,50)
#i = 0
#fig = plt.figure(figsize=(16,10))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#while i <= clustnum-1:
#    tmp = rmsem[0:int(countoverall[i]),i]
#    tmp2 = rmsem2[0:int(countoverall2[i]),i]
#    hist, bins = np.histogram(tmp, bins = binsrmse)
#    hist2, bins2 = np.histogram(tmp2, bins = binsrmse)
#    width = 0.7 * (bins[1]-bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    fig.add_subplot(3,3,i+1)    
#    plt.bar(center, (hist/countoverall[i])*100, align='center', width=width)
#    plt.bar(center, (hist2/(3640-countoverall[i])) * 100, align='center', width=width, color='tab:gray')
#    plt.ylim(0,100,10)
#    plt.xlim(0,250,25)
#    if( i !=clustnum):
#        name = "WT" + str(i+1)
#    else:
#        name = "WT" + str(i)
#    plt.title(name)
#    i = i + 1
#fig.suptitle('RMSE for MSLP')
#plt.savefig(outdir+'rmsemslp.png', bbox_inches='tight')
#
##create the histogram for bias
#binsbias = np.arange(-200,200,20)
#i = 0
#fig = plt.figure(figsize=(16,10))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#while i <= clustnum-1:
#    tmp = biasm[0:int(countoverall[i]),i]
#    tmp2 = biasm2[0:int(countoverall2[i]),i]
#    hist, bins = np.histogram(tmp, bins = binsbias)
#    hist2, bins2 = np.histogram(tmp2, bins = binsbias)
#    width = 0.7 * (bins[1]-bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    fig.add_subplot(3,3,i+1)    
#    plt.bar(center, (hist/countoverall[i])*100, align='center', width=width)
#    plt.bar(center, (hist2/(3640-countoverall[i])) * 100, align='center', width=width, color='tab:gray')
#    plt.ylim(0,100,10)
#    plt.xlim(-200,200,50)
#    if( i !=clustnum):
#        name = "WT" + str(i+1)
#    else:
#        name = "WT" + str(i)
#    plt.title(name)
#    i = i + 1
#fig.suptitle('Bias for MSLP')
#plt.savefig(outdir+'biasmslp.png', bbox_inches='tight')
#
##Create for u850 winds
##Create the histogram for correlation coefficient
#binscorr = np.arange(-1.,1.,.1)
#i = 0
#fig = plt.figure(figsize=(16,10))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#while i <= clustnum -1:
#    tmp = corrsu[0:int(countoverall[i]),i]
#    tmp2 = corrsu2[0:int(countoverall2[i]),i]
#    hist, bins = np.histogram(tmp, bins = binscorr)
#    hist2, bins2 = np.histogram(tmp2, bins = binscorr)
#    width = 0.7 * (bins[1]-bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    fig.add_subplot(3,3,i+1)    
#    plt.bar(center, (hist/countoverall[i])*100, align='center', width=width)
#    plt.bar(center, (hist2/(3640-countoverall[i])) * 100, align='center', width=width, color='tab:gray')
#    plt.ylim(0,100,10)
#    plt.xlim(-1,1,.1)
#    if( i !=clustnum):
#        name = "WT" + str(i+1)
#    else:
#        name = "WT" + str(i)
#    plt.title(name)
#    i = i + 1
#fig.suptitle('Correlation Coefficients for u850')
#plt.savefig(outdir+'correlationu850.png', bbox_inches='tight')
#
#
##create the histogram for rmse
#binsrmse = np.arange(4,8,0.1)
#i = 0
#fig = plt.figure(figsize=(16,10))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#while i <= clustnum-1:
#    tmp = rmseu[0:int(countoverall[i]),i]
#    tmp2 = rmseu2[0:int(countoverall2[i]),i]
#    hist, bins = np.histogram(tmp, bins = binsrmse)
#    hist2, bins2 = np.histogram(tmp2, bins = binsrmse)
#    width = 0.7 * (bins[1]-bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    fig.add_subplot(3,3,i+1)    
#    plt.bar(center, (hist/countoverall[i])*100, align='center', width=width)
#    plt.bar(center, (hist2/(3640-countoverall[i])) * 100, align='center', width=width, color='tab:gray')
#    plt.ylim(0,100,10)
#    plt.xlim(4,8,.25)
#    if( i !=clustnum):
#        name = "WT" + str(i+1)
#    else:
#        name = "WT" + str(i)
#    plt.title(name)
#    i = i + 1
#fig.suptitle('RMSE for u850')
#plt.savefig(outdir+'rmseu850.png', bbox_inches='tight')
#
##create the histogram for bias
#binsbias = np.arange(-5,5,0.1)
#i = 0
#fig = plt.figure(figsize=(16,10))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#while i <= clustnum-1:
#    tmp = biasu[0:int(countoverall[i]),i]
#    tmp2 = biasu2[0:int(countoverall2[i]),i]
#    hist, bins = np.histogram(tmp, bins = binsbias)
#    hist2, bins2 = np.histogram(tmp2, bins = binsbias)
#    width = 0.7 * (bins[1]-bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    fig.add_subplot(3,3,i+1)    
#    plt.bar(center, (hist/countoverall[i])*100, align='center', width=width)
#    plt.bar(center, (hist2/(3640-countoverall[i])) * 100, align='center', width=width, color='tab:gray')
#    plt.ylim(0,100,10)
#    plt.xlim(-5,5,0.25)
#    if( i !=clustnum):
#        name = "WT" + str(i+1)
#    else:
#        name = "WT" + str(i)
#    plt.title(name)
#    i = i + 1
#fig.suptitle('Bias for u850')
#plt.savefig(outdir+'biasu850.png', bbox_inches='tight')
#
##Create for v850 winds
##Create the histogram for correlation coefficient
#binscorr = np.arange(-1.,1.,.1)
#i = 0
#fig = plt.figure(figsize=(16,10))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#while i <= clustnum -1:
#    tmp = corrsv[0:int(countoverall[i]),i]
#    tmp2 = corrsv2[0:int(countoverall2[i]),i]
#    hist, bins = np.histogram(tmp, bins = binscorr)
#    hist2, bins2 = np.histogram(tmp2, bins = binscorr)
#    width = 0.7 * (bins[1]-bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    fig.add_subplot(3,3,i+1)    
#    plt.bar(center, (hist/countoverall[i])*100, align='center', width=width)
#    plt.bar(center, (hist2/(3640-countoverall[i])) * 100, align='center', width=width, color='tab:gray')
#    plt.ylim(0,100,10)
#    plt.xlim(-1,1,.1)
#    if( i !=clustnum):
#        name = "WT" + str(i+1)
#    else:
#        name = "WT" + str(i)
#    plt.title(name)
#    i = i + 1
#fig.suptitle('Correlation Coefficients for v850')
#plt.savefig(outdir+'correlationv850.png', bbox_inches='tight')
#
#
##create the histogram for rmse
#binsrmse = np.arange(4,8,0.1)
#i = 0
#fig = plt.figure(figsize=(16,10))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#while i <= clustnum-1:
#    tmp = rmsev[0:int(countoverall[i]),i]
#    tmp2 = rmsev2[0:int(countoverall2[i]),i]
#    hist, bins = np.histogram(tmp, bins = binsrmse)
#    hist2, bins2 = np.histogram(tmp2, bins = binsrmse)
#    width = 0.7 * (bins[1]-bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    fig.add_subplot(3,3,i+1)    
#    plt.bar(center, (hist/countoverall[i])*100, align='center', width=width)
#    plt.bar(center, (hist2/(3640-countoverall[i])) * 100, align='center', width=width, color='tab:gray')
#    plt.ylim(0,100,10)
#    plt.xlim(4,8,.25)
#    if( i !=clustnum):
#        name = "WT" + str(i+1)
#    else:
#        name = "WT" + str(i)
#    plt.title(name)
#    i = i + 1
#fig.suptitle('RMSE for v850')
#plt.savefig(outdir+'rmsev850.png', bbox_inches='tight')
#
##create the histogram for bias
#binsbias = np.arange(-5,5,0.25)
#i = 0
#fig = plt.figure(figsize=(16,10))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#while i <= clustnum-1:
#    tmp = biasv[0:int(countoverall[i]),i]
#    tmp2 = biasv2[0:int(countoverall2[i]),i]
#    hist, bins = np.histogram(tmp, bins = binsbias)
#    hist2, bins2 = np.histogram(tmp2, bins = binsbias)
#    width = 0.7 * (bins[1]-bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    fig.add_subplot(3,3,i+1)    
#    plt.bar(center, (hist/countoverall[i])*100, align='center', width=width)
#    plt.bar(center, (hist2/(3640-countoverall[i])) * 100, align='center', width=width, color='tab:gray')
#    plt.ylim(0,100,10)
#    plt.xlim(-5,5,0.1)
#    if( i !=clustnum):
#        name = "WT" + str(i+1)
#    else:
#        name = "WT" + str(i)
#    plt.title(name)
#    i = i + 1
#fig.suptitle('Bias for v850')
#plt.savefig(outdir+'biasv850.png', bbox_inches='tight')
#
#pd.DataFrame(dateindices).to_csv("wt2correlations.csv", header = None, index = None)
#
###Create for Persistence >= 5 days
###create the histogram for correlation coefficient
##binscorr = np.arange(-1.,1.,.1)
##i = 0
##fig = plt.figure(figsize=(16,10))
##fig.subplots_adjust(hspace=0.4, wspace=0.4)
##while i <= clustnum -1:
##    tmp = corrsh3[0:int(countoverall3[i]),i]
##    hist, bins = np.histogram(tmp, bins = binscorr)
##    width = 0.7 * (bins[1]-bins[0])
##    center = (bins[:-1] + bins[1:]) / 2
##    fig.add_subplot(3,3,i+1)    
##    plt.bar(center, (hist/countoverall3[i])*100, align='center', width=width)
##    plt.ylim(0,100,10)
##    plt.xlim(-1,1,.1)
##    if( i !=clustnum):
##        name = "WT" + str(i+1)
##    else:
##        name = "WT" + str(i)
##    plt.title(name)
##    i = i + 1
##fig.suptitle('Correlation Coefficients for 500 hPa Heights w/ Persistence >= 5 Days')
##plt.savefig(outdir+'correlationh500persist.png', bbox_inches='tight')
##
##
###create the histogram for rmse
##binsrmse = np.arange(0,250,20)
##i = 0
##fig = plt.figure(figsize=(16,10))
##fig.subplots_adjust(hspace=0.4, wspace=0.4)
##while i <= clustnum-1:
##    tmp = rmseh3[0:int(int(countoverall3[i])),i]
##    hist, bins = np.histogram(tmp, bins = binsrmse)
##    width = 0.7 * (bins[1]-bins[0])
##    center = (bins[:-1] + bins[1:]) / 2
##    fig.add_subplot(3,3,i+1)    
##    plt.bar(center, (hist/countoverall3[i])*100, align='center', width=width)
##    plt.ylim(0,100,10)
##    plt.xlim(0,250,25)
##    if( i !=clustnum):
##        name = "WT" + str(i+1)
##    else:
##        name = "WT" + str(i)
##    plt.title(name)
##    i = i + 1
##fig.suptitle('RMSE for 500 hPa heights w/ Persistence >= 5 Days')
##plt.savefig(outdir+'rmseh500persist.png', bbox_inches='tight')
##
###create the histogram for bias
##binsbias = np.arange(-200,200,20)
##i = 0
##fig = plt.figure(figsize=(16,10))
##fig.subplots_adjust(hspace=0.4, wspace=0.4)
##while i <= clustnum-1:
##    tmp = biash3[0:int(countoverall3[i]),i]
##    hist, bins = np.histogram(tmp, bins = binsbias)
##    width = 0.7 * (bins[1]-bins[0])
##    center = (bins[:-1] + bins[1:]) / 2
##    fig.add_subplot(3,3,i+1)    
##    plt.bar(center, (hist/countoverall3[i])*100, align='center', width=width)
##    plt.ylim(0,100,10)
##    plt.xlim(-200,200,50)
##    if( i !=clustnum):
##        name = "WT" + str(i+1)
##    else:
##        name = "WT" + str(i)
##    plt.title(name)
##    i = i + 1
##fig.suptitle('Bias for 500 hPa heights w/ Persistence >= 5 Days')
##plt.savefig(outdir+'biash500persist.png', bbox_inches='tight')
#
###create the histogram for correlation coefficient
##binscorr = np.arange(-1.,1.,.1)
##i = 0
##fig = plt.figure(figsize=(16,10))
##fig.subplots_adjust(hspace=0.4, wspace=0.4)
##while i <= clustnum -1:
##    tmp = corrsm3[0:int(countoverall3[i]),i]
##    hist, bins = np.histogram(tmp, bins = binscorr)
##    width = 0.7 * (bins[1]-bins[0])
##    center = (bins[:-1] + bins[1:]) / 2
##    fig.add_subplot(3,3,i+1)    
##    plt.bar(center, (hist/countoverall3[i])*100, align='center', width=width)
##    plt.ylim(0,100,10)
##    plt.xlim(-1,1,.1)
##    if( i !=clustnum):
##        name = "WT" + str(i+1)
##    else:
##        name = "WT" + str(i)
##    plt.title(name)
##    i = i + 1
##fig.suptitle('Correlation Coefficients for MSLP w/ Persistence >= 5 Days')
##plt.savefig(outdir+'correlationmslppersist.png', bbox_inches='tight')
##
##
###create the histogram for rmse
##binsrmse = np.arange(0,250,20)
##i = 0
##fig = plt.figure(figsize=(16,10))
##fig.subplots_adjust(hspace=0.4, wspace=0.4)
##while i <= clustnum-1:
##    tmp = rmsem3[0:int(int(countoverall3[i])),i]
##    hist, bins = np.histogram(tmp, bins = binsrmse)
##    width = 0.7 * (bins[1]-bins[0])
##    center = (bins[:-1] + bins[1:]) / 2
##    fig.add_subplot(3,3,i+1)    
##    plt.bar(center, (hist/countoverall3[i])*100, align='center', width=width)
##    plt.ylim(0,100,10)
##    plt.xlim(0,250,25)
##    if( i !=clustnum):
##        name = "WT" + str(i+1)
##    else:
##        name = "WT" + str(i)
##    plt.title(name)
##    i = i + 1
##fig.suptitle('RMSE for MSLP w/ Persistence >= 5 Days')
##plt.savefig(outdir+'rmsemslppersist.png', bbox_inches='tight')
##
###create the histogram for bias
##binsbias = np.arange(-200,200,20)
##i = 0
##fig = plt.figure(figsize=(16,10))
##fig.subplots_adjust(hspace=0.4, wspace=0.4)
##while i <= clustnum-1:
##    tmp = biasm3[0:int(countoverall3[i]),i]
##    hist, bins = np.histogram(tmp, bins = binsbias)
##    width = 0.7 * (bins[1]-bins[0])
##    center = (bins[:-1] + bins[1:]) / 2
##    fig.add_subplot(3,3,i+1)    
##    plt.bar(center, (hist/countoverall3[i])*100, align='center', width=width)
##    plt.ylim(0,100,10)
##    plt.xlim(-200,200,50)
##    if( i !=clustnum):
##        name = "WT" + str(i+1)
##    else:
##        name = "WT" + str(i)
##    plt.title(name)
##    i = i + 1
##fig.suptitle('Bias for MSLP w/ Persistence >= 5 Days')
##plt.savefig(outdir+'biasmslppersist.png', bbox_inches='tight')
##
###create the histogram for correlation coefficient
##binscorr = np.arange(-1.,1.,.1)
##i = 0
##fig = plt.figure(figsize=(16,10))
##fig.subplots_adjust(hspace=0.4, wspace=0.4)
##while i <= clustnum -1:
##    tmp = corrsu3[0:int(countoverall3[i]),i]
##    hist, bins = np.histogram(tmp, bins = binscorr)
##    width = 0.7 * (bins[1]-bins[0])
##    center = (bins[:-1] + bins[1:]) / 2
##    fig.add_subplot(3,3,i+1)    
##    plt.bar(center, (hist/countoverall3[i])*100, align='center', width=width)
##    plt.ylim(0,100,10)
##    plt.xlim(-1,1,.1)
##    if( i !=clustnum):
##        name = "WT" + str(i+1)
##    else:
##        name = "WT" + str(i)
##    plt.title(name)
##    i = i + 1
##fig.suptitle('Correlation Coefficients for U 850 w/ Persistence >= 5 Days')
##plt.savefig(outdir+'correlationu850persist.png', bbox_inches='tight')
##
##
###create the histogram for rmse
##binsrmse = np.arange(0,250,20)
##i = 0
##fig = plt.figure(figsize=(16,10))
##fig.subplots_adjust(hspace=0.4, wspace=0.4)
##while i <= clustnum-1:
##    tmp = rmseu3[0:int(int(countoverall3[i])),i]
##    hist, bins = np.histogram(tmp, bins = binsrmse)
##    width = 0.7 * (bins[1]-bins[0])
##    center = (bins[:-1] + bins[1:]) / 2
##    fig.add_subplot(3,3,i+1)    
##    plt.bar(center, (hist/countoverall3[i])*100, align='center', width=width)
##    plt.ylim(0,100,10)
##    plt.xlim(0,250,25)
##    if( i !=clustnum):
##        name = "WT" + str(i+1)
##    else:
##        name = "WT" + str(i)
##    plt.title(name)
##    i = i + 1
##fig.suptitle('RMSE for U 850 w/ Persistence >= 5 Days')
##plt.savefig(outdir+'rmseu850persist.png', bbox_inches='tight')
##
###create the histogram for bias
##binsbias = np.arange(-200,200,20)
##i = 0
##fig = plt.figure(figsize=(16,10))
##fig.subplots_adjust(hspace=0.4, wspace=0.4)
##while i <= clustnum-1:
##    tmp = biasu3[0:int(countoverall3[i]),i]
##    hist, bins = np.histogram(tmp, bins = binsbias)
##    width = 0.7 * (bins[1]-bins[0])
##    center = (bins[:-1] + bins[1:]) / 2
##    fig.add_subplot(3,3,i+1)    
##    plt.bar(center, (hist/countoverall3[i])*100, align='center', width=width)
##    plt.ylim(0,100,10)
##    plt.xlim(-200,200,50)
##    if( i !=clustnum):
##        name = "WT" + str(i+1)
##    else:
##        name = "WT" + str(i)
##    plt.title(name)
##    i = i + 1
##fig.suptitle('Bias for U 850 w/ Persistence >= 5 Days')
##plt.savefig(outdir+'biasu850persist.png', bbox_inches='tight')
##
###create the histogram for correlation coefficient
##binscorr = np.arange(-1.,1.,.1)
##i = 0
##fig = plt.figure(figsize=(16,10))
##fig.subplots_adjust(hspace=0.4, wspace=0.4)
##while i <= clustnum -1:
##    tmp = corrsv3[0:int(countoverall3[i]),i]
##    hist, bins = np.histogram(tmp, bins = binscorr)
##    width = 0.7 * (bins[1]-bins[0])
##    center = (bins[:-1] + bins[1:]) / 2
##    fig.add_subplot(3,3,i+1)    
##    plt.bar(center, (hist/countoverall3[i])*100, align='center', width=width)
##    plt.ylim(0,100,10)
##    plt.xlim(-1,1,.1)
##    if( i !=clustnum):
##        name = "WT" + str(i+1)
##    else:
##        name = "WT" + str(i)
##    plt.title(name)
##    i = i + 1
##fig.suptitle('Correlation Coefficients for V 850 w/ Persistence >= 5 Days')
##plt.savefig(outdir+'correlationv850persist.png', bbox_inches='tight')
##
##
###create the histogram for rmse
##binsrmse = np.arange(0,250,20)
##i = 0
##fig = plt.figure(figsize=(16,10))
##fig.subplots_adjust(hspace=0.4, wspace=0.4)
##while i <= clustnum-1:
##    tmp = rmsev3[0:int(int(countoverall3[i])),i]
##    hist, bins = np.histogram(tmp, bins = binsrmse)
##    width = 0.7 * (bins[1]-bins[0])
##    center = (bins[:-1] + bins[1:]) / 2
##    fig.add_subplot(3,3,i+1)    
##    plt.bar(center, (hist/countoverall3[i])*100, align='center', width=width)
##    plt.ylim(0,100,10)
##    plt.xlim(0,250,25)
##    if( i !=clustnum):
##        name = "WT" + str(i+1)
##    else:
##        name = "WT" + str(i)
##    plt.title(name)
##    i = i + 1
##fig.suptitle('RMSE for V 850 w/ Persistence >= 5 Days')
##plt.savefig(outdir+'rmsev850persist.png', bbox_inches='tight')
##
###create the histogram for bias
##binsbias = np.arange(-200,200,20)
##i = 0
##fig = plt.figure(figsize=(16,10))
##fig.subplots_adjust(hspace=0.4, wspace=0.4)
##while i <= clustnum-1:
##    tmp = biasv3[0:int(countoverall3[i]),i]
##    hist, bins = np.histogram(tmp, bins = binsbias)
##    width = 0.7 * (bins[1]-bins[0])
##    center = (bins[:-1] + bins[1:]) / 2
##    fig.add_subplot(3,3,i+1)    
##    plt.bar(center, (hist/countoverall3[i])*100, align='center', width=width)
##    plt.ylim(0,100,10)
##    plt.xlim(-200,200,50)
##    if( i !=clustnum):
##        name = "WT" + str(i+1)
##    else:
##        name = "WT" + str(i)
##    plt.title(name)
##    i = i + 1
##fig.suptitle('Bias for V 850 w/ Persistence >= 5 Days')
##plt.savefig(outdir+'biasv850persist.png', bbox_inches='tight')

ncorrsh = []
ncorrsm = []
ncorrsu = []
ncorrsv = []
nrmseh = []
nrmsem = []
nrmseu = []
nrmsev = []
n2corrsh = []
n2corrsm = []
n2corrsu = []
n2corrsv = []
n2rmseh = []
n2rmsem = []
n2rmseu = []
n2rmsev = []

for i in range(7):
    count = countoverall[i]
    count1 = countoverall2[i]
    for j in range(int(count)):
        ncorrsh.append(corrsh[j,i])
        ncorrsm.append(corrsm[j,i])
        ncorrsu.append(corrsu[j,i])
        ncorrsv.append(corrsv[j,i])
        nrmsem.append(rmsem[j,i])
        nrmseh.append(rmseh[j,i])
        nrmseu.append(rmseu[j,i])
        nrmsev.append(rmsev[j,i])
    for k in range(int(count1)):
        n2corrsh.append(corrsh2[k,i])
        n2corrsm.append(corrsm2[k,i])
        n2corrsu.append(corrsu2[k,i])
        n2corrsv.append(corrsv2[k,i])
        n2rmsem.append(rmsem2[k,i])
        n2rmseh.append(rmseh2[k,i])
        n2rmseu.append(rmseu2[k,i])
        n2rmsev.append(rmsev2[k,i])

binscorr = np.arange(-1.,1.,.1)
fig, ax = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.8,wspace=0.8)

tmp = ncorrsh[:]
tmp2 = n2corrsh[:]
hist, bins = np.histogram(tmp, bins = binscorr)
hist2, bins2 = np.histogram(tmp2, bins = binscorr)
width = 0.7 * (bins[1]-bins[0])
center = (bins[:-1] + bins[1:]) / 2   
ax[0,0].bar(center, (hist/float(np.sum(hist)))*100, align='center', width=width)
ax[0,0].bar(center, (hist2/float(np.sum(hist2))) * 100, align='center', width=width, color='tab:gray')
#ax.ylim(0,100,10)
#ax.xlim(-1,1,.1)
ax[0,0].title.set_text('Correlation Coefficient H500')
ax[0,0].set_ylabel('% of days')

tmp = ncorrsm[:]
tmp2 = n2corrsm[:]
hist, bins = np.histogram(tmp, bins = binscorr)
hist2, bins2 = np.histogram(tmp2, bins = binscorr)
width = 0.7 * (bins[1]-bins[0])
center = (bins[:-1] + bins[1:]) / 2   
ax[0,1].bar(center, (hist/float(np.sum(hist)))*100, align='center', width=width)
ax[0,1].bar(center, (hist2/float(np.sum(hist2))) * 100, align='center', width=width, color='tab:gray')
ax[0,1].title.set_text('Correlation Coefficient MSLP')
ax[0,1].set_ylabel('% of days')

tmp = ncorrsu[:]
tmp2 = n2corrsu[:]
hist, bins = np.histogram(tmp, bins = binscorr)
hist2, bins2 = np.histogram(tmp2, bins = binscorr)
width = 0.7 * (bins[1]-bins[0])
center = (bins[:-1] + bins[1:]) / 2   
ax[1,0].bar(center, (hist/float(np.sum(hist)))*100, align='center', width=width)
ax[1,0].bar(center, (hist2/float(np.sum(hist2))) * 100, align='center', width=width, color='tab:gray')
ax[1,0].title.set_text('Correlation Coefficient U850')
ax[1,0].set_ylabel('% of days')

tmp = ncorrsv[:]
tmp2 = n2corrsv[:]
hist, bins = np.histogram(tmp, bins = binscorr)
hist2, bins2 = np.histogram(tmp2, bins = binscorr)
width = 0.7 * (bins[1]-bins[0])
center = (bins[:-1] + bins[1:]) / 2   
ax[1,1].bar(center, (hist/float(np.sum(hist)))*100, align='center', width=width)
ax[1,1].bar(center, (hist2/float(np.sum(hist2))) * 100, align='center', width=width, color='tab:gray')
ax[1,1].title.set_text('Correlation Coefficient V850')
ax[1,1].set_ylabel('% of days')

for (m,n), subplot in np.ndenumerate(ax):
    subplot.set_xlim(-1,1)
    subplot.set_ylim(0,20)

plt.savefig(outdir+'cc_overall.png', bbox_inches='tight')

#create the histogram for rmse
binsrmse = np.arange(4,8,0.1)
fig, ax = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.8,wspace=0.8)
binsrmse = np.arange(0,200,10)
tmp = nrmseh[:]
tmp2 = n2rmseh[:]
hist, bins = np.histogram(tmp, bins = binsrmse)
hist2, bins2 = np.histogram(tmp2, bins = binsrmse)
width = 0.7 * (bins[1]-bins[0])
center = (bins[:-1] + bins[1:]) / 2    
ax[0,0].bar(center, (hist/float(np.sum(hist)))*100, align='center', width=width)
ax[0,0].bar(center, (hist2/float(np.sum(hist2))) * 100, align='center', width=width, color='tab:gray')
ax[0,0].set_ylim(0,20,5)
ax[0,0].set_xlim(0,200,20)
ax[0,0].title.set_text('RMSE for H500')
ax[0,0].set_ylabel('% of days')

binsrmse = np.arange(0,500,25)
tmp = nrmsem[:]
tmp2 = n2rmsem[:]
hist, bins = np.histogram(tmp, bins = binsrmse)
hist2, bins2 = np.histogram(tmp2, bins = binsrmse)
width = 0.7 * (bins[1]-bins[0])
center = (bins[:-1] + bins[1:]) / 2    
ax[0,1].bar(center, (hist/float(np.sum(hist)))*100, align='center', width=width)
ax[0,1].bar(center, (hist2/float(np.sum(hist2))) * 100, align='center', width=width, color='tab:gray')
ax[0,1].title.set_text('RMSE for MSLP')
ax[0,1].set_ylim(0,20,5)
ax[0,1].set_ylabel('% of days')

binsrmse = np.arange(0,10,.1)
tmp = nrmseu[:]
tmp2 = n2rmseu[:]
hist, bins = np.histogram(tmp, bins = binsrmse)
hist2, bins2 = np.histogram(tmp2, bins = binsrmse)
width = 0.7 * (bins[1]-bins[0])
center = (bins[:-1] + bins[1:]) / 2    
ax[1,0].bar(center, (hist/float(np.sum(hist)))*100, align='center', width=width)
ax[1,0].bar(center, (hist2/float(np.sum(hist2))) * 100, align='center', width=width, color='tab:gray')
ax[1,0].title.set_text('RMSE for U850')
ax[1,0].set_ylim(0,20,5)
ax[1,0].set_xlim(0,10,1)
ax[1,0].set_ylabel('% of days')


binsrmse = np.arange(0,10,.1)
tmp = nrmsev[:]
tmp2 = n2rmsev[:]
hist, bins = np.histogram(tmp, bins = binsrmse)
hist2, bins2 = np.histogram(tmp2, bins = binsrmse)
width = 0.7 * (bins[1]-bins[0])
center = (bins[:-1] + bins[1:]) / 2    
ax[1,1].bar(center, (hist/float(np.sum(hist)))*100, align='center', width=width)
ax[1,1].bar(center, (hist2/float(np.sum(hist2))) * 100, align='center', width=width, color='tab:gray')
ax[1,1].title.set_text('RMSE for V850')
ax[1,1].set_ylim(0,20,5)
ax[1,1].set_xlim(0,10,1)
ax[1,1].set_ylabel('% of days')

plt.savefig(outdir+'rmse_overall.png', bbox_inches='tight')

#Now put the data into a dataframe for the h500 correlation and each cluster

df = pd.DataFrame()
df['Cluster'] = mat_contents2['K'][:,6]
df['Correlation'] = ncorrsh

#Group together by cluster
groups = df.groupby(df.Cluster).groups
min_val = np.min(df.groupby(df.Cluster).size())

#Now pull out the highest correlations up to the number of min val and put in new dataframe
df_new = pd.DataFrame()
for i in range(7):
    temp = df.Correlation[groups[i+1].values].sort_values(ascending=False).index.values
    df_new[str(i+1)] = temp[0:min_val]
df_new.to_excel('indices.xlsx')