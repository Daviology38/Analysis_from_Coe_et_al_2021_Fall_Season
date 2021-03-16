# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:07:01 2019

@author: CoeFamily
"""

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
import random

# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)



df = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/era5_son_sep2020_12z2/clust_vals.csv")
df.columns = ["C","Clust"]
vals = df["Clust"].values
clustvals = list(map(int,vals))
clustvals = [x + 1 for x in clustvals]
clustvals = [2 if x == 1 else 5 if x == 2 else 7 if x == 3 else 1 if x == 4 else 4 if x == 5 else 3 if x == 6 else 6 for x in clustvals]

storage = np.zeros((3,7))
storage2 = np.zeros((3,7))
top = np.zeros((3,7))
bot = np.zeros((3,7))

y = 0
for j in range (3640):
 
    if( y <= 30):
        storage[0,clustvals[j]-1] = storage[0,clustvals[j]-1] + 1
    elif( y > 30 and y <= 61):
        storage[1,clustvals[j]-1] = storage[1,clustvals[j]-1] + 1      
    elif( y > 61 and y <= 91):
        storage[2,clustvals[j]-1] = storage[2,clustvals[j]-1] + 1 
    else:
     y = 0;
    
    if (y == 91):
       y = 0; 
    
    y = y + 1;
 

storagesum = sum(storage,1)

storage[0,:] = (storage[0,:] / 1200) * 100
storage[1,:] = (storage[1,:] / 1240) * 100
storage[2,:] = (storage[2,:] / 1200) * 100

count =1;
y = 0;
p=1;
storage2 = np.zeros((1000,3,7));
for j in range(1000):
    df = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/era5_son_sep2020_12z2/clust_vals.csv")
    df.columns = ["C","Clust"]
    vals = df["Clust"].values
    clustvals = list(map(int,vals))
    clustvals = [x + 1 for x in clustvals]
    clustvals = [2 if x == 1 else 5 if x == 2 else 7 if x == 3 else 1 if x == 4 else 4 if x == 5 else 3 if x == 6 else 6 for x in clustvals]
    random.shuffle(clustvals)

    for i in range(3640):


        if( y <= 30):
            storage2[j,0,clustvals[i]-1] = storage2[j,0,clustvals[i]-1] + 1
        elif( y > 30 and y <= 61):
            storage2[j,1,clustvals[i]-1] = storage2[j,1,clustvals[i]-1] + 1
        elif( y > 61 and y <= 91):
            storage2[j,2,clustvals[i]-1] = storage2[j,2,clustvals[i]-1] + 1
        else:
         y = 0;

        if (y == 91):
           y = 0;

        y = y + 1;
        
sep_sort = np.sort(storage2[:,0,:], axis=0)
oct_sort = np.sort(storage2[:,1,:], axis=0)
nov_sort = np.sort(storage2[:,2,:], axis=0)

storage25 = np.zeros((3,7))
storage975 = np.zeros((3,7))

storage25[0,:] = (sep_sort[49,:] / 1200) * 100
storage25[1,:] = (oct_sort[49,:] / 1240) * 100
storage25[2,:] = (nov_sort[49,:] / 1200) * 100

storage975[0,:] = (sep_sort[949,:] / 1200) * 100
storage975[1,:] = (oct_sort[949,:] / 1240) * 100
storage975[2,:] = (nov_sort[949,:] / 1200) * 100


for k in range(7):
    for l in range(3):
        xx = storage25[l,k]
        yy = storage975[l,k]
        

        if( np.floor(xx) == np.floor(yy) or np.ceil(xx) == np.ceil(yy)):
            top[l,k] = xx
            bot[l,k] = yy
        elif(np.floor(xx) == np.ceil(yy)):
            top[l,k] = xx
            bot[l,k] = yy
        elif(np.floor(yy) == np.ceil(xx)):
            top[l,k] = yy
            bot[l,k] = xx
        elif(xx < yy):
            top[l,k] = yy
            bot[l,k] = xx
        else:
            top[l,k] = xx
            bot[l,k] = yy


c1 = np.zeros((3,7));
C = pd.Series(['red','blue','gray'])

cc = 0
dd = 0

while dd < 7:
    while cc < 3:
        if(storage[cc,dd] > top[cc,dd]):
            c1[cc,dd] = 0
        elif(storage[cc,dd] < bot[cc,dd]):
            c1[cc,dd] = 1
        else:
            c1[cc,dd] = 2
        cc = cc + 1
    cc = 0
    dd = dd + 1
            

fig= plt.figure()
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,:]) 
fig.subplots_adjust(hspace=.5,wspace = .5,top=.75)
    
color1 = list(C[c1[0,:]])
ax1.bar(['1','2','3','4','5','6','7'],storage[0,:], color=color1)
ax1.set_title('September')
ax1.set_xlabel('WT')
ax1.set_ylabel('% of days')
ax1.set_ylim(0,50)
ax1.plot(('1','1'),(bot[0][0]+0.25, top[0][0]-0.25),color='black',linewidth=5.0)
ax1.plot(('2','2'),(bot[0][1]+0.25, top[0][1]-0.25),color='black',linewidth=5.0)
ax1.plot(('3','3'),(bot[0][2]+0.25, top[0][2]-0.25),color='black',linewidth=5.0)
ax1.plot(('4','4'),(bot[0][3]+0.25, top[0][3]-0.25),color='black',linewidth=5.0)
ax1.plot(('5','5'),(bot[0][4]+0.25, top[0][4]-0.25),color='black',linewidth=5.0)
ax1.plot(('6','6'),(bot[0][5]+0.25, top[0][5]-0.25),color='black',linewidth=5.0)
ax1.plot(('7','7'),(bot[0][6]+0.25, top[0][6]-0.25),color='black',linewidth=5.0)
color1 = list(C[c1[1,:]])
ax2.bar(['1','2','3','4','5','6','7'],storage[1,:], color=color1)
ax2.set_title('October')
ax2.set_xlabel('WT')
ax2.set_ylabel('% of days')
ax2.set_ylim(0,50)
ax2.plot(('1','1'),(bot[1][0]+0.25, top[1][0]-0.25),color='black',linewidth=5.0)
ax2.plot(('2','2'),(bot[1][1]+0.25, top[1][1]-0.25),color='black',linewidth=5.0)
ax2.plot(('3','3'),(bot[1][2]+0.25, top[1][2]-0.25),color='black',linewidth=5.0)
ax2.plot(('4','4'),(bot[1][3]+0.25, top[1][3]-0.25),color='black',linewidth=5.0)
ax2.plot(('5','5'),(bot[1][4]+0.25, top[1][4]-0.25),color='black',linewidth=5.0)
ax2.plot(('6','6'),(bot[1][5]+0.25, top[1][5]-0.25),color='black',linewidth=5.0)
ax2.plot(('7','7'),(bot[1][6]+0.25, top[1][6]-0.25),color='black',linewidth=5.0)
color1 = list(C[c1[2,:]])
ax3.bar(['1','2','3','4','5','6','7'],storage[2,:], color=color1)
ax3.set_title('November')
ax3.set_xlabel('WT')
ax3.set_ylabel('% of days')
ax3.set_ylim(0,50)
ax3.plot(('1','1'),(bot[2][0]+0.25, top[2][0]-0.25),color='black',linewidth=5.0)
ax3.plot(('2','2'),(bot[2][1]+0.25, top[2][1]-0.25),color='black',linewidth=5.0)
ax3.plot(('3','3'),(bot[2][2]+0.25, top[2][2]-0.25),color='black',linewidth=5.0)
ax3.plot(('4','4'),(bot[2][3]+0.25, top[2][3]-0.25),color='black',linewidth=5.0)
ax3.plot(('5','5'),(bot[2][4]+0.25, top[2][4]-0.25),color='black',linewidth=5.0)
ax3.plot(('6','6'),(bot[2][5]+0.25, top[2][5]-0.25),color='black',linewidth=5.0)
ax3.plot(('7','7'),(bot[2][6]+0.25, top[2][6]-0.25),color='black',linewidth=5.0)
fig.suptitle("Monthly WT Occurence", fontsize=20)

plt.savefig("WT_monthly_son_sep2020.png", bbox_inches='tight')
