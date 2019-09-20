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

# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)


mat = scipy.io.loadmat('H:/era5_son_new/Full_data/CI_results.mat')

clust = mat['K']

clustvals = clust[:,6]

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

for j in range(1000):
    mat = scipy.io.loadmat('H:/era5_son_new/Full_data/CI_results.mat')

    clust = mat['K']

    clustvals = clust[:,6]
    np.random.shuffle(clustvals)
    storage2 = np.zeros((3,7));
    count =0;
    if(j == 25 or j == 975):
        for i in range(3640):
            
      
            if( y <= 30):
                storage2[0,clustvals[i]-1] = storage2[0,clustvals[i]-1] + 1
            elif( y > 30 and y <= 61):
                storage2[1,clustvals[i]-1] = storage2[1,clustvals[i]-1] + 1      
            elif( y > 61 and y <= 91):
                storage2[2,clustvals[i]-1] = storage2[2,clustvals[i]-1] + 1 
            else:
             y = 0;
            
            if (y == 91):
               y = 0; 
            
            y = y + 1;
        
        if(j == 25):
            storage25 = storage2[:,:]
    
        
        if( j == 975):
            storage975 = storage2[:,:]

storagesum = sum(storage25,1)

storage25[0,:] = (storage25[0,:] / 1200) * 100
storage25[1,:] = (storage25[1,:] / 1240) * 100
storage25[2,:] = (storage25[2,:] / 1200) * 100

storagesum = sum(storage975,1)
storage975[0,:] = (storage975[0,:] / 1200) * 100
storage975[1,:] = (storage975[1,:] / 1240) * 100
storage975[2,:] = (storage975[2,:] / 1200) * 100


for k in range(7):
    for l in range(3):
        xx = storage25[l,k]
        yy = storage975[l,k]
        

        if( np.floor(xx) == np.floor(yy) or np.ceil(xx) == np.ceil(yy)):
            top[l,k] = xx + 1
            bot[l,k] = yy
        elif(np.floor(xx) == np.ceil(yy)):
            top[l,k] = xx + 1
            bot[l,k] = yy
        elif(np.floor(yy) == np.ceil(xx)):
            top[l,k] = yy+1
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
ax1.plot(('1','1'),(bot[0][0], top[0][0]),color='black',linewidth=10.0)
ax1.plot(('2','2'),(bot[0][1], top[0][1]),color='black',linewidth=10.0)
ax1.plot(('3','3'),(bot[0][2], top[0][2]),color='black',linewidth=10.0)
ax1.plot(('4','4'),(bot[0][3], top[0][3]),color='black',linewidth=10.0)
ax1.plot(('5','5'),(bot[0][4], top[0][4]),color='black',linewidth=10.0)
ax1.plot(('6','6'),(bot[0][5], top[0][5]),color='black',linewidth=10.0)
ax1.plot(('7','7'),(bot[0][6], top[0][6]),color='black',linewidth=10.0)
color1 = list(C[c1[1,:]])
ax2.bar(['1','2','3','4','5','6','7'],storage[1,:], color=color1)
ax2.set_title('October')
ax2.set_xlabel('WT')
ax2.set_ylabel('% of days')
ax2.plot(('1','1'),(bot[1][0], top[1][0]),color='black',linewidth=10.0)
ax2.plot(('2','2'),(bot[1][1], top[1][1]),color='black',linewidth=10.0)
ax2.plot(('3','3'),(bot[1][2], top[1][2]),color='black',linewidth=10.0)
ax2.plot(('4','4'),(bot[1][3], top[1][3]),color='black',linewidth=10.0)
ax2.plot(('5','5'),(bot[1][4], top[1][4]),color='black',linewidth=10.0)
ax2.plot(('6','6'),(bot[1][5], top[1][5]),color='black',linewidth=10.0)
ax2.plot(('7','7'),(bot[1][6], top[1][6]),color='black',linewidth=10.0)
color1 = list(C[c1[2,:]])
ax3.bar(['1','2','3','4','5','6','7'],storage[2,:], color=color1)
ax3.set_title('November')
ax3.set_xlabel('WT')
ax3.set_ylabel('% of days')
ax3.plot(('1','1'),(bot[2][0], top[2][0]),color='black',linewidth=10.0)
ax3.plot(('2','2'),(bot[2][1], top[2][1]),color='black',linewidth=10.0)
ax3.plot(('3','3'),(bot[2][2], top[2][2]),color='black',linewidth=10.0)
ax3.plot(('4','4'),(bot[2][3], top[2][3]),color='black',linewidth=10.0)
ax3.plot(('5','5'),(bot[2][4], top[2][4]),color='black',linewidth=10.0)
ax3.plot(('6','6'),(bot[2][5], top[2][5]),color='black',linewidth=10.0)
ax3.plot(('7','7'),(bot[2][6], top[2][6]),color='black',linewidth=10.0)
fig.suptitle("Monthly WT Occurence", fontsize=20)

plt.savefig("WT_progression_son.png", bbox_inches='tight')
