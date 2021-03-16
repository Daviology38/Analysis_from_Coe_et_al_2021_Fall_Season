# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:38:20 2019

@author: CoeFamily
"""

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
import random

# Create 3X3 sub plots
gs = gridspec.GridSpec(3,3)

df = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Coe_clusters_new.csv")
vals = df["0"].values
clustvals = list(map(int,vals))
ldf = clustvals
#Make an array to fill with the data. 
#Rows signify each individual WT
#Columns signify the WT that follows after each row
array = np.zeros((7,7))

#Get the clusters by month
#Make overall array to hold the cluster number and month values
son = np.zeros((3640,2))
i = 0
son = list(map(int,ldf))
mon = []

k = 0
y = 1
while k < 3640:
    if(y<=30):
        mon.append('Sep')
    elif(y > 30 and y<=61):
        mon.append('Oct')
    else:
        mon.append('Nov')
    k = k + 1
    y = y + 1
    if(y==92):
        y = 1
        
#Make the list of months and clusters each into their own Series
son = pd.Series(son)
mon = pd.Series(mon)
#Turn the two Series into a DataFrame with columns cluster and month
dff = pd.concat([son,mon],axis=1)   
dff.columns = ['Cluster','Month']

#group the values by month
group = dff.groupby('Month')
#put the grouped values into arrays based on what month they are grouped in to
s = group.get_group('Sep')
temp1 = s.Cluster.values
temp2 = s.Month.values
s = pd.concat((pd.Series(temp1),pd.Series(temp2)),axis=1)
o = group.get_group('Oct')
temp1 = o.Cluster.values
temp2 = o.Month.values
o = pd.concat((pd.Series(temp1),pd.Series(temp2)),axis=1) 
n = group.get_group('Nov')
temp1 = n.Cluster.values
temp2 = n.Month.values
n = pd.concat((pd.Series(temp1),pd.Series(temp2)),axis=1)

#Loop over all the values in the clustvals array
#Loop over all the values in the clustvals array
arrays = np.zeros((7,7))

for j in range((len(son)-1)):
    #Take the initial value and then the next value (current day and next day)
    b = son[j]-1;
    a = son[j+1]-1;
    #Add one to the overall array 
    arrays[b,a] = arrays[b,a] + 1;

    
#Totals array gets the sum of all values of each WT for all years
totals = np.sum(arrays,axis=0)  
#Divide each row by the total number of the WT that row represents
arrays[0,:] = arrays[0,:] * 100 / totals[0]
arrays[1,:] = arrays[1,:] * 100 / totals[1]
arrays[2,:] = arrays[2,:] * 100 / totals[2]
arrays[3,:] = arrays[3,:] * 100 / totals[3]
arrays[4,:] = arrays[4,:] * 100 / totals[4]
arrays[5,:] = arrays[5,:] * 100 / totals[5]
arrays[6,:] = arrays[6,:] * 100 / totals[6]

#Make new arrays to repeat the current analysis 1000 times
arrayones = np.zeros((10000,7))
arraytwos = np.zeros((10000,7))
arraythrees = np.zeros((10000,7))
arrayfours = np.zeros((10000,7))
arrayfives = np.zeros((10000,7))
arraysixs = np.zeros((10000,7))
arraysevens = np.zeros((10000,7))

for i in range(10000):

    #Load in all the values from the matlab file
    df = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Coe_clusters_new.csv")
    vals = df["0"].values
    clustvals = list(map(int, vals))
    ldf = clustvals

    
    random.shuffle(ldf)
    son = np.zeros((3640,2))
    son = list(map(int,ldf))

    # Make the list of months and clusters each into their own Series
    son = pd.Series(son)
    mon = pd.Series(mon)
    # Turn the two Series into a DataFrame with columns cluster and month
    dff = pd.concat([son, mon], axis=1)
    dff.columns = ['Cluster', 'Month']

    # group the values by month
    group = dff.groupby('Month')
    # put the grouped values into arrays based on what month they are grouped in to
    s = group.get_group('Sep')
    temp1 = s.Cluster.values
    temp2 = s.Month.values
    s = pd.concat((pd.Series(temp1), pd.Series(temp2)), axis=1)
    o = group.get_group('Oct')
    temp1 = o.Cluster.values
    temp2 = o.Month.values
    o = pd.concat((pd.Series(temp1), pd.Series(temp2)), axis=1)
    n = group.get_group('Nov')
    temp1 = n.Cluster.values
    temp2 = n.Month.values
    n = pd.concat((pd.Series(temp1), pd.Series(temp2)), axis=1)

    # Loop over all the values in the clustvals array
    # Loop over all the values in the clustvals array
    arraynews = np.zeros((7, 7))

    for j in range((len(son) - 1)):
        # Take the initial value and then the next value (current day and next day)
        b = son[j] - 1;
        a = son[j + 1] - 1;
        # Add one to the overall array
        arraynews[b, a] = arraynews[b, a] + 1;

    # Totals array gets the sum of all values of each WT for all years
    totals = np.sum(arraynews, axis=0)
    # Divide each row by the total number of the WT that row represents
    arraynews[0, :] = arraynews[0, :] * 100 / totals[0]
    arraynews[1, :] = arraynews[1, :] * 100 / totals[1]
    arraynews[2, :] = arraynews[2, :] * 100 / totals[2]
    arraynews[3, :] = arraynews[3, :] * 100 / totals[3]
    arraynews[4, :] = arraynews[4, :] * 100 / totals[4]
    arraynews[5, :] = arraynews[5, :] * 100 / totals[5]
    arraynews[6, :] = arraynews[6, :] * 100 / totals[6]

    arrayones[i,:] = arraynews[0, :]
    arraytwos[i,:] = arraynews[1, :]
    arraythrees[i,:] = arraynews[2, :]
    arrayfours[i,:] = arraynews[3, :]
    arrayfives[i,:] = arraynews[4, :]
    arraysixs[i,:] = arraynews[5, :]
    arraysevens[i,:] = arraynews[6, :]

arrayones = np.sort(arrayones,axis=0)
arraytwos = np.sort(arraytwos,axis=0)
arraythrees = np.sort(arraythrees,axis=0)
arrayfours = np.sort(arrayfours,axis=0)
arrayfives = np.sort(arrayfives,axis=0)
arraysixs = np.sort(arraysixs,axis=0)
arraysevens = np.sort(arraysevens,axis=0)

array25s = np.zeros((7,7))
array975s = np.zeros((7,7))

array25s[0,:] = arrayones[249,:]
array25s[1,:] = arraytwos[249,:]
array25s[2,:] = arraythrees[249,:]
array25s[3,:] = arrayfours[249,:]
array25s[4,:]= arrayfives[249,:]
array25s[5,:] = arraysixs[249,:]
array25s[6,:] = arraysevens[249,:]

array975s[0,:] = arrayones[9750,:]
array975s[1,:] = arraytwos[9750,:]
array975s[2,:] = arraythrees[9750,:]
array975s[3,:] = arrayfours[9750,:]
array975s[4,:]= arrayfives[9750,:]
array975s[5,:] = arraysixs[9750,:]
array975s[6,:] = arraysevens[9750,:]

#arrays[[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]] = arrays[[2,5,3,1,6,0,4],[2,5,3,1,6,0,4]]
#array25s[[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]] = array25s[[2,5,3,1,6,0,4],[2,5,3,1,6,0,4]]
#array975s[[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]] = array975s[[2,5,3,1,6,0,4],[2,5,3,1,6,0,4]]

top = np.zeros((7,7))
bot = np.zeros((7,7))

for l in range(7):
    for k in range(7):
        xx = array25s[l,k]
        yy = array975s[l,k]
        
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
            bot[l,k] = xx
            top[l,k] = yy
        else:
            top[l,k] = xx
            bot[l,k] = yy
            
            
c1 = np.zeros((7,7))
C = pd.Series(['red','blue','gray'])
c2 = np.zeros((7,7))
c3 = np.zeros((7,7))

cc = 0
dd = 0

while cc < 7:
    while dd < 7:
        if(arrays[cc,dd] > top[cc,dd]):
            c1[cc,dd] = 0
        elif(arrays[cc,dd] < bot[cc,dd]):
            c1[cc,dd] = 1
        else:
            c1[cc,dd] = 2
        dd = dd + 1
    dd = 0
    cc = cc + 1
fig= plt.figure()
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2]) 
ax4 = fig.add_subplot(gs[1,0]) 
ax5 = fig.add_subplot(gs[1,1]) 
ax6 = fig.add_subplot(gs[1,2]) 
ax7 = fig.add_subplot(gs[2,1])             

fig.subplots_adjust(hspace=1.5,wspace = .5,top=.80)
    
color1 = list(C[c1[0,:]])
ax1.bar(['1','2','3','4','5','6','7'],arrays[0,:], color=color1)
ax1.set_title('WT1')
ax1.set_xlabel('WT')
ax1.set_ylabel('% of days')
ax1.set_ylim(0,50)
ax1.plot(('1','1'),(bot[0][0], top[0][0]),color='black',linewidth=4.0)
ax1.plot(('2','2'),(bot[0][1], top[0][1]),color='black',linewidth=4.0)
ax1.plot(('3','3'),(bot[0][2], top[0][2]),color='black',linewidth=4.0)
ax1.plot(('4','4'),(bot[0][3], top[0][3]),color='black',linewidth=4.0)
ax1.plot(('5','5'),(bot[0][4], top[0][4]),color='black',linewidth=4.0)
ax1.plot(('6','6'),(bot[0][5], top[0][5]),color='black',linewidth=4.0)
ax1.plot(('7','7'),(bot[0][6], top[0][6]),color='black',linewidth=4.0)
color1 = list(C[c1[1,:]])
ax2.bar(['1','2','3','4','5','6','7'],arrays[1,:], color=color1)
ax2.set_title('WT2')
ax2.set_xlabel('WT')
ax2.set_ylabel('% of days')
ax2.set_ylim(0,50)
ax2.plot(('1','1'),(bot[1][0], top[1][0]),color='black',linewidth=4.0)
ax2.plot(('2','2'),(bot[1][1], top[1][1]),color='black',linewidth=4.0)
ax2.plot(('3','3'),(bot[1][2], top[1][2]),color='black',linewidth=4.0)
ax2.plot(('4','4'),(bot[1][3], top[1][3]),color='black',linewidth=4.0)
ax2.plot(('5','5'),(bot[1][4], top[1][4]),color='black',linewidth=4.0)
ax2.plot(('6','6'),(bot[1][5], top[1][5]),color='black',linewidth=4.0)
ax2.plot(('7','7'),(bot[1][6], top[1][6]),color='black',linewidth=4.0)
color1 = list(C[c1[2,:]])
ax3.bar(['1','2','3','4','5','6','7'],arrays[2,:], color=color1)
ax3.set_title('WT3')
ax3.set_xlabel('WT')
ax3.set_ylabel('% of days')
ax3.set_ylim(0,50)
ax3.plot(('1','1'),(bot[2][0], top[2][0]),color='black',linewidth=4.0)
ax3.plot(('2','2'),(bot[2][1], top[2][1]),color='black',linewidth=4.0)
ax3.plot(('3','3'),(bot[2][2], top[2][2]),color='black',linewidth=4.0)
ax3.plot(('4','4'),(bot[2][3], top[2][3]),color='black',linewidth=4.0)
ax3.plot(('5','5'),(bot[2][4], top[2][4]),color='black',linewidth=4.0)
ax3.plot(('6','6'),(bot[2][5], top[2][5]),color='black',linewidth=4.0)
ax3.plot(('7','7'),(bot[2][6], top[2][6]),color='black',linewidth=4.0)
color1 = list(C[c1[3,:]])
ax4.bar(['1','2','3','4','5','6','7'],arrays[3,:], color=color1)
ax4.set_title('WT4')
ax4.set_xlabel('WT')
ax4.set_ylabel('% of days')
ax4.set_ylim(0,50)
ax4.plot(('1','1'),(bot[3][0], top[3][0]),color='black',linewidth=4.0)
ax4.plot(('2','2'),(bot[3][1], top[3][1]),color='black',linewidth=4.0)
ax4.plot(('3','3'),(bot[3][2], top[3][2]),color='black',linewidth=4.0)
ax4.plot(('4','4'),(bot[3][3], top[3][3]),color='black',linewidth=4.0)
ax4.plot(('5','5'),(bot[3][4], top[3][4]),color='black',linewidth=4.0)
ax4.plot(('6','6'),(bot[3][5], top[3][5]),color='black',linewidth=4.0)
ax4.plot(('7','7'),(bot[3][6], top[3][6]),color='black',linewidth=4.0)
color1 = list(C[c1[4,:]])
ax5.bar(['1','2','3','4','5','6','7'],arrays[4,:], color=color1)
ax5.set_title('WT5')
ax5.set_xlabel('WT')
ax5.set_ylabel('% of days')
ax5.set_ylim(0,50)
ax5.plot(('1','1'),(bot[4][0], top[4][0]),color='black',linewidth=4.0)
ax5.plot(('2','2'),(bot[4][1], top[4][1]),color='black',linewidth=4.0)
ax5.plot(('3','3'),(bot[4][2], top[4][2]),color='black',linewidth=4.0)
ax5.plot(('4','4'),(bot[4][3], top[4][3]),color='black',linewidth=4.0)
ax5.plot(('5','5'),(bot[4][4], top[4][4]),color='black',linewidth=4.0)
ax5.plot(('6','6'),(bot[4][5], top[4][5]),color='black',linewidth=4.0)
ax5.plot(('7','7'),(bot[4][6], top[4][6]),color='black',linewidth=4.0)
color1 = list(C[c1[5,:]])
ax6.bar(['1','2','3','4','5','6','7'],arrays[5,:], color=color1)
ax6.set_title('WT6')
ax6.set_xlabel('WT')
ax6.set_ylabel('% of days')
ax6.set_ylim(0,50)
ax6.plot(('1','1'),(bot[5][0], top[5][0]),color='black',linewidth=4.0)
ax6.plot(('2','2'),(bot[5][1], top[5][1]),color='black',linewidth=4.0)
ax6.plot(('3','3'),(bot[5][2], top[5][2]),color='black',linewidth=4.0)
ax6.plot(('4','4'),(bot[5][3], top[5][3]),color='black',linewidth=4.0)
ax6.plot(('5','5'),(bot[5][4], top[5][4]),color='black',linewidth=4.0)
ax6.plot(('6','6'),(bot[5][5], top[5][5]),color='black',linewidth=4.0)
ax6.plot(('7','7'),(bot[5][6], top[5][6]),color='black',linewidth=4.0)
color1 = list(C[c1[6,:]])
ax7.bar(['1','2','3','4','5','6','7'],arrays[6,:], color=color1)
ax7.set_title('WT7')
ax7.set_xlabel('WT')
ax7.set_ylabel('% of days')
ax7.set_ylim(0,50)
ax7.plot(('1','1'),(bot[6][0], top[6][0]),color='black',linewidth=4.0)
ax7.plot(('2','2'),(bot[6][1], top[6][1]),color='black',linewidth=4.0)
ax7.plot(('3','3'),(bot[6][2], top[6][2]),color='black',linewidth=4.0)
ax7.plot(('4','4'),(bot[6][3], top[6][3]),color='black',linewidth=4.0)
ax7.plot(('5','5'),(bot[6][4], top[6][4]),color='black',linewidth=4.0)
ax7.plot(('6','6'),(bot[6][5], top[6][5]),color='black',linewidth=4.0)
ax7.plot(('7','7'),(bot[6][6], top[6][6]),color='black',linewidth=4.0)
fig.suptitle("WT Progression", fontsize=20)
plt.savefig("son_progression.png", bbox_inches = 'tight')