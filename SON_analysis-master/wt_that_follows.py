# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:40:35 2019

@author: CoeFamily
"""

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Create 2x2 sub plots
gs = gridspec.GridSpec(3, 3)


mat = scipy.io.loadmat('C:/Users/CoeFamily/Documents/Research/era5_son_oct2019_run/era5_son_new/CI_results.mat')

clust = mat['K']

clustvals = clust[:,6]
array = np.zeros((7,7))

for j in range((len(clustvals)-1)):
    b = clustvals[j]-1;
    a = clustvals[j+1]-1;
    
    array[b,a] = array[b,a] + 1;

totals = np.sum(array,axis=0)    
array = array * 100 / np.sum(array,axis=0)


for i in range(1000):
    mat = scipy.io.loadmat('C:/Users/CoeFamily/Documents/Research/era5_son_oct2019_run/era5_son_new/CI_results.mat')

    clust = mat['K']

    clustvals = clust[:,6]
    np.random.shuffle(clustvals)
    array_temp = np.zeros((7,7))
    temp = np.zeros(7)
    count = 0
    if(i == 50 or i == 950):
        for value in totals:
            x = np.random.randint(0,int(len(clustvals)-1),size=(1,int(value)))
            x = x + 1
            y = clustvals[x]
            unique, counts = np.unique(y, return_counts=True)
            array_temp[count,:] = counts;
            count = count + 1
        if( i == 50):
            array25 = array_temp
        if(i == 950):
            array975 = array_temp
array25 = array25 * 100 / np.sum(array25,axis=0)
array975 = array975 * 100 / np.sum(array975,axis=0)

top = np.zeros((7,7))
bot = np.zeros((7,7))

for k in range(7):
    for l in range(7):
        xx = array25[l,k]
        yy = array975[l,k]
        
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
            
c1 = np.zeros((7,7));
C = pd.Series(['red','blue','gray'])

cc = 0
dd = 0

while dd < 7:
    while cc < 7:
        if(array[cc,dd] > top[cc,dd]):
            c1[cc,dd] = 0
        elif(array[cc,dd] < bot[cc,dd]):
            c1[cc,dd] = 1
        else:
            c1[cc,dd] = 2
        cc = cc + 1
    cc = 0
    dd = dd + 1

fig= plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2]) 
ax4 = fig.add_subplot(gs[1,0]) 
ax5 = fig.add_subplot(gs[1,1]) 
ax6 = fig.add_subplot(gs[1,2]) 
ax7 = fig.add_subplot(gs[2,1]) 
fig.subplots_adjust(hspace=1.5,wspace = .5,top=.80)
    
color1 = list(C[c1[0,:]])
ax1.bar(['1','2','3','4','5','6','7'],array[0,:], color=color1)
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
ax2.bar(['1','2','3','4','5','6','7'],array[1,:], color=color1)
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
ax3.bar(['1','2','3','4','5','6','7'],array[2,:], color=color1)
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
ax4.bar(['1','2','3','4','5','6','7'],array[3,:], color=color1)
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
ax5.bar(['1','2','3','4','5','6','7'],array[4,:], color=color1)
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
ax6.bar(['1','2','3','4','5','6','7'],array[5,:], color=color1)
ax6.set_title('WT6')
ax6.set_xlabel('WT')
ax6.set_ylabel('% of days')
ax6.set_ylim(0,50)
ax6.plot(('1','1'),(bot[5][0], top[5][0]),color='black',linewidth=4.0)
ax6.plot(('2','2'),(bot[5][1], top[5][1]),color='black',linewidth=4.0)
ax6.plot(('3','3'),(bot[5][2], top[5][2]),color='black',linewidth=4.0)
ax6.plot(('4','4'),(bot[5][3], top[5][3]),color='black',linewidth=4.0)
ax6.plot(('5','5'),(bot[5][4]-1, top[5][4]-1),color='black',linewidth=4.0)
ax6.plot(('6','6'),(bot[5][5], top[5][5]),color='black',linewidth=4.0)
ax6.plot(('7','7'),(bot[5][6], top[5][6]),color='black',linewidth=4.0)
color1 = list(C[c1[6,:]])
ax7.bar(['1','2','3','4','5','6','7'],array[6,:], color=color1)
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