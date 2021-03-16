#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:59:08 2019

@author: mariofire
"""

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Coe_clusters_new.csv"
)
df.columns = ["C", "Clust"]
vals = df["Clust"].values
clust = list(map(int, vals))

son = np.zeros((3640, 2))
i = 0
son = clust[:]
mon = []

k = 0
y = 1
while k < 3640:
    if y <= 30:
        mon.append("Sep")
    elif y > 30 and y <= 61:
        mon.append("Oct")
    else:
        mon.append("Nov")
    k = k + 1
    y = y + 1
    if y == 92:
        y = 1

son = pd.Series(son)
mon = pd.Series(mon)
dff = pd.concat([son, mon], axis=1)
dff.columns = ["Cluster", "Month"]

dff.to_csv("cluster_month.csv")

group = dff.groupby("Month")

s = group.get_group("Sep")
temp1 = s.Cluster.values
temp2 = s.Month.values
s = pd.concat((pd.Series(temp1), pd.Series(temp2)), axis=1)
o = group.get_group("Oct")
temp1 = o.Cluster.values
temp2 = o.Month.values
o = pd.concat((pd.Series(temp1), pd.Series(temp2)), axis=1)
n = group.get_group("Nov")
temp1 = n.Cluster.values
temp2 = n.Month.values
n = pd.concat((pd.Series(temp1), pd.Series(temp2)), axis=1)

y = 1
p = 0
count = 0
count1 = 0
count2 = 0
september = np.zeros((40, 7))
october = np.zeros((40, 7))
november = np.zeros((40, 7))

while i < 3640:
    x = son[i]
    if y <= 30:
        september[p, x - 1] = september[p, x - 1] + 1
        count = count + 1
    elif y > 30 and y <= 61:
        october[p, x - 1] = october[p, x - 1] + 1
        count1 = count1 + 1
    else:
        november[p, x - 1] = november[p, x - 1] + 1
        count2 = count2 + 1
    y = y + 1
    i = i + 1
    if y == 92:
        y = 1
        p = p + 1

ind = []

y = "1979"
p = 0
q = 0

while q < 120:
    ind.append(y)
    p = p + 1
    q = q + 1
    y = int(y)
    y = y + 1
    y = str(y)
    if p == 40:
        y = "1979"
        p = 0

month2 = [
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
]

months = [
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Sep",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Oct",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
    "Nov",
]
overall = np.concatenate([september, october, november])
df = pd.DataFrame(
    data=overall, columns=["WT1", "WT2", "WT3", "WT4", "WT5", "WT6", "WT7"]
)
df["EarlySeason"] = df["WT1"] + df["WT6"] + df["WT2"]
df["LateSeason"] = df["WT3"] + df["WT4"] + df["WT7"] + df["WT5"]
df["Month"] = months
df["Year"] = ind
df["Year"] = df["Year"].astype(int)

df.to_csv("dates_months.csv")

pre = df[df.Year <= 1998].groupby("Month").sum()
post = df[df.Year > 1998].groupby("Month").sum()

es_s_pre = pre.loc["Sep"].EarlySeason
es_o_pre = pre.loc["Oct"].EarlySeason
es_n_pre = pre.loc["Nov"].EarlySeason

es_s_post = post.loc["Sep"].EarlySeason
es_o_post = post.loc["Oct"].EarlySeason
es_n_post = post.loc["Nov"].EarlySeason

ls_s_pre = pre.loc["Sep"].LateSeason
ls_o_pre = pre.loc["Oct"].LateSeason
ls_n_pre = pre.loc["Nov"].LateSeason

ls_s_post = post.loc["Sep"].LateSeason
ls_o_post = post.loc["Oct"].LateSeason
ls_n_post = post.loc["Nov"].LateSeason

wt5_s_pre = pre.loc["Sep"].WT5
wt5_o_pre = pre.loc["Oct"].WT5
wt5_n_pre = pre.loc["Nov"].WT5

wt5_s_post = post.loc["Sep"].WT5
wt5_o_post = post.loc["Oct"].WT5
wt5_n_post = post.loc["Nov"].WT5

wt2_s_pre = pre.loc["Sep"].WT2
wt2_o_pre = pre.loc["Oct"].WT2
wt2_n_pre = pre.loc["Nov"].WT2

wt2_s_post = post.loc["Sep"].WT2
wt2_o_post = post.loc["Oct"].WT2
wt2_n_post = post.loc["Nov"].WT2

wt1_s_pre = pre.loc["Sep"].WT1
wt1_o_pre = pre.loc["Oct"].WT1
wt1_n_pre = pre.loc["Nov"].WT1

wt1_s_post = post.loc["Sep"].WT1
wt1_o_post = post.loc["Oct"].WT1
wt1_n_post = post.loc["Nov"].WT1

wt3_s_pre = pre.loc["Sep"].WT3
wt3_o_pre = pre.loc["Oct"].WT3
wt3_n_pre = pre.loc["Nov"].WT3

wt3_s_post = post.loc["Sep"].WT3
wt3_o_post = post.loc["Oct"].WT3
wt3_n_post = post.loc["Nov"].WT3

wt4_s_pre = pre.loc["Sep"].WT4
wt4_o_pre = pre.loc["Oct"].WT4
wt4_n_pre = pre.loc["Nov"].WT4

wt4_s_post = post.loc["Sep"].WT4
wt4_o_post = post.loc["Oct"].WT4
wt4_n_post = post.loc["Nov"].WT4

wt6_s_pre = pre.loc["Sep"].WT6
wt6_o_pre = pre.loc["Oct"].WT6
wt6_n_pre = pre.loc["Nov"].WT6

wt6_s_post = post.loc["Sep"].WT6
wt6_o_post = post.loc["Oct"].WT6
wt6_n_post = post.loc["Nov"].WT6

wt7_s_pre = pre.loc["Sep"].WT7
wt7_o_pre = pre.loc["Oct"].WT7
wt7_n_pre = pre.loc["Nov"].WT7

wt7_s_post = post.loc["Sep"].WT7
wt7_o_post = post.loc["Oct"].WT7
wt7_n_post = post.loc["Nov"].WT7

# Take the values for each month and divide by the total number of days of that month throughout the time period. Then multiply by 100 for a percentage
pre_s = pre.loc["Sep"] * 100 / 600
pre_o = pre.loc["Oct"] * 100 / 620
pre_n = pre.loc["Nov"] * 100 / 600

post_s = post.loc["Sep"] * 100 / 600
post_o = post.loc["Oct"] * 100 / 620
post_n = post.loc["Nov"] * 100 / 600

# Put them back into new arrays
pre = pd.concat((pre_s, pre_o, pre_n), axis=1).transpose()
post = pd.concat((post_s, post_o, post_n), axis=1).transpose()

# Pull out the individual arrays we need
pre_es = pre.EarlySeason
pre_ls = pre.LateSeason
pre_wt5 = pre.WT5
pre_wt2 = pre.WT2
pre_wt1 = pre.WT1
pre_wt3 = pre.WT3
pre_wt4 = pre.WT4
pre_wt6 = pre.WT6
pre_wt7 = pre.WT7

post_es = post.EarlySeason
post_ls = post.LateSeason
post_wt5 = post.WT5
post_wt2 = post.WT2
post_wt1 = post.WT1
post_wt3 = post.WT3
post_wt4 = post.WT4
post_wt6 = post.WT6
post_wt7 = post.WT7

# Put together the first 20 and last 20 years into one dataframe

total_es = pd.concat([pre_es, post_es], axis=1)
total_ls = pd.concat([pre_ls, post_ls], axis=1)
total_wt5 = pd.concat([pre_wt5, post_wt5], axis=1)
total_wt2 = pd.concat([pre_wt2, post_wt2], axis=1)
total_wt1 = pd.concat([pre_wt1, post_wt1], axis=1)
total_wt3 = pd.concat([pre_wt3, post_wt3], axis=1)
total_wt4 = pd.concat([pre_wt4, post_wt4], axis=1)
total_wt6 = pd.concat([pre_wt6, post_wt6], axis=1)
total_wt7 = pd.concat([pre_wt7, post_wt7], axis=1)

# Name the columns appropriately
total_es.columns = ["1979-1998", "1999-2018"]
total_ls.columns = ["1979-1998", "1999-2018"]
total_wt5.columns = ["1979-1998", "1999-2018"]
total_wt2.columns = ["1979-1998", "1999-2018"]
total_wt1.columns = ["1979-1998", "1999-2018"]
total_wt3.columns = ["1979-1998", "1999-2018"]
total_wt4.columns = ["1979-1998", "1999-2018"]
total_wt6.columns = ["1979-1998", "1999-2018"]
total_wt7.columns = ["1979-1998", "1999-2018"]

# j = 0
# september = np.zeros((1000,7))
# october = np.zeros((1000,7))
# november = np.zeros((1000,7))
# september1 = np.zeros((1000,7))
# october1 = np.zeros((1000,7))
# november1 = np.zeros((1000,7))
# while j <1000:
#
#     news = s
#     newo = o
#     newn = n
#
#     p = 0
#     pp = 0
#     ppp = 0
#     count = 0
#     count1 = 0
#     count2 = 0
#
#     indices = np.zeros(3640)
#     for i in range(3640):
#         indices[i] = i
#
#
#     #randomly select half the data from each month
#     #If using newx.drop then assigns the unselected data to x
#     ys = np.random.choice(indices,1820,replace=False)
#     xs = np.delete(indices,ys)
#     #xs = indices.sample(frac=0.5)
#     #yo = indices.sample(frac=0.5)
#     #xo = indices.drop(yo.index)
#     #xo = .sample(frac=0.5)
#     #xn = newn.sample(frac=0.5)
#     #Only run if the 25th of 975th run
#
#     for value in ys:
#         value = int(value)
#         clust = dff.iloc[value][0]
#         month = dff.iloc[value][1]
#
#         if(month == 'Sep'):
#             september[j,clust-1] = september[j,clust-1] + 1
#         elif(month =='Oct'):
#             october[j,clust-1] = october[j,clust-1]+1
#         else:
#             november[j,clust-1] = november[j,clust-1]+1
#
#     for value in xs:
#         value = int(value)
#         clust = dff.iloc[value][0]
#         month = dff.iloc[value][1]
#
#         if(month == 'Sep'):
#             september1[j,clust-1] = september1[j,clust-1] + 1
#         elif(month =='Oct'):
#             october1[j,clust-1] = october1[j,clust-1]+1
#         else:
#             november1[j,clust-1] = november1[j,clust-1]+1
#
#
#     j = j + 1
#
# ##Sum the data arrays over the second dimension(20 year summation)
# september = september * 100 / 600
# september1 = september1 * 100 / 600
# october = october * 100 / 620
# october1 = october1 * 100 / 620
# november = november * 100 / 600
# november1 = november1 * 100 / 600
#
# #Get the difference by subtracting the latter 20 years - first 20 years
# septot = september1 - september
# octtot = october1 - october
# novtot = november1 - november
#
# #Find the early and late season values
# sepe = septot[:,0] + septot[:,1] + septot[:,5]
# sepl = septot[:,2] + septot[:,3] + septot[:,4] + septot[:,6]
# octe = octtot[:,0] + octtot[:,1] + octtot[:,5]
# octl = octtot[:,2] + octtot[:,3] + octtot[:,4] + octtot[:,6]
# nove = novtot[:,0] + novtot[:,1] + novtot[:,5]
# novl = novtot[:,2] + novtot[:,3] + novtot[:,4] +novtot[:,6]
# temp = ((september1*600/100)+(october1*620/100)+(november1*600/100) - ((september*600/100)+(october*620/100)+(november*600/100)) )*100 / 1820
# tote = temp[:,0] + temp[:,1] + temp[:,5]
# totl = temp[:,2] + temp[:,3] + temp[:,4] + temp[:,6]
#
#
# #Sort the arrays
# sepe = np.sort(sepe,axis=0)
# octe = np.sort(octe,axis=0)
# nove = np.sort(nove,axis=0)
# tote = np.sort(tote,axis=0)
#
# sepl = np.sort(sepl,axis=0)
# octl = np.sort(octl,axis=0)
# novl = np.sort(novl,axis=0)
# totl = np.sort(totl,axis=0)
#
# #get the 25th and 975th values
# sepe25 = sepe[49]
# sepe975 = sepe[949]
# octe25 = octe[49]
# octe975 = octe[949]
# nove25 = nove[49]
# nove975 = nove[949]
# tote25 = tote[49]
# tote975 = tote[949]
#
# #get the 25th and 975th values
# sepl25 = sepl[49]
# sepl975 = sepl[949]
# octl25 = octl[49]
# octl975 = octl[949]
# novl25 = novl[49]
# novl975 = novl[949]
# totl25 = totl[49]
# totl975 = totl[949]
#
# #
# fig, axes = plt.subplots(nrows=1, ncols=2)
# fig.subplots_adjust(hspace=.5,wspace=.5,top=.75)
# #
# error1 = np.zeros((2,3))
# error2 = np.zeros((2,3))
# error3 = np.zeros((2,3))
# error4 = np.zeros((2,3))
# error5 = np.zeros((2,3))
# error6 = np.zeros((2,3))
# error7 = np.zeros((2,3))
# error8 = np.zeros((2,3))
# error9 = np.zeros((2,3))
# error10 = np.zeros((2,3))
# error11 = np.zeros((2,3))
# #
# error1[0][0] = sepe25
# error1[0][1] = octe25
# error1[0][2] = nove25
# error1[1][0] = sepe975
# error1[1][1] = octe975
# error1[1][2] = nove975
#
# error2[0][0] = sepl25
# error2[0][1] = octl25
# error2[0][2] = novl25
# error2[1][0] = sepl975
# error2[1][1] = octl975
# error2[1][2] = novl975
#
# error3[0][0] = 0
# error3[0][1] = 0
# error3[0][2] = 0
# error3[1][0] = 0
# error3[1][1] = 0
# error3[1][2] = 0
#
# error4[0][0] = 0
# error4[0][1] =0
# error4[0][2] = 0
# error4[1][0] = 0
# error4[1][1] = 0
# error4[1][2] = 0
#
# error5[0][0] = 0
# error5[0][1] = 0
# error5[0][2] = 0
# error5[1][0] = 0
# error5[1][1] =0
# error5[1][2] = 0
#
# error6[0][0] = 0
# error6[0][1] =0
# error6[0][2] = 0
# error6[1][0] = 0
# error6[1][1] = 0
# error6[1][2] =0
#
# error7[0][0] = 0
# error7[0][1] = 0
# error7[0][2] = 0
# error7[1][0] = 0
# error7[1][1] = 0
# error7[1][2] = 0
#
# error8[0][0] = 0
# error8[0][1] = 0
# error8[0][2] =0
# error8[1][0] = 0
# error8[1][1] = 0
# error8[1][2] = 0
#
# error9[0][0] = 0
# error9[0][1] = 0
# error9[0][2] = 0
# error9[1][0] = 0
# error9[1][1] = 0
# error9[1][2] = 0
#
# error10[0][0] = tote25
# error10[1][0] = tote975
#
# error11[0][0] = totl25
# error11[1][0] =totl975
#
# color0 = []
# color1 = []
# color2 = []
# color3 = []
# color4 = []
# color5 = []
# color6 = []
# color7 = []
# color8 = []
# color9 = []
# color10 = []
#
# essep = (es_s_post - es_s_pre) * 100 / 600
# esoct = (es_o_post - es_o_pre) * 100 / 620
# esnov = (es_n_post - es_n_pre) * 100 / 600
# lssep = (ls_s_post - ls_s_pre) * 100 / 600
# lsoct = (ls_o_post - ls_o_pre) * 100 / 620
# lsnov = (ls_n_post - ls_n_pre) * 100 / 600
# wt5sep = (wt5_s_post - wt5_s_pre) * 100 / 600
# wt5oct = (wt5_o_post - wt5_o_pre) * 100 / 620
# wt5nov = (wt5_n_post - wt5_n_pre) * 100 / 600
# wt2sep = (wt2_s_post - wt2_s_pre) * 100 / 600
# wt2oct = (wt2_o_post - wt2_o_pre) * 100 / 620
# wt2nov = (wt2_n_post - wt2_n_pre) * 100 / 600
# wt1sep = (wt1_s_post - wt1_s_pre) * 100 / 600
# wt1oct = (wt1_o_post - wt1_o_pre) * 100 / 620
# wt1nov = (wt1_n_post - wt1_n_pre) * 100 / 600
# wt3sep = (wt3_s_post - wt3_s_pre) * 100 / 600
# wt3oct = (wt3_o_post - wt3_o_pre) * 100 / 620
# wt3nov = (wt3_n_post - wt3_n_pre) * 100 / 600
# wt4sep = (wt4_s_post - wt4_s_pre) * 100 / 600
# wt4oct = (wt4_o_post - wt4_o_pre) * 100 / 620
# wt4nov = (wt4_n_post - wt4_n_pre) * 100 / 600
# wt6sep = (wt6_s_post - wt6_s_pre) * 100 / 600
# wt6oct = (wt6_o_post - wt6_o_pre) * 100 / 620
# wt6nov = (wt6_n_post - wt6_n_pre) * 100 / 600
# wt7sep = (wt7_s_post - wt7_s_pre) * 100 / 600
# wt7oct = (wt7_o_post - wt7_o_pre) * 100 / 620
# wt7nov = (wt7_n_post - wt7_n_pre) * 100 / 600
# estot = (es_s_post+es_o_post+es_n_post)*100/1820 - (es_s_pre+es_o_pre +es_n_pre)*100/1820
# lstot = (ls_s_post+ls_o_post+ls_n_post)*100/1820 - (ls_s_pre+ls_o_pre+ls_n_pre)*100/1820
#
#
# #
# i = 0
# #
# #
# if(essep < error1[0][0]):
#     color0.append('Blue')
# elif(essep > error1[1][0]):
#     color0.append('Red')
# else:
#     color0.append('Gray')
# if(esoct < error1[0][1]):
#     color0.append('Blue')
# elif(esoct > error1[1][1]):
#     color0.append('Red')
# else:
#     color0.append('Gray')
# if(esnov < error1[0][2]):
#     color0.append('Blue')
# elif(esnov > error1[1][2]):
#     color0.append('Red')
# else:
#     color0.append('Gray')
# if(estot < error10[0][0]):
#     color0.append('Blue')
# elif(estot > error10[1][0]):
#     color0.append('Red')
# else:
#     color0.append('Gray')
# if(lssep < error2[0][0]):
#     color1.append('Blue')
# elif(lssep > error2[1][0]):
#     color1.append('Red')
# else:
#     color1.append('Gray')
# if(lsoct < error2[0][1]):
#     color1.append('Blue')
# elif(lsoct > error2[1][1]):
#     color1.append('Red')
# else:
#     color1.append('Gray')
# if(lsnov < error2[0][2]):
#     color1.append('Blue')
# elif(lsnov > error2[1][2]):
#     color1.append('Red')
# else:
#     color1.append('Gray')
# if(lstot < error11[0][0]):
#     color1.append('Blue')
# elif(lstot > error11[1][0]):
#     color1.append('Red')
# else:
#     color1.append('Gray')
# if(wt5sep < error3[0][0]):
#     color2.append('Blue')
# elif(wt5sep > error3[1][0]):
#     color2.append('Red')
# else:
#     color2.append('Gray')
# if(wt5oct < error3[0][1]):
#     color2.append('Blue')
# elif(wt5oct > error3[1][1]):
#     color2.append('Red')
# else:
#     color2.append('Gray')
# if(wt5nov < error3[0][2]):
#     color2.append('Blue')
# elif(wt5nov > error3[1][2]):
#     color2.append('Red')
# else:
#     color2.append('Gray')
# if(wt2sep < error4[0][0]):
#     color3.append('Blue')
# elif(wt2sep > error4[1][0]):
#     color3.append('Red')
# else:
#     color3.append('Gray')
# if(wt2oct < error4[0][1]):
#     color3.append('Blue')
# elif(wt2oct > error4[1][1]):
#     color3.append('Red')
# else:
#     color3.append('Gray')
# if(wt2nov < error4[0][2]):
#     color3.append('Blue')
# elif(wt2nov > error4[1][2]):
#     color3.append('Red')
# else:
#     color3.append('Gray')
#
#
# if(wt1sep < error5[0][0]):
#     color4.append('Blue')
# elif(wt1sep > error5[1][0]):
#     color4.append('Red')
# else:
#     color4.append('Gray')
# if(wt1oct < error5[0][1]):
#     color4.append('Blue')
# elif(wt1oct > error5[1][1]):
#     color4.append('Red')
# else:
#     color4.append('Gray')
# if(wt1nov < error5[0][2]):
#     color4.append('Blue')
# elif(wt1nov > error5[1][2]):
#     color4.append('Red')
# else:
#     color4.append('Gray')
#
# if(wt3sep < error6[0][0]):
#     color5.append('Blue')
# elif(wt3sep > error6[1][0]):
#     color5.append('Red')
# else:
#     color5.append('Gray')
# if(wt3oct < error6[0][1]):
#     color5.append('Blue')
# elif(wt3oct > error6[1][1]):
#     color5.append('Red')
# else:
#     color5.append('Gray')
# if(wt3nov < error6[0][2]):
#     color5.append('Blue')
# elif(wt3nov > error6[1][2]):
#     color5.append('Red')
# else:
#     color5.append('Gray')
#
# if(wt4sep < error7[0][0]):
#     color6.append('Blue')
# elif(wt4sep > error7[1][0]):
#     color6.append('Red')
# else:
#     color6.append('Gray')
# if(wt4oct < error7[0][1]):
#     color6.append('Blue')
# elif(wt4oct > error7[1][1]):
#     color6.append('Red')
# else:
#     color6.append('Gray')
# if(wt4nov < error7[0][2]):
#     color6.append('Blue')
# elif(wt4nov > error7[1][2]):
#     color6.append('Red')
# else:
#     color6.append('Gray')
#
# if(wt6sep < error8[0][0]):
#     color7.append('Blue')
# elif(wt6sep > error8[1][0]):
#     color7.append('Red')
# else:
#     color7.append('Gray')
# if(wt6oct < error8[0][1]):
#     color7.append('Blue')
# elif(wt6oct > error8[1][1]):
#     color7.append('Red')
# else:
#     color7.append('Gray')
# if(wt6nov < error8[0][2]):
#     color7.append('Blue')
# elif(wt6nov > error8[1][2]):
#     color7.append('Red')
# else:
#     color7.append('Gray')
#
# if(wt7sep < error9[0][0]):
#     color8.append('Blue')
# elif(wt7sep > error9[1][0]):
#     color8.append('Red')
# else:
#     color8.append('Gray')
# if(wt7oct < error9[0][1]):
#     color8.append('Blue')
# elif(wt7oct > error9[1][1]):
#     color8.append('Red')
# else:
#     color8.append('Gray')
# if(wt7nov < error9[0][2]):
#     color8.append('Blue')
# elif(wt7nov > error9[1][2]):
#     color8.append('Red')
# else:
#     color8.append('Gray')
#
# if(estot > error10[1][0]):
#     color9.append('Red')
# elif(estot < error10[0][0]):
#     color9.append('Blue')
# else:
#     color9.append('Gray')
#
# if(lstot > error11[1][0]):
#     color10.append('Red')
# elif(lstot < error11[0][0]):
#     color10.append('Blue')
# else:
#     color10.append('Gray')
# #
# #
# ##m = np.zeros(8)
# ##
# ##
# ##m[0] = np.min(es_tot.values)
# ##m[1] = np.min(ls_tot.values)
# ##m[2] = np.min(wt5_tot.values)
# ##m[3] = np.min(wt2_tot.values)
# ##m[4] = np.min(error1)
# ##m[5] = np.min(error2)
# ##m[6] = np.min(error3)
# ##m[7] = np.min(error4)
# ##
# ##om = np.min(m)
# ##
# ##es_tot = es_tot - om
# ##ls_tot = ls_tot - om
# ##wt5_tot = wt5_tot - om
# ##wt2_tot = wt2_tot - om
# ##error1 = error1 - om
# ##error2 = error2 - om
# ##error3 = error3 - om
# ##error4 = error4 - om
# #
#
# total_e = estot
# total_l = lstot
# data = {'Sep':essep,'Oct': esoct,'Nov':esnov,'Total':total_e}
# data2 = {'Sep':lssep,'Oct': lsoct,'Nov':lsnov,'Total':total_l}
# data3 = {'Sep':wt5sep,'Oct': wt5oct,'Nov':wt5nov}
# data4 = {'Sep':wt2sep,'Oct': wt2oct,'Nov':wt2nov}
# data5 = {'Sep':wt1sep,'Oct': wt1oct,'Nov':wt1nov}
# data6 = {'Sep':wt3sep,'Oct': wt3oct,'Nov':wt3nov}
# data7 = {'Sep':wt4sep,'Oct': wt4oct,'Nov':wt4nov}
# data8 = {'Sep':wt6sep,'Oct': wt6oct,'Nov':wt6nov}
# data9 = {'Sep':wt7sep,'Oct': wt7oct,'Nov':wt7nov}
#
#
# es_tot = pd.Series(data).to_frame()
# ls_tot = pd.Series(data2).to_frame()
# wt5_tot = pd.Series(data3).to_frame()
# wt2_tot = pd.Series(data4).to_frame()
# wt1_tot = pd.Series(data5).to_frame()
# wt3_tot = pd.Series(data6).to_frame()
# wt4_tot = pd.Series(data7).to_frame()
# wt6_tot = pd.Series(data8).to_frame()
# wt7_tot = pd.Series(data9).to_frame()
#
#
# es_tot.plot(kind='bar',ax=axes[0],ylim=[-10,10],legend=False,rot=0,title='Early Season',color = [color0])
# axes[0].plot(('Sep','Sep'),(error1[0][0], error1[1][0]),color='gray',linewidth=7.0)
# axes[0].plot(('Oct','Oct'),(error1[0][1], error1[1][1]),color='gray',linewidth=7.0)
# axes[0].plot(('Nov','Nov'),(error1[0][2], error1[1][2]),color='gray',linewidth=7.0)
# axes[0].plot(('Total','Total'),(error10[0][0], error10[1][0]-.2), color = 'gray', linewidth = 7.0)
# axes[0].set_ylabel('% Difference')
# ls_tot.plot(kind='bar',ax=axes[1],ylim = [-10,10],legend=False,rot=0,title='Late Season',color=[color1])
# axes[1].plot(('Sep','Sep'),(error2[0][0], error2[1][0]),color='gray',linewidth=7.0)
# axes[1].plot(('Oct','Oct'),(error2[0][1], error2[1][1]),color='gray',linewidth=7.0)
# axes[1].plot(('Nov','Nov'),(error2[0][2], error2[1][2]),color='gray',linewidth=7.0)
# axes[1].plot(('Total','Total'),(error11[0][0]+.2, error11[1][0]),color='gray',linewidth=7.0)
# axes[1].set_ylabel('% Difference')
# #wt5_tot.plot(kind='bar',ax=axes[1,0],ylim = [-10,10],legend=False,rot=0,title='WT5',color=[color2])
# #axes[1,0].plot(('Sep','Sep'),(error3[0][0], error3[1][0]),color='black',linewidth=8.0)
# #axes[1,0].plot(('Oct','Oct'),(error3[0][1], error3[1][1]),color='black',linewidth=8.0)
# #axes[1,0].plot(('Nov','Nov'),(error3[0][2], error3[1][2]),color='black',linewidth=8.0)
# #axes[1,0].set_ylabel('% change')
# #wt2_tot.plot(kind='bar',ax=axes[1,1],ylim = [-10,10],legend=False,rot=0,title='WT2',color=[color3])
# #axes[1,1].plot(('Sep','Sep'),(error4[0][0], error4[1][0]),color='black',linewidth=8.0)
# #axes[1,1].plot(('Oct','Oct'),(error4[0][1], error4[1][1]),color='black',linewidth=8.0)
# #axes[1,1].plot(('Nov','Nov'),(error4[0][2], error4[1][2]),color='black',linewidth=8.0)
# #axes[1,1].set_ylabel('% change')
#
# fig.suptitle('Difference in Monthly Occurrence from 1979-1998 to 1999-2018', fontsize=20)
#
# plt.savefig('monte_carlo_change_per_time_of_season.png',bbox_inches='tight')

# fig, axes = plt.subplots(nrows=3, ncols=3)
# fig.subplots_adjust(hspace=.5,wspace=.5,top=.75)
# wt1_tot.plot(kind='bar',ax=axes[0,0],ylim=[-10,10],legend=False,rot=0,title='WT1',color = [color4])
# axes[0,0].plot(('Sep','Sep'),(error5[0][0], error5[1][0]),color='black',linewidth=8.0)
# axes[0,0].plot(('Oct','Oct'),(error5[0][1], error5[1][1]),color='black',linewidth=8.0)
# axes[0,0].plot(('Nov','Nov'),(error5[0][2], error5[1][2]),color='black',linewidth=8.0)
# axes[0,0].set_ylabel('% change')
# wt2_tot.plot(kind='bar',ax=axes[0,1],ylim = [-10,10],legend=False,rot=0,title='WT2',color=[color3])
# axes[0,1].plot(('Sep','Sep'),(error4[0][0], error4[1][0]),color='black',linewidth=8.0)
# axes[0,1].plot(('Oct','Oct'),(error4[0][1], error4[1][1]),color='black',linewidth=8.0)
# axes[0,1].plot(('Nov','Nov'),(error4[0][2], error4[1][2]),color='black',linewidth=8.0)
# axes[0,1].set_ylabel('% change')
# wt3_tot.plot(kind='bar',ax=axes[0,2],ylim = [-10,10],legend=False,rot=0,title='WT3',color=[color5])
# axes[0,2].plot(('Sep','Sep'),(error6[0][0], error6[1][0]),color='black',linewidth=8.0)
# axes[0,2].plot(('Oct','Oct'),(error6[0][1], error6[1][1]),color='black',linewidth=8.0)
# axes[0,2].plot(('Nov','Nov'),(error6[0][2], error6[1][2]),color='black',linewidth=8.0)
# axes[0,2].set_ylabel('% change')
# wt4_tot.plot(kind='bar',ax=axes[1,0],ylim = [-10,10],legend=False,rot=0,title='WT4',color=[color6])
# axes[1,0].plot(('Sep','Sep'),(error7[0][0], error7[1][0]),color='black',linewidth=8.0)
# axes[1,0].plot(('Oct','Oct'),(error7[0][1], error7[1][1]),color='black',linewidth=8.0)
# axes[1,0].plot(('Nov','Nov'),(error7[0][2], error7[1][2]),color='black',linewidth=8.0)
# axes[1,0].set_ylabel('% change')
# wt5_tot.plot(kind='bar',ax=axes[1,1],ylim=[-10,10],legend=False,rot=0,title='WT5',color = [color2])
# axes[1,1].plot(('Sep','Sep'),(error3[0][0], error3[1][0]),color='black',linewidth=8.0)
# axes[1,1].plot(('Oct','Oct'),(error3[0][1], error3[1][1]),color='black',linewidth=8.0)
# axes[1,1].plot(('Nov','Nov'),(error3[0][2], error3[1][2]),color='black',linewidth=8.0)
# axes[1,1].set_ylabel('% change')
# wt6_tot.plot(kind='bar',ax=axes[1,2],ylim = [-10,10],legend=False,rot=0,title='WT6',color=[color7])
# axes[1,2].plot(('Sep','Sep'),(error8[0][0], error8[1][0]),color='black',linewidth=8.0)
# axes[1,2].plot(('Oct','Oct'),(error8[0][1], error8[1][1]),color='black',linewidth=8.0)
# axes[1,2].plot(('Nov','Nov'),(error8[0][2], error8[1][2]),color='black',linewidth=8.0)
# axes[1,2].set_ylabel('% change')
# wt7_tot.plot(kind='bar',ax=axes[2,1],ylim = [-10,10],legend=False,rot=0,title='WT7',color=[color8])
# axes[2,1].plot(('Sep','Sep'),(error9[0][0], error9[1][0]),color='black',linewidth=8.0)
# axes[2,1].plot(('Oct','Oct'),(error9[0][1], error9[1][1]),color='black',linewidth=8.0)
# axes[2,1].plot(('Nov','Nov'),(error9[0][2], error9[1][2]),color='black',linewidth=8.0)
# axes[2,1].set_ylabel('% change')
#
# axes[2,0].axis('off')
# axes[2,2].axis('off')
#
# fig.suptitle('Change in Monthly Occurence from 1979-1998 to 1999-2018', fontsize=20)
#
# plt.savefig('monte_carlo_change_per_wt.png',bbox_inches='tight')

df_early = df.groupby("Year").sum().EarlySeason
df_late = df.groupby("Year").sum().LateSeason


sep_month = df.groupby("Month").groups["Sep"]
oct_month = df.groupby("Month").groups["Oct"]
nov_month = df.groupby("Month").groups["Nov"]

indx = [
    "1979",
    "1980",
    "1981",
    "1982",
    "1983",
    "1984",
    "1985",
    "1986",
    "1987",
    "1988",
    "1989",
    "1990",
    "1991",
    "1992",
    "1993",
    "1994",
    "1995",
    "1996",
    "1997",
    "1998",
    "1999",
    "2000",
    "2001",
    "2002",
    "2003",
    "2004",
    "2005",
    "2006",
    "2007",
    "2008",
    "2009",
    "2010",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
]

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.subplots_adjust(hspace=0.75, wspace=0.75, top=0.8)

essep = df.EarlySeason[sep_month].rolling(5, min_periods=1).mean()
essep.index = indx

esoct = df.EarlySeason[oct_month].rolling(5, min_periods=1).mean()
esoct.index = indx

esnov = df.EarlySeason[nov_month].rolling(5, min_periods=1).mean()
esnov.index = indx

lssep = df.LateSeason[sep_month].rolling(5, min_periods=1).mean()
lssep.index = indx

lsoct = df.LateSeason[oct_month].rolling(5, min_periods=1).mean()
lsoct.index = indx

lsnov = df.LateSeason[nov_month].rolling(5, min_periods=1).mean()
lsnov.index = indx

es_overall = df_early.rolling(5, min_periods=1).mean()
es_overall.index = indx

ls_overall = df_late.rolling(5, min_periods=1).mean()
ls_overall.index = indx
essep.plot(
    kind="line", ax=axes[0, 0], ylim=[0, 40], legend=False, rot=90, color="black"
)
lssep.plot(kind="line", ax=axes[0, 0], ylim=[0, 40], legend=False, rot=90, color="red")
esoct.plot(
    kind="line", ax=axes[0, 1], ylim=[0, 40], legend=False, rot=90, color="black"
)
lsoct.plot(kind="line", ax=axes[0, 1], ylim=[0, 40], legend=False, rot=90, color="red")
esnov.plot(
    kind="line", ax=axes[1, 0], ylim=[0, 40], legend=False, rot=90, color="black"
)
lsnov.plot(kind="line", ax=axes[1, 0], ylim=[0, 40], legend=False, rot=90, color="red")
es_overall.plot(
    kind="line", ax=axes[1, 1], ylim=[0, 95], legend=False, rot=90, color="black"
)
ls_overall.plot(
    kind="line", ax=axes[1, 1], ylim=[0, 95], legend=False, rot=90, color="red"
)
axes[0, 0].set_ylabel("# of Days")
axes[0, 1].set_ylabel("# of Days")
axes[1, 0].set_ylabel("# of Days")
axes[1, 1].set_ylabel("# of Days")
axes[0, 0].set_title("September")
axes[0, 1].set_title("October")
axes[1, 0].set_title("November")
axes[1, 1].set_title("Overall")
box = axes[0, 0].get_position()
axes[0, 0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
box = axes[0, 1].get_position()
axes[0, 1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
box = axes[1, 0].get_position()
axes[1, 0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
box = axes[1, 1].get_position()
axes[1, 1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
axes[0, 0].legend(
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    ncol=1,
    fancybox=True,
    shadow=True,
    prop={"size": 6},
)
axes[0, 1].legend(
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    ncol=1,
    fancybox=True,
    shadow=True,
    prop={"size": 6},
)
axes[1, 0].legend(
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    ncol=1,
    fancybox=True,
    shadow=True,
    prop={"size": 6},
)
axes[1, 1].legend(
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    ncol=1,
    fancybox=True,
    shadow=True,
    prop={"size": 6},
)
# axes[0,0].legend(loc='best')
# axes[0,1].legend(loc='best')
# axes[1,0].legend(loc='best')
# axes[1,1].legend(loc='best')
plt.suptitle("Early and Late Season WT Days per Month")


# from pandas.plotting import autocorrelation_plot

# autocorrelation_plot(df.EarlySeason)

plt.savefig("monthly_evl_montecarlo.png")
