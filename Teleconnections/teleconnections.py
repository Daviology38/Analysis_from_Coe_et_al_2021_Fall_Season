import pandas as pd
import numpy as np
import xarray as xr
import scipy as sp
import scipy.ndimage as ndimage
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
import random
from itertools import groupby

# Create 3X3 sub plots
gs = gridspec.GridSpec(2,2)

#Read in the files for each teleconnection
ao = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Teleconnections/ao.csv")
nao = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Teleconnections/nao.csv")
pna = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Teleconnections/pna.csv")
nino = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Teleconnections/nino.csv")
mjo = pd.read_csv("C:/Users/CoeFamily/Documents/David College Class Work/WT_analysis/Teleconnections/mjo_mam.csv")

data = sio.loadmat(
        "C:/Users/CoeFamily/OneDrive - University of Massachusetts Lowell - UMass Lowell/WTs_general/era5_mam_jan21_85/CI_results.mat")
clustvals = data["K"][:, 6]
ldf = clustvals

#Parse the teleconnection data for 1979-2018
ao = ao.loc[(ao["Year"]>=1979) & (ao["Year"] <= 2018) & ((ao["M\t"] == 3) | (ao["M\t"] == 4) | (ao["M\t"] == 5))]
nao = nao.loc[(nao["Year"]>=1979) & (nao["Year"] <= 2018) & ((nao["M"] == 3) | (nao["M"] == 4) | (nao["M"] == 5))]
pna = pna.loc[(pna["Year"]>=1979) & (pna["Year"] <= 2018) & ((pna["M"] == 3) | (pna["M"] == 4) | (pna["M"] == 5))]

# #Parse the el nino data and extend each monthly value by day for each WT
nino = nino.loc[(nino["MON"] ==3) | (nino["MON"] == 4) | (nino["MON"] == 5)]

#Make a list of it for each day for each year
daily_nino = []
for i in range(1979,2019,1):
    y = 1
    while y < 93:
        if(i < 1982):
            daily_nino.append(np.nan)
        else:
            if (y <= 31):
                daily_nino.append(nino.loc[nino["YR"]==i]["ANOM.3"].values[0])
            elif (y > 31 and y <= 61):
                daily_nino.append(nino.loc[nino["YR"]==i]["ANOM.3"].values[1])
            else:
                daily_nino.append(nino.loc[nino["YR"]==i]["ANOM.3"].values[2])
        y = y + 1



#Put all the indices and WT daily numbers into a dictionary
d = {"WT":ldf,"AO":ao.AO.values,"NAO":nao.NAO.values,"PNA":pna.PNA.values,"NINO": daily_nino}

#Put that data into a DataFrame to have one concise dataset
overall_data = pd.DataFrame(d)

#Get the AO, NAO, PNA and Nino values for each WT
conds = [overall_data.values <= -1.0, overall_data.values >= 1.0, np.isnan(overall_data.values)]
choices = ["Negative", "Positive", np.nan]

dff = pd.DataFrame(np.select(conds, choices, default='Neutral'), index = overall_data.index, columns = overall_data.columns)
dff.WT = ldf
dff["MJO"] = mjo["Phase"].values.astype(str)

AO_actual = np.zeros((3,7))
NAO_actual = np.zeros((3,7))
PNA_actual = np.zeros((3,7))
NINO_actual = np.zeros((3,7))
MJO_actual = np.zeros((8,7))

AO_actual25 = np.zeros((3,7))
NAO_actual25 = np.zeros((3,7))
PNA_actual25 = np.zeros((3,7))
NINO_actual25 = np.zeros((3,7))
MJO_actual25 = np.zeros((8,7))

AO_actual975 = np.zeros((3,7))
NAO_actual975 = np.zeros((3,7))
PNA_actual975 = np.zeros((3,7))
NINO_actual975 = np.zeros((3,7))
MJO_actual975 = np.zeros((8,7))

AO_actualtemp = np.zeros((1000,3,7))
NAO_actualtemp = np.zeros((1000,3,7))
PNA_actualtemp = np.zeros((1000,3,7))
NINO_actualtemp = np.zeros((1000,3,7))
MJO_actualtemp = np.zeros((1000,8,7))

for i in range(1,8,1):
    AO_actual[0,i-1] = dff.groupby(dff.WT).get_group(i).AO.value_counts()["Negative"]
    AO_actual[1, i-1] = dff.groupby(dff.WT).get_group(i).AO.value_counts()["Neutral"]
    AO_actual[2, i-1] = dff.groupby(dff.WT).get_group(i).AO.value_counts()["Positive"]

    NAO_actual[0, i-1] = dff.groupby(dff.WT).get_group(i).NAO.value_counts()["Negative"]
    NAO_actual[1, i-1] = dff.groupby(dff.WT).get_group(i).NAO.value_counts()["Neutral"]
    NAO_actual[2, i-1] = dff.groupby(dff.WT).get_group(i).NAO.value_counts()["Positive"]

    PNA_actual[0, i-1] = dff.groupby(dff.WT).get_group(i).PNA.value_counts()["Negative"]
    PNA_actual[1, i-1] = dff.groupby(dff.WT).get_group(i).PNA.value_counts()["Neutral"]
    PNA_actual[2, i-1] = dff.groupby(dff.WT).get_group(i).PNA.value_counts()["Positive"]

    NINO_actual[0, i-1] = dff.groupby(dff.WT).get_group(i).NINO.value_counts()["Negative"]
    NINO_actual[1, i-1] = dff.groupby(dff.WT).get_group(i).NINO.value_counts()["Neutral"]
    NINO_actual[2, i-1] = dff.groupby(dff.WT).get_group(i).NINO.value_counts()["Positive"]

    MJO_actual[0, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["1"]
    MJO_actual[1, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["2"]
    MJO_actual[2, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["3"]
    MJO_actual[3, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["4"]
    MJO_actual[4, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["5"]
    MJO_actual[5, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["6"]
    MJO_actual[6, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["7"]
    MJO_actual[7, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["8"]

values = np.sum(AO_actual,0)
AO_actual = AO_actual
NAO_actual = NAO_actual
PNA_actual = PNA_actual
NINO_actual = NINO_actual

for j in range(1000):
    ls = ldf
    random.shuffle(ls)
    # Put all the indices and WT daily numbers into a dictionary
    d = {"WT": ls, "AO": ao.AO.values, "NAO": nao.NAO.values, "PNA": pna.PNA.values, "NINO": daily_nino}

    # Put that data into a DataFrame to have one concise dataset
    overall_data = pd.DataFrame(d)

    # Get the AO, NAO, PNA and Nino values for each WT
    conds = [overall_data.values <= -1.0, overall_data.values >= 1.0, np.isnan(overall_data.values)]
    choices = ["Negative", "Positive", np.nan]

    dff = pd.DataFrame(np.select(conds, choices, default='Neutral'), index=overall_data.index,
                       columns=overall_data.columns)
    dff.WT = ls
    dff["MJO"] = mjo["Phase"].values.astype(str)

    for i in range(1, 8, 1):
        AO_actualtemp[j,0, i - 1] = dff.groupby(dff.WT).get_group(i).AO.value_counts()["Negative"]
        AO_actualtemp[j,1, i - 1] = dff.groupby(dff.WT).get_group(i).AO.value_counts()["Neutral"]
        AO_actualtemp[j,2, i - 1] = dff.groupby(dff.WT).get_group(i).AO.value_counts()["Positive"]

        NAO_actualtemp[j,0, i - 1] = dff.groupby(dff.WT).get_group(i).NAO.value_counts()["Negative"]
        NAO_actualtemp[j,1, i - 1] = dff.groupby(dff.WT).get_group(i).NAO.value_counts()["Neutral"]
        NAO_actualtemp[j,2, i - 1] = dff.groupby(dff.WT).get_group(i).NAO.value_counts()["Positive"]

        PNA_actualtemp[j,0, i - 1] = dff.groupby(dff.WT).get_group(i).PNA.value_counts()["Negative"]
        PNA_actualtemp[j,1, i - 1] = dff.groupby(dff.WT).get_group(i).PNA.value_counts()["Neutral"]
        PNA_actualtemp[j,2, i - 1] = dff.groupby(dff.WT).get_group(i).PNA.value_counts()["Positive"]

        NINO_actualtemp[j,0, i - 1] = dff.groupby(dff.WT).get_group(i).NINO.value_counts()["Negative"]
        NINO_actualtemp[j,1, i - 1] = dff.groupby(dff.WT).get_group(i).NINO.value_counts()["Neutral"]
        NINO_actualtemp[j,2, i - 1] = dff.groupby(dff.WT).get_group(i).NINO.value_counts()["Positive"]

        MJO_actualtemp[j, 0, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["1"]
        MJO_actualtemp[j, 1, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["2"]
        MJO_actualtemp[j, 2, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["3"]
        MJO_actualtemp[j, 3, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["4"]
        MJO_actualtemp[j, 4, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["5"]
        MJO_actualtemp[j, 5, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["6"]
        MJO_actualtemp[j, 6, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["7"]
        MJO_actualtemp[j, 7, i - 1] = dff.groupby(dff.WT).get_group(i).MJO.value_counts()["8"]



for k in range(7):
    for kk in range(3):
        AO_actual25[kk,k] = np.sort(AO_actualtemp[:,kk,:],axis=0)[24,k]
        NAO_actual25[kk,k] = np.sort(NAO_actualtemp[:,kk,:],axis=0)[24,k]
        PNA_actual25[kk,k] = np.sort(PNA_actualtemp[:,kk,:],axis=0)[24,k]
        NINO_actual25[kk,k] = np.sort(NINO_actualtemp[:,kk,:],axis=0)[24,k]

        AO_actual975[kk,k] = np.sort(AO_actualtemp[:,kk,:],axis=0)[974,k]
        NAO_actual975[kk,k] = np.sort(NAO_actualtemp[:,kk,:],axis=0)[974,k]
        PNA_actual975[kk,k] = np.sort(PNA_actualtemp[:,kk,:],axis=0)[974,k]
        NINO_actual975[kk,k] = np.sort(NINO_actualtemp[:,kk,:],axis=0)[974,k]

for lm in range(7):
    for mm in range(8):
        MJO_actual25[mm,lm] = np.sort(MJO_actualtemp[:,mm,:],axis=0)[24,lm]
        MJO_actual975[mm, lm] = np.sort(MJO_actualtemp[:, mm, :], axis=0)[974, lm]

topAO = np.zeros((3, 7))
botAO = np.zeros((3, 7))

topNAO = np.zeros((3, 7))
botNAO = np.zeros((3, 7))

topPNA = np.zeros((3, 7))
botPNA = np.zeros((3, 7))

topNINO = np.zeros((3, 7))
botNINO = np.zeros((3, 7))

topMJO = np.zeros((8, 7))
botMJO = np.zeros((8, 7))

for l in range(3):
    for k in range(7):
        xx = AO_actual25[l, k]
        yy = AO_actual975[l, k]

        if (np.floor(xx) == np.floor(yy) or np.ceil(xx) == np.ceil(yy)):
            topAO[l, k] = xx
            botAO[l, k] = yy
        elif (np.floor(xx) == np.ceil(yy)):
            topAO[l, k] = xx
            botAO[l, k] = yy
        elif (np.floor(yy) == np.ceil(xx)):
            topAO[l, k] = yy
            botAO[l, k] = xx
        elif (xx < yy):
            botAO[l, k] = xx
            topAO[l, k] = yy
        else:
            topAO[l, k] = xx
            botAO[l, k] = yy

        xx = NAO_actual25[l, k]
        yy = NAO_actual975[l, k]

        if (np.floor(xx) == np.floor(yy) or np.ceil(xx) == np.ceil(yy)):
            topNAO[l, k] = xx
            botNAO[l, k] = yy
        elif (np.floor(xx) == np.ceil(yy)):
            topNAO[l, k] = xx
            botNAO[l, k] = yy
        elif (np.floor(yy) == np.ceil(xx)):
            topNAO[l, k] = yy
            botNAO[l, k] = xx
        elif (xx < yy):
            botNAO[l, k] = xx
            topNAO[l, k] = yy
        else:
            topNAO[l, k] = xx
            botNAO[l, k] = yy

        xx = PNA_actual25[l, k]
        yy = PNA_actual975[l, k]

        if (np.floor(xx) == np.floor(yy) or np.ceil(xx) == np.ceil(yy)):
            topPNA[l, k] = xx
            botPNA[l, k] = yy
        elif (np.floor(xx) == np.ceil(yy)):
            topPNA[l, k] = xx
            botPNA[l, k] = yy
        elif (np.floor(yy) == np.ceil(xx)):
            topPNA[l, k] = yy
            botPNA[l, k] = xx
        elif (xx < yy):
            botPNA[l, k] = xx
            topPNA[l, k] = yy
        else:
            topPNA[l, k] = xx
            botPNA[l, k] = yy

        xx = NINO_actual25[l, k]
        yy = NINO_actual975[l, k]

        if (np.floor(xx) == np.floor(yy) or np.ceil(xx) == np.ceil(yy)):
            topNINO[l, k] = xx
            botNINO[l, k] = yy
        elif (np.floor(xx) == np.ceil(yy)):
            topNINO[l, k] = xx
            botNINO[l, k] = yy
        elif (np.floor(yy) == np.ceil(xx)):
            topNINO[l, k] = yy
            botNINO[l, k] = xx
        elif (xx < yy):
            botNINO[l, k] = xx
            topNINO[l, k] = yy
        else:
            topNINO[l, k] = xx
            botNINO[l, k] = yy
for l in range(8):
    for k in range(7):
        xx = MJO_actual25[l, k]
        yy = MJO_actual975[l, k]

        if (np.floor(xx) == np.floor(yy) or np.ceil(xx) == np.ceil(yy)):
            topMJO[l, k] = xx
            botMJO[l, k] = yy
        elif (np.floor(xx) == np.ceil(yy)):
            topMJO[l, k] = xx
            botMJO[l, k] = yy
        elif (np.floor(yy) == np.ceil(xx)):
            topMJO[l, k] = yy
            botMJO[l, k] = xx
        elif (xx < yy):
            botMJO[l, k] = xx
            topMJO[l, k] = yy
        else:
            topMJO[l, k] = xx
            botMJO[l, k] = yy

cAO = np.zeros((3, 7))
cNAO = np.zeros((3, 7))
cPNA = np.zeros((3, 7))
cNINO = np.zeros((3, 7))
cMJO = np.zeros((8, 7))
C = pd.Series(['blue','red', 'black'])

cc = 0
dd = 0

while cc < 3:
    while dd < 7:
        if (AO_actual[cc, dd] > topAO[cc, dd]):
            cAO[cc, dd] = 0
        elif (AO_actual[cc, dd] < botAO[cc, dd]):
            cAO[cc, dd] = 1
        else:
            cAO[cc, dd] = 2

        if (NAO_actual[cc, dd] > topNAO[cc, dd]):
            cNAO[cc, dd] = 0
        elif (NAO_actual[cc, dd] < botNAO[cc, dd]):
            cNAO[cc, dd] = 1
        else:
            cNAO[cc, dd] = 2

        if (PNA_actual[cc, dd] > topPNA[cc, dd]):
            cPNA[cc, dd] = 0
        elif (PNA_actual[cc, dd] < botPNA[cc, dd]):
            cPNA[cc, dd] = 1
        else:
            cPNA[cc, dd] = 2

        if (NINO_actual[cc, dd] > topNINO[cc, dd]):
            cNINO[cc, dd] = 0
        elif (NINO_actual[cc, dd] < botNINO[cc, dd]):
            cNINO[cc, dd] = 1
        else:
            cNINO[cc, dd] = 2
        dd = dd + 1
    dd = 0
    cc = cc + 1

cc = 0
dd = 0
while cc < 8:
    while dd < 7:
        if (MJO_actual[cc, dd] > topMJO[cc, dd]):
            cMJO[cc, dd] = 0
        elif (MJO_actual[cc, dd] < botMJO[cc, dd]):
            cMJO[cc, dd] = 1
        else:
            cMJO[cc, dd] = 2
        dd = dd + 1
    dd = 0
    cc = cc + 1

fig = plt.figure()
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

values = np.sum(AO_actual,0)
AO_actual = AO_actual * 100 / values
NAO_actual = NAO_actual * 100 / values
PNA_actual = PNA_actual * 100 / values
NINO_actual = NINO_actual * 100 / values

fig.subplots_adjust(top=.90)
# set width of bar
barWidth = 0.25
bars1 = AO_actual[0,:]
# Set position of bar on X axis
r1 = np.arange(len(bars1))
#r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r1]


color1 = list(C[cAO[0, :]])
color2 = list(C[cAO[1, :]])
color3 = list(C[cAO[2, :]])

rr1 = ax1.bar(r1, bars1, color=color1, width=barWidth, edgecolor='white', label='-')
#rr2 = ax1.bar(r2, AO_actual[1,:], color=color2, width=barWidth, edgecolor='white', label='~')
rr3 = ax1.bar(r3, AO_actual[2,:], color=color3, width=barWidth, edgecolor='white', label='+')
ax1.set_xticklabels(['0','1','2','3','4','5','6','7'])

for rect in rr1:
    height = rect.get_height()
    label = '-'
    ax1.annotate('{}'.format(label),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# for rect in rr2:
#     height = rect.get_height()
#     label = '~'
#     ax1.annotate('{}'.format(label),
#                 xy=(rect.get_x() + rect.get_width() / 2, height),
#                 xytext=(0, 3),  # 3 points vertical offset
#                 textcoords="offset points",
#                 ha='center', va='bottom')

for rect in rr3:
    height = rect.get_height()
    label = '+'
    ax1.annotate('{}'.format(label),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Create legend & Show graphic
ax1.set_ylabel('% of days')
ax1.set_ylim(0,50,5)
ax1.set_title('AO')

# set width of bar
barWidth = 0.25
bars1 = NAO_actual[0,:]
# Set position of bar on X axis
r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r1]
color1 = list(C[cNAO[0, :]])
color2 = list(C[cNAO[1, :]])
color3 = list(C[cNAO[2, :]])

rr1 = ax2.bar(r1, bars1, color=color1, width=barWidth, edgecolor='white', label='-')
#rr2 = ax2.bar(r2, NAO_actual[1,:], color=color2, width=barWidth, edgecolor='white', label='~')
rr3 = ax2.bar(r3, NAO_actual[2,:], color=color3, width=barWidth, edgecolor='white', label='+')
ax2.set_xticklabels(['0','1','2','3','4','5','6','7'])

for rect in rr1:
    height = rect.get_height()
    label = '-'
    ax2.annotate('{}'.format(label),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# for rect in rr2:
#     height = rect.get_height()
#     label = '~'
#     ax2.annotate('{}'.format(label),
#                 xy=(rect.get_x() + rect.get_width() / 2, height),
#                 xytext=(0, 3),  # 3 points vertical offset
#                 textcoords="offset points",
#                 ha='center', va='bottom')

for rect in rr3:
    height = rect.get_height()
    label = '+'
    ax2.annotate('{}'.format(label),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Create legend & Show graphic
ax2.set_ylabel('% of days')
ax2.set_ylim(0,50,5)
ax2.set_title('NAO')

# set width of bar
barWidth = 0.25
bars1 = PNA_actual[0,:]
# Set position of bar on X axis
r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r1]
color1 = list(C[cPNA[0, :]])
color2 = list(C[cPNA[1, :]])
color3 = list(C[cPNA[2, :]])

rr1 = ax3.bar(r1, bars1, color=color1, width=barWidth, edgecolor='white', label='-')
#rr2 = ax3.bar(r2, PNA_actual[1,:], color=color2, width=barWidth, edgecolor='white', label='~')
rr3 = ax3.bar(r3, PNA_actual[2,:], color=color3, width=barWidth, edgecolor='white', label='+')
ax3.set_xticklabels(['0','1','2','3','4','5','6','7'])

for rect in rr1:
    height = rect.get_height()
    label = '-'
    ax3.annotate('{}'.format(label),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# for rect in rr2:
#     height = rect.get_height()
#     label = '~'
#     ax3.annotate('{}'.format(label),
#                 xy=(rect.get_x() + rect.get_width() / 2, height),
#                 xytext=(0, 3),  # 3 points vertical offset
#                 textcoords="offset points",
#                 ha='center', va='bottom')

for rect in rr3:
    height = rect.get_height()
    label = '+'
    ax3.annotate('{}'.format(label),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Create legend & Show graphic
ax3.set_xlabel('WT')
ax3.set_ylabel('% of days')
ax3.set_title('PNA')
ax3.set_ylim(0,50,5)

# set width of bar
barWidth = 0.25
bars1 = NINO_actual[0,:]
# Set position of bar on X axis
r1 = np.arange(len(bars1))
#r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r1]
color1 = list(C[cNINO[0, :]])
color2 = list(C[cNINO[1, :]])
color3 = list(C[cNINO[2, :]])

rr1 = ax4.bar(r1, bars1, color=color1, width=barWidth, edgecolor='white', label='-')
#rr2 = ax4.bar(r2, NINO_actual[1,:], color=color2, width=barWidth, edgecolor='white', label='~')
rr3 = ax4.bar(r3, NINO_actual[2,:], color=color3, width=barWidth, edgecolor='white', label='+')
ax4.set_xticklabels(['0','1','2','3','4','5','6','7'])

for rect in rr1:
    height = rect.get_height()
    label = '-'
    ax4.annotate('{}'.format(label),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# for rect in rr2:
#     height = rect.get_height()
#     label = '~'
#     ax4.annotate('{}'.format(label),
#                 xy=(rect.get_x() + rect.get_width() / 2, height),
#                 xytext=(0, 3),  # 3 points vertical offset
#                 textcoords="offset points",
#                 ha='center', va='bottom')

for rect in rr3:
    height = rect.get_height()
    label = '+'
    ax4.annotate('{}'.format(label),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Create legend & Show graphic
ax4.set_xlabel('WT')
ax4.set_ylim(0,50,5)
ax4.set_ylabel('% of days')
ax4.set_title('El Nino/ La Nina')

fig.suptitle("Teleconnections", fontsize=20)
plt.savefig("mam_teleconnections_12z.png", bbox_inches='tight')


for jj in range(8):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # set width of bar
    barWidth = 0.25
    bars1 = MJO_actual[jj,:]
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    #r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r1]
    color1 = list(C[cMJO[jj, :]])


    rr1 = ax.bar(r1, bars1, color=color1, width=barWidth, edgecolor='white', label='-')
    #rr2 = ax4.bar(r2, NINO_actual[1,:], color=color2, width=barWidth, edgecolor='white', label='~')
    #rr3 = ax4.bar(r3, MJO_actual[2,:], color=color3, width=barWidth, edgecolor='white', label='+')
    ax.set_xticklabels(['0','1','2','3','4','5','6','7'])

    # for rect in rr1:
    #     height = rect.get_height()
    #     label = '-'
    #     ax4.annotate('{}'.format(label),
    #                 xy=(rect.get_x() + rect.get_width() / 2, height),
    #                 xytext=(0, 3),  # 3 points vertical offset
    #                 textcoords="offset points",
    #                 ha='center', va='bottom')
    plt.savefig('mjo_phase_mam_'+str(jj)+'.png')