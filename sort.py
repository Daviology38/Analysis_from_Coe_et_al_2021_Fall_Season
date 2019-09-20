import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import xlsxwriter
import pandas as pd

#Load in the Matlab File
mat_contents = sio.loadmat('H:\era5_son_new\CI_results.mat')
mat_contents2 = sio.loadmat('H:\era5_son_new\CI_results.mat')
#Put Variables into an array

K = mat_contents['K']
K1 = mat_contents2['K']

#Convert numbers to a python index style
ci1 = 3
ci2 = 7

arrayi1 = ci1
arrayi2 = ci2

#Place data into new arrays for comparison
dset1 = K[:,ci1-1]
dset2 = K1[:,ci2-1]

#Make storage array for comparison data
storage = np.zeros((arrayi1,arrayi2))

#Compare the days of each CI against each other for occurrence rates/dates
for i in range(dset1.shape[0]):
	val1 = dset1[i]
	val2 = dset2[i]
	storage[val1-1,val2-1] = storage[val1-1,val2-1] + 1

#Label the rows and columns
row_names = np.zeros((arrayi1))
column_names = np.zeros((arrayi2))

for i in range(arrayi1):
	row_names[i] = str(i+1)

for j in range(arrayi2):
	column_names[j] = str(j+1)

#Plot the figure
fig = plt.figure()
ax = Axes3D(fig)

#Get dimensions of the array
lx = storage.shape[0]
ly = storage.shape[1]


#Set up x and y arrays to mesh
xpos = np.arange(0,ly,1)
ypos = np.arange(0,lx,1)
xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)



#Convert positions to 1D array
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(lx*ly)
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = storage.flatten()

print column_names
print row_names
#cs = ['r', 'g', 'b', 'y', 'c'] * ly

ax.bar3d(xpos,ypos,zpos,dx,dy,dz)

ax.set_yticks(np.arange(len(row_names)))
ax.w_xaxis.set_ticklabels(column_names)
ax.w_yaxis.set_ticklabels(row_names)
ax.set_xlabel('CI')
ax.set_ylabel('CI')
ax.set_zlabel('Days')

name = str(ci1) + str(ci2) + '.png'
plt.savefig(name)
#Export data to Excel for plotting
filename = str(ci1) + str(ci2) + 'data.xlsx'

#Convert array to Dataframe
df = pd.DataFrame(storage)
df.to_excel(filename,index=False)




