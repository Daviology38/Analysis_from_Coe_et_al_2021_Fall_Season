import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

data = np.random.random([100, 100]) * 10

# create yellow colormaps
N = 256
green = np.ones((N, 4))
green[:, 0] = np.linspace(55/256, 1, N) # R = 55
green[:, 1] = np.linspace(69/256, 1, N) # G = 69
green[:, 2] = np.linspace(33/256, 1, N)  # B = 33
green_cmp = ListedColormap(green)

brown = np.ones((N, 4))
brown[:, 0] = np.linspace(102/256, 1, N) # R = 102
brown[:, 1] = np.linspace(58/256, 1, N) # G = 58
brown[:, 2] = np.linspace(20/256, 1, N) # B = 20
brown_cmp = ListedColormap(brown)

newcolors2 = np.vstack((brown_cmp(np.linspace(0, 1, 128)),
                       green_cmp(np.linspace(1, 0, 128))))
double = ListedColormap(newcolors2, name='double')
plt.figure(figsize=(7, 6))
plt.pcolormesh(data, cmap=double)
plt.colorbar()

