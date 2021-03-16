from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np


f = Dataset("era5_land_snowfall.nc")