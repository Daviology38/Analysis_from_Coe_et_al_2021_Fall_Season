import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gs


def make_grid(rows=0, cols=0, proj=ccrs.PlateCarree(), cbar=False):
    axlist = []
    fig2 = plt.figure(figsize=(20, 20))
    if cbar == False:
        spec2 = gs.GridSpec(ncols=cols, nrows=rows, figure=fig2)

        for x in range(rows):
            for y in range(cols):
                axlist.append(fig2.add_subplot(spec2[x, y], projection=proj))
    else:
        spec2 = gs.GridSpec(ncols=cols + 1, nrows=rows, figure=fig2)

        for x in range(rows):
            for y in range(cols):
                axlist.append(fig2.add_subplot(spec2[x, y], projection=proj))

        axlist.append(fig2.add_subplot(spec2[:, 3]))

    return axlist
