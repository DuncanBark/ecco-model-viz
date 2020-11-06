import os
import sys
import math
import pickle
import imageio
import numpy as np
import xarray as xr
import matplotlib as mpl
from pathlib import Path
import cartopy.crs as ccrs
from datetime import datetime
import matplotlib.pyplot as plt


def main(start='1995-01', end='2015-01', loc='', data_name='SEA_SURFACE_HEIGHT', field_name='SSH'):
    if not loc:
        loc = Path(
            f'{Path(__file__).resolve().parents[2]}/model_output/{data_name}')

    dates_in_range = np.arange(f'{start}', f'{end}', dtype='datetime64[M]')

    # Get files
    files_to_use = []
    for date in dates_in_range:
        files_to_use.append(list(loc.glob(f'*{date}*'))[0])

    # Open and save PNG of data
    image_output = f'{Path(__file__).resolve().parents[2]}/images/{data_name}/'
    if not os.path.exists(image_output):
        os.makedirs(image_output)

    data_max = -math.inf
    data_min = math.inf
    for datafile in files_to_use:
        ds = xr.open_dataset(datafile)
        data_var = ds[field_name]

        cur_max = np.nanmax(data_var.values[0])
        cur_min = np.nanmin(data_var.values[0])
        if cur_max > data_max:
            data_max = cur_max
        if cur_min < data_min:
            data_min = cur_min

    new_long = -180
    for datafile in files_to_use:
        image_name = f'{image_output}{os.path.basename(datafile)[:-2]}png'
        if not os.path.exists(image_name):
            ds = xr.open_dataset(datafile)
            data_var = ds[field_name]

            fig = plt.figure(figsize=(10, 10))

            ax = fig.add_subplot(1, 2, 1, projection=ccrs.Orthographic(central_longitude=new_long))
            # ax.background_img(name='BM', resolution='low')
            cs = ax.imshow(data_var.values[0], transform=ccrs.PlateCarree(), origin='lower')
            plt.title(f'{ds.time.values[0]}'[:7])

            ax = fig.add_subplot(1, 2, 2, projection=ccrs.Orthographic(central_longitude=new_long))
            # ax.background_img(name='BM', resolution='low')
            cs = ax.imshow(data_var.values[0], transform=ccrs.PlateCarree(), origin='lower')
            plt.title(f'{ds.time.values[0]}'[:7])

            cax, _ = mpl.colorbar.make_axes(ax, location='right', fraction=0.046, pad=0.04)
            mpl.colorbar.ColorbarBase(cax, norm=mpl.colors.Normalize(vmin=data_min, vmax=data_max)) 
            cs.set_clim(vmin=data_min, vmax=data_max)   

            fig.savefig(image_name, dpi=50,
                        bbox_inches='tight', pad_inches=0.1)

            # plt.show()
            plt.close()

        new_long += 360/len(files_to_use)
        if new_long > 180:
            new_long = -180

    # Create GIF
    images_to_use = [f for f in os.listdir(image_output) if 'png' in f]
    images_to_use.sort()
    images = []
    for img in images_to_use:
        img = f'{image_output}{img}'
        images.append(imageio.imread(img))
    imageio.mimsave(f'{image_output}animation.gif', images, fps=15, subrectangles=True)


if __name__ == "__main__":
    main()
