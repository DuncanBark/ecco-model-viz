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

# # #
# Need to get corresponding data files.
#   Daily data needs to be filtered to monthly
#   Need to handle missing files...use tolerance?
# Need to mak sure data projection matches ecco projection
# # #


# Caculates min and max value for a list of files (typically a list of files for a single dataset)
def calc_min_max(files, field_name):
    data_max = -math.inf
    data_min = math.inf

    for datafile in files:
        ds = xr.open_dataset(datafile)
        data_var = ds[field_name]

        cur_max = np.nanmax(data_var.values[0])
        cur_min = np.nanmin(data_var.values[0])
        if cur_max > data_max:
            data_max = cur_max
        if cur_min < data_min:
            data_min = cur_min

    return data_min, data_max


# Takes two lists of files and field names (and an output path) and creates side by side pngs with a single colorbar
def make_screenshots(ecco_files_to_use, ecco_field_name, data_files_to_use, data_field_name, image_output):
    ecco_min, ecco_max = calc_min_max(ecco_files_to_use, ecco_field_name)
    data_min, data_max = calc_min_max(data_files_to_use, data_field_name)

    global_min = min(ecco_min, data_min)
    global_max = max(ecco_max, data_max)

    # Create and save pngs
    new_long = -180
    for datafile in ecco_files_to_use:
        image_name = f'{image_output}{os.path.basename(datafile)[:-2]}png'
        if not os.path.exists(image_name):
            ds = xr.open_dataset(datafile)
            data_var = ds[ecco_field_name]

            fig, axes = plt.subplots(ncols=2, figsize=(15, 15),
                                     subplot_kw={'projection': ccrs.Orthographic(central_longitude=new_long)})

            for ax in axes.flat:
                im = ax.imshow(
                    data_var.values[0], transform=ccrs.PlateCarree(), origin='lower', vmin=global_min, vmax=global_max)
                ax.set_title(f'{ds.time.values[0]}'[:7])

            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.35)

            fig.savefig(image_name, dpi=50,
                        bbox_inches='tight', pad_inches=0.1)

            plt.close()

        new_long += 360/len(ecco_files_to_use)
        if new_long > 180:
            new_long = -180


# Creates a gif from a list of files
def create_gif(images_to_use, image_output):
    images = []

    for img in images_to_use:
        img = f'{image_output}{img}'
        images.append(imageio.imread(img))

    imageio.mimsave(f'{image_output}animation.gif',
                    images, fps=15, subrectangles=True)


def main(start='1995-01', end='2015-01', ecco_loc='', ecco_field_name='SSH', data_name='SEA_SURFACE_HEIGHT'):
    if not ecco_loc:
        ecco_loc = Path(
            f'{Path(__file__).resolve().parents[2]}/model_output/{data_name}')

    dates_in_range = np.arange(f'{start}', f'{end}', dtype='datetime64[M]')

    image_output = f'{Path(__file__).resolve().parents[2]}/images/{data_name}/'
    if not os.path.exists(image_output):
        os.makedirs(image_output)

    # Get ecco files
    ecco_files_to_use = []
    for date in dates_in_range:
        ecco_files_to_use.append(list(ecco_loc.glob(f'*{date}*'))[0])

    # Get data files
    #
    #

    # TODO Currently uses the ecco data for both plots.
    make_screenshots(ecco_files_to_use, ecco_field_name,
                     ecco_files_to_use, ecco_field_name, image_output)

    # Create GIF
    images_to_use = [f for f in os.listdir(image_output) if 'png' in f]
    images_to_use.sort()

    create_gif(images_to_use, image_output)


if __name__ == "__main__":
    main()
