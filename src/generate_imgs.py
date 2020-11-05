import os
import sys
import pickle
import imageio
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


def main(start='2000-01', end='2010-01', loc='', data_name='SEA_SURFACE_HEIGHT'):
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

    counter = 0
    for datafile in files_to_use:
        counter += 1
        if counter < 20:
            image_name = f'{image_output}{os.path.basename(datafile)[:-2]}png'
            if not os.path.exists(image_name):
                fig = plt.figure()
                ds = xr.open_dataset(datafile)
                data_var = ds.SSH

                plt.imshow(data_var[0])
                plt.gca().invert_yaxis()
                plt.gca().set_axis_off()
                fig.savefig(image_name, dpi=150,
                            bbox_inches='tight', pad_inches=0)

                plt.close()

    # Create GIF
    images_to_use = [f for f in os.listdir(image_output) if 'png' in f]
    images_to_use.sort()
    images = []
    print(images_to_use)
    for img in images_to_use:
        img = f'{image_output}{img}'
        print(img)
        images.append(imageio.imread(img))
    imageio.mimsave(f'{image_output}animation.gif', images)


if __name__ == "__main__":
    main()
