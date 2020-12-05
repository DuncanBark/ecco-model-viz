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
#       Daily data needs to be filtered to monthly
#       Need to handle missing files...use tolerance?
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
def make_screenshots(ecco_files_to_use, ecco_field_name, data_files_to_use, data_field_name, image_output, dim):
    ecco_min, ecco_max = calc_min_max(ecco_files_to_use, ecco_field_name)
    data_min, data_max = calc_min_max(data_files_to_use, data_field_name)

    global_min = min(ecco_min, data_min)
    global_max = max(ecco_max, data_max)

    # Create and save pngs
    new_long = -180
    for ecco_file, data_file in zip(ecco_files_to_use, data_files_to_use):
        image_name = f'{image_output}{os.path.basename(ecco_file)[:-2]}png'

        if not os.path.exists(image_name):
            ecco_ds = xr.open_dataset(ecco_file)
            ecco_data_var = ecco_ds[ecco_field_name]

            data_ds = xr.open_dataset(data_file)

            if (max(data_ds.coords['Longitude']) > 180):
                data_ds.coords['Longitude'] = (
                    data_ds.coords['Longitude'] + 180) % 360 - 180
                data_ds = data_ds.sortby(data_ds['Longitude'])

            data_data_var = data_ds[data_field_name].values.T

            # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15),
            #                          subplot_kw={'projection': ccrs.Orthographic(central_longitude=new_long)})

            # fig = plt.figure()

            # gs = fig.add_gridspec(2, 2)
            # f_ax1 = fig.add_subplot(gs[0, 0])
            # f_ax1.imshow(data_data_var)
            # f_ax1.set_title(f'DATA\n{data_file[-10:-3]}')

            # f_ax2 = fig.add_subplot(gs[0, 1])
            # im2 = f_ax2.imshow(ecco_data_var.values[0])
            # f_ax2.set_title(f'ECCO\n{f"{ecco_ds.time.values[0]}"[:7]}')

            # im1 = axes[0][0].imshow(
            #     data_data_var, transform=ccrs.PlateCarree(), origin='lower', vmin=global_min, vmax=global_max)
            # axes[0].set_title(f'DATA\n{data_file[-10:-3]}')

            # im2 = axes[0][1].imshow(
            #     ecco_data_var.values[0], transform=ccrs.PlateCarree(), origin='lower', vmin=global_min, vmax=global_max)
            # axes[1].set_title(f'ECCO\n{f"{ecco_ds.time.values[0]}"[:7]}')

            # plt.colorbar(im2, ax=[f_ax1, f_ax2], shrink=0.35)

            # f_ax3 = fig.add_subplot(gs[1, :])
            # im2 = f_ax3.imshow(ecco_data_var.values[0])
            # f_ax3.set_title(f'ECCO\n{f"{ecco_ds.time.values[0]}"[:7]}')

            # plt.show()

            fig = plt.figure(figsize=(15, 15))
            ax1 = plt.subplot(221, projection=ccrs.Orthographic(
                central_longitude=new_long))
            ax1.imshow(data_data_var, transform=ccrs.PlateCarree(),
                       origin='lower', vmin=global_min, vmax=global_max)
            ax1.set_title(f'DATA\n{data_file[-10:-3]}')

            ax2 = plt.subplot(222, projection=ccrs.Orthographic(
                central_longitude=new_long))
            im2 = ax2.imshow(ecco_data_var.values[0], transform=ccrs.PlateCarree(
            ), origin='lower', vmin=global_min, vmax=global_max)
            ax2.set_title(f'ECCO\n{f"{ecco_ds.time.values[0]}"[:7]}')

            ax3 = plt.subplot(212)
            ax3.plot(np.linspace(-np.pi, np.pi, 100))
            ax3.set_title(f'Comparison plot')

            plt.colorbar(im2, ax=[ax1, ax2], shrink=.8)

            fig.savefig(image_name, dpi=50,
                        bbox_inches='tight', pad_inches=0.1)

            plt.close()

        new_long += 360/len(ecco_files_to_use)
        if new_long > 180:
            new_long = -180


# Creates a gif from a list of files
def create_gif(images_to_use, image_output, fps=15):
    images = []

    for img in images_to_use:
        img = f'{image_output}{img}'
        images.append(imageio.imread(img))

    imageio.mimsave(f'{image_output}animation.gif',
                    images, fps=fps, subrectangles=True)


def monthly_aggregate(data_loc, dates, date_format, data_time_scale, image_output, dim, data_field_name):
    # Iterate through dates, mean the collected files for that date range (month), meaned dataarray gets appended to list
    # Should have 12*20 dataarrays in list
    monthly_aggregate_files = []

    output_dir = f'{image_output}aggregated_files/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    years = list(set([f'{date}'[:4] for date in dates]))

    for year in years:
        print(f'Working on {year}')
        daily_paths = dict()

        # find all etcdf files in this directory that have the year and suffix
        all_netcdf_files_year = np.sort(
            list(data_loc.glob(f'**/*{year}*')))

        dates_in_year = []
        if data_time_scale == 'monthly':
            for i in range(1, 13):
                if date_format == 'yyyymm':
                    dates_in_year.append(str(year) + f'{i:02d}')
                elif date_format == 'yyyy_mm':
                    dates_in_year.append(str(year) + '_' + f'{i:02d}')
                elif date_format == 'yyyyddd':
                    dates_in_year.append(str(year) + f'{i:02d}')
        elif data_time_scale == 'daily':
            if date_format == 'yyyyddd':
                for i in range(1, 367):
                    dates_in_year.append(str(year) + f'{i:03d}')
            elif date_format == 'yyyymmdd':
                dates_in_year = np.arange(f'{year}-01-01', f'{int(year)+1}-01-01',
                                          dtype='datetime64[D]')

        # make empty list that will contain the dates in this year in iso format
        # yyyy-mm-dd
        dates_in_year_iso = []
        # loop through every day in the year
        for date in dates_in_year:
            if data_time_scale == 'monthly':
                if date_format == 'yyyymm':
                    date_str_iso_obj = datetime.strptime(date, '%Y%m')
                    date_str = date
                elif date_format == 'yyyy_mm':
                    date_str_iso_obj = datetime.strptime(date, '%Y_%m')
                    date_str = date
                elif date_format == 'yyyyddd':
                    date_str_iso_obj = datetime.strptime(date, '%Y%m')
                    date_str = datetime.strftime(date_str_iso_obj, '%Y%j')
            elif data_time_scale == 'daily':
                if date_format == 'yyyyddd':
                    print('TODO: implement daily yyyyddd date support')
                    date_str = date
                elif date_format == 'yyyymmdd':
                    date_str = str(date.tolist().year) + \
                        str(date.tolist().month).zfill(2) + \
                        str(date.tolist().day).zfill(2)
                    date_str_iso_obj = datetime.strptime(date_str, '%Y%m%d')

            # add iso format date to dates_in_year_iso
            date_str_iso = datetime.strftime(date_str_iso_obj, '%Y-%m-%d')
            dates_in_year_iso.append(date_str_iso)

            # find the path that matches this day
            paths_date = []
            # Extracting date from path
            # Need to isolate date from the data name and excess information at the end
            for netcdf_path in all_netcdf_files_year:
                if str.find(str(netcdf_path), date_str) >= 0:
                    paths_date = netcdf_path

            # add this path to the dictionary with the date_str_iso as the key
            daily_paths[date_str_iso] = paths_date

        for month in range(1, 13):
            files_in_month = []
            month = f'{month:02d}'
            output_path = f'{output_dir}{year}-{month}.nc'

            # COMMENT ME IF YOU WANT TO RUN AGAIN
            if os.path.exists(output_path):
                monthly_aggregate_files.append(output_path)
                continue

            for date in daily_paths.keys():
                if date[:4] == year and date[5:7] == month:
                    if daily_paths[date]:
                        files_in_month.append(daily_paths[date])

            data_DA_month = []

            for file_path in files_in_month:
                if file_path:
                    data_DS = xr.open_dataset(file_path)
                    data_DA = data_DS[data_field_name]
                    data_DA_month.append(data_DA)

            data_DA_month_merged = xr.concat((data_DA_month), dim=dim)
            data_DA_month_meaned = data_DA_month_merged.mean(
                axis=0, skipna=True, keep_attrs=True)

            data_DA_month_meaned.to_netcdf(output_path)

            monthly_aggregate_files.append(output_path)
            print(f' - {year}-{month} complete')

    return monthly_aggregate_files


def main(start='1995-01', end='2015-01', ecco_field_name='SSH', ecco_data_name='SEA_SURFACE_HEIGHT', model_output_loc='',
         data_field_name='SLA', data_name='sla_MEaSUREs_JPL1812',  data_loc='', date_format='yyyymmdd', data_time_scale='daily', dim='Time'):

    dates_in_range = np.arange(f'{start}', f'{end}', dtype='datetime64[M]')

    image_output = f'{Path(__file__).resolve().parents[2]}/images/{ecco_data_name}/'

    if not os.path.exists(image_output):
        os.makedirs(image_output)

    # Get ecco files
    if not model_output_loc:
        model_output_loc = Path(
            f'{Path(__file__).resolve().parents[2]}/model_output/{ecco_data_name}')

    ecco_files_to_use = []
    for date in dates_in_range:
        ecco_files_to_use.append(list(model_output_loc.glob(f'*{date}*'))[0])
    ecco_files_to_use.sort()

    # Get data files
    if not data_loc:
        data_loc = Path(
            f'/Users/kevinmarlis/Developer/JPL/pipeline_output/{data_name}/harvested_granules/')

    data_files_to_use = monthly_aggregate(
        data_loc, dates_in_range, date_format, data_time_scale, image_output, dim, data_field_name)
    data_files_to_use.sort()

    # ecco_files_to_use, ecco_field_name, data_files_to_use, data_field_name, image_output
    make_screenshots(ecco_files_to_use, ecco_field_name,
                     data_files_to_use, data_field_name, image_output, dim)

    # Create GIF
    images_to_use = [f for f in os.listdir(image_output) if 'png' in f]
    images_to_use.sort()

    create_gif(images_to_use, image_output)


if __name__ == "__main__":
    main()
