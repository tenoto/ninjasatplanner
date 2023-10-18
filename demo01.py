# import packages
import cubesat
import datetime
import argparse
import os
import yaml
import glob
import math
import ephem
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cubesat as cubesat
import importlib
importlib.reload(cubesat)
import skyfield.sgp4lib as sgp4lib
from astropy import coordinates as coord, units as u
from astropy.time import Time

# setup file links
# setup_yamlfile='C:/Users/Owner/Box/Per_201442_kentaro Taniguchi/ninjasat/data/ninjasat_setup.yaml'
# data_yamlfile='C:/Users/Owner/Box/Per_201442_kentaro Taniguchi/ninjasat/data/ninjasat_data_setup.yaml'
setup_yamlfile='C:/Users/tkent/Box/Per_201442_kentaro Taniguchi/ninjasat_github/data/ninjasat_setup.yaml'
data_yamlfile='C:/Users/tkent/Box/Per_201442_kentaro Taniguchi/ninjasat_github/data/ninjasat_data_setup.yaml'

# setup parameters
input_start_date = '2023-11-01 00:00:00'
input_end_date = '2023-11-01 05:00:00'
timebin_minute = 5
time_str = input_start_date
utc_time = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

# setup initial object
ninjasat = cubesat.CubeSat(setup_yamlfile, data_yamlfile, input_start_date, input_end_date, timebin_minute)

# setup main dataframe
ninjasat.setup_df()

# propagate satellite location (lon, lat, alt) from start date to end date
ninjasat.simulate_orbit()

# add satellite position in TEME and J2000
ninjasat.add_position_TEME()
ninjasat.add_position_J2000()

# add sun and moon position in J2000
ninjasat.add_sun_moon_position()

# add maxi rbm rate
ninjasat.setup_maxi_rbm_index()
ninjasat.add_maxi_rbm_index()

# add cutoff rigidity
ninjasat.set_cutoff_rigidity()
ninjasat.add_cutoff_rigidity()

# add communication path and elevation angles
ninjasat.add_observer_vis()

# add target position and visibility
ninjasat.setup_target_df()
ninjasat.add_target_vis()

# add orbit number and define satellite mode for any orbit
ninjasat.add_orbit_num()
# ninjasat.add_satellite_mode()

# output main dataframe
print(ninjasat.df)
ninjasat.save_df_to_csv(ninjasat.df, "result/df.csv")

# output multiple figures describing position relationship between orbit plane and targets direction
ninjasat.plot_orbit_plane_target_vis_multiple_plot()

# add satellite mode
ninjasat.add_satellite_mode()

# output result dataframe about satellite mode allocation
print(ninjasat.result_df)
ninjasat.save_df_to_csv(ninjasat.result_df, "result/result_df.csv")

# output result of plot_one_orbit
ninjasat.plot_one_orbit(2)

