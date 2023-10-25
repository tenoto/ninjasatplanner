# -*- coding: utf-8 -*-

import os
import yaml
import glob
import math
import ephem
import pickle
import datetime
import numpy as np
import pandas as pd
# import healpy as hp


from pyorbital.orbital import Orbital
from pyorbital.tlefile import Tle
from pyorbital import tlefile

from skyfield.api import EarthSatellite
from skyfield.api import load

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

# COLUMN_KEYWORDS = ['utc_time','longitude_deg','latitude_deg','altitude_km','sun_elevation_deg','sat_elevation_deg','sat_azimuth_deg','cutoff_rigidity_GV',
# 	'maxi_rbm_rate_cps','hv_allowed_flag']

COLUMN_KEYWORDS = ['utc_time','longitude_deg','latitude_deg','altitude_km','dcm_x','sat_elevation_deg','sat_azimuth_deg','cutoff_rigidity_GV',
	'maxi_rbm_rate_cps','hv_allowed_flag']

class CubeSat():
	def __init__(self,setup_yamlfile,data_yamlfile,start_date_utc,end_date_utc,timebin_minute):

		print("init")

		self.param = yaml.load(open(setup_yamlfile),Loader=yaml.SafeLoader)

		self.orbital_tle = Tle(self.param['tle_name'],self.param['tle_file'])
		# self.orbital_tle = tlefile.read(self.param['tle_file'])
		self.orbital_orbit = Orbital(self.param['tle_name'], line1=self.orbital_tle.line1, line2=self.orbital_tle.line2)
		self.ts = load.timescale()
		self.skyfield_satellite = EarthSatellite(self.orbital_tle.line1, self.orbital_tle.line2, 'ISS', self.ts)

		self.ephem_sun = ephem.Sun()
		self.ephem_sat = ephem.Observer()

		self.df_column_keywords = ["Time", "Longitude", "Latitude", "Altitude"]
		self.df = None
		self.utc_time_init = start_date_utc
		self.utc_time_end = end_date_utc
		self.time_bin_min = timebin_minute
		# self.dcm_init = None
		# self.q_init = np.array([0, 0, 0, 0])
		# self.dcm_init = np.array([[1, 0, 0],
		#         					[0, 1, 0],
		#         					[0, 0, 1]])

		self.target_df = pd.read_csv(self.param['obs_target_list'],skiprows=[0,1],sep=',')
		self.gs_df = pd.read_csv(self.param['ground_station_list'],sep=',')

		self.sun_sep_angle = self.param["sun_separation_angle"] # deg
		self.moon_sep_angle = self.param["moon_separation_angle"] # deg

		self.track_dict = {}
		for keyword in self.df_column_keywords:
			self.track_dict[keyword] = []

		self.DEG2RAD = math.pi / 180
		self.RAD2DEG = 180 / math.pi
		self.ARCSEC2RAD = self.DEG2RAD /36000
		self.MJD_J2000 = 51544.5 # The value depends on the context, this is a commonly used value
		self.EARTH_RADIUS = 6371 # in kilometers
		self.MOON_RADIUS = 1738 # in kilometers
		self.EPS = 1e-9

		# 図の見やすさは以下の文字フォントの大きさで調整している
		plt.rcParams["font.size"] = 15


## Create dataframe and simulate satellite position vs time

	def simulate_orbit(self):
		"""
		Input : self
		Output : time is propageted to the time_end and full orbit is simulated
					full dataframe is returned
		"""
		last_row = self.df.iloc[-1]
		while pd.to_datetime(last_row['Time']) < self.str2datetime(self.utc_time_end):
			self.update_df()
			last_row = self.df.iloc[-1]
			# print(pd.to_datetime(last_row['Time']))

	def setup_df(self):
		"""
		set up initial dataframe which has the following keywords
		["Time" : string
		"Longitude""Altitude" float
		"""
		self.df = pd.DataFrame(columns=self.df_column_keywords)
		longitude, latitude, altitude = self.get_position(self.str2datetime(self.utc_time_init))
		# q_1d = self.q_init
		# row = [self.utc_time_init, longitude, latitude, altitude] + q_1d.tolist()
		row = [self.utc_time_init, longitude, latitude, altitude]
		new_df = pd.DataFrame([row], columns=self.df_column_keywords)
		self.df = pd.concat([self.df, new_df], ignore_index=True)

	def update_df(self):
		"""
		with adding the time delta to the time in the last row in the dataframe, append a new position with the new time to the dataframe
		"""
		last_row = self.df.iloc[-1]
		new_time = pd.to_datetime(last_row['Time']) + pd.Timedelta(minutes=self.time_bin_min)
		new_longitude, new_latitude, new_altitude = self.get_position(new_time)
		row = [self.datetime2str(new_time), new_longitude, new_latitude, new_altitude]
		new_df = pd.DataFrame([row], columns=self.df_column_keywords)
		self.df = pd.concat([self.df, new_df], ignore_index=True)

	def read_df(self, csv_path):
		self.df = pd.read_csv(csv_path)

	def save_df_to_csv(self, dataframe, save_csv_path):
		dataframe.to_csv(save_csv_path)

	def add_position_TEME(self):
		for index, row in self.df.iterrows():
			datetime_value = self.str2datetime(row["Time"])
			p, v = self.orbital_orbit.get_position(datetime_value, False)
			self.df.loc[index, "xTEME"] = p[0]
			self.df.loc[index, "yTEME"] = p[1]
			self.df.loc[index, "zTEME"] = p[2]

	def add_position_J2000(self):
		for index, row in self.df.iterrows():
			datetime_value = self.str2datetime(row["Time"])
			x_pos, y_pos, z_pos = self.get_position_J2000(datetime_value)
			self.df.loc[index, 'xJ2000'] = x_pos
			self.df.loc[index, 'yJ2000'] = y_pos
			self.df.loc[index, 'zJ2000'] = z_pos

	def get_position_J2000(self, datetime_value):
		# print(datetime_value)
		year = datetime_value.year
		month = datetime_value.month
		day = datetime_value.day
		hour = datetime_value.hour
		minute = datetime_value.minute
		second = datetime_value.second
		t = self.ts.utc(year, month, day, hour, minute, second)
		geocentric = self.skyfield_satellite.at(t)
		# print(geocentric.position.km)
		# print(geocentric.position.km[0])
		x_pos = geocentric.position.km[0]
		y_pos = geocentric.position.km[1]
		z_pos = geocentric.position.km[2]
		return x_pos, y_pos, z_pos

	def add_orbit_num(self):
		crossings = np.where(np.diff(np.sign(self.df['xJ2000'])) > 0)[0]
		self.df['Orbit']=np.zeros(len(self.df))
		# 周回数を計算
		for i, crossing in enumerate(crossings):
			self.df.loc[crossing+1:, 'Orbit'] = i+1

	def add_sun_moon_position(self):
		for index, row in self.df.iterrows():
				datetime_value = self.str2datetime(row["Time"])
				mjd_value = self.datetime2mjd(datetime_value)
				x_pos, y_pos, z_pos = self.get_position_J2000(datetime_value)
				satVect = np.array([x_pos, y_pos, z_pos])
				# moonVect = self.atMoon(mjd_value)[0] / np.linalg.norm(self.atMoon(mjd_value)[0])
				# sunVect = self.atSun(mjd_value)/ np.linalg.norm(self.atSun(mjd_value))

				# self.df.loc[index, 'xSunVect'] = sunVect[0]
				# self.df.loc[index, 'ySunVect'] = sunVect[1]
				# self.df.loc[index, 'zSunVect'] = sunVect[2]

				sun = ephem.Sun()
				sun.compute(self.df['Time'][index])
				self.df.loc[index, 'xSunVect'] = np.cos(sun.dec) * np.cos(sun.ra)
				self.df.loc[index, 'ySunVect'] = np.cos(sun.dec) * np.sin(sun.ra)
				self.df.loc[index, 'zSunVect'] = np.sin(sun.dec)

				# self.df.loc[index, 'xMoonVect'] = moonVect[0]
				# self.df.loc[index, 'yMoonVect'] = moonVect[1]
				# self.df.loc[index, 'zMoonVect'] = moonVect[2]

				moon = ephem.Moon()
				moon.compute(self.df['Time'][index])
				self.df.loc[index, 'xMoonVect'] = np.cos(moon.dec) * np.cos(moon.ra)
				self.df.loc[index, 'yMoonVect'] = np.cos(moon.dec) * np.sin(moon.ra)
				self.df.loc[index, 'zMoonVect'] = np.sin(moon.dec)

				moonVect = np.array([self.df.loc[index, 'xSunVect'],
					self.df.loc[index, 'ySunVect'],
					self.df.loc[index, 'zSunVect']])
				sunVect = np.array([self.df.loc[index, 'xMoonVect'],
					self.df.loc[index, 'yMoonVect'],
					self.df.loc[index, 'zMoonVect']])

				# self.df.loc[index, 'sunMoonSepAngle_epehm'] = ephem.separation(sun,moon)
				# self.df.loc[index, 'sunMoonSepAngle'] = self.ang_distance(moon_ephem, sun_ephem)

				flag, el, x_dist = self.at_earth_occult(satVect, sunVect, sunVect)
				self.df.loc[index, 'visFlag_Sun'] = flag
				self.df.loc[index, 'visEl_Sun'] = el

				flag, el, x_dist = self.at_earth_occult(satVect, moonVect, sunVect)
				self.df.loc[index, 'visFlag_Moon'] = flag
				self.df.loc[index, 'visEl_Moon'] = el


	def add_target_vis(self):
		for index, row in self.target_df.iterrows():
			# print(row)
			# print(index)
			target_ra = row["RA (radians)"]
			target_dec = row["DEC (radians)"]
			xVect = self.get_direction_vector(target_ra, target_dec)

			for index2, row2 in self.df.iterrows():
				datetime_value = self.str2datetime(row2["Time"])
				mjd_value = self.datetime2mjd(datetime_value)
				x_pos, y_pos, z_pos = self.get_position_J2000(datetime_value)
				satVect = np.array([x_pos, y_pos, z_pos])
				# moonVect = self.atMoon(mjd_value)[0] / np.linalg.norm(self.atMoon(mjd_value)[0])
				moonVect = np.array([self.df.loc[index2, 'xMoonVect'],
					 self.df.loc[index2, 'yMoonVect'],
					 self.df.loc[index2, 'zMoonVect']])
				# sunVect = self.atSun(mjd_value)/ np.linalg.norm(self.atSun(mjd_value))
				sunVect = np.array([self.df.loc[index2, 'xSunVect'],
					 self.df.loc[index2, 'ySunVect'],
					 self.df.loc[index2, 'zSunVect']])
				earthVect = -satVect

				self.df.loc[index2, 'xTargVec_'+row["Name"]] = xVect[0]
				self.df.loc[index2, 'yTargVec_'+row["Name"]] = xVect[1]
				self.df.loc[index2, 'zTargVec_'+row["Name"]] = xVect[2]

				flag, el, x_dist = self.at_earth_occult(satVect, xVect, sunVect)
				self.df.loc[index2, 'visFlag_'+row["Name"]] = flag
				self.df.loc[index2, 'visEl_'+row["Name"]] = el
				self.df.loc[index2, 'angDistEarth_'+row["Name"]] = x_dist

				self.df.loc[index2, 'angDistMoon_'+row["Name"]] = self.ang_distance(xVect, moonVect)
				self.df.loc[index2, 'angDistSun_'+row["Name"]] = self.ang_distance(xVect, sunVect)

				self.df.loc[index2, 'angDistFlagMoon_'+row["Name"]] = (self.ang_distance(xVect, moonVect)*self.RAD2DEG > self.moon_sep_angle)
				self.df.loc[index2, 'angDistFlagSun_'+row["Name"]] = (self.ang_distance(xVect, sunVect)*self.RAD2DEG > self.sun_sep_angle)



				self.df.loc[index2, 'obsFlag_'+row["Name"]] = (self.df.loc[index2, 'visFlag_'+row["Name"]] == 0)*(self.df.loc[index2, 'angDistFlagMoon_'+row["Name"]] == 1)*(self.df.loc[index2, 'angDistFlagSun_'+row["Name"]] == 1)

	def setup_target_df(self):
		self.target_df = pd.read_csv(self.param['obs_target_list'],skiprows=[0,1],sep=',')
		self.target_df['RA (radians)'] = self.target_df['RA (J2000)'].apply(self.angtime2radians)
		self.target_df['DEC (radians)'] = self.target_df['DEC (J2000)'].apply(lambda x: self.angtime2radians(x, ra=False))

	def plot_target_vis(self):
		fig = plt.figure(figsize=(12,8))
		gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
		ax0 = plt.subplot(gs[0])
		ax1 = plt.subplot(gs[1])

		for index_target, row_target in self.target_df.iterrows():
			datetime_series = self.df['Time'].apply(self.str2datetime)
			ax0.step(datetime_series, self.df['visFlag_'+row_target["Name"]], label=row_target['Name'])

			values = [0.0, 1.0, 2.0]
			visFlag_df = self.df['visFlag_'+row_target["Name"]]
			counts = [visFlag_df[visFlag_df == value].count() for value in values]
			# print(counts)

			# バーの位置を設定
			bar_width = 0.05
			index = np.arange(len(values))
			# print(index + bar_width * index_target)
			# バーのプロット
			ax1.barh(index + bar_width * index_target, counts, height=bar_width, label=row_target["Name"])

		ax0.legend()
		ax0.set_yticks([0, 1, 2])
		ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
		ax0.xaxis.set_major_locator(mdates.HourLocator(interval=2))
		ax0.tick_params(axis='x', rotation=60)
		ax0.set_ylabel('visFlag')
		# ax1.legend()
		ax1.set_xlabel('Frequency')
		# y軸のラベルを変更
		y_labels = ["visible", "dark\nearth", "bright\nearth"]
		y_ticks = [i + bar_width * len(self.target_df) / 2 for i in range(len(y_labels))]
		ax1.set_yticks(y_ticks)
		ax1.set_yticklabels(y_labels)
		plt.show()

	def plot_target_vis_v2(self):
		# Create a figure with custom size
		fig, axes = plt.subplots(5, 1, figsize=(12,16), sharex=False)
		# Adjust the space between subplots
		plt.subplots_adjust(hspace=0.3)
		# Convert the 'Time' column to datetime
		self.df['Time'] = pd.to_datetime(self.df['Time'])
		# Get the minimum and maximum time
		min_time = self.df['Time'].min()
		max_time = self.df['Time'].max()
		# Calculate the total time range in seconds
		total_seconds = (max_time - min_time).total_seconds()
		# Calculate the time range for each subplot in seconds
		subplot_seconds = total_seconds / 5
		# Define the time ranges for each subplot
		time_ranges = [(min_time + pd.Timedelta(seconds=i*subplot_seconds), min_time + pd.Timedelta(seconds=(i+1)*subplot_seconds)) for i in range(5)]
		for ax, (start_time, end_time) in zip(axes, time_ranges):
			for index_target, row_target in self.target_df.iterrows():
				# Filter the data for the current time range
				mask = (self.df['Time'] >= start_time) & (self.df['Time'] <= end_time)
				filtered_df = self.df.loc[mask]

				datetime_series = filtered_df['Time']
				# digitized flag plot by stepping functions
				# ax.step(datetime_series, filtered_df['visFlag_'+row_target["Name"]], label=row_target['Name'])
				# elevation plot of target bodies
				ax.plot(datetime_series, filtered_df['visEl_'+row_target["Name"]]/np.pi*180, label=row_target['Name'])
			ax.legend(fontsize='small')
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
			ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
			ax.set_ylabel('Elevation from\n Earth Rim [deg]')

		plt.show()

		# Convert the 'Time' column back to string
		self.df['Time'] = self.df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')


	def plot_target_vis_v2_head100(self):

		df_tent = self.df[:100]
		# Create a figure with custom size
		fig, axes = plt.subplots(2, 1, figsize=(12,8), sharex=False)
		# Adjust the space between subplots
		plt.subplots_adjust(hspace=0.2)
		# Convert the 'Time' column to datetime
		df_tent['Time'] = pd.to_datetime(df_tent['Time'])
		# Get the minimum and maximum time
		min_time = df_tent['Time'].min()
		max_time = df_tent['Time'].max()
		# Calculate the total time range in seconds
		total_seconds = (max_time - min_time).total_seconds()
		# Calculate the time range for each subplot in seconds
		subplot_seconds = total_seconds / 5
		# Define the time ranges for each subplot
		time_ranges = [(min_time + pd.Timedelta(seconds=i*subplot_seconds), min_time + pd.Timedelta(seconds=(i+1)*subplot_seconds)) for i in range(5)]
		for ax, (start_time, end_time) in zip(axes, time_ranges):
			for index_target, row_target in self.target_df.iterrows():
				# Filter the data for the current time range
				mask = (df_tent['Time'] >= start_time) & (df_tent['Time'] <= end_time)
				filtered_df = df_tent.loc[mask]

				datetime_series = filtered_df['Time']
				# digitized flag plot by stepping functions
				# ax.step(datetime_series, filtered_df['visFlag_'+row_target["Name"]], label=row_target['Name'])
				# elevation plot of target bodies
				ax.plot(datetime_series, filtered_df['visEl_'+row_target["Name"]]/np.pi*180, label=row_target['Name'])
			ax.legend()
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
			ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
			ax.set_ylabel('Elevation from\n Earth Rim [deg]')
		plt.show()

	def plot_target_vis_v3(self):
		# Create a figure with custom size
		fig, axes = plt.subplots(3, 1, figsize=(12,12), sharex=False)
		# Adjust the space between subplots
		plt.subplots_adjust(hspace=0.4)
		# Convert the 'Time' column to datetime
		self.df['Time'] = pd.to_datetime(self.df['Time'])
		# Get the minimum and maximum time
		min_time = self.df['Time'].min()
		max_time = self.df['Time'].max()
		# Calculate the total time range in seconds
		total_seconds = (max_time - min_time).total_seconds()
		# Calculate the time range for each subplot in seconds
		subplot_seconds = total_seconds / 3
		# Define the time ranges for each subplot
		time_ranges = [(min_time + pd.Timedelta(seconds=i*subplot_seconds), min_time + pd.Timedelta(seconds=(i+1)*subplot_seconds)) for i in range(7)]
		# Create a separate axis for the legend
		legend_ax = fig.add_axes([0.85, 0.1, 0.1, 0.8])
		for ax, (start_time, end_time) in zip(axes, time_ranges):
			for index_target, row_target in self.target_df.iterrows():
				# Filter the data for the current time range
				mask = (self.df['Time'] >= start_time) & (self.df['Time'] <= end_time)
				filtered_df = self.df.loc[mask]

				datetime_series = filtered_df['Time']
				# digitized flag plot by stepping functions
				# ax.step(datetime_series, filtered_df['visFlag_'+row_target["Name"]], label=row_target['Name'])
				# elevation plot of target bodies
				line, = ax.plot(datetime_series, filtered_df['visEl_'+row_target["Name"]]*self.RAD2DEG, label='EarthEl_'+row_target['Name'])
				color = line.get_color()

				ax.plot(datetime_series, filtered_df['angDistSun_'+row_target["Name"]]*self.RAD2DEG, label='SunSepAng_'+row_target["Name"], linestyle='-.', color=color)
				ax.plot(datetime_series, filtered_df['angDistMoon_'+row_target["Name"]]*self.RAD2DEG, label='MoonSepAng_'+row_target["Name"], linestyle=':', color=color)

				# Check the condition and fill the background if it is satisfied
				if (filtered_df['visFlag_'+row_target["Name"]] == 0).any():
					# Find the consecutive intervals where the condition is satisfied
					intervals = []
					start_index = None
					for i, value in enumerate(filtered_df['obsFlag_'+row_target["Name"]]):
						if value == 1 and start_index is None:
							start_index = i
						elif value != 1 and start_index is not None:
							intervals.append((filtered_df.iloc[start_index]['Time'], filtered_df.iloc[i-1]['Time']))
							start_index = None
					if start_index is not None:  # If the last interval continues until the end
						intervals.append((filtered_df.iloc[start_index]['Time'], filtered_df.iloc[-1]['Time']))

					# Fill the background for each interval
					for interval in intervals:
						ax.axvspan(interval[0], interval[1], facecolor=color, alpha=0.3)  # Fill the background

			ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
			ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
			ax.axhline(y=self.param["sight_elevation"], color='r', linestyle='--', label='Elev. Earth')
			ax.axhline(y=self.param["sun_separation_angle"], color='r', linestyle='--', label='Elev. Sun')
			ax.axhline(y=self.param["moon_separation_angle"], color='r', linestyle='--', label='Elev. Moon')
			ax.set_yticks([-90, 0, 90])
			ax.legend().set_visible(False)  # Hide the legend in the subplots

		# Create the legend in the separate axis
		legend_ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1])
		legend_ax.axis('off')  # Turn off the axis for the legend
		legend_ax.set_xlim(0, 1)
		legend_ax.set_ylim(0, 1)

		plt.show()

		# Convert the 'Time' column back to string
		self.df['Time'] = self.df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')


	def plot_target_vis_v4(self):
		# Create a figure with custom size
		fig, ax = plt.subplots(1, 1, figsize=(8,8), sharex=False)
		# Adjust the space between subplots
		plt.subplots_adjust(hspace=0.4)
		plt.rcParams.update({'font.size': 16})
		# Convert the 'Time' column to datetime
		self.df['Time'] = pd.to_datetime(self.df['Time'])
		# Get the minimum and maximum time
		min_time = self.df['Time'].min()
		max_time = self.df['Time'].max()

		for index_target, row_target in self.target_df.iterrows():
			# Filter the data for the current time range
			line, = ax.plot(self.df['Time'], self.df['angDistSun_'+row_target["Name"]]*self.RAD2DEG, label='Sun_'+row_target["Name"], linestyle='-')
			ax.plot(self.df['Time'], self.df['angDistMoon_'+row_target["Name"]]*self.RAD2DEG, label='Moon_'+row_target["Name"], linestyle=':',color=line.get_color())
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
		ax.xaxis.set_major_locator(mdates.DayLocator(interval=80))
		ax.axhline(y=self.param["sun_separation_angle"], color='r', linestyle='--', label='Elev. Sun')
		ax.axhline(y=self.param["moon_separation_angle"], color='r', linestyle='--', label='Elev. Moon')
		ax.set_yticks([0, 90, 180])
		ax.legend().set_visible(False)  # Hide the legend in the subplots
		# Create the legend in the separate axis
		ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1], bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.show()
		# Convert the 'Time' column back to string
		self.df['Time'] = self.df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

	def plot_3D_loc_target_vis(self):
		# 地球の半径を天文単位に変換（1天文単位＝約149597870.7キロメートル）
		earth_radius = 6378.137
		# プロットする矢印の大きさを調整するスケール
		scale = earth_radius/3

		# self.target_dfの行の数だけプロットを作成するように変更する
		n_plots = len(self.target_df)
		n_cols = 1
		n_rows = int(np.ceil(n_plots / n_cols))

		# プロットエリアのサイズを指定
		fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 16), subplot_kw={'projection': '3d'})
		axs = axs.ravel()  # Flatten the array for easy iterating

		# 地球を描画（半径は地球の半径、中心は原点）
		u = np.linspace(0, 2 * np.pi, 100)
		v = np.linspace(0, np.pi, 100)
		x = earth_radius * np.outer(np.cos(u), np.sin(v))
		y = earth_radius * np.outer(np.sin(u), np.sin(v))
		z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

		for ax, (_, row_target) in zip(axs, self.target_df.iterrows()):
			ax.plot_wireframe(x, y, z, color='lightblue', alpha=0.3)
			# print(row_target['Name'])
			ax.set_title(row_target['Name'])

			# 各行の時刻それぞれに対してループを回す
			for index, row in self.df.iterrows():
				# print('xTargVec_'+row_target['Name'])
				# print(self.df.loc[index, 'xTargVec_'+row_target['Name']])
				# 対象となる天体の方向ベクトルは、self.dfから取得する
				xVect = np.array([self.df.loc[index, 'xTargVec_'+row_target['Name']],
								  self.df.loc[index, 'yTargVec_'+row_target['Name']],
								  self.df.loc[index, 'zTargVec_'+row_target['Name']]])*scale

				# satellite location.
				satVect = np.array([self.df.loc[index, 'xJ2000'],
							 self.df.loc[index, 'yJ2000'],
							 self.df.loc[index, 'zJ2000']])
				# Sun vector
				sunVect = np.array([self.df.loc[index, 'xSunVect'],
					 self.df.loc[index, 'ySunVect'],
					 self.df.loc[index, 'zSunVect']])*scale
				# moon vector
				moonVect = np.array([self.df.loc[index, 'xMoonVect'],
					 self.df.loc[index, 'yMoonVect'],
					 self.df.loc[index, 'zMoonVect']])*scale

				# 衛星の位置をプロット # xVectを描画
				if self.df.loc[index, 'visEl_'+row_target["Name"]] > 0:
					ax.scatter(satVect[0], satVect[1], satVect[2], color='r')
					ax.quiver(satVect[0], satVect[1], satVect[2],
						  xVect[0], xVect[1], xVect[2], color='b')
					ax.quiver(satVect[0], satVect[1], satVect[2],
						  moonVect[0], moonVect[1], moonVect[2], color='y')
					ax.quiver(satVect[0], satVect[1], satVect[2],
						  sunVect[0], sunVect[1], sunVect[2], color='g')
				else:
					ax.scatter(satVect[0], satVect[1], satVect[2], color='k')
					ax.quiver(satVect[0], satVect[1], satVect[2],
						  xVect[0], xVect[1], xVect[2], color='b')
					ax.quiver(satVect[0], satVect[1], satVect[2],
						  moonVect[0], moonVect[1], moonVect[2], color='y')
					ax.quiver(satVect[0], satVect[1], satVect[2],
						  sunVect[0], sunVect[1], sunVect[2], color='g')

				ax.plot([0, satVect[0]], [0, satVect[1]], [0, satVect[2]], color='black')
				ax.set_title('Green:Sun, Yellow:Moon, Blue:' + row_target['Name'])

			ax.set_xlabel('X')
			ax.set_ylabel('Y')
			ax.set_zlabel('Z')

		plt.show()


	def plot_orbit_plane_target_vis_single_plot(self):
		# 地球の半径を天文単位に変換（1天文単位＝約149597870.7キロメートル）
		earth_radius = 6378.137
		# プロットする矢印の大きさを調整するスケール
		scale = earth_radius
		# プロットエリアのサイズを指定

		fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={'projection': '3d'})

		# 地球を描画（半径は地球の半径、中心は原点）
		u = np.linspace(0, 2 * np.pi, 100)
		v = np.linspace(0, np.pi, 100)
		x = earth_radius * np.outer(np.cos(u), np.sin(v))
		y = earth_radius * np.outer(np.sin(u), np.sin(v))
		z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
		ax.plot_wireframe(x, y, z, color='lightblue', alpha=0.3, rcount=10, ccount=10)

		counter = 0

		# 各行の時刻それぞれに対してループを回す
		for index, row in self.df.iterrows():
			counter += 1
			ax.set_xlabel('X')
			ax.set_ylabel('Y')
			ax.set_zlabel('Z')

			for (_, row_target) in self.target_df.iterrows():
				# 対象となる天体の方向ベクトルは、self.dfから取得する
				xVect = np.array([self.df.loc[index, 'xTargVec_'+row_target['Name']],
								  self.df.loc[index, 'yTargVec_'+row_target['Name']],
								  self.df.loc[index, 'zTargVec_'+row_target['Name']]])*scale
				ax.quiver(0, 0, 0, xVect[0], xVect[1], xVect[2], color='b')
				ax.text(xVect[0], xVect[1], xVect[2], row_target['Name'], color='b')

			# satellite location.
			satVect = np.array([self.df.loc[index, 'xJ2000'],
						 self.df.loc[index, 'yJ2000'],
						 self.df.loc[index, 'zJ2000']])
			# Sun vector
			sunVect = np.array([self.df.loc[index, 'xSunVect'],
				 self.df.loc[index, 'ySunVect'],
				 self.df.loc[index, 'zSunVect']])*scale
			# moon vector
			moonVect = np.array([self.df.loc[index, 'xMoonVect'],
				 self.df.loc[index, 'yMoonVect'],
				 self.df.loc[index, 'zMoonVect']])*scale

			# 衛星の位置をプロット # xVectを描画
			ax.scatter(satVect[0], satVect[1], satVect[2], color='k')

			# 2時間に1回だけ月太陽ベクトルをプロットする
			if counter % int(2*60/self.time_bin_min) == 1:
				ax.quiver(0, 0, 0,	moonVect[0], moonVect[1], moonVect[2], color='y')
				ax.quiver(0, 0, 0,	sunVect[0], sunVect[1], sunVect[2], color='g')

			# 一番最初に月太陽ベクトルのラベルをプロットする
			if counter == 1:
				ax.text(moonVect[0], moonVect[1], moonVect[2], 'Moon', color='y')
				ax.text(sunVect[0], sunVect[1], sunVect[2], 'Sun', color='g')

			# 1日に1回だけ日時をプロットする
			if counter % int(60*24/self.time_bin_min) == 1:
				ax.text(satVect[0], satVect[1], satVect[2], self.df.loc[index, 'Time'], color='k', fontsize=8)

			ax.set_xlim([-6500, 6500])
			ax.set_ylim([-6500, 6500])
			ax.set_zlim([-6500, 6500])

		plt.show()

	def plot_orbit_plane_target_vis_multiple_plot(self):
		# 地球の半径を天文単位に変換（1天文単位＝約149597870.7キロメートル）
		earth_radius = 6378.137
		# プロットする矢印の大きさを調整するスケール
		scale = earth_radius
		# プロットエリアのサイズを指定

		fig, axes = plt.subplots(4, 2, figsize=(18, 36), subplot_kw={'projection': '3d'})

		# elevationとazimuthの組み合わせ
		views = [
			(40, 150),
			(40, 120),
			(40, 60),
			(40, 30),
			(40, -30),
			(40, -60),
			(40, -120),
			(40, -150)
		]


		for ax, view in zip(axes.ravel(), views):
			elevation, azimuth = view
			ax.view_init(elev=elevation, azim=azimuth)

			# 地球を描画（半径は地球の半径、中心は原点）
			u = np.linspace(0, 2 * np.pi, 100)
			v = np.linspace(0, np.pi, 100)
			x = earth_radius * np.outer(np.cos(u), np.sin(v))
			y = earth_radius * np.outer(np.sin(u), np.sin(v))
			z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
			ax.plot_wireframe(x, y, z, color='lightblue', alpha=0.5, rcount=10, ccount=10)

			counter = 0

			# 各行の時刻それぞれに対してループを回す
			for index, row in self.df.iterrows():
				counter += 1
				ax.set_xlabel('X')
				ax.set_ylabel('Y')
				ax.set_zlabel('Z')

				for (_, row_target) in self.target_df.iterrows():
					# 対象となる天体の方向ベクトルは、self.dfから取得する
					xVect = np.array([self.df.loc[index, 'xTargVec_'+row_target['Name']],
									  self.df.loc[index, 'yTargVec_'+row_target['Name']],
									  self.df.loc[index, 'zTargVec_'+row_target['Name']]])*scale
					ax.quiver(0, 0, 0, xVect[0], xVect[1], xVect[2], color='b')
					ax.text(xVect[0], xVect[1], xVect[2], row_target['Name'], color='b')

				# satellite location.
				satVect = np.array([self.df.loc[index, 'xJ2000'],
							 self.df.loc[index, 'yJ2000'],
							 self.df.loc[index, 'zJ2000']])
				# Sun vector
				sunVect = np.array([self.df.loc[index, 'xSunVect'],
					 self.df.loc[index, 'ySunVect'],
					 self.df.loc[index, 'zSunVect']])*scale
				# moon vector
				moonVect = np.array([self.df.loc[index, 'xMoonVect'],
					 self.df.loc[index, 'yMoonVect'],
					 self.df.loc[index, 'zMoonVect']])*scale

				# 衛星の位置をプロット # xVectを描画
				ax.scatter(satVect[0], satVect[1], satVect[2], color='k')

				# 2時間に1回だけ月太陽ベクトルをプロットする
				if counter % int(2*60/self.time_bin_min) == 1:
					ax.quiver(0, 0, 0,	moonVect[0], moonVect[1], moonVect[2], color='y')
					ax.quiver(0, 0, 0,	sunVect[0], sunVect[1], sunVect[2], color='g')

				# 一番最初に月太陽ベクトルのラベルをプロットする
				if counter == 1:
					ax.text(moonVect[0], moonVect[1], moonVect[2], 'Moon', color='y')
					ax.text(sunVect[0], sunVect[1], sunVect[2], 'Sun', color='g')

				# 1日に1回だけ日時をプロットする
				if counter % int(60*24/self.time_bin_min) == 1:
					ax.text(satVect[0], satVect[1], satVect[2], self.df.loc[index, 'Time'], color='k', fontsize=8)

				ax.set_xlim([-6500, 6500])
				ax.set_ylim([-6500, 6500])
				ax.set_zlim([-6500, 6500])

		plt.tight_layout()
		# plt.show()
		plt.savefig("result/result_plot_orbit_plane_target_vis_multiple_plot.pdf", format="pdf")


	def add_satellite_mode(self):

		# operation mode, 0 not-difined, 1  communication, 2 charging, 3 observation
		self.df['operation_mode'] = 0
		# communication is priority
		self.df.loc[((self.df['gsFlag_Lithuania'] == 1) | (self.df['gsFlag_Los Angeles'] == 1)) & (self.df['operation_mode'] == 0), 'operation_mode'] = 1
		# chargint is semi-priority
		self.df.loc[(self.df['hMaxiFlag'] == 1) & (self.df['visFlag_Sun'] == 0) & (self.df['operation_mode'] == 0), 'operation_mode'] = 2

		orbits = self.df['Orbit'].unique()
		result_list = []

		for orbit in orbits:
			df_part = self.df[self.df['Orbit'] == orbit]
			count_orbit = df_part.shape[0]*self.time_bin_min
			count_nondefined = df_part[df_part['operation_mode'] == 0].shape[0]*self.time_bin_min
			count_comm = df_part[df_part['operation_mode'] == 1].shape[0]*self.time_bin_min
			count_charging = df_part[df_part['operation_mode'] == 2].shape[0]*self.time_bin_min
			result_list.append([count_orbit,count_nondefined, count_comm, count_charging])

		# 結果をDataFrameに変換
		df_results = pd.DataFrame(result_list, columns=['orbit[min]', 'not-defined[min]', 'comm[min]', 'charge[min]'], index=orbits)

		###########################################
		##### adjust satellite charging mode ######
		###########################################
		# 各周回の48%の時間を計算
		df_results['orbit_48perc'] = df_results['orbit[min]'] * 0.48
		# それが充電時間に足りていない場合の差分を計算
		df_results['charge_diff'] = df_results['orbit_48perc'] - df_results['charge[min]']
		# 充電時間が48%以上の場合は、その差分は0とする
		df_results['charge_diff'] = df_results['charge_diff'].apply(lambda x: max(0, x))
		df_results['additional_charge_bins'] = df_results['charge_diff'].apply(lambda x: math.ceil(x / self.time_bin_min))

		for orbit in orbits:
			df_part = self.df[self.df['Orbit'] == orbit].copy()
			counts_required = int(df_results.loc[orbit, 'additional_charge_bins'])
			# operation_modeが2であるインデックスを取得し、その連続性を確認
			counts_required = df_results.loc[orbit, 'additional_charge_bins']
			is_charging = df_part['operation_mode'] == 2
			# operation_modeが変化する直後のインデックスだけを取得
			switch_points = is_charging.diff()[is_charging.diff() != 0].index
			# switch_pointsのすぐ後ろのインデックスが0のインデックスを取得
			switch_points_after = [point for point in switch_points
							if point + counts_required in df_part.index
							and all(df_part.loc[point+1:point + counts_required, 'operation_mode'] == 0)]

			print(switch_points_after)

			if len(switch_points_after) > 0:
				# 追加の充電が必要なbin数だけ、既存の充電モードに連続してoperation_modeを加える
				for index in range(switch_points_after[0], switch_points_after[0] + counts_required):
					df_part.loc[index, 'operation_mode'] = 2
				# 元のDataFrameに反映
				self.df.loc[self.df['Orbit'] == orbit] = df_part
			else:
				pass

		######################################
		##### add final result dataframe #####
		######################################
		result_list = []
		for orbit in orbits:
			df_part = self.df[self.df['Orbit'] == orbit]
			count_orbit = df_part.shape[0]*self.time_bin_min
			count_nondefined = df_part[df_part['operation_mode'] == 0].shape[0]*self.time_bin_min
			count_comm = df_part[df_part['operation_mode'] == 1].shape[0]*self.time_bin_min
			count_charging = df_part[df_part['operation_mode'] == 2].shape[0]*self.time_bin_min
			result_list.append([count_orbit,count_nondefined, count_comm, count_charging])
		self.result_df = pd.DataFrame(result_list, columns=['orbit[min]', 'not-defined[min]', 'comm[min]', 'charge[min]'], index=orbits)
		self.result_df['orbit_48perc'] = self.result_df['orbit[min]'] * 0.48
		# それが充電時間に足りていない場合の差分を計算
		self.result_df['charge_diff'] = self.result_df['orbit_48perc'] - self.result_df['charge[min]']
		# 各Orbitの値に対してループを回す
		for orbit in orbits:
			df_part = self.df[self.df['Orbit'] == orbit]
			for index_target, row_target in self.target_df.iterrows():
				selec_df = df_part.loc[(df_part['obsFlag_'+row_target["Name"]] == 1) & (df_part['operation_mode'] == 0), :]
				self.result_df.loc[orbit, 'obsFlag_'+row_target["Name"]] = selec_df.shape[0]*self.time_bin_min

		df_part = self.df[self.df['Orbit'] == 5]
		print(df_part)
		for index_target, row_target in self.target_df.iterrows():
			selec_df = df_part.loc[(df_part['obsFlag_'+row_target["Name"]] == 1) & (df_part['operation_mode'] == 0), :]
			print(selec_df)
			self.result_df.loc[orbit, 'obsFlag_'+row_target["Name"]] = selec_df.shape[0]*self.time_bin_min


	def plot_one_orbit(self, orb_num):
		df_part = self.df[self.df['Orbit'] == orb_num].reset_index()

		plt.rcParams['font.size'] = 16
		# 時刻列を日時型に変換
		df_part.loc[:, 'Time'] = pd.to_datetime(df_part['Time'])
		# Figureオブジェクトを作成
		fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, sharex=True, figsize=(12, 15), gridspec_kw={'hspace': 0})

		# ax1: hMaxi and zMaxi
		ax1.plot(df_part['Time'], df_part['hMaxi'], label='hMaxi', marker='x', markersize=12)
		ax1.plot(df_part['Time'], df_part['zMaxi'], label='zMaxi')
		ax1.set_yscale('log')
		ax1.set_ylabel('RBM counts s$^{-1}$')
		ax1.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
		ax12 = ax1.twinx()
		ax12.plot(df_part['Time'], 1 - df_part['visFlag_Sun'], label='Vis Flag Sun')
		ax12.legend(bbox_to_anchor=(1.05, 0.3), loc='upper left')
		ax12.set_yticks([0, 1])
		ax12.set_ylim([-0.2, 1.2])
		ax12.set_yticklabels(['Invisible', 'Visible'])

		# ax2: visFlag with Sun Moon Separation Angle

		offsets = np.linspace(0.92, 1.08, num=self.target_df.shape[0])
		for offset, (index_target, row_target) in zip(offsets, self.target_df.iterrows()):
			line, = ax2.step(df_part['Time'], (2-df_part['visFlag_' + row_target["Name"]]) * offset, label=row_target['Name'],
					 linestyle='--') # 2-visFlag で「2 visible 1 dark earth 0 bright earth」となる
			color = line.get_color()
			mask = (df_part['angDistFlagMoon_' + row_target["Name"]] == 0) | (df_part['angDistFlagSun_' + row_target["Name"]] == 0)
			ax2.plot(df_part['Time'][mask], (2-df_part['visFlag_' + row_target["Name"]])[mask] * offset, 'x', color=color, markersize=12)
		ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		ax2.set_yticks([0, 1, 2])
		ax2.set_ylim([-0.2, 2.2])
		ax2.set_yticklabels(['Bright Earth', 'Dark Earth', 'Visible'])


		# ax3: obsFlag
		offsets = np.linspace(0.92, 1.08, num=self.target_df.shape[0])
		for offset, (index_target, row_target) in zip(offsets, self.target_df.iterrows()):
			ax3.step(df_part['Time'], df_part['obsFlag_' + row_target["Name"]] * offset, label=row_target['Name'],
					 linestyle='--')
		ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		ax3.set_yticks([0, 1])
		ax3.set_ylim([-0.2, 1.2])
		ax3.set_yticklabels(['Unobservable', 'Observable'])

		# ax4: gsFlag
		offsets = np.linspace(0.95, 1.05, num=self.gs_df.shape[0])
		for offset, (index_gs, row_gs) in zip(offsets, self.gs_df.iterrows()):
			ax4.step(df_part['Time'], df_part['gsFlag_' + row_gs["Name"]] * offset, label=row_gs['Name'], linestyle='-')
		ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		ax4.set_yticks([0, 1])
		ax4.set_ylim([-0.2, 1.2])
		ax4.set_yticklabels(['Invisible', 'Visible'])

		# ax5: operation_mode and Latitude
		ax5.step(df_part['Time'], df_part['operation_mode'], label='operation_mode')
		ax5.set_yticks([0, 1, 2])
		ax5.set_yticklabels(['Obs.', 'Comm.', 'Chg.'])
		ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

		# 均等な間隔で日時を生成（この例では6つのポイントを生成）
		start_time = df_part['Time'].min()
		end_time = df_part['Time'].max()
		number_of_ticks = 6
		time_ticks = pd.date_range(start=start_time, end=end_time, periods=number_of_ticks)
		# ax5にxticksを設定
		ax5.set_xticks(time_ticks)
		ax5.set_xticklabels(time_ticks.strftime('%Y-%m-%d\n%H:%M'))
		ax5.tick_params(axis='x', rotation=60)


		ax5.set_xlabel('Time')
		plt.tight_layout()
		# グラフ表示
		# plt.show()
		plt.savefig("result/result_plot_one_orbit.pdf", format="pdf")



	def plot_maxi_rbm_map_only(self):
		fig = plt.figure(figsize=(12,8))
		plt.clf()
		ax = plt.axes(projection=ccrs.PlateCarree())
		# map rbm values
		im = ax.imshow(self.maxi_rbm_map_h.T,
			origin="lower",extent=[-180, 180, -90, 90],
			transform=ccrs.PlateCarree(), alpha=0.4, norm=LogNorm(), cmap='Greys')
		cbar=plt.colorbar(im, shrink=0.5, pad=0.1)
		cbar.set_label("RBM counts s$^{-1}$ [%s]" % self.param['maxi_rbm_selected_basename_h'])
		# contour rbm levels
		contours=ax.contour(np.where(self.maxi_rbm_map_h.T <= 0, np.nan, self.maxi_rbm_map_h.T), origin="lower"
				,levels=[0.1, 1, 10, 100, 1000, 10000],	norm=LogNorm(), extent=[-180, 180, -90, 90], cmap='Greys')
		ax.clabel(contours, inline=1, fontsize=8)
		ax.set_global()
		ax.coastlines(linewidth=0.5)
		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
			linewidth=1, color='gray', alpha=0.3, linestyle='--')
		ax.set_xlim(-180.0,180.0)
		ax.set_ylim(-90.0,90.0)
		ax.set_xlabel('Longitude (deg), [{}], threshold={}'.format(self.param['maxi_rbm_selected_basename_h'], self.param['hv_allowed_maxi_rbm_threshold']))
		ax.set_ylabel('Latitude (deg)')
		ax.set_xticks([-180,-120,-60,0,60,120,180])
		ax.set_yticks([-90,-60,-30,0,30,60,90])


	def at_earth_occult(self, sat_vect, x_vect, sun_vect):
		sat_v, earth_vect, x_v = None, None, None
		earth_vect = -sat_vect
		sat_distance = self.norm(earth_vect)
		earth_size = np.arcsin(self.EARTH_RADIUS / sat_distance)
		x_dist = self.ang_distance(x_vect, earth_vect)
		el = x_dist - earth_size

		flag = 0
		if el <= -self.EPS:
			rm = self.set_rot_mat_zx(sun_vect, sat_vect)
			sat_v = self.rot_vect(rm, sat_vect)
			x_v = self.rot_vect(rm, x_vect)
			dot = self.scal_prod(earth_vect, x_vect)
			z_cross = sat_v[2] + x_v[2] * (dot
				- np.sqrt(self.EARTH_RADIUS * self.EARTH_RADIUS
				- sat_distance * sat_distance + dot * dot))
			if z_cross < 0.:
				flag = 1  # Dark Earth
			else:
				flag = 2  # Bright Earth
		return flag, el, x_dist


	def get_direction_vector(self, lon_rad, lat_rad):

		# XYZ座標を計算
		# ra ; 赤経 ; 経度 ; long
		# dec ;　赤緯 ; 緯度 ; lat
		x = np.cos(lat_rad) * np.cos(lon_rad)
		y = np.cos(lat_rad) * np.sin(lon_rad)
		z = np.sin(lat_rad)

		# 方向ベクトルを返す
		return np.array([x, y, z])


	def add_observer_vis(self):
		for index, row in self.gs_df.iterrows():
			for index2, row2 in self.df.iterrows():
				datetime_value = self.str2datetime(row2["Time"])
				st_azi_deg, st_ele_deg = self.orbital_orbit.get_observer_look(datetime_value, row['long'], row['lat'], row['alt'])
				self.df.loc[index2, 'gsElev_'+row["Name"]] = st_ele_deg
				self.df.loc[index2, 'gsAzim_'+row["Name"]] = st_azi_deg
				self.df.loc[index2, 'gsFlag_'+row["Name"]] = st_ele_deg>self.param['sight_elevation']

	def plot_observer_vis(self):
		fig = plt.figure(figsize=(12,8))
		plt.clf()
		ax = plt.axes(projection=ccrs.PlateCarree())

		# Loop for dataframe, time series
		for index2, row2 in self.df.iterrows():
			# total index flags, 0 means no communication with any ground station, 1 means communication with a ground station
			obs_vis_ind = 0
			# Loop for ground stations, location series
			for index, row in self.gs_df.iterrows():

				if self.df.loc[index2, 'gsFlag_'+row["Name"]] == 1:
					obs_vis_ind = 1
					# a ground station has been found, plot the location as a communication point
					break
				else:
					obs_vis_ind = 0

			if obs_vis_ind == 1:
				ax.scatter(self.df.loc[index2, 'Longitude'], self.df.loc[index2, 'Latitude'],
					transform=ccrs.PlateCarree(), marker='x', s=20, c="r")
			else:
				ax.scatter(self.df.loc[index2, 'Longitude'], self.df.loc[index2, 'Latitude'],
					transform=ccrs.PlateCarree(), marker='x', s=20, c="b")

		# add text label on each location
		for i in range(len(self.df)):
			ax.text(self.df['Longitude'][i], self.df['Latitude'][i], self.df['Time'][i], fontsize=8)

		# plot ground station location, adustment of "+2" for avoiding overlap
		for index_gs, row_gs in self.gs_df.iterrows():
			ax.scatter(row_gs['long'], row_gs['lat'],
				transform=ccrs.PlateCarree(), marker='D', s=30, c="m")
			ax.text(row_gs['long']+2, row_gs['lat']+2, row_gs['Name'], transform=ccrs.PlateCarree(), fontsize=10, c="m")

		ax.set_global()
		ax.coastlines(linewidth=0.5)
		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
			linewidth=1, color='gray', alpha=0.3, linestyle='--')
		ax.set_xlim(-180.0,180.0)
		ax.set_ylim(-90.0,90.0)
		ax.set_xlabel('Longitude (deg)')
		ax.set_ylabel('Latitude (deg)')
		ax.set_xticks([-180,-120,-60,0,60,120,180])
		ax.set_yticks([-90,-60,-30,0,30,60,90])
		plt.savefig("result/result_plot_observer_vis.pdf", format="pdf")



	def setup_maxi_rbm_index(self):

		maxi_rbm_image_list = []
		with open(self.param['maxi_rbm_pic_directory'], 'rb') as maxi:
			maxi_rbm_image_lst = pickle.load(maxi)

		self.maxi_rbm_map_h = maxi_rbm_image_lst[self.param['maxi_rbm_selected_basename_h']]
		self.maxi_rbm_map_h = np.nan_to_num(self.maxi_rbm_map_h, nan=1e5)
		maxi_rbm_map_index_bool_h = self.maxi_rbm_map_h > self.param['hv_allowed_maxi_rbm_threshold']
		self.maxi_rbm_map_index_h = maxi_rbm_map_index_bool_h.astype(int)

		self.maxi_rbm_map_z = maxi_rbm_image_lst[self.param['maxi_rbm_selected_basename_z']]
		self.maxi_rbm_map_z = np.nan_to_num(self.maxi_rbm_map_z, nan=1e5)
		maxi_rbm_map_index_bool_z = self.maxi_rbm_map_z > self.param['hv_allowed_maxi_rbm_threshold']
		self.maxi_rbm_map_index_z = maxi_rbm_map_index_bool_z.astype(int)

	def add_maxi_rbm_index(self):
		for index, row in self.df.iterrows():
			datetime_value = self.str2datetime(row["Time"])

			theta = np.linspace(-180, 180, 1441) # longitude
			phi = np.linspace(-90,90, 481) # lattitude

			long=self.orbital_orbit.get_lonlatalt(datetime_value)[0]
			lat=self.orbital_orbit.get_lonlatalt(datetime_value)[1]

			index_phi = np.digitize(lat, phi)
			index_theta = np.digitize(long, theta)
			maxi_rbm_h = self.maxi_rbm_map_h.T[index_phi][index_theta]
			maxi_rbm_z = self.maxi_rbm_map_z.T[index_phi][index_theta]

			maxi_rbm_index_h = self.maxi_rbm_map_index_h.T[index_phi][index_theta] # np.array[row][column]
			maxi_rbm_index_z = self.maxi_rbm_map_index_z.T[index_phi][index_theta] # = np.array[long_ind][phi_ind]

			self.df.loc[index, 'hMaxi'] = maxi_rbm_h
			self.df.loc[index, 'zMaxi'] = maxi_rbm_z
			self.df.loc[index, 'hMaxiFlag'] = maxi_rbm_index_h
			self.df.loc[index, 'zMaxiFlag'] = maxi_rbm_index_z

	def plot_maxi_rbm_flag(self):
		fig = plt.figure(figsize=(12,8))
		plt.clf()
		ax = plt.axes(projection=ccrs.PlateCarree())
		im = ax.imshow(self.maxi_rbm_map_index_h.T,
			origin="lower",extent=[-180, 180, -90, 90],
			transform=ccrs.PlateCarree(), cmap='Greys',	alpha=0.4)
		cbar=plt.colorbar(im, shrink=0.5, pad=0.1)
		cbar.set_label("RBM counts s$^{-1}$ [%s]" % self.param['maxi_rbm_selected_basename_h'])

		ax.scatter(self.df['Longitude'], self.df['Latitude'],transform=ccrs.PlateCarree(), marker='x', s=20, c="r")
		for i in range(len(self.df)):
			ax.text(self.df['Longitude'][i], self.df['Latitude'][i], self.df['Time'][i], fontsize=8)

		ax.set_global()
		ax.coastlines(linewidth=0.5)
		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
			linewidth=1, color='gray', alpha=0.3, linestyle='--')
		ax.set_xlim(-180.0,180.0)
		ax.set_ylim(-90.0,90.0)
		ax.set_xlabel('Longitude (deg), [{}], threshold={}'.format(self.param['maxi_rbm_selected_basename_h'], self.param['hv_allowed_maxi_rbm_threshold']))
		ax.set_ylabel('Latitude (deg)')
		ax.set_xticks([-180,-120,-60,0,60,120,180])
		ax.set_yticks([-90,-60,-30,0,30,60,90])
		plt.savefig("result/result_plot_maxi_rbm_flag.pdf", format="pdf")

	def plot_maxi_rbm_map(self):
		fig = plt.figure(figsize=(12,8))
		plt.clf()
		ax = plt.axes(projection=ccrs.PlateCarree())
		# map rbm values
		im = ax.imshow(self.maxi_rbm_map_h.T,
			origin="lower",extent=[-180, 180, -90, 90],
			transform=ccrs.PlateCarree(), alpha=0.4, norm=LogNorm(), cmap='Greys')
		cbar=plt.colorbar(im, shrink=0.5, pad=0.1)
		cbar.set_label("RBM counts s$^{-1}$ [%s]" % self.param['maxi_rbm_selected_basename_h'])
		# contour rbm levels
		contours=ax.contour(np.where(self.maxi_rbm_map_h.T <= 0, np.nan, self.maxi_rbm_map_h.T), origin="lower"
				,levels=[0.1, 1, 10, 100, 1000, 10000],	norm=LogNorm(), extent=[-180, 180, -90, 90], cmap='Greys')
		ax.clabel(contours, inline=1, fontsize=8)
		# scatter satellite location
		ax.scatter(self.df['Longitude'], self.df['Latitude'],transform=ccrs.PlateCarree(), marker='x', s=20, c="r")
		for i in range(len(self.df)):
			ax.text(self.df['Longitude'][i], self.df['Latitude'][i], self.df['Time'][i], fontsize=8)
		ax.set_global()
		ax.coastlines(linewidth=0.5)
		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
			linewidth=1, color='gray', alpha=0.3, linestyle='--')
		ax.set_xlim(-180.0,180.0)
		ax.set_ylim(-90.0,90.0)
		ax.set_xlabel('Longitude (deg), [{}], threshold={}'.format(self.param['maxi_rbm_selected_basename_h'], self.param['hv_allowed_maxi_rbm_threshold']))
		ax.set_ylabel('Latitude (deg)')
		ax.set_xticks([-180,-120,-60,0,60,120,180])
		ax.set_yticks([-90,-60,-30,0,30,60,90])
		plt.savefig("result/result_plot_maxi_rbm_map.pdf", format="pdf")

	def set_cutoff_rigidity(self):
		print("--set_cutoff_rigidity_map")
		print('file_cutoffrigidity: {}'.format(self.param['file_cutoffrigidity']))

		# file_cutoffrigidityは、yamlファイルからのインプット

		# 新しいインスタンス変数、COR = Cut Off Rigidity
		self.df_cor = pd.read_csv(self.param['file_cutoffrigidity'],
			skiprows=1,delim_whitespace=True,
			names=['latitude_deg','longitude_deg','cutoff_rigidity'],
			dtype={'latitude_deg':'float64','longitude_deg':'float64','cutoff_rigidity':'float64'})
		# print("df_cor")
		# print(self.df_cor)

		# 新しいインスタンス変数
		self.cutoff_rigidity_map, self.cormap_longitude_edges, self.cormap_latitude_edges = np.histogram2d([],[],
			bins=[self.param["cormap_lon_nbin"],self.param["cormap_lat_nbin"]],
			range=[[-180.,180.],[-90.,90.]])
		self.df_cor["longitude_index"] = np.digitize(self.df_cor['longitude_deg'], self.cormap_longitude_edges)
		self.df_cor["longitude_index"] = pd.to_numeric(self.df_cor["longitude_index"],downcast='signed')
		self.df_cor["latitude_index"] = np.digitize(self.df_cor['latitude_deg'], self.cormap_latitude_edges)
		self.df_cor["latitude_index"] = pd.to_numeric(self.df_cor["latitude_index"],downcast='signed')
		"""
		H: two dimentional matrix
		x: longitude
		y: latitude

		H[0][2]		...
		H[0][1]		H[1][1]		...
		H[0][0]		H[1][0]		...
		"""

		for index, row in self.df_cor.iterrows():
			i = int(row['longitude_index'])-1
			j = int(row['latitude_index'])-1
			self.cutoff_rigidity_map[i][j] = row['cutoff_rigidity']

		self.cutoff_rigidity_map_T = self.cutoff_rigidity_map.T
		self.cormap_longitude_centers = (self.cormap_longitude_edges[:-1] + self.cormap_longitude_edges[1:]) / 2.
		self.cormap_latitude_centers = (self.cormap_latitude_edges[:-1] + self.cormap_latitude_edges[1:]) / 2.

	def add_cutoff_rigidity(self):
		print("get_cutoff_rigidity")
		for index, row in self.df.iterrows():
			datetime_value = self.str2datetime(row["Time"])

			long = self.orbital_orbit.get_lonlatalt(datetime_value)[0]
			lat = self.orbital_orbit.get_lonlatalt(datetime_value)[1]

			i = int(np.digitize(long, self.cormap_longitude_edges)) - 1
			j = int(np.digitize(lat, self.cormap_latitude_edges)) - 1

			self.df.loc[index, 'COR'] = self.cutoff_rigidity_map[i][j]


	def plot_cutoff_rigidity_map(self,foutname_base='cormap'):
		print("plot_cutoff_rigidity_map")

		fig = plt.figure(figsize=(12,8))
		ax = plt.axes(projection=ccrs.PlateCarree())
		ax.stock_img()
		ax.coastlines(resolution='110m')

		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
			linewidth=1, color='gray', alpha=0.3, linestyle='--')
		ax.set_xlim(-180.0,180.0)
		ax.set_ylim(-90.0,90.0)
		ax.set_xlabel('Longitude (deg)')
		ax.set_ylabel('Latitude (deg)')
		ax.set_xticks([-180,-120,-60,0,60,120,180])
		ax.set_yticks([-90,-60,-30,0,30,60,90])
		X, Y = np.meshgrid(self.cormap_longitude_edges, self.cormap_latitude_edges)
		# scatter satellite location
		ax.scatter(self.df['Longitude'], self.df['Latitude'],transform=ccrs.PlateCarree(), marker='x', s=20, c="r")
		for i in range(len(self.df)):
			ax.text(self.df['Longitude'][i], self.df['Latitude'][i], self.df['Time'][i], fontsize=8)
		ax.pcolormesh(X, Y, self.cutoff_rigidity_map_T, alpha=0.5)
		cormap_contour = ax.contour(
			self.cormap_longitude_centers,
			self.cormap_latitude_centers,
			self.cutoff_rigidity_map_T,
			levels=10, colors='Black',
			transform=ccrs.PlateCarree())
		cormap_contour.clabel(fmt='%1.1f', fontsize=12)
		plt.savefig("result/result_plot_cutoff_rigidity_map.pdf", format="pdf")




	# def set_hvoff_region(self):
	# 	print("set_hvoff_region")
	# 	# https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
	# 	open(self.param["lookuptable_hvoff"])



	def datetime2mjd(self,time_datetime):
		julian_day = time_datetime.toordinal() + 1721425.5
		mjd = julian_day - 2400000.5
		return mjd

	def datetime2jd(self,time_datetime):
		julian_day = time_datetime.toordinal() + 1721425.5
		return julian_day

	def str2datetime(self, time_str):
		"""
		Assume time string such as '2021-08-08 23:59:00', and return as datetime object
		"""
		time_datetime = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
		return time_datetime

	def datetime2str(self, time_datetime):
		"""
		Assume time in datetime such as '2021-08-08 23:59:00', and return as string
		"""
		time_str = time_datetime.strftime('%Y-%m-%d %H:%M:%S')
		return time_str

	def get_position(self,utc_time):
		# print("get_position")
		"""
		return satellite position: longitude (deg), latitude (deg), and altitude (km)
		"""
		return self.orbital_orbit.get_lonlatalt(utc_time)



	def norm(self, vector):
		return np.linalg.norm(vector)

	def norm_vect(self, vector):
		norm = np.linalg.norm(vector)
		if norm == 0:
			return None
		return vector / norm

	def set_rot_mat_zx(self, z_axis, x_axis):
		y_axis = self.vect_prod(z_axis, x_axis)
		z = self.norm_vect(z_axis)
		y = self.norm_vect(y_axis)
		if z is None or y is None:
			return None
		x = self.vect_prod(y, z)

		rm = np.array([x, y, z])
		return rm

	def ang_distance(self, vector1, vector2):
		return np.arccos(np.dot(vector1, vector2) / (self.norm(vector1) * self.norm(vector2)))

	def vect_prod(self, vector1, vector2):
		return np.cross(vector1, vector2)

	def rot_vect(self, matrix, vector):
		return matrix.dot(vector)

	def scal_prod(self, vector1, vector2):
		return np.dot(vector1, vector2)

	def angtime2radians(self, time_str, ra=True):
		t = [int(i) for i in time_str.split(':')]

		if ra: # if the string is RA
			total_degrees = 15 * (t[0] + t[1]/60 + t[2]/3600)
		else: # if the string is DEC
			sign = -1 if time_str[0] == '-' else 1
			total_degrees = sign * (abs(t[0]) + t[1]/60 + t[2]/3600)
		return math.radians(total_degrees)


	def atPrecession(self, mjd0, x0, mjd):
		rm = np.zeros((3, 3))
		rm = self.atPrecessRM(mjd0, mjd)
		x = np.dot(rm, x0)
		return x

	def atPrecessRM(self, mjd0, mjd):
		RmAto2000 = np.zeros((3, 3))
		RmBto2000 = np.zeros((3, 3))
		Rm2000toB = np.zeros((3, 3))
		rm = np.zeros((3, 3))

		RmAto2000 = self.atPrecessRMJ2000(mjd0)
		RmBto2000 = self.atPrecessRMJ2000(mjd)
		Rm2000toB = np.transpose(RmBto2000)
		rm = np.dot(RmAto2000, Rm2000toB)

		return rm



	def atPrecessRMJ2000(self, mjd):
		t = (mjd - self.MJD_J2000) / 36525.0

		zeta = (2306.2181 + (0.30188 + 0.017998*t)*t)*t * self.ARCSEC2RAD
		z = (2306.2181 + (1.09468 + 0.018203*t)*t)*t * self.ARCSEC2RAD
		theta = (2004.3109 - (0.42665 + 0.041833*t)*t)*t * self.ARCSEC2RAD

		cos_zeta = math.cos(zeta)
		sin_zeta = math.sin(zeta)

		cos_z = math.cos(z)
		sin_z = math.sin(z)

		cos_theta = math.cos(theta)
		sin_theta = math.sin(theta)

		rm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
		rm[0][0] = cos_zeta*cos_theta*cos_z - sin_zeta*sin_z
		rm[1][0] = -sin_zeta*cos_theta*cos_z - cos_zeta*sin_z
		rm[2][0] = -sin_theta*cos_z
		rm[0][1] = cos_zeta*cos_theta*sin_z + sin_zeta*cos_z
		rm[1][1] = -sin_zeta*cos_theta*sin_z + cos_zeta*cos_z
		rm[2][1] = -sin_theta*sin_z
		rm[0][2] = cos_zeta*sin_theta
		rm[1][2] = -sin_zeta*sin_theta
		rm[2][2] = cos_theta

		return rm

	def atSun(self, mjd):
		MJD_B1950 = 33281.923
		DEG2RAD = np.pi / 180.0

		rm1950_2000 = self.atPrecessRMJ2000(MJD_B1950)

		t = mjd - 4.5e4
		m = ( np.fmod(t * .985600267, 360.0) + 27.26464 ) * DEG2RAD
		sin_2m = np.sin(2*m)

		l = ( np.fmod(t * .985609104, 360.0) - 50.55138
			  + np.sin(m) * 1.91553 + sin_2m * .0201 ) * DEG2RAD
		sin_l = np.sin(l)

		r = 1.00014 - np.cos(m) * .01672 - sin_2m * 1.4e-4

		x = np.array([r * np.cos(l), r * .91744 * sin_l, r * .39788 * sin_l])

		pos = np.dot(rm1950_2000, x)
		return pos

	def atMoon(self, mjd):  # input: time in MJD
		ta, a, b, c, d, e, g, j, l, m, n, v, w = self.moonag(mjd)
		mx, my, mz = self.moonth(ta, a, b, c, d, e, g, j, l, m, n, v, w)

		r_xy = math.sqrt(my - mx * mx)
		sin_delta = mz / r_xy
		cos_delta = math.sqrt(1. - sin_delta * sin_delta)
		sin_c = math.sin(c)
		cos_c = math.cos(c)

		# R.A. of moon = mean longitude (c) + delta
		distan = self.EARTH_RADIUS * math.sqrt(my)
		x_tod = [0, 0, 0]
		x_tod[0] = self.EARTH_RADIUS * r_xy * (cos_delta * cos_c - sin_delta * sin_c)
		x_tod[1] = self.EARTH_RADIUS * r_xy * (sin_delta * cos_c + cos_delta * sin_c)
		x_tod[2] = self.EARTH_RADIUS * mx

		size = math.atan(self.MOON_RADIUS / distan)
		phase = d % (np.pi * 2)
		pos = self.atPrecession(mjd, x_tod, self.MJD_J2000)

		return pos, size, phase, distan  # return as a tuple

	def moonag(self, mjd):
		DEG2RAD = np.pi / 180.0
		ta = (mjd - 15019.5) / 36525.
		tb = ta * ta

		a = DEG2RAD*(ta * 4.77e5 +296.1044608 + ta * 198.849108 + tb * .009192)
		b = DEG2RAD*(ta * 483120. + 11.250889 + ta * 82.02515 - tb * .003211)
		c = DEG2RAD*(ta * 480960. +270.434164 + ta * 307.883142 - tb * .001133)
		d = DEG2RAD*(ta * 444960 + 350.737486 + ta * 307.114217 - tb * .001436)
		e = DEG2RAD*(ta * 35640 + 98.998753 + ta * 359.372886)
		g = DEG2RAD*(ta * 35999.04975 + 358.475833 - tb * 1.5e-4)
		j = DEG2RAD*(ta * 2880 + 225.444651 + ta * 154.906654)
		l = DEG2RAD*(ta * 36000.76892 + 279.696678 + tb * 3.03e-4)
		m = DEG2RAD*(ta * 19080 + 319.529425 + ta * 59.8585 + tb * 1.81e-4)
		n = DEG2RAD*(259.183275 - ta * 1800 - ta * 134.142008 + tb * .002078)
		v = DEG2RAD*(ta * 58320 + 212.603219 + ta * 197.803875 + tb * .001286)
		w = DEG2RAD*(ta * 58320 + 342.767053 + ta * 199.211911 * 3.1e-4 * tb)

		return ta, a, b, c, d, e, g, j, l, m, n, v, w

	def moonth(self, ta, a, b, c, d, e, g, j, l, m, n, v, w):
		# MOON THETA
		mx = math.sin(a + b - d * 4.0) * -0.00101
		mx -= math.sin(a - b - d * 4.0 - n) * 0.00102
		mx -= ta * 0.00103 * math.sin(a - b - n)
		mx -= math.sin(a - g - b - d * 2.0 - n) * 0.00107
		mx -= math.sin(a * 2.0 - b - d * 4.0 - n) * 0.00121
		mx += math.sin(a * 3.0 + b + n) * 0.0013
		mx -= math.sin(a + b - n) * 0.00131
		mx += math.sin(a + b - d + n) * 0.00136
		mx -= math.sin(g + b) * 0.00145
		mx -= math.sin(a + g - b - d * 2.0) * 0.00149
		mx += math.sin(g - b + d - n) * 0.00157
		mx -= math.sin(g - b) * 0.00159
		mx += math.sin(a - g + b - d * 2.0 + n) * 0.00184
		mx -= math.sin(b - d * 2.0 - n) * 0.00194
		mx -= math.sin(g - b + d * 2.0 - n) * 0.00196
		mx += math.sin(b - d) * 0.002
		mx -= math.sin(a + g - b) * 0.00205
		mx += math.sin(a - g - b) * 0.00235
		mx += math.sin(a - b * 3 - n) * 0.00246
		mx -= math.sin(a * 2 + b - d * 2.0) * 0.00262
		mx -= math.sin(a + g + b - d * 2.0) * 0.00283
		mx -= math.sin(g - b - d * 2.0 - n) * 0.00339
		mx += math.sin(a - b + n) * 0.00345
		mx -= math.sin(g - b + d * 2.0) * 0.00347
		mx -= math.sin(b + d + n) * 0.00383
		mx -= math.sin(a + g + b + n) * 0.00411
		mx -= math.sin(a * 2 - b - d * 2.0 - n) * 0.00442
		mx += math.sin(a - b + d * 2.0) * 0.00449
		mx -= math.sin(b * 3 - d * 2.0 + n) * 0.00456
		mx += math.sin(a + b + d * 2.0 + n) * 0.00466
		mx += math.sin(a * 2 - b) * 0.0049
		mx += math.sin(a * 2 + b) * 0.00561
		mx += math.sin(a - g + b + n) * 0.00564
		mx -= math.sin(a + g - b - n) * 0.00638
		mx -= math.sin(a + g - b - d * 2.0 - n) * 0.00713
		mx -= math.sin(g + b - d * 2.0) * 0.00929
		mx -= math.sin(a * 2 - b - n) * 0.00947
		mx += math.sin(a - g - b - n) * 0.00965
		mx += math.sin(b + d * 2.0) * 0.0097
		mx += math.sin(b - d + n) * 0.01064
		mx -= ta * 0.0125 * math.sin(b + n)
		mx -= math.sin(g + b - d * 2.0 + n) * 0.01434
		mx -= math.sin(a + g + b - d * 2.0 + n) * 0.01652
		mx -= math.sin(a * 2 + b - d * 2.0 + n) * 0.01868
		mx += math.sin(a * 2 + b + n) * 0.027
		mx -= math.sin(a - b - d * 2.0) * 0.02994
		mx -= math.sin(g + b + n) * 0.03759
		mx -= math.sin(g - b - n) * 0.03982
		mx += math.sin(b + d * 2.0 + n) * 0.04732
		mx -= math.sin(b - n) * 0.04771
		mx -= math.sin(a + b - d * 2.0) * 0.06505
		mx += math.sin(a + b) * 0.13622
		mx -= math.sin(a - b - d * 2.0 - n) * 0.14511
		mx -= math.sin(b - d * 2.0) * 0.18354
		mx -= math.sin(b - d * 2.0 + n) * 0.20017
		mx -= math.sin(a + b - d * 2.0 + n) * 0.38899
		mx += math.sin(a - b) * 0.40248
		mx += math.sin(a + b + n) * 0.65973
		mx += math.sin(a - b - n) * 1.96763
		mx += math.sin(b) * 4.95372
		mx += math.sin(b + n) * 23.89684

		# MOON RHO
		my = math.cos(a * 2 + g) * 0.05491
		my += math.cos(a + d) * 0.0629
		my -= math.cos(d * 4) * 0.06444
		my -= math.cos(a * 2 - g) * 0.06652
		my -= math.cos(g - d * 4) * 0.07369
		my += math.cos(a - d * 3) * 0.08119
		my -= math.cos(a + d * 4) * 0.09261
		my += math.cos(a - b * 2 + d * 2) * 0.10177
		my += math.cos(a + g + d * 2) * 0.10225
		my -= math.cos(a + g * 2 - d * 2) * 0.10243
		my -= math.cos(b * 2) * 0.12291
		my -= math.cos(a * 2 - b * 2) * 0.12291
		my -= math.cos(a + g - d * 4) * 0.12428
		my -= math.cos(a * 3) * 0.14986
		my -= math.cos(a - g + d * 2) * 0.1607
		my -= math.cos(a - d) * 0.16949
		my += math.cos(a + b * 2 - d * 2) * 0.17697
		my -= math.cos(a * 2 - d * 4) * 0.18815
		my -= math.cos(g * 2 - d * 2) * 0.19482
		my += math.cos(b * 2 - d * 2) * 0.22383
		my += math.cos(a * 3 - d * 2) * 0.22594
		my += math.cos(a * 2 + g - d * 2) * 0.24454
		my -= math.cos(g + d) * 0.31717
		my -= math.cos(a - d * 4) * 0.36333
		my += math.cos(a - g - d * 2) * 0.47999
		my += math.cos(g + d * 2) * 0.63844
		my += math.cos(g) * 0.8617
		my += math.cos(a - b * 2) * 1.50534
		my -= math.cos(a + d * 2) * 1.67417
		my += math.cos(a + g) * 1.99463
		my += math.cos(d) * 2.07579
		my -= math.cos(a - g) * 2.455
		my -= math.cos(a + g - d * 2) * 2.74067
		my -= math.cos(g - d * 2) * 3.83002
		my -= math.cos(a * 2) * 5.37817
		my += math.cos(a * 2 - d * 2) * 6.60763
		my -= math.cos(d * 2) * 53.97626
		my -= math.cos(a - d * 2) * 68.62152
		my -= math.cos(a) * 395.13669
		my += 3649.33705

		# MOON PHI
		mz = math.sin(a - g - b * 2 - n * 2) * -0.001
		mz -= math.sin(a + g - d * 4) * 0.001
		mz += math.sin(a * 2 - g) * 0.001
		mz += math.sin(a - g + d * 2) * 0.00102
		mz -= math.sin(a * 2 - b * 2 - n) * 0.00106
		mz -= math.sin(a * 2 + n) * 0.00106
		mz -= math.sin(a + b * 2 - d * 2) * 0.00109
		mz -= math.sin(b * 2 - d + n * 2) * 0.0011
		mz += math.sin(d * 4) * 0.00112
		mz -= math.sin(a * 2 - n) * 0.00122
		mz -= math.sin(a * 2 + b * 2 + n) * 0.00122
		mz += math.sin(g + b * 2 - d * 2 + n * 2) * 0.00149
		mz -= math.sin(a * 2 - d * 4) * 0.00157
		mz += math.sin(a + g + b * 2 - d * 2 + n * 2) * 0.00171
		mz -= math.sin(a * 2 + g - d * 2) * 0.00175
		mz -= math.sin(g * 2 - d * 2) * 0.0019
		mz += math.sin(a + e * 16 - w * 18) * 0.00193
		mz += math.sin(a * 2 + b * 2 - d * 2 + n * 2) * 0.00194
		mz += math.sin(g - d * 2 - n) * 0.00201
		mz += math.sin(g + b * 2 - d * 2 + n) * 0.00201
		mz -= math.sin(a + g * 2 - d * 2) * 0.00207
		mz -= math.sin(g * 2) * 0.0021
		mz -= math.sin(d * 2 - n) * 0.00213
		mz -= math.sin(b * 2 + d * 2 + n) * 0.00213
		mz -= math.sin(a * 3 - d * 2) * 0.00215
		mz -= math.sin(a - d * 4) * 0.00247
		mz -= math.sin(a - b * 2 + d * 2) * 0.00253
		mz += ta * 0.00279 * math.sin(b * 2 + n * 2)
		mz -= math.sin(a * 2 + b * 2 + n * 2) * 0.0028
		mz += math.sin(a * 3) * 0.00312
		mz -= math.sin(a + b * 2) * 0.00317
		mz -= math.sin(a + e * 16 - w * 18) * 0.0035
		mz += math.sin(g + b * 2 + n * 2) * 0.0039
		mz += math.sin(g - b * 2 - n * 2) * 0.00413
		mz -= math.sin(n * 2) * 0.0049
		mz -= math.sin(b * 2 + d * 2 + n * 2) * 0.00491
		mz += math.sin(g + d) * 0.00504
		mz += math.sin(a - d) * 0.00516
		mz -= math.sin(g + d * 2) * 0.00621
		mz += math.sin(a - b * 2 - d * 2 - n) * 0.00648
		mz += math.sin(a - d * 2 + n) * 0.00648
		mz += math.sin(a - g - d * 2) * 0.007
		mz += math.sin(a + d * 2) * 0.01122
		mz += math.sin(a - d * 2 - n) * 0.0141
		mz += math.sin(a + b * 2 - d * 2 + n) * 0.0141
		mz += math.sin(a - b * 2) * 0.01424
		mz += math.sin(a - b * 2 - d * 2 - n * 2) * 0.01506
		mz -= math.sin(b * 2 - d * 2) * 0.01567
		mz += math.sin(b * 2 - d * 2 + n * 2) * 0.02077
		mz -= math.sin(a + g) * 0.02527
		mz -= math.sin(a - n) * 0.02952
		mz -= math.sin(a + b * 2 + n) * 0.02952
		mz -= math.sin(d) * 0.03487
		mz += math.sin(a - g) * 0.03684
		mz -= math.sin(d * 2 + n) * 0.03983
		mz += math.sin(b * 2 - d * 2 + n) * 0.03983
		mz += math.sin(a + b * 2 - d * 2 + n * 2) * 0.04037
		mz += math.sin(a * 2) * 0.04221
		mz -= math.sin(g - d * 2) * 0.04273
		mz -= math.sin(a * 2 - d * 2) * 0.05566
		mz -= math.sin(a + g - d * 2) * 0.05697
		mz -= math.sin(a + b * 2 + n * 2) * 0.06846
		mz -= math.sin(a - b * 2 - n) * 0.08724
		mz -= math.sin(a + n) * 0.08724
		mz -= math.sin(b * 2) * 0.11463
		mz -= math.sin(g) * 0.18647
		mz -= math.sin(a - b * 2 - n * 2) * 0.20417
		mz += math.sin(d * 2) * 0.59616
		mz += math.sin(n) * 1.07142
		mz -= math.sin(b * 2 + n) * 1.07447
		mz -= math.sin(a - d * 2) * 1.28658
		mz -= math.sin(b * 2 + n * 2) * 2.4797
		mz += math.sin(a) * 6.32962

		return mx, my, mz
