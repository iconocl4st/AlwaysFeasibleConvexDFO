
import numpy as np
import os

import scipy.stats

from algorithm.sample_regions.numerical import FeasibleRegion, determine_ellipsoid
from settings import EnvironmentSettings
from utils.ellipsoid import Ellipsoid
from utils.plotting import Plotting
from scipy.stats import ortho_group
from utils.cone_contains_sphere import cone_contains_ellipsoid


# d = 2 * np.random.random(2)
# r = scipy.stats.ortho_group.rvs(2)
# q = r.T @ np.diag(d) @ r
# c = (2 * np.random.random(2) - 1)


class PlotLogger:
	def __init__(self, fr, nb):
		self.fr = fr
		self.image_counter = 0
		self.max_volume = -1
		self.nb = nb

	def plot_update(self, volume, center, rot, diag):
		if volume < self.max_volume:
			return
		q = rot.T @ np.diag(diag) @ rot
		if not self.fr.contains_ellipsoid(q, center):
			print('uh oh')
		self.max_volume = volume
		self.image_counter += 1

		filename = EnvironmentSettings.get_output_path(
			['visualizations', 'maximize_ellipsoid', str(self.nb), 'ellipsoid_' + str(self.image_counter) + '.png'])
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		plot = Plotting.create_plot_on(
			filename, [-6, -6], [6, 6], name='volume = ' + str(volume))
		self.fr.add_to_plot(plot)
		plot.add_contour(
			lambda x: (x - center).T @ q @ (x - center) - 1,
			label='ellipsoid',
			color='b',
			lvls=[-0.1, 0])
		plot.add_point(center, label='ellipsoid center', color='b')
		plot.save()

		for idx, v in enumerate(self.fr.vs):
			filename = EnvironmentSettings.get_output_path(
				['visualizations', 'maximize_ellipsoid', str(self.nb),
				'ellipsoid_' + str(self.image_counter) + '_solution_check_' + str(idx) + '.png'])
			plot = Plotting.create_plot_on(
				filename, [-6, -6], [6, 6], name='volume = ' + str(volume))
			plot.add_wedge(v, -v, self.fr.b, 20.0, color=(1.0, 0.8, 0.6, 0.5), label='cone')
			nv = np.linalg.norm(v)
			vh = v / nv
			contained, plotter = cone_contains_ellipsoid(
				nv, vh,
				self.fr.b, (rot.T @ np.diag(diag) @ rot), center)
			plotter(plot)
			plot.add_contour(
				lambda x: (x - center) @ (rot.T @ np.diag(diag) @ rot) @ (x - center) - 1,
				label='ellipsoid', color='b', lvls=[-0.1, 0])
			plot.add_point(center, label='ellipsoid center', color='b')
			plot.save()



def visualize_projection_onto_ellipsoid():
	tol = 1e-8

	for i in range(10):
		fr = FeasibleRegion()
		fr.tr_center = np.random.random(2)
		fr.tr_radius = 3.0
		fr.vs = 5 * (2 * np.random.random((3, 2)) - 1)
		fr.b = np.random.random()
		fr.pltr = PlotLogger(fr, i)

		center = (2 * np.random.random(2) - 1)
		volume, center, rot, diag = determine_ellipsoid(fr, center, tol)


if __name__ == '__main__':
	np.random.seed(1776)
	visualize_projection_onto_ellipsoid()























#
# def determine_rotation(feasible_region, center, tol):
# 	image_counter = 0
# 	n = len(center)
# 	delta = 1.0
#
# 	rot = np.eye(n)
# 	volume, diag = determine_diagonals(feasible_region, center, rot, tol)
#
# 	iterations_since_improvement = 0
# 	while iterations_since_improvement < 10:
#
# 		alt_volume, alt_diag = determine_diagonals(feasible_region, center, alt_rot, tol)
# 		if alt_volume <= volume:
# 			continue
# 		iterations_since_improvement = 0
# 		rot = alt_rot
# 		diag = alt_diag
# 		volume = alt_volume
# 		feasible_region.pltr.plot_update(volume, center, rot, diag)
# 	return volume, rot, diag


# def compute_maximum_ellipsoid(feasible_region, tol=1e-8):
# 	center = feasible_region.tr_center
# 	n = len(center)
# 	delta = 1.0
# 	while delta > tol:
# 		iterations_since_improvement = 0
# 		while iterations_since_improvement < 100:
# 			other_center = delta * np.random.random(n)
# 		delta /= 2
