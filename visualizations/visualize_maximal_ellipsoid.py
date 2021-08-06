
import numpy as np
import os

import scipy.stats

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
			contained, plotter = cone_contains_ellipsoid(v, self.fr.b, (rot.T @ np.diag(diag) @ rot), center)
			plotter(plot)
			plot.add_contour(
				lambda x: (x - center) @ (rot.T @ np.diag(diag) @ rot) @ (x - center) - 1,
				label='ellipsoid', color='b', lvls=[-0.1, 0])
			plot.add_point(center, label='ellipsoid center', color='b')
			plot.save()


class FeasibleRegion:
	def __init__(self):
		self.tr_center = None
		self.tr_radius = None
		self.vs = None
		self.b = None
		self.pltr = None

	def add_to_plot(self, plot):
		plot.add_point(self.tr_center, label='tr center', color='r', marker='x')
		for v in self.vs:
			plot.add_wedge(v, -v, self.b, 20.0, color=(1.0, 0.8, 0.6, 0.5), label='cone')
		plot.add_linf_tr(self.tr_center, self.tr_radius, 'trust region', color='b')

	def contains_ellipsoid(self, q, c):
		for v in self.vs:
			if not cone_contains_ellipsoid(v, self.b, q, c)[0]:
				return False
		return True


def determine_ellipsoid(feasible_region, init_center, tol):
	n = len(init_center)
	init_diag = np.ones_like(init_center)
	init_rot = np.eye(n)
	while not feasible_region.contains_ellipsoid(np.diag(init_diag), init_center):
		init_diag *= 2
		if init_diag[0] > 1e30:
			return -1, init_center, init_rot, None
	init_volume = np.prod(1/init_diag)

	max_volume = None
	max_diag = None
	max_center = None
	max_rot = None

	for i in range(5):
		print(i)
		volume = init_volume.copy()
		center = init_center.copy()
		rot = init_rot
		diag = init_diag

		delta = 1.0
		while delta > 1e-6:
			iterations_since_improvement = 0
			while iterations_since_improvement < 50:
				iterations_since_improvement += 1
				alt_diag = diag + np.random.normal(loc=0, scale=delta, size=n)
				if min(alt_diag) < tol:
					continue
				alt_volume = np.prod(1/alt_diag)
				if alt_volume < volume:
					continue
				alt_rot = scipy.stats.ortho_group.rvs(n)
				alt_center = center + np.random.normal(loc=0, scale=2 * delta, size=n)

				alt_q = alt_rot.T @ np.diag(alt_diag) @ alt_rot
				ellipsoid = Ellipsoid.create(alt_q, alt_center, r=1.0)
				if not ellipsoid.contained_within_tr(
						feasible_region.tr_center, feasible_region.tr_radius, tol):
					continue

				if not feasible_region.contains_ellipsoid(alt_q, alt_center):
					continue
				diag = alt_diag
				center = alt_center
				rot = alt_rot
				volume = alt_volume
				iterations_since_improvement = 0
				if feasible_region.pltr:
					feasible_region.pltr.plot_update(volume, center, rot, diag)

			delta /= 2.0

		if max_volume is None or volume > max_volume:
			max_volume = volume
			max_diag = diag
			max_center = center
			max_rot = rot
	return max_volume, max_center, max_rot, max_diag


def visualize_projection_onto_ellipsoid():
	tol = 1e-8

	for i in range(10):
		fr = FeasibleRegion()
		fr.tr_center = np.zeros(2)
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
