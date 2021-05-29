
import numpy as np
import os
import scipy.stats

from settings import EnvironmentSettings
from utils.ellipsoid import Ellipsoid
from utils.plotting import Plotting
from utils.polyhedron import Polyhedron


def visualize_containment(ellipsoid, center, radius, name):
	is_contained = ellipsoid.contained_within_tr(center, radius, 1e-12)

	title = 'ellipse is ' + ('' if is_contained else 'not ') + 'contained'
	filename = EnvironmentSettings.get_sub_path('visualizations/containment/ellipse_contained_' + name + '.png')
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	plot = Plotting.create_plot_on(filename, center - 5, center + 5, name=title)
	plot.add_point(center, label='center of trust region', marker='x', color='red')
	plot.add_linf_tr(center, radius, label='trust region', color='red')
	plot.add_point(ellipsoid.center, label='center of ellipsoid', marker='x', color='blue')
	plot.add_contour(lambda x: ellipsoid.evaluate(x), label='ellipsoid', color='blue', lvls=[0])
	plot.save()


def visualize_containments1():
	tr_center = np.array([5, 5])
	tr_radius = 2.0

	ell_center = np.array([5, 5])
	ell_radius = 1.0
	q = np.eye(2)
	ellipsoid = Ellipsoid.create(q, ell_center, ell_radius)
	visualize_containment(ellipsoid, tr_center, tr_radius, 'simple_interior')


def visualize_containments2():
	tr_center = np.array([5, 5])
	tr_radius = 2.0

	ell_center = np.array([5, 5])
	ell_radius = 2.0
	q = np.eye(2)
	ellipsoid = Ellipsoid.create(q, ell_center, ell_radius)
	visualize_containment(ellipsoid, tr_center, tr_radius, 'simple_boundary')


def visualize_containments3():
	tr_center = np.array([5, 5])
	tr_radius = 2.0

	ell_center = np.array([5, 5])
	ell_radius = 3.0
	q = np.eye(2)
	ellipsoid = Ellipsoid.create(q, ell_center, ell_radius)
	visualize_containment(ellipsoid, tr_center, tr_radius, 'simple_not_contained')


def visualize_containments4():
	tr_center = np.array([5, 5])
	tr_radius = 2.0

	ell_center = np.array([5.5, 5])
	ell_radius = 1.5
	q = np.diag([1.0, 1])
	ellipsoid = Ellipsoid.create(q, ell_center, ell_radius)
	visualize_containment(ellipsoid, tr_center, tr_radius, 'different eigenvalues')


def visualize_containments0():
	tr_center = 10 * (2 * np.random.random(2) - 1)
	tr_radius = 0.5 + 3 * np.random.random()

	ell_center = tr_center + np.random.random(2)
	ell_radius = 0.25 + 0.5 * np.random.random()
	v = scipy.stats.ortho_group.rvs(2)
	d = 1.0 + 0.85 * (2 * np.random.random(2) - 1)
	q = v @ np.diag(d) @ v.T
	print(np.linalg.eig(q))
	ellipsoid = Ellipsoid.create(q, ell_center, ell_radius)
	visualize_containment(ellipsoid, tr_center, tr_radius, 'random')


def visualize_containments():
	visualize_containments0()
	visualize_containments1()
	visualize_containments2()
	visualize_containments3()
	visualize_containments4()


if __name__ == '__main__':
	visualize_containments()
