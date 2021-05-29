
import numpy as np
import os

from settings import EnvironmentSettings
from utils.plotting import Plotting
from utils.project_onto_cone import sphere_is_contained_within_cone


# dist = theta ** 2 * nrm ** 2 \
	# 	+ (1 - 2 * theta ** 2) * (x @ direction) ** 2 \
	# 	- 2 * theta * x @ direction * np.sqrt((1 - theta ** 2) * (nrm ** 2 - (x @ direction) ** 2))


def visualize_cone_contains_sphere(M, i, v, u, theta, x, radius):
	is_contained, projection = sphere_is_contained_within_cone(v, u, theta, x, radius)

	filename = EnvironmentSettings.get_sub_path('visualizations/cone_contains_sphere/projection_' + str(i) + '.png')
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	plot = Plotting.create_plot_on(filename, [-1.2 * M, -1.2 * M], [M, M], name='is contained = ' + str(is_contained))
	plot.add_wedge(v, u, theta, 2.0, label='cone', color='yellow')
	plot.add_arrow(v, v + u, color='green', label='u')
	plot.add_arrow(np.zeros_like(x), x, color='red', label='x')
	plot.add_circle(center=x, radius=radius, color='red')
	plot.add_arrow(x, projection, color='blue', label='x')
	plot.save()


def visualize_several_cone_contains_sphere():
	M = 5

	visualize_cone_contains_sphere(
		M, -1, np.array([1, 0]), np.array([-1, 0]), 0.5, np.zeros(2), 0.1)
	visualize_cone_contains_sphere(
		M, -2, np.array([-1, 0]), np.array([-1, 0]), 0.5, np.zeros(2), 0.1)

	for i in range(100):
		u = 2 * np.random.random(2) - 1
		u /= np.linalg.norm(u)

		x = M * (2 * np.random.random(2) - 1)
		v = M * (2 * np.random.random(2) - 1)
		theta = 2 * np.random.random() - 1
		radius = 0.25 * np.random.random()

		visualize_cone_contains_sphere(M, i, v, u, theta, x, radius)


if __name__ == '__main__':
	np.random.seed(1776)
	visualize_several_cone_contains_sphere()
