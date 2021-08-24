
import numpy as np
import os

import scipy.stats

from settings import EnvironmentSettings
from utils.plotting import Plotting
from scipy.stats import ortho_group
from utils.cone_contains_sphere import cone_contains_ellipsoid


# dist = theta ** 2 * nrm ** 2 \
	# 	+ (1 - 2 * theta ** 2) * (x @ direction) ** 2 \
	# 	- 2 * theta * x @ direction * np.sqrt((1 - theta ** 2) * (nrm ** 2 - (x @ direction) ** 2))


def visualize_projection_onto_ellipsoid():
	M = 5
	for i in range(50):
		# d = 2 * np.random.random(2)
		d = 4 * np.random.random(2)
		r = scipy.stats.ortho_group.rvs(2)
		q = r.T @ np.diag(d) @ r
		# c = (2 * np.random.random(2) - 1)
		c = 2 * (2 * np.random.random(2) - 1)

		v = 5 * (2 * np.random.random(2) - 1)
		b = np.random.random()

		nv = np.linalg.norm(v)
		vh = v / nv
		if np.random.random() > 0.5 or True:
			nv = 0
			v = nv * vh
		contained, plotter = cone_contains_ellipsoid(nv, vh, b, q, c)

		filename = EnvironmentSettings.get_output_path(
			['visualizations', 'cone_contains_ellipsoid', 'projection_' + str(i) + '.png'])
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		plot = Plotting.create_plot_on(
			filename, [-1.2 * M, -1.2 * M], [M, M], name='contained = ' + str(contained))
		plot.add_point(c, label='center', color='r', marker='x')

		plot.add_wedge(v, -vh, b, 20.0, color=(1.0, 0.8, 0.6, 0.5), label='cone')
		plot.add_contour(lambda x: (x - c) @ q @ (x - c) - 1, label='ellipsoid', color='b', lvls=[-0.1, 0])

		# plot.add_point(np.array([-1.37293283,  0.3511065]), label='test', color='k')
		if plotter:
			plotter(plot)
		# plot.add_arrow(p, proj, color='k', label='projection')
		plot.save()


if __name__ == '__main__':
	np.random.seed(1776)
	visualize_projection_onto_ellipsoid()
