
import numpy as np
import os

import scipy.stats

from settings import EnvironmentSettings
from utils.plotting import Plotting
from scipy.stats import ortho_group
from utils.cone_contains_sphere import anti_project_point_onto_ellipsoid


# dist = theta ** 2 * nrm ** 2 \
	# 	+ (1 - 2 * theta ** 2) * (x @ direction) ** 2 \
	# 	- 2 * theta * x @ direction * np.sqrt((1 - theta ** 2) * (nrm ** 2 - (x @ direction) ** 2))


def visualize_projection_onto_ellipsoid():
	M = 3
	for i in range(50):
		p = 2 * (2 * np.random.random(2) - 1)
		d = 2 * np.random.random(2)
		if np.random.random() < 0.25:
			d[1] = d[0]
		v = scipy.stats.ortho_group.rvs(2)
		q = v.T @ np.diag(d) @ v

		success, proj, dist = anti_project_point_onto_ellipsoid(q, p)

		filename = EnvironmentSettings.get_output_path(
			['visualizations', 'project_onto_ellipsoid', 'projection_' + str(i) + '.png'])
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		plot = Plotting.create_plot_on(
			filename, [-1.2 * M, -1.2 * M], [M, M], name='distance to projection = ' + str(dist))
		plot.add_point(p, label='point', color='r', marker='x')
		plot.add_point(proj, label='projection', color='g', marker='x')
		plot.add_contour(lambda x: x @ q @ x - 1, label='ellipsoid', color='b', lvls=[-0.1, 0])
		plot.add_arrow(p, proj, color='k', label='projection')
		plot.save()


if __name__ == '__main__':
	np.random.seed(1776)
	visualize_projection_onto_ellipsoid()
