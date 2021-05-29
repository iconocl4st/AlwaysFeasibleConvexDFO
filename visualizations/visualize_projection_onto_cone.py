
import numpy as np
import os

from settings import EnvironmentSettings
from utils.plotting import Plotting
from utils.project_onto_cone import project_onto_edge_of_cone


# dist = theta ** 2 * nrm ** 2 \
	# 	+ (1 - 2 * theta ** 2) * (x @ direction) ** 2 \
	# 	- 2 * theta * x @ direction * np.sqrt((1 - theta ** 2) * (nrm ** 2 - (x @ direction) ** 2))


def visualize_projection_onto_cone():
	M = 5

	for i in range(100):
		u = 2 * np.random.random(2) - 1
		u /= np.linalg.norm(u)

		x = M * (2 * np.random.random(2) - 1)
		v = M * (2 * np.random.random(2) - 1)
		theta = 2 * np.random.random() - 1

		proj, dist, feasible = project_onto_edge_of_cone(v, u, theta, x)

		filename = EnvironmentSettings.get_sub_path('visualizations/project_onto_cone/projection_' + str(i) + '.png')
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		plot = Plotting.create_plot_on(filename, [-1.2 * M, -1.2 * M], [M, M], name='distance to projection = ' + str(dist))
		plot.add_wedge(v, u, theta, 2.0, label='cone', color='yellow')
		plot.add_arrow(v, v + u, color='green', label='u')
		plot.add_arrow(np.zeros_like(x), x, color='red', label='x')
		if not feasible:
			plot.add_arrow(v, proj, color='m', label='direction')
		plot.add_arrow(x, proj, color='blue', label='projection')
		plot.save()


if __name__ == '__main__':
	np.random.seed(1776)
	visualize_projection_onto_cone()
