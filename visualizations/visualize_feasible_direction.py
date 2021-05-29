
import numpy as np
import os

from settings import EnvironmentSettings
from utils.plotting import Plotting
from pyomo_opt.feasible_direction import find_feasible_direction


def create_gradients(initial_num, direction, tolerance):
	gradients = 2 * np.random.random([initial_num, 2]) - 1
	gradients = gradients / np.linalg.norm(gradients, axis=1)[:, None]
	gradients = gradients[-gradients@direction > tolerance]
	return gradients


def visualize_feasible_direction():
	direction = 2 * np.random.random(2) - 1
	direction /= np.linalg.norm(direction)

	gradients = create_gradients(15, direction, 1e-2)
	success, u, t = find_feasible_direction(gradients)

	filename = EnvironmentSettings.get_sub_path('visualizations/feasible_direction.png')
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	plot = Plotting.create_plot_on(
		filename,
		[-1.2, -1.2], [1.2, 1.2],
		name='feasible direction, success=' + str(success) + ', t=' + str(t)
	)
	for gradient in gradients:
		plot.add_arrow(np.zeros_like(gradient), gradient, label='gradient ' + str(gradient))

	plot.add_arrow(np.zeros_like(direction), direction, color='yellow', label='generated direction')
	plot.add_arrow(np.zeros_like(u), u, color='green', label='computed direction')

	plot.save()


if __name__ == '__main__':
	visualize_feasible_direction()
