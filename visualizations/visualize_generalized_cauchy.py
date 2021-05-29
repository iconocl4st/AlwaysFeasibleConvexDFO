import os

import numpy as np

from settings import EnvironmentSettings
from utils.generalized_cauchy import GeneralizedCauchyParams
from utils.generalized_cauchy import compute_generalized_cauchy_point
from utils.plotting import Plotting


def visualize_generalized_cauchy():
	scale = 100
	func = lambda x: (x[0] - 10) ** 4 / scale + (3 * x[1] - 5) ** 2 / scale
	grad = lambda x: np.array([4 * (x[0] - 10) ** 3, 2 * 3 * (3 * x[1] - 5)]) / scale

	class MockModel:
		def evaluate(self, x):
			return func(x)

	gcp = GeneralizedCauchyParams()
	gcp.radius = 2
	gcp.x = np.array([7.5, 2])
	gcp.gradient = grad(gcp.x)
	gcp.model = MockModel()
	gcp.A = np.array([
		[1, 1],
		[-1, 1],
		[1, 0],
	])
	gcp.b = gcp.A @ gcp.x + np.random.random()
	gcp.cur_val = func(gcp.x)

	s, val, projections, tangent_cone_projections = compute_generalized_cauchy_point(gcp)

	filename = EnvironmentSettings.get_sub_path('visualizations/gc/generalized_cauchy.png')
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	plt = Plotting.create_plot_on(filename, gcp.x - 5, gcp.x + 5, 'Generalized Cauchy point')
	plt.add_lines(gcp.A, gcp.b, label='constraint', color='m', lvls=[-0.1, 0])
	plt.add_contour(func, label='objective', color='k')
	plt.add_point(gcp.x, label='center', color='green')
	plt.add_point(gcp.x - gcp.gradient, label='negative gradient', color='red')
	plt.add_arrow(gcp.x, gcp.x - gcp.gradient, label='negative gradient', color='c', width=0.05)
	plt.add_points(np.array(projections), label='projected gradient path', color='blue')
	plt.add_points(np.array(tangent_cone_projections), label='tangent cone path', color='yellow')
	plt.save()


if __name__ == '__main__':
	visualize_generalized_cauchy()
