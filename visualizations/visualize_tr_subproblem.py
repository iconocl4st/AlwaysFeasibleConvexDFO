
import numpy as np
import os

import scipy.stats
from settings import EnvironmentSettings
from utils.plotting import Plotting
from utils.tr import solve_tr_subproblem


def visualize_tr_subproblem(g, q, image_path):
	success, minimizer, value = solve_tr_subproblem(g, q)
	assert success, 'failure'

	os.makedirs(os.path.dirname(image_path), exist_ok=True)
	plot = Plotting.create_plot_on(
		image_path,
		[-1.2, -1.2], [1.2, 1.2],
		name='trust region subproblem'
	)
	plot.add_contour(lambda x: g@x + 0.5 * x.T@q@x, label='quadratic')
	plot.add_point(minimizer, label='minimum', s=50)
	plot.add_circle(center=np.zeros(2), radius=1)
	plot.save()


def visualize_tr_subproblems():
	q = np.array([
		[2., -0.41421356],
		[-0.41421356, 0.]
	])
	g = np.array([1., 0.])
	filename = EnvironmentSettings.get_sub_path('visualizations/tr_subproblem/problem.png')
	visualize_tr_subproblem(-g, -q, filename)

	g = 2 * np.random.random(2) - 1
	g /= np.linalg.norm(g)
	g *= 10

	v = scipy.stats.ortho_group.rvs(2)
	d = 1 + 5 * np.random.random(2)
	h = v @ np.diag(d) @ v.T
	filename = EnvironmentSettings.get_sub_path('visualizations/tr_subproblem/pd_exterior.png')
	visualize_tr_subproblem(g, h, filename)

	filename = EnvironmentSettings.get_sub_path('visualizations/tr_subproblem/pd_interior.png')
	visualize_tr_subproblem(g / 5, h, filename)

	d = np.array([-1, 2])
	h = v @ np.diag(d) @ v.T
	filename = EnvironmentSettings.get_sub_path('visualizations/tr_subproblem/id_simple.png')
	visualize_tr_subproblem(g, h, filename)

	d = np.array([-1, -2])
	h = v @ np.diag(d) @ v.T
	filename = EnvironmentSettings.get_sub_path('visualizations/tr_subproblem/nd.png')
	visualize_tr_subproblem(g, h, filename)

	d = np.array([-1, 2])
	h = v @ np.diag(d) @ v.T
	filename = EnvironmentSettings.get_sub_path('visualizations/tr_subproblem/difficult.png')
	visualize_tr_subproblem(v[:, 1], h, filename)

	d = np.array([-1, -1])
	h = v @ np.diag(d) @ v.T
	filename = EnvironmentSettings.get_sub_path('visualizations/tr_subproblem/difficult_trivial.png')
	visualize_tr_subproblem(v[:, 1], h, filename)

	d = np.array([1, 0])
	h = v @ np.diag(d) @ v.T
	filename = EnvironmentSettings.get_sub_path('visualizations/tr_subproblem/psd.png')
	visualize_tr_subproblem(g, h, filename)

	d = np.array([-1, 0])
	h = v @ np.diag(d) @ v.T
	filename = EnvironmentSettings.get_sub_path('visualizations/tr_subproblem/nsd.png')
	visualize_tr_subproblem(g, h, filename)

	d = np.array([0, 0])
	h = v @ np.diag(d) @ v.T
	filename = EnvironmentSettings.get_sub_path('visualizations/tr_subproblem/zd.png')
	visualize_tr_subproblem(g, h, filename)

	d = np.array([0, 0])
	h = v @ np.diag(d) @ v.T
	filename = EnvironmentSettings.get_sub_path('visualizations/tr_subproblem/zero.png')
	visualize_tr_subproblem(np.zeros_like(g), h, filename)




if __name__ == '__main__':
	visualize_tr_subproblems()
