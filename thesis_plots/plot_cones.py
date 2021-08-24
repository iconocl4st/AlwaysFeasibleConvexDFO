import os
import numpy as np
import scipy.optimize

from settings import EnvironmentSettings
from utils.plotting import Plotting


def plot_unshifted_cone():
	filename = EnvironmentSettings.get_output_path(
		['visualizations', 'thesis_plots', 'unshifted_cone.png'])
	os.makedirs(os.path.dirname(filename), exist_ok=True)

	plot = Plotting.create_plot_on(
		filename, [0, 0], [10, 10], name='Unshifted cone')

	a = np.array([
			[0.25, 1],
			[1, -0.5],
			[-0.25, 1],
		])
	b = np.array([6, 5, 5])
	nrms = np.linalg.norm(a, axis=1)
	a = a / nrms[:, np.newaxis]
	b = b / nrms

	plot.add_lines(a=a, b=b, label='Polyhedron')

	active_a = a[1:]
	active_b = b[1:]
	xk = np.linalg.solve(active_a, active_b)
	plot.add_point(xk, label='x^k', color='g', marker='x', s=50)

	minimization_result = scipy.optimize.minimize(
		fun=lambda x: -min(-x@ai for ai in active_a),
		x0=np.array([-1, -1]),  # np.random.random(2),
		constraints=[scipy.optimize.NonlinearConstraint(lambda x: x@x, 1, 1)],
	)
	u = minimization_result.x
	pi = -minimization_result.fun
	beta = np.sqrt(1 - pi ** 2)

	plot.add_wedge(xk, u, beta, 5.0, color=(1.0, 0.8, 0.6, 0.5), label='cone')

	# plot.add_point(np.array([-1.37293283,  0.3511065]), label='test', color='k')
	plot.add_arrow(xk, xk + u, color='k', label='u')
	plot.save()


def plot_shifted_cone():
	pi = 0.5
	filename = EnvironmentSettings.get_output_path(
		['visualizations', 'thesis_plots', 'shifted_cone.png'])
	os.makedirs(os.path.dirname(filename), exist_ok=True)

	plot = Plotting.create_plot_on(
		filename, [0, -1], [2, 1], name='Shifted cone, pi=' + str(pi))

	e1 = np.zeros(2)
	e1[0] = 1
	s = np.array([0, pi])

	plot.add_wedge(np.zeros(2), e1, e1 @ (s + e1) / np.linalg.norm(s + e1), 2.0, color=(1.0, 0.8, 0.6, 0.5), label='cone')
	# np.sqrt(1 - pi ** 2)
	# plot.add_point(np.array([-1.37293283,  0.3511065]), label='test', color='k')
	plot.add_arrow(np.zeros(2), e1, color='b', label='e_1')
	plot.add_arrow(e1, e1 + s, color='r', label='s')
	plot.save()


def plot_bad_feasible_region():
	filename = EnvironmentSettings.get_output_path(
		['visualizations', 'thesis_plots', 'small_sample_region.png'])
	os.makedirs(os.path.dirname(filename), exist_ok=True)

	r = 1.25
	plot = Plotting.create_plot_on(
		filename, [-r, -r], [r, r],
		name='Small feasible region')

	a = np.array([
		[1, 0],
		[-1, 0],
		[0, 1],
		[0, -1],
	], dtype=np.float64)
	b = np.array([1, 1, 1, 1], dtype=np.float64)

	alpha = 0.05
	fa = np.array([
		[-alpha, 1],
		[-alpha, -1],
	])
	fb = np.array([alpha, alpha])

	sample_points = np.empty((5, 2), dtype=np.float64)
	for i in range(sample_points.shape[0]):
		point = np.random.random(2)
		while (fa @ point >= fb).any():
			point = np.random.random(2)
		sample_points[i] = point

	sample_points[0] = np.array([-1, 0])
	sample_points[1] = np.array([1, 2 * alpha])
	sample_points[2] = np.array([1, -2 * alpha])

	plot.add_points(sample_points, label='sample points', color='y', s=20)

	plot.add_lines(a=a, b=b, label='Outer trust region')
	plot.add_point(np.zeros(2), label='x^k', color='g', marker='x', s=50)

	plot.add_lines(
		fa,
		fb,
		color='r',
		label='Feasible region'
	)

	plot.save()


def plot_spokes():
	filename = EnvironmentSettings.get_output_path(
		['visualizations', 'thesis_plots', 'spokes.png'])
	os.makedirs(os.path.dirname(filename), exist_ok=True)

	r = 1.5
	plot = Plotting.create_plot_on(
		filename, [-r, -r], [r, r],
		name='')

	x = np.array([1, 1], dtype=np.float64)
	y = np.array([1, -1.2], dtype=np.float64)
	x /= 1.1 * np.linalg.norm(x)
	y /= 1.2 * np.linalg.norm(y)
	u = np.array([1.2, x[1] + 0.1])
	v = np.array([1.1, y[1]])
	plot.add_circle(center=np.zeros(2), radius=1, color='k')
	# plot.add_arrow(np.zeros_like(x), x, color='b', label='x')
	# plot.add_arrow(np.zeros_like(y), y, color='g', label='y')
	plot.add_arrow(np.zeros_like(y), np.array([1, 0], dtype=np.float64), color='y', label='a')
	plot.add_arrow(x, u, color='r', label='epsilon')
	plot.add_arrow(y, v, color='r', label='epsilon')
	plot.add_point(x, color='b', s=50, marker='o', label='x')
	plot.add_point(y, color='g', s=50, marker='o', label='y')
	plot.add_point(u, color='b', s=50, marker='+', label='u')
	plot.add_point(v, color='g', s=50, marker='+', label='v')

	plot.add_lines(a=np.array([[1.0, 0.0]]), b=np.array([1]), label='ax >= |a|^2')

	plot.save()


if __name__ == '__main__':
	plot_unshifted_cone()
	plot_shifted_cone()
	plot_bad_feasible_region()
	plot_spokes()
