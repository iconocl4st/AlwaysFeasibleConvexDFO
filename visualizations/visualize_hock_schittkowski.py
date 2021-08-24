import os
import sys

from settings import EnvironmentSettings
from utils.ellipsoid import Ellipsoid
from utils.plotting import Plotting
from utils.bounds import Bounds
from hock_schittkowski.problems import HockSchittkowski
from utils.finite_difference import find_feasible_start


def add_problem_to_plot(problem, plt):
	plt.add_contour(problem.objective, label='objective', color='k')
	for i, ci in enumerate(problem.bounds.to_constraint_functions()):
		plt.add_contour(ci, label='bound constraint ' + str(i), lvls=[-0.1, 0], color='b')

	for i, ci in enumerate(problem.constraints):
		if ci.sense == '==0':
			plt.add_contour(ci.expr, label='equality constraint ' + str(i), lvls=[0], color='r')
		elif ci.sense == '>=0':
			plt.add_contour(lambda x: -ci.expr(x), label='ge constraint ' + str(i), lvls=[-0.1, 0], color='g')
		else:
			plt.add_contour(ci.expr, label='lt constraint ' + str(i), lvls=[-0.1, 0], color='m')

	plt.add_point(problem.solution.xstar, label='solution', color='m', marker='o')
	plt.add_point(problem.initial.x0, label='initial-point', color='red')


def plot_hock_schittkowski_problem(problem):
	if problem.n != 2:
		return False
	filename = EnvironmentSettings.get_output_path([
		'visualizations',
		'schittowski',
		str(problem.number).rjust(3, '0') + '_.png'])
	os.makedirs(os.path.dirname(filename), exist_ok=True)

	bounds = Bounds() \
		.extend(problem.initial.x0) \
		.extend(problem.solution.xstar)

	if problem.bounds.has_lb():
		bounds = bounds.extend(problem.bounds.lb)
	if problem.bounds.has_ub():
		bounds = bounds.extend(problem.bounds.ub)

	has_ellipsoid, center, q, r = find_feasible_start(problem)
	if has_ellipsoid:
		bounds.extend_tr(center, r)

	bounds = bounds.buffer(0.1).expand(1.2)

	plt = Plotting.create_plot_on(filename, bounds.lb, bounds.ub, 'Hott-Schittowski-' + str(problem.number))

	if has_ellipsoid:
		plt.add_point(center, label='first ellipsoid center', color='c')
		ellipsoid = Ellipsoid.create(q, center, r)
		plt.add_contour(ellipsoid.evaluate, label='initial ellipsoid', lvls=[0], color='c')

	add_problem_to_plot(problem, plt)

	plt.save()


if __name__ == '__main__':
	for problem in HockSchittkowski.PROBLEMS:
	# for problem in [HottSchittowski.get_problem_by_number(30)]:
		print('problem:', problem.number)
		plot_hock_schittkowski_problem(problem)
