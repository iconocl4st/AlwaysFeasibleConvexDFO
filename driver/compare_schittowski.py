
import numpy as np
import sys
from hott_schittowski.problems import HottSchittowski, Data
from hott_schittowski.fortran_interface import create_problem_description


def bound_cmp(x, y):
	if x is None and y is None:
		return False
	if x is None:
		return any(yi is not None for yi in y)
	if y is None:
		return any(xi is not None for xi in x)

	for xi, yi in zip(x, y):
		if xi is None and yi is None:
			continue
		if xi is None or yi is None:
			return True
		if abs(xi - yi) > 0.01 * abs(yi):
			return True
	return False


def arr_cmp(x, y):
	return (
		(x is None and y is not None) or
		(x is not None and y is None) or
		np.linalg.norm(x - y) > 0.01 * np.linalg.norm(y)
	)


def str_cmp(x, y):
	return len(x) != len(y) or any(xi != yi for xi, yi in zip(x, y))


def bool_cmp(x, y):
	return x != y


def get_func_cmp(problem):
	x1 = problem.initial.x0
	x2 = problem.solution.xstar
	num_points = 100
	noise_r = 5
	noise = noise_r * (2 * np.random.random((num_points, problem.n)) - 1)
	coeff = np.random.random((num_points, 1))
	trial_points = noise + coeff * x1 + (1 - coeff) * x2

	def func_cmp(f1, f2):
		y1 = np.array([f1(x) for x in trial_points], dtype=np.float64)
		y2 = np.array([f2(x) for x in trial_points], dtype=np.float64)
		ret = arr_cmp(y1, y2)
		if ret:
			print(y1[:3])
			print(y2[:3])
		return ret
	return func_cmp


def compare_implementation(problem):
	library_problem = create_problem_description(problem.number)

	library_initial_evaluation = library_problem.evaluate(library_problem.x0)
	# TODO: Handle equality constraints...
	library_is_feasible = (library_initial_evaluation.constraint_values >= 0).all()

	comparison = [
		['n', problem.n, library_problem.n, arr_cmp],
		['minimum', problem.solution.fstar, library_problem.f_min, arr_cmp],
		['minimizer', problem.solution.xstar, library_problem.x_min, arr_cmp],
		['lower-bounds', problem.bounds.lb, library_problem.lower_bounds, bound_cmp],
		['upper-bounds', problem.bounds.ub, library_problem.upper_bounds, bound_cmp],
		['initial-x', problem.initial.x0, library_problem.x0, arr_cmp],
		['initial-f', problem.initial.f0, library_initial_evaluation.f, arr_cmp],
		['is-feasible', problem.initial.is_feasible, library_is_feasible, bool_cmp],
		['constraint-types', problem.get_constraint_types(), library_problem.get_constraint_types(), str_cmp],
		['solution-is-exact',
			problem.classification['solution-is-exact'],
			library_problem.solution_is_exact, bool_cmp],
		['objective', problem.objective, lambda x: library_problem.evaluate_objective(x), get_func_cmp(problem)],
	] + [
		[
			'constraint-' + str(idx),
			constraint.expr,
			library_problem.get_constraint_evaluator(idx),
			get_func_cmp(problem)
		]
		for idx, constraint in enumerate(problem.constraints)
	]
	# objective = lambda x: (10 * (x[0] - x[1]) ** 2 + (x[0] - 1) ** 2) ** 4,
	# constraints = [],

	print('========================================')
	print('number', problem.number)
	if (
		False or
		len(problem.constraints) == 0 or
		any(c.sense == '==0' for c in problem.constraints) or
		not problem.initial.is_feasible or
		problem.notes is not None
	):
		return
		# print('Ignoring this problem')

	f01 = problem.objective(problem.initial.x0)
	f02 = problem.initial.f0
	if arr_cmp(f01, f02):
		print('The initial value in the book did not match the function value', f01, f02)

	f01 = problem.objective(problem.solution.xstar)
	f02 = problem.solution.fstar
	if arr_cmp(f01, f02):
		print('The solution value in the book did not match the function value', f01, f02)

	for [key, python_impl, library_impl, cmp] in comparison:
		if cmp(python_impl, library_impl):
			print('\t', key)
			print('\t\t', 'python implementation', python_impl)
			print('\t\t', 'library implementation', library_impl)


def compare_all_implementations():
	for problem in HottSchittowski.PROBLEMS:
		# try:
		compare_implementation(problem)
		# except:
		# 	print('Unable to load problem', problem.number)


if __name__ == '__main__':
	# for idx, yi in enumerate(Data.p105_y):
	# 	print(idx+1, yi)

	if len(sys.argv) < 2:
		compare_all_implementations()
	else:
		problem_num = int(sys.argv[-1])
		problem = HottSchittowski.get_problem_by_number(problem_num)
		compare_implementation(problem)


