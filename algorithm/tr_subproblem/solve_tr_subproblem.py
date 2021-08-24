import numpy as np

from algorithm.tr_subproblem.gc_solution import gc_search_for_cauchy
from algorithm.tr_subproblem.penalty_buffered import search_penalty_buffered
from algorithm.tr_subproblem.pyomo_search import pyomo_search
from algorithm.tr_subproblem.quadratic_buffered import search_quadratic_buffered
from algorithm.tr_subproblem.scale_linear import scale_linear_solutions
from algorithm.tr_subproblem.sphere_search import linear_sphere_search, conic_sphere_search
from algorithm.tr_subproblem.stochastic_search import conic_stochastic_search, linear_stochastic_search
from algorithm.tr_subproblem.tr_solution import TrSolution
from utils.assertions import make_assertion

from utils.trial_value import TrialValue


def add_xk(model, solutions):
	cones_sol = TrSolution()
	cones_sol.type = TrSolution.Types.CONES
	cones_sol.name = 'iterate'
	cones_sol.trial = model.shifted_x
	cones_sol.value = model.shifted_objective.evaluate(cones_sol.trial)
	solutions.append(cones_sol)

	# linear_sol = TrSolution()
	# linear_sol.type = TrSolution.Types.LINEAR
	# linear_sol.name = cones_sol.name
	# linear_sol.trial = cones_sol.trial
	# linear_sol.value = cones_sol.value
	# solutions.append(linear_sol)


def _find_minimum_solution(solutions, solution_types):
	trial_value = TrialValue()
	for solution in solutions:
		if solution.type not in solution_types:
			continue
		trial_value.accept(solution.trial, solution.value)
	make_assertion(
		trial_value.trial is not None,
		'No solution found for: [' + ','.join(solution_types) + ']')
	return trial_value


def solve_tr_subproblem(state, model, buffered_region):
	state.logger.start_step('Solving the trust region subproblem')

	tr_solutions = []

	add_xk(model, tr_solutions)
	state.logger.info('computing generalized Cauchy')
	gc_search_for_cauchy(state, model, tr_solutions)
	state.logger.info('stochastic linear minimization')
	linear_stochastic_search(model, tr_solutions)
	state.logger.info('spherical linear minimization')
	linear_sphere_search(state, model, tr_solutions)

	scale_linear_solutions(model, buffered_region, tr_solutions)
	state.logger.info('stochastic conic minimization')
	conic_stochastic_search(model, buffered_region, tr_solutions)
	state.logger.info('spherical conic minimization')
	conic_sphere_search(state, model, buffered_region, tr_solutions)

	state.logger.info('pyomo minimization')
	pyomo_search(state, model, buffered_region, tr_solutions)

	for solution in tr_solutions:
		make_assertion(
			np.max(model.shifted_A @ solution.trial - model.shifted_b) < 1e-4,
			'trial point is infeasible')

	for sol in tr_solutions:
		print(sol)

	linear_solution = _find_minimum_solution(
		tr_solutions,
		[TrSolution.Types.LINEAR])
	cones_solution = _find_minimum_solution(
		tr_solutions,
		[TrSolution.Types.LINEAR, TrSolution.Types.CONES])

	state.logger.log_matrix('minimum linear solution', linear_solution.trial)
	state.logger.verbose('with value: ' + str(linear_solution.value))
	state.logger.log_matrix('minimum trial point', cones_solution.trial)
	state.logger.verbose('with value: ' + str(cones_solution.value))

	state.logger.stop_step()
	# Need to be unshifted...
	state.current_plot.add_point(
		model.x + model.r * cones_solution.trial, **state.params.plot_options['trial-point'])
	state.current_plot.add_point(
		model.x + model.r * linear_solution.trial, **state.params.plot_options['projected-gradient-descent'])

	if 'quadratic-buffered' in state.params.tr_heuristics:
		search_quadratic_buffered(state, model, buffered_region, tr_solutions)
	if (
		'penalty-buffered' in state.params.tr_heuristics or
		'convex-penalty-buffered' in state.params.tr_heuristics):
		search_penalty_buffered(state, model, buffered_region, tr_solutions)

	return {
		'linear': linear_solution,
		'trial': cones_solution,
		'heuristics': [
			TrialValue.create(sol.trial, sol.value)
			for sol in tr_solutions
			if sol.type == TrSolution.Types.HEURISTIC
			if sol.name in state.params.tr_heuristics
		],
	}

# state.logger.verbose_json('computed projected gradient', projection)

# Pyomo sometimes returns answers slightly worse than provided
# Thus, we attempt hot start values as well
# trial_value.accept(hotstart.x, model.shifted_objective.evaluate(hotstart.x))

