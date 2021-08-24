import numpy as np

from algorithm.tr_subproblem.tr_solution import TrSolution

import utils.stochastic_search as sss


def conic_stochastic_search(model, br, solutions):
	# This one is time consuming...
	search_params = sss.StochasticSearchParams()
	search_params.num_iterations = 100
	search_params.x0 = model.shifted_x
	search_params.objective = lambda x: tuple([
		np.max(x) < 1 and np.min(x) > -1 and np.all([
			br.cones[i].decompose(x)[0]
			for i in range(br.num_active_constraints)
		]),
		model.shifted_objective.evaluate(x)])
	trial_value = sss.simple_stochastic_search(search_params)

	solutions.append(TrSolution.create(
		TrSolution.Types.CONES,
		'stochastic',
		trial_value
	))


def linear_stochastic_search(model, solutions, tol=1e-8):
	search_params = sss.StochasticSearchParams()
	search_params.x0 = model.shifted_x
	search_params.tolerance = tol
	search_params.objective = lambda x: tuple([
		np.max(x) < 1 and np.min(x) > -1 and np.all(model.shifted_A @ x <= model.shifted_b + tol),
		model.shifted_objective.evaluate(x)])

	def multi_eval(x):
		obj = model.shifted_objective.multi_eval(x)
		infeas = np.max((x @ model.shifted_A.T - model.shifted_b), axis=1) >= 0
		infeas = np.logical_or(infeas, np.max(x, axis=1) > model.shifted_r)
		infeas = np.logical_or(infeas, np.min(x, axis=1) < model.shifted_r)
		obj[infeas] = np.nan
		return obj

	search_params.multi_eval = multi_eval
	trial_value = sss.multi_eval_stochastic_search(search_params)

	solutions.append(TrSolution.create(
		TrSolution.Types.LINEAR,
		'stochastic',
		trial_value
	))
