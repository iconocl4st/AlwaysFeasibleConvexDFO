import numpy as np

from utils.assertions import make_assertion
from utils.trial_value import TrialValue


class StochasticSearchParams:
	def __init__(self):
		self.x0 = None
		self.objective = None
		self.initial_radius = 1
		self.num_iterations = 1000
		self.tolerance = 1e-6
		self.multi_eval = None
		self.required_objective = None

	def completed(self, radius, trial_value):
		return (
			radius < self.tolerance or
			(
				self.required_objective is not None and
				trial_value.value is not None and
				trial_value.value < self.required_objective
			)
		)


def simple_stochastic_search(search_params):
	radius = search_params.initial_radius
	trial_value = TrialValue()
	trial_value.trial = search_params.x0
	feasible, trial_value.value = search_params.objective(trial_value.trial)
	make_assertion(feasible, "search does not start at a feasible point")

	while not search_params.completed(radius, trial_value):
		count = 0
		while count < search_params.num_iterations:
			trial_direction = (2 * np.random.random(trial_value.n) - 1)
			trial_direction /= np.linalg.norm(trial_direction)
			trial_direction *= radius
			while not search_params.completed(radius, trial_value):
				trial_x = trial_value.trial + trial_direction
				feasible, value = search_params.objective(trial_x)
				if not trial_value.accept(trial_x, value, feasible):
					count += 1
					break
				count = 0
		radius /= 2

	return trial_value


def multi_eval_stochastic_search(search_params):
	radius = search_params.initial_radius

	trial_value = TrialValue()
	trial_value.trial = search_params.x0
	feasible, trial_value.value = search_params.objective(trial_value.trial)
	make_assertion(feasible, "search does not start at a feasible point")

	while not search_params.completed(radius, trial_value):
		decreased = True
		while decreased and not search_params.completed(radius, trial_value):
			decreased = False
			ds = (2 * np.random.random((search_params.num_iterations, trial_value.n)) - 1)
			ds = radius * ds / np.linalg.norm(ds, axis=1)[:, np.newaxis]
			obj = search_params.multi_eval(trial_value.trial + ds)
			if np.all(np.isnan(obj)):
				radius /= 2
				break

			min_idx = np.nanargmin(obj)
			while not search_params.completed(radius, trial_value):
				trial_x = trial_value.trial + ds[min_idx]
				feasible, value = search_params.objective(trial_x)
				if trial_value.accept(trial_x, value, feasible):
					decreased = True
				else:
					break
		radius /= 2

	return trial_value


def stochastic_projection(x0, A, b, tol=None):
	if tol is None:
		tol = 1e-12
	feasibility_search = StochasticSearchParams()
	feasibility_search.x0 = x0
	feasibility_search.objective = lambda x: (True, np.max(A @ x - b))
	feasibility_search.multi_eval = lambda x: np.max((x @ A.T - b), axis=1)
	feasibility_search.required_objective = -tol
	feasibility_search.tolerance = tol
	feasibility_trial_value = multi_eval_stochastic_search(feasibility_search)
	if feasibility_trial_value.value >= tol:
		feasibility_search.x0 = np.zeros_like(x0)
		feasibility_trial_value = multi_eval_stochastic_search(feasibility_search)

	if feasibility_trial_value.value >= tol:
		return None

	projection_search = StochasticSearchParams()
	projection_search.x0 = feasibility_trial_value.trial
	projection_search.objective = lambda x: (np.max(A @ x - b) <= tol, np.linalg.norm(x - x0))

	def multi_eval(x):
		obj = np.linalg.norm(x - x0, axis=1)
		infeas = np.max((x @ A.T - b), axis=1) >= tol
		obj[infeas] = np.nan
		return obj

	projection_search.multi_eval = multi_eval
	projection_search.tolerance = tol
	projection_value = multi_eval_stochastic_search(feasibility_search)
	return projection_value
