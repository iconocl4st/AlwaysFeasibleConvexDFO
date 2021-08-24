import numpy as np

from algorithm.tr_subproblem.tr_solution import TrSolution
from utils.assertions import make_assertion
import utils.spherical_descent as scd


def get_initials(state, model):
	yield (state.sample_region.center - model.x) / model.r
	yield model.shifted_x


def _do_sphere_search(state, model, solutions, search_params, search_type):
	if not search_params.sphere_is_feasible(search_params.x0, 1e-12):
		# make_assertion(False, 'initial sphere is not feasible')
		return

	search_params.objective = model.shifted_objective.evaluate

	trial_value = scd.spherical_descent_search(search_params)
	if trial_value is None:
		return

	solutions.append(TrSolution.create(search_type, 'spherical', trial_value))


def linear_sphere_search(state, model, solutions):
	poly = model.get_all_shifted_constraints()

	for x0 in get_initials(state, model):
		search_params = scd.SphereSearchParams()
		search_params.sphere_is_feasible = lambda x, r: \
			poly.contains(x, tolerance=1e-8) and poly.distance_to_closest_constraint(x) >= r
		search_params.x0 = x0
		_do_sphere_search(state, model, solutions, search_params, TrSolution.Types.LINEAR)


def conic_sphere_search(state, model, br, solutions):
	for x0 in get_initials(state, model):
		search_params = scd.SphereSearchParams()
		search_params.sphere_is_feasible = lambda x, r: \
			np.max(x + r) < 1 and np.min(x - r) > -1 and np.all([
				br.cones[i].contains_sphere(x, r)
				for i in range(br.num_active_constraints)
			])
		search_params.x0 = x0
		_do_sphere_search(state, model, solutions, search_params, TrSolution.Types.CONES)
