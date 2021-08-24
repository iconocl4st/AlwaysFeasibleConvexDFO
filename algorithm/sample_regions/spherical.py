
import numpy as np

from utils.ellipsoid import Ellipsoid
from utils.stochastic_search import StochasticSearchParams, simple_stochastic_search


def construct_maximal_volume_sphere(state, model, br):
	# poly = model.get_all_unshifted_constraints()
	# ssp.objective = lambda x: poly.distance_to_closest_constraint
	unshifted_cones = [
		cone.unshift(model.x, model.r)
		for cone in br.cones]

	ssp = StochasticSearchParams()
	ssp.x0 = model.x

	def objective(x):
		dist_above = state.current_iterate + state.outer_tr_radius - x
		dist_below = x - state.current_iterate + state.outer_tr_radius
		feasible = np.all(dist_above >= 0) and np.all(dist_below >= 0)
		min_dist = max(0, min(np.min(dist_above), np.min(dist_below)))
		for cone in unshifted_cones:
			feasiblei, distance = cone.get_distance_to(x)
			feasible = feasible and feasiblei
			min_dist = min(min_dist, distance)
		return feasible, -min_dist

	ssp.objective = objective
	ssp.initial_radius = model.r / 4.0
	trial_value = simple_stochastic_search(ssp)

	return Ellipsoid.create(
		q=np.eye(model.n),
		center=trial_value.trial,
		r=-trial_value.value
	)
