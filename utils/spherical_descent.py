import numpy as np

from utils.assertions import make_assertion
from utils.trial_value import TrialValue


class SphereSearchParams:
	def __init__(self):
		self.x0 = None
		self.objective = None
		self.sphere_is_feasible = None
		self.initial_radius = 1
		self.tolerance = 1e-8


def generate_coordinate_descents(n):
	ei = np.zeros(n, dtype=np.float64)
	for i in range(n):
		ei[i] = 1
		yield ei
		ei[i] = -1
		yield ei
		ei[i] = 0


def spherical_descent_search(search_params):
	radius = search_params.initial_radius
	trial_value = TrialValue()
	trial_value.trial = search_params.x0
	trial_value.value = search_params.objective(trial_value.trial)

	while not search_params.sphere_is_feasible(trial_value.trial, radius):
		radius /= 2
		if radius < search_params.tolerance:
			return None

		# make_assertion(radius > search_params.tolerance, "search does not start at a feasible point")

	while radius >= search_params.tolerance:
		for coordinate_direction in generate_coordinate_descents(trial_value.n):
			trial_direction = radius * coordinate_direction
			t = 1
			while t >= 1:
				trial_x = trial_value.trial + t * trial_direction
				feasible = search_params.sphere_is_feasible(trial_x, radius)
				if not feasible:
					break
				value = search_params.objective(trial_x)
				if trial_value.accept(trial_x, value, feasible):
					t *= 2
				else:
					t /= 2
		radius /= 2

	return trial_value




