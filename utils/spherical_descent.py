import numpy as np


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
	current_x = search_params.x0
	current_f = search_params.objective(current_x)

	while not search_params.sphere_is_feasible(current_x, radius):
		radius /= 2
		if radius < search_params.tolerance:
			print('about to fail assertion')
			search_params.sphere_is_feasible(current_x, radius)
		assert radius > search_params.tolerance, "search does not start at a feasible point"

	while radius >= search_params.tolerance:
		for coordinate_direction in generate_coordinate_descents(len(current_x)):
			trial_direction = radius * coordinate_direction
			while True:
				trial_x = current_x + trial_direction
				feasible = search_params.sphere_is_feasible(trial_x, radius)
				if not feasible:
					break
				value = search_params.objective(trial_x)
				if value >= current_f:
					break
				current_x = trial_x
				current_f = value
		radius /= 2

	return current_x, current_f




