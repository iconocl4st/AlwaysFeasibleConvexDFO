import numpy as np


class StochasticSearchParams:
	def __init__(self):
		self.x0 = None
		self.objective = None
		self.initial_radius = 1
		self.num_iterations = 1000
		self.tolerance = 1e-6


def simple_stochastic_search(search_params):
	radius = search_params.initial_radius
	current_x = search_params.x0
	feasible, current_f = search_params.objective(current_x)
	assert feasible, "search does not start at a feasible point"

	while radius >= search_params.tolerance:
		count = 0
		while count < search_params.num_iterations:
			trial_direction = (2 * np.random.random(len(current_x)) - 1)
			trial_direction /= np.linalg.norm(trial_direction)
			trial_direction *= radius
			while True:
				trial_x = current_x + trial_direction
				feasible, value = search_params.objective(trial_x)
				if not feasible or value >= current_f:
					count += 1
					break
				current_x = trial_x
				current_f = value
				count = 0
		radius /= 2

	return current_x, current_f




