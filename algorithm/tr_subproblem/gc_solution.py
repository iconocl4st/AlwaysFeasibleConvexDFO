import numpy as np

from utils.generalized_cauchy import GeneralizedCauchyParams
from utils.generalized_cauchy import compute_generalized_cauchy_point

from .tr_solution import TrSolution


def gc_search_for_cauchy(state, model, solutions):
	shifted_poly = model.get_shifted_model_constraints()

	gcp = GeneralizedCauchyParams()
	gcp.radius = model.shifted_r
	gcp.x = model.shifted_x
	gcp.model = model.shifted_objective
	# This might should be divided by the model radius
	gcp.gradient = model.gradient_of_shifted  # / model.r
	gcp.A = shifted_poly.A
	gcp.b = shifted_poly.b
	gcp.cur_val = gcp.model.evaluate(gcp.x)
	gcp.tol = 1e-12
	gcp.plotter = state.plotter
	gcp.logger = state.logger

	projection, _, _, _ = compute_generalized_cauchy_point(gcp)

	if projection is None:
		return

	gc_solution = TrSolution()
	gc_solution.type = TrSolution.Types.LINEAR
	gc_solution.name = 'generalized-cauchy'
	gc_solution.trial = projection
	gc_solution.value = model.shifted_objective.evaluate(projection)

	solutions.append(gc_solution)
