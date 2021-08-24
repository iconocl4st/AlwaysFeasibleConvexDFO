import numpy as np

from algorithm.tr_subproblem.tr_solution import TrSolution
from utils.assertions import make_assertion


def _scale(tol, br, sol):
	tmin = 0.0
	tmax = 1.0
	ret = 0, np.zeros_like(sol.trial)
	make_assertion(br.is_buffered(ret[1], tol=tol), 'no scaling possible')
	for _ in range(100):
		# point = (1 - tmid) * x + tmid * ugc
		tmid = (tmax + tmin) / 2.0
		point = tmid * sol.trial
		if br.is_buffered(point):
			ret = tmid, point
			tmin = tmid
		else:
			tmax = tmid
		if tmax - tmin < tol:
			return ret
	make_assertion(False, 'Binary search should have concluded')


def scale_linear_solutions(model, br, solutions, tol=1e-8):
	solutions_to_add = []

	for solution in solutions:
		if not solution.type == TrSolution.Types.LINEAR:
			continue

		_, trial = _scale(tol, br, solution)

		if trial is None:
			continue
		if np.linalg.norm(trial - model.shifted_x) > 1e-12:
			continue

		sol = TrSolution()
		sol.trial = trial
		sol.type = TrSolution.Types.CONES
		sol.name = 'scaled'
		sol.value = model.shifted_objective.evaluate(sol.trial)
		solutions_to_add.append(sol)

		sol = TrSolution()
		sol.trial = trial / 2.0
		sol.type = TrSolution.Types.CONES
		sol.name = 'half-scaled'
		sol.value = model.shifted_objective.evaluate(sol.trial)
		solutions_to_add.append(sol)

	solutions.extend(solutions_to_add)
