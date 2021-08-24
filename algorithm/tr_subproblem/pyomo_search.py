import numpy as np

from algorithm.tr_subproblem.tr_solution import TrSolution
from utils.assertions import make_assertion

from pyomo_opt.buffered_tr_subproblem2 import solve_buffered_tr_subproblem, TrustRegionSubproblem


class BufferedHotStart:
	def __init__(self):
		self.x = None
		self.s = None
		self.t = None

	def apply(self, pyomo_model, use_st=True):
		n = len(self.x)
		m = len(self.t)

		for i in range(n):
			pyomo_model.x[i].set_value(self.x[i])
		if not use_st:
			return

		for i in range(m):
			pyomo_model.t[i].set_value(self.t[i])
		for i in range(m):
			for j in range(n):
				pyomo_model.s[i, j].set_value(self.s[i, j])

	@staticmethod
	def create(x, br):
		n = len(x)
		hotstart = BufferedHotStart()
		hotstart.x = x
		hotstart.s = np.zeros([br.num_active_constraints, n])
		hotstart.t = np.zeros(br.num_active_constraints)
		for i in range(br.num_active_constraints):
			_, hotstart.s[i], hotstart.t[i] = br.cones[i].decompose(x)
		return hotstart


def pyomo_search(state, model, br, solutions):
	solutions_to_add = []

	tr_subproblem = TrustRegionSubproblem()
	tr_subproblem.m = br.num_active_constraints
	tr_subproblem.n = model.shifted_A.shape[1]
	tr_subproblem.x = model.shifted_x
	tr_subproblem.radius = model.shifted_r
	tr_subproblem.ws = br.ws[br.active_indices]
	tr_subproblem.unit_gradients = model.shifted_A[br.active_indices]
	make_assertion(
		tr_subproblem.m == 0 or np.max(abs(1 - np.linalg.norm(tr_subproblem.unit_gradients, axis=1))) < 1e-12,
		'gradients not normalized')
	tr_subproblem.beta = br.bdpb
	tr_subproblem.objective = model.shifted_objective
	tr_subproblem.tol = 1e-8

	for solution in solutions:
		if solution.type != TrSolution.Types.CONES:
			continue
		tr_subproblem.hotstart = BufferedHotStart.create(solution.trial, br)
		success, trial, value = solve_buffered_tr_subproblem(tr_subproblem, state.logger)
		if not success:
			print('unable to compute trial step')
			continue

		pyomo_solution = TrSolution()
		pyomo_solution.type = TrSolution.Types.CONES
		pyomo_solution.name = 'pyomo'
		pyomo_solution.trial = trial
		pyomo_solution.value = value
		solutions_to_add.append(pyomo_solution)

	solutions.extend(solutions_to_add)
