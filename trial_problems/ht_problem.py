import traceback

import numpy as np

from trial_problems.base_problem import BaseProblem
from trial_problems.evaluation import Evaluation
from trial_problems.infeasible_strategies import InfeasibleStrategies
from utils.assertions import make_assertion
from utils.finite_difference import find_feasible_start

from hott_schittowski.problems import HottSchittowski


class HottSschittowskiProblem(BaseProblem):
	def __init__(self, inf_strategy, problem, q, r, center, constraints):
		super().__init__(inf_strategy)
		self.problem = problem
		self.q = q
		self.r = r
		self.center = center
		self.constraints = constraints

	@property
	def n(self):
		return len(self.center)

	def get_name(self):
		return 'hott_schittowski_' + str(self.problem.number)

	@property
	def num_constraints(self):
		return len(self.constraints)

	def get_initial_x(self):
		return self.problem.initial.x0

	def get_initial_r(self):
		dist = np.linalg.norm(self.get_initial_x() - self.get_initial_center())
		if dist < 1e-8:
			return self.get_initial_delta()
		return 2 * dist

	def get_initial_q(self):
		return self.q

	def get_initial_center(self):
		return self.center

	def get_initial_delta(self):
		return self.r

	def construct_evaluation(self, x):
		evaluation = Evaluation()
		evaluation.x = x
		evaluation.objective = self.problem.objective(x)
		evaluation.constraints = np.array([
			c(x)
			for c in self.constraints
		])
		return evaluation

	def to_json(self):
		return {
			'type': 'hott-schittowski',
			'problem-number': self.problem.number,
			'strategy': self.inf_strategy
		}

	@staticmethod
	def parse_json(json):
		if json['type'] != 'hott-schittowski':
			raise Exception('Unrecognized problem type: ' + str(json['type']))
		ht_problem = HottSchittowski.get_problem_by_number(json['problem-number'])
		strategy = InfeasibleStrategies.parse_json(json['strategy'])
		success, problem = HottSschittowskiProblem.create_schittowski_problem(ht_problem, strategy)
		make_assertion(success, 'unable to create hott schittowski problem ' + str(json['problem-number']))
		return problem


	@staticmethod
	def create_schittowski_problem(problem, strategy):
		if problem.n > 10:
			return False, None
		if not problem.initial.is_feasible:
			return False, None
		if len(problem.constraints) == 0:
			return False, None

		success, constraints = problem.get_all_le_constraints()
		if not success:
			return False, None

		try:
			success, center, q, r = find_feasible_start(problem)
		except:
			print('unable to find feasible start for problem number', problem.number)
			traceback.print_exc()
			raise
		# if not success:
		# 	return False, None

		return True, HottSschittowskiProblem(strategy, problem, q, r, center, constraints)
