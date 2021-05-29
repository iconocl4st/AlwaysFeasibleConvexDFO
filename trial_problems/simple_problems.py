import traceback

import numpy as np

from hott_schittowski.problems import HottSchittowski
from utils.finite_difference import construct_initial_ellipsoid
from utils.evaluation import Evaluation


def find_feasible_start(problem):
	if not problem.initial.is_feasible:
		return False, None, None, None
	if problem.number == 215:
		return True, np.array([0.5, 1.5]), np.array([[1.0, 0.0], [0.0, 1.0]]), 0.5
	elif problem.number == 218:
		return True, np.array([9, 100]), np.array([[1.0, 0.0], [0.0, 1.0]]), 0.75
	elif problem.number == 221:
		return True, np.array([0.2, 0.25]), np.array([[1.0, 0.0], [0.0, 1.0]]), 0.1
	elif problem.number == 222:
		return True, np.array([1.4, 0.1]), np.array([[1.0, 0.0], [0.0, 1.0]]), 0.1
	elif problem.number == 223:
		return True, np.array([0.1, 3.4]), np.array([[1.0, 0.0], [0.0, 1.0]]), 0.025
	elif problem.number == 224:
		return True, np.array([1.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0]]), 1.0
	elif problem.number == 225:
		return True, np.array([3.0, 0.5]), np.array([[1.0, 0.0], [0.0, 1.0]]), 1.0
	elif problem.number == 226:
		return True, np.array([0.7, 0.2]), np.array([[1.0, 0.0], [0.0, 1.0]]), 0.2
	elif problem.number == 227:
		return True, np.array([0.5, 0.5]), np.array([[1.0, 0.0], [0.0, 1.0]]), 0.1
	elif problem.number == 228:
		return True, np.array([0, 0]), np.array([[1.0, 0.0], [0.0, 1.0]]), 0.5
	elif problem.number == 231:
		return True, np.array([-1.2, 1]), np.array([[1.0, 0.0], [0.0, 1.0]]), 1/3.0
	elif problem.number == 232:
		return True, np.array([2, 0.5]), np.array([[1.0, 0.0], [0.0, 1.0]]), 1/3.0
	elif problem.number == 233:
		return True, np.array([1.2, 1]), np.array([[1.0, 0.0], [0.0, 1.0]]), 0.1
	elif problem.number == 249:
		return True, np.array([2, 1, 1]), np.eye(3), 1

	success, constraints = problem.get_all_le_constraints()
	if not success:
		return False, None, None, None
	success, center, q, radius = construct_initial_ellipsoid(constraints, problem.initial.x0)
	if not success:
		return False, None, None, None
	return success, center, q, radius


class TestProblems:
	class HottSschittowskiProblem:
		def __init__(self, problem, q, r, center, constraints):
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

		def to_json(self):
			return {'type': 'hott-schittowski', 'problem-number': self.problem.number}

		def evaluate(self, x):
			evaluation = Evaluation()
			evaluation.x = x
			evaluation.objective = self.problem.objective(x)
			evaluation.constraints = np.array([
				c(x)
				for c in self.constraints
			])
			evaluation.success = True
			return evaluation.filter()

		@staticmethod
		def create_schittowski_problem(problem):
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

			return True, TestProblems.HottSschittowskiProblem(problem, q, r, center, constraints)

	class EasyTrialProblem:
		def get_name(self):
			return 'simple'

		@property
		def num_constraints(self):
			return 1

		def get_initial_x(self):
			return np.array([0, 5])

		def get_initial_r(self):
			return 1.0

		def get_initial_q(self):
			return np.diag([2, 1])

		def get_initial_center(self):
			return self.get_initial_x()

		def get_initial_delta(self):
			return self.get_initial_r()

		def to_json(self):
			return {'type': 'simple'}

		def evaluate(self, x):
			evaluation = Evaluation()
			evaluation.x = x
			evaluation.objective = x[0] ** 2 + (x[1] + 5) ** 2
			evaluation.constraints = np.array([
				x[0] ** 2 - x[1]
			])
			evaluation.success = True
			return evaluation.filter()

	class NarrowTrialProblem:
		def get_name(self):
			return 'narrow'

		def __init__(self, a=0.5):
			self.a = a

		@property
		def num_constraints(self):
			return 2

		def get_initial_x(self):
			return np.array([5, 0])

		def get_initial_r(self):
			return 1

		def get_initial_q(self):
			return np.identity(2)

		def get_initial_center(self):
			return self.get_initial_x()

		def get_initial_delta(self):
			return self.get_initial_r()

		def to_json(self):
			return {'type': 'narrow'}

		def evaluate(self, x):
			evaluation = Evaluation()
			evaluation.x = x

			evaluation.objective = x[0] + 10 * (x[1] - 0.75 * self.a * x[0] * np.sin(x[0])) ** 2
			# x[1] <= +a * x[0] => -a * x[0] + x[1] <= 0
			# x[1] >= -a * x[0] => -a * x[0] - x[1] <= 0
			evaluation.constraints = np.array([
				-self.a * x[0] + x[1],
				-self.a * x[0] - x[1],
			])
			evaluation.success = True
			return evaluation.filter()

	class Rosenbrock:
		def get_name(self):
			return 'rosenbrock'

		@property
		def num_constraints(self):
			return 3

		def get_initial_x(self):
			return np.array([0, 5])

		def get_initial_r(self):
			return 1

		def get_initial_q(self):
			return np.identity(2)

		def get_initial_center(self):
			return self.get_initial_x()

		def get_initial_delta(self):
			return self.get_initial_r()

		def to_json(self):
			return {'type': 'rosenbrock'}

		def evaluate(self, x):
			evaluation = Evaluation()
			evaluation.x = x
			evaluation.objective = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
			evaluation.constraints = np.array([
				x[0] ** 2 - x[1],
				(x[0] - 0) ** 2 - (x[1] - 5) ** 2 - 5 ** 2,
				x[0] + x[1] - 100,
			])
			evaluation.success = True
			return evaluation.filter()

	@staticmethod
	def parse_json(json):
		if json['type'] == 'hott-schittowski':
			ht_problem = HottSchittowski.get_problem_by_number(json['problem-number'])
			success, problem = TestProblems.HottSschittowskiProblem.create_schittowski_problem(ht_problem)
			assert success, 'unable to create hott schittowski problem ' + str(json['problem-number'])
			return problem
		raise Exception('Unrecognized problem type: ' + str(json['type']))

