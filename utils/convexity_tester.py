import numpy as np


def _get_definiteness(quad, tol=1e-4):
	definiteness = set()
	try:
		for d in np.linalg.eig(quad.Q)[0]:
			if d > tol:
				definiteness.add('+')
			elif d < -tol:
				definiteness.add('-')
			else:
				definiteness.add('0')
		return '/'.join(sorted([d for d in definiteness]))
	except:
		raise


class ConvexityTester:
	def __init__(self, n):
		self.objective_definiteness = set()
		self.constraint_definiteness = [set() for _ in range(n)]

	def add_model(self, model):
		self.add_objective(model.unshifted_objective)
		self.add_constraints(model.unshifted_constraints)

	def add_objective(self, quadratic):
		self.objective_definiteness.add(_get_definiteness(quadratic))

	def add_constraints(self, constraint_quadratics):
		for constraint_quadratic, definiteness_set in zip(constraint_quadratics, self.constraint_definiteness):
			definiteness_set.add(_get_definiteness(constraint_quadratic))

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		return ','.join(
			['f=[' + ','.join([d for d in self.objective_definiteness]) + ']'] + [
				'c' + str(idx) + '=[' + ','.join([d for d in cd]) + ']'
				for idx, cd in enumerate(self.constraint_definiteness)])

	def to_json(self):
		return {
			'objective': [d for d in self.objective_definiteness],
			'constraints': [[d for d in cd] for cd in self.constraint_definiteness]
		}

	@staticmethod
	def parse_json(json_object):
		if json_object is None:
			return None
		checker = ConvexityTester(len(json_object['constraints']))
		checker.objective_definiteness = set(d for d in json_object['objective'])
		checker.constraint_definiteness = [
			set(d for d in definiteness_set)
			for definiteness_set in json_object['constraints']
		]
		return checker

