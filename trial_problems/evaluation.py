import numpy as np

from utils.default_stringable import DefaultStringable


def _is_informative(value):
	return value is not None and not np.isnan(value)


class Evaluation(DefaultStringable):
	def __init__(self):
		self.x = None
		self.objective = None
		self.constraints = None
		self.failure = False

	def has_information(self):
		return (
			np.isfinite(self.objective) and
			np.isfinite(self.constraints).all())

	def copy(self):
		ret = Evaluation()
		ret.x = self.x.copy()
		ret.objective = self.objective
		ret.constraints = [c for c in self.constraints]
		ret.failure = self.failure
		return ret

	def is_feasible(self):
		for ci in self.constraints:
			if not np.isfinite(ci) or ci > 0:
				return False
		return True

	def to_json(self):
		return {
			'x': self.x,
			'objective-value': self.objective,
			'constraint-values': self.constraints,
			'failure': self.failure,
		}

	@staticmethod
	def parse_json(json):
		evaluation = Evaluation()
		evaluation.x = np.array(json['x'])
		evaluation.objective = json['objective-value']
		evaluation.constraints = np.array(json['constraint-values'])
		evaluation.failure = json['failure']
		return evaluation




	# def filter(self):
	# 	if not self.is_feasible():
	# 		return self.fail()
	# 	return self

	# def fail(self):
	# 	evaluation = Evaluation()
	# 	evaluation.x = self.x.copy()
	# 	evaluation.objective = None
	# 	evaluation.constraints = [None for _ in range(len(self.constraints))]
	# 	evaluation.success = False
	# 	return evaluation