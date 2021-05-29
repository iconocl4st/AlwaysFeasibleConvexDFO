import numpy as np


def _is_informative(value):
	return value is not None and not np.isnan(value)


class Evaluation:
	def __init__(self):
		self.x = None
		self.objective = None
		self.constraints = None
		self.success = None

	def has_information(self):
		return _is_informative(self.objective) or np.any(
			_is_informative(ci) for ci in self.constraints)

	def copy(self):
		ret = Evaluation()
		ret.x = self.x.copy()
		ret.objective = self.objective
		ret.constraints = [c for c in self.constraints]
		ret.success = self.success
		ret.has_information = self.has_information
		return ret

	def is_feasible(self):
		for ci in self.constraints:
			if ci > 0:
				return False
		return True

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

	def to_json(self):
		return {
			'x': self.x,
			'objective-value': self.objective,
			'constraint-values': self.constraints,
			'success': self.success,
		}

	@staticmethod
	def parse_json(json):
		evaluation = Evaluation()
		evaluation.x = np.array(json['x'])
		evaluation.objective = json['objective-value']
		evaluation.constraints = np.array(json['constraint-values'])
		evaluation.success = json['success']
		return evaluation
