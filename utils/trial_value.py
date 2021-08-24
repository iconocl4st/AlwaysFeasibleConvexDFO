
# TODO: Use this in more places
from utils.default_stringable import DefaultStringable


class TrialValue:
	def __init__(self):
		self.trial = None
		self.value = None

	def has_value(self):
		return self.trial is not None

	@property
	def n(self):
		return len(self.trial)

	def accept(self, alt_x, alt_value, feasible=True):
		if not feasible:
			return False
		if self.trial is not None and self.value <= alt_value:
			return False
		self.trial = alt_x
		self.value = alt_value
		return True

	@staticmethod
	def create(x, val):
		trial_value = TrialValue()
		trial_value.trial = x
		trial_value.value = val
		return trial_value

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		return '[value=' + str(self.value) + ',trial=' + str(self.trial) + ']'

	def to_json(self):
		return {
			'trial': self.trial,
			'value': self.value
		}

