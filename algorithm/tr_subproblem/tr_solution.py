import numpy as np


class TrSolution:
	class Types:
		LINEAR = 'linear'
		CONES = 'cones'
		HEURISTIC = 'heuristic'

	def __init__(self):
		self.type = None
		self.name = None
		self.trial = None
		self.value = None

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return ('[' +
			'type=' + str(self.type) + ',' +
			'name=' + str(self.name) + ',' +
			'trial=' + str(self.trial) + ',' +
			'value=' + str(self.value) + ']')

	@staticmethod
	def create(type, name, trial_value):
		sol = TrSolution()
		sol.type = type
		sol.name = name
		sol.trial = trial_value.trial
		sol.value = trial_value.value
		return sol
