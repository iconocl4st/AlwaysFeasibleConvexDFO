import abc
from abc import ABC

import numpy as np


class InfeasibleStrategies:
	FAIL_WITH_NO_INFORMATION = 'fail-with-no-information'
	FAIL_WITH_GARBAGE = 'fail-with-garbage'
	FAIL_WITH_INFORMATION = 'fail-with-information'
	SUCCEED = 'no-failures'

	@staticmethod
	def parse_json(json_object):
		if json_object is None:
			return None
		return InfeasibleStrategies.get_infeasible_strategy(json_object['type'])

	@staticmethod
	def get_infeasible_strategy(strategy_type):
		if strategy_type == InfeasibleStrategies.FAIL_WITH_NO_INFORMATION:
			return InfeasibleStrategies.FailWithNoInformation()
		elif strategy_type == InfeasibleStrategies.FAIL_WITH_GARBAGE:
			return InfeasibleStrategies.FailWithGarbage()
		elif strategy_type == InfeasibleStrategies.FAIL_WITH_INFORMATION:
			return InfeasibleStrategies.FailWithInformation()
		elif strategy_type == InfeasibleStrategies.SUCCEED:
			return InfeasibleStrategies.Succeed()
		else:
			raise Exception('Unable to parse: ' + str(strategy_type))

	class InfeasibleStrategy(ABC):
		def __init__(self, name):
			self.name = name

		def __str__(self):
			return self.name

		def __repr__(self):
			return self.__str__()

		def apply(self, evaluation):
			if evaluation.is_feasible():
				evaluation.success = True
				return evaluation
			evaluation.success = False
			self.handle_infeasible_evaluation(evaluation)
			return evaluation

		@abc.abstractmethod
		def handle_infeasible_evaluation(self, evaluation):
			raise Exception('implement me')

		def to_json(self):
			return {'type': self.name}

	class FailWithNoInformation(InfeasibleStrategy):
		def __init__(self):
			super().__init__(InfeasibleStrategies.FAIL_WITH_NO_INFORMATION)

		def handle_infeasible_evaluation(self, evaluation):
			evaluation.objective = np.nan
			evaluation.constraints = [np.nan for _ in range(len(evaluation.constraints))]

	class FailWithGarbage(InfeasibleStrategy):
		def __init__(self):
			super().__init__(InfeasibleStrategies.FAIL_WITH_GARBAGE)

		def handle_infeasible_evaluation(self, evaluation):
			evaluation.objective = 1e300
			evaluation.constraints = [1.0 for _ in range(len(evaluation.constraints))]

	class FailWithInformation(InfeasibleStrategy):
		def __init__(self):
			super().__init__(InfeasibleStrategies.FAIL_WITH_INFORMATION)

		def handle_infeasible_evaluation(self, e):
			pass

	class Succeed(InfeasibleStrategy):
		def __init__(self):
			super().__init__(InfeasibleStrategies.SUCCEED)

		def handle_infeasible_evaluation(self, e):
			e.success = True

