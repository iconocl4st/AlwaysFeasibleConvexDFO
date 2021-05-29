from hott_schittowski.problems import HottSchittowski
from settings import EnvironmentSettings
from utils.convexity_tester import ConvexityTester
from utils.formatting import Formatting
import numpy as np
import os

from utils.history import History


class RunParams:
	def __init__(self):
		self.map = {}

	def set(self, param_name, param_value):
		self.map[param_name] = param_value

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		return ','.join(str(key) + '=' + str(value) for key, value in self.map.items())

	def serialize(self):
		if len(self.map) == 0:
			return 'no_params'

		return '_'.join(
			map(
				lambda x: str(x[1]),
				sorted([(key, value) for key, value in self.map.items()], key=lambda x: x[0])))

	def to_json(self):
		return {'parameter-map': self.map}

	@staticmethod
	def create(params_map):
		params = RunParams()
		params.map = params_map
		return params

	@staticmethod
	def parse_json(json_object):
		params = RunParams()
		params.map = json_object['parameter-map']
		return params


class Column:
	def __init__(self, name, getter):
		self.name = name
		self.getter = getter


class RunResult:
	COLUMNS = [
		Column('#', lambda run: run.ht_problem.number),
		Column('algorithm', lambda run: run.algorithm_name),
		# Column('params', lambda run: run.run_params),
		Column('n', lambda run: run.ht_problem.n),
		Column('# succ', lambda run: run.history.get_successful_num()),
		Column('# fail', lambda run: run.history.get_unsuccessful_num()),

		Column('optimal', lambda run: 'yes' if run.is_optimal() else 'no'),

		Column('min', lambda run: Formatting.format_float(run.get_minimum_evaluation().objective)),
		Column('true min', lambda run: Formatting.format_float(run.ht_problem.solution.fstar)),

		Column('minimizer', lambda run: Formatting.format_vector(run.get_minimum_evaluation().x)),
		Column('true minimizer', lambda run: Formatting.format_vector(run.ht_problem.solution.xstar)),

		Column('x-error', lambda run: Formatting.format_float(run.get_error_in_x())),

		Column('status', lambda run: run.status),
		Column('details', lambda run: run.status_details),
		Column('# iter', lambda run: run.num_iterations),
		Column('definity', lambda run: run.definiteness),
	]

	def __init__(self):
		self.algorithm_name = None
		self.ht_problem = None
		self.history = None
		self.status = None
		self.status_details = None
		self.num_iterations = None
		self.min_cache = None
		self.run_params = None
		self.definiteness = None

	def ensure_output_directory(self):
		os.makedirs(self.get_output_folder(), exist_ok=True)

	def unique_identifier(self):
		return str(self.ht_problem.number) + '_' + self.algorithm_name + '_' + self.run_params.serialize()

	def get_output_folder(self):
		return EnvironmentSettings.get_output_path(['runs', self.unique_identifier()])

	def get_result_file(self):
		return os.path.join(self.get_output_folder(), 'result.json')

	def has_evaluations(self):
		return self.history.get_successful_num() > 0

	def get_minimum_evaluation(self):
		if self.min_cache is None:
			self.min_cache = self.history.get_minimum_evaluation()
		return self.min_cache

	def get_error_in_x(self):
		return np.linalg.norm(self.get_minimum_evaluation().x - self.ht_problem.solution.xstar)

	def is_optimal(self):
		# Maybe I should use the difference in x
		return (
			abs(self.get_minimum_evaluation().objective - self.ht_problem.solution.fstar) /
			max(1, abs(self.ht_problem.solution.fstar)) < 1e-3)

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		return ', '.join(n + '=' + str(s) for n, s in zip(RunResult.get_headers(), self.to_row()))

	@property
	def dimension(self):
		return self.ht_problem.n

	def to_row(self):
		return [column.getter(self) for column in RunResult.COLUMNS]

	def to_json(self):
		return {
			'algorithm-name': self.algorithm_name,
			'problem-number': self.ht_problem.number,
			'history': self.history,
			'status': self.status,
			'status-details': self.status_details,
			'num-iterations': self.num_iterations,
			'params': self.run_params,
			'definity': self.definiteness,
		}

	@staticmethod
	def parse_json(json_object):
		result = RunResult()
		result.algorithm_name = json_object['algorithm-name']
		result.ht_problem = HottSchittowski.get_problem_by_number(json_object['problem-number'])
		result.history = History.parse_json(json_object['history'])
		result.status = json_object['status']
		result.status_details = json_object['status-details']
		result.num_iterations = json_object['num-iterations']
		result.run_params = RunParams.parse_json(json_object['params'])
		result.definiteness = ConvexityTester.parse_json(json_object['definity'])
		return result

	@staticmethod
	def get_headers():
		return [column.name for column in RunResult.COLUMNS]

	@staticmethod
	def create(algorithm_name, run_params, ht_problem):
		result = RunResult()
		result.algorithm_name = algorithm_name
		result.run_params = run_params
		result.ht_problem = ht_problem
		result.history = History()
		result.num_iterations = 0
		return result
