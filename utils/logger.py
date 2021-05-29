import numpy as np
import os

from utils.json_utils import JsonUtils


class Logger:
	VERBOSE = 0
	INFO = 1
	WARNING = 2

	def __init__(self):
		self.steps = []
		self.log_file = None
		self.log_level = None

	def flush(self):
		self.log_file.flush()

	def get_base(self):
		return "[" + "] [".join(self.steps) + "] "

	def start_step(self, step):
		self.steps.append(step)
		self.write(self.get_base() + "Begin", Logger.VERBOSE)

	def stop_step(self):
		self.write(self.get_base() + "End", Logger.VERBOSE)
		self.steps = self.steps[:-1]

	def log_trial_point_evaluation(self, solution, evaluation, rho):
		self.write((
				self.get_base() +
				"Evaluated " + str(evaluation.objective_value) + " at " + str(evaluation.x) +
				" instead of " + str(solution.trial_objective_value) +
				" for a rho= " + str(rho)
		), Logger.INFO)

	def log_matrix(self, message, mat, log_level=VERBOSE):
		self.log_message(message + "\n" + np.array2string(mat), log_level)

	def info(self, message):
		self.log_message(message, Logger.INFO)

	def verbose(self, message):
		self.log_message(message, Logger.VERBOSE)

	def log_message(self, message, log_level):
		self.write(self.get_base() + message, log_level)

	def verbose_json(self, str, json_object):
		self.log_json(str, json_object, Logger.VERBOSE)

	def info_json(self, str, json_object):
		self.log_json(str, json_object, Logger.INFO)

	def log_json(self, str, json_object, log_level):
		self.write(self.get_base() + str, log_level)
		self.write(self.get_base() + JsonUtils.dumps(json_object), log_level)

	def log_trial_point(self):
		pass

	def log_rho(self):
		pass

	def log_new_center(self):
		pass

	def log_radius_change(self):
		pass

	def write(self, str, log_level):
		if log_level >= self.log_level:
			print(str)
		self.log_file.write(str + "\n")
		self.log_file.flush()

	def close(self):
		self.log_file.close()

	@staticmethod
	def create(log_path):
		logger = Logger()
		os.makedirs(os.path.dirname(log_path), exist_ok=True)
		logger.log_file = open(log_path, "w")
		logger.log_level = Logger.INFO
		return logger
