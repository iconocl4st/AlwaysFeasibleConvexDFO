import traceback

import PyNomad
import sys
import os
import traceback
import numpy as np
import json
import re

from hott_schittowski.problems import HottSchittowski
from trial_problems.simple_problems import TestProblems
from utils.formatting import Formatting
from utils.history import History
from utils.json_utils import JsonUtils
from utils.run_result import RunResult
from utils.run_result import RunParams


class PyNomadParams:
	def __init__(self, ht_problem):
		self.display_degree = 0
		self.stats_file = 'stats.txt'
		self.output_type = 'OBJ PB EB'
		self.max_evaluations = 10000
		self.max_time = 60
		self.lbs = ht_problem.bounds.nomad_lb()
		self.ubs = ht_problem.bounds.nomad_ub()

	def to_params(self):
		return [
			'DISPLAY_DEGREE ' + str(self.display_degree),
			'STATS_FILE ' + str(self.stats_file),
			'BB_OUTPUT_TYPE ' + self.output_type,
			'MAX_BB_EVAL ' + str(self.max_evaluations),
			'MAX_TIME ' + str(self.max_time),
		] + self.lbs + self.ubs


class InfeasibleStrategy:
	FAIL_WITH_NO_INFORMATION = 'fail-with-no-information'
	FAIL_WITH_GARBAGE = 'fail-with-garbage'
	FAIL_WITH_INFORMATION = 'fail-with-information'
	SUCCEED = 'no-failures'


def get_bb(problem, constraints, strategy, history):
	def bb(x):
		try:
			npX = np.array([
				x.get_coord(i)
				for i in range(x.get_n())
			], dtype=np.float64)

			evaluation = problem.evaluate(npX)
			history.add_evaluation(-1, evaluation)
			if evaluation.success or strategy == InfeasibleStrategy.SUCCEED:
				x.set_bb_output(0, ht_problem.objective(npX))
				for idx, constraint in enumerate(constraints):
					x.set_bb_output(idx + 1, constraint(npX))
				return 1
			elif strategy == InfeasibleStrategy.FAIL_WITH_NO_INFORMATION:
				return 0
			elif strategy == InfeasibleStrategy.FAIL_WITH_GARBAGE:
				x.set_bb_output(0, 50000)
				for idx, constraint in enumerate(constraints):
					x.set_bb_output(idx + 1, 1.0)
				return 0
			elif strategy == InfeasibleStrategy.FAIL_WITH_INFORMATION:
				x.set_bb_output(0, ht_problem.objective(npX))
				for idx, constraint in enumerate(constraints):
					x.set_bb_output(idx + 1, constraint(npX))
				return 0
			else:
				raise Exception('Unknown case')
		except:
			# print("Unexpected eval error", sys.exc_info()[0])
			return 0
	return bb


def run_hott_schittowski_problem(ht_problem, strategy):
	if not ht_problem.initial.is_feasible:
		return
	if len(ht_problem.constraints) == 0:
		return
	success, problem = TestProblems.HottSschittowskiProblem.create_schittowski_problem(ht_problem)
	if not success:
		return
	success, constraints = ht_problem.get_explicit_constraints()
	if not success:
		return

	run_result = RunResult.create(
		'nomad',
		RunParams.create({'strategy': strategy}),
		ht_problem)
	run_result.status = 'failure'
	run_result.status_details = 'run not completed'

	run_result.ensure_output_directory()
	result_file = run_result.get_result_file()
	with open(result_file, 'w') as output:
		JsonUtils.dump(run_result, output)

	params = PyNomadParams(ht_problem)
	print(ht_problem.number, ht_problem.n, strategy)
	try:
		output = PyNomad.optimize(
			get_bb(problem, constraints, strategy, run_result.history),
			[xi for xi in ht_problem.initial.x0],
			[], [],  # bounds come from the params...
			params.to_params())
	except:
		traceback.print_exc()
		return

	run_result.status = 'successful'
	[_, _, _, _, run_result.num_iterations, run_result.status_details] = output
	with open(result_file, 'w') as output:
		JsonUtils.dump(run_result, output)


if __name__ == '__main__':
	np.seterr(all='raise')
	try:
		problem_num = int(sys.argv[-2])
		strategy = sys.argv[-1]
	except:
		problem_num = 215
		strategy = InfeasibleStrategy.FAIL_WITH_NO_INFORMATION
		print('no problem specified')
	else:
		# if True:
		ht_problem = HottSchittowski.get_problem_by_number(problem_num)
		if ht_problem is not None:
			run_hott_schittowski_problem(ht_problem, strategy)

