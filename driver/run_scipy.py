from scipy.optimize import minimize

from driver.infeasible_strategies import InfeasibleStrategies


import traceback

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


def run_scipy(ht_problem, strategy):
	success, problem = TestProblems.HottSschittowskiProblem.create_schittowski_problem(ht_problem)
	if not success:
		return
	success, constraints = ht_problem.get_explicit_constraints()
	if not success:
		return

	run_result = RunResult.create(
		'scipy',
		RunParams.create({'strategy': strategy}),
		ht_problem)
	run_result.status = 'failure'
	run_result.status_details = 'run not completed'

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

	run_result.ensure_output_directory()
	result_file = run_result.get_result_file()
	run_result.status = 'successful'
	[_, _, _, _, run_result.num_iterations, run_result.status_details] = output
	with open(result_file, 'w') as output:
		JsonUtils.dump(run_result, output)


if __name__ == '__main__':
	problem_num = 215
	run_params = RunParams.create({
		'strategy': InfeasibleStrategies.FailWithNoInformation(),
		'method': 'BGFS',
	})
	ht_problem = HottSchittowski.get_problem_by_number(problem_num)
	if ht_problem is not None:
		run_scipy(ht_problem, run_params)

