import json
import os
import re
import sys

from algorithm.iteration_result import IterationResult
from settings import EnvironmentSettings
import traceback

from trial_problems.simple_problems import TestProblems
from algorithm.algorithm_state import AlgorithmState
from algorithm.algorithm import run_algorithm

from hott_schittowski.problems import HottSchittowski
from rerun import get_state_from
from run_algorithm import run_problem
from utils.run_result import RunResult, RunParams


def has_converged(result_file):
	if not os.path.exists(result_file):
		return False

	with open(result_file, 'r') as result_input:
		run_result = RunResult.parse_json(json.load(result_input))

	return run_result.status == 'completed' and 'converged' in run_result.status_details


def run_on_all_problems(rerun, dry_run):
	for ht in HottSchittowski.PROBLEMS:
		run_on(ht, dry_run, rerun)


def run_on(ht_problem, rerun, dry_run):
	if ht_problem.number in [359, 67]:
		print('TODO: These problems take a long time to run...')
		return

	success, test_problem = TestProblems.HottSschittowskiProblem.create_schittowski_problem(ht_problem)
	if not success:
		print('Unable to create problem for ' + str(ht_problem.number))
		return

	run_result = RunResult.create('always_feasible', RunParams.create({}), ht_problem)
	if not rerun and has_converged(run_result.get_result_file()):
		print(ht_problem.number, 'has already converged')
		return

	print("Running on ht=", ht_problem.number, 'dim=', ht_problem.n)
	run_problem(run_result, test_problem, dry_run)


if __name__ == '__main__':
	# run_on_all_problems(rerun=False, dry_run=False)

	# 33, 34, 43, 44, 66, 71, 268
	for problem_no in [93]:
		run_on(HottSchittowski.get_problem_by_number(problem_no), rerun=False, dry_run=False)
