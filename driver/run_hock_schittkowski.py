import json
import os
import numpy as np

from hock_schittkowski.problems import HockSchittkowski
from run_algorithm import run_problem
from trial_problems.ht_problem import HockSchittkowskiProblem as htp
from trial_problems.infeasible_strategies import InfeasibleStrategies
from utils.run_result import RunResult, RunParams


def has_converged(result_file):
	if not os.path.exists(result_file):
		return False

	with open(result_file, 'r') as result_input:
		run_result = RunResult.parse_json(json.load(result_input))

	return run_result.status == 'completed' and 'converged' in run_result.status_details


def run_on(strategy, ht_problem, dry_run, rerun, params):
	success, test_problem = htp.create_schittkowski_problem(ht_problem, strategy)
	if not success:
		print('Unable to create problem for ' + str(ht_problem.number))
		return

	run_result = RunResult.create('always_feasible', RunParams.create(params), ht_problem)
	if not rerun and has_converged(run_result.get_result_file()):
		print(ht_problem.number, 'has already converged')
		return

	print("Running on ht=", ht_problem.number, 'dim=', ht_problem.n)
	run_problem(run_result, test_problem, dry_run, params)


def run_on_all_problems(rerun, dry_run, strategy, params):
	shuffled = [ht for ht in HockSchittkowski.PROBLEMS]
	np.random.shuffle(shuffled)
	for ht in shuffled:
		if ht.number not in [33, 34, 66, 76, 93, 105, 113, 249, 264, 329, 337, 339,]:
			continue
		run_on(strategy, ht, dry_run, rerun, params)


if __name__ == '__main__':

	# the unbound radius thing....
	# Re run on problem 12...
	strategy = InfeasibleStrategies.FailWithNoInformation()
	for params in [{
	# }, {
	# 	'basis': 'linear',
	# }, {
	# 	'tr-heuristics': ['penalty-buffered']
	# }, {
	# 	'tr-heuristics': ['convex-penalty-buffered']
	# }, {
	# 	'tr-heuristics': ['quadratic-buffered']
	# }, {
	# 	'on-empty-sample': 'recover'
	# }, {
	# 	'sr-strategy': 'max-volume'
	# }, {
	# 	'sr-strategy': 'spherical'
	}]:
		run_on_all_problems(rerun=False, dry_run=False, strategy=strategy, params=params)
		# 33, 34, 66, 76, 93, 105, 113, 249, 264, 329, 337, 339,
		# 221 is not regular...