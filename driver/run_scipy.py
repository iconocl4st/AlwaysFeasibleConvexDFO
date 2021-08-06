import scipy.optimize

from hott_schittowski import problems
from trial_problems.ht_problem import HottSschittowskiProblem
from trial_problems.infeasible_strategies import InfeasibleStrategies

import traceback
import numpy as np

import scipy.optimize

from hott_schittowski.problems import HottSchittowski
from utils.json_utils import JsonUtils
from utils.run_result import RunResult
from utils.run_result import RunParams


def get_bb(problem, history):
	def bb(x):
		evaluation = problem.evaluate(x)
		history.add_evaluation(-1, evaluation)
		return evaluation.objective
	return bb


# def callback(*params, **kparams):
# 	return False

def create_nonlinear_constraint(problem, index):
	return scipy.optimize.NonlinearConstraint(
		fun=lambda x: problem.evaluate(x).constraints[index],
		lb=-np.inf,
		ub=0,
		# keep_feasible=True
	)


def ensure_bounds(b, dim):
	return b if b is not None else [None] * dim


def run_scipy(ht_problem, params, strategy):
	success, problem = HottSschittowskiProblem.create_schittowski_problem(ht_problem, strategy)
	if not success:
		return

	run_result = RunResult.create(
		'scipy',
		RunParams.create({}),
		ht_problem)
	print(ht_problem.number, ht_problem.n, strategy)
	try:
		output = scipy.optimize.minimize(
			fun=get_bb(problem, run_result.history),
			x0=ht_problem.initial.x0,
			# args=(),
			# method=params.map['method'],
			# jac=None,
			# hess=None,
			# hessp=None,
			bounds=scipy.optimize.Bounds(
				lb=problems.Bounds.to_pdfo(ht_problem.bounds.lb, -np.inf, ht_problem.n),
				ub=problems.Bounds.to_pdfo(ht_problem.bounds.ub, np.inf, ht_problem.n),
				# keep_feasible=True
			),
			constraints=[
				create_nonlinear_constraint(problem, i)
				for i in range(problem.num_constraints)
			],
			tol=1e-4,
			# callback=None,
			options={
				'maxiter': 10000,
				'disp': False,
			},
		)
	except:
		traceback.print_exc()
		return

	print(output)

	run_result.status_details = output.message
	run_result.num_iterations = output.nit
	run_result.status = 'successful' if output.success else 'failed'

	run_result.ensure_output_directory()
	result_file = run_result.get_result_file()
	with open(result_file, 'w') as output:
		JsonUtils.dump(run_result, output)


if __name__ == '__main__':
	run_params = RunParams.create({
		# 'method': 'Powell',
	})
	strategy = InfeasibleStrategies.Succeed()
	for ht_problem in HottSchittowski.PROBLEMS:
		if ht_problem.number == 67:
			continue
		run_scipy(ht_problem, run_params, strategy)

