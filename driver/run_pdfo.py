from pdfo import pdfo

from pdfo import Bounds as PdfoBounds
from pdfo import NonlinearConstraint

import traceback

from hott_schittowski.problems import HottSchittowski
from trial_problems.ht_problem import HottSschittowskiProblem
from trial_problems.infeasible_strategies import InfeasibleStrategies
from utils.json_utils import JsonUtils
from utils.run_result import RunResult, RunParams


def create_objective(problem, run_result):
	def objective_function(x):
		evaluation = problem.evaluate(x)
		run_result.history.add_evaluation(-1, evaluation)
		return evaluation.objective
	return objective_function


def create_constraint(problem, index):
	return NonlinearConstraint(fun=lambda x: problem.evaluate(x).constraints[index], lb=None, ub=0.0)


def run_on(ht_problem):
	strategy = InfeasibleStrategies.FailWithGarbage()
	success, problem = HottSschittowskiProblem.create_schittowski_problem(ht_problem, strategy)
	if not success:
		return

	run_result = RunResult.create('pdfo', RunParams.create({}), ht_problem)

	print("Running on ht=", ht_problem.number, 'dim=', ht_problem.n)
	# Solvers: UOBYQA, NEWUOA, BOBYQA, LINCOA, COBYLA
	pdfo(
		fun=create_objective(problem, run_result),
		x0=ht_problem.initial.x0,
		# method=,
		bounds=PdfoBounds(
			lb=ht_problem.bounds.to_pdfo_lb(ht_problem.n),
			ub=ht_problem.bounds.to_pdfo_ub(ht_problem.n)),
		constraints=[
			create_constraint(problem, index)
			for index in range(problem.num_constraints)],
		options={
			'honour_x0': True,  # only used by BOBYQA
			'quiet': False,
			# rhobeg: trust region radius
			# rhoend: final trust region radius, default is 1e-6
			# maxfev: maximum number of objective function calls
			# npt: number of interpolation points (for NEWUOA, BOBYQA, or LINCOA)
			# ftarget: stop if objective becomes lower than this value
			# scale: scale according to the bound constraints
			# classical: call classical code, defaults to False
			# debug: defaults to False
			# chkfunval: bool, optional
			#     Flag used when debugging. If both `options['debug']` and `options['chkfunval']` are True, an extra
			#     function/constraint evaluation would be performed to check whether the returned values of objective
			#     function and constraint match the returned x. By default, it is False.
		}
	)

	run_result.ensure_output_directory()
	output_file = run_result.get_result_file()
	with open(output_file, 'w') as output_stream:
		JsonUtils.dump(run_result, output_stream)


def run_on_all_problems():
	for ht in HottSchittowski.PROBLEMS:
		try:
			run_on(ht)
		except:
			traceback.print_exc()


if __name__ == '__main__':
	run_on_all_problems()
	# run_on(HottSchittowski.get_problem_by_number(215))
