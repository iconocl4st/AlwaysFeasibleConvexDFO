from hock_schittkowski.problems import HockSchittkowski
from trial_problems.ht_problem import HockSchittkowskiProblem as htp
from trial_problems.infeasible_strategies import InfeasibleStrategies


def filter_problem(problem):
	success, problem = htp.create_schittkowski_problem(
		problem, InfeasibleStrategies.FailWithNoInformation())
	if not success:
		return False
	if problem.n != 2:
		return False
	return True


if __name__ == '__main__':
	print(' '.join(
		str(problem.number)
		for problem in HockSchittkowski.PROBLEMS
		if not filter_problem(problem)
	))
