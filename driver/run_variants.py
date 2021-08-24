from driver.run_hock_schittkowski import run_on
from hock_schittkowski.problems import HockSchittkowski
from trial_problems.infeasible_strategies import InfeasibleStrategies

if __name__ == '__main__':
	strategy = InfeasibleStrategies.FailWithNoInformation()
	for params in [{
	}, {
		'basis': 'linear',
	}, {
		'tr-heuristics': ['penalty-buffered']
	}, {
		'tr-heuristics': ['convex-penalty-buffered']
	}, {
		'tr-heuristics': ['quadratic-buffered']
	}, {
		'on-empty-sample': 'recover'
	}, {
		'sr-strategy': 'max-volume'
	}, {
		'sr-strategy': 'spherical'
	}]:
		# The lagrange interpolation fails for 34...
		# [34, 66, 76, 93, 251, 264, 329, 337, 339, 354, 359]:
		for problem_no in [12]:
			run_on(
				strategy,
				HockSchittkowski.get_problem_by_number(problem_no),
				rerun=False, dry_run=False,
				params=params
			)