#
#
# from settings import EnvironmentSettings
#
#
# from trial_problems.simple_problems import TestProblems
# from algorithm.algorithm_state import AlgorithmState
# from algorithm.algorithm import run_algorithm
#
#
# def test_algorithm():
# 	for test_problem in [
# 		TestProblems.EasyTrialProblem(),
# 		TestProblems.NarrowTrialProblem(),
# 		TestProblems.Rosenbrock(),
# 	]:
# 		EnvironmentSettings.set_all_output_prefix(test_problem.get_name())
# 		EnvironmentSettings.remove_images()
#
# 		state = AlgorithmState.create(test_problem, EnvironmentSettings.get_current_prefix())
# 		run_result = run_algorithm(state)
# 		state.history.create_plot(state.plotter)
# 		state.logger.log_json('history', state.history)
# 		state.logger.flush()
# 		print(run_result)
#
#
# if __name__ == '__main__':
# 	test_algorithm()
