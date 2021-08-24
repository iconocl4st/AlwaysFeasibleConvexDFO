import os
import sys
import traceback

from algorithm.algorithm import run_algorithm
from algorithm.algorithm_state import AlgorithmState
from settings import EnvironmentSettings
from utils.json_utils import JsonUtils


def run_problem(run_result, problem, dry_run, user_params=None):
	output_folder = run_result.get_output_folder()
	result_file = run_result.get_result_file()

	try:
		if dry_run:
			return

		run_result.ensure_output_directory()
		EnvironmentSettings.remove_files(output_folder)

		state = AlgorithmState.create(problem, output_folder, user_params)
		it_result = run_algorithm(state)
		print(it_result)
		# state.logger.info_json('history', state.history)
		state.logger.flush()

		run_result.status = 'completed' if it_result.completed else 'incomplete'
		run_result.status_details = ('converged' if it_result.converged else 'failed') + ': ' + it_result.description
		run_result.history = state.history
		run_result.num_iterations = state.iteration
		run_result.definiteness = state.convexity_tester

		with open(result_file, 'w') as output:
			JsonUtils.dump(run_result, output)
	except:
		print('Unable to complete')
		traceback.print_exc(file=sys.stdout)
		with open(os.path.join(output_folder, 'exception.txt'), 'w') as exc_out:
			traceback.print_exc(file=exc_out)

