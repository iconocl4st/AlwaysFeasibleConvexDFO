import os
import re
import json
import traceback

from algorithm.algorithm import run_algorithm
from algorithm.algorithm_state import AlgorithmState
from algorithm.iteration_result import IterationResult


def get_state_from(iterations_directory, new_root):
	if not os.path.exists(iterations_directory):
		return None, None
	iteration_pattern = re.compile('iteration_([0-9]*).json')
	for filename in sorted(
			os.listdir(iterations_directory),
			key=lambda filename: -int(iteration_pattern.match(filename).group(1))):
		try:
			with open(os.path.join(iterations_directory, filename), 'r') as input_file:
				state_json = json.load(input_file)
			state = AlgorithmState.parse_json(state_json['state'], new_root=new_root)
			result = IterationResult.parse_json(state_json['iteration-result'])
			print('found usable state at ' + filename)
			return state, result
		except:
			traceback.print_exc()
			continue

	print('Could not find a suitable iteration')
	return None, None


def continue_from(iterations_directory):
	EnvironmentSettings.set_all_output_prefix('rerun')
	new_root = EnvironmentSettings.get_current_prefix()
	EnvironmentSettings.remove_files(new_root)
	state, _ = get_state_from(iterations_directory, new_root)
	if state is None:
		return

	# create run result
	# set the output directory

	run_result = run_algorithm(state)
	state.history.create_plot(state.plotter, verbose=False)
	state.logger.info_json('history', state.history)
	state.logger.flush()
	print(run_result)


if __name__ == '__main__':
	continue_from(
		'/home/thallock/Pictures/ConvexConstraintsOutput/hott_schittowski_218.old/iteration_json/')
