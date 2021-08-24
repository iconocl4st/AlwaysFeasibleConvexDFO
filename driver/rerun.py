import os
import re
import json
import traceback

from algorithm.algorithm import run_algorithm
from algorithm.algorithm_state import AlgorithmState
from algorithm.iteration_result import IterationResult
from settings import EnvironmentSettings


def get_state_from(iterations_directory, new_root):
	if not os.path.exists(iterations_directory):
		return None, None

	history_path = os.path.join(iterations_directory, 'history.json')
	if not os.path.exists(history_path):
		return None, None
	with open(history_path, 'r') as input_file:
		history_json = json.load(input_file)['history']

	iteration_pattern = re.compile('iteration_([0-9]*).json')
	iteration_jsons = os.listdir(iterations_directory)
	iteration_jsons = filter(lambda filename: iteration_pattern.match(filename), iteration_jsons)
	iteration_jsons = sorted(
		iteration_jsons,
		key=lambda filename: -int(iteration_pattern.match(filename).group(1)))

	for filename in iteration_jsons:
		try:
			with open(os.path.join(iterations_directory, filename), 'r') as input_file:
				state_json = json.load(input_file)
			state = AlgorithmState.parse_json(
				state_json['state'],
				history_json=history_json,
				new_root=new_root)
			state.history.remove_future_evaluations(state.iteration)
			result = IterationResult.parse_json(state_json['iteration-result'])
			print('found usable state at ' + filename)
			return state, result
		except:
			traceback.print_exc()
			continue

	print('Could not find a suitable iteration')
	return None, None


def continue_from(iterations_directory):
	new_root = EnvironmentSettings.get_output_path(['rerun'])
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
		'/home/thallock/Pictures/ConvexConstraintsOutput/runs/' +
		'215_always_feasible_linear' +
		'/iteration_json')
