import os
import json

from utils.formatting import Formatting
from utils.run_result import RunResult


def generate_table(root_directory):
	for filename in os.listdir(root_directory):
		result_file = os.path.join(root_directory, filename, 'result.json')
		if not os.path.exists(result_file):
			continue

		with open(result_file, 'r') as input_file:
			run_result = RunResult.parse_json(json.load(input_file))

		if not run_result.has_evaluations():
			continue

		yield run_result


def print_table(root_directory):
	sorted_results = sorted(
		[e for e in generate_table(root_directory)],
		key=lambda x: (str(x.ht_problem.number), str(x.algorithm_name)))
	sorted_results = filter(
		lambda x: x.algorithm_name == 'always_feasible',
		sorted_results
	)
	print(Formatting.format_strings(
		[RunResult.get_headers()] + [
			te.to_row()
			for te in sorted_results]))


if __name__ == '__main__':
	print_table(root_directory='/home/thallock/Pictures/ConvexConstraintsOutput/runs')
