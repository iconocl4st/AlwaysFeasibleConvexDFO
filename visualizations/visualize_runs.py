import os
import re
import json

from utils.evaluation import Evaluation
from hott_schittowski.problems import HottSchittowski
from utils.convexity_tester import ConvexityTester
from utils.formatting import Formatting
from utils.run_result import RunResult


def should_show_result(run_result):
	if run_result.algorithm_name == 'nomad':
		return run_result.run_params.map['strategy'] == 'fail-with-garbage'
	return run_result.ht_problem.n < 10


def generate_table(root_directory):
	for filename in os.listdir(root_directory):
		result_file = os.path.join(root_directory, filename, 'result.json')
		if not os.path.exists(result_file):
			continue

		with open(result_file, 'r') as input_file:
			run_result = RunResult.parse_json(json.load(input_file))

		if not run_result.has_evaluations():
			continue
		if not should_show_result(run_result):
			continue

		yield run_result


def print_table(root_directory):
	sorted_results = sorted(
		[e for e in generate_table(root_directory)],
		key=lambda x: (str(x.ht_problem.number), str(x.algorithm_name)))
	print(Formatting.format_strings(
		[RunResult.get_headers()] + [
			te.to_row()
			for te in sorted_results]))


if __name__ == '__main__':
	print_table(root_directory='/home/thallock/Pictures/ConvexConstraintsOutput/runs')
