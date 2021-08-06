import json
import os
import numpy as np

from settings import EnvironmentSettings
from visualizations.visualize_runs import generate_table
import utils.performance as perf

import matplotlib.pyplot as plt


def remove_duplicates(x_and_ys):
	ret = []
	for i in range(len(x_and_ys) - 1):
		if x_and_ys[i][1] < x_and_ys[i + 1][1]:
			ret.append(x_and_ys[i])
	return ret


class PerformanceProfile:
	def __init__(self, optimality_ratio):
		self.problem_2_alg_2_iters = {}
		# self.problem_2_min_iters = {}

		self.problem_minimums = {}
		self.problems = set()
		self.algorithms = set()
		self.optimality_ratio = optimality_ratio

	def add(self, algorithm, problem, values):
		self.algorithms.add(algorithm)
		self.problems.add(problem.number)
		self.problem_minimums[problem.number] = problem.solution.fstar

		if problem.number not in self.problem_2_alg_2_iters:
			self.problem_2_alg_2_iters[problem.number] = {}

		for idx, value in zip(values[0], values[1]):
			true_minimum = self.problem_minimums[problem.number]
			relative_error = abs(value - true_minimum) / max(1, abs(true_minimum))
			if idx <= 0:
				# TODO: look into this...
				continue
			if relative_error < self.optimality_ratio:
				self.problem_2_alg_2_iters[problem.number][algorithm] = idx
				return

	def get_prob_2_alg_2_iters(self):
		return {
			problem: {
				algorithm: (
					self.problem_2_alg_2_iters[problem][algorithm]
					if problem in self.problem_2_alg_2_iters and
						algorithm in self.problem_2_alg_2_iters[problem]
					else np.inf
				)
				for algorithm in self.algorithms
			}
			for problem in self.problems
		}

	def plot_performance(self, path):
		prob_2_alg_2_iters = self.get_prob_2_alg_2_iters()
		print(json.dumps(prob_2_alg_2_iters, indent=2))
		minimum_num_iterations = {
			problem: min(
				prob_2_alg_2_iters[problem][algorithm]
				for algorithm in self.algorithms)
			for problem in self.problems
		}
		print(json.dumps(minimum_num_iterations, indent=2))
		ratio_of_iterations = {
			problem: {
				algorithm: (
					prob_2_alg_2_iters[problem][algorithm] /
					minimum_num_iterations[problem]
					if not np.isinf(minimum_num_iterations[problem])
					else np.inf)
				for algorithm in self.algorithms
			}
			for problem in self.problems
		}
		print(json.dumps(ratio_of_iterations, indent=2))
		problem_to_ratio = {
			algorithm: sorted([
				algs[algorithm]
				for problem, algs in ratio_of_iterations.items()])
			for algorithm in self.algorithms
		}
		print(json.dumps(problem_to_ratio, indent=2))
		profile = {
			algorithm: remove_duplicates([
				(idx / len(values), x_value)
				for idx, x_value in enumerate(values)])
			for algorithm, values in problem_to_ratio.items()
		}
		print(json.dumps(profile, indent=2))

		for algorithm, values in profile.items():
			plt.step(
				x=[v[1] for v in values],
				y=[v[0] for v in values],
				where='post',
				label=algorithm)
		plt.grid(axis='x')
		plt.legend(title='Performance Profile')
		plt.xlim([0, 20])
		plt.ylim([0, 1.0])
		plt.savefig(path)
		plt.show()


def create_performance_profile(root_directory):
	performance_profile = PerformanceProfile(optimality_ratio=0.01)
	for run_result in generate_table(root_directory):
		performance_profile.add(
			run_result.algorithm_name,
			run_result.ht_problem,
			perf.get_performance(
				run_result.history, perf.PerformancePlotType.ALL))

	performance_path = EnvironmentSettings.get_output_path(['results', 'performance_profile.png'])
	os.makedirs(os.path.dirname(performance_path), exist_ok=True)
	performance_profile.plot_performance(performance_path)


if __name__ == '__main__':
	create_performance_profile(root_directory='/home/thallock/Pictures/ConvexConstraintsOutput/runs')

