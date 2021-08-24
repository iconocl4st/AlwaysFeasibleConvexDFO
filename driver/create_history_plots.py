
import matplotlib.pyplot as plt
import numpy as np

import os

from settings import EnvironmentSettings
from utils.plotting import Plotting
from visualizations.visualize_hock_schittkowski import add_problem_to_plot
from visualizations.visualize_runs import generate_table
import utils.performance as perf

# state.history.create_plot(state.plotter, verbose=False)


class MockPlotter:
	def __init__(self, path):
		self.path = path

	def create_plot(self, arg1, bounds, title, arg4):
		return Plotting.create_plot(title, self.path, bounds)


def create_performance_plot(folder, problem_no, run_results, perf_type):
	if len(run_results) < 2:
		return

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.legend(loc='lower left')

	plt.title('Performance plot for ' + str(problem_no) + ', type=' + perf_type)

	for run_result in run_results:
		xs, ys = perf.get_performance(run_result.history, perf_type)
		plt.plot(xs, ys, label=run_result.unique_identifier())

	plt.show()
	ax.legend()
	ax.grid(True)

	image_path = os.path.join(folder, str(problem_no) + '_' + perf_type + '.png')
	fig.savefig(image_path)
	plt.close()


def print_stats(by_problem):
	count = 0
	total = 0
	total_total = 0
	for problem_no, run_results in sorted(by_problem.items(), key=lambda x: x[0]):
		num_evaluations = sorted([
			(run_result, run_result.history.get_successful_num() + run_result.history.get_unsuccessful_num())
			for run_result in run_results
			if run_result.is_optimal()
		], key=lambda x: x[1])
		if len(num_evaluations) < 1:
			continue
		print(str(problem_no))
		for run_result, num_evals in num_evaluations:
			print('\t' + run_result.unique_identifier() + ': ' + str(num_evals))

		total_total += 1
		for idx, run_result in enumerate(num_evaluations):
			if run_result[0].algorithm_name == 'always_feasible':
				total += 1
				if idx == 0:
					count += 1
	print(count / total)
	print(count / total_total)


def create_plots(root_directory, plot_histories=False, plot_performance=False):
	history_folder = EnvironmentSettings.get_output_path(['results', 'histories'])
	os.makedirs(history_folder, exist_ok=True)

	by_problem = {}
	for run_result in generate_table(root_directory):
		problem_no = run_result.ht_problem.number
		if problem_no not in by_problem:
			by_problem[problem_no] = []
		by_problem[problem_no].append(run_result)

		dim = run_result.ht_problem.n
		if dim == 2 and plot_histories:
			history_path = os.path.join(history_folder, run_result.unique_identifier() + '.png')
			plt = run_result.history.create_plot_but_dont_save(MockPlotter(history_path))
			add_problem_to_plot(run_result.ht_problem, plt)
			plt.save()

	if plot_performance:
		performance_folder = EnvironmentSettings.get_output_path(['results', 'performance'])
		os.makedirs(performance_folder, exist_ok=True)
		for problem_no, run_results in by_problem.items():
			for perf_type in perf.PerformancePlotType.ALL_TYPES:
				create_performance_plot(performance_folder, problem_no, run_results, perf_type)

	print_stats(by_problem)


if __name__ == '__main__':
	create_plots(
		root_directory='/home/thallock/Pictures/ConvexConstraintsOutput/runs',
		plot_histories=True,
		plot_performance=True
	)
