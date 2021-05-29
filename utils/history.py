
import numpy as np

from trial_problems.simple_problems import Evaluation
from utils.bounds import Bounds
from utils.ellipsoid import Ellipsoid


class History:
	def __init__(self):
		self.bounds = Bounds()
		self.evaluations = []
		self.sample_regions = []
		self.outer_trust_regions = []

	def add_outer_tr(self, iteration, iterate, radius):
		self.outer_trust_regions.append([iteration, iterate, radius])

	def add_sample_region(self, iteration, sample_region):
		self.sample_regions.append([iteration, sample_region])

	def get_minimum_evaluation(self):
		return min((e for _, e in self.evaluations if e.success), key=lambda x: x.objective, default=None)

	def get_successful_num(self):
		return sum(1 for _, e in self.evaluations if e.success)

	def get_unsuccessful_num(self):
		return sum(1 for _, e in self.evaluations if not e.success)

	def get_evaluations(self, filter_method):
		return [
			(idx, evaluation.copy())
			for idx, (_, evaluation) in enumerate(self.evaluations)
			if evaluation.success and filter_method(evaluation.x)
		]

	def find_evaluation(self, x):
		evaluation = None
		dist = None
		idx = None
		for i, (_, e) in enumerate(self.evaluations):
			edist = np.linalg.norm(x - e.x)
			if evaluation is None or edist < dist:
				evaluation = e
				dist = edist
				idx = i
		assert dist < 1e-12, 'could not find evaluation'
		return idx, evaluation.copy()

	def get_evaluation(self, idx):
		return self.evaluations[idx][1].copy()

	def add_evaluation(self, iteration, evaluation):
		self.evaluations.append([iteration, evaluation.copy()])
		self.bounds.extend(evaluation.x)

	def get_plot_bounds(self):
		return self.bounds.expand()

	def create_plot_but_dont_save(self, plotter, subfolder=None, iterations=None, verbose=True):
		if iterations is None:
			min_iter, max_iter = -1, np.inf
		else:
			min_iter, max_iter = iterations
		evaluations_to_plot = [
			x
			for iteration, x in self.evaluations
			if min_iter <= iteration <= max_iter
		]
		outer_trs_to_plot = [
			[center, radius]
			for iteration, center, radius in self.outer_trust_regions
			if min_iter <= iteration <= max_iter
		]
		sr_to_plot = [
			x
			for iteration, x in self.sample_regions
			if min_iter <= iteration <= max_iter
		]

		b = Bounds()
		for evaluation in evaluations_to_plot:
			b = b.extend(evaluation.x)
		for center, radius in outer_trs_to_plot:
			b = b.extend(center + radius).extend(center - radius)

		successful = np.array([
			evaluation.x
			for evaluation in evaluations_to_plot
			if evaluation.success
		])
		unsuccessful = np.array([
			evaluation.x
			for evaluation in evaluations_to_plot
			if not evaluation.success
		])
		title = 'successful=' + str(successful.shape[0]) + ', unsuccessful=' + str(unsuccessful.shape[0])
		plt = plotter.create_plot('history', b.expand(), title, subfolder)

		for center, radius in outer_trs_to_plot:
			plt.add_linf_tr(center, radius, label=None, color='b')

		for sample_region in sr_to_plot:
			plt.add_contour(sample_region.evaluate, label=None, color='m', lvls=0)

		plt.add_points(successful, label='successful evaluations', color='g', s=50, marker="+")
		if unsuccessful.shape[0] > 0:
			plt.add_points(unsuccessful, label='unsuccessful evaluations', color='r', s=20, marker="x")
		return plt

	def create_plot(self, plotter, subfolder=None, iterations=None, verbose=True):
		plt = self.create_plot_but_dont_save(plotter, subfolder, iterations, verbose)
		plt.save()

	def to_json(self):
		return {
			'bounds': self.bounds,
			'evaluations': self.evaluations,
			'sample-regions': self.sample_regions,
			'outer-trust-regions': self.outer_trust_regions,
		}

	@staticmethod
	def parse_json(json):
		history = History()
		history.bounds = Bounds.parse_json(json['bounds'])
		history.evaluations = [
			[iteration, Evaluation.parse_json(evalJson)]
			for iteration, evalJson in json['evaluations']
		]
		history.outer_trust_regions = [
			[iteration, np.array(iterate), radius]
			for iteration, iterate, radius in json['outer-trust-regions']
		]
		history.sample_regions = [
			[iteration, Ellipsoid.parse_json(sr_json)]
			for iteration, sr_json in json['sample-regions']
		]
		return history
