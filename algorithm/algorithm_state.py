import os

import numpy as np

from utils.assertions import make_assertion
from utils.convexity_tester import ConvexityTester
from utils.default_stringable import DefaultStringable
from utils.plotting import Plotter
from utils.history import History
from utils.logger import Logger

from utils.ellipsoid import Ellipsoid
from utils.polynomial import MultiIndex

from trial_problems.ht_problem import HockSchittkowskiProblem


class UserParams(DefaultStringable):
	def __init__(self):
		self.basis_type = 'quadratic'
		self.tr_heuristics = []
		self.sample_region_strategy = 'conservative-ellipsoid'
		self.empty_sr_method = 'scale'
		self.overrides = {}

		self.threshold_criticality = 1e-3
		self.threshold_tr_radius = 1e-4
		self.threshold_regularity = 1e-3
		self.threshold_reduction_sufficient = 0.9
		self.threshold_reduction_minimum = 0.1
		self.tr_update_dec = 0.5
		self.tr_update_inc = 1.5
		self.alpha = 0.99
		self.beta = 0.99
		self.kappa_crit = 1e-2
		self.p_beta = 0.99
		self.p_alpha = 0.99
		self.p_chi = None
		self.p_delta = 0.99 * min(self.p_alpha, self.p_beta)
		self.maximum_iterations = 500
		self.sample_point_filter = 'trust-region-radius'
		self.xsi_replace = 1e-2
		self.ka = np.sqrt(2)
		self.plot_lagrange_maximizations = False

		self.plot_options = {
			'current-iterate': {
				'color': 'green',
				'marker': '*',
				's': 100,
				'label': 'current iterate',
			},
			'sample-region': {
				'color': 'red',
				'label': 'sample region',
				'lvls': [0],
			},
			'trust-region': {
				'color': 'blue',
				'label': 'outer trust region',
			},
			'active-region': {
				'color': 'yellow',
				'label': 'active region',
			},
			'sample-points': {
				'color': 'red',
				'label': 'sample points',
				's': 30,
				'marker': 'D',
			},
			'objective-contour': {
				'color': 'k',
				'label': 'objective',
			},
			'constraint-contour': {
				'color': 'c',
				'label': 'constraint'
			},
			'zik': {
				'color': 'k',
				'label': 'constraint zeros',
				's': 20,
				'marker': 'x'
			},
			'wik': {
				'color': 'm',
				'label': 'buffering cone vertices',
				's': 20,
				'marker': '+'
			},
			'u': {
				'label': 'feasible direction',
			},
			'trial-point': {
				'color': 'm',
				'label': 'trial point',
				's': 50,
				'marker': 'o'
			},
			'projected-gradient-descent': {
				'color': 'tab:orange',
				'label': 'projected gradient descent',
				's': 50,
				'marker': 'x'
			},
			'negative-gradient': {
				'color': 'm',
				'label': 'steepest descent direction',
			}
		}

	def create_basis(self, n):
		if self.basis_type == 'linear':
			return MultiIndex.get_linear_basis(n)
		else:
			return MultiIndex.get_quadratic_basis(n)

	def to_json(self):
		return {'overrides': self.overrides}

	@staticmethod
	def parse_json(json_object):
		return UserParams.create(json_object['overrides'])

	@staticmethod
	def create(overrides):
		params = UserParams()
		params.overrides = overrides
		if overrides is None:
			return params

		if 'basis' in overrides and overrides['basis'] == 'linear':
			params.basis_type = 'linear'

		if 'tr-heuristics' in overrides:
			params.tr_heuristics = overrides['tr-heuristics']

		if 'sr-strategy' in overrides:
			params.sample_region_strategy = overrides['sr-strategy']

		if 'on-empty-sample' in overrides:
			params.empty_sr_method = overrides['on-empty-sample']

		return params


class AlgorithmState(DefaultStringable):
	def __init__(self):
		self.root_directory = None

		self.problem = None
		self.params = None

		self.basis = None

		self.iteration = 0
		self.current_iterate = None
		self.outer_tr_radius = None
		self.sample_region = None

		self.history = None
		self.logger = None
		self.plotter = None

		self.convexity_tester = None

		self.current_plot = None
		self.buffering_plot = None
		self.evaluation_count = None

		self.unbound_radius = None

	@property
	def dim(self):
		return len(self.current_iterate)

	@property
	def num_constraints(self):
		return self.problem.num_constraints

	def evaluate(self, x):
		self.evaluation_count += 1
		evaluation = self.problem.evaluate(x)
		self.history.add_evaluation(self.iteration, evaluation)
		self.logger.info('Evaluated at ' + str(x) + ', feasible = ' + str(evaluation.success))
		return evaluation

	def to_json(self, show_history=True):
		return {
			'problem': self.problem,
			'root-directory': self.root_directory,
			'params': self.params,
			'basis': self.basis,
			'current-iterate': self.current_iterate,
			'outer-tr-radius': self.outer_tr_radius,
			'sample-region': self.sample_region,
			'plotter': self.plotter,
			'evaluation-count': self.evaluation_count,
			'iteration': self.iteration,
			'convexity': self.convexity_tester,
			'unbound-radius': self.unbound_radius,
			**({'history': self.history} if show_history else {}),
		}

	@staticmethod
	def parse_json(json, history_json, new_root=None):
		if new_root is None:
			new_root = json['root-directory']

		algo_state = AlgorithmState()
		algo_state.root_directory = new_root

		algo_state.problem = HockSchittkowskiProblem.parse_json(json['problem'])
		algo_state.params = UserParams.parse_json(json['params'])

		algo_state.iteration = json['iteration']
		algo_state.current_iterate = np.array(json['current-iterate'], dtype=np.float64)
		algo_state.outer_tr_radius = json['outer-tr-radius']
		algo_state.sample_region = Ellipsoid.parse_json(json['sample-region'])
		algo_state.evaluation_count = json['evaluation-count']
		algo_state.convexity_tester = ConvexityTester.parse_json(json['convexity'])

		if history_json is not None:
			algo_state.history = History.parse_json(history_json)
		elif 'history' in json:
			algo_state.history = History.parse_json(json['history'])
		else:
			raise Exception('Unable to find history')
		algo_state.logger = Logger.create(os.path.join(new_root, 'log_file.txt'))
		algo_state.plotter = Plotter.parse_json(json['plotter'], new_image_path=os.path.join(new_root, 'images'))
		algo_state.unbound_radius = json['unbound-radius']

		algo_state.basis = algo_state.params.create_basis(len(algo_state.current_iterate))

		return algo_state

	@staticmethod
	def create(problem_spec, root_directory, user_params=None):
		algo_state = AlgorithmState()
		algo_state.root_directory = root_directory
		algo_state.evaluation_count = 0
		algo_state.problem = problem_spec
		algo_state.params = UserParams.create(user_params)
		algo_state.history = History()
		algo_state.convexity_tester = ConvexityTester(problem_spec.num_constraints)

		algo_state.logger = Logger.create(os.path.join(root_directory, 'log_file.txt'))
		algo_state.plotter = Plotter.create(os.path.join(root_directory, 'images'))
		algo_state.unbound_radius = False

		algo_state.iteration = 0
		algo_state.current_iterate = problem_spec.get_initial_x()
		algo_state.outer_tr_radius = problem_spec.get_initial_r()
		algo_state.sample_region = Ellipsoid.create(
			problem_spec.get_initial_q(),
			problem_spec.get_initial_center(),
			problem_spec.get_initial_delta()
		)
		algo_state.basis = algo_state.params.create_basis(len(algo_state.current_iterate))
		initial_evaluation = algo_state.evaluate(algo_state.current_iterate)
		make_assertion(initial_evaluation.success, 'Initial point not feasible')
		return algo_state
