import os
import numpy as np
import scipy.optimize

from algorithm.tr_subproblem.quadratic_buffered import QuadraticBufferedTr
from settings import EnvironmentSettings
from utils.plotting import Plotting
from utils.quadratic import Quadratic
from utils.stochastic_search import multi_eval_stochastic_search, StochasticSearchParams


def plot_quadratic_tr():
	qbtr = QuadraticBufferedTr()

	qbtr.A = np.array([
			[1, 1],
			[1, -0.5],
			[-0.25, 1],
		])
	qbtr.b = np.array([6, 5, 5])
	qbtr.xk = np.array([5, 5])
	qbtr.r = 4
	qbtr.buffer_dist = 0.1
	qbtr.buffer_rate = 0.1

	qbtr.objective = Quadratic.create(
		c=0,
		g=np.array([0, 0], dtype=np.float64),
		Q=np.array([[1, 0], [0, 1]], dtype=np.float64)
	)

	filename = EnvironmentSettings.get_output_path(
		['visualizations', 'quadratic_tr', 'attempt_1.png'])
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	plot = Plotting.create_plot_on(
		filename, [0, 0], [10, 10], name='Unshifted cone')

	qbtr.create_constraints()
	qbtr.add_to_plt(plot)

	ssp = StochasticSearchParams()
	ssp.x0 = qbtr.xk
	ssp.objective = qbtr.stochastic_obj
	ssp.multi_eval = qbtr.stochastic_multi_eval
	ssp.initial_radius = qbtr.r

	trial_value = multi_eval_stochastic_search(ssp)
	plot.add_point(trial_value.trial, label='minimizer', marker='o', color='c')

	plot.save()


if __name__ == '__main__':
	plot_quadratic_tr()
