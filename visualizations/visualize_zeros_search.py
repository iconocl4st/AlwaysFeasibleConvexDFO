import os
import numpy as np

import matplotlib.pyplot as plt

from settings import EnvironmentSettings
from utils.zeros_search import find_zeros


def objective(asymptotes, lows):
	def obj(x):
		if x <= asymptotes[0]:
			return 1 / (asymptotes[0] - x) - 2
		if x >= asymptotes[-1]:
			return 1 / (x - asymptotes[-1]) - 2
		for i in range(len(asymptotes) - 1):
			al = asymptotes[i]
			ah = asymptotes[i+1]
			if x < al:
				continue
			if x > ah:
				continue
			return (
				(x - (0.3 * al + 0.7 * ah)) ** 2 / ((x - al) * (ah - x)) - lows[i]
			)
		raise Exception('this should not be able to happen')
	return obj


def visualize_find_lagrange(num):
	ymin = -3
	ymax = 5
	asymptotes = [-1, 1, 1, 3]
	lows = [-2, 3, 3]
	obj = objective(asymptotes, lows)

	zeros = sorted([z for z, _ in find_zeros(obj, asymptotes, tol=1e-12)])
	print(zeros)

	t = np.linspace(min(asymptotes) - 1, max(asymptotes) + 1, 1000)
	y = np.array([obj(ti) for ti in t])

	plt.plot(t, y, t, np.zeros_like(t))
	plt.ylim([ymin, ymax])
	plt.vlines(asymptotes, ymin, ymax, colors='red')
	plt.vlines(zeros, ymin, ymax, colors='green')
	plt.ylabel('function values')

	filename = EnvironmentSettings.get_output_path(
		['visualizations', 'find_zeros', 'zero_' + str(num) + '.png'])
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	plt.savefig(filename)
	plt.close()


if __name__ == '__main__':
	visualize_find_lagrange(0)
