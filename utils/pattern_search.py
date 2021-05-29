
import numpy as np


class PatternSearchParams:
	def __init__(self):
		self.objective_func = None
		self.constraint_func = None
		self.initial = None


def _increment_index(idx, shape):
	index = 0
	while True:
		if index >= len(idx):
			return False
		if idx[index] < shape[index] - 1:
			break
		index += 1

	idx[index] += 1
	index -= 1

	while index >= 0:
		idx[index] = 0
		index -= 1

	return True


def _generate_pattern(arr):
	shape = arr.shape
	num_dims = len(shape)
	idx = [0] * num_dims
	while True:
		z = np.zeros_like(arr)
		z[tuple(idx)] = 1.0
		yield z
		z = np.zeros_like(arr)
		z[tuple(idx)] = -1.0
		yield z
		if not _increment_index(idx, shape):
			break



def pattern_search(search_params):
	pass


if __name__ == '__main__':
	# shape = [2, 4, 6, 2, 1]
	shape = [2, 4]
	arr = np.random.random(shape)
	for direction in _generate_pattern(arr):
		print(direction)
