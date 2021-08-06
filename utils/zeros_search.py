import numpy as np
import json

from utils.assertions import make_assertion


class Pattern:
	def __init__(self, obj, x_low, x_high):
		self.obj = obj
		self.outer_diameter = x_high - x_low
		self.center_x = (x_high + x_low) / 2.0
		self.t_radius = 1.0
		self.history = []
		self.t_low, self.t_med, self.t_high = -self.t_radius, 0, self.t_radius
		self.o_low, self.o_med, self.o_high = None, None, None
		self.ensure_evaluations()

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		return json.dumps({
			'low': {'t': round(self.t_low, 3), 'x': round(self.map_to_x(self.t_low), 3), 'y': round(self.o_low, 3)},
			'mid': {'t': round(self.t_med, 3), 'x': round(self.map_to_x(self.t_med), 3), 'y': round(self.o_med, 3)},
			'high': {'t': round(self.t_high, 3), 'x': round(self.map_to_x(self.t_high), 3), 'y': round(self.o_high, 3)},
		})

	def map_to_x(self, t):
		return self.center_x + self.outer_diameter * np.arctan(t) / np.pi

	def evaluate(self, t):
		x = self.map_to_x(t)
		self.history.append(x)
		return t, self.obj(x)

	def ensure_evaluations(self):
		if self.o_low is None:
			self.t_low, self.o_low = self.evaluate(self.t_low)
		if self.o_med is None:
			self.t_med, self.o_med = self.evaluate(self.t_med)
		if self.o_high is None:
			self.t_high, self.o_high = self.evaluate(self.t_high)

	def left(self):
		make_assertion(self.t_low > -1e300, 'Not bounded to the left')
		self.t_high, self.o_high = self.t_med, self.o_med
		self.t_med, self.o_med = self.t_low, self.o_low
		self.t_low, self.o_low = self.evaluate(self.t_med - self.t_radius)
		self.ensure_evaluations()

	def right(self):
		make_assertion(self.t_high < 1e300, 'Not bounded to the right')
		self.t_low, self.o_low = self.t_med, self.o_med
		self.t_med, self.o_med = self.t_high, self.o_high
		self.t_high, self.o_high = self.evaluate(self.t_med + self.t_radius)
		self.ensure_evaluations()

	def center(self):
		self.t_radius /= 2.0
		self.t_low, self.o_low = self.evaluate(self.t_med - self.t_radius)
		self.t_high, self.o_high = self.evaluate(self.t_med + self.t_radius)

	def expand(self):
		self.t_radius *= 2.0
		self.o_low = None
		self.o_high = None

	def find_lowest_evaluation(self):
		return np.argmin([self.o_low, self.o_med, self.o_high])


def pattern_search(obj, x_low, x_high, tol=1e-12):
	pattern = Pattern(obj, x_low, x_high)
	count_left = 0
	count_right = 0
	count_iters = 0
	while pattern.t_radius > tol and pattern.o_med > -tol:
		count_iters += 1
		make_assertion(count_iters < 10000, 'This is taking too long')
		arg_min = pattern.find_lowest_evaluation()
		make_assertion(arg_min in [0, 1, 2], 'a 3 element array can only contain 3 elements')
		if arg_min == 0:
			count_left += 1
			count_right = 0
			if count_left > 10:
				pattern.expand()
			pattern.left()
		elif arg_min == 1:
			count_left = 0
			count_right = 0
			pattern.center()
		elif arg_min == 2:
			count_left = 0
			count_right += 1
			if count_right > 10:
				pattern.expand()
			pattern.right()
	return pattern.map_to_x(pattern.t_med), pattern.o_med, pattern.history


class BinarySearchBounds:
	def __init__(self, obj, singularity, direction):
		self.obj = obj
		self.singularity = singularity
		self.direction = direction
		self.inner_magnitude = None
		self.inner_value = None
		self.outer_magnitude = None
		self.outer_value = None

	def map_to_x(self, magnitude):
		return self.singularity + magnitude * self.direction

	def map_to_magnitude(self, x):
		return (x - self.singularity) / self.direction

	def set_outer_bound(self, maximum_bound):
		self.outer_value = self.obj(maximum_bound)
		make_assertion(self.outer_value < 0, 'outer bound not found')
		self.outer_magnitude = self.map_to_magnitude(maximum_bound)
		make_assertion(
			abs(maximum_bound - self.map_to_x(self.outer_magnitude)) < 1e-4,
			'unmapping of bound did not work')

	def find_outer_bound(self):
		self.outer_magnitude = 1
		while True:
			x = self.map_to_x(self.outer_magnitude)
			make_assertion(abs(x) < 1e300, 'outer bound too far out')
			self.outer_value = self.obj(x)
			if self.outer_value < 0:
				return

			self.inner_magnitude = self.outer_magnitude
			self.inner_value = self.outer_value
			self.outer_magnitude *= 2

	def find_inner_bound(self):
		if self.inner_magnitude is not None:
			return

		self.inner_magnitude = min(1, self.outer_magnitude / 2)
		while True:
			x = self.map_to_x(self.inner_magnitude)
			self.inner_value = self.obj(x)
			if self.inner_value > 0:
				return

			self.inner_magnitude /= 2.0
			make_assertion(self.inner_magnitude > 1e-16, 'inner bound too close')


def binary_search(obj, singularity, direction, tol=1e-12, maximum_bound=None):
	bounds = BinarySearchBounds(obj, singularity, direction)
	if maximum_bound is not None:
		bounds.set_outer_bound(maximum_bound)
	else:
		bounds.find_outer_bound()
	bounds.find_inner_bound()
	while True:
		rad = (bounds.outer_magnitude - bounds.inner_magnitude) / 2.0
		med = (bounds.outer_magnitude + bounds.inner_magnitude) / 2.0
		x = bounds.map_to_x(med)
		val = bounds.obj(x)
		if rad < tol:
			return x, val

		if val < 0:
			bounds.outer_magnitude = med
			bounds.outer_value = val
		else:
			bounds.inner_magnitude = med
			bounds.inner_value = val


def remove_duplicates(asymptotes, tol):
	asymptotes = sorted(asymptotes)
	r = [asymptotes[0]]
	for i in range(1, len(asymptotes)):
		a = asymptotes[i]
		if abs(r[-1] - a) > tol:
			r.append(a)
	return r


def find_zeros(obj, vertical_asymptotes, tol=1e-12):
	vertical_asymptotes = remove_duplicates(vertical_asymptotes, tol)
	yield binary_search(obj, vertical_asymptotes[+0], -1, tol)
	yield binary_search(obj, vertical_asymptotes[-1], +1, tol)
	for i in range(len(vertical_asymptotes) - 1):
		left = vertical_asymptotes[i]
		rght = vertical_asymptotes[i+1]
		x_middle, f, history = pattern_search(obj, left, rght, tol)
		if f > 0:
			continue
		yield binary_search(obj, left, +1.0, tol, x_middle)
		yield binary_search(obj, rght, -1.0, tol, x_middle)

