import numpy as np


class Bounds:
	def __init__(self):
		self.ub = None
		self.lb = None

	def radius(self):
		return np.max(self.ub - self.lb)

	def extend_tr(self, center, radius):
		self.extend(center + radius)
		self.extend(center - radius)
		return self

	def sample(self):
		if self.ub is None or self.lb is None:
			return None
		return self.lb + np.multiply(
			np.random.random(len(self.ub)), self.ub - self.lb
		)

	def extend(self, x):
		if self.ub is None:
			self.ub = np.copy(x)
		if self.lb is None:
			self.lb = np.copy(x)

		for i in range(len(x)):
			if x[i] > self.ub[i]:
				self.ub[i] = x[i]
			if x[i] < self.lb[i]:
				self.lb[i] = x[i]
		return self

	def buffer(self, amount):
		self.ub += amount
		self.lb -= amount
		return self

	def expand(self, factor=1.2):
		b = Bounds()
		b.ub = np.copy(self.ub)
		b.lb = np.copy(self.lb)
		for i in range(len(self.ub)):
			expansion = (factor - 1.0) * (b.ub[i] - b.lb[i])
			b.ub[i] = b.ub[i] + expansion
			b.lb[i] = b.lb[i] - expansion
		return b

	def to_json(self):
		return {'lower-bound': self.lb, 'upper-bound': self.ub}

	@staticmethod
	def parse_json(json):
		bounds = Bounds()
		bounds.lb = np.array(json['lower-bound'], dtype=np.float64)
		bounds.ub = np.array(json['upper-bound'], dtype=np.float64)
		return bounds

	def __str__(self):
		return '[' + str(self.lb) + '-' + str(self.ub) + ']'

	@staticmethod
	def create(lb, ub):
		bounds = Bounds()
		bounds.extend(np.array([xi for xi in lb]))
		bounds.extend(np.array([xi for xi in ub]))
		return bounds
