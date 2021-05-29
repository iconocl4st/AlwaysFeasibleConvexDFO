import itertools
import numpy as np


class Polyhedron:
	def __init__(self, A, b):
		self.A = A.astype(np.float64)
		self.b = b.astype(np.float64)

	def evaluate(self, x):
		return self.A @ x - self.b

	def add_single_constraint(self, a, b):
		return Polyhedron(
			np.vstack([self.A, a]),
			np.concatenate([self.b, [b]])
		)

	def add_lb(self, idx, c):
		return self.add_single_constraint(
			np.pad([-1.0], pad_width=[[idx, self.A.shape[1] - idx - 1]], mode='constant'),
			-c
		)

	def add_ub(self, idx, c):
		return self.add_single_constraint(
			np.pad([1.0], pad_width=[[idx, self.A.shape[1] - idx - 1]], mode='constant'),
			c
		)

	def distance_to_closest_constraint(self, x):
		return min(
			np.divide(abs(numpy.dot(self.A, x) - self.b), np.linalg.norm(self.A, axis=1))
		)

	def clone(self):
		return Polyhedron(np.copy(self.A), np.copy(self.b))

	def add_to_pyomo(self, model):
		for r in range(self.A.shape[0]):
			model.constraints.add(
				sum(model.x[c] * self.A[r, c] for c in model.dimension) <= self.b[r]
			)

	def contains(self, point, tolerance=1e-10):
		return (np.dot(self.A, point) <= self.b + tolerance).all()

	def shrink(self, center, factor):
		A = np.copy(self.A)
		b = np.copy(self.b)
		for i in range(A.shape[0]):
			n = np.linalg.norm(A[i])
			A[i] /= n
			b[i] /= n
		return Polyhedron(A, b * factor + (1-factor) * np.dot(A, center))

	def translate(A, b, center):
		return Polyhedron(np.copy(A), b + np.dot(A, center))

	def rotate(self, theta):
		if self.A.shape[1] != 2:
			raise Exception("Not supported")
		rotation = np.array([
			[+np.cos(theta), -np.sin(theta)],
			[+np.sin(theta), +np.cos(theta)]
		])
		return Polyhedron(np.dot(self.A, rotation), np.copy(self.b))

	def intersect(self, other):
		return Polyhedron(
			np.append(self.A, other.A, axis=0),
			np.append(self.b, other.b)
		)

	def enumerate_vertices(self):
		dimension = self.A.shape[1]
		num_constraints = self.A.shape[0]

		for indices in itertools.combinations(range(num_constraints), dimension):
			sub_a = self.A[list(indices), :]
			sub_b = self.b[list(indices)]

			try:
				x = np.linalg.solve(sub_a, sub_b)
			except np.linalg.LinAlgError:
				continue

			if not self.contains(x):
				continue

			yield x, indices

	def get_feasible_point(self, tolerance=1e-4):
		vertices = np.array([v[0] for v in self.enumerate_vertices()])
		central_point = numpy.mean(vertices, axis=0)
		if vertices.shape[0] < vertices.shape[1] + 1:
			directions = self.A[self.A@central_point > self.b - tolerance, :]
			direction = np.mean(np.diag(-1/np.linalg.norm(directions, axis=1)) @ directions, axis=0)
			direction /= np.linalg.norm(direction)
			t = 1
			while not self.contains(central_point + t * direction, tolerance):
				t /= 2
			if not self.contains(central_point + t * direction, tolerance):
				raise Exception("this algorithm did not work")
			return central_point + t * direction
		return central_point

	def get_diameter(self):
		diam = -1
		vertices = np.array([v for v in self.enumerate_vertices()])
		for idx1, idx2 in itertools.combinations(range(len(vertices)), 2):
			d = np.linalg.norm(vertices[idx1] - vertices[idx2])
			if d > diam:
				diam = d
		return diam

	def shift(self, center, radius):
		return Polyhedron(
			self.A * radius,
			self.b - np.dot(self.A, center)
		)

	def normalize(self, use_a=False):
		''' TODO '''
		A = self.A.copy()
		b = self.b.copy()
		for i in range(A.shape[0]):
			row_scale = 1.0 / abs(b[i])
			if use_a or numpy.isinf(row_scale) or numpy.isnan(row_scale) or row_scale < 1e-12:
				row_scale = 1.0 / numpy.linalg.norm(A[i])
			A[i] *= row_scale
			b[i] *= row_scale
		return Polyhedron(A, b)

	def to_json(self):
		return {'A': self.A, 'b': self.b}

	@staticmethod
	def parse_json(json_object):
		return Polyhedron(
			np.array(json_object['A']),
			np.array(json_object['b']),
		)

	@staticmethod
	def create_from_tr(center, radius):
		n = len(center)
		A = np.zeros([2 * n, n], dtype=np.float)

		for i in range(n):
			A[2 * i + 0, i] = +1
			A[2 * i + 1, i] = -1

		return Polyhedron(A, A @ center + radius)

	@staticmethod
	def create(A, b):
		return Polyhedron(A, b)
