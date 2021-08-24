import numpy as np
import math
import itertools

from utils.quadratic import Quadratic


class MultiIndex:
	def __init__(self, powers):
		self.powers = powers
		for a in powers:
			if type(a).__name__ != 'int':
				raise Exception('cannot use non-integer powers')

	def __eq__(self, other):
		return self.compare_to(other) == 0

	def __gt__(self, other):
		return self.compare_to(other) > 0

	def __str__(self):
		return "<" + ",".join([str(a) for a in self.powers]) + ">"

	def __repr__(self):
		return self.__str__()

	def to_json(self):
		return {'powers': self.powers}

	@property
	def degree(self):
		return np.sum(self.powers)

	@property
	def dimension(self):
		return len(self.powers)

	def add(self, other):
		return MultiIndex([a + other.powers[i] for i, a in enumerate(self.powers)])

	def subtract(self, other):
		ret = MultiIndex([a - other.powers[i] for i, a in enumerate(self.powers)])
		if not ret.non_negative():
			raise Exception("This shouldn't happen, should it?")
		return ret

	def compare_to(self, other):
		if len(self.powers) != len(other.powers):
			return len(self.powers) - len(other.powers)
		if self.degree != other.degree:
			return self.degree - other.degree
		for i, a in enumerate(self.powers):
			if other.powers[i] != a:
				return a - other.powers[i]
		return 0

	def non_negative(self):
		return min(self.powers) >= 0

	def copy(self):
		return MultiIndex([a for a in self.powers])

	def factorial(self):
		# return math.factorial(self.degree)
		ret = 1.0
		for a in self.powers:
			ret *= math.factorial(a)
		return ret

	def safe_factorial(self):
		ret = 1.0
		for a in self.powers:
			if a < 0:
				return False, None
			ret *= math.factorial(a)
		return True, ret

	def as_exponent(self, x):
		product = 1.0
		for i, xi in enumerate(x):
			for _ in range(self.powers[i]):
				product = product * x[i]
		return product

	def to_pyomo(self, model):
		ret = 1.0
		for i, power in enumerate(self.powers):
			for _ in range(power):
				ret = ret * model.x[i]
		return ret

	def translate(self, center):
		ret = Polynomial([Monomial(MultiIndex(tuple([0 for _ in range(self.dimension)])))])
		for i, p in enumerate(self.powers):
			if p == 0:
				continue
			ret = ret.multiply(Polynomial([
				Monomial(
					powers=MultiIndex.single_power(i, self.dimension, p - j),
					coefficient=c * (-center[i]) ** j)
				for j, c in enumerate(_construct_pascals_triangle(p))
			]))
		return ret

	@staticmethod
	def single_power(i, dimension, power):
		return MultiIndex(tuple([
			power if j == i else 0
			for j in range(dimension)
		]))

	@staticmethod
	def get_indices(degree, dimension):
		return sorted([
			MultiIndex([len([x for x in product if x == i]) for i in range(dimension)])
			for product in itertools.combinations_with_replacement([i for i in range(dimension)], degree)
		])

	@staticmethod
	def get_first_n_indices(n, dimension):
		ret = []
		degree = 0
		while True:
			for index in MultiIndex.get_indices(degree, dimension):
				if len(ret) >= n:
					return ret
				ret.append(index)
			degree += 1

	@staticmethod
	def get_preceding_indices(alpha, dimension):
		ret = []
		degree = 0
		while True:
			for index in MultiIndex.get_indices(degree, dimension):
				if index.compare_to(alpha) >= 0:
					return ret
				ret.append(index)
			degree += 1

	@staticmethod
	def get_quadratic_basis(dimension):
		return (
			MultiIndex.get_indices(0, dimension) +
			MultiIndex.get_indices(1, dimension) +
			MultiIndex.get_indices(2, dimension)
		)

	@staticmethod
	def get_linear_basis(dimension):
		return (
			MultiIndex.get_indices(0, dimension) +
			MultiIndex.get_indices(1, dimension)
		)

	@staticmethod
	def construct_vandermonde(basis, sample):
		return np.array([
			[index.as_exponent(point) for index in basis]
			for point in sample
		])


class Monomial:
	def __init__(self, powers, coefficient=1.0):
		if powers.__class__.__name__ in ['tuple', 'list']:
			self.powers = MultiIndex(powers)
		else:
			self.powers = powers
		self.coefficient = coefficient

	@property
	def dimension(self):
		return self.powers.dimension

	def evaluate(self, x):
		return self.coefficient * np.product(np.power(x, self.powers.powers))

	def to_pyomo(self, model):
		return self.coefficient * self.powers.to_pyomo(model)

	@property
	def degree(self):
		return self.powers.degree

	def differentiate_single_component(self, idx):
		if self.powers.powers[idx] == 0:
			return Monomial(
				MultiIndex([0 for _ in self.powers.powers]),
				0.0
			)
		if self.powers.powers[idx] == 1:
			return Monomial(
				MultiIndex([a if i != idx else 0 for i, a in enumerate(self.powers.powers)]),
				self.coefficient
			)
		return Monomial(
			MultiIndex([a if i != idx else a-1 for i, a in enumerate(self.powers.powers)]),
			self.coefficient * self.powers.powers[idx]
		)

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		if self.degree == 0:
			return str(self.coefficient)
		return " * ".join(
			([str(self.coefficient)] if self.coefficient != 1 else []) +
			[
				"x[" + str(i) + "]" + ("**" + str(a) if a != 1 else "")
				for i, a in enumerate(self.powers.powers)
				if a != 0
			]
		) if self.coefficient != 0.0 else "0"

	def copy(self):
		return Monomial(self.powers.copy(), self.coefficient)

	def multiply(self, other):
		return Monomial(
			self.powers.add(other.powers),
			self.coefficient * other.coefficient
		)

	def multiply_by_constant(self, c):
		return Monomial(
			self.powers.copy(),
			self.coefficient * c
		)

	def differentiate(self, d):
		current = self.copy()
		for i, a in enumerate(d.alpha):
			for _ in range(a):
				current = current.differentiate_single_component(i)
		return current

	def translate(self, center):
		return Polynomial([
			Monomial(monomial.powers, coefficient=self.coefficient * monomial.coefficient)
			for monomial in self.powers.translate(center).monomials
		])

	def transform(self, transformation):
		poly = Polynomial.constant_polynomial(self.coefficient, self.dimension)
		for i in range(self.dimension):
			d = self.powers.powers[i]
			if d == 0:
				continue
			component_poly = Polynomial([
				Monomial(
					MultiIndex([1 if idx == j else 0 for idx in range(self.dimension)]),
					coefficient=transformation[i, j]
				)
				for j in range(self.dimension)
			])
			for _ in range(d):
				poly = poly.multiply(component_poly)
		return poly

	def scale(self, radius):
		return Monomial(self.powers.copy(), self.coefficient / radius ** self.degree)

	@staticmethod
	def constant_monomial(value, dimension):
		return Monomial([0 for _ in range(dimension)], value)


def _group_by(l, key=lambda x: x):
	ret = {}
	for item in l:
		k = key(item)
		if k in ret:
			ret[k].append(item)
		else:
			ret[k] = [item]
	return ret


class Polynomial:
	def __init__(self, monomials=[]):
		self.monomials = monomials

	def collect(self):
		return Polynomial([
			Monomial(
				MultiIndex(list(index)),
				sum(t.coefficient for t in terms)
			)
			for index, terms in _group_by(
				self.monomials,
				lambda monomial: tuple(monomial.powers.powers)
			).items()
		])

	def negate(self):
		return Polynomial([
			Monomial(
				monomial.powers.copy(),
				-monomial.coefficient
			)
			for monomial in self.monomials
		])

	def copy(self):
		return Polynomial([m.copy() for m in self.monomials])

	def add(self, other):
		return Polynomial(
			self.copy().monomials + other.copy().monomials
		).collect()

	def subtract(self, other):
		return self.add(other.negate())

	def multiply(self, other):
		return Polynomial([
			x.multiply(y)
			for x in self.monomials
			for y in other.monomials
		]).collect()

	def multiply_by_constant(self, c):
		return Polynomial([
			m.multiply_by_constant(c)
			for m in self.monomials
		])

	def evaluate(self, x):
		return sum(m.evaluate(x) for m in self.monomials)

	def to_pyomo(self, model):
		term = 0
		for monomial in self.monomials:
			term = term + monomial.to_pyomo(model)
		return term

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return " + ".join([str(m) for m in self.monomials])

	@property
	def degree(self):
		return max(m.degree for m in self.monomials)

	@property
	def dimension(self):
		return self.monomials[0].dimension

	def to_matrix_form(self):
		Q = np.zeros((self.dimension, self.dimension))
		b = np.zeros(self.dimension)
		c = None
		for m in self.monomials:
			if m.degree == 0:
				c = m.coefficient
			elif m.degree == 1:
				b[np.array(m.powers.powers) > 0] = m.coefficient
			elif m.degree == 2:
				idxs = np.argwhere(np.array(m.powers.powers)).flatten()
				if len(idxs) == 1:
					Q[idxs[0], idxs[0]] = m.coefficient
				else:
					Q[idxs[0], idxs[1]] = 0.5 * m.coefficient
					Q[idxs[1], idxs[0]] = 0.5 * m.coefficient
			else:
				raise Exception('polynomial degree too high to convert to matrix form')
		return Quadratic.create(c, b, Q)

	def differentiate_single_component(self, component):
		return Polynomial([
			monomial.differentiate_single_component(component)
			for monomial in self.monomials
		])

	def evaluate_gradient(self, x):
		return [
			self.differentiate_single_component(i).evaluate(x)
			for i in range(self.dimension)
		]

	def translate(self, center):
		poly = None
		for monomial in self.monomials:
			p = monomial.translate(center)
			if poly is None:
				poly = p
			else:
				poly = poly.add(p)
		return poly

	def scale(self, radius):
		return Polynomial([
			monomial.scale(radius)
			for monomial in self.monomials
		])

	def transform(self, transformation):
		poly = None
		for monomial in self.monomials:
			p = monomial.transform(transformation)
			if poly is None:
				poly = p
			else:
				poly = poly.add(p)
		return poly

	def unshift(self, center, radius):
		return self.scale(radius).translate(center)

	def shift(self, center, radius):
		return self.translate(-center).scale(1.0/radius)

	@staticmethod
	def constant_polynomial(value, dimension):
		return Polynomial([Monomial.constant_monomial(value, dimension)])

	@staticmethod
	def construct_polynomial(basis, coeff):
		return Polynomial([Monomial(b, c) for b, c in zip(basis, coeff)])


def _construct_pascals_triangle(depth):
	prev = [0, 1, 0]
	for d in range(int(depth+1)):
		cur = [0] + [
			prev[i] + prev[i+1]
			for i in range(d+1)
		] + [0]
		prev = cur
	return prev[1:-1]

#
#
# N = 15
# poly = construct_polynomial(get_first_n_indices(N, 3), 2 * np.random.random(N) - 1)
# print(poly)
#
# for _ in range(10):
# 	center = np.random.random(3)
# 	radius = np.random.random() * 5
# 	# radius = 1.0
# 	point = 10 * np.random.random(3) - 5
#
# 	expected = poly.evaluate((point - center) / radius)
# 	actual = poly.shift(center, radius).evaluate(point)
#
# 	print(expected)
# 	print(actual)
#
# 	expected = poly.evaluate(point)
# 	actual = poly.shift(center, radius).unshift(center, radius).evaluate(point)
#
# 	print(expected)
# 	print(actual)




#
#
# n = 3
# N = 30
# poly = construct_polynomial(get_first_n_indices(N, n), 2 * np.random.random(N) - 1)
# print(poly)
#
# for _ in range(10):
# 	transform = np.random.random((n, n))
# 	point = np.random.random(n)
#
# 	expected = poly.evaluate(transform@point)
# 	actual = poly.transform(transform).evaluate(point)
#
# 	print(expected)
# 	print(actual)
#
# 	expected = poly.evaluate(point)
# 	actual = poly.transform(transform).transform(np.linalg.inv(transform)).evaluate(point)
#
# 	print(expected)
# 	print(actual)
