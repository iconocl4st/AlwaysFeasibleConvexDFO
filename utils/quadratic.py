
import numpy as np

from utils.assertions import make_assertion
from utils.default_stringable import DefaultStringable


class Quadratic(DefaultStringable):
	def __init__(self):
		self.c = None
		self.g = None
		self.Q = None

	def evaluate(self, x):
		return self.c + self.g @ x + x @ self.Q @ x

	def multi_eval(self, xs):
		return self.c + xs @ self.g + np.sum(xs * (self.Q @ xs.T).T, axis=1)

	def evaluate_gradient(self, x):
		return self.g + 2 * self.Q @ x

	def to_pyomo(self, x):
		n = len(self.g)
		return (
			self.c +
			sum(self.g[i] * x[i] for i in range(n)) +
			# sum(x[i] * self.Q[i, j] * x[j] for i in range(n) for j in range(n))
			sum(self.Q[i, i] * x[i] ** 2 for i in range(n)) +
			sum(2 * self.Q[i, j] * x[i] * x[j] for i in range(n) for j in range(i))
		)

	def unshift(self, ellipsoid):
		return self.compose(ellipsoid.l, ellipsoid.center, ellipsoid.r)

	def compose(self, L, center, r):
		# x = np.random.random(2)
		# s = L @ (x - center) / r
		# print('x', x)
		# print('shifted', s)
		# print('tmp-0', self.c + self.g @ s + s @ self.Q @ s)
		# print('tmp-1', self.c
		# 	  	+ self.g @ L @ (x - center) / r
		# 		+ (x - center) @ L.T @ self.Q @ L @ (x - center) / (r ** 2)
		# )
		# print('tmp-2', self.c
		# 	  + self.g @ L @ x / r
		# 	  - self.g @ L @ center / r
		# 	  + (x - center) @ L.T @ self.Q @ L @ (x - center) / (r ** 2)
		# )
		# print('tmp-3', self.c
		# 	  + self.g @ L @ x / r
		# 	  - self.g @ L @ center / r
		# 	  +  x @ L.T @ self.Q @ L @ x / (r ** 2)
		# 	  - 2 * center @ L.T @ self.Q @ L @ x / (r ** 2)
		# 	  + center @ L.T @ self.Q @ L @ center / (r ** 2)
		# )
		# print('tmp-4', self.c
		# 	  - self.g @ L @ center / r
		# 	  + center @ L.T @ self.Q @ L @ center / (r ** 2)
		# 	  + self.g @ L @ x / r
		# 	  - 2 * center @ L.T @ self.Q @ L @ x / (r ** 2)
		# 	  +  x @ L.T @ self.Q @ L @ x / (r ** 2)
		# )
		# print('tmp-5', self.c
		# 	  - self.g @ L @ center / r
		# 	  + center @ L.T @ self.Q @ L @ center / (r ** 2)
		# 	  + (self.g @ L / r
		# 	  - 2 * center @ L.T @ self.Q @ L / (r ** 2)) @ x
		# 	  +  x @ L.T @ self.Q @ L @ x / (r ** 2)
		# )

		# (x-c).T q (x-c) = x.T q x - 2 c.T q x + c.T q c
		quad0 = self.c \
			- self.g @ L @ center / r \
			+ center @ L.T @ self.Q @ L @ center / (r ** 2)
		quad1 = self.g @ L / r \
			- 2 * center @ L.T @ self.Q @ L / (r ** 2)
		quad2 = L.T @ self.Q @ L / (r ** 2)
		# print('tmp-6', quad0 + quad1 @ x + x @ quad2 @ x)
		ret = Quadratic.create(quad0, quad1, quad2)
		# print('tmp-7', ret.evaluate(x))
		return ret

	def __mul__(self, other):
		if np.isscalar(other):
			return Quadratic.create(other * self.c, other * self.g, other * self.Q)
		else:
			assert False, 'unknown type'

	def __add__(self, other):
		if type(other).__name__ == 'Quadratic':
			return Quadratic.create(other.c + self.c, other.g + self.g, other.Q + self.Q)
		else:
			assert False, 'unknown type'

	def to_json(self):
		return {'c': self.c, 'b': self.g, 'Q': self.Q}

	@staticmethod
	def parse_json(json_object):
		quad = Quadratic()
		quad.c = float(json_object['c'])
		quad.g = np.array(json_object['b'], dtype=np.float64)
		quad.Q = np.array(json_object['Q'], dtype=np.float64)
		return quad

	@staticmethod
	def create(c, g, Q):
		make_assertion(not np.isnan(c), 'nan in quadratic')
		make_assertion(not np.isnan(g).any(), 'nan in quadratic')
		make_assertion(not np.isnan(Q).any(), 'nan in quadratic')

		quad = Quadratic()
		quad.c = c
		quad.g = g
		quad.Q = Q
		return quad

	@staticmethod
	def zeros(n):
		return Quadratic.create(0.0, np.zeros(n), np.zeros([n, n]))
