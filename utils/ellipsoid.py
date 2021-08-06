
import numpy as np
import logging


class Ellipsoid:
	def __init__(self):
		self.q = None
		self.center = None
		self.l = None
		self.r = None

		self.qinv = None
		self.linv = None

	def contained_within_polyhedron(self, poly, tol):
		if np.max(poly.evaluate(self.center)) > tol:
			return False
		A, b = poly.A, poly.b
		bbar = b - A @ self.center
		qinv = self.r ** 2 * self.qinv / 2.0
		for i in range(A.shape[0]):
			if A[i] @ qinv @ A[i] > 0.5 * bbar[i] ** 2 + tol:
				return False
		return True

	def contained_within_tr(self, center, radius, tol):
		try:
			if not np.max(np.abs(self.center - center)) < radius:
				return False
		except:
			print('here')
		# from utils.polyhedron import Polyhedron
		# poly = Polyhedron.create_from_linf_tr(center, radius)
		qinv = self.r ** 2 * self.qinv / 2.0
		for i in range(len(center)):
			bi = center[i] - self.center[i] + radius
			if qinv[i, i] > 0.5 * bi ** 2 + tol:
				return False
			bi = -center[i] + self.center[i] + radius
			if qinv[i, i] > 0.5 * bi ** 2 + tol:
				return False
		return True

	def scale_towards(self, point, t):
		# s = point + t * (x - point)
		# x = point + (s - point) / t

		# newcenter = t * self.center + (1 - t) * point

		# x - self.center
		# = point + (s - point) / t - self.center
		# = s / t - point / t + point - self.center
		# = s / t - [self.center - (t - 1) * point / t]
		# = s / t - [t * self.center - (t - 1) * point] / t
		# = s / t - newcenter / t

		# (x - self.center) @ self.q @ (x - self.center) <= r ** 2
		# (point + (s - point) / t - self.center) @ self.q @ (point + (s - point) / t - self.center) <= r ** 2
		# (s - newcenter) / t @ self.q @ (s - newcenter) / t <= r ** 2
		# (s - newcenter) @ self.q @ (s - newcenter) <= (t * r) ** 2

		if np.isnan(t) or np.isinf(t):
			print('here')
		scaled = Ellipsoid()
		scaled.center = t * self.center + (1 - t) * point
		scaled.r = t * self.r

		scaled.q = self.q
		scaled.l = self.l
		scaled.qinv = self.qinv
		scaled.linv = self.linv
		return scaled

	def evaluate(self, v):
		return (v - self.center) @ self.q @ (v - self.center) - self.r ** 2

	def contains(self, v):
		return self.evaluate(v) <= 0

	def shift_sample(self, sample):
		return (self.l @ (sample - self.center).T / self.r).T

	def shift_point(self, sample):
		return self.l @ (sample - self.center) / self.r

	def unshift_point(self, x):
		# self.l @ (sample - self.center) / self.r = y
		return self.center + self.r * self.linv @ x

	def shift_polynomial(self, poly):
		return poly.transform(self.linv).multiply_by_constant(self.r).translate(self.center)

	def unshift_polynomial(self, poly):
		return poly.translate(-self.center).multiply_by_constant(1.0/self.r).transform(self.l)

	def add_to_plot(self, plot_object, color='g'):
		plot_object.add_point(self.center, label='trust region center', color=color, s=20, marker="*")
		plot_object.add_contour(
			lambda x: -self.evaluate(x),
			label='ellipsoid',
			color=color,
			lvls=[-0.1, 0.0]
		)
		# if not detailed:
		# 	return
		# for d in self.ds:
		# 	plot_object.add_arrow(self.center, self.center + d, color="c")

	def to_json(self):
		return {'q': self.q, 'center': self.center, 'r': self.r}

	@staticmethod
	def parse_json(json):
		return Ellipsoid.create(
			np.array(json['q']),
			np.array(json['center']),
			json['r']
		)

	@staticmethod
	def create(q, center, r=1.0):
		ellipsoid = Ellipsoid()
		ellipsoid.r = r
		ellipsoid.center = center
		ellipsoid.q = q
		ellipsoid.l = np.linalg.cholesky(q).T

		if len(ellipsoid.l.shape) != 2:
			assert False, 'Bad shape of l'

		ellipsoid.linv = np.linalg.pinv(ellipsoid.l)
		try:
			ellipsoid.qinv = np.linalg.pinv(ellipsoid.q)
		except Exception:
			logging.exception("Unable to invert q")
			ellipsoid.qinv = ellipsoid.linv @ ellipsoid.linv.T
		return ellipsoid

	# def sample_shifted_region(self, num_points):
	# 	ret = [numpy.zeros(len(self.center))]
	# 	for d in sample_search_directions(len(self.center), num_points, include_axis=False):
	# 		ret.append(numpy.random.random() * d)
	# 	return ret



#
# q = np.random.random([5, 5])
# q = q.T@q
# l = np.linalg.cholesky(q).T
# linv = np.linalg.pinv(l)
#
# qinv = np.linalg.pinv(q)
# np.linalg.norm(q@linv.T@linv)
#
# np.linalg.norm(l@l.T - q)
# x = np.random.random(5)
# c = np.random.random(5)
#
#
# r = 1
# (x - c).T@q@(x - c) - r
# (x-c).T@l@l.T@(x-c) - r
# (l.T@x-l.T@c).T@(l.T@x-l.T@c) - r
#
#
