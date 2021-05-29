import numpy as np


def a(l):
	return np.array(l, dtype=np.float64)


class Constraint:
	GE = '>=0'
	LE = '<=0'
	EQ = '==0'

	def __init__(self, expr, sense):
		self.expr = expr
		self.sense = sense


class Initial:
	def __init__(self, x0, f0, is_feasible):
		self.x0 = x0
		self.f0 = f0
		self.is_feasible = is_feasible


class Solution:
	def __init__(self, xstar, fstar, rstar):
		self.xstar = xstar
		self.fstar = fstar
		self.rstar = rstar


def _does_not_contain_none(vec):
	return vec is not None and np.all([xi is not None for xi in vec])


class Bounds:
	def __init__(self, lb, ub):
		self.lb = lb
		self.ub = ub

	@staticmethod
	def to_nomad(prefix, bound):
		if bound is None:
			return []
		return [
			prefix + ' ' + str(idx) + ' ' + str(b)
			for idx, b in enumerate(bound)
			if b is not None
		]

	@staticmethod
	def to_pdfo(bound, default, dim):
		if bound is None:
			return np.array([default] * dim, dtype=np.float64)
		return np.array([b if b is not None else default for b in bound], dtype=np.float64)

	def nomad_lb(self):
		return Bounds.to_nomad('LOWER_BOUND', self.lb)  # np.finfo(np.float32).min)

	def nomad_ub(self):
		return Bounds.to_nomad('UPPER_BOUND', self.ub)  # np.finfo(np.float32).max)

	def to_pdfo_lb(self, dim):
		# return Bounds.to_pdfo(np.finfo(np.float64).min, self.lb, dim)
		return Bounds.to_pdfo(self.lb, -np.infty, dim)

	def to_pdfo_ub(self, dim):
		# return Bounds.to_pdfo(np.finfo(np.float64).max, self.ub, dim)
		return Bounds.to_pdfo(self.ub, np.infty, dim)

	def has_lb(self):
		return _does_not_contain_none(self.lb)

	def has_ub(self):
		return _does_not_contain_none(self.ub)

	def to_constraint_functions(self):
		def glc(i, lbi):
			return lambda x: lbi - x[i]
		def guc(i, ubi):
			return lambda x: x[i] - ubi
		return (
			([] if self.lb is None else [
				glc(i, lbi)
				for i, lbi in enumerate(self.lb)
				if lbi is not None
			]) +
			([] if self.ub is None else [
				guc(i, ubi)
				for i, ubi in enumerate(self.ub)
				if ubi is not None
			])
		)
		# constraints = []
		# if self.lb is not None:
		# 	for i, lbi in enumerate(self.lb):
		# 		if lbi is None:
		# 			continue
		# 		constraints.append(lambda x: lbi - x[i])
		# if self.ub is not None:
		# 	for i, ubi in enumerate(self.ub):
		# 		if ubi is None:
		# 			continue
		# 		constraints.append(lambda x: x[i] - ubi)
		# return constraints


class Problem:
	def __init__(
		self,
		number,
		objective,
		constraints,
		bounds,
		initial,
		solution,
		name=None
	):
		self.number = number
		self.objective = objective
		self.constraints = constraints
		self.bounds = bounds
		self.initial = initial
		self.solution = solution
		self.name = name

	@property
	def n(self):
		return len(self.initial.x0)

	# def nomad_lb(self):
	# 	return Bounds.to_nomad(self.lb, '-inf')  # np.finfo(np.float32).min)
	#
	# def nomad_ub(self):
	# 	return Bounds.to_nomad(self.ub, 'inf')  # np.finfo(np.float32).max)

	def get_explicit_constraints(self):
		def get_ge_constraint(c):
			return lambda x: -c.expr(x)

		constraints = []
		for constraint in self.constraints:
			if constraint.sense == '==0':
				return False, None
			elif constraint.sense == '>=0':
				constraints.append(get_ge_constraint(constraint))
			else:
				constraints.append(constraint.expr)

		return True, constraints

	def get_all_le_constraints(self):
		def get_ge_constraint(c):
			return lambda x: -c.expr(x)

		bound_constraints = self.bounds.to_constraint_functions()
		success, explicit_constraints = self.get_explicit_constraints()
		if not success:
			return False, None
		return True, bound_constraints + explicit_constraints


class Data:
	p236_f = lambda x: a([
		1.0,
		x[0],
		x[0] ** 2,
		x[0] ** 3,
		x[0] ** 4,
		x[1],
		x[0] * x[1],
		x[0] ** 2 * x[1],
		x[0] ** 3 * x[1],
		x[0] ** 4 * x[1],
		x[1] ** 2,
		x[1] ** 3,
		x[1] ** 4,
		1 / (x[1] + 1),
		x[0] ** 2 * x[1] ** 2,
		x[0] ** 3 * x[1] ** 2,
		x[0] ** 3 * x[1] ** 3,
		x[0] * x[1] ** 2,
		x[0] * x[1] ** 3,
		np.exp(0.0005 * x[0] * x[1])
	])
	p236_b = a([
		75.1963666677,
		-3.8112755343,
		+0.1269366345,
		-0.0020567665,
		+0.0000103450,
		-6.8306567613,
		+0.0302344793,
		-0.0012813448,
		+0.0000352559,
		-0.0000002266,

		+0.2564581253,
		-0.0034604030,
		+0.0000135139,
		-28.1064434908,
		-0.0000052375,
		-0.0000000063,
		+0.0000000007,
		+0.0003405462,
		-2.8673112392,

		1.0
	])
	p236_obj = lambda x: -Data.p236_b @ Data.p236_f(x)

	# TODO: verify B = (H^TH)^{-1}
	p204_H = a([[-0.564255, 0.392417], [-0.404979, 0.927589], [-0.0735084, 0.535493]])
	p204_B = a([[5.66598, 2.77141], [2.77141, 2.12413]])
	p204_A = a([0.13294, -0.244378, 0.325895])
	p204_D = a([2.5074, -1.36401, 1.02282])

	p243_A = a([0.14272, -0.184918, -0.521869, -0.684306])
	p243_D = a([1.75168, -1.35195, -0.479048, -0.3648])
	p243_G = a([
		[-0.564255, +0.392417,  -0.404979],
		[+0.927589, -0.0735083, +0.535493],
		[+0.658799, -0.636666,  -0.681091],
		[-0.869487, +0.586387,  +0.289826]])
	p243_B = a([
		[+2.95137, +4.87407, -2.0506],
		[+4.87407, +9.39321, -3.93181],
		[-2.0506,  -3.93189, +2.64745]])

	p253_c = a([1] * 8)
	p253_a = a([
		[0, 10, 10, 0, 0, 10, 10, 0],
		[0, 0, 10, 10, 0, 0, 10, 10],
		[0, 0, 0, 0, 10, 10, 10, 10]
	])

	p268_D = a([
		[-74, 80, 18, -11, -4],
		[14, -69, 21, 28, 0],
		[66, -72, -5, 7, 1],
		[-12, 66, -30, -23, 3],
		[3, 8, -7, -4, 1],
		[4, -12, 4, 4, 0],
	])
	p268_d = a([51, -61, -56, 69, 10, -12])

	p284_c = a([20, 40, 400, 20, 80, 20, 40, 140, 380, 280, 80, 40, 140, 40, 120])
	p284_b = a([385, 470, 560, 565, 645, 430, 485, 455, 390, 460])
	p284_a = a([
		[100, 100,  10,   5,  10,   0,   0,  25,   0,  10,  55,   5,  45,  20,   0],
		[ 90, 100,  10,  35,  20,   5,   0,  35,  55,  25,  20,   0,  40,  25,  10],
		[ 70,  50,   0,  55,  25, 100,  40,  50,   0,  30,  60,  10,  30,   0,  40],
		[ 50,   0,   0,  65,  35, 100,  35,  60,   0,  15,   0,  75,  35,  30,  65],
		[ 50,  10,  70,  60,  45,  45,   0,  35,  65,   5,  75, 100,  75,  10,   0],
		[ 40,   0,  50,  95,  50,  35,  10,  60,   0,  45,  15,  20,   0,   5,   5],
		[ 30,  60,  30,  90,   0,  30,   5,  25,   0,  70,  20,  25,  70,  15,  15],
		[ 20,  30,  40,  25,  40,  25,  15,  10,  80,  20,  30,  30,   5,  65,  20],
		[ 10,  70,  10,  35,  25,  65,   0,  30,   0,   0,  25,   0,  15,  50,  55],
		[  5,  10, 100,   5,  20,   5,  10,  35,  95,  70,  20,  10,  35,  10,  30],
	])
	p284_get_constraint = lambda i: lambda x: Data.p284_b[i] - Data.p284_a[i] @ (x ** 2)

	p285_a = a([
		[100, 100,  10,   5,  10,   0,   0,  25,   0,  10,  55,   5,  45,  20,   0],
		[ 90, 100,  10,  35,  20,   5,   0,  35,  55,  25,  20,   0,  40,  25,  10],
		[ 70,  50,   0,  55,  25, 100,  40,  50,   0,  30,  60,  10,  30,   0,  40],
		[ 50,   0,   0,  65,  35, 100,  35,  60,   0,  15,   0,  75,  35,  30,  65],
		[ 50,  10,  70,  60,  45,  45,   0,  35,  65,   5,  75, 100,  75,  10,   0],
		[ 40,   0,  50,  95,  50,  35,  10,  60,   0,  45,  15,  20,   0,   3,   3],
		[ 30,  60,  30,  90,   0,  30,   5,  25,   0,  70,  20,  25,  70,  15,  15],
		[ 20,  30,  40,  25,  40,  25,  15,  10,  80,  20,  30,  30,   5,  65,  20],
		[ 10,  70,  10,  35,  25,  65,   0,  30,   0,   0,  25,   0,  15,  50,  55],
		[  5,  10, 100,   5,  20,   5,  10,  35,  95,  70,  20,  10,  35,  10,  30],
	])
	p285_b = a([385, 470, 560, 565, 645, 430, 485, 455, 390, 460])
	p285_get_constraint = lambda i: lambda x: Data.p285_b[i] - Data.p285_a[i] @ (x ** 2)

	p332_tis = [np.pi * (1 / 3.0 + (i - 1) / 180.0) for i in range(1, 101)]
	p332_pmax = lambda x: np.max([
		180 / np.pi * np.arctan(abs((1 / ti - x[0]) / (np.log(ti) + x[1])))
		for ti in Data.p332_tis
	])

	p334_y = a([0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.25, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39])

	p353_q = lambda x: (
		(0.53 * x[0]) ** 2 + (0.44 * x[1]) ** 2 + (4.5 * x[2]) ** 2 + (0.79 * x[3]) ** 2
	)

	p359_data = a([
		[-8720288.849, -145421.402, -155011.1084, -326669.5104],
		[150512.5253, 2931.1506, 4360.53352, 7390.68412],
		[-156.6950325, -40.428932, 12.9492344, -27.8986976],
		[476470.3222, 5106.192, 10236.884, 16643.076],
		[729482.8271, 15711.36, 13176.786, 30988.146]
	])
	p359_a = p359_data[:, 0]
	p359_b = p359_data[:, 1]
	p359_c = p359_data[:, 2]
	p359_d = p359_data[:, 3]

	p57_a = a([8] * 2 + [10] * 4 + [12] * 4 + [14] * 3 + [16] * 3 + [18] * 2 + [20] * 3 + [22] * 3 + [24] * 3 +
			  [26] * 3 + [28] * 2 + [30] * 3 + [32] * 2 + [34] + [36] * 2 + [38] * 2 + [40, 42])
	p57_b = a([.49, .49, .48, .47, .48, .47, .46, .46, .45, .43, .45, .43, .43, .44, .43, .43, .46, .45, .42, .42, .43, .41,
			   .41, .40, .42, .40, .40, .41, .40, .41, .41, .40, .40, .40, .38, .41, .40, .40, .41, .38, .40, .40, .39, .39])
	p57_fi = lambda x, i: Data.p57_b[i] - x[0] - (0.49 - x[0]) * np.exp(-x[1] * (Data.p57_a[i] - 8))

	p105_y = a([
		95 + i * 5
		for i, length in enumerate([
			1, 0, 1, 6-3, 10-7, 25-11, 40-26, 55-41, 68-56, 89-69, 101-90, 118-102, 122-119, 142-123, 150-143, 167-151,
			175-168, 181-176, 187-182, 194-188, 198-195, 201-199, 204-202, 212-205, 1, 219-214, 0, 224-220, 1, 232-226, 1, 235-234])
		for _ in range(length)])

	p105_a = lambda i, x: x[0] / x[5] * np.exp(-(Data.p105_y[i] - x[2]) ** 2 / (2 * x[5] ** 2))
	p105_b = lambda i, x: x[1] / x[6] * np.exp(-(Data.p105_y[i] - x[3]) ** 2 / (2 * x[6] ** 2))
	p105_c = lambda i, x: (1 - x[1] - x[0]) / x[7] * np.exp(-(Data.p105_y[i] - x[4]) ** 2 / (2 * x[7] ** 2))

	@staticmethod
	def p67_y(x):
		y2 = 1.6 * x[0]
		while True:
			y3 = 1.22 * y2 - x[0]
			y6 = (x[1] + y3) / x[0]
			y2c = 0.01 * x[0] * (112 + 13.167 * y6 - 0.6667 * y6 ** 2)
			if abs(y2c - y2) <= 0.001:
				break
			y2 = y2c
		y4 = 93
		while True:
			y5 = 86.35 + 1.098 * y6 - 0.038 * y6 ** 2 + 0.325 * (y4 - 89)
			y8 = 3 * y5 - 133
			y7 = 35.82 - 0.222 * y8
			y4c = 98000 * x[2] / (y2 * y7 + 1000 * x[2])
			if abs(y4c - y4) <= 0.001:
				break
			y4 = y4c
		#		  0,  1,  2,  3,  4,  5,  6,  7,  8
		return a([0,  0, y2, y3, y4, y5, y6, y7, y8])

	p67_a = a([0, 0, 85, 91, 3, .01, 145, 5000, 2000, 93, 95, 12, 4, 162])
	@staticmethod
	def p67_obj(x):
		y = Data.p67_y(x)
		return -(0.063 * y[2] * y[5] - 5.04 * x[0] - 3.36 * y[3] - 0.35 * x[1] - 10 * x[2])
	p67_c1 = lambda i: lambda x: Data.p67_y(x)[i + 1] - Data.p67_a[i - 1]
	p67_c2 = lambda i: lambda x: Data.p67_a[i - 1] - Data.p67_y(x)[i - 6]

	p84_a = a([-24345, -8720288.849, 150512.5253, -156.6950325, 476470.3222, 729482.8271, -145421.402, 2931.1506, -40.427932, 5106.192,
			   15711.36, -155011.1084, 4360.53352, 12.9492344, 10236.844, 13176.785, -326669.5104, 7390.68412, -27.8986976, 16643.076, 30988.146])
	p84_c1 = lambda x: Data.p84_a[ 6] * x[0] + Data.p84_a[ 7] * x[0] * x[1] + Data.p84_a[ 8] * x[0] * x[2] + Data.p84_a[ 9] * x[0] * x[2] + Data.p84_a[10] * x[0] * x[4]
	p84_c2 = lambda x: Data.p84_a[11] * x[0] + Data.p84_a[12] * x[0] * x[1] + Data.p84_a[13] * x[0] * x[2] + Data.p84_a[14] * x[0] * x[2] + Data.p84_a[15] * x[0] * x[4]
	p84_c3 = lambda x: Data.p84_a[16] * x[0] + Data.p84_a[17] * x[0] * x[1] + Data.p84_a[18] * x[0] * x[2] + Data.p84_a[19] * x[0] * x[2] + Data.p84_a[20] * x[0] * x[4]

	p86_data = a([
		[  -15,   -27,   -36,   -18,   -12],
		[   30,   -20,   -10,    32,   -10],
		[  -20,    39,    -6,   -31,    32],
		[  -10,    -6,    10,    -6,   -10],
		[   32,   -31,    -6,    39,   -20],
		[  -10,    32,   -10,   -20,    30],
		[    4,     8,    10,     6,     2],
		[  -16,    20,     0,     1,     0],
		[    0,    -2,     0,     4,     2],
		[ -3.5,     0,     2,     0,     0],
		[    0,    -2,     0,    -4,    -1],
		[    0,    -9,    -2,     1,  -2.8],
		[    2,     0,    -4,     0,     0],
		[   -1,    -1,    -1,    -1,    -1],
		[   -1,    -2,    -3,    -2,    -1],
		[    1,     2,     3,     4,     5],
		[    1,     1,     1,     1,     1],
		[  -40,    -2, -0.25,    -4,    -4],
		[   -1,   -40,   -60,     5,     1],
	])
	p86_e = p86_data[0, :]
	p86_c = p86_data[1:6, :]
	p86_d = p86_data[6, :]
	p86_a = p86_data[7:17, :]
	p86_b = np.reshape(p86_data[17:19, :], newshape=(10,))


class HottSchittowski:
	@staticmethod
	def get_problem_by_number(number):
		for problem in HottSchittowski.PROBLEMS:
			if problem.number == number:
				return problem
		return None

	PROBLEMS = [
		############################################################################
		## Book 1
		############################################################################
		Problem(
			number=201,
			objective=lambda x: 4 * (x[0] - 5) ** 2 + (x[1] - 6) ** 2,
				# array_expr=lambda x: (x - a([5, 6])) ** 2 @ a([4, 1]),
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([8, 9]), f0=45, is_feasible=True),
			solution=Solution(xstar=a([5, 6]), fstar=0, rstar=0),
		),
		Problem(
			number=202,
			objective=lambda x:
				(-13 + x[0] - 2 * x[1] + 5 * x[1] ** 2 - x[1] ** 3) ** 2 +
				(-29 + x[0] - 14 * x[1] + x[1] ** 2 + x[1] ** 3) ** 2,
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([15, -2]), f0=1256, is_feasible=True),
			solution=Solution(xstar=a([5, 4]), fstar=0, rstar=0)
		),
		Problem(
			number=203,
			objective=lambda x: sum([
				ui ** 2
				for ui in [
					ci - x[0] * (1 - x[1] ** (i + 1))
					for i, ci in enumerate(a([1.5, 2.25, 2.625]))]]),
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([2, 0.2]), f0=0.529781, is_feasible=True),
			solution=Solution(xstar=a([3, 0.5]), fstar=0, rstar=0)
		),
		Problem(
			number=204,
			objective=lambda x: sum(
				(Data.p204_A + Data.p204_H @ x + 0.5 * (x @ Data.p204_B @ x) * Data.p204_D) ** 2
			),
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([0.1, 0.1]), f0=0.190330, is_feasible=True),
			solution=Solution(xstar=a([0, 0]), fstar=0.183601, rstar=0)
		),
		Problem(
			number=205,
			objective=lambda x: (
				(1.5 - x[0] * (1 - x[1])) ** 2 +
				(2.25 - x[0] * (1 - x[1] ** 2)) ** 2 +
				(2.625 - x[0] * (1 - x[1] ** 3)) ** 2
			),
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([0, 0]), f0=14.2031, is_feasible=True),
			solution=Solution(xstar=a([3, 0.5]), fstar=0, rstar=0)
		),
		Problem(
			number=206,
			objective=lambda x: (x[1] - x[0] ** 2) ** 2 + 100 * (1 - x[0]) ** 2,
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-1.2, 1]), f0=484.194, is_feasible=True),
			solution=Solution(xstar=a([1, 1]), fstar=0, rstar=0)
		),
		Problem(
			name='banana function',
			number=207,
			objective=lambda x: (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-1.2, 1]), f0=5.03360, is_feasible=True),
			solution=Solution(xstar=a([1, 1]), fstar=0, rstar=0)
		),
		Problem(
			name='banana function',
			number=208,
			objective=lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-1.2, 1]), f0=24.2, is_feasible=True),
			solution=Solution(xstar=a([1, 1]), fstar=0, rstar=0)
		),
		Problem(
			name='banana function',
			number=209,
			objective=lambda x: 10000 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-1.2, 1]), f0=1940.84, is_feasible=True),
			solution=Solution(xstar=a([1, 1]), fstar=0, rstar=0)
		),
		Problem(
			name='banana function',
			number=210,
			objective=lambda x: 1000000 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-1.2, 1]), f0=193605, is_feasible=True),
			solution=Solution(xstar=a([1, 1]), fstar=0, rstar=0)
		),
		Problem(
			number=211,
			objective=lambda x: 100 * (x[1] - x[0] ** 3) ** 2 + (1 - x[0]) ** 2,
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-1.2, 1]), f0=749.038, is_feasible=True),
			solution=Solution(xstar=a([1, 1]), fstar=0, rstar=0)
		),
		Problem(
			number=212,
			objective=lambda x:
				(4 * (x[0] + x[1])) ** 2 +
				(
					4 * (x[0] + x[1]) +
					(x[0] - x[1]) * ((x[0] - 2) ** 2 + x[1] ** 2 - 1)
				) ** 2,
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([2, 0]), f0=100, is_feasible=True),
			solution=Solution(xstar=a([0, 0]), fstar=0, rstar=0)
		),
		Problem(
			number=213,
			objective=lambda x: (10 * (x[0] - x[1]) ** 2 + (x[0] - 1) ** 2) ** 4,
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([3, 1]), f0=0.374810e7, is_feasible=True),
			solution=Solution(xstar=a([0, 0]), fstar=0, rstar=0)
		),
		Problem(
			number=214,
			objective=lambda x: (10 * (x[0] - x[1]) ** 2 + (x[0] - 1) ** 2) ** 0.25,
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-1.2, 1]), f0=2.70122, is_feasible=True),
			solution=Solution(xstar=a([1, 1]), fstar=0, rstar=0)
		),
		Problem(
			number=215,
			objective=lambda x: x[1],
			constraints=[
				Constraint(expr=lambda x: x[1] - x[0] ** 2, sense=Constraint.GE)],
			bounds=Bounds(lb=[0, None], ub=None),
			initial=Initial(x0=a([1, 1]), f0=1, is_feasible=True),
			solution=Solution(xstar=a([0, 0]), fstar=0, rstar=0)
		),
		Problem(
			number=216,
			objective=lambda x: 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2,
			constraints=[
				Constraint(expr=lambda x: x[0] * (x[0] - 4) - 2 * x[1] + 12, sense=Constraint.EQ)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-1.2, 1]), f0=24.2, is_feasible=False),
			solution=Solution(xstar=a([2, 4]), fstar=1, rstar=0)
		),
		Problem(
			number=217,
			objective=lambda x: -x[1],
			constraints=[
				Constraint(expr=lambda x: 1 + x[0] - 2 * x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] ** 2 + x[1] ** 2 - 1, sense=Constraint.EQ)],
			bounds=Bounds(lb=[0, None], ub=None),
			initial=Initial(x0=a([10, 10]), f0=-10, is_feasible=False),
			solution=Solution(xstar=a([0.6, 0.8]), fstar=-0.8, rstar=0)
		),
		Problem(
			number=218,
			objective=lambda x: x[1],
			constraints=[
				Constraint(expr=lambda x: x[1] - x[0] ** 2, sense=Constraint.GE)],
			bounds=Bounds(lb=[None, 0], ub=None),
			initial=Initial(x0=a([9, 100]), f0=100, is_feasible=True),
			solution=Solution(xstar=a([0, 0]), fstar=0, rstar=0)
		),
		Problem(
			number=219,
			objective=lambda x: -x[0],
			constraints=[
				Constraint(expr=lambda x: x[1] - x[0] ** 3 - x[2] ** 2, sense=Constraint.EQ),
				Constraint(expr=lambda x: x[0] ** 2 - x[1] - x[3] ** 2, sense=Constraint.EQ)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([10, 10, 10, 10]), f0=-10, is_feasible=False),
			solution=Solution(xstar=a([1, 1, 0, 0]), fstar=-1, rstar=0)
		),
		Problem(
			number=220,
			objective=lambda x: x[0],
			constraints=[
				Constraint(expr=lambda x: (x[0] - 1) ** 3 - x[1], sense=Constraint.EQ)],
			bounds=Bounds(lb=[1, 0], ub=None),
			initial=Initial(x0=a([25000, 25000]), f0=-25000, is_feasible=True),
			solution=Solution(xstar=a([1, 0]), fstar=1, rstar=0)
		),
		Problem(
			number=221,
			objective=lambda x: -x[0],
			constraints=[
				Constraint(expr=lambda x: (1 - x[0]) ** 3 - x[1], sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0], ub=None),
			initial=Initial(x0=a([0.25, 0.25]), f0=-0.25, is_feasible=True),
			solution=Solution(xstar=a([1, 0]), fstar=-1, rstar=0)
		),
		Problem(
			number=222,
			objective=lambda x: -x[0],
			constraints=[
				Constraint(expr=lambda x: 0.125 - x[1] + (1 - x[0]) ** 2, sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0], ub=None),
			initial=Initial(x0=a([1.3, 0.2]), f0=-1.3, is_feasible=False),
			solution=Solution(xstar=a([1.5, 0]), fstar=-1.5, rstar=0)
		),
		Problem(
			number=223,
			objective=lambda x: -x[0],
			constraints=[
				Constraint(expr=lambda x: np.exp(np.exp(x[0])), sense=Constraint.GE),
				Constraint(expr=lambda x: x[1] - np.exp(np.exp(x[0])), sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0], ub=[10, 10]),
			initial=Initial(x0=a([0.1, 3.3]), f0=-0.1, is_feasible=True),
			solution=Solution(xstar=a([np.log(np.log(10)), 10]), fstar=-np.log(np.log(10)), rstar=0)
			# solution=Solution(xstar=a([0.834, 10]), fstar=-0.834032, rstar=0)
		),
		Problem(
			number=224,
			objective=lambda x: 2 * x[0] ** 2 + x[1] ** 2 - 48 * x[0] - 40 * x[1],
			constraints=[
				Constraint(expr=lambda x: x[0] + 3 * x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: 18 - x[0] - 3 * x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] + x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: 8 - x[0] - x[1], sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0], ub=[6, 6]),
			initial=Initial(x0=a([0.1, 0.1]), f0=-8.77, is_feasible=True),
			solution=Solution(xstar=a([4, 4]), fstar=-304, rstar=0)
		),
		Problem(
			number=225,
			objective=lambda x: x[0] ** 2 + x[1] ** 2,
			constraints=[
				Constraint(expr=lambda x: x[0] + x[1] - 1, sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] ** 2 + x[1] ** 2 - 1, sense=Constraint.GE),
				Constraint(expr=lambda x: 9 * x[0] ** 2 + x[1] ** 2 - 9, sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] ** 2 - x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: -x[1] ** 2 + x[0], sense=Constraint.GE)],  # Changing this from the books
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([3, 1]), f0=10, is_feasible=True),  # How is this feasible????
			solution=Solution(xstar=a([1, 1]), fstar=2, rstar=0)
		),
		Problem(
			number=226,
			objective=lambda x: -x[0] * x[1],
			constraints=[
				Constraint(expr=lambda x: x[0] ** 2 + x[1] ** 2, sense=Constraint.GE),
				Constraint(expr=lambda x: 1 - x[0] ** 2 - x[1] ** 2, sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0], ub=None),
			initial=Initial(x0=a([0.8, 0.05]), f0=-0.04, is_feasible=True),
			solution=Solution(xstar=a([1.0 / np.sqrt(2.0)] * 2), fstar=-0.5, rstar=0)
		),
		Problem(
			number=227,
			objective=lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
			constraints=[
				Constraint(expr=lambda x: -x[0] ** 2 + x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] - x[1] ** 2, sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([0.5, 0.5]), f0=2.5, is_feasible=True),
			solution=Solution(xstar=a([1, 1]), fstar=1, rstar=0)
		),
		Problem(
			number=228,
			objective=lambda x: x[0] ** 2 + x[1],
			constraints=[
				Constraint(expr=lambda x: -x[0] - x[1] + 1, sense=Constraint.GE),
				Constraint(expr=lambda x: -(x[0] ** 2 + x[1] ** 2) + 9, sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([0, 0]), f0=0, is_feasible=True),
			solution=Solution(xstar=a([0, -3]), fstar=-3, rstar=0)
		),
		Problem(
			number=229,
			objective=lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
			constraints=[],
			bounds=Bounds(lb=[-2, -2], ub=[2, 2]),
			initial=Initial(x0=a([-1.2, 1]), f0=24.2, is_feasible=True),
			solution=Solution(xstar=a([1, 1]), fstar=0, rstar=0)
		),
		Problem(
			name='Chamberlain-problem',
			number=230,
			objective=lambda x: x[1],
			constraints=[
				Constraint(expr=lambda x: -2 * x[0] ** 2 + x[0] ** 3 + x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: -2 * (1 - x[0]) ** 2 + (1 - x[0]) ** 3 + x[1], sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([0, 0]), f0=0, is_feasible=False),
			solution=Solution(xstar=a([0.5, 0.375]), fstar=0.375, rstar=0)
		),
		Problem(
			number=231,
			objective=lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
			constraints=[
				Constraint(expr=lambda x: +x[0] / 3 + x[1] + 0.1, sense=Constraint.GE),
				Constraint(expr=lambda x: -x[0] / 3 + x[1] + 0.1, sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-1.2, 1]), f0=24.2, is_feasible=True),
			solution=Solution(xstar=a([1, 1]), fstar=0, rstar=0)
		),
		Problem(
			number=232,
			objective=lambda x: -(9 - (x[0] - 3) ** 2) * x[1] ** 3 / (27 * np.sqrt(3)),
			constraints=[
				Constraint(expr=lambda x: x[0] / np.sqrt(3) - x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] + np.sqrt(3) * x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: 6 - x[0] - np.sqrt(3) * x[1], sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0], ub=None),
			initial=Initial(x0=a([2, 0.5]), f0=-0.0213833, is_feasible=True),
			solution=Solution(xstar=a([3, np.sqrt(3)]), fstar=-1, rstar=0)
		),
		Problem(
			number=233,
			objective=lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
			constraints=[
				Constraint(expr=lambda x: x[0] ** 2 + x[1] ** 2 - 0.25, sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([1.2, 1]), f0=19.4, is_feasible=True),
			solution=Solution(xstar=a([1, 1]), fstar=0, rstar=0)
		),
		Problem(
			number=234,
			objective=lambda x: (x[1] - x[0]) ** 4 - (1 - x[0]),
			constraints=[
				Constraint(expr=lambda x: -x[0] ** 2 - x[1] ** 2 + 1, sense=Constraint.GE)],
			bounds=Bounds(lb=[0.2, 0.2], ub=[2, 2]),
			initial=Initial(x0=a([0, 0]), f0=-1, is_feasible=False),
			solution=Solution(xstar=a([0.2, 0.2]), fstar=-0.8, rstar=0)
		),
		Problem(
			number=235,
			objective=lambda x: 0.01 * (x[0] - 1) ** 2 + (x[1] - x[0] ** 2) ** 2,
			constraints=[
				Constraint(expr=lambda x: x[0] + x[2] ** 2 + 1, sense=Constraint.EQ)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-2, 3, 1]), f0=1.09, is_feasible=True),
			solution=Solution(xstar=a([-1, 1, 0]), fstar=0.04, rstar=0)
		),
		Problem(
			number=236,
			objective=Data.p236_obj,
			constraints=[
				Constraint(expr=lambda x: x[0] * x[1] - 700, sense=Constraint.GE),
				Constraint(expr=lambda x: x[1] - 5 * (x[0] / 25) ** 2, sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0], ub=[75, 65]),
			initial=Initial(x0=a([95, 10]), f0=86.0371, is_feasible=False),
			solution=Solution(xstar=a([75, 65]), fstar=-58.9034, rstar=0)
		),
		Problem(
			number=237,
			objective=Data.p236_obj,
			constraints=[
				Constraint(expr=lambda x: x[0] * x[1] - 700, sense=Constraint.GE),
				Constraint(expr=lambda x: x[1] - 5 * (x[0] / 25) ** 2, sense=Constraint.GE),
				Constraint(expr=lambda x: (x[1] - 50) ** 2 - 5 * (x[0] - 55), sense=Constraint.GE)],
			bounds=Bounds(lb=[54, None], ub=[75, 65]),
			initial=Initial(x0=a([95, 10]), f0=86.0371, is_feasible=False),
			solution=Solution(xstar=a([75, 65]), fstar=-58.9034, rstar=0)
		),
		Problem(
			number=238,
			objective=Data.p236_obj,
			constraints=[
				Constraint(expr=lambda x: x[0] * x[1] - 700, sense=Constraint.GE),
				Constraint(expr=lambda x: x[1] - 5 * (x[0] / 25) ** 2, sense=Constraint.GE),
				Constraint(expr=lambda x: (x[1] - 50) ** 2 - 5 * (x[0] - 55), sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=[75, 65]),
			initial=Initial(x0=a([95, 10]), f0=86.0371, is_feasible=False),
			solution=Solution(xstar=a([75, 65]), fstar=-58.9034, rstar=0)
		),
		Problem(
			number=239,
			objective=Data.p236_obj,
			constraints=[
				Constraint(expr=lambda x: x[0] * x[1] - 700, sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0], ub=[75, 65]),
			initial=Initial(x0=a([95, 10]), f0=86.0371, is_feasible=False),
			solution=Solution(xstar=a([75, 65]), fstar=-58.9034, rstar=0)
		),
		Problem(
			number=240,
			objective=lambda x: (x[0] - x[1] + x[2]) ** 2 + (-x[0] + x[1] + x[2]) ** 2 + (x[0] + x[1] - x[2]) ** 2,
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([100, -1, 2.5]), f0=29726.8, is_feasible=False),
			solution=Solution(xstar=a([0, 0, 0]), fstar=0, rstar=0)
		),
		Problem(
			number=241,
			objective=lambda y: sum([
				fi(y) ** 2
				for fi in [
					lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1,
					lambda x: x[0] ** 2 + x[1] ** 2 + (x[2] - 2) ** 2 - 1,
					lambda x: x[0] + x[1] + x[2] - 1,
					lambda x: x[0] + x[1] - x[2] + 1,
					lambda x: x[0] ** 3 + 3 * x[1] ** 2 + (5 * x[2] - x[0] + 1) ** 2 - 36
				]
			]),
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([1, 2, 0]), f0=629, is_feasible=True),
			solution=Solution(xstar=a([0, 0, 1]), fstar=0, rstar=0)
		),
		Problem(
			number=242,
			objective=lambda x: sum([
				(
					np.exp(-x[0] * ti) - np.exp(-x[1] * ti) -
					x[2] * (np.exp(-ti) - np.exp(-10 * ti))
				) ** 2
				for ti in [0.1 * i for i in range(1, 11)]
			]),
			constraints=[],
			bounds=Bounds(lb=a([0, 0, 0]), ub=a([10, 10, 10])),
			initial=Initial(x0=a([2.5, 10, 10]), f0=275.881, is_feasible=True),
			solution=Solution(xstar=a([1, 10, 1]), fstar=0, rstar=0)
		),
		Problem(
			number=243,
			objective=lambda x: sum(
				(Data.p243_A + Data.p243_G @ x + 0.5 * (x @ Data.p243_B @ x) * Data.p243_D) ** 2
			),
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([0.1, 0.1, 0.1]), f0=0.939853, is_feasible=True),
			solution=Solution(xstar=a([0, 0, 0]), fstar=0.7966, rstar=0)
		),
		Problem(
			name="Bigg's Function",
			number=244,
			objective=lambda x: sum([
				(np.exp(-x[0] * zi) - x[2] * np.exp(-x[1] * zi) - yi) ** 2
				for yi, zi in [
					(np.exp(-zi) - 5 * np.exp(-10 * zi), zi)
					for zi in [0.1 * i for i in range(1, 11)]]]),
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([1, 2, 1]), f0=1.55347, is_feasible=True),
			solution=Solution(xstar=a([1, 10, 5]), fstar=0, rstar=0)
		),
		Problem(
			name="Bigg's Function",
			number=244,
			objective=lambda x: sum([
				(np.exp(-x[0] * zi) - x[2] * np.exp(-x[1] * zi) - yi) ** 2
				for yi, zi in [
					(np.exp(-zi) - 5 * np.exp(-10 * zi), zi)
					for zi in [0.1 * i for i in range(1, 11)]]]),
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([1, 2, 1]), f0=1.55347, is_feasible=True),
			solution=Solution(xstar=a([1, 10, 5]), fstar=0, rstar=0)
		),
		Problem(
			number=245,
			objective=lambda x: sum([
				(
					np.exp(-i * x[0] / 10) - np.exp(-i * x[1] / 10) -
					x[2] * (np.exp(-i / 10) - np.exp(-i))
				) ** 2
				for i in range(1, 11)]),
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([0, 10, 20]), f0=1031.15, is_feasible=True),
			solution=Solution(xstar=a([1, 10, 1]), fstar=0, rstar=0)
		),
		Problem(
			number=246,
			objective=lambda x: (
				100 * (x[2] - ((x[0] + x[1]) / 2) ** 2) ** 2 +
				(1 - x[0]) ** 2 + (1 - x[1]) ** 2
			),
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-1.2, 2, 0]), f0=8.4, is_feasible=True),
			solution=Solution(xstar=a([1, 1, 1]), fstar=0, rstar=0)
		),
		# TODO: Problem 247 Missing
		Problem(
			name='Around-the-World-Problem',
			number=248,
			objective=lambda x: -x[1],
			constraints=[
				Constraint(expr=lambda x: 1 - 2 * x[1] + x[0], sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1, sense=Constraint.EQ)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-0.1, -1, 0.1]), f0=1, is_feasible=True),
			solution=Solution(xstar=a([0.6, 0.8, 0]), fstar=-0.8, rstar=0)
		),
		Problem(
			name='Around-the-World-Problem',
			number=249,
			objective=lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2,
			constraints=[
				Constraint(expr=lambda x: x[0] ** 2 + x[1] ** 2 - 1, sense=Constraint.GE)],
			bounds=Bounds(lb=[1, None, None], ub=None),
			initial=Initial(x0=a([1, 1, 1]), f0=3, is_feasible=True),
			solution=Solution(xstar=a([1, 0, 0]), fstar=1, rstar=0)
		),
		Problem(
			name="Rosenbrock's-Post-Office-Prob.",
			number=250,
			objective=lambda x: -x[0] * x[1] * x[2],
			constraints=[
				Constraint(expr=lambda x: x[0] + 2 * x[1] + 2 * x[2], sense=Constraint.GE),
				Constraint(expr=lambda x: 72 - x[0] - 2 * x[1] - 2 * x[2], sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0], ub=[20, 11, 42]),
			initial=Initial(x0=a([10, 10, 10]), f0=-1000, is_feasible=True),
			solution=Solution(xstar=a([20, 11, 15]), fstar=-3300, rstar=0)
		),
		Problem(
			number=251,
			objective=lambda x: -x[0] * x[1] * x[2],
			constraints=[
				Constraint(expr=lambda x: x[0] + 2 * x[1] + 2 * x[2], sense=Constraint.GE),
				Constraint(expr=lambda x: 72 - x[0] - 2 * x[1] - 2 * x[2], sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0], ub=[42, 42, 42]),
			initial=Initial(x0=a([10, 10, 10]), f0=-1000, is_feasible=True),
			solution=Solution(xstar=a([24, 12, 12]), fstar=-3456, rstar=0)
		),
		Problem(
			number=252,
			objective=lambda x: 0.01 * (x[0] - 1) ** 2 + (x[1] - x[0] ** 2) ** 2,
			constraints=[
				Constraint(expr=lambda x: x[0] + x[2] ** 2 + 1, sense=Constraint.EQ)],
			bounds=Bounds(lb=None, ub=[-1, None, None]),
			initial=Initial(x0=a([-1, 2, 2]), f0=1.04, is_feasible=False),
			solution=Solution(xstar=a([-1, 1, 0]), fstar=0.04, rstar=0)
		),
		Problem(
			number=253,
			# TODO: Could make this neater...
			objective=lambda x: sum([
				Data.p253_c[i] * np.sqrt(
					(Data.p253_a[0, i] - x[0]) ** 2 +
					(Data.p253_a[1, i] - x[1]) ** 2 +
					(Data.p253_a[2, i] - x[2]) ** 2
				)
				for i in range(8)
			]),
			constraints=[
				Constraint(expr=lambda x: 30 - 3 * x[0] - 3 * x[2], sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0], ub=None),
			initial=Initial(x0=a([0, 2, 0]), f0=86.5395, is_feasible=True),
			solution=Solution(xstar=a([0.3333, 0.3333, 0.3333]), fstar=87.3794, rstar=0)
		),
		Problem(
			number=254,
			objective=lambda x: np.log(x[2]) - x[1],
			constraints=[
				Constraint(expr=lambda x: x[1] ** 2 + x[2] ** 2 - 4, sense=Constraint.EQ),
				Constraint(expr=lambda x: x[2] - 1 - x[0] ** 2, sense=Constraint.EQ)],
			bounds=Bounds(lb=[None, None, 1], ub=None),
			initial=Initial(x0=a([1, 1, 1]), f0=-1, is_feasible=False),
			solution=Solution(xstar=a([0, np.sqrt(3), 1]), fstar=-np.sqrt(3), rstar=0)
		),
		Problem(
			number=255,
			objective=lambda x: (
				100 * (x[1] - x[0] ** 2) + (1 - x[0]) ** 2 + 90 * (x[3] - x[2] ** 2) + (1 - x[2]) ** 2 +
				10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) + 19.8 * (x[1] - 1) * (x[3] - 1)
			),
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-3, 1, -3, 1]), f0=-1488, is_feasible=True),
			solution=Solution(xstar=a([1, 1, 1, 1]), fstar=0, rstar=0)
		),
		Problem(
			name="Powell's Function",
			number=256,
			objective=lambda x:
				(x[0] + 10 * x[1]) ** 2 + 5 * (x[2] - x[3]) ** 2 + (x[1] - 2 * x[2]) ** 4 + 10 * (x[0] - x[3]) ** 4,
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([3, -1, 0, 1]), f0=215, is_feasible=True),
			solution=Solution(xstar=a([0, 0, 0, 0]), fstar=0, rstar=0)
		),
		Problem(
			number=257,
			objective=lambda x: (
				100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2 + 90 * (x[2] ** 2 - x[3]) ** 2 + (x[2] - 1) ** 2 +
				10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) + 19.8 * (x[0] - 1) * (x[3] - 1)
			),
			constraints=[],
			bounds=Bounds(lb=[0, None, 0, None], ub=None),
			initial=Initial(x0=a([-3, -1, -3, -1]), f0=19271.2, is_feasible=False),
			solution=Solution(xstar=a([1, 1, 1, 1]), fstar=0, rstar=0)
		),
		Problem(
			name="Wood's Function",
			number=258,
			objective=lambda x: (
				100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 + 90 * (x[3] - x[2] ** 2) ** 2 + (1 - x[2]) ** 2 +
				10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) + 19.8 * (x[1] - 1) * (x[3] - 1)
			),
			constraints=[],
			bounds=Bounds(lb=[0, None, 0, None], ub=None),
			initial=Initial(x0=a([-3, -1, -3, -1]), f0=19192, is_feasible=True),
			solution=Solution(xstar=a([1, 1, 1, 1]), fstar=0, rstar=0)
		),
		Problem(
			number=259,
			objective=lambda x: (
				100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 + 90 * (x[3] - x[2] ** 2) ** 2 + (1 - x[2]) ** 2 +
				10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) + 19.8 * (x[1] - 1) * (x[3] - 1)
			),
			constraints=[],
			bounds=Bounds(lb=None, ub=[None, None, None, 1]),
			initial=Initial(x0=a([0, 0, 0, 0]), f0=32.9, is_feasible=True),
			solution=Solution(xstar=a([1, 1, 1, 1]), fstar=0, rstar=0)
		),
		Problem(
			number=260,
			objective=lambda x: (
				100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 + 90 * (x[3] - x[2] ** 2) ** 2 + (1 - x[2]) ** 2 +
				10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) + 19.8 * (x[1] - 1) * (x[3] - 1)
			),
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-3, -1, -3, -1]), f0=19192, is_feasible=True),
			solution=Solution(xstar=a([1, 1, 1, 1]), fstar=0, rstar=0)
		),
		Problem(
			number=261,
			objective=lambda x: (
				(np.exp(x[0]) - x[1]) ** 4 + 100 * (x[1] - x[2]) ** 6 + np.tan(x[2] - x[3]) ** 4 +
				x[0] ** 8 + (x[3] - 1) ** 2
			),
			constraints=[],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([0, 0, 0, 0]), f0=2, is_feasible=True),
			solution=Solution(xstar=a([0, 1, 1, 1]), fstar=0, rstar=0)
		),
		Problem(
			number=262,
			objective=lambda x: -0.5 * x[0] - x[1] - 0.5 * x[2] - x[3],
			constraints=[
				Constraint(expr=lambda x: 10 - x[0] - x[1] - x[2] - x[3], sense=Constraint.GE),
				Constraint(expr=lambda x: 10 - 0.2 * x[0] - 0.5 * x[1] - x[2] - 2 * x[3], sense=Constraint.GE),
				Constraint(expr=lambda x: 10 - 2 * x[0] - x[1] - 0.5 * x[2] - 0.2 * x[3], sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] + x[1] + x[2] - 2 * x[3] - 6, sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0, 0], ub=None),
			initial=Initial(x0=a([1, 1, 1, 1]), f0=-3, is_feasible=True),
			solution=Solution(xstar=a([0, 8.667, 0, 1.333]), fstar=-10, rstar=0)
		),
		Problem(
			number=263,
			objective=lambda x: -x[0],
			constraints=[
				Constraint(expr=lambda x: x[1] - x[0] ** 3, sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] ** 2 - x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: x[1] - x[0] ** 3 - x[2] ** 2, sense=Constraint.EQ),
				Constraint(expr=lambda x: x[0] ** 2 - x[1] - x[3] ** 2, sense=Constraint.EQ)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([10, 10, 10, 10]), f0=-10, is_feasible=False),
			solution=Solution(xstar=a([1, 1, 0, 0]), fstar=-1, rstar=0)
		),
		Problem(
			name='modified Rosen-Suzuki-Problem',
			number=264,
			objective=lambda x: (
					x[0] ** 2 + x[1] ** 2 + 2 * x[2] ** 2 + x[3] ** 2
					- 5 * x[0] - 5 * x[1] - 21 * x[2] + 7 * x[3]
			),
			constraints=[
				Constraint(expr=lambda x: (
					-x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - x[3] ** 2 - x[0] - x[1] + x[2] + x[3] + 8
				), sense=Constraint.GE),
				Constraint(expr=lambda x: (
					-x[0] ** 2 - 2 * x[1] ** 2 - x[2] ** 2 - 2 * x[3] ** 2 + x[0] + x[3] + 9
				), sense=Constraint.GE),
				Constraint(expr=lambda x: (
					-2 * x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - 2 * x[0] + x[1] + x[3] + 5
				), sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([0, 0, 0, 0]), f0=0, is_feasible=True),
			solution=Solution(xstar=a([0, 1, 2, -1]), fstar=-44, rstar=0)
		),
		Problem(
			number=265,
			objective=lambda x: sum([
				1 - np.exp(-10 * x[i] * np.exp(-x[2 + i]))
				for i in range(2)
			]),
			constraints=[
				Constraint(expr=lambda x: x[0] + x[1] - 1, sense=Constraint.EQ),
				Constraint(expr=lambda x: x[2] + x[3] - 1, sense=Constraint.EQ)],
			bounds=Bounds(lb=[0, 0, 0, 0], ub=[1, 1, 1, 1]),
			initial=Initial(x0=a([0, 0, 0, 0]), f0=0, is_feasible=False),
			solution=Solution(xstar=a([1, 0, 1, 0]), fstar=1 - np.exp(-10 * np.exp(-1)), rstar=0)
		),
		# TODO: skipped 266, 267
		Problem(
			number=268,
			objective=lambda x: (
				x @ (Data.p268_D.T @ Data.p268_D) @ x
				- (2 * Data.p268_d @ Data.p268_D) @ x
				+ Data.p268_d @ Data.p268_d
			),
			constraints=[
				Constraint(expr=lambda x: -x[0] - x[1] - x[2] - x[3] - x[4] + 5, sense=Constraint.GE),
				Constraint(expr=lambda x: 10 * x[0] + 10 * x[1] - 3 * x[2] + 5 * x[3] + 4 * x[4] - 20, sense=Constraint.GE),
				Constraint(expr=lambda x: -8 * x[0] + x[1] - 2 * x[2] - 5 * x[3] + 3 * x[4] + 3 * x[4] + 40, sense=Constraint.GE),
				Constraint(expr=lambda x: 8 * x[0] - x[1] + 2 * x[2] + 5 * x[3] - 3 * x[4] - 11, sense=Constraint.GE),
				Constraint(expr=lambda x: -4 * x[0] - 2 * x[2] + 3 * x[2] - 5 * x[3] + x[4] + x[4] + 30, sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([1, 1, 1, 1, 1]), f0=12048, is_feasible=True),
			solution=Solution(xstar=a([1, 2, -1, 3, -4]), fstar=0, rstar=0)
		),
		# TODO: skipped 269
		Problem(
			number=270,
			objective=lambda x: (
				x[0] * x[1] * x[2] * x[3] - 3 * x[0] * x[1] * x[3] - 4 * x[0] * x[1] * x[2]
					+ 12 * x[0] * x[1] - x[1] * x[2] * x[3] + 3 * x[1] * x[3]
				+ 4 * x[1] * x[2] - 12 * x[1] - 2 * x[0] * x[2] * x[3] + 6 * x[0] * x[3]
					+ 8 * x[0] * x[2] - 24 * x[0] + 2 * x[2] * x[3]
				- 6 * x[3] - 8 * x[2] + 24 + 1.5 * x[4] ** 4 - 5.75 * x[4] ** 3 + 5.25 * x[4] ** 2
			),
			constraints=[
				Constraint(expr=lambda x:
					34 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - x[3] ** 2 - x[4] ** 2, sense=Constraint.GE)],
			bounds=Bounds(lb=[1, 2, 3, 4, None], ub=None),
			initial=Initial(x0=a([1.1, 2.1, 3.1, 4.1, -1]), f0=12.5001, is_feasible=True),
			solution=Solution(xstar=a([1, 2, 3, 4, 2]), fstar=-1, rstar=0)
		),
		# TODO: skipped 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283
		Problem(
			number=284,
			objective=lambda x: Data.p284_c @ x,
			constraints=[
				Constraint(expr=Data.p284_get_constraint(i), sense=Constraint.GE)
				for i in range(Data.p284_a.shape[0])
			],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([0] * 15), f0=0, is_feasible=True),
			solution=Solution(xstar=a([1] * 15), fstar=-1840, rstar=0)
		),
		Problem(
			number=285,
			objective=lambda x: (
				-486 * x[0] - 640 * x[1] - 758 * x[2] - 776 * x[3] - 477 * x[4] - 707 * x[5] - 175 * x[6]
				-619 * x[7] - 627 * x[8] - 614 * x[9] - 475 * x[10] - 377 * x[11] - 524 * x[12]
				-468 * x[13] - 529 * x[14]
			),
			constraints=[
				Constraint(expr=Data.p285_get_constraint(i), sense=Constraint.GE)
				for i in range(Data.p285_a.shape[0])],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([0] * 15), f0=0, is_feasible=True),
			solution=Solution(xstar=a([1] * 15), fstar=-8252, rstar=0)
		),
		# TODO: skipped 286, 287, 288, 289, 290, 291
		############################################################################
		## Book 2
		############################################################################
		# TODO: skipped 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 307, 308, 309, 310, 311,
		#  312, 313, 314
		Problem(
			number=315,
			objective=lambda x: -x[1],
			constraints=[
				Constraint(expr=lambda x: 1 - 2 * x[1] + x[0], sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] ** 2 + x[1] ** 2, sense=Constraint.GE),
				Constraint(expr=lambda x: 1 - x[0] ** 2 - x[1] ** 2, sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([-0.1, -0.9]), f0=0.9, is_feasible=True),
			solution=Solution(xstar=a([0.6, 0.8]), fstar=-0.8, rstar=0.8640e-11)
		),
		# TODO: skipped 316, 317, 318, 319, 320, 321, 322
		Problem(
			number=323,
			objective=lambda x: x[0] ** 2 + x[1] ** 2 - 4 * x[0] + 4,
			constraints=[
				Constraint(expr=lambda x: x[0] - x[1] + 2, sense=Constraint.GE),
				Constraint(expr=lambda x: -x[0] ** 2 + x[1] - 1, sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0], ub=None),
			initial=Initial(x0=a([0, 1]), f0=5, is_feasible=True),
			solution=Solution(xstar=a([0.5536, 1.306]), fstar=3.79894, rstar=0.3130e-8)
		),
		############################################################################
		## Book 2  (Odd pages)
		############################################################################
		# TODO: skipped 324
		Problem(
			number=326,
			objective=lambda x: x[0] ** 2 + x[1] ** 2 - 16 * x[0] - 10 * x[1],
			constraints=[
				Constraint(expr=lambda x: 11 - x[0] ** 2 + 6 * x[0] - 4 * x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] * x[1] - 3 * x[1] - np.exp(x[0] - 3) + 1, sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0], ub=None),
			initial=Initial(x0=a([4, 3]), f0=-69, is_feasible=True),
			solution=Solution(xstar=a([5.240, 3.746]), fstar=-79.8078, rstar=0.3980e-11)
		),
		# TODO: skipped 328, 330
		#######################################################################################
		# Problem 332 is supposed to be feasible...
		#######################################################################################
		Problem(
			number=332,
			objective=lambda x: np.pi / 3.6 * sum([
				(np.log(ti) + x[1] * np.sin(ti) + x[0] * np.cos(ti)) ** 2 +
				(np.log(ti) + x[1] * np.cos(ti) + x[0] * np.sin(ti)) ** 2
				for ti in Data.p332_tis
			]),
			constraints=[
				Constraint(expr=lambda x: 30 - Data.p332_pmax(x), sense=Constraint.GE),
				Constraint(expr=lambda x: Data.p332_pmax(x) - 30, sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0], ub=[1.5, 1.5]),
			initial=Initial(x0=a([0.75, 0.75]), f0=217.361, is_feasible=False),
			solution=Solution(xstar=a([0.9114, 0.02928]), fstar=114.95, rstar=0.1530e-7)
		),
		# TODO: skipped 334, 336, 338, 340
		Problem(
			number=342,
			objective=lambda x: -x[0] * x[1] * x[2],
			constraints=[
				Constraint(expr=lambda x: 48 - x[0] ** 2 - 2 * x[1] ** 2 - 4 * x[2] ** 2, sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0], ub=None),
			initial=Initial(x0=a([100, 100, 100]), f0=-1, is_feasible=True),
			solution=Solution(xstar=a([4, 2.828, 2]), fstar=-22.6274, rstar=0.175e-7)
		),
		# TODO: skipped 344, 346, 348, 351
		Problem(
			number=353,
			objective=lambda x: -(24.55 * x[0] + 26.75 * x[1] + 39 * x[2] + 40.5 * x[3]),
			constraints=[
				Constraint(expr=lambda x: 2.3 * x[0] + 5.6 * x[1] + 11.1 * x[2] + 1.3 * x[2] - 5, sense=Constraint.GE),
				Constraint(expr=lambda x: 12 * x[0] + 11.9 * x[1] + 41.8 * x[2] + 52.1 * x[3]
										  - 1.645 * np.sqrt(Data.p353_q(x)) - 12, sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] + x[1] + x[2] + x[3] - 1, sense=Constraint.EQ)],
			bounds=Bounds(lb=[0, 0, 0, 0], ub=None),
			initial=Initial(x0=a([0, 0, 0.4, 0.6]), f0=-39.9, is_feasible=True),
			solution=Solution(xstar=a([0, 0, 0.3776, 0.6224]), fstar=-39.9337, rstar=0)
		),
		# TODO: skipped 355, 357
		Problem(
			number=359,
			objective=lambda x: -(Data.p359_a @ x - 24345),
			constraints=[
				Constraint(expr=lambda x: 2.4 * x[0] - x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: -1.2 * x[0] + x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: 60 * x[1] - x[2], sense=Constraint.GE),
				Constraint(expr=lambda x: -20 * x[0] + x[2], sense=Constraint.GE),
				Constraint(expr=lambda x: 9.3 * x[0] - x[3], sense=Constraint.GE),
				Constraint(expr=lambda x: -9.0 * x[0] + x[3], sense=Constraint.GE),
				Constraint(expr=lambda x: 7 * x[0] - x[4], sense=Constraint.GE),
				Constraint(expr=lambda x: -6.5 * x[0] + x[4], sense=Constraint.GE),
				Constraint(expr=lambda x: Data.p359_b @ x, sense=Constraint.GE),
				Constraint(expr=lambda x: Data.p359_c @ x, sense=Constraint.GE),
				Constraint(expr=lambda x: Data.p359_d @ x, sense=Constraint.GE),
				Constraint(expr=lambda x: 294000 - Data.p359_b @ x, sense=Constraint.GE),
				Constraint(expr=lambda x: 294000 - Data.p359_c @ x, sense=Constraint.GE),
				Constraint(expr=lambda x: 294000 - Data.p359_d @ x, sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([2.52, 5.04, 94.5, 23.31, 17.14]), f0=-0.235124e7, is_feasible=True),
			solution=Solution(xstar=a([4.3537, 10.89, 272.2, 42.2, 31.76]), fstar=-0.528042e7, rstar=0)
		),

		# TODO: skipped 361,
		# 364, 356,  on p 49
		# TODO: skipped 368, 370, 372, 374
		# 376
		# TODO: skipped 378, 380
		# 382, 384, 386, 388 on page 58
		# TODO: skipped 391
		# 393
		# TODO: skipped 395

		# First publication:
		# TODO: skipped 2, 3, 4, 5, 6, 7, 8
		Problem(
			number=1,
			objective=lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
			constraints=[],
			bounds=Bounds(lb=[None, -1.5], ub=None),
			initial=Initial(x0=a([-2, 1]), f0=909, is_feasible=True),
			solution=Solution(xstar=a([1, 1]), fstar=0, rstar=0)
		),
		# Problem(
		# 	number=9,
		# 	objective=lambda x: np.sin(np.pi * x[0] / 12) * np.cos(np.pi * x[1] / 16),
		# 	constraints=[
		# 		Constraint(expr=lambda x: 4 * x[0] - 3 * x[1], sense=Constraint.EQ)],
		# 	bounds=Bounds(lb=None, ub=None),
		# 	initial=Initial(x0=a([0, 0]), f0=0, is_feasible=True),
		# 	solution=Solution(xstar=a([0, 0, 0, 0]), fstar=0, rstar=0)
		# ),

		Problem(
			number=12,
			objective=lambda x: 0.5 * x[0] ** 2 + x[1] ** 2 - x[0] * x[1] - 7 * x[0] - 7 * x[1],
			constraints=[
				Constraint(expr=lambda x: 25 - 4 * x[0] ** 2 - x[1] ** 2, sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([0, 0]), f0=0, is_feasible=True),
			solution=Solution(xstar=a([2, 3]), fstar=-30, rstar=0)
		),
		Problem(
			number=24,
			objective=lambda x: 1 / (27 * np.sqrt(3)) * ((x[0] - 3) ** 2 - 9) * x[1] ** 3,
			constraints=[
				Constraint(expr=lambda x: x[0] / np.sqrt(3) - x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] + np.sqrt(3) * x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: -x[0] - np.sqrt(3) * x[1] + 6, sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0], ub=None),
			initial=Initial(x0=a([1, 0.5]), f0=-0.01336459, is_feasible=True),
			solution=Solution(xstar=a([3, np.sqrt(3)]), fstar=-1, rstar=0)
		),
		# TODO: Skipped 25, 26, 27, 28
		Problem(
			number=29,
			objective=lambda x: -x[0] * x[1] * x[2],
			constraints=[
				Constraint(expr=lambda x: -x[0] ** 2 - 2 * x[1] ** 2 - 4 * x[2] ** 2 + 48, sense=Constraint.GE),],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([1, 1, 1]), f0=-1, is_feasible=True),
			solution=Solution(xstar=a([4, 2 * np.sqrt(2), 2]), fstar=-16 * np.sqrt(2), rstar=0)  # One of many...
			# (a, b, c), (a, -b, -c), (-a, b, -c), (-a, -b, c)
		),
		Problem(
			number=30,
			objective=lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2,
			constraints=[
				Constraint(expr=lambda x: x[0] ** 2 + x[1] ** 2 - 1, sense=Constraint.GE)],
			bounds=Bounds(lb=[1, -10, -10], ub=[10, 10, 10]),
			initial=Initial(x0=a([1, 1, 1]), f0=3, is_feasible=True),
			solution=Solution(xstar=a([1, 0, 0]), fstar=1, rstar=0)
		),
		Problem(
			number=31,
			objective=lambda x: 9 * x[0] ** 2 + x[1] ** 2 + 9 * x[2] ** 2,
			constraints=[
				Constraint(expr=lambda x: x[0] * x[1] - 1, sense=Constraint.GE)],
			bounds=Bounds(lb=[-10, 1, -10], ub=[10, 10, 1]),
			initial=Initial(x0=a([1, 1, 1]), f0=19, is_feasible=True),
			solution=Solution(xstar=a([1/np.sqrt(3), np.sqrt(3), 0]), fstar=6, rstar=0)
		),
		# TODO: Skipped 32
		Problem(
			number=33,
			objective=lambda x: (x[0] - 1) * (x[0] - 2) * (x[0] - 3) + x[2],
			constraints=[
				Constraint(expr=lambda x: x[2] ** 2 - x[1] ** 2 - x[0] ** 2, sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 4, sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0], ub=[None, None, 5]),
			initial=Initial(x0=a([0, 0, 3]), f0=-3, is_feasible=True),
			solution=Solution(xstar=a([0, np.sqrt(2), np.sqrt(2)]), fstar=np.sqrt(2) - 6, rstar=0)
		),
		Problem(
			number=34,
			objective=lambda x: -x[0],
			constraints=[
				Constraint(expr=lambda x: x[1] - np.exp(x[0]), sense=Constraint.GE),
				Constraint(expr=lambda x: x[2] - np.exp(x[1]), sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0], ub=[100, 100, 10]),
			initial=Initial(x0=a([0, 1.05, 2.9]), f0=0, is_feasible=True),
			solution=Solution(xstar=a([np.log(np.log(10)), np.log(10), 10]), fstar=-np.log(np.log(10)), rstar=0)
		),
		Problem(
			number=35,
			objective=lambda x: 9 - 8 * x[0] - 6 * x[1] - 4 * x[2] + 2 * x[0] ** 2 + 2 * x[1] ** 2 + x[2] ** 2
								+ 2 * x[0] * x[1] + 2 * x[0] * x[2],
			constraints=[
				Constraint(expr=lambda x: 3 - x[0] - x[1] - 2 * x[2], sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0], ub=None),
			initial=Initial(x0=a([0.5, 0.5, 0.5]), f0=2.25, is_feasible=True),
			solution=Solution(xstar=a([4/3, 7/9, 4/9]), fstar=1/9, rstar=0)
		),
		Problem(
			number=36,
			objective=lambda x: -x[0] * x[1] * x[2],
			constraints=[
				Constraint(expr=lambda x: 72 - x[0] - 2 * x[1] - 2 * x[2], sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0], ub=[20, 11, 42]),
			initial=Initial(x0=a([10, 10, 10]), f0=-1000, is_feasible=True),
			solution=Solution(xstar=a([20, 11, 15]), fstar=-3300, rstar=0)
		),
		Problem(
			number=37,
			objective=lambda x: -x[0] * x[1] * x[2],
			constraints=[
				Constraint(expr=lambda x: 72 - x[0] - 2 * x[1] - 2 * x[2], sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] + 2 * x[1] + 2 * x[2], sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0], ub=[42, 42, 42]),
			initial=Initial(x0=a([10, 10, 10]), f0=-1000, is_feasible=True),
			solution=Solution(xstar=a([24, 12, 12]), fstar=-3456, rstar=0)
		),
		# TODO: Skipped 38, 39, 40, 41, 42
		Problem(
			number=43,
			objective=lambda x: x[0] ** 2 + x[1] ** 2 + 2 * x[2] ** 2 + x[3] ** 2 - 5 * x[0] - 5 * x[1] - 21 * x[2] +
				+ 7 * x[3],
			constraints=[
				Constraint(expr=lambda x: 8 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - x[3] ** 2 - x[0] + x[1] - x[2] + x[3], sense=Constraint.GE),
				Constraint(expr=lambda x: 10 - x[0] ** 2- 2 * x[1] ** 2 - x[2] ** 2 - 2 * x[3] ** 2 + x[0] + x[3], sense=Constraint.GE),
				Constraint(expr=lambda x: 5 - 2 * x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - 2 * x[0] + x[1] + x[3], sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([0, 0, 0, 0]), f0=0, is_feasible=True),
			solution=Solution(xstar=a([0, 1, 2, -1]), fstar=-44, rstar=0)
		),
		Problem(
			number=44,
			objective=lambda x: x[0] - x[1] - x[2] - x[0] * x[2] + x[0] * x[2] + x[1] * x[2] - x[1] * x[3],
			constraints=[
				Constraint(expr=lambda x: 8 - x[0] - 2 * x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: 12 - 4 * x[0] - x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: 12 - 3 * x[0] - 4 * x[1], sense=Constraint.GE),
				Constraint(expr=lambda x: 8 - 2 * x[2] - x[3], sense=Constraint.GE),
				Constraint(expr=lambda x: 8 - x[2] - 2 * x[3], sense=Constraint.GE),
				Constraint(expr=lambda x: 5 - x[2] - x[3], sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0, 0], ub=None),
			initial=Initial(x0=a([0, 0, 0, 0]), f0=0, is_feasible=True),
			solution=Solution(xstar=a([0, 3, 0, 4]), fstar=-15, rstar=0)
		),
		# TODO: Skipped 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56
		Problem(
			number=57,
			objective=lambda x: sum([Data.p57_fi(x, i) ** 2 for i in range(44)]),
			constraints=[
				Constraint(expr=lambda x: 0.49 * x[1] - x[0] * x[1] - 0.09, sense=Constraint.GE)],
			bounds=Bounds(lb=[0.4, -4], ub=None),
			initial=Initial(x0=a([.42, 5]), f0=0.0307098602, is_feasible=True),
			solution=Solution(xstar=a([.419952675, 1.284845629]), fstar=.02845966972, rstar=0)
		),
		# Where did problem 58 go??
		Problem(
			number=59,
			objective=lambda x: -75.196 + 3.8112 * x[0] + 0.0020567 * x[0] ** 3 - 1.0345e-5 * x[0] ** 4
				+ 6.8306 * x[1] - 0.030234 * x[0] * x[1] + 1.28134e-3 * x[1] * x[0] ** 2
				+ 2.266e-7 * x[0] ** 4 * x[1] - 0.25645 * x[1] ** 2 + 0.0034604 * x[1] ** 3 - 1.3514e-5 * x[1] ** 4
				+ 28.106 / (x[1] + 1) + 5.2375e-6 * x[0] ** 2 * x[1] ** 2 + 6.3e-8 * x[0] ** 3 * x[1] ** 2
				- 7e-10 * x[0] ** 3 * x[1] ** 3 - 3.405e-4 * x[0] * x[1] ** 2 + 1.6638e-6 * x[0] * x[1] ** 3
				+ 2.8673 * np.exp(0.0005 * x[0] * x[1]) - 3.5256e-5 * x[0] ** 3 * x[1],
			constraints=[
				Constraint(expr=lambda x: x[0] * x[1] - 700, sense=Constraint.GE),
				Constraint(expr=lambda x: x[1] - x[0] ** 2 / 125, sense=Constraint.GE),
				Constraint(expr=lambda x: (x[1] - 50) ** 2 - 5 * (x[0] - 55), sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0], ub=[75, 65]),
			initial=Initial(x0=a([90, 10]), f0=86.878639, is_feasible=False),
			solution=Solution(xstar=a([13.55010424, 51.66018129]), fstar=-7.804226324, rstar=0)
		),
		Problem(
			number=66,
			objective=lambda x: 0.2 * x[2] - 0.8 * x[0],
			constraints=[
				Constraint(expr=lambda x: x[1] - np.exp(x[0]), sense=Constraint.GE),
				Constraint(expr=lambda x: x[2] - np.exp(x[1]), sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0], ub=[100, 100, 10]),
			initial=Initial(x0=a([0, 1.05, 2.9]), f0=0.58, is_feasible=True),
			solution=Solution(xstar=a([0.1841264879, 1.202167843, 3.327322322]), fstar=0.5181632741, rstar=0.58e-10)
		),
		Problem(
			number=67,
			objective=Data.p67_obj,
			constraints=
				[Constraint(expr=Data.p67_c1(i), sense=Constraint.GE) for i in range(1, 8)] +
				[Constraint(expr=Data.p67_c2(i), sense=Constraint.GE) for i in range(8, 15)],
			bounds=Bounds(lb=[1e-5, 1e-5, 1e-5], ub=[2e3, 1.6e4, 1.2e2]),
			initial=Initial(x0=a([1745, 12000, 110]), f0=868.6458, is_feasible=True),
			solution=Solution(xstar=a([1728.371286, 16000, 98.14151402]), fstar=-1162.36507, rstar=0)
		),
		# TODO: Skipped 70, not really any reason to, though
		Problem(
			number=71,
			objective=lambda x: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
			constraints=[
				Constraint(expr=lambda x: x[0] * x[1] * x[2] * x[3] - 25, sense=Constraint.GE),
				Constraint(expr=lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 - 40, sense=Constraint.EQ)],
			bounds=Bounds(lb=[1, 1, 1, 1], ub=[5, 5, 5, 5]),
			initial=Initial(x0=a([1, 5, 5, 1]), f0=16, is_feasible=True),
			solution=Solution(xstar=a([1, 4.7429994, 3.8211503, 1.3794082]), fstar=17.0140173, rstar=0)
		),
		Problem(
			number=76,
			objective=lambda x: x[0] ** 2 + 0.5 * x[1] ** 2 + x[2] ** 2 + 0.5 * x[3] ** 2 - x[0] * x[2] + x[2] * x[3]
				- x[0] - 3 * x[1] + x[2] - x[3],
			constraints=[
				Constraint(expr=lambda x: 5 - x[0] - 2 * x[1] - x[2] - x[3], sense=Constraint.GE),
				Constraint(expr=lambda x: 4 - 3 * x[0] - x[1] - 2 * x[2] + x[3], sense=Constraint.GE),
				Constraint(expr=lambda x: x[1] + 4 * x[2] - 1.5, sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0, 0], ub=None),
			initial=Initial(x0=a([0.5, 0.5, 0.5, 0.5]), f0=-1.25, is_feasible=True),
			solution=Solution(xstar=a([0.2727273, 0.2090909, 0.26e-10, .5454545]), fstar=-4.681818181, rstar=0)
		),
		Problem(
			number=84,
			objective=lambda x: -Data.p84_a[0] - Data.p84_a[1] * x[0] - Data.p84_a[2] * x[0] * x[1]
				- Data.p84_a[3] * x[0] * x[2] - Data.p84_a[4] * x[0] * x[3] - Data.p84_a[5] * x[0] * x[4],
			constraints=[
				Constraint(expr=lambda x: 294000 - Data.p84_c1(x), sense=Constraint.GE),
				Constraint(expr=lambda x: Data.p84_c1(x), sense=Constraint.GE),
				Constraint(expr=lambda x: 294000 - Data.p84_c2(x), sense=Constraint.GE),
				Constraint(expr=lambda x: Data.p84_c2(x), sense=Constraint.GE),
				Constraint(expr=lambda x: 277200 - Data.p84_c3(x), sense=Constraint.GE),
				Constraint(expr=lambda x: Data.p84_c3(x), sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 1.2, 20, 9, 6.5], ub=[1000, 2.4, 60, 9.3, 7]),
			initial=Initial(x0=a([2.52, 2, 37.5, 9.25, 6.8]), f0=-2351243.5, is_feasible=True),
			solution=Solution(xstar=a([4.53743097, 2.4, 60, 9.3, 7]), fstar=-5290335.133, rstar=0)
		),
		# Skipped problem # 85 (only that it is code...)
		Problem(
			number=86,
			objective=lambda x: (
				sum([Data.p86_e[j] * x[j] for j in range(5)]) +
				sum([Data.p86_c[i, j] * x[i] * x[j] for i in range(5) for j in range(5)]) +
				sum([Data.p86_d[j] * x[j] ** 3 for j in range(5)])
			),
			constraints=[
				Constraint(expr=lambda x: sum([Data.p86_a[0, j] * x[j] - Data.p86_b[0] for j in range(5)]), sense=Constraint.GE),
				Constraint(expr=lambda x: sum([Data.p86_a[1, j] * x[j] - Data.p86_b[1] for j in range(5)]), sense=Constraint.GE),
				Constraint(expr=lambda x: sum([Data.p86_a[2, j] * x[j] - Data.p86_b[2] for j in range(5)]), sense=Constraint.GE),
				Constraint(expr=lambda x: sum([Data.p86_a[3, j] * x[j] - Data.p86_b[3] for j in range(5)]), sense=Constraint.GE),
				Constraint(expr=lambda x: sum([Data.p86_a[4, j] * x[j] - Data.p86_b[4] for j in range(5)]), sense=Constraint.GE),
				Constraint(expr=lambda x: sum([Data.p86_a[5, j] * x[j] - Data.p86_b[5] for j in range(5)]), sense=Constraint.GE),
				Constraint(expr=lambda x: sum([Data.p86_a[6, j] * x[j] - Data.p86_b[6] for j in range(5)]), sense=Constraint.GE),
				Constraint(expr=lambda x: sum([Data.p86_a[7, j] * x[j] - Data.p86_b[7] for j in range(5)]), sense=Constraint.GE),
				Constraint(expr=lambda x: sum([Data.p86_a[8, j] * x[j] - Data.p86_b[8] for j in range(5)]), sense=Constraint.GE),
				Constraint(expr=lambda x: sum([Data.p86_a[9, j] * x[j] - Data.p86_b[9] for j in range(5)]), sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0, 0, 0], ub=None),
			initial=Initial(x0=a([0, 0, 0, 0, 1]), f0=20, is_feasible=True),
			solution=Solution(xstar=a([0.3, 0.33346761, 0.4, 0.42831010, 0.22396487]), fstar=-32.34867897, rstar=0)
		),
		Problem(
			number=93,
			objective=lambda x: 0.0204 * x[0] * x[3] * (x[0] + x[1] + x[2]) + 0.0187 * x[1] * x[2] * (x[0] + 1.57 * x[1] + x[3])
				+ 0.0607 * x[0] * x[3] * x[4] ** 2 * (x[0] + x[1] + x[2])
				+ 0.0437 * x[1] * x[2] * x[5] ** 2 * (x[0] + 1.57 * x[1] + x[3]),
			constraints=[
				Constraint(expr=lambda x: 0.001 * x[0] * x[1] * x[2] * x[3] * x[4] * x[5] - 2.07, sense=Constraint.GE),
				Constraint(expr=lambda x: 1 - 0.00062 * x[0] * x[3] * x[4] ** 2 * (x[0] + x[1] + x[2])
					- 0.00058 * x[1] * x[2] * x[5] ** 2 * (x[0] + 1.57 * x[1] + x[3]), sense=Constraint.GE)],
			bounds=Bounds(lb=[0, 0, 0, 0, 0, 0], ub=None),
			initial=Initial(x0=a([5.54, 4.4, 12.02, 11.82, 0.702, 0.852]), f0=137.066, is_feasible=True),
			solution=Solution(xstar=a([5.332666, 4.656744, 10.43299, 12.08230, 0.7526074, 0.87865284]), fstar=135.075961, rstar=0)
		),
		Problem(
			number=100,
			objective=lambda x: (x[0] - 10) ** 2 + 5 * (x[1] - 12) ** 2 + x[2] ** 4 + 3 * (x[3] - 11) ** 2
				+ 10 * x[4] ** 6 + 7 * x[5] ** 2 + x[6] ** 4 - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6],
			constraints=[
				Constraint(expr=lambda x: 127 - 2 * x[0] ** 2 - 3 * x[1] ** 3 - x[2] - 4 * x[3] ** 2 - 5 * x[4], sense=Constraint.GE),
				Constraint(expr=lambda x: 282 - 7 * x[0] - 3 * x[1] - 10 * x[2] ** 2 - x[3] + x[4], sense=Constraint.GE),
				Constraint(expr=lambda x: 196 - 23 * x[0] - x[1] ** 2 - 6 * x[5] ** 2 + 8 * x[6], sense=Constraint.GE),
				Constraint(expr=lambda x: -4 * x[0] ** 2 - x[1] ** 2 + 3 * x[0] * x[1] - 2 * x[2] ** 2 - 5 * x[5] + 11 * x[6] + 11 * x[6], sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([1, 2, 0, 4, 0, 1, 1]), f0=714, is_feasible=True),
			solution=Solution(xstar=a([2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227]), fstar=680.6300573, rstar=0)
		),
		Problem(
			number=105,
			objective=lambda x: -sum([
				np.log((Data.p105_a(i, x) + Data.p105_b(i, x) + Data.p105_c(i, x)) / np.sqrt(2 * np.pi))
				for i in range(235)
			]),
			constraints=[
				Constraint(expr=lambda x: 1 - x[0] - x[1], sense=Constraint.GE)],
			bounds=Bounds(lb=[0.001, 0.001, 100, 130, 170, 5, 5, 5], ub=[0.499, 0.499, 180, 210, 240, 25, 25, 25]),
			initial=Initial(x0=a([.1, .2, 100, 125, 175, 11.2, 13.2, 15.8]), f0=1297.6693, is_feasible=True),
			solution=Solution(xstar=a([0.4128928, 0.4033526, 131.2613, 164.3135,
				217.4222, 12.28018, 15.77170, 20.74682]), fstar=1138.416240, rstar=0)
		),
		Problem(
			number=110,
			# Not too sure where the closing bracket is...
			objective=lambda x: sum([
				np.log(x[i] - 2) ** 2 + np.log(10 - x[i]) ** 2
				for i in range(10)
			]) - np.prod([
				x[i]
				for i in range(10)
			]) ** 0.2,
			constraints=[],
			bounds=Bounds(lb=[2.001] * 10, ub=[9.999] * 10),
			initial=Initial(x0=a([9] * 10), f0=-43.134337, is_feasible=True),
			solution=Solution(xstar=a([9.35025655] * 10), fstar=-45.77846971, rstar=0)
		),
		Problem(
			number=113,
			objective=lambda x: x[0] ** 2 + x[1] ** 2 + x[0] * x[1] - 14 * x[0] - 16 * x[1] + (x[2] - 10) ** 2
				+ 4 * (x[3] - 5) ** 2 + (x[4] - 3) ** 2 + 2 * (x[5] - 1) ** 2 + 5 * x[6] ** 2
				+ 7 * (x[7] - 11) ** 2 + 2 * (x[8] - 10) ** 2 + (x[9] - 7) ** 2 + 45,
			constraints=[
				Constraint(expr=lambda x: 105 - 4 * x[0] - 5 * x[1] + 3 * x[6] - 9 * x[7], sense=Constraint.GE),
				Constraint(expr=lambda x: -10 * x[0] + 8 * x[1] + 17 * x[6] - 2 * x[7], sense=Constraint.GE),
				Constraint(expr=lambda x: 8 * x[0] - 2 * x[1] - 5 * x[8] + 2 * x[9] + 12, sense=Constraint.GE),
				Constraint(expr=lambda x: -3 * (x[0] - 2) ** 2 - 4 * (x[1] - 3) ** 2 - 2 * x[2] ** 2 + 7 * x[3] + 120, sense=Constraint.GE),
				Constraint(expr=lambda x: -5 * x[0] ** 2 - 8 * x[1] - (x[2] - 6) ** 2 + 2 * x[3] + 40, sense=Constraint.GE),
				Constraint(expr=lambda x: -0.5 * (x[0] - 8) ** 2 - 2 * (x[1] - 4) ** 2 - 3 * x[4] ** 2 + x[5] + 30, sense=Constraint.GE),
				Constraint(expr=lambda x: -x[0] - 2 * (x[1] - 2) ** 2 + 2 * x[0] * x[1] - 14 * x[4] + 6 * x[5], sense=Constraint.GE),
				Constraint(expr=lambda x: 3 * x[0] - 6 * x[1] - 12 * (x[8] - 8) ** 2 + 7 * x[9], sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([2, 3, 5, 5, 1, 2, 7, 3, 6, 10]), f0=753, is_feasible=True),
			solution=Solution(
				xstar=a([
					2.171996, 2.363683, 8.773926, 5.095984,
					0.9906548, 1.430574, 1.321644, 9.828426,
					8.280092, 8.375927]), fstar=24.3062091, rstar=12e-8)
		),
		Problem(
			number=117,
			objective=lambda x: (
				sum([Data.p86_b[j] * x[j] for j in range(10)]) +
				sum([Data.p86_c[j, k] * x[10 + j] * x[10 + k] for j in range(5) for k in range(5)]) +
				sum([Data.p86_d[j] * x[10 + j] ** 3 for j in range(5)])
			),
			constraints=[
				Constraint(expr=lambda x: 2 * sum([Data.p86_c[k, 0] * x[10 + k] for k in range(5)]) + 3 * Data.p86_d[0] * x[10 + 0] ** 2 + Data.p86_e[0] - sum([Data.p86_a[k, 0] * x[k] ** 3 for k in range(5)]), sense=Constraint.GE),
				Constraint(expr=lambda x: 2 * sum([Data.p86_c[k, 1] * x[10 + k] for k in range(5)]) + 3 * Data.p86_d[1] * x[10 + 1] ** 2 + Data.p86_e[1] - sum([Data.p86_a[k, 1] * x[k] ** 3 for k in range(5)]), sense=Constraint.GE),
				Constraint(expr=lambda x: 2 * sum([Data.p86_c[k, 2] * x[10 + k] for k in range(5)]) + 3 * Data.p86_d[2] * x[10 + 2] ** 2 + Data.p86_e[2] - sum([Data.p86_a[k, 2] * x[k] ** 3 for k in range(5)]), sense=Constraint.GE),
				Constraint(expr=lambda x: 2 * sum([Data.p86_c[k, 3] * x[10 + k] for k in range(5)]) + 3 * Data.p86_d[3] * x[10 + 3] ** 2 + Data.p86_e[3] - sum([Data.p86_a[k, 3] * x[k] ** 3 for k in range(5)]), sense=Constraint.GE),
				Constraint(expr=lambda x: 2 * sum([Data.p86_c[k, 4] * x[10 + k] for k in range(5)]) + 3 * Data.p86_d[4] * x[10 + 4] ** 2 + Data.p86_e[4] - sum([Data.p86_a[k, 4] * x[k] ** 3 for k in range(5)]), sense=Constraint.GE),],
			bounds=Bounds(lb=[0] * 15, ub=None),
			initial=Initial(x0=0.001 * a([1] * 6 + [60000] + [1] * 8), f0=2400.1053, is_feasible=True),
			solution=Solution(xstar=
				a([0, 0, 5.174136, 0, 3.061093, 11.83968, 0, 0,
					0.1039071, 0, 0.2999929, 0.3334709, 0.3999910,
					0.4283145, 0.2239607]), fstar=32.348679, rstar=0)
		),
		Problem(
			number=118,
			objective=lambda x: sum([
				2.3 * x[3 * k] + 0.0001 * x[3 * k] ** 2 + 1.7 * x[3 * k + 1] + 0.0001 * x[3 * k + 1] ** 2
				+ 2.2 * x[3 * k + 2] + 0.00015 * x[3 * k + 2] ** 2
				for k in range(5)
			]),
			constraints=[
				Constraint(expr=lambda x:       x[3 * 1] - x[3 * 1 - 3] + 7, sense=Constraint.GE),
				Constraint(expr=lambda x: 13 - (x[3 * 1] - x[3 * 1 - 3] + 7), sense=Constraint.GE),
				Constraint(expr=lambda x:       x[3 * 2] - x[3 * 2 - 3] + 7, sense=Constraint.GE),
				Constraint(expr=lambda x: 13 - (x[3 * 2] - x[3 * 2 - 3] + 7), sense=Constraint.GE),
				Constraint(expr=lambda x:       x[3 * 3] - x[3 * 3 - 3] + 7, sense=Constraint.GE),
				Constraint(expr=lambda x: 13 - (x[3 * 3] - x[3 * 3 - 3] + 7), sense=Constraint.GE),
				Constraint(expr=lambda x:       x[3 * 4] - x[3 * 4 - 3] + 7, sense=Constraint.GE),
				Constraint(expr=lambda x: 13 - (x[3 * 4] - x[3 * 4 - 3] + 7), sense=Constraint.GE),

				Constraint(expr=lambda x:       x[3 * 1 + 2] - x[3 * 1 - 1] + 7, sense=Constraint.GE),
				Constraint(expr=lambda x: 13 - (x[3 * 1 + 2] - x[3 * 1 - 1] + 7), sense=Constraint.GE),
				Constraint(expr=lambda x:       x[3 * 2 + 2] - x[3 * 2 - 1] + 7, sense=Constraint.GE),
				Constraint(expr=lambda x: 13 - (x[3 * 2 + 2] - x[3 * 2 - 1] + 7), sense=Constraint.GE),
				Constraint(expr=lambda x:       x[3 * 3 + 2] - x[3 * 3 - 1] + 7, sense=Constraint.GE),
				Constraint(expr=lambda x: 13 - (x[3 * 3 + 2] - x[3 * 3 - 1] + 7), sense=Constraint.GE),
				Constraint(expr=lambda x:       x[3 * 4 + 2] - x[3 * 4 - 1] + 7, sense=Constraint.GE),
				Constraint(expr=lambda x: 13 - (x[3 * 4 + 2] - x[3 * 4 - 1] + 7), sense=Constraint.GE),

				Constraint(expr=lambda x:       x[3 * 1 + 1] - x[3 * 1 - 2] + 7, sense=Constraint.GE),
				Constraint(expr=lambda x: 14 - (x[3 * 1 + 1] - x[3 * 1 - 2] + 7), sense=Constraint.GE),
				Constraint(expr=lambda x:       x[3 * 2 + 1] - x[3 * 2 - 2] + 7, sense=Constraint.GE),
				Constraint(expr=lambda x: 14 - (x[3 * 2 + 1] - x[3 * 2 - 2] + 7), sense=Constraint.GE),
				Constraint(expr=lambda x:       x[3 * 3 + 1] - x[3 * 3 - 2] + 7, sense=Constraint.GE),
				Constraint(expr=lambda x: 14 - (x[3 * 3 + 1] - x[3 * 3 - 2] + 7), sense=Constraint.GE),
				Constraint(expr=lambda x:       x[3 * 4 + 1] - x[3 * 4 - 2] + 7, sense=Constraint.GE),
				Constraint(expr=lambda x: 14 - (x[3 * 4 + 1] - x[3 * 4 - 2] + 7), sense=Constraint.GE),

				Constraint(expr=lambda x: x[ 0] + x[ 1] + x[ 2] -  60, sense=Constraint.GE),
				Constraint(expr=lambda x: x[ 3] + x[ 4] + x[ 5] -  50, sense=Constraint.GE),
				Constraint(expr=lambda x: x[ 6] + x[ 7] + x[ 8] -  70, sense=Constraint.GE),
				Constraint(expr=lambda x: x[ 9] + x[10] + x[11] -  85, sense=Constraint.GE),
				Constraint(expr=lambda x: x[12] + x[13] + x[14] - 100, sense=Constraint.GE)],
			bounds=Bounds(lb=[8, 43, 3] + [0, 0, 0] * 4, ub=[21, 57, 16] + [90, 120, 60] * 4),
			initial=Initial(x0=a([20, 55, 15, 20, 60, 20, 20, 60, 20, 20, 60, 20, 20, 60, 20]), f0=664.82045, is_feasible=True),
			solution=Solution(xstar=a([8, 49, 3, 1, 56, 0, 1, 63, 6, 3, 70, 12, 5, 77, 18]), fstar=664.82045, rstar=0)
			# Not able to decrease? Why change x?
		),
	]
	'''
		Problem(
			number=268,
			objective=lambda x: ,
			constraints=[
				Constraint(expr=lambda x: , sense=Constraint.GE),
				Constraint(expr=lambda x: , sense=Constraint.GE),
				Constraint(expr=lambda x: , sense=Constraint.GE),
				Constraint(expr=lambda x: , sense=Constraint.GE)],
			bounds=Bounds(lb=None, ub=None),
			initial=Initial(x0=a([0, 0, 0, 0]), f0=0, is_feasible=True),
			solution=Solution(xstar=a([0, 0, 0, 0]), fstar=0, rstar=0)
		),
		'''


if __name__ == '__main__':
	for problem in HottSchittowski.PROBLEMS:
		print('problem', problem.number)
		assert problem.n == len(problem.initial.x0)
		assert problem.n == len(problem.solution.xstar)
		init_error = abs(problem.objective.expr(problem.initial.x0) - problem.initial.f0)
		sol_error = abs(problem.objective.expr(problem.solution.xstar) - problem.solution.fstar)
		print('\terror at starting point', init_error)
		print('\terror at solution', sol_error)
		# check is feasible...
		#
