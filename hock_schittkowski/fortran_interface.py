import schittkowski_library as schit
import numpy as np

'''

SCHIT_SRC_LOCATION='/work/research/schittowski_library/2011_version/unzipped'
python -m numpy.f2py -c $SCHIT_SRC_LOCATION/PROB.FOR $SCHIT_SRC_LOCATION/CONV.FOR  -m schittkowski_library

'''

'''
schit.l4.gf
schit.l5.gg

schit.l9.index1
schit.l10.index2
'''



'''
Problem 71: It says Nex is 0, but it still has a solution...


'''

class ProblemEvaluation:
	def __init__(self):
		self.f = None
		self.x = None
		self.constraint_values = None


class ProblemDescription:
	def __init__(self, problem_no):
		self.problem_no = problem_no
		self.n = None
		self.nb_linear_ineq = None
		self.nb_nonlinear_ineq = None
		self.nb_linear_equality = None
		self.nb_nonlinear_equality = None
		self.x0 = None
		self.lower_bounds = None
		self.upper_bounds = None
		self.solution_is_exact = None
		self.num_extremum = None
		self.f_min = None
		self.minimizers = None

	@property
	def x_min(self):
		if self.minimizers.shape[0] == 0:
			return None
		return self.minimizers[0]

	@property
	def m(self):
		return (
				self.nb_linear_equality +
				self.nb_linear_ineq +
				self.nb_nonlinear_equality +
				self.nb_nonlinear_ineq
		)

	def get_constraint_types(self):
		return (
			['inequality'] * (self.nb_linear_ineq + self.nb_nonlinear_ineq) +
			['equality'] * (self.nb_linear_equality + self.nb_nonlinear_equality)
		)

	def evaluate_objective(self, x):
		# padded_x = np.pad(x, (0, cls.NMAX - len(x)))
		schit.l2.x[:self.n] = x
		schit.l8.ntp = self.problem_no

		schit.conv(mode=2)
		return float(schit.l6.fx)

	def get_constraint_evaluator(self, idx):
		def evaluator(x):
			schit.l2.x[:self.n] = x
			schit.l8.ntp = self.problem_no
			schit.l9.index1[:self.m] = 0
			schit.l9.index1[idx] = 1
			schit.conv(mode=4)
			return float(schit.l3.g[idx])
		return evaluator

	def evaluate(self, x):
		# padded_x = np.pad(x, (0, cls.NMAX - len(x)))
		schit.l2.x[:self.n] = x
		schit.l8.ntp = self.problem_no

		evaluation = ProblemEvaluation()
		evaluation.x = x.copy()

		schit.conv(mode=2)
		evaluation.f = float(schit.l6.fx)

		m = self.m
		schit.l9.index1[:m] = 1
		schit.conv(mode=4)
		evaluation.constraint_values = np.array(schit.l3.g[:m], dtype=np.float64)

		return evaluation


class SchittkowskiCache:
	NMAX=101
	MMAX=50

	PROBLEM_DESCRIPTIONS = {}


def create_problem_description(problem_no):
	if problem_no in SchittkowskiCache.PROBLEM_DESCRIPTIONS:
		return SchittkowskiCache.PROBLEM_DESCRIPTIONS[problem_no]

	desc = ProblemDescription(problem_no)
	schit.l8.ntp = problem_no
	schit.conv(mode=1)

	desc.n = int(schit.l1.n)
	desc.nb_linear_ineq = int(schit.l1.nili)
	desc.nb_nonlinear_ineq = int(schit.l1.ninl)
	desc.nb_linear_equality = int(schit.l1.neli)
	desc.nb_nonlinear_equality = int(schit.l1.nenl)

	desc.x0 = np.array(schit.l2.x[:desc.n], dtype=np.float64)

	desc.lower_bounds = [None] * desc.n
	desc.upper_bounds = [None] * desc.n
	for i in range(desc.n):
		if bool(schit.l11.lxl[i]):
			desc.lower_bounds[i] = schit.l13.xl[i]
		if bool(schit.l12.lxu[i]):
			desc.upper_bounds[i] = schit.l14.xu[i]

	desc.solution_is_exact = bool(schit.l20.lex)
	desc.num_extremum = int(schit.l20.nex)

	if desc.num_extremum < 0:
		# Infinitely many solutions
		num_sols_to_read = 1
	elif desc.num_extremum == 0:
		# Problem 71 still has a solution...
		num_sols_to_read = 1
	else:
		num_sols_to_read = desc.num_extremum

	desc.minimizers = np.array(
		[
			schit.l20.xex[idx * desc.n:(idx + 1) * desc.n]
			for idx in range(num_sols_to_read)
		],
		dtype=np.float64
	)

	desc.f_min = schit.l20.fex

	SchittkowskiCache.PROBLEM_DESCRIPTIONS[problem_no] = desc
	return desc


if __name__ == '__main__':
	desc = create_problem_description(24)
	for key, value in vars(desc).items():
		print(key, ':', value)

	x = np.array([4, 2], dtype=np.float64)
	evaluation = desc.evaluate(x)

	for key, value in vars(evaluation).items():
		print(key, ':', value)
