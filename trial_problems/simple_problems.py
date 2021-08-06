import numpy as np

from trial_problems.evaluation import Evaluation
from trial_problems.base_problem import BaseProblem


class EasyTrialProblem(BaseProblem):
    def get_name(self):
        return 'simple'

    @property
    def num_constraints(self):
        return 1

    def get_initial_x(self):
        return np.array([0, 5])

    def get_initial_r(self):
        return 1.0

    def get_initial_q(self):
        return np.diag([2, 1])

    def get_initial_center(self):
        return self.get_initial_x()

    def get_initial_delta(self):
        return self.get_initial_r()

    def to_json(self):
        return {'type': 'simple'}

    def evaluate(self, x):
        evaluation = Evaluation()
        evaluation.x = x
        evaluation.objective = x[0] ** 2 + (x[1] + 5) ** 2
        evaluation.constraints = np.array([
            x[0] ** 2 - x[1]
        ])
        evaluation.success = True
        return evaluation.filter()


class NarrowTrialProblem(BaseProblem):
    def get_name(self):
        return 'narrow'

    def __init__(self, a=0.5):
        self.a = a

    @property
    def num_constraints(self):
        return 2

    def get_initial_x(self):
        return np.array([5, 0])

    def get_initial_r(self):
        return 1

    def get_initial_q(self):
        return np.identity(2)

    def get_initial_center(self):
        return self.get_initial_x()

    def get_initial_delta(self):
        return self.get_initial_r()

    def to_json(self):
        return {'type': 'narrow'}

    def evaluate(self, x):
        evaluation = Evaluation()
        evaluation.x = x

        evaluation.objective = x[0] + 10 * (x[1] - 0.75 * self.a * x[0] * np.sin(x[0])) ** 2
        # x[1] <= +a * x[0] => -a * x[0] + x[1] <= 0
        # x[1] >= -a * x[0] => -a * x[0] - x[1] <= 0
        evaluation.constraints = np.array([
            -self.a * x[0] + x[1],
            -self.a * x[0] - x[1],
        ])
        evaluation.success = True
        return evaluation.filter()


class Rosenbrock(BaseProblem):
    def get_name(self):
        return 'rosenbrock'

    @property
    def num_constraints(self):
        return 3

    def get_initial_x(self):
        return np.array([0, 5])

    def get_initial_r(self):
        return 1

    def get_initial_q(self):
        return np.identity(2)

    def get_initial_center(self):
        return self.get_initial_x()

    def get_initial_delta(self):
        return self.get_initial_r()

    def to_json(self):
        return {'type': 'rosenbrock'}

    def evaluate(self, x):
        evaluation = Evaluation()
        evaluation.x = x
        evaluation.objective = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        evaluation.constraints = np.array([
            x[0] ** 2 - x[1],
            (x[0] - 0) ** 2 - (x[1] - 5) ** 2 - 5 ** 2,
            x[0] + x[1] - 100,
        ])
        evaluation.success = True
        return evaluation.filter()

