from abc import *
import numpy as np
from trial_problems.evaluation import Evaluation


class BaseProblem(ABC):
    def __init__(self, inf_strategy):
        self.inf_strategy = inf_strategy

    @property
    def n(self):
        return len(self.get_initial_x())

    @abstractmethod
    def num_constraints(self):
        pass

    def get_initial_r(self):
        return 1.0

    def get_initial_q(self):
        return np.eye(self.n)

    def get_initial_center(self):
        return self.get_initial_x()

    def get_initial_delta(self):
        return 1.0

    def evaluate(self, x):
        return self.inf_strategy.apply(self.construct_evaluation(x))

    @abstractmethod
    def construct_evaluation(self, x):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_initial_x(self):
        pass

    @abstractmethod
    def to_json(self):
        pass
