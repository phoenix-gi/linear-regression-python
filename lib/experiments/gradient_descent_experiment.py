from .experiment import Experiment
from ..algorithms.gradient_descent import GradientDescent


class GradientDescentExperiment(Experiment):
    def setup(self):
        self.set_func(self.setup_func())

    def run_experiment(self):
        gd_solver = GradientDescent(self.func)
        self.theta = gd_solver.solve(
            theta=self.start_theta, rate=self.learning_rate, max_iterations=self.max_iterations, stop_threshold=self.stop_threshold
        )
        self.history = gd_solver.get_history()

    def set_func(self, func):
        self.func = func

    def get_func(self):
        return self.func

    def setup_func(self):
        raise NotImplementedError

    def set_start_theta(self, start_theta):
        self.start_theta = start_theta

    def get_start_theta(self):
        return self.start_theta

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.learning_rate

    def set_max_iterations(self, max_iterations):
        self.max_iterations = max_iterations

    def get_max_iterations(self):
        return self.max_iterations

    def set_stop_threshold(self, stop_threshold):
        self.stop_threshold = stop_threshold

    def get_stop_threshold(self):
        return self.stop_threshold

    def get_history(self):
        return self.history

    def get_history_values(self):
        return self.history['values']

    def get_history_iterations(self):
        return self.history['iterations']
