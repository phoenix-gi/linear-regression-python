import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from square_function import SquareFunction
from gradient_descent_experiment import GradientDescentExperiment


class SquareFunctionExperiment(GradientDescentExperiment):
    def __init__(self) -> None:
        super().__init__()

    def setup_func(self):
        return SquareFunction(k=1, shift_y=-60)

    def output_results(self):
        iterations = self.get_history_iterations()
        values = self.get_history_values()
        c_v = np.frompyfunc(self.get_func(), 1, 1)

        fig, ax = plt.subplots()
        x1 = np.arange(-10, 10, 0.5)
        y1 = c_v(x1)
        ax.plot(x1, y1, 'r.-')
        plt.show()

        print(self.theta)
        print(self.get_func()(*self.theta))

        fig, ax = plt.subplots()
        ax.plot(list(map(lambda values: values[0], values)), iterations, 'g.-')
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(list(range(0, len(iterations))), iterations, 'b.-')
        plt.show()


sfe = SquareFunctionExperiment()
sfe.set_stop_threshold(0.01)
sfe.set_start_theta([-10])
sfe.set_max_iterations(1000)

rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
         0.8, 0.9, 0.99, 1.0, 1.001, 1.01, 1.05]

for rate in rates:
    sfe.set_learning_rate(rate)
    sfe.run_all()
