import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from .gradient_descent_experiment import GradientDescentExperiment
from ..functions.quadratic_error_cost_function import QuadraticErrorCostFunction



class LinearRegressionExperiment(GradientDescentExperiment):
    def setup_func(self):
        self.real_theta = [3*(np.random.rand()-0.5), 5*(np.random.rand()-0.5)]
        self.x = np.arange(-10, 10, 0.5)
        self.y = list(map(lambda u: u - 2*(np.random.rand()-0.5),
                          list(self.real_theta[0]+self.real_theta[1]*self.x))
                      )
        return QuadraticErrorCostFunction(list(map(lambda val: [1, val], self.x)), self.y)

    def output_results(self):
        iterations = self.get_history_iterations()
        values = self.get_history_values()
        c_v = np.frompyfunc(self.get_func(), 2, 1)
        Y = np.arange(-15, 15, 0.1)
        X = np.arange(-80, 80, 0.5)
        X, Y = np.meshgrid(X, Y)
        Z = c_v(X, Y)

        print('result theta')
        print(self.theta)
        print('real theta')
        print(self.real_theta)

        fig, ax = plt.subplots()
        ax.plot(self.x, self.y, 'b.')
        x1 = np.arange(-10, 10, 0.5)
        y1 = self.theta[0] + self.theta[1] * x1
        ax.plot(x1, y1, 'r.')
        plt.show()

        plt.style.use('_mpl-gallery')

        x2 = np.arange(-10, 10, 0.25)
        y2 = c_v(x2, 1)

        fig, ax = plt.subplots()
        ax.plot(x2, y2)
        plt.show()

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.view_init(40, -10)
        ax.plot_surface(X, Y, Z, cmap=cm.Blues)
        ax.plot3D(list(map(lambda values: values[0], values)), list(map(
            lambda values: values[1], values)), list(map(lambda v: c_v(*v), values)), 'g.')
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(list(range(0, len(iterations))), iterations, 'b.')
        plt.show()

        fig, ax = plt.subplots()
        ax.contour(X, Y, np.array(Z.tolist()), levels=np.arange(-200, 3000, 150))
        ax.plot(list(map(lambda values: values[0], values)),
                list(map(lambda values: values[1], values)), 'g.')
        ax.plot([self.theta[0]], [self.theta[1]], 'r.')
        plt.show()

