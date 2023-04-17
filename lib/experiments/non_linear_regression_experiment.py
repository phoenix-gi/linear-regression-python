import matplotlib.pyplot as plt
import numpy as np
from ..functions.quadratic_error_cost_function import QuadraticErrorCostFunction
from .gradient_descent_experiment import GradientDescentExperiment
from matplotlib.animation import FuncAnimation
import calendar
import time


class NonLinearRegressionExperiment(GradientDescentExperiment):
    def setup_func(self):
        self.real_theta = [5, 0, 3]
        self.x = np.arange(2, 20, 0.5)
        self.y = list(map(lambda u: u - 0.5*(np.random.rand()-0.5),
                          list(self.real_theta[0] +
                               self.real_theta[1]*self.x +
                               self.real_theta[2]*(np.sqrt(self.x-2))
                               ))
                      )
        new_x = list(map(lambda val: [1, val, np.sqrt(val-2)], self.x))

        return QuadraticErrorCostFunction(new_x, self.y)

    def output_results(self):
        iterations = self.get_history_iterations()
        values = self.get_history_values()

        print('result theta')
        print(self.theta)
        print('real theta')
        print(self.real_theta)

        fig, ax = plt.subplots()
        ax.plot(self.x, self.y, 'b.')
        x1 = self.x
        y1 = self.theta[0] + self.theta[1] * x1 + \
            self.theta[2] * np.sqrt(x1-2)
        ax.plot(x1, y1, 'r.')
        plt.show()

        plt.style.use('_mpl-gallery')

        fig, ax = plt.subplots()
        ax.plot(list(range(0, len(iterations))), iterations, 'b.')
        plt.show()

        print('theta=', values[-1], ' J(theta)=', iterations[-1])

        # make animation of learning process
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y, 'b.')

        line, = ax.plot([], [], 'r.')
        num_of_frames = 500

        def init():
            line.set_data([], [])
            return line,
        def animate(i):
            step = len(values) // num_of_frames
            shift = len(values) - (num_of_frames - 1) * step - 1
            t = values[i * step + shift]
            x1 = np.arange(2, 20, 0.5)
            y1 = t[0] + t[1] * x1 + t[2] * np.sqrt(x1-2)
            line.set_data(x1, y1)
            return line,

        anim = FuncAnimation(fig, animate, init_func=init,
                                    frames=num_of_frames, interval=200, blit=True)

        plt.show()
        anim.save(f'non_linear_regression_{calendar.timegm(time.gmtime())}.gif', writer='imagemagick')
