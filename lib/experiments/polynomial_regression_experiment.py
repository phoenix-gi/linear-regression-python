import matplotlib.pyplot as plt
import numpy as np
from ..functions.quadratic_error_cost_function import QuadraticErrorCostFunction
from .gradient_descent_experiment import GradientDescentExperiment
from matplotlib.animation import FuncAnimation
import calendar
import time

class PolynomialRegressionExperiment(GradientDescentExperiment):
    def setup_func(self):
        self.real_theta = [2, 0.1, 0.1, 0.005]
        self.x = np.arange(-30, 20, 0.5)
        self.y = list(map(lambda u: u - (np.random.rand()-0.5),
                          list(self.real_theta[0] +
                               self.real_theta[1]*self.x +
                               self.real_theta[2]*(self.x**2) +
                               self.real_theta[3]*(self.x**3)))
                      )
        # our source data is not linear
        # so we add new features
        new_x = list(map(lambda val: [1, val, (val)**2, (val)**3], self.x))
        # but new features are to big
        # so we apply feature scaling
        means = np.array([0, 0, 0, 0])
        mins = [new_x[0][0], new_x[0][1], new_x[0][2], new_x[0][3]]
        maxs = [new_x[0][0], new_x[0][1], new_x[0][2], new_x[0][3]]

        for j in range(0, len(new_x)):
            for i in range(1, len(mins)):
                e = new_x[j][i]
                means[i] = means[i] + e
                if e < mins[i]:
                    mins[i] = e
                if e > maxs[i]:
                    maxs[i] = e
        means = means / len(new_x)
        print('means ', means)
        print('mins ', mins)
        print('maxs ', maxs)
        print(new_x)
        for j in range(0, len(new_x)):
            for i in range(1, len(mins)):
                new_x[j][i] = (new_x[j][i] - means[i])/(maxs[i] - mins[i])
        print(new_x)
        self.means = means
        self.maxs = maxs
        self.mins = mins

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
        x1 = np.arange(-30, 20, 1.5)
        y1 = self.theta[0] + self.theta[1] * (x1 - self.means[1])/(self.maxs[1]-self.mins[1]) + \
            self.theta[2] * ((x1**2 - self.means[2])/(self.maxs[2]-self.mins[2])) + \
            self.theta[3] * (x1**3 - self.means[3])/(self.maxs[3]-self.mins[3])
        ax.plot(x1, y1, 'r.')
        plt.show()

        plt.style.use('_mpl-gallery')

        fig, ax = plt.subplots()
        ax.plot(list(range(0, len(iterations))), iterations, 'b.')
        plt.show()

        print(values)
        print('Num of iterations: ', len(iterations))
        print('theta=', values[-1], ' J(theta)=', iterations[-1])

        # make animation of learning process
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y, 'b.')

        line, = ax.plot([], [], 'r.')

        num_of_frames = len(values)

        def init():
            line.set_data([], [])
            return line,
        def animate(i):
            step = len(values) // num_of_frames
            shift = len(values) - (num_of_frames - 1) * step - 1
            t = values[i * step + shift]
            x1 = np.arange(-30, 20, 1.5)
            y1 = t[0] + t[1] * (x1 - self.means[1])/(self.maxs[1]-self.mins[1]) + \
                 t[2] * ((x1**2 - self.means[2])/(self.maxs[2]-self.mins[2])) + \
                 t[3] * (x1**3 - self.means[3])/(self.maxs[3]-self.mins[3])
            line.set_data(x1, y1)
            return line,

        anim = FuncAnimation(fig, animate, init_func=init,
                                    frames=num_of_frames, interval=200, blit=True)

        plt.show()
        anim.save(f'polynomial_regression_{calendar.timegm(time.gmtime())}.gif', writer='imagemagick')
