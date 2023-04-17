import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from .gradient_descent_experiment import GradientDescentExperiment
from ..functions.logistic_cost_function import LogisticCostFunction
from matplotlib.animation import FuncAnimation
import calendar
import time

class BinaryClassificationExperiment(GradientDescentExperiment):
    def setup_func(self):
        self.real_theta = [3*(np.random.rand()-0.5), 3 *
                           (np.random.rand()-0.5), 5*(np.random.rand()-0.5)]
        self.x = list(map(lambda i: [1, np.random.rand()*10-5, np.random.rand()*10-5], range(0, 100)))
        self.y = list(map(lambda x:  1 if (self.real_theta[0]*x[0] + self.real_theta[1]*x[1]+self.real_theta[2]*x[2] >= 0) else 0, self.x))
        return LogisticCostFunction(self.x, self.y)

    def output_results(self):
        iterations = self.get_history_iterations()
        values = self.get_history_values()

        fig, ax = plt.subplots()
        x1 = list(map(lambda x: x[1], self.x))
        x2 = list(map(lambda x: x[2], self.x))
        c = list(map(lambda y: 'blue' if y == 1 else 'green', self.y))
        ax.scatter(x1, x2, c=c)
        x1 = np.arange(-5,5,0.1)

        t = self.theta
        x2 = list(map(lambda i: (-t[0]-t[1]*x1[i])/t[2],range(0,len(x1))))
        ax.plot(x1,x2,'r')
        t = self.real_theta
        x2 = list(map(lambda i: (-t[0]-t[1]*x1[i])/t[2],range(0,len(x1))))
        ax.plot(x1,x2,'m')
        plt.show()

        print('result theta')
        print(self.theta)
        print('real theta')
        print(self.real_theta)

        plt.style.use('_mpl-gallery')

        fig, ax = plt.subplots()
        ax.plot(list(range(0, len(iterations))), iterations, 'b.')
        plt.show()

        # make animation of learning process
        fig, ax = plt.subplots()
        x1 = list(map(lambda x: x[1], self.x))
        x2 = list(map(lambda x: x[2], self.x))
        c = list(map(lambda y: 'blue' if y == 1 else 'green', self.y))
        ax.scatter(x1, x2, c=c)
        x1 = np.arange(-5,5,0.1)

        t = self.real_theta
        x2 = list(map(lambda i: (-t[0]-t[1]*x1[i])/t[2],range(0,len(x1))))
        ax.plot(x1,x2,'m')

        line, = ax.plot([], [], 'r')

        num_of_frames = 200
        if len(values) < num_of_frames:
            num_of_frames = len(values)

        def init():
            line.set_data([], [])
            return line,
        def animate(i):
            step = len(values) // num_of_frames
            shift = len(values) - (num_of_frames - 1) * step - 1
            t = values[i * step + shift]
            x = np.arange(-5,5,0.1)
            y = list(map(lambda j: (-t[0]-t[1]*x[j])/t[2],range(0,len(x))))
            line.set_data(x, y)
            return line,

        anim = FuncAnimation(fig, animate, init_func=init,
                                    frames=num_of_frames, interval=40, blit=True)

        plt.show()
        anim.save(f'binary_classification_{calendar.timegm(time.gmtime())}.gif', writer='imagemagick')
