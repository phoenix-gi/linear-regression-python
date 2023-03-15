import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from logistic_cost_function import LogisticCostFunction
from gradient_descent_experiment import GradientDescentExperiment
from matplotlib.animation import FuncAnimation

np.seterr(all='raise')


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


lre = BinaryClassificationExperiment()
lre.set_stop_threshold(0.00001)
lre.set_start_theta([-10, -10, -10])
lre.set_max_iterations(100000)

rates = [0.00001, 0.001, 0.01, 0.05]

for rate in rates:
    lre.set_learning_rate(rate)
    lre.run_all()
