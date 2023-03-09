import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from parabaloid_function import ParabaloidFunction
from gradient_descent_experiment import GradientDescentExperiment


class ParabaloidFunctionExperiment(GradientDescentExperiment):
    def __init__(self) -> None:
        super().__init__()

    def setup_func(self):
        return ParabaloidFunction(sx=5,sy=-5,sz=2)

    def output_results(self):
        iterations = self.get_history_iterations()
        values = self.get_history_values()
        c_v = np.frompyfunc(self.get_func(), 2, 1)

        print(self.theta)
        print(self.get_func()(*self.theta))

        fig, ax = plt.subplots()
        ax.plot(list(range(0, len(iterations))), iterations, 'b.-')
        plt.show()

        Y = np.arange(-20, 20, 0.1)
        X = np.arange(-20, 20, 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = c_v(X, Y)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.view_init(40, 10)
        ax.plot_wireframe(X, Y, Z, cmap=cm.Blues)
        ax.plot3D(list(map(lambda values: values[0], values)), list(map(
            lambda values: values[1], values)), iterations, 'r.')
        plt.show()

        Z = np.array(c_v(X, Y).tolist())
        fig, ax = plt.subplots()
        ax.contour(X,Y,Z,levels=np.arange(-100, 1200, 10))
        ax.plot(list(map(lambda values: values[0], values)),
                list(map(lambda values: values[1], values)), 'b.')
        ax.plot([self.theta[0]],[self.theta[1]], 'r.')

        plt.show()




sfe = ParabaloidFunctionExperiment()
sfe.set_start_theta([-20,-20])
sfe.set_max_iterations(1000)

rates = [0.1, 0.7, 1.0005]

for rate in rates:
    sfe.set_learning_rate(rate)
    sfe.run_all()
