from .base_function import BaseFunction


class QuadraticErrorCostFunction(BaseFunction):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, *theta):
        return sum(
            map(lambda i:
                (self.h(theta, self.x[i]) - self.y[i]) ** 2, range(0, len(self.x)))
        )/(2.0 * len(self.x))

    def h(self, theta, x_i):
        return sum(map(lambda j: theta[j]*x_i[j], range(0, len(theta))))

    def derivative(self, partial_index):
        return lambda *theta: (
            sum(
                map(lambda i:
                    self.x[i][partial_index]*(self.h(theta, self.x[i]) - self.y[i]), range(0, len(self.x)))
            )/len(self.x)
        )
