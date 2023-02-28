from base_function import BaseFunction


class CostFunction(BaseFunction):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, *theta):
        return sum(
            map(lambda i:
                (sum(map(lambda j: theta[j]*self.x[i][j], range(0, len(theta)))) - self.y[i]) ** 2, range(0, len(self.x)))
        )/(2.0 * len(self.x))

    def derivative(self, partial_index):
        return lambda *theta: (
            sum(
                map(lambda i:
                    self.x[i][partial_index]*(sum(map(lambda j: theta[j]*self.x[i][j], range(0, len(theta)))) - self.y[i]), range(0, len(self.x)))
            )/len(self.x)
        )
