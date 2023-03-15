from base_function import BaseFunction
import math

class LogisticCostFunction(BaseFunction):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, *theta):
        m = len(self.x)
        return sum(
            map(lambda i: self.cost(
                self.h(theta, self.x[i]), self.y[i]), range(0, m)
                )
        )/m

    def cost(self, h, y):
        # -y * math.log(h)-(1-y)*math.log(1-h)
        z = (h**y)*((1-h)**(1-y))
        if z < 1E-323:
            z = 1E-323
        return -math.log(z)

    def h(self, theta, x_i):
        return 1/(1+math.exp(-sum(map(lambda j: theta[j]*x_i[j], range(0, len(theta))))))

    def derivative(self, partial_index):
        return lambda *theta: (
            sum(
                map(lambda i:
                    self.x[i][partial_index]*(self.h(theta, self.x[i]) - self.y[i]), range(0, len(self.x)))
            )/len(self.x)
        )
