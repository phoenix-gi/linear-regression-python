from base_function import BaseFunction

class CostFunction(BaseFunction):
    def __init__(self, m, x, y):
        self.m = m
        self.x = x
        self.y = y
        super().__init__(
            lambda theta_0, theta_1:
            sum(map(lambda i:
                (theta_0 + theta_1 * self.x[i] - self.y[i]) ** 2, range(0, m)))/(2.0 * self.m)
        )

    def derivative(self, partial_index):
        if partial_index == 0:
            return (
                lambda theta_0, theta_1:
                sum(map(lambda i:
                    (theta_0+theta_1 * self.x[i]-self.y[i]), range(0, self.m)))/self.m
            )
        if partial_index == 1:
            return (
                lambda theta_0, theta_1:
                sum(map(lambda i:
                    self.x[i]*(theta_0+theta_1 * self.x[i] -
                               self.y[i]), range(0, self.m))
                    )/self.m
            )
