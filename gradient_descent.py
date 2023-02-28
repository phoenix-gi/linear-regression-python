from base_solver import BaseSolver


class GradientDescent(BaseSolver):
    def __init__(self, func) -> None:
        self.history = []
        super().__init__(func)

    def solve(self, rate=0.005):
        theta = list(map(lambda i: 2, range(0, len(self.func.x[0]))))
        convergence = False
        values = [theta]
        history_iterations = [self.func(*theta)]
        while not convergence:
            theta = list(map(lambda j: theta[j] - rate*self.func.derivative(j)(*theta), range(0, len(theta))))

            values.append(theta)
            history_iterations.append(self.func(*theta))
            convergence = abs(history_iterations[-2] - history_iterations[-1]) <= 0.01
        self.history = {
            "values": values,
            "iterations": history_iterations
        }
        return theta

    def get_history(self):
        return self.history
