from .base_solver import BaseSolver


class GradientDescent(BaseSolver):
    def __init__(self, func) -> None:
        self.history = []
        super().__init__(func)

    def solve(self, theta, rate=0.005, max_iterations=10000, stop_threshold=0.01):
        convergence = False
        values = [theta]
        history_iterations = [self.func(*theta)]
        while not convergence:
            theta = list(map(lambda j: theta[j] - rate*self.func.derivative(j)(*theta), range(0, len(theta))))

            values.append(theta)
            history_iterations.append(self.func(*theta))
            convergence = abs(history_iterations[-2] - history_iterations[-1]) <= stop_threshold or len(history_iterations) >= max_iterations
        self.history = {
            "values": values,
            "iterations": history_iterations
        }
        return theta

    def get_history(self):
        return self.history
