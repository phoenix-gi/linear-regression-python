from base_solver import BaseSolver

class GradientDescent(BaseSolver):
    def __init__(self, func) -> None:
        self.history = []
        super().__init__(func)

    def solve(self):
        theta_0 = -9
        theta_1 = -10
        rate = 0.005
        convergence = False
        history_0 = [theta_0]
        history_1 = [theta_1]
        while not convergence:
            c_0 = self.func.derivative(0)(theta_0, theta_1)
            c_1 = self.func.derivative(1)(theta_0, theta_1)
            t_0 = theta_0 - rate*c_0
            t_1 = theta_1 - rate*c_1
            theta_0 = t_0
            theta_1 = t_1
            history_0.append(theta_0)
            history_1.append(theta_1)
            convergence = abs(c_0) <= 0.001 and abs(c_1) <= 0.001

        self.history = [history_0, history_1]
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        return theta_0, theta_1

    def get_history(self):
        return self.history
