import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from cost_function import CostFunction
from gradient_descent import GradientDescent
from base_function import BaseFunction

class SquareFunction(BaseFunction):
    def __init__(self, k = 1, shift_x=0, shift_y=0) -> None:
        self.k = k
        self.shift_x = shift_x
        self.shift_y = shift_y

    def __call__(self, *theta):
        return self.k*(theta[0]-self.shift_x)**2+self.shift_y

    def derivative(self, partial_index):
        if partial_index > 0:
          return 0
        return lambda *theta: 2*self.k*(theta[0]-self.shift_x)

c = SquareFunction(k=1, shift_y=-60)
c_v = np.frompyfunc(c, 1, 1)

fig, ax = plt.subplots()
x1 = np.arange(-10, 10, 0.5)
y1 = c_v(x1)
ax.plot(x1, y1, 'r.-')
plt.show()

gd_solver = GradientDescent(c)
theta = gd_solver.solve(theta=[-10],rate=0.9)
history = gd_solver.get_history()
values = history['values']
iterations = history['iterations']

print(theta)
print(c(*theta))

fig, ax = plt.subplots()
ax.plot(list(map(lambda values: values[0], values)), iterations, 'g.-')
plt.show()

fig, ax = plt.subplots()
ax.plot(list(range(0, len(iterations))), iterations, 'b.-')
plt.show()
