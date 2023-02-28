import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from cost_function import CostFunction
from gradient_descent import GradientDescent

# random linear sample data
real_theta_0 = 3*(np.random.rand()-0.5)
real_theta_1 = 5*(np.random.rand()-0.5)
x = np.arange(-10, 10, 0.5)
y = list(map(lambda u: u - 2*(np.random.rand()-0.5),
         list(real_theta_0+real_theta_1*x)))

c = CostFunction(list(map(lambda val: [1,val], x)), y)

c_v = np.frompyfunc(c, 2, 1)

gd_solver = GradientDescent(c)
theta_0, theta_1 = gd_solver.solve()
history = gd_solver.get_history()
values = history['values']
iterations = history['iterations']

print('result')
print(theta_0, theta_1)
print(real_theta_0, real_theta_1)

fig, ax = plt.subplots()
ax.plot(x, y, 'b.')
x1 = np.arange(-10, 10, 0.5)
y1 = theta_0 + theta_1 * x1
ax.plot(x1, y1, 'r.')
plt.show()

plt.style.use('_mpl-gallery')

x2 = np.arange(-10, 10, 0.25)
y2 = c_v(x2, 1)

fig, ax = plt.subplots()
ax.plot(x2, y2)
plt.show()

# Make data
X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = c_v(X, Y)

# Plot the surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.view_init(40, -10)
ax.plot_surface(X, Y, Z, cmap=cm.Blues)
ax.plot3D(list(map(lambda values: values[0], values)), list(map(lambda values: values[1], values)), list(map(lambda v: c_v(*v), values)), 'g.')

plt.show()

fig, ax = plt.subplots()
ax.plot(list(range(0, len(iterations))), iterations, 'b.')
plt.show()
