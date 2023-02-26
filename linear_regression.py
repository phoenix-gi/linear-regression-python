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
m = len(x)
fig, ax = plt.subplots()
ax.plot(x, y, 'b.')
plt.show()

c = CostFunction(m, x, y)

c_v = np.frompyfunc(c, 2, 1)

gd_solver = GradientDescent(c)
theta_0, theta_1 = gd_solver.solve()
history_x, history_y = gd_solver.get_history()

print('result')
print(theta_0, theta_1)
print(real_theta_0, real_theta_1)


x = np.arange(-10, 10, 0.5)
y = theta_0 + theta_1 * x
fig, ax = plt.subplots()
ax.plot(x, y, 'r.')
plt.show()


plt.style.use('_mpl-gallery')

x = np.arange(-10, 10, 0.25)
y = list(map(lambda theta_1: c(1, theta_1), x))
y = c_v(x, 1)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()

# Make data
X = np.arange(-10, 10, 0.1)
Y = np.arange(-10, 10, 0.1)
X, Y = np.meshgrid(X, Y)
Z = c_v(X, Y)

# Plot the surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.view_init(40, 50)
ax.plot_surface(X, Y, Z, cmap=cm.Blues)
ax.plot3D(history_x, history_y, c_v(history_x, history_y), 'g.')

plt.show()
