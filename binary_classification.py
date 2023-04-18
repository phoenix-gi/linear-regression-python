from lib.experiments.binary_classification_experiment import BinaryClassificationExperiment
import numpy as np
np.seterr(all='raise')

real_theta = [
    3*(np.random.rand()-0.5),
    3 * (np.random.rand()-0.5),
    5*(np.random.rand()-0.5)
]

x = list(
    map(
        lambda i: [1, np.random.rand()*10-5,
                   np.random.rand()*10-5], range(0, 100)
    )
)

y = list(
    map(
        lambda x:  1 if (
            real_theta[0]*x[0] + real_theta[1]*x[1]+real_theta[2]*x[2] >= 0
        ) else 0, x
    )
)

lre = BinaryClassificationExperiment()
lre.set_real_theta(real_theta)
lre.set_x(x)
lre.set_y(y)

lre.set_stop_threshold(0.00001)
lre.set_start_theta([-10, -10, -10])
lre.set_max_iterations(100000)

rates = [0.00001, 0.001, 0.01, 0.05]

for rate in rates:
    lre.set_learning_rate(rate)
    lre.run_all()
