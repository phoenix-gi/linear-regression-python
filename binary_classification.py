import numpy as np
np.seterr(all='raise')

from lib.experiments.binary_classification_experiment import BinaryClassificationExperiment


lre = BinaryClassificationExperiment()
lre.set_stop_threshold(0.00001)
lre.set_start_theta([-10, -10, -10])
lre.set_max_iterations(100000)

rates = [0.00001, 0.001, 0.01, 0.05]

for rate in rates:
    lre.set_learning_rate(rate)
    lre.run_all()
