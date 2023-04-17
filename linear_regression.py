from lib.experiments.linear_regression_experiment import LinearRegressionExperiment

lre = LinearRegressionExperiment()
lre.set_stop_threshold(0.01)
lre.set_start_theta([-15, 15])
lre.set_max_iterations(10000)

rates = [0.001, 0.01, 0.05]

for rate in rates:
    lre.set_learning_rate(rate)
    lre.run_all()
