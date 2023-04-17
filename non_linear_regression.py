from lib.experiments.non_linear_regression_experiment import NonLinearRegressionExperiment

nlre = NonLinearRegressionExperiment()
nlre.set_stop_threshold(0.000001)
nlre.set_start_theta([3, 10, 10])
nlre.set_max_iterations(500000)

rates = [0.01, 0.001, 0.001]

for rate in rates:
    nlre.set_learning_rate(rate)
    nlre.run_all()
