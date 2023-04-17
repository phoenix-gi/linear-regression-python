from lib.experiments.polynomial_regression_experiment import PolynomialRegressionExperiment

pre = PolynomialRegressionExperiment()
pre.set_stop_threshold(0.000000001)
pre.set_start_theta([-100, -200, -2, -2])
pre.set_max_iterations(10000)

rates = [0.9]

for rate in rates:
    pre.set_learning_rate(rate)
    pre.run_all()
