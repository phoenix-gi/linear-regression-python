from lib.experiments.square_function_experiment import SquareFunctionExperiment

sfe = SquareFunctionExperiment()
sfe.set_stop_threshold(0.01)
sfe.set_start_theta([-10])
sfe.set_max_iterations(1000)

rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
         0.8, 0.9, 0.99, 1.0, 1.001, 1.01, 1.05]

for rate in rates:
    sfe.set_learning_rate(rate)
    sfe.run_all()
