from lib.experiments.parabaloid_function_experiment import ParabaloidFunctionExperiment

sfe = ParabaloidFunctionExperiment()
sfe.set_stop_threshold(0.01)
sfe.set_start_theta([-20,-20])
sfe.set_max_iterations(1000)

rates = [0.1, 0.7, 1.0005]

for rate in rates:
    sfe.set_learning_rate(rate)
    sfe.run_all()
