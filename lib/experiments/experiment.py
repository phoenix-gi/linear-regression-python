class Experiment:
    def run_all(self):
        self.setup()
        self.run_experiment()
        self.output_results()

    def setup(self):
        raise NotImplementedError

    def run_experiment(self):
        raise NotImplementedError

    def output_results(self):
        raise NotImplementedError
