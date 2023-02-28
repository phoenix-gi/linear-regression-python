class BaseFunction:
    def __call__(self, *arguments):
        raise NotImplementedError

    def derivative(self, partial_index):
        raise NotImplementedError(
            "Derivative function must be implemented in subclass"
        )
