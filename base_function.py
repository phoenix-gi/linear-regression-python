class BaseFunction:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwds):
        return self.func(*args, **kwds)

    def derivative(self, partial_index):
        raise NotImplementedError(
            "Derivative function must be implemented in subclass"
        )
