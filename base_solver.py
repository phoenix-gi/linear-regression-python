class BaseSolver:
    def __init__(self, func) -> None:
        self.func = func

    def solve(self):
        raise NotImplementedError
