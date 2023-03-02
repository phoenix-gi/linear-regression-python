from base_function import BaseFunction


class SquareFunction(BaseFunction):
    def __init__(self, k=1, shift_x=0, shift_y=0) -> None:
        self.k = k
        self.shift_x = shift_x
        self.shift_y = shift_y

    def __call__(self, *theta):
        return self.k*(theta[0]-self.shift_x)**2+self.shift_y

    def derivative(self, partial_index):
        if partial_index > 0:
            return 0
        return lambda *theta: 2*self.k*(theta[0]-self.shift_x)
