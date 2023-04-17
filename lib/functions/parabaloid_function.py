from .base_function import BaseFunction


class ParabaloidFunction(BaseFunction):
    def __init__(self, a=1, b=1, sx=0, sy=0, sz=0) -> None:
        self.a = a
        self.b = b
        self.sx = sx
        self.sy = sy
        self.sz = sz

    def __call__(self, *theta):
        return self.sz+self.a * ((theta[0]+self.sx)**2)+self.b * ((theta[1]+self.sy)**2)

    def derivative(self, partial_index):
        if partial_index == 0:
            return lambda *theta: 2*(theta[0]+self.sx)*self.a
        if partial_index == 1:
            return lambda *theta: 2*(theta[1]+self.sy)*self.b
