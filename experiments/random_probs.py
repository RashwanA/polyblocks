"""Random problem generators for testing solvers."""

from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

from polyblocks.abstract import ABPolyblock


class MonotoneProb(ABC):
    def __init__(self, vars: int, x_max=None, x_min=None, seed=0):
        self.rng = np.random.default_rng(seed=seed)
        self.vars = vars
        self.x_max = (
            np.ones(vars, dtype=np.float32)
            if x_max is None
            else np.broadcast_to(x_max, vars)
        )
        self.x_min = (
            np.zeros(vars, dtype=np.float32)
            if x_min is None
            else np.broadcast_to(x_min, vars)
        )
        self.dtype = self.x_min.dtype
        self.ub_trg: ndarray
        self.lb_trg: ndarray
        self.params: dict[str, dict[str, ndarray]] = {}
        self.reroll()

    def reroll(self):
        for type in ("obj", "ub", "lb"):
            self.params[type] = self.generate_params()
        self.rand_trg()

    @abstractmethod
    def generate_params(self) -> dict[str, ndarray]:
        """Generates parameters for class of functions."""

    @abstractmethod
    def eval(self, x: ndarray, **kwargs: ndarray) -> ndarray:
        """Evaluate function given `x` and generated parameters."""

    def _apply(self, x: ndarray, type: str) -> ndarray:
        """Applies the function of type `type` to a batched input`x`."""
        return self.eval(x, **self.params[type])

    def obj(self, x):
        return self._apply(x, "obj")

    def ub_oracle(self, x):
        return self._apply(x, "ub") <= self.ub_trg

    def lb_oracle(self, x):
        return self._apply(x, "lb") >= self.lb_trg

    def rand_x(self, reduce=1.0):
        rnd_val = self.rng.random(self.vars, dtype=self.dtype)
        rnd_x = rnd_val * reduce * (self.x_max - self.x_min) + self.x_min
        return rnd_x

    def rand_trg(self, leeway=0.01):
        x = self.rand_x()[None]
        self.ub_trg = self._apply(x + leeway, "ub")
        self.lb_trg = self._apply(x - leeway, "lb")

    def solve(self, solver: type[ABPolyblock], **kwargs):
        return solver.solve(
            obj=self.obj,
            x_l=self.x_min,
            x_u=self.x_max,
            ub_oracle=self.ub_oracle,
            lb_oracle=self.lb_oracle,
            **kwargs,
        )


class RandQCQP(MonotoneProb):
    def __init__(self, vars, x_max=None, x_min=None, seed=0):
        super().__init__(vars=vars, x_max=x_max, x_min=x_min, seed=seed)

    def generate_params(self):
        random = self.rng.random
        dtype = self.dtype

        vector = random(self.vars, dtype=dtype)
        mat = random((self.vars, self.vars), dtype=dtype) + 1
        return {"vector": vector, "mat": mat}

    def eval(self, x, vector, mat):
        x = x.clip(self.x_min, self.x_max)
        y = x @ vector
        y += np.einsum("bi,ij,bj->b", x, mat, x)
        return y / self.vars


class RandMonotoneNets(MonotoneProb):
    def __init__(self, vars, hidden=16, x_max=None, x_min=None, seed=0):
        self.hidden = hidden
        super().__init__(vars=vars, x_max=x_max, x_min=x_min, seed=seed)

    def generate_params(self):
        rnd = self.rng
        h, x = self.hidden, self.vars
        dtype = self.x_max.dtype

        mat1 = rnd.random((h, x), dtype=dtype)
        bias1 = rnd.random(h, dtype=dtype) - 0.5
        mat2 = rnd.random(h, dtype=dtype)
        return {"mat1": mat1, "bias1": bias1, "mat2": mat2}

    def eval(self, x, mat1, bias1, mat2):
        mm = (x @ mat1.T) / self.hidden
        l1 = np.maximum(mm + bias1, 0.0)
        return l1 @ mat2


class RandSteps(RandMonotoneNets):
    def __init__(self, vars, hidden=16, x_max=None, x_min=None, seed=0):
        super().__init__(vars, hidden=hidden, x_max=x_max, x_min=x_min, seed=seed)

    def eval(self, x, mat1, bias1, mat2):
        out = super().eval(x, mat1, bias1, mat2) / 2
        return out.round(2) * 2
