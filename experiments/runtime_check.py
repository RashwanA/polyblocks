from time import perf_counter

import numpy as np
from random_probs import RandMonotoneNets, RandQCQP, RandSteps

from polyblocks import BalancedPOA, BasePOA, TreePOA

SEED = 0
DIM = 5
PROB = RandQCQP(DIM, seed=SEED)
SOLVER = TreePOA

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    print(f"Solver: {SOLVER.__name__}")
    st = perf_counter()
    PROB.solve(
        SOLVER,
        verbose_gap=100,
    )
    t1 = perf_counter() - st

    print(f"Runtime: {t1:.4f} s")
