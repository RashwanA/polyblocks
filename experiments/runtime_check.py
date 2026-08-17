import numpy as np
from random_probs import RandMonotoneNets, RandQCQP, RandSteps

from polyblocks import BalancedPOA, BasePOA, TreePOA

SEED = 0
DIM = 5
PROB = RandMonotoneNets(DIM, seed=SEED)
SOLVER = TreePOA

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    print(f"Solver: {SOLVER.__name__}")
    PROB.solve(
        SOLVER,
        verbose_gap=100,
    )
