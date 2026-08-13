"""Runs all experiments shown in the paper."""

import os
from itertools import product
from time import perf_counter

import numpy as np
import pandas as pd
from random_probs import RandMonotoneNets, RandQCQP, RandSteps

from polyblocks import BalancedPOA, BasePOA, TreePOA
from polyblocks.utils import print_row


def save_csv(fname, **kwargs):
    if fname[-4:] != ".csv":
        fname += ".csv"
    header = not os.path.exists(fname)
    entry = pd.DataFrame([kwargs])
    entry.to_csv(fname, index=False, header=header, mode="a")


def experiments(seed=0, n_probs=20, fname="results.csv"):
    dims = range(4, 7)
    solvers = (
        type("Vectorised", (TreePOA,), {"PROJECTED_VERTICES": 8}),
        type("TreeBased", (TreePOA,), {"PROJECTED_VERTICES": 1}),
        type("Relaxed", (BalancedPOA,), {}),
        type("Balanced", (BalancedPOA,), {}),
        type("Base", (BasePOA,), {}),
    )
    prob_clss = (RandSteps, RandMonotoneNets, RandQCQP)

    parent_dir = os.path.dirname(__file__)
    full_path = os.path.join(parent_dir, fname)

    ## Tests solvers on all problem classes at all dimensions
    for clss, dim, solver in product(prob_clss, dims, solvers):
        prob = clss(vars=dim, seed=seed)
        for i in range(n_probs):
            st = perf_counter()
            sol = prob.solve(
                solver,
                delta=0.0 if solver.__name__ in ("Balanced", "Base") else 1e-3,
            )

            runtime = perf_counter() - st
            print_row(
                header=i % 50 == 0,
                **{
                    "Itr": i,
                    "Problem Class": clss.__name__,
                    "Solver Variant": solver.__name__,
                    "Runtime (sec)": runtime,
                    "Termination status": sol.status,
                },
            )
            prob.reroll()

            save_csv(
                full_path,
                problem=clss.__name__,
                solver=solver.__name__,
                dim=dim,
                n_probs=n_probs,
                instance=i,
                runtime=round(runtime, 3),
                success=sol.success,
                obj=round(sol.obj, 3) if sol.obj > -np.inf else np.nan,
            )


if __name__ == "__main__":
    experiments()
