"""Runs all experiments shown in the paper."""

import os
from collections.abc import Iterable
from itertools import product

import numpy as np
import pandas as pd
from random_probs import MonotoneProb, RandMonotoneNets, RandQCQP, RandSteps

from polyblocks import ABPolyblock, BalancedPOA, BasePOA, TreePOA
from polyblocks.utils import print_row


def save_csv(fname, **kwargs):
    if fname[-4:] != ".csv":
        fname += ".csv"
    header = not os.path.exists(fname)
    entry = pd.DataFrame([kwargs])
    entry.to_csv(fname, index=False, header=header, mode="a")


def experiments(
    seed: int = 0,
    n_probs: int = 20,
    fname: str = "results.csv",
    prob_clss: Iterable[type[MonotoneProb]] = (RandSteps, RandMonotoneNets, RandQCQP),
    dims: Iterable[int] = range(4, 7),
    solver_names: tuple[str, ...] | None = None,
) -> None:
    """
    Run every solver variant on `n_probs` instances of each problem class and dimension, appending to `fname`.

    `prob_clss`, `dims` and `solver_names` restrict the run to a subset, e.g. to re-run a single block.
    """

    solvers: tuple[type[ABPolyblock], ...] = (
        type("Vectorised", (TreePOA,), {"PROJECTED_VERTICES": 8}),
        type("TreeBased", (TreePOA,), {"PROJECTED_VERTICES": 1}),
        type("Relaxed", (BalancedPOA,), {}),
        type("Balanced", (BalancedPOA,), {}),
        type("Base", (BasePOA,), {}),
    )
    if solver_names is not None:
        solvers = tuple(s for s in solvers if s.__name__ in solver_names)

    parent_dir = os.path.dirname(__file__)
    full_path = os.path.join(parent_dir, fname)

    ## Tests solvers on all problem classes at all dimensions
    for clss, dim, solver in product(prob_clss, dims, solvers):
        prob = clss(vars=dim, seed=seed)
        for i in range(n_probs):
            sol = prob.solve(
                solver,
                delta=0.0 if solver.__name__ in ("Balanced", "Base") else 1e-3,
            )

            print_row(
                header=i % 50 == 0,
                **{
                    "Itr": i,
                    "Problem Class": clss.__name__,
                    "Solver Variant": solver.__name__,
                    "Runtime (sec)": sol.runtime,
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
                runtime=round(sol.runtime, 3),
                status=sol.status,
                obj=round(sol.obj, 5) if sol.obj > -np.inf else np.nan,
                max_polyblock=sol.max_polyblock,
                n_iter=sol.n_iter,
            )


if __name__ == "__main__":
    experiments()
