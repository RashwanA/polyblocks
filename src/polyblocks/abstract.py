from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import monotone_proj, print_row, tighten


@dataclass
class Solution:
    """
    The result returned by `ABPolyblock.solve`.

    Attributes:
        x: Optimal (or best attained) solution.
        obj: Optimal (or best attained) objective value.
        best_bound: Best upper-bound on the optimal objective.
        success: Whether the solver returned an optimal solution.
        status: Description of the termination status.
        n_iter: Number of iterations executed.
    """

    x: NDArray | None = None
    obj: float = -np.inf
    best_bound: float = np.inf
    success: bool = False
    status: str = "Maximum iterations reached"
    n_iter: int = 0


class ABPolyblock(ABC):
    """
    An abstract class for custom implementations of the Polyblock Outer-approximation (POA) algorithm.

    POA maximises an increasing objective over `G ∩ H ∩ [x_l, x_u]`, where `G` is a normal set given by `ub_oracle` and `H` a co-normal set given by `lb_oracle`.
    It does so by maintaining a *polyblock*: a finite vertex set `V` whose union of boxes `(-inf, v]` contains every feasible point which could still improve the incumbent.
    Each iteration projects vertices of `V` onto the boundary of `G` and cuts the
    infeasible cone strictly above each projection out of the polyblock, so the outer-approximation tightens until `V` empties and the incumbent is certified optimal.

    Child classes choose how that vertex set is stored and refined, and `solve` drives the representation through one cycle per iteration:

    1. `projection_pairs` selects the vertices to refine, each paired with a feasible anchor below it.
    2. `solve` projects each vertex towards its anchor onto the boundary of the eroded set `G_delta`, giving `proj`, and takes the shifted candidates `min(proj + delta, x_u)` as feasible solutions.
    3. `set_min_obj` is called with a new objective cut-off, but only when a candidate improves the incumbent.
    4. `new_vertices` cuts the cones above `proj` out of the polyblock and returns the replacement vertices.
    5. `solve` evaluates the objective on those vertices and passes a feasibility mask to `update`, which reports whether the vertex set is now exhausted.

    Implementing `__init__`, `projection_pairs`, `set_min_obj`, `new_vertices`, and  `update` is therefore enough to define a solver.
    The `size` property is used for limiting memory usage, while the `best_bound` property is optional and only used for progress reporting.

    Attributes:
        POLYBLOCK_LIMIT: Maximum permitted `size` of the polyblock container.
    """

    POLYBLOCK_LIMIT = 2 * int(1e8)

    @abstractmethod
    def __init__(self, lower: NDArray, upper: NDArray) -> None:
        """
        Initialises polyblock representation using lower and upper points which define the feasible rectangle.

        Both points have already been tightened against the oracles by `solve`, so the initial polyblock is the single vertex `upper`.

        Args:
            lower: Lower point of the feasible rectangle, of shape `(dim,)`.
            upper: Upper point of the feasible rectangle, of shape `(dim,)`.
        """

    @abstractmethod
    def projection_pairs(self) -> tuple[NDArray, NDArray]:
        """
        Returns pairs of anchors and vertices for computing monotone projections.

        Called once at the start of each iteration to choose which vertices to refine.
        The selection policy is free, subject only to the requirement that the *maximal* vertex, the one of greatest objective value in the polyblock, is always among those returned; convergence rests on it.
        Returning several pairs cuts several cones per iteration, where downstream routines apply in a vectorised fashion.

        Each anchor need only lie in `G` and strictly below its vertex; the `delta` shift keeping it feasible for the eroded set is applied downstream.

        Returns:
            A tuple `(anchors, vertices)`, each of shape `(num_pairs, dim)`:
                anchors: Feasible points which project their paired vertex.
                vertices: Polyblock vertices to project onto their paired anchor.
        """

    @abstractmethod
    def set_min_obj(self, obj: float) -> None:
        """
        Update minimum objective value of future candidate solutions, discarding those which fall below it.

        The polyblock only has to cover feasible points which beat the incumbent by the optimality tolerance, so a vertex falling below `obj` may be dropped: its objective bounds the whole box below it.
        Called only when the incumbent improves, so `obj` increases monotonically over a solve.

        Args:
            obj: Objective cut-off for retained vertices.
        """

    @abstractmethod
    def new_vertices(self, proj: NDArray, delta: float) -> NDArray:
        """
        Returns refined polyblock vertices to be checked for feasibility.

        Cuts the cones above `proj` out of the polyblock, returning the vertices which replace those they remove.
        A vertex lying above a projection is refined by reducing each of its components in turn to the matching component of that projection, giving up to `dim` replacements.
        The choice of which vertices are refined, a vertex lying above several projections is handled is left to the implementation.
        The implementation decides which vertices are refined, the projections used to refine them, and how redundancy checking is handled.

        New vertiecs are internally retained, since `update` reports feasibility against it positionally.

        Args:
            proj: Monotone projections of this iteration's vertices, of shape `(num_pairs, dim)`.
            delta: Erosion factor, giving the margin within which a vertex need not be refined for partial refinements.

        Returns:
            Candidate vertices of shape `(num_new, dim)`, pending the feasibility check in `update`.
            All vertices should lie within the feasible box.
        """

    @abstractmethod
    def update(self, new_mask: NDArray[np.bool], new_obj: NDArray) -> bool:
        """
        Updates internal polyblock representation after checking new vertex feasibility.

        `new_mask` indexes the array returned by `new_vertices` in the same order, while `new_obj` holds objective values for the masked entries alone.
        Masked-out vertices either violate the co-normal constraints or fall below the objective cut-off, and should be discarded rather than stored.

        Args:
            new_mask: Refined vertex feasibility, of shape `(num_new,)`.
            new_obj: Objective values of feasible refined vertices, of shape `(new_mask.sum(),)`.

        Returns:
            True once the vertex set is exhausted, which terminates `solve` and certifies the incumbent as optimal, or the problem as infeasible if no candidate was ever found.
        """

    @property
    @abstractmethod
    def size(self) -> int:
        """
        Size of the container used to store polyblocks. Used for limiting memory usage.

        Polled every iteration, and `solve` gives up once it exceeds `POLYBLOCK_LIMIT`.
        Units are left to the implementation, as the limit is only ever compared against this property.
        """

    @property
    def best_bound(self) -> float:
        """
        Best upper-bound on optimal objective value (optional).

        Defaults to `nan` for implementations which do not track a bound.
        This property is only used for reporting purposes and does not affect the solve.
        """
        return np.nan

    @classmethod
    def solve(
        cls,
        obj: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        x_l: ArrayLike,
        x_u: ArrayLike,
        ub_oracle: Callable[[NDArray[np.floating]], NDArray[np.bool]],
        lb_oracle: Callable[[NDArray[np.floating]], NDArray[np.bool]] | None = None,
        eps_obj_abs: float = 1e-6,
        eps_obj_rel: float = 1e-2,
        eps_ls: float = 1e-3,
        delta: float = 1e-3,
        verbose_gap: int | None = None,
        time_limit: int = 3600,
        iteration_limit: int = int(1e8),
    ) -> Solution:
        """
        A template for the Polyblock Outer-approximation algorithm.

        Maximises `obj` over the points of `[x_l, x_u]` accepted by both oracles.
        The objective must be non-decreasing in every coordinate, while `ub_oracle` and `lb_oracle` should be boolean-valued functions querying upper- and lower-bound constraints, respectively.
        All three functions are called with 2D arrays batching points along the first axis.

        A returned solution is always feasible, and no feasible point beats it by more than the objective tolerances, except possibly points lying within `delta` of the upper-bound constraint boundary.

        Args:
            obj: Non-decreasing objective function.
            x_l: Lower point defining feasible rectangle [x_l, x_u], must broadcast with `x_u` to a valid input for the objective and oracles.
            x_u: Upper point defining feasible rectangle [x_l, x_u], must broadcast with `x_l` to a valid input for the objective and oracles.
            ub_oracle: Oracle querying upper-bound (normal) constraints.
            lb_oracle: Oracle querying lower-bound (co-normal) constraints (optional).
            eps_obj_abs: Absolute tolerance between incumbent objective and upper-bound.
            eps_obj_rel: Relative tolerance between incumbent objective and upper-bound.
            eps_ls: Line-search tolerance.
            delta: Optimality relaxation for upper-bound constraints. The objective is optimised only over the `delta` eroded normal set.
            verbose_gap: Number of iterations between printout (optional).
            iteration_limit: Maximum number of POA iterations before the solver gives up.
            time_limit: Maximum solver runtime in seconds.

        Returns:
            The best solution found, see `Solution`. Note that `success` reports whether the solver terminated on its own certificate rather than a limit, so it is also set when the problem is proven infeasible.

        Raises:
            ValueError: If `x_u` is smaller than `x_l` in any coordinate.
        """

        lb_exists = lb_oracle is not None
        verbose = verbose_gap is not None

        ## returned solution
        sol = Solution()

        ## initial checks
        x_l, x_u = np.broadcast_arrays(x_l, x_u)
        if (x_u - x_l < 0).any():
            raise ValueError("`x_u` must be no smaller than `x_l` element-wise.")
        elif not ub_oracle(x_l[None]):
            sol.success = True
            sol.status = "Problem Infeasible since `x_l` not in normal set."
            if verbose:
                print(sol.status)
            return sol

        ## tighten initial box
        x_u = tighten(x_l, x_u, oracle=ub_oracle)
        x_l = tighten(x_u, x_l, oracle=lb_oracle) if lb_exists else x_l.copy()

        mono_proj = partial(monotone_proj, oracle=ub_oracle, eps=eps_ls, delta=delta)
        polyblock = cls(x_l, x_u)
        min_obj = -np.inf

        ## main loop
        i = -1
        start_time = perf_counter()
        for i in range(iteration_limit):
            ## compute projections from lower and upper points
            lower_data, upper_data = polyblock.projection_pairs()
            proj = mono_proj(lower_data, upper_data)
            candidates = (proj + delta).clip(max=x_u)

            ## find feasible candidates
            feas_mask = (candidates >= x_l).all(-1)
            if lb_exists:
                feas_mask &= lb_oracle(candidates)

            ## update best solution
            if feas_mask.any():
                cand_feas = candidates[feas_mask]
                cand_obj = obj(cand_feas)
                best_cand = cand_obj.argmax()
                best_obj_cand = cand_obj[best_cand]
                if best_obj_cand > sol.obj:
                    sol.obj = best_obj_cand.item()
                    sol.x = cand_feas[best_cand].copy()
                    min_obj = eps_obj_abs + sol.obj * (eps_obj_rel + 1)
                    polyblock.set_min_obj(min_obj)

            ## update polyblock representation
            new = polyblock.new_vertices(proj, delta=delta)
            new_obj = obj(new).flatten()
            new_mask = new_obj >= min_obj
            if lb_exists:
                new_mask &= lb_oracle(new)
            new_obj = new_obj[new_mask]
            empty = polyblock.update(new_mask, new_obj)

            ## check termination
            if empty:
                sol.success = True
                if sol.x is not None:
                    sol.status = "Optimal!"
                else:
                    sol.status = "Infeasible under current relaxation."
                break
            elif polyblock.size > cls.POLYBLOCK_LIMIT:
                sol.status = "Maximum polyblock size exceeded."
                break
            elif perf_counter() - start_time > time_limit:
                sol.status = "Time limit exceeded."
                break

            if verbose and i % verbose_gap == 0:
                header = i % (verbose_gap * 50) == 0
                print_row(
                    header=header,
                    iter=i,
                    obj=sol.obj,
                    best_bound=polyblock.best_bound,
                    poly_size=polyblock.size,
                    dist_to_boundary=(upper_data - proj).mean().item(),
                )

        sol.n_iter = i + 1
        sol.best_bound = polyblock.best_bound
        if verbose:
            print(
                f"{sol.status}, "
                f"itr: {sol.n_iter}, "
                f"Best solution: {sol.x}, "
                f"Obj: {sol.obj:.3f}"
            )

        return sol
