import pytest
from numpy import allclose, array

from polyblocks import BalancedPOA, BasePOA, TreePOA


def test_1d():
    """
    A trivial 1D problem, solved in a single POA iteration:
        max     x
        s.t.    x <= 0.5
                0 <= x <= 1
    """

    sol = TreePOA.solve(
        obj=lambda x: x[:, 0],
        ub_oracle=lambda x: x[:, 0] <= 0.5,
        x_l=(0.0,),
        x_u=(1.0,),
    )

    assert sol.success
    assert sol.x is not None
    assert allclose(sol.x, (0.5), atol=0.01)


def test_manysolutions():
    """
    This example demonstrates that POA always returns solutions on the (eroded) boundary of the feasible set:
        max     x
        s.t.    x + y <= 1.5
                0 <= x <= 1
                0 <= y <= 1
    """

    sol = TreePOA.solve(
        obj=lambda x: x[:, 0],
        ub_oracle=lambda x: x.sum(1) <= 1.5,
        x_l=(0.0, 0.0),
        x_u=(1.0, 1.0),
    )

    assert sol.success
    assert sol.x is not None
    assert allclose(sol.x, (1.0, 0.5), atol=0.01)


def test_simple0():
    """
    A minimal example which solves the monotone problem:
        max     x^2 + y^2
        s.t.    x + y <= 1
                0 <= x <= 1
                0 <= y <= 0.5

    """

    def obj(x):
        return (x**2).sum(1)

    def ub_oracle(x):
        return x.sum(1) <= 1.0

    sol_tree = TreePOA.solve(
        obj=obj,
        ub_oracle=ub_oracle,
        x_l=(0.0, 0.0),
        x_u=(1.0, 0.5),
    )

    assert sol_tree.success
    assert sol_tree.x is not None
    assert allclose(sol_tree.x, (1.0, 0.0), atol=0.01)


@pytest.mark.parametrize("solver", (TreePOA, BasePOA, BalancedPOA))
def test_simple1(solver):
    """
    Every solver must reach the same optimum of a problem with lower-bound constraints:
        max     2x + y
        s.t.    x + y <= 1
                xy >= 0.16
                0 <= x <= 1
                0 <= y <= 1

    Optimum at `(0.8, 0.2)` with value `1.8`.
    """

    def obj(x):
        return 2 * x[:, 0] + x[:, 1]

    def ub_oracle(x):
        return x.sum(1) <= 1.0

    def lb_oracle(x):
        return x.prod(1) >= 0.16

    sol = solver.solve(
        obj=obj,
        ub_oracle=ub_oracle,
        lb_oracle=lb_oracle,
        x_l=(0.0, 0.0),
        x_u=(1.0, 1.0),
    )

    assert sol.success
    assert sol.x is not None
    assert allclose(sol.x, (0.8, 0.2), atol=0.01)
    assert allclose(sol.obj, 1.8, atol=0.01)


@pytest.mark.parametrize("solver", (TreePOA, BasePOA, BalancedPOA))
def test_solvers_agree(solver):
    """
    Every solver must reach the same optimum of a linear objective over a sphere:
        max     2x + 3y + z
        s.t.    x^2 + y^2 + z^2 <= 1
                0 <= x <= 1
                0 <= y <= 1
                0 <= z <= 1

    Optimum at `c / ||c||` where `c = (2, 3, 1)` are the linear coefficients.
    """

    c = array((2.0, 3.0, 1.0))

    def obj(x):
        return x @ c

    def ub_oracle(x):
        return (x**2).sum(1) <= 1.0

    eps_rel = 1e-2
    eps_abs = 1e-6
    sol = solver.solve(
        obj=obj,
        ub_oracle=ub_oracle,
        x_l=(0.0, 0.0, 0.0),
        x_u=(1.0, 1.0, 1.0),
        eps_obj_rel=eps_rel,
        eps_obj_abs=eps_abs,
    )

    x_opt = c / 14**0.5

    assert sol.success
    assert sol.x is not None
    assert allclose(sol.obj, obj(x_opt), atol=0.01)
    assert allclose(sol.x, x_opt, atol=0.02)

    no_bound = sol.best_bound != sol.best_bound
    assert no_bound or sol.best_bound - sol.obj < eps_rel * sol.obj + eps_abs
