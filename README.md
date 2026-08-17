Polyblocks
=============

This package provides a set of solvers for monotonic optimisation problems, which take the general form:

$$
\begin{aligned}
\max_{x \in \mathbb{R}^n} \quad & f(x) \\
\text{s.t.} \ \quad & x_l \le x \le x_u, \\
& g(x) \le 0, \\
& h(x) \ge 0,
\end{aligned}
$$

where $f$, $g$, and $h$ are non-decreasing in each coordinate.
This covers a broad class of non-convex problems, including polynomial programming along with many radio resource allocation problems in communications.

Solvers are built around the **Polyblock Outer-approximation (POA)** algorithm: a branch-and-bound algorithm which iteratively refines a rectangular outer-approximation of the solution space.
Full details are available in the [accompanying paper](https://arxiv.org/pdf/2608.13694).

## Installation

```bash
pip install polyblocks
```

or, for local development:

```bash
git clone https://github.com/RashwanA/polyblocks.git
cd polyblocks
pip install .
```

## Solvers

The package includes three built-in solvers: `BasePOA`, `BalancedPOA`, and `TreePOA`.
Of these `TreePOA` typically has the best empirical performance as it uses an efficient tree-based representation of the solution space.
All solvers implement the common `ABPolyblock` interface, which exposes a set of subroutines used by the POA algorithm. See the [interface documentation](https://rashwana.github.io/polyblocks/interface/) for details.
Users may also define custom solvers by implementing this interface.

## Usage

Each solver is called via its `solve` classmethod, which takes the objective, the bounding box, and oracles for querying feasibility.

Ball optimisation example:

```python
from polyblocks import TreePOA

# maximise 3x + 4y
# subject to x^2 + y^2 <= 1

def obj(x):
    return x @ [3, 4]

def ub_oracle(x):
    return (x ** 2).sum(-1) <= 1

sol = TreePOA.solve(
    obj=obj,
    ub_oracle=ub_oracle,
    x_l=(0., 0.),
    x_u=(1., 1.),
)

# solution: x=0.6, y=0.8
print(f"status={sol.status}, obj={sol.obj:.2f}, x={sol.x.round(2)}")
```

Geometric programming example:

```python
from polyblocks import TreePOA

# maximise x*y*z
# subject to x*(y + z) <= 4,
#            y*z <= 4,
#            x, y, z ∈ [0, 4]

def obj(x):
    return x.prod(1)

def ub_oracle(x):
    cons1 = x[:,0] * (x[:,1:].sum(1)) <= 4.
    cons2 = x[:,1:].prod(1) <= 4.
    return cons1 * cons2

# solution: x=1, y=2, z=2
sol = TreePOA.solve(
    obj=obj,
    ub_oracle=ub_oracle,
    x_l=(0., 0., 0.),
    x_u=4.,             # bounds are broadcast
    verbose_gap=50,     # print progress every 50 iterations
)
```

Objectives and oracles are expected to accept batched inputs of shape `(num_points, dim)`.
