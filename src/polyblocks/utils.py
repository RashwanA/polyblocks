from collections.abc import Callable

import numpy as np
from numerize.numerize import numerize
from numpy.linalg import norm
from numpy.typing import NDArray


def monotone_proj(
    vertices: NDArray,
    anchors: NDArray,
    oracle: Callable,
    eps=1e-4,
    delta=0.0,
) -> NDArray:
    """
    Perform a vectorised bisection search to compute monotone projections onto the `delta`-eroded normal set corresponding to the given oracle.

    Args:
        vertices: Vertices to project using paired anchors, of shape `(num_pairs, dim)`.
        anchors: Feasible anchor points for projecting paired vertex, of shape `(num_pairs, dim)`. Shifted by `delta` to account for erosion.
        oracle: An oracle for querying normal set feasibility.
        eps: The numerical tolerance for the line search.
        delta: The erosion factor for the normal set.

    Returns:
        Batched monotone projections of shape `(num_pairs, dim)`.
    """

    ## resolve erosion
    vert_offset = vertices + delta
    anchors = anchors - delta

    ## scale epsilon
    diff = anchors - vertices
    eps = eps / norm(diff, ord=2, axis=-1, keepdims=True)

    ub = np.ones_like(eps)
    lb = -eps
    x = np.full_like(eps, 0.5)

    while (ub - lb > eps).any():
        mask = oracle(diff * x + vert_offset)
        n_mask = ~mask
        ub[n_mask] = x[n_mask]
        lb[mask] = x[mask]
        x = (ub + lb) / 2

    return lb * diff + vertices


def tighten(root: NDArray, reduced: NDArray, oracle: Callable, eps=1e-4) -> NDArray:
    """Tighten the box anchored at `root` using a monotone oracle."""

    reduced = reduced.copy()
    for ind in range(reduced.shape[-1]):
        c_end = root.copy()
        c_end[ind] = reduced[ind]
        reduced[ind] = monotone_proj(root[None], c_end[None], oracle, eps=eps)[0, ind]
    return reduced


def _center(s: str, width: int) -> str:
    pad = max(width - len(s), 0)
    left = pad - pad // 2
    return " " * left + s + " " * (pad - left)


def print_row(*, header: bool = True, **columns):
    """
    Print a row of a progress table, drawing a header/rule above it if `header` is set.

    Args:
        header: Whether to (re)print the column header and box rules before the row.
        **columns: Column name -> value. Column names are converted to title case for display,
            and the column width is derived from the label length so header and rows always align.
    """

    labels = {name: name.replace("_", " ").title() for name in columns}
    widths = {name: max(len(label) + 2, 8) for name, label in labels.items()}

    if header:
        print("┌" + "┬".join("─" * w for w in widths.values()) + "┐")
        print(
            "│"
            + "│".join(_center(label, widths[name]) for name, label in labels.items())
            + "│"
        )
        print("├" + "┼".join("─" * w for w in widths.values()) + "┤")

    cells = []
    for name, value in columns.items():
        if isinstance(value, int):
            value = f"{numerize(value, decimals=2)}"
        elif isinstance(value, float):
            value = f"{value:#.4g}"
        else:
            value = str(value)

        cells.append(_center(value, widths[name]))

    print("│" + "│".join(cells) + "│")
