import numpy as np
from numpy.typing import NDArray

from .abstract import ABPolyblock
from .containers import DynamicArray
from .jit_funcs import new_block


class BasePOA(ABPolyblock):
    """A naive implementation of POA which directly stores polyblock vertices in dynamic arrays."""

    POLYBLOCK_LIMIT = 3 * int(1e6)
    RHO = 0.2

    def __init__(self, lower, upper):
        self.lower = lower
        const = self.RHO / (1 - self.RHO)
        self.lower_offset = lower - const * (upper - lower).max()
        self.lower_offset = self.lower_offset[None]
        self.new: NDArray

        self.vertices = DynamicArray(dim=lower.shape[0], dtype=lower.dtype)
        self.obj_vals = DynamicArray(dim=1, dtype=lower.dtype)
        self.vertices.append(upper)
        self.obj_vals.append(np.inf)

    def projection_pairs(self) -> tuple[NDArray, NDArray]:
        best_vtx = self.obj_vals.array.argmax()
        return self.lower_offset, self.vertices[None, best_vtx].copy()

    def set_min_obj(self, obj) -> None:
        removed_idx = (self.obj_vals.array < obj).nonzero()[0]
        self.vertices.delete(removed_idx)
        self.obj_vals.delete(removed_idx)

    def new_vertices(self, proj, delta) -> NDArray:
        proj = proj.flatten()
        idx_mask = proj >= self.lower
        vertices = self.vertices
        obj_vals = self.obj_vals

        removed_idx, self.new = new_block(vertices.array, proj, idx_mask, delta)
        vertices.delete(removed_idx)
        obj_vals.delete(removed_idx)
        return self.new

    def update(self, new_mask, new_obj) -> bool:
        new = self.new[new_mask]
        self.vertices.append(new)
        self.obj_vals.append(new_obj)
        return self.vertices.length == 0

    @property
    def size(self):
        return self.vertices.length


class BalancedPOA(BasePOA):
    """A variant of the naive POA implementation which uses balanced anchors."""

    def projection_pairs(self) -> tuple[NDArray, NDArray]:
        blk_best = self.obj_vals.array.argmax()
        best_upper = self.vertices[None, blk_best]
        best_lower = best_upper - (best_upper - self.lower).max()
        return best_lower, best_upper.copy()
