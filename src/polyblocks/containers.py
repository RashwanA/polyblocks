import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .jit_funcs import delete, find_best, query_multi, rebuild, update_obj


class DynamicArray:
    """
    A 2D numpy array with over-allocated memory to improve append performance along first axis.

    Data range is not checked when indexing array.
    """

    __slots__ = ("data", "length")

    def __init__(self, dim=1, dtype: DTypeLike = float, start_sz=100):
        """Initialise dynamic container consisting of `dim` dimensional vectors of type `dtype`"""

        size = (start_sz,) if dim == 1 else (start_sz, dim)
        self.data = np.empty(size, dtype=dtype)
        self.length = 0

    def append(self, new: ArrayLike) -> None:
        """Append vectors in `new` to the end of the array."""

        data = self.data
        row_shape = data.shape[1:]
        new = np.asarray(new, dtype=data.dtype).reshape(-1, *row_shape)

        max_size = data.shape[0]
        length = self.length
        new_len = length + new.shape[0]
        if new_len > max_size:
            new_size = max(max_size * 2, new_len)
            new_data = np.empty((new_size, *row_shape), dtype=data.dtype)
            new_data[:length] = data[:length]
            self.data = data = new_data
        data[length:new_len] = new
        self.length = new_len

    @property
    def array(self):
        return self.data[: self.length]

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, val):
        self.data[index] = val

    def delete(self, removed_idx: NDArray[np.intp]) -> None:
        """Deletes elements at `removed_idx` by shifting remaining elements down."""

        delete(self.array, removed_idx)
        self.length -= removed_idx.shape[0]


class Tree:
    """Maintains a tree-representation of polyblock vertices."""

    IDX_TYPE = np.int32
    COMPONENT_TYPE = np.int8

    def __init__(self, first: NDArray[np.float32 | np.float64]):
        """Initialise tree using the root node vertex."""

        self.first = first

        ## tree representation
        float_type = first.dtype
        cvo_type = np.dtype(
            [("comp", self.COMPONENT_TYPE), ("value", float_type), ("obj", float_type)]
        )
        self.cvo = DynamicArray(dim=1, dtype=cvo_type)
        self.idx_range = DynamicArray(dim=2, dtype=self.IDX_TYPE)
        self.parent = DynamicArray(dim=1, dtype=self.IDX_TYPE)

        ## add first points
        self.idx_range.append([-1, -1])
        self.cvo.append((-1, -1, np.inf))
        self.parent.append(-1)

    def query(self, x: NDArray, lower: NDArray, min_obj=-np.inf, delta=1e-3) -> tuple:
        """
        Find and expand all leaf nodes which lie in the upper orthant of points `x`.

        If a leaf node lies in more than one orthant, it is expanded using only the first valid point in `x`.

        Args:
            x: Array of shape `(num_points, dim)` containing points to query.
            lower: Component-wise lower-bounds on tree vertices.
            min_obj: Lower-bound on leaf objective permitted.
            delta: Minimum distance from `x` required to expand a leaf.

        Returns:
            A tuple `(values, indices, vect, comp, cval)`:
                values: Expanded leaf values.
                indices: Indices of expanded leaves in the tree.
                vect: Indices into `values` giving the parent leaf of each new node.
                comp: Reduced component of each new node.
                cval: New component values.
        """

        return query_multi(
            x,
            self.cvo.array,
            self.idx_range.array,
            self.first,
            lower,
            min_obj=min_obj,
            delta=delta,
        )

    def find_best(self, num=1) -> NDArray:
        """Find up to `num` different leaf node values, the first of which has the best objective."""

        return find_best(self.cvo.array, self.idx_range.array, self.first, num=num)

    def add(
        self,
        expanded: NDArray[np.intp],
        exp_idx: NDArray[np.intp],
        comp_idx: NDArray[np.intp],
        comp_vals: NDArray[np.float32 | np.float64],
        new_obj: NDArray[np.float32 | np.float64],
    ) -> None:
        """
        Update internal tree representation by expanding leaf nodes.

        Args:
            expanded: Indices of expanded nodes.
            exp_idx: Indices in `expanded` for parents of new nodes.
            comp_idx: New expanded components.
            comp_vals: New expanded component values.
            new_obj: New objective values

        """

        cvo = self.cvo
        parents = self.parent
        idx_range = self.idx_range

        ## collect new data
        new_cvo = np.empty(comp_vals.shape[0], dtype=self.cvo.dtype)
        new_cvo["comp"] = comp_idx
        new_cvo["value"] = comp_vals
        new_cvo["obj"] = new_obj

        ## update parent neighbour ranges
        num_expand = np.bincount(exp_idx, minlength=expanded.shape[0])
        cumsum_expand = np.cumulative_sum(num_expand, include_initial=True)
        cs_offset = cumsum_expand + cvo.length
        par_ranges = np.column_stack((cs_offset[:-1], cs_offset[1:]))
        idx_range[expanded] = par_ranges

        ## append to tree
        cvo.append(new_cvo)
        parent_idx = expanded[exp_idx]  # exp_idx is assumed sorted
        parents.append(parent_idx)
        added_len = exp_idx.shape[0]
        idx_range.append(np.full((added_len, 2), -1, dtype=idx_range.dtype))

        ## backtrack obj values
        update_obj(expanded, parents.array, cvo.array, idx_range.array)

    def rebuild(self, min_obj: float) -> None:
        """Remove childless nodes and nodes with objectives below `min_obj`."""

        parent = self.parent
        idx_range = self.idx_range
        cvo = self.cvo

        n_removed = rebuild(cvo.array, idx_range.array, parent.array, min_obj)
        for arr in (cvo, parent, idx_range):
            arr.length -= n_removed
