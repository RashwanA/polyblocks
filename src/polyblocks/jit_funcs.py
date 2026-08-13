"""A collection of functions which are jit compiled using numba."""

import numpy as np
from numba import njit, prange


@njit(parallel=True, nogil=True, cache=True)
def rebuild(cvo, idx_range, parents, min_obj) -> int:
    """
    Remove nodes whose objective falls to `min_obj` or below, and repair the indices of those remaining.

    Surviving nodes are shifted down in-place, so the caller must reduce its own record of the array lengths by the returned count.

    Args:
        cvo: Node data of shape `(num_nodes,)` with `comp`, `value` and `obj` fields. Modified in-place.
        idx_range: Child index ranges of shape `(num_nodes, 2)`. Modified in-place.
        parents: Parent node indices of shape `(num_nodes,)`. Modified in-place.
        min_obj: Nodes whose objective does not exceed this value are removed.

    Returns:
        Number of nodes removed.
    """

    ## find indices to remove
    o = cvo["obj"]
    removed_idx = (o <= min_obj).nonzero()[0]

    ## prune tree
    delete(cvo, removed_idx)
    delete(parents, removed_idx)
    delete(idx_range, removed_idx)

    ## shift index values down
    for i in prange(cvo.shape[0] - removed_idx.shape[0]):
        p_offset = np.searchsorted(removed_idx, parents[i])
        parents[i] -= p_offset

        st, end = idx_range[i]
        if st == end:
            continue
        else:
            st_offset = np.searchsorted(removed_idx, st)
            end_offset = np.searchsorted(removed_idx[st_offset:], end)
        idx_range[i] = st - st_offset, end - end_offset - st_offset

    return removed_idx.shape[0]


@njit(nogil=True, cache=True)
def query(x, cvo, idx_range, first, min_obj=-np.inf):
    """
    Query polyblock tree for all vertices `v` such that `v >= x` and `obj[v] >= min_obj`.

    Args:
        x: Query point of shape `(dim,)`.
        cvo: Node data with `comp`, `value` and `obj` fields.
        idx_range: Child index ranges of shape `(num_nodes, 2)`.
        first: Root vertex value of shape `(dim,)`.
        min_obj: Subtrees whose objective falls below this value are not descended into.

    Returns:
        A tuple `(values, indices)`:
            values: Matching leaf vertices of shape `(num_leaves, dim)`.
            indices: Indices of those leaves in `cvo`, of shape `(num_leaves,)`.
    """

    idx_type = idx_range.dtype.type
    node_stack = [
        (idx_type(0), first),
    ]
    leaf_idx = []
    leaf_values = []

    while node_stack:
        ## search top node
        node_idx, node_val = node_stack.pop()
        st, end = idx_range[node_idx]

        if st == -1:
            node_idx = idx_range.dtype.type(node_idx)
            leaf_idx.append(node_idx)
            leaf_values.append(node_val)

        for i in range(st, end):
            ci = cvo[i]
            if ci["obj"] >= min_obj and x[ci["comp"]] <= ci["value"]:
                child_val = node_val.copy()
                child_val[ci["comp"]] = ci["value"]
                node_stack.append((i, child_val))

    ## collect leaf values
    l_idx = np.array(leaf_idx, dtype=idx_type)
    n_leaves = l_idx.shape[0]
    l_vals = np.empty((n_leaves, x.shape[0]), dtype=x.dtype)
    for i in range(n_leaves):
        l_vals[i] = leaf_values[i]

    return l_vals, l_idx


@njit(parallel=True, nogil=True, cache=True)
def query_multi(x_batch, cvo, idx_range, first, lower, min_obj=-np.inf, delta=1e-3):
    """
    Query and refine a polyblock tree using a batch of points `x_batch`.

    Each point is queried independently, and the leaves it matches are refined along every component which yields a non-redundant vertex.
    A leaf matching more than one query point is refined against only the first of them.

    Args:
        x_batch: Query points of shape `(num_points, dim)`.
        cvo: Node data with `comp`, `value` and `obj` fields.
        idx_range: Child index ranges of shape `(num_nodes, 2)`.
        first: Root vertex value of shape `(dim,)`.
        lower: Component-wise lower-bounds on tree vertices, of shape `(dim,)`.
        min_obj: Subtrees whose objective falls below this value are not descended into.
        delta: Minimum separation from the query point required to refine a leaf.

    Returns:
        A tuple `(values, indices, vect, comp, cval)`:
            values: Refined leaf vertices of shape `(num_refined, dim)`.
            indices: Indices of those leaves in `cvo`, of shape `(num_refined,)`.
            vect: Index into `values` of the parent leaf of each new node.
            comp: Component reduced by each new node.
            cval: New value taken by that component.
    """

    b = x_batch.shape[0]
    idx_dtype = idx_range.dtype
    float_dtype = x_batch.dtype
    x_batch_delta = x_batch + delta

    indices = [np.empty(0, dtype=idx_dtype) for _ in range(b)]
    values = [np.empty((0, 0), dtype=float_dtype) for _ in range(b)]
    vects = [np.empty((0), dtype=np.int64) for _ in range(b)]
    comps = [np.empty((0), dtype=np.int64) for _ in range(b)]
    comp_val = [np.empty((0), dtype=float_dtype) for _ in range(b)]

    ## parallel queries and redundancy checks
    for i in prange(b):
        x = x_batch[i]
        x_delta = x_batch_delta[i]
        value, index = query(x, cvo, idx_range, first, min_obj)

        ## only explore vertices further than delta and break ties
        refine_mask = all_row(value > x_delta)
        if i > 0:
            for idx in range(value.shape[0]):
                if refine_mask[idx]:
                    v_idx = value[idx]
                    for j in range(i):
                        x_delta_j = x_batch_delta[j]
                        feas_j = (v_idx > x_delta_j).all()
                        if feas_j:
                            refine_mask[idx] = False
                            break

        idx_mask = x >= lower
        vect, comp = find_redundant(value, idx_mask, refine_mask)

        indices[i] = index[refine_mask]
        values[i] = value[refine_mask]
        vects[i] = vect
        comps[i] = comp
        comp_val[i] = x[comp]

    ## merge all data while shifting indices
    v_full = cat(values)
    ind_full = cat(indices)

    cumsum = len(indices[0])
    for i in range(1, b):
        vects[i] += cumsum
        cumsum += len(indices[i])

    vect_full = cat(vects)
    comp_full = cat(comps)
    cval_full = cat(comp_val)

    return v_full, ind_full, vect_full, comp_full, cval_full


@njit(nogil=True, cache=True)
def cat(list_of_arrays):
    """Concatenate list of arrays along first axis."""

    size = sum([arr.shape[0] for arr in list_of_arrays])
    ar0 = list_of_arrays[0]
    combined = np.empty((size,) + ar0.shape[1:], dtype=ar0.dtype)
    init_pos = 0
    for arr in list_of_arrays:
        combined[init_pos : init_pos + len(arr)] = arr
        init_pos += len(arr)
    return combined


@njit(nogil=True, cache=True, parallel=True, inline="always")
def all_row(arr):
    """Equivalent to `np.all(arr, axis=1)`"""
    rows = arr.shape[0]
    mask = np.empty(rows, dtype=np.bool)
    for i in prange(rows):
        mask[i] = arr[i].all()
    return mask


@njit(parallel=True, nogil=True, cache=True)
def find_best(cvo, idx_range, first, num=1):
    """
    Find up to `num` distinct leaf vertices, the first of which has the best objective.

    Attempts to find distinct leaves by performing `num` tree descents.
    Descent `i` carries an offset of `i` which diverts its path: at a node with `n` children it steps up to `n - 1` ranks below the best child, taking as many as the remaining offset allows and deducting them from it.
    Once the offset reaches zero the descent follows best children the rest of the way down.
    Descent `0` is never diverted and so reaches the best leaf, while larger offsets are spent as high in the tree as possible, giving paths that diverge earlier and hence distinct leaves.
    A descent `i` still holding offset when it reaches a leaf is discarded as its path is identical to that of a descent with a smaller index `j < i`, so fewer than `num` vertices may be returned.

    Args:
        cvo: Node data with `comp`, `value` and `obj` fields.
        idx_range: Child index ranges of shape `(num_nodes, 2)`.
        first: Root vertex value of shape `(dim,)`.
        num: Number of descents to attempt.

    Returns:
        Leaf vertices of shape `(num_found, dim)`, where `num_found <= num`.
        The first leaf has the best objective.
    """

    dim = first.shape[0]
    values = np.empty((num, dim), dtype=first.dtype)
    for i in range(num):
        values[i] = first

    curr_idx = np.zeros(num, dtype=idx_range.dtype)
    skipped_mask = np.zeros(num, dtype=np.bool)
    for i in prange(num):
        skip = np.int32(i)
        while True:
            st, end = idx_range[curr_idx[i]]
            if st == end:
                if skip == 0:
                    skipped_mask[i] = True
                break
            order = end - st - 1
            if skip > 0:
                less = min(skip, end - st - 1)
                order -= less
                skip -= less

            o = cvo[st:end]["obj"]
            chosen_child = np.argpartition(o, order)[order] + st
            child_cvo = cvo[chosen_child]
            values[i][child_cvo["comp"]] = child_cvo["value"]
            curr_idx[i] = chosen_child

    return values[skipped_mask]


@njit(parallel=True, nogil=True, cache=True)
def new_block(block, added, idx_mask, delta=1e-3):
    """
    Remove infeasible cone given by `added` from `block`.

    Args:
        block: Current polyblock vertices of shape `(num_vertices, dim)`.
        added: Vertex of the cone to remove, of shape `(dim,)`.
        idx_mask: Components eligible for reduction, of shape `(dim,)`.
        delta: Minimum separation from `added` required for a vertex to be cut.

    Returns:
        A tuple `(removed_idx, new_vertices)`:
            removed_idx: Indices of vertices in `block` refined by the cut.
            new_vertices: New vertices generated by the cut.
    """

    ## perform range query
    vertex_idx = []
    old_size, dims = block.shape
    for i in range(old_size):
        row = block[i]
        inside = True
        for d in range(dims):
            if added[d] > row[d]:
                inside = False
                break
        if inside:
            vertex_idx.append(i)

    vertex_idx = np.array(vertex_idx, dtype=np.int64)
    vertices = block[vertex_idx]
    removed_mask = all_row(vertices > added + delta)

    ## compute new vertices
    vect, comp = find_redundant(vertices, idx_mask, removed_mask)
    apen = vertices[removed_mask][vect]
    for i in prange(apen.shape[0]):
        c_comp = comp[i]
        apen[i, c_comp] = added[c_comp]

    return vertex_idx[removed_mask], apen


@njit(nogil=True, cache=True)
def delete(array, removed_idx):
    """Delete indices in-place by shifting remaining elements down. This function also sorts removed_idx."""

    ## check if removed_idx is sorted
    r_prev = -np.inf
    for r in removed_idx:
        if r >= r_prev:
            r_prev = r
        else:
            removed_idx.sort()
            break

    ## shift down
    n_removed = removed_idx.shape[0]
    old_size = array.shape[0]
    for down_shift in range(n_removed + 1):
        st = removed_idx[down_shift - 1] + 1 if down_shift > 0 else 0
        end = removed_idx[down_shift] if down_shift < n_removed else old_size
        for j in range(st, end):
            array[j - down_shift] = array[j]


@njit(parallel=True, nogil=True, inline="always", cache=True)
def find_redundant(arr, idx_mask, eps_mask=None):
    """
    Find the vertex refinements which are dominated by another vertex in `arr`.

    Reducing component `d` of vertex `arr[i]` is redundant when some other vertex `arr[j]` already dominates the result, which happens exactly when `arr[i]` exceeds `arr[j]` in dimension `d` alone.

    Args:
        arr: Candidate vertices of shape `(num_vertices, dim)`.
        idx_mask: Components eligible for reduction, of shape `(dim,)`.
        eps_mask: Optional mask of shape `(num_vertices,)` selecting the vertices to refine. All vertices are refined when omitted.

    Returns:
        A tuple `(vect, comp)` of equal length, holding one entry per non-redundant refinement:
            vect: Index of the vertex to refine, relative to the rows selected by `eps_mask`.
            comp: Component of that vertex to reduce.
    """

    if eps_mask is None:
        eps_idx = np.arange(arr.shape[0])
    else:
        eps_idx = eps_mask.nonzero()[0]

    n_exp = eps_idx.shape[0]
    dims = arr.shape[1]
    mask = np.empty((n_exp, dims), dtype=np.bool)
    for i in prange(n_exp):
        mask[i] = idx_mask

    for i in prange(n_exp):
        ai = arr[eps_idx[i]]
        for aj in arr:
            fail_i = -1
            nfail_i = 0

            for d in range(dims):
                dom_ij = ai[d] > aj[d]

                if dom_ij:
                    nfail_i += 1
                    fail_i = d
                    if nfail_i > 1:
                        break

            if nfail_i == 1:
                mask[i, fail_i] = False

    vect, comp = mask.nonzero()
    return vect, comp


@njit(nogil=True, cache=True)
def update_obj(expanded, parents, cvo, idx_range):
    """
    Update the objective attribute of polyblock tree by propagating the maximum objective value from children to parents.

    Args:
        expanded: Indices of the nodes whose children have changed.
        parents: Parent node indices of shape `(num_nodes,)`.
        cvo: Node data with `comp`, `value` and `obj` fields. The `obj` field is modified in-place.
        idx_range: Child index ranges of shape `(num_nodes, 2)`.
    """

    curr_layer = expanded
    while curr_layer.shape[0] > 0:
        layer_mask = np.zeros_like(curr_layer, dtype=np.bool)

        for i in range(curr_layer.shape[0]):
            ## get child data
            curr_idx = curr_layer[i]
            st, end = idx_range[curr_idx]

            ## update self if best obj changes
            best_obj = cvo[st:end]["obj"].max() if st != end else -np.inf
            if best_obj < cvo[curr_idx]["obj"]:
                cvo[curr_idx]["obj"] = best_obj
                if curr_idx > 0:
                    layer_mask[i] = True

        ## find unique set of parents
        curr_layer = parents[curr_layer[layer_mask]]
        curr_layer = np.unique(curr_layer)
