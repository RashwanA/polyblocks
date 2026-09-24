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
    Query polyblock tree for all vertices `v` such that `v >= x` and `obj[v] > min_obj`.

    Args:
        x: Query point of shape `(dim,)`.
        cvo: Node data with `comp`, `value` and `obj` fields.
        idx_range: Child index ranges of shape `(num_nodes, 2)`.
        first: Root vertex value of shape `(dim,)`.
        min_obj: Subtrees whose objective does not exceed this value are not descended into.

    Returns:
        A tuple `(values, indices)`:
            values: Matching leaf vertices of shape `(num_leaves, dim)`.
            indices: Indices of those leaves in `cvo`, of shape `(num_leaves,)`.
    """

    idx_type = idx_range.dtype.type
    node_stack = [(idx_type(0), 0)]
    node_val = first.copy()
    undo = []
    leaf_idx = []
    leaf_values = []

    while node_stack:
        ## rewind `node_val` to the parent of the current node, then apply its modification
        node_idx, depth = node_stack.pop()
        while len(undo) >= depth > 0:
            comp, prev = undo.pop()
            node_val[comp] = prev
        if depth > 0:
            c = cvo[node_idx]["comp"]
            undo.append((c, node_val[c]))
            node_val[c] = cvo[node_idx]["value"]

        st, end = idx_range[node_idx]
        if st == -1:
            leaf_idx.append(node_idx)
            leaf_values.extend(node_val)

        for i in range(st, end):
            ci = cvo[i]
            if ci["obj"] > min_obj and x[ci["comp"]] <= ci["value"]:
                node_stack.append((idx_type(i), depth + 1))

    ## collect leaf values
    l_idx = np.array(leaf_idx, dtype=idx_type)
    l_vals = np.array(leaf_values, dtype=x.dtype).reshape(-1, x.shape[0])

    return l_vals, l_idx


@njit(parallel=True, nogil=True, cache=True)
def query_multi(x_batch, cvo, idx_range, first, lower, min_obj=-np.inf, delta=1e-3):
    """
    Query and refine a polyblock tree using a batch of points `x_batch`.

    Each point is queried independently, and the leaves it matches are refined along every component which yields a non-redundant vertex.
    A leaf matching more than one query point is refined against only the first whose cone contains it, and only if it clears that point by `delta`.

    Args:
        x_batch: Query points of shape `(num_points, dim)`.
        cvo: Node data with `comp`, `value` and `obj` fields.
        idx_range: Child index ranges of shape `(num_nodes, 2)`.
        first: Root vertex value of shape `(dim,)`.
        lower: Component-wise lower-bounds on tree vertices, of shape `(dim,)`.
        min_obj: Subtrees whose objective does not exceed this value are not descended into.
        delta: Minimum separation from the query point required to refine a leaf.

    Returns:
        A tuple `(values, indices, vect, comp, cval)`:
            values: Refined leaf vertices of shape `(num_refined, dim)`.
            indices: Indices of those leaves in `cvo`, of shape `(num_refined,)`.
            vect: Index into `values` of the parent leaf of each new node.
            comp: Component reduced by each new node.
            cval: New value taken by that component.
    """

    b, dim = x_batch.shape

    matched = [np.empty((0, 0), dtype=x_batch.dtype) for _ in range(b)]
    refined = [np.empty((0, 0), dtype=x_batch.dtype) for _ in range(b)]
    indices = [np.empty(0, dtype=idx_range.dtype) for _ in range(b)]
    projections = [np.empty(0, dtype=np.int64) for _ in range(b)]

    ## parallel queries
    for i in prange(b):
        value, index = query(x_batch[i], cvo, idx_range, first, min_obj)

        ## only explore vertices further than delta, while leaving those an earlier cone claims
        refine_mask = all_row(value > x_batch[i] + delta)
        for idx in range(value.shape[0]):
            if refine_mask[idx]:
                for j in range(i):
                    inside = True
                    for d in range(dim):
                        if value[idx, d] < x_batch[j, d]:
                            inside = False
                            break
                    if inside:
                        refine_mask[idx] = False
                        break

        matched[i] = value
        refined[i] = value[refine_mask]
        indices[i] = index[refine_mask]
        projections[i] = np.full(refined[i].shape[0], i, dtype=np.int64)

    ## perform redundancy checks in parallel for each matched leaf
    v_full = cat(refined)
    projection = cat(projections)

    mask = np.empty((v_full.shape[0], dim), dtype=np.bool)
    for t in prange(v_full.shape[0]):
        p = projection[t]
        for d in range(dim):
            mask[t, d] = x_batch[p, d] >= lower[d]
        mark_redundant(v_full[t], matched[p], mask[t])

    vect, comp = mask.nonzero()
    cval = np.empty(vect.shape[0], dtype=x_batch.dtype)
    for k in range(vect.shape[0]):
        cval[k] = x_batch[projection[vect[k]], comp[k]]

    return v_full, cat(indices), vect, comp, cval


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


@njit(nogil=True, cache=True)
def find_best(cvo, idx_range, first, min_obj, num=1):
    """
    Find up to `num` distinct leaf vertices, the first of which has the best objective.

    The tree is expanded level by level from the root until the frontier holds at least `num` subtrees that clear the `min_obj` threshold, or until no node can be expanded further.
    As the subtrees are disjoint their top leaves are distinct, and splitting as high in the tree as possible spreads them apart while still focusing high objective vertices.
    The frontier's best subtree contains the best overall leaf, returned first.
    Fewer than `num` vertices are returned when the tree holds fewer than `num` leaves clearing `min_obj`.

    Args:
        cvo: Node data with `comp`, `value` and `obj` fields.
        idx_range: Child index ranges of shape `(num_nodes, 2)`.
        first: Root vertex value of shape `(dim,)`.
        min_obj: Subtrees whose objective does not exceed this value are not descended into.
        num: Number of descents to attempt.

    Returns:
        Leaf vertices of shape `(num_found, dim)`, where `num_found <= num`.
        The first leaf has the best objective.
    """

    ## expand tree to find `num` distinct subtrees
    idx_type = idx_range.dtype.type
    frontier = [(idx_type(0), first.copy())]
    expand_flag = True
    while expand_flag and len(frontier) < num:
        expand_flag = False
        children = []
        for node, val in frontier:
            st, end = idx_range[node]
            if st == -1:
                children.append((node, val))
                continue
            for i in range(st, end):
                expand_flag = True
                ci = cvo[i]
                if ci["obj"] > min_obj:
                    child_val = val.copy()
                    child_val[ci["comp"]] = ci["value"]
                    children.append((idx_type(i), child_val))
        frontier = children

    ## descend to the best leaf of each of the best `num` subtrees
    frontier_obj = np.array([cvo[node]["obj"] for node, _ in frontier])
    chosen = np.argsort(-frontier_obj, kind="mergesort")[:num]
    values = np.empty((chosen.shape[0], first.shape[0]), dtype=first.dtype)
    for i, f in enumerate(chosen):
        node, val = frontier[f]
        values[i] = val
        st, end = idx_range[node]
        while st != -1:
            node = st + cvo[st:end]["obj"].argmax()
            values[i, cvo[node]["comp"]] = cvo[node]["value"]
            st, end = idx_range[node]

    return values


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
            removed_idx: Indices of vertices in `block` refined by the cut, in ascending order.
            new_vertices: New vertices generated by the cut.
    """

    n, dim = block.shape

    ## range query, marking the vertices which clear `added` by `delta`
    inside = np.empty(n, dtype=np.bool)
    cleared = np.empty(n, dtype=np.bool)
    for i in prange(n):
        inside[i], cleared[i] = cone_position(block, i, added, delta)

    vertex_idx = inside.nonzero()[0]
    vertices = block[vertex_idx]
    refined = cleared[vertex_idx].nonzero()[0]

    ## find redundant refinements
    mask = np.empty((refined.shape[0], dim), dtype=np.bool)
    mask[:] = idx_mask
    for t in prange(refined.shape[0]):
        mark_redundant(vertices[refined[t]], vertices, mask[t])

    ## apply refinements
    vect, comp = mask.nonzero()
    new_vertices = vertices[refined[vect]]
    for k in range(vect.shape[0]):
        new_vertices[k, comp[k]] = added[comp[k]]

    return vertex_idx[refined], new_vertices


@njit(nogil=True, inline="always", cache=True)
def cone_position(block, i, added, delta):
    """Whether `block[i] >= added`, and if so whether also `block[i] > added + delta`."""

    cleared = True
    for d in range(block.shape[1]):
        if block[i, d] < added[d]:
            return False, False
        cleared &= block[i, d] > added[d] + delta
    return True, cleared


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


@njit(nogil=True, inline="always", cache=True)
def mark_redundant(ai, arr, mask):
    """
    Mark components of `mask` along which refining vertex `ai` is redundant.

    Reducing component `d` of `ai` is redundant when some other vertex `aj` in `arr` already dominates the result, which happens exactly when `ai` exceeds `aj` in dimension `d` alone.

    Args:
        ai: Vertex to refine, of shape `(dim,)`.
        arr: Candidate vertices which may dominate the refinements, of shape `(num_vertices, dim)`.
        mask: Components eligible for reduction, of shape `(dim,)`. Modified in-place.
    """

    n_left = mask.sum()
    for j in range(arr.shape[0]):
        ## count components in which `ai` exceeds `arr[j]`
        n_exceed = 0
        exceed_d = 0
        for d in range(ai.shape[0]):
            exceeds = ai[d] > arr[j, d]
            n_exceed += exceeds
            exceed_d = d if exceeds else exceed_d

        if n_exceed == 1 and mask[exceed_d]:
            mask[exceed_d] = False
            n_left -= 1
            if n_left == 0:
                break


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
