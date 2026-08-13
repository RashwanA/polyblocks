import numpy as np

from .abstract import ABPolyblock
from .containers import Tree


class TreePOA(ABPolyblock):
    """
    An implementation of POA which stores polyblock vertices in a tree.

    Rather than storing the current vertices `V` alone, this solver keeps the whole tree of refinements the search has produced, with `V` as its leaves.
    The representation is compressed: a cut replaces a vertex by copies differing from it in one component, so a node need only record the component its cut modified and that component's new value, with the root holding the initial upper point.
    A vertex is recovered by descending from the root and tracing these modifications at each node.

    In return, each node bounds both the region and objective of its subtree, so a search can rule out the whole subtree without visiting its leaves. POA's searches over `V` thus become descents rather than scans:

    - `projection_pairs` finds the maximal vertex by descending through the highest-objective child at each
      step, using descents diverted into lower-ranked children to return up to `PROJECTED_VERTICES`
      distinct leaves
    - `new_vertices` collects the vertices above a projection by recursing from the root, entering a child
      only when its component clears the projection and its objective attribute clears the incumbent.
    - `best_bound` is read off the root, whose objective attribute is the maximum over all vertices.

    Under a positive `delta` the depth of the tree is bounded independently of how many vertices it holds, so a descent costs `O(dim * depth)` against the `O(len(V))` of a flat array.

    Pruned nodes are not removed eagerly, as repairing every index is expensive for large trees.
    Instead, their objective attributes are used to keep them out of both tree traversals for free.
    `update` rebuilds the tree every `REBUILD_GAP` updates, removing childless nodes and those which can no longer improve the incumbent.

    Attributes:
        REBUILD_GAP: Number of updates between tree rebuilds, which prunes redundant nodes.
        PROJECTED_VERTICES: Maximum number of vertices projected per iteration.
    """

    REBUILD_GAP: int = 10000
    PROJECTED_VERTICES: int = 8

    def __init__(self, lower, upper) -> None:
        self.lower = lower
        self.tree = Tree(upper)
        self.best_obj = -np.inf
        self.clean_counter = 0

        self.expanded_idx: np.ndarray
        self.vect: np.ndarray
        self.comp: np.ndarray
        self.cval: np.ndarray

    def projection_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        vertices = self.tree.find_best(self.PROJECTED_VERTICES)
        anchors = vertices - (vertices - self.lower).max(-1, keepdims=True)
        return anchors, vertices

    def set_min_obj(self, obj) -> None:
        self.best_obj = obj

    def new_vertices(self, proj, delta) -> np.ndarray:
        ## query range using tree search
        (expanded_data, self.expanded_idx, vect, comp, cval) = self.tree.query(
            proj, self.lower, min_obj=self.best_obj, delta=delta
        )

        ## compute new vertices
        new_data = expanded_data[vect]
        np.put_along_axis(new_data, comp[:, None], cval[:, None], axis=1)

        self.vect, self.comp = vect, comp
        self.cval = cval
        return new_data

    def update(self, new_mask, new_obj) -> bool:
        vect, comp = self.vect[new_mask], self.comp[new_mask]
        cval = self.cval[new_mask]
        self.tree.add(self.expanded_idx, vect, comp, cval, new_obj)

        ## rebuild tree periodically
        if self.clean_counter == self.REBUILD_GAP:
            self.tree.rebuild(self.best_obj)
            self.clean_counter = 0
        else:
            self.clean_counter += 1

        return self.best_bound <= self.best_obj

    @property
    def size(self):
        return self.tree.cvo.length

    @property
    def best_bound(self) -> float:
        return self.tree.cvo[0]["obj"].item()
