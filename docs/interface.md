# Interface

Every solver is a subclass of `ABPolyblock` and is called through its inherited `solve` classmethod, so the parameters and return value documented here apply to all of them.
The remaining methods are the sub-routines a solver implements, and are only of interest when writing a custom one.

::: polyblocks.ABPolyblock
    options:
      group_by_category: false
      members:
        - solve
        - projection_pairs
        - set_min_obj
        - new_vertices
        - update
        - size
        - best_bound

::: polyblocks.Solution

::: polyblocks.Status
