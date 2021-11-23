import numpy as np
from numba import jit, vectorize, guvectorize


class Rectangle:

    def __init__(self, width, height, dx, dy):
        assert width / dx == round(width / dx), "`width` must be evenly divisible by `dx`"
        assert height / dx == round(height / dx), "`height` must be evenly divisible by `dx`"
        self.dofs_per_row = round(width / dx) + 1
        self.dofs_per_col = round(height / dy) + 1
        self.ndofs = self.dofs_per_row * self.dofs_per_col
        self.nelems = 2 * (self.dofs_per_row - 1) * (self.dofs_per_col - 1)
        self.dx = dx
        self.dy = dy
        self.width = width
        self.height = height

    @property
    def dofs(self):
        return np.arange(self.ndofs)

    @property
    def elems(self):
        return np.arange(self.nelems)

    def neighboors_interior(self, indexes):
        return self._neighboors_interior(
            indexes, self.dofs_per_row, np.ones((1, 3), np.int64))

    def neighboors_exterior(self, indexes):
        return self._neighboors_exterior(
            indexes, self.dofs_per_row, self.dofs_per_col, np.ones((1, 2), np.int64))

    def coords(self, indexes):
        x = self.dx * (indexes % self.dofs_per_row)
        y = self.dy * (indexes // self.dofs_per_row)
        return x, y

    def boundary_dofs(self, e):
        if e == 0:
            return np.arange(self.dofs_per_row)
        if e == 1:
            return np.arange(self.dofs_per_col) * self.dofs_per_row + (self.dofs_per_row - 1)
        if e == 2:
            total_dofs = self.dofs_per_col * self.dofs_per_row - 1
            return total_dofs - np.arange(self.dofs_per_row)
        if e == 3:
            return np.arange(self.dofs_per_col - 1, -1, -1) * self.dofs_per_row
        raise ValueError(f"boundary index must be in (0, 1, 2, 3)")

    def boundary_elems(self, e):
        elems_per_row = self.dofs_per_row - 1
        elems_per_col = self.dofs_per_col - 1
        if e == 0:
            return np.arange(elems_per_row)
        if e == 1:
            return np.arange(elems_per_col) + elems_per_row
        if e == 2:
            return np.arange(elems_per_row) + elems_per_col + elems_per_row
        if e == 3:
            return np.arange(elems_per_col) + elems_per_col + 2 * elems_per_row
        raise ValueError(f"boundary index must be in (0, 1, 2, 3)")

    @staticmethod
    @guvectorize(
       "void(int64[:], int64, int64[:], int64[:])",
       "(),(),(n)->(n)",
       target="parallel",
       nopython=True,
    )
    def _neighboors_interior(_element_index, dofs_per_row, _, n):
        element_index = _element_index[0]
        elems_per_row = 2 * (dofs_per_row - 1)
        row = element_index // elems_per_row
        dof_index_if_all_elements_on_one_row = element_index // 2

        if element_index % 2 == 0:
            n[0] = dof_index_if_all_elements_on_one_row + row
            n[1] = dof_index_if_all_elements_on_one_row + row + 1
            n[2] = dof_index_if_all_elements_on_one_row + row + dofs_per_row

        else:
            n[0] = dof_index_if_all_elements_on_one_row + row + dofs_per_row + 1
            n[1] = dof_index_if_all_elements_on_one_row + row + dofs_per_row
            n[2] = dof_index_if_all_elements_on_one_row + row + 1

    @staticmethod
    @guvectorize(
        "void(int64[:], int64, int64, int64[:], int64[:])",
        "(),(),(),(n)->(n)",
        target="parallel",
        nopython=True,
    )
    def _neighboors_exterior(_element_index, dofs_per_row, dofs_per_col, _, n):
        # N = dofs per row
        element_index = _element_index[0]
        elems_per_row = dofs_per_row - 1
        elems_per_col = dofs_per_col - 1

        if 0 <= element_index < elems_per_row:
            # bottom
            n[0] = element_index
            n[1] = element_index + 1

        elif elems_per_row <= element_index < elems_per_row + elems_per_col:
            # right
            row = element_index - elems_per_row
            n[0] = (row + 1) * dofs_per_row - 1
            n[1] = (row + 2) * dofs_per_row - 1

        elif (
            elems_per_row + elems_per_col
            <= element_index
            < elems_per_row * 2 + elems_per_col
        ):
            # top
            dof_max_index = dofs_per_row * dofs_per_col - 1
            column = element_index - elems_per_row - elems_per_col
            n[0] = dof_max_index - column
            n[1] = dof_max_index - column - 1

        else:
            # left
            row = elems_per_col - (element_index - 2 * elems_per_row - elems_per_col) - 1
            n[0] = (row + 1) * dofs_per_row
            n[1] = row * dofs_per_row
