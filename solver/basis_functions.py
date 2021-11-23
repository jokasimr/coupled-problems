from numba import jit, vectorize, guvectorize


@vectorize(
    ["float64(float64, float64, float64, float64)"], nopython=True, target="parallel"
)
def _hatv(_x, _y, dx, dy):
    " Computes value of canonical hat function at point x, y"
    x, y = _x / dx, _y / dy

    if not -1.0 <= (x + y) <= 1.0:
        return 0.0
    if not -1.0 <= x <= 1.0:
        return 0.0
    if not -1.0 <= y <= 1.0:
        return 0.0

    if x >= 0.0:
        if y >= 0.0:
            return 1.0 - y - x
        if x >= -y:
            return 1.0 - x
        else:
            return 1.0 + y
    else:
        if y <= 0.0:
            return 1.0 + y + x
        if x <= -y:
            return 1.0 + x
        else:
            return 1.0 - y


@vectorize(
    ["float64(float64, float64, float64, float64, float64, float64)"],
    nopython=True,
    target="parallel",
)
def _ghatv(_x, _y, nx, ny, dx, dy):
    " Computes gradient \dot (nx, ny) of a canonical hat function at point x, y"
    x, y = _x / dx, _y / dy

    if not -1.0 <= (x + y) <= 1.0:
        return 0.0
    if not -1.0 <= x <= 1.0:
        return 0.0
    if not -1.0 <= y <= 1.0:
        return 0.0

    if x >= 0.0:
        if y >= 0.0:
            return -ny / dy - nx / dx
        if x >= -y:
            return -nx / dx
        else:
            return ny / dy
    else:
        if y <= 0.0:
            return ny / dy + nx / dx
        if x <= -y:
            return nx / dx
        else:
            return -ny / dy



