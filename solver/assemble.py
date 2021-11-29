import numpy as np
from scipy import sparse


def assemble(m, neighboors):
    m = np.repeat(m[np.newaxis, :, :], len(neighboors), 0)
    return _assemble(m, neighboors)


def _assemble(ms, neighboors):
    _, k = neighboors.shape
    i = np.stack(
        [np.repeat(neighboors, k, axis=1).ravel(),
         np.tile(neighboors, k).ravel()]
    )
    ms = ms.reshape(-1)
    return sparse.coo_matrix((ms, i)).tocsr()
