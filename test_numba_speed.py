import time
t0 = time.time()
import numba
import numpy as np

try:
    @numba.njit
    def _test_empty():
        return np.empty((1,), dtype=np.bool_)
    _test_empty()
except Exception as e:
    pass
print(f"Time: {time.time()-t0}")
