import numpy as np


def read_file(fileid, n_rows, n_cols, dtype):
    with open(fileid, "rb") as f:
        b = f.read()
        a_flat = np.frombuffer(b, dtype=dtype)
    a = a_flat.reshape((n_rows, n_cols))
    return a