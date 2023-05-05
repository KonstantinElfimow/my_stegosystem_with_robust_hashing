import numpy as np


def my_hamming(h_1: np.uint64, h_2: np.uint64) -> np.uint8:
    d: np.uint8 = np.uint8(0)
    h: np.uint64 = np.bitwise_xor(h_1, h_2)
    while h:
        h = np.bitwise_and(np.uint64(h), np.uint64(h - 1))
        d += 1
    return d
