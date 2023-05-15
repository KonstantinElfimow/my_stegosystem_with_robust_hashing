import numpy as np


def my_hamming(h_1, h_2) -> int:
    distance = 0
    h = np.bitwise_xor(h_1, h_2)
    while h:
        h = np.bitwise_and(h, h - 1)
        distance += 1
    return distance
