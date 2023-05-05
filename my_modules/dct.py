import numpy as np


def my_dct(grayMat: np.uint8) -> np.array:
    """ Двумерное дискретное косинусное преобразование """
    assert grayMat.shape == (8, 8)
    assert grayMat.dtype == np.uint8
    # Создаем матрицу коэффициентов
    coeffs = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i == 0:
                coeffs[i, j] = 1 / np.sqrt(8)
            else:
                coeffs[i, j] = np.sqrt(2 / 8) * np.cos((np.pi * (2 * j + 1) * i) / (2 * 8))

    # Вычисляем DCT для блока
    dct_block = np.dot(np.dot(coeffs, grayMat), coeffs.T)
    return dct_block
