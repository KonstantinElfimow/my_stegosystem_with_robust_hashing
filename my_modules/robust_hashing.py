import cv2
import numpy as np
from my_modules.pca import my_pca
from my_modules.dct import my_dct


def __phash(block: np.array) -> np.uint64:
    """ Перцептивное хэширование """
    assert block.shape == (8, 8) and block.dtype == np.uint8

    dct = my_dct(block)
    dct = dct.reshape(-1)
    x = (np.arange(64))[dct >= dct.mean()]
    h = np.uint64(np.sum(np.power(2, x)))
    return h


def improved_phash(image: np.array) -> np.uint64:
    # # Исходное изображение
    # io.imshow(image)
    # io.show()

    # Преобразуем в оттенки серого
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Разбиение на блоки размером 8x8
    blocks = []
    for i in range(8, img.shape[0], 8):
        for j in range(8, img.shape[1], 8):
            block = img[i - 8: i, j - 8: j]
            blocks.append(block)
    vectors = np.array(blocks, dtype=np.uint8).reshape(-1, 64)
    del blocks

    # Восстановленные блоки
    k = len(vectors) // 2
    reconstructed = np.array(my_pca(vectors, k), dtype=np.uint8).reshape(-1, 8, 8)

    # Восстановленное изображение
    reconstructed_image = np.zeros(img.shape, dtype=np.uint8)
    count = 0
    for i in range(8, img.shape[0], 8):
        for j in range(8, img.shape[1], 8):
            reconstructed_image[i - 8: i, j - 8: j] = reconstructed[count, :, :]
            count += 1
    # Масштабируем полученное изображение до (8, 8) для вычисления pHash
    reconstructed_image = cv2.resize(reconstructed_image, (8, 8))

    # Вычисление pHash
    hash_value = __phash(reconstructed_image)

    # # Изображение, от которого вычислялся хэш
    # io.imshow(reconstructed_image)
    # io.show()

    return hash_value
