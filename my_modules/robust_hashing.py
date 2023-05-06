from PIL import Image
import numpy as np
from my_modules.pca import my_pca
from my_modules.dct import my_dct


def __phash(block: np.array) -> np.uint16:
    """ Перцептивное хэширование """
    assert block.shape == (8, 8) and block.dtype == np.uint8

    dct = my_dct(block)

    dct_low_freq = dct[0: 4, 0: 4].copy().reshape(-1)
    x = (np.arange(16))[dct_low_freq >= dct_low_freq[1:].mean()]
    h = np.uint16(np.sum(np.power(2, x)))
    return h


def improved_phash(image: Image) -> np.uint16:
    # Преобразуем в оттенки серого
    img = np.asarray(image.convert('L'), dtype=np.uint8).copy()

    # Исходное изображение
    # temp = Image.fromarray(img)
    # temp.show()

    # Разбиение на блоки размером 8x8
    blocks = []
    for i in range(8, img.shape[0], 8):
        for j in range(8, img.shape[1], 8):
            block = img[i - 8: i, j - 8: j]
            blocks.append(block)
    vectors = np.asarray(blocks, dtype=np.uint8).reshape(-1, 64)
    del blocks

    # Восстановленные блоки
    k = len(vectors) // 2
    reconstructed = np.asarray(my_pca(vectors, k), dtype=np.uint8).reshape(-1, 8, 8)

    # Восстановленное изображение
    reconstructed_image = np.zeros(img.shape, dtype=np.uint8)
    count = 0
    for i in range(8, img.shape[0], 8):
        for j in range(8, img.shape[1], 8):
            reconstructed_image[i - 8: i, j - 8: j] = reconstructed[count, :, :]
            count += 1
    # Масштабируем полученное изображение до (8, 8) для вычисления pHash
    reconstructed_image = np.asarray(Image.fromarray(reconstructed_image).convert('L').resize((8, 8)), dtype=np.uint8)

    # Вычисление pHash
    hash_value = __phash(reconstructed_image)

    # Изображение, от которого вычислялся хэш
    # temp = Image.fromarray(reconstructed_image)
    # temp.show()

    return hash_value
