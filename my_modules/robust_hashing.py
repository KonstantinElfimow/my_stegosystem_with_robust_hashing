from PIL import Image
import numpy as np
from my_modules.pca import my_pca
from my_modules.dct import my_dct, my_idct


def improved_phash(image: Image) -> np.uint16:
    # Преобразуем в оттенки серого
    img = np.asarray(image.convert('L'), dtype=np.uint8).copy()

    height, width = img.shape[0: 2]

    # Разбиение на блоки размером 8x8
    blocks = []
    for i in range(0, height - 7, 8):
        for j in range(0, width - 7, 8):
            block = img[i: i + 8, j: j + 8]
            blocks.append(block)
    vectors = np.asarray(blocks, dtype=np.uint8).reshape(-1, 64)
    del blocks

    # Ухудшаем качество перед вычислением pHash
    k = len(vectors) // 2
    reconstructed = np.asarray(my_pca(vectors, k), dtype=np.complex64).real.astype(np.uint8).reshape(-1, 8, 8)

    # Восстановленное изображение
    reconstructed_image = np.zeros(img.shape, dtype=np.uint8)
    count = 0
    for i in range(0, height - 7, 8):
        for j in range(0, width - 7, 8):
            reconstructed_image[i: i + 8, j: j + 8] = reconstructed[count, :, :]
            count += 1
    # Масштабируем полученное изображение до (8, 8) для вычисления pHash
    reconstructed_image = np.asarray(Image.fromarray(reconstructed_image).convert('L').resize((8, 8)), dtype=np.uint8)

    # Вычисление pHash
    dct = my_dct(reconstructed_image)

    # Вычисление медианы
    median = np.median(dct)

    # Вычисление больше или равно
    flags = [np.median(dct[i: i + 2, j: j + 2]) >= median for i in range(0, 8, 2) for j in range(0, 8, 2)]

    # Вычисление pHash
    h = np.uint16(np.sum(np.power(2, np.arange(16)[flags])))
    return h
