from PIL import Image
import numpy as np
from my_modules.pca import my_pca
from my_modules.dct import my_dct


def improved_phash(image: Image) -> np.uint8:
    # Преобразуем в оттенки серого
    img = np.asarray(image.convert('L'), dtype=np.uint8).copy()

    height, width = img.shape[0: 2]
    # Исходное изображение
    # temp = Image.fromarray(img)
    # temp.show()

    # Разбиение на блоки размером 8x8
    blocks = []
    for i in range(0, height - 7, 8):
        for j in range(0, width - 7, 8):
            block = img[i: i + 8, j: j + 8]
            blocks.append(block)
    vectors = np.asarray(blocks, dtype=np.uint8).reshape(-1, 64)
    del blocks

    # Восстановленные блоки
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

    # Изображение, от которого вычисляется хэш
    # temp = Image.fromarray(reconstructed_image)
    # temp.show()

    # Вычисление pHash
    dct = my_dct(reconstructed_image)

    dct_low_freq = dct[0: 3, 0: 3].copy().reshape(-1)
    x = (np.arange(8))[dct_low_freq[1:] >= dct_low_freq[1:].mean()]
    h = np.uint8(np.sum(np.power(2, x)))
    return h
