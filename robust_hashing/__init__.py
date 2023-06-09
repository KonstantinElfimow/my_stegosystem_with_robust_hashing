import numpy as np
from PIL import Image
import scipy.fftpack

""" Библиотека работает с hash_size  4, 16, 64 или 256 """

""" ANTIALIAS -это фильтр, используемый в библиотеке Pillow для улучшения качества изображений при изменении их размера. 
Он используется для сглаживания краев изображения и предотвращения появления артефактов при изменении размера. ANTIALIAS 
был переименован в Lanczos в версии Pillow 2.7.0, но по-прежнему используется для улучшения качества изображений. """

try:
    ANTIALIAS = Image.Resampling.LANCZOS
except AttributeError:
    ANTIALIAS = Image.ANTIALIAS


def _binary_array_to_hex(arr):
    """Внутренняя функция для создания шестнадцатеричной строки из двоичного массива."""
    bit_string = ''.join(str(b) for b in 1 * arr)
    width = int(np.ceil(len(bit_string) / 4))
    return '{:0>{width}x}'.format(int(bit_string, 2), width=width)


class ImageHash:
    """Инкапсуляция хеша. Может использоваться для ключей словаря и сравнений."""

    def __init__(self, binary_array: np.array):
        self.__binary_array = binary_array.flatten()
        self.__value = sum([2 ** i for i, v in enumerate(self.__binary_array) if v])

    @property
    def binary_array(self):
        return self.__binary_array

    @property
    def value(self):
        return self.__value

    def __str__(self):
        return _binary_array_to_hex(self.binary_array)

    def __repr__(self):
        return repr(self.value)

    def __sub__(self, other) -> int:
        if other.__class__ is self.__class__:
            if self.binary_array.size != other.binary_array.size:
                raise TypeError('ImageHash должны иметь одинаковую длину.', self.binary_array.size, other.hash.size)
            return np.count_nonzero(self.binary_array != other.binary_array)
        else:
            return NotImplemented

    def __eq__(self, other) -> bool:
        if other.__class__ is self.__class__:
            return np.array_equal(self.binary_array, other.binary_array)
        else:
            return NotImplemented

    def __ne__(self, other) -> bool:
        if other.__class__ is self.__class__:
            return not np.array_equal(self.binary_array, other.binary_array)
        else:
            return NotImplemented

    def __hash__(self) -> int:
        # Возвращает целое число
        return self.value

    def __len__(self):
        # Возвращает битовую длину хеша
        return self.binary_array.size


def uint_to_hash(integer: np.uint) -> ImageHash:
    hash_size = integer.nbytes * 8
    arr = np.asarray([bool(int(x)) for x in (bin(integer)[2:]).zfill(hash_size)])
    return ImageHash(arr)


def average_hash(image: Image.Image, hash_size: int) -> ImageHash:
    """
    Вычисление Average Hash

    @image должно быть экземпляром PIL.
    """
    if hash_size < 4:
        raise ValueError('Размер хеша должен быть больше или равен 4')

    # уменьшить размер и сложность, затем преобразовать в оттенки серого
    image = image.convert('L').resize((int(np.sqrt(hash_size)), int(np.sqrt(hash_size))), ANTIALIAS)

    # найти среднее значение пикселя; «пиксели» — это массив значений пикселей в диапазоне от 0 (черный) до 255 (белый).
    pixels = np.asarray(image, dtype=np.uint8)
    avg = np.mean(pixels)

    # создать строку битов
    diff = np.asarray(pixels > avg)
    # составить хеш
    return ImageHash(diff)


def phash(image: Image.Image, hash_size: int, midfreq_factor: int) -> ImageHash:
    """
    Вычисление Perceptual hash

    @image должно быть экземпляром PIL.
    """
    if hash_size < 4:
        raise ValueError('Размер хеша должен быть больше или равен 4')

    img_size = int(np.sqrt(hash_size)) * midfreq_factor
    image = image.convert('L').resize((img_size, img_size), ANTIALIAS)
    pixels = np.asarray(image, dtype=np.uint8)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
    dctlowfreq = dct[:int(np.sqrt(hash_size)), :int(np.sqrt(hash_size))]
    med = np.median(dctlowfreq)
    diff = np.asarray(dctlowfreq > med)
    return ImageHash(diff)


def dhash(image: Image.Image, hash_size: int) -> ImageHash:
    """
    Вычисление Difference Hash

    вычисляет различия по горизонтали

    @image должно быть экземпляром PIL.
    """
    if hash_size < 4:
        raise ValueError('Размер хеша должен быть больше или равен 4')

    image = image.convert('L').resize((int(np.sqrt(hash_size)) + 1, int(np.sqrt(hash_size))), ANTIALIAS)
    pixels = np.asarray(image, dtype=np.uint8)
    # вычисляет различия между столбцами
    diff = np.asarray(pixels[:, 1:] > pixels[:, :-1])
    return ImageHash(diff)
