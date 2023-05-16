import numpy as np
from PIL import Image


class ImageReconstruction:
    @staticmethod
    def reconstruction_with_pca(image: Image.Image, k: int) -> Image.Image:
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

        # Вычисляем среднее по векторам
        mean = np.mean(vectors, axis=0)
        # Центрирование данных (sum(Xi)=0)
        centered_data = vectors - mean
        # Вычисление ковариационной матрицы
        covariance_matrix = np.cov(centered_data, rowvar=False)
        # Вычисление собственных значений и собственных векторов ковариационной матрицы
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        del covariance_matrix
        # Сортировка собственных векторов в порядке убывания собственных значений
        indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues, sorted_eigenvectors = eigenvalues[indices], eigenvectors[:, indices]

        # Выбор первых k собственных векторов
        selected_eigenvectors = sorted_eigenvectors[:, :k]

        # Преобразование данных в новое пространство признаков
        X_reduced = np.dot(centered_data, selected_eigenvectors)
        # Обратное преобразование данных в исходное пространство признаков
        X_recovered = np.dot(X_reduced, selected_eigenvectors.T) + mean
        X_recovered = np.asarray(X_recovered, dtype=np.complex64).real.astype(np.uint8)
        X_recovered = X_recovered.reshape(-1, 8, 8)

        # Восстановленное изображение
        reconstruction = np.zeros(img.shape, dtype=np.uint8)
        count = 0
        for i in range(0, height - 7, 8):
            for j in range(0, width - 7, 8):
                reconstruction[i: i + 8, j: j + 8] = X_recovered[count, :, :]
                count += 1
        reconstructed_image = Image.fromarray(reconstruction)
        # reconstructed_image.show()
        return reconstructed_image
