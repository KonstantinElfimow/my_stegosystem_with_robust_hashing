import numpy as np


def my_pca(vectors: np.array, k: int) -> np.array:
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
    return X_recovered
