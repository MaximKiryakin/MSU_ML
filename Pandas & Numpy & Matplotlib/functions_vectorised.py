import numpy as np


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    diagonal = np.diag(X)
    return diagonal[diagonal >= 0].sum() if len(diagonal[diagonal >= 0]) else -1


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return np.array_equal(np.bincount(x), np.bincount(y))


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x,
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    if len(x) <= 1:
        return -1
    x1, x2 = np.roll(x, 1), np.roll(x, -1)
    x1[0], x2[-1] = 0, 0
    tmp1, tmp2 = x * x1, x * x2
    ans = max(max(tmp1[tmp1 % 3 == 0]), max(tmp2[tmp2 % 3 == 0]))
    if ans == 0:
      if 0 in x:
        return ans
      else:
        return -1
    else:
      return ans


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return np.sum(image * weights, axis=2)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x1 = np.repeat(x.T[0], x.T[1])
    y1 = np.repeat(y.T[0], y.T[1])
    return np.dot(x1, y1) if len(x1) == len(y1) else -1


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    div = np.outer(np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1))
    d = np.dot(X, Y.T)
    tmp = div.copy()
    div[tmp == 0] = 1
    d[tmp == 0] = 1
    return d / div