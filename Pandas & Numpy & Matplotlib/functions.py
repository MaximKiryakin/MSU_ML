from typing import List

def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    sum = 0
    find = 0
    for i in range(min(len(X), len(X[0]))):
        if X[i][i] >= 0:
            sum += X[i][i]
            find = 1
    return sum if find else -1


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    y1, x1 = y[:], x[:]
    for elem in x1:
        if elem in y1:
            y1.remove(elem)
        else:
            return False

    return not y1

def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    if len(x) < 2:
        return -1
    max = 0
    find = 0
    for i in range(len(x) - 1):
        if (x[i] * x[i + 1]) % 3 == 0:
            if not find:
                max = x[i] * x[i + 1]
                find = 1
            else:
                if x[i] * x[i + 1] > max:
                    max = x[i] * x[i + 1]
    return max if find else -1

def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    ans = [[0] * len(image[0]) for _ in range(len(image))]
    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(len(weights)):
                ans[i][j] += image[i][j][k] * weights[k]

    return ans


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    a = []
    b = []
    for i in range(0, len(x), 1):
        a += [int(x[i][0])] * int(x[i][1])
    for i in range(0, len(y), 1):
        b += [int(y[i][0])] * int(y[i][1])

    if len(a) != len(b):
        return -1
    else:
        sum = 0
        for i in range(len(a)):
            sum += a[i] * b[i]
        return sum


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    m = [[0.0] * len(X[0]) for _ in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y)):
            len_x = 0.0
            len_y = 0.0
            for k in range(len(X[0])):
                len_x += X[i][k] ** 2
                len_y += Y[j][k] ** 2
                m[i][j] += X[i][k] * Y[j][k]
            if len_x == 0 or len_y == 0:
                m[i][j] = 1.0
            else:
                m[i][j] = m[i][j] / ((len_x ** 0.5) * (len_y ** 0.5))
    return m
