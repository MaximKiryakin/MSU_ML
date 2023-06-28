import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype
        self.col_unique = []

    def fit(self, X, Y=None):
        # создание массивов уникальных значений для каждой колонки
        # for column in X.columns:
        #  self.col_unique += [sorted(X[column].unique())]
        X_1 = np.array(X)
        X_1 = X_1.T
        for column in X_1:
            self.col_unique += [sorted(np.unique(column))]

    def transform(self, X):
        X_1 = np.array(X)
        X_1 = X_1.T
        ans = []
        for i in range(len(X_1[0])):
            tmp1 = []
            for index in range(len(X_1)):
                tmp2 = [0]*len(self.col_unique[index])
                tmp2[self.col_unique[index].index(X_1[index, i])] = 1
                tmp1 += tmp2
            ans += [tmp1]
        return np.array(ans)

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.list_dict = []

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        # создаю список из словарей для каждой колонки
        self.list_dict = [{} for _ in range(X.shape[1])]

        for i in range(X.shape[0]):
            for index, column in enumerate(X.columns):
                if not X.iloc[i, index] in self.list_dict[index]:
                    tmp = [0] * 3
                    tmp[0] = Y[X[column] == X.iloc[i, index]].mean()
                    tmp[1] = Y[X[column] == X.iloc[i, index]].count() / X.shape[0]
                    self.list_dict[index][X.iloc[i, index]] = tmp.copy()

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        ans = []
        for i in range(X.shape[0]):
            tmp1 = []
            for index, column in enumerate(X.columns):
                tmp = self.list_dict[index][X.iloc[i, index]].copy()
                tmp[2] = (tmp[0] + a) / (tmp[1] + b)
                tmp1 += tmp
            ans += [tmp1]
        return np.array(ans)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        # словарь, где ключ - это кортеж из обучающей выборки
        self.info = {}

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        gen = group_k_fold(X.shape[0], self.n_folds, seed)
        for condition in gen:
            self.info[tuple(condition[1])] = [{} for _ in range(X.shape[1])]

        # заполняю словарь для каждого фолда
        for fld in self.info:
            for i in fld:
                for index, column in enumerate(X.columns):
                    if not X.iloc[i, index] in self.info[fld][index]:
                        sub_X = np.array(X.iloc[list(fld), index])
                        sub_Y = np.array(Y.iloc[list(fld)])
                        tmp = [0] * 3
                        tmp[0] = sub_Y[sub_X == X.iloc[i, index]].mean()
                        tmp[1] = sub_Y[sub_X == X.iloc[i, index]].size / len(fld)
                        self.info[fld][index][X.iloc[i, index]] = tmp.copy()

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        ans = []
        for i in range(X.shape[0]):
            tmp1 = []
            for index, column in enumerate(X.columns):
                # поиск фолда из которого брать значения
                for fold in self.info:
                    if i not in fold:
                        tmp = self.info[fold][index][X.iloc[i, index]].copy()
                        tmp[2] = (tmp[0] + a) / (tmp[1] + b)
                        tmp1 += tmp
                        break
            ans += [tmp1]
        return np.array(ans)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    enc = MyOneHotEncoder(dtype=int)
    enc.fit(x.reshape(len(x), 1))
    onehot = enc.transform(x.reshape(len(x), 1))
    tmp = np.sum(onehot[y == 1], axis=0)
    return tmp / (tmp + np.sum(onehot[y == 0], axis=0))
