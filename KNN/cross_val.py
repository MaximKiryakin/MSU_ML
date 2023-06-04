import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without i-th fold while the 2nd one contains
                                       i-th fold
    """
    array = np.arange(num_objects)
    fold_len = num_objects // num_folds
    ans = []
    for i in range(num_folds):
        tmp = np.array(array.copy())
        if i != num_folds - 1:
            ans += [(np.delete(tmp, [_ for _ in range(i*fold_len, (i+1)*fold_len)]), tmp[i*fold_len:(i+1)*fold_len])]
        else:
            ans += [(tmp[:i*fold_len], tmp[i*fold_len:])]
    return ans


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations)

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_true, y_predict) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    d = {}
    for normalizer_name in parameters['normalizers']:
        for n_neighbors in parameters['n_neighbors']:
            for metric in parameters['metrics']:
                for weight in parameters['weights']:
                    key = (normalizer_name[1], n_neighbors, metric, weight)
                    model = knn_class(n_neighbors=n_neighbors, weights=weight, metric=metric)
                    tmp = []
                    for train, test in folds:
                        x_train, y_train = X[train], y[train]
                        x_test, y_test = X[test], y[test]
                        if not normalizer_name[0] is None:
                            normalizer = normalizer_name[0]
                            normalizer.fit(x_train)
                            x_train = normalizer.transform(x_train)
                            x_test = normalizer.transform(x_test)
                        model.fit(x_train, y_train)
                        predict = model.predict(x_test)
                        tmp += [score_function(y_test, predict)]
                    d[key] = sum(tmp)/len(tmp)
    return d