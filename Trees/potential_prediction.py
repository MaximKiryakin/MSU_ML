import os


from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline

import numpy as np


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        x_cop = x.copy()
        for i in range(0, x_cop.shape[0]):
            row = np.argmin(x_cop[i]) // 256       # строка
            col = 256 - np.argmin(x_cop[i]) % 256  # столбец

            x_cop[i] = np.roll(x_cop[i], (-1)*(128 - col), axis=1)

            if len(np.unique(x_cop[i])) > 10:
                shift = 256 - 128 + col

                tmp = np.split(x_cop[i], [128], axis=1)
                if shift > 0:
                    tmp[1] = np.flip(tmp[0], axis=1)
                else:
                    tmp[0] = np.flip(tmp[1], axis=1)

                tmp = np.column_stack((tmp[0], tmp[1]))
                x_cop[i] = tmp

            x_cop[i] = np.roll(x_cop[i].T, (128 - row), axis=1).T
            if len(np.unique(x_cop[i])) > 10:
                x_cop[i] = x_cop[i].T

                shift = 256 + 128 - row

                tmp = np.split(x_cop[i], [128], axis=1)
                if shift > 0:
                    tmp[1] = np.flip(tmp[0], axis=1)
                else:
                    tmp[0] = np.flip(tmp[1], axis=1)

                tmp = np.column_stack((tmp[0], tmp[1]))
                x_cop[i] = tmp.T

        return x_cop.reshape((x_cop.shape[0], -1))


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    # model_reg = ExtraTreesRegressor(n_estimators=15, max_depth=6)
    model_reg = ExtraTreesRegressor(n_estimators=974,
                                    min_samples_split=2,
                                    min_samples_leaf=2,
                                    max_features="sqrt",
                                    max_depth=12,
                                    bootstrap=False)
    regressor = Pipeline([('vectorizer', PotentialTransformer()), ('decision_tree', model_reg)])
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
