import pandas as pd
import warnings
from numpy import ndarray
from sklearn.preprocessing import LabelEncoder
import numpy as np
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    def find_common_dummies(train, test, category):
        dummies_train = pd.get_dummies(train[category].apply(pd.Series).stack()).sum(level=0)
        columns_train = dummies_train.columns

        dummies_test = pd.get_dummies(test[category].apply(pd.Series).stack()).sum(level=0)
        columns_test = dummies_test.columns

        common_genres = columns_train[np.in1d(columns_train, columns_test)]
        dummies_test, dummies_train = dummies_test[common_genres], dummies_train[common_genres]

        train = pd.concat([train, dummies_train.reindex(train.index)], axis=1)
        test = pd.concat([test, dummies_test.reindex(test.index)], axis=1)

        del test[category]
        del train[category]

        return train, test, common_genres

    def transform_dataset(dff_train, dff_test):
        category, cl = [], LabelEncoder()

        for i in range(3):
            cl.fit(dff_train[f"actor_{i}_gender"])

            dff_train[f"actor_{i}_gender"] = cl.transform(dff_train[f"actor_{i}_gender"])
            dff_test[f"actor_{i}_gender"] = cl.transform(dff_test[f"actor_{i}_gender"])

            dff_train[f"actor_{i}_gender"] = dff_train[f"actor_{i}_gender"].astype('category')
            dff_test[f"actor_{i}_gender"] = dff_test[f"actor_{i}_gender"].astype('category')
            category += [f"actor_{i}_gender"]

        dff_train, dff_test, common_genres = find_common_dummies(dff_train, dff_test, "genres")
        category += list(common_genres)

        dff_train, dff_test, common_dir = find_common_dummies(dff_train, dff_test, "directors")
        category += list(common_dir)

        dff_train = dff_train.drop(columns=["keywords", "filming_locations"])
        dff_test = dff_test.drop(columns=["keywords", "filming_locations"])

        return dff_train, dff_test, category

    train = pd.read_json(train_file, lines=True)
    test = pd.read_json(test_file, lines=True)

    train, test, category = transform_dataset(train, test)

    y = train.awards
    X = train.drop(columns='awards')

    model = LGBMRegressor(**{'learning_rate': 0.0026330215519286913, 'max_depth': 9, 'n_estimators': 3500})

    model.fit(X, y, categorical_feature=category)
    return model.predict(test.to_numpy())
