import numpy as np


class MinMaxScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)


    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        return (data - self.min) / (self.max - self.min)


class StandardScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)


    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        return (data - self.mean) / self.std
