import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractLearner(metaclass=ABCMeta):

    def __init__(self):
        self.classes = []
        self.labels = []

    def train(self, x_train, y_train):
        self._extract_classes(y_train)

        self._train(x_train, self.labels)

    @abstractmethod
    def _train(self, x_train, y_train):
        pass

    def _extract_classes(self, y_train):
        self.classes = list(set(y_train))
        class_to_index = {key: index for index, key in enumerate(self.classes)}
        self.labels = np.concatenate(np.array([[class_to_index[name] for name in y_train]]))

    @staticmethod
    def fit_array(x):
        if isinstance(x, list):
            return np.array(x, ndmin=2, dtype=np.float64)
        else:
            dimensions = len(x.shape)
            return x.reshape(1, -1) if dimensions == 1 else x

    def predict(self, x):
        x = AbstractLearner.fit_array(x)
        indices = self._predict(x)
        return [self.classes[idx] for idx in indices]

    def predict_proba(self, x):
        return self._predict_proba(AbstractLearner.fit_array(x))

    @abstractmethod
    def _predict(self, x):
        pass

    @abstractmethod
    def _predict_proba(self, x):
        pass

    def __str__(self):
        return type(self).__name__
