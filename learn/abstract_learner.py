from abc import ABCMeta, abstractmethod


class AbstractLearner(metaclass=ABCMeta):

    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x):
        pass
