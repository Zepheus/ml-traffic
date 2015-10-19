from learn import AbstractLearner
from sklearn.svm import LinearSVC


class LinearSVCTrainer(AbstractLearner):

    def __init__(self):
        self.learner = LinearSVC()

    def _train(self, x_train, y_train):
        self.learner = self.learner.fit(x_train, y_train)

    def _predict(self, x):
        return self.learner.predict(x)

