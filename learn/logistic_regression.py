from learn import AbstractLearner
from sklearn.linear_model import LogisticRegression


class LogisticRegression(AbstractLearner):

    def __init__(self):
        self.learner = LogisticRegression()

    def train(self, x_train, y_train):
        self.learner = self.learner.fit(x_train, y_train)

    def predict(self, x):
        return self.learner.predict(x)

