from learn import AbstractLearner
from sklearn.naive_bayes import GaussianNB


class NaiveBayes(AbstractLearner):

    def __init__(self):
        self.learner = GaussianNB()

    def train(self, x_train, y_train):
        self.learner = self.learner.fit(x_train, y_train)

    def predict(self, x):
        return self.learner.predict(x)

