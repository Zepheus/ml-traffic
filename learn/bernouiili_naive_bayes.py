from learn import AbstractLearner
from sklearn.naive_bayes import BernoulliNB


class BernoulliNaiveBayes(AbstractLearner):

    def __init__(self):
        self.learner = BernoulliNB()

    def _train(self, x_train, y_train):
        self.learner = self.learner.fit(x_train, y_train)

    def _predict(self, x):
        return self.learner.predict(x)

    def _predict_proba(self, x):
        return self.learner.predict_proba(x)

