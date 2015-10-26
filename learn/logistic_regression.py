from learn import AbstractLearner
from sklearn.linear_model import LogisticRegression


class LogisticRegressionTrainer(AbstractLearner):

    def __init__(self):
        self.learner = LogisticRegression(penalty='l2', multi_class='ovr')

    def _train(self, x_train, y_train):
        self.learner = self.learner.fit(x_train, y_train)

    def _predict(self, x):
        return self.learner.predict(x)

    def predict_proba(self, x):
        return self.learner.predict_proba(x)

