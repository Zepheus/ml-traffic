from learn import AbstractLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier


class BaggingLearner(AbstractLearner):

    def __init__(self):
        self.learner = BaggingClassifier(LogisticRegression(), max_samples=0.5, max_features=0.5)

    def _train(self, x_train, y_train):
        self.learner = self.learner.fit(x_train, y_train)

    def _predict(self, x):
        return self.learner.predict(x)

    def predict_proba(self, x):
        return self.learner.predict_proba(x)

