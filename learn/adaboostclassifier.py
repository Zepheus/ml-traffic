from learn import AbstractLearner
from sklearn.ensemble import AdaBoostClassifier


class AdaLearner(AbstractLearner):

    def __init__(self, n_estimators=100):
        self.learner = AdaBoostClassifier(n_estimators=n_estimators)

    def _train(self, x_train, y_train):
        self.learner = self.learner.fit(x_train, y_train)

    def _predict(self, x):
        return self.learner.predict(x)

    def _predict_proba(self, x):
        return self.learner.predict_proba(x)

