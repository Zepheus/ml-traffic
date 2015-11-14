from learn import AbstractLearner
from sklearn.svm import SVC


class NormalSVCTrainer(AbstractLearner):

    def __init__(self, kernel='rbf', penalty=1.0):
        self.learner = SVC(C=penalty, kernel=kernel, probability=True)

    def _train(self, x_train, y_train):
        self.learner = self.learner.fit(x_train, y_train)

    def _predict(self, x):
        return self.learner.predict(x)

    def predict_proba(self, x):
        return self.learner.predict_proba(x)

