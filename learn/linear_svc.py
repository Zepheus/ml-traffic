from learn import AbstractLearner
from sklearn.svm import SVC


class LinearSVCTrainer(AbstractLearner):

    def __init__(self):
        self.learner = SVC(C=1.0, kernel='linear', probability=True)

    def _train(self, x_train, y_train):
        self.learner = self.learner.fit(x_train, y_train)

    def _predict(self, x):
        return self.learner.predict(x)

    def predict_proba(self, x):
        return self.learner.predict_proba(x)

