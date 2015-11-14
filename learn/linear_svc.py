from learn import AbstractLearner
from sklearn.svm import LinearSVC
from sklearn import preprocessing


class LinearSVCTrainer(AbstractLearner):

    def __init__(self, penalty=1.0,  scale=True):
        self.learner = LinearSVC(C=1.0)
        self.penalty = penalty
        self.scale = scale

    def _train(self, x_train, y_train):
        if self.scale:
            self.scaler = preprocessing.StandardScaler().fit(x_train)
            x_scaled = self.scaler.transform(x_train)
            self.learner = self.learner.fit(x_scaled, y_train)
        else:
            self.learner = self.learner.fit(x_train, y_train)

    def _predict(self, x):
        if self.scale:
            x_scaled = self.scaler.transform(x)
            return self.learner.predict(x_scaled)
        else:
            return self.learner.predict(x)

    def __str__(self):
        return 'LinearSVC (penalty: %f)' % (self.penalty)

