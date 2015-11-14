from learn import AbstractLearner
from sklearn.svm import SVC
from sklearn import preprocessing


class NormalSVCTrainer(AbstractLearner):

    def __init__(self, kernel='linear', penalty=1.0, cache=200, scale=True, scheme='ovr'):
        self.learner = SVC(C=penalty, kernel=kernel, probability=True, cache_size=cache, decision_function_shape=scheme)
        self.kernel = kernel
        self.penalty = penalty
        self.scheme = scheme
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

    def _predict_proba(self, x):
        if self.scale:
            x_scaled = self.scaler.transform(x)
            return self.learner.predict_proba(x_scaled)
        else:
            return self.learner.predict_proba(x)

    def __str__(self):
        return 'SVC (kernel=%s, penalty: %f, scheme: %s)' % (self.kernel, self.penalty, self.scheme)

