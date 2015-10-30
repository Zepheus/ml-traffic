from learn import AbstractLearner
from sklearn.linear_model import LogisticRegression


class LogisticRegressionTrainer(AbstractLearner):

    def __init__(self, regularization=1.0, penalty='l2'):
        self.regularization = regularization
        self.penalty = penalty
        self.learner = LogisticRegression(penalty=penalty, multi_class='ovr', C=regularization)

    def _train(self, x_train, y_train):
        self.learner = self.learner.fit(x_train, y_train)

    def _predict(self, x):
        return self.learner.predict(x)

    def predict_proba(self, x):
        return self.learner.predict_proba(x)

    def __str__(self):
        return 'LogisticRegression (reg=%f, penalty: %s)' % (self.regularization, self.penalty)

