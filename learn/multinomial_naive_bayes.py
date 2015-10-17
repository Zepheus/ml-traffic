from learn import AbstractLearner
from sklearn.naive_bayes import MultinomialNB


class MultinomialNaiveBayes(AbstractLearner):

    def __init__(self):
        self.learner = MultinomialNB()

    def train(self, x_train, y_train):
        self.learner = self.learner.fit(x_train, y_train)

    def predict(self, x):
        return self.learner.predict(x)

