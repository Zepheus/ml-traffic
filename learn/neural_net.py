from learn import AbstractLearner
from sknn.mlp import Classifier, Layer, Convolution
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy import stats
from operator import itemgetter
import numpy as np

class NeuralNet(AbstractLearner):

    def __init__(self, num_units=150):
        self.units = num_units
        self.learner = Classifier(
            layers=[
                Layer('Sigmoid', units=100),
                Layer('Sigmoid', units=81),
                Layer('Softmax')],
            learning_rate=0.006,
            valid_size=0.2,
            n_stable=10)

        self.learner = RandomizedSearchCV(self.learner, param_distributions={
            'learning_rate': stats.uniform(0.001, 0.05),
            'hidden0__units': stats.randint(20, 300),
            'hidden0__type': ["Rectifier"],
            'hidden1__units': stats.randint(20, 300),
            'hidden1__type': ["Rectifier"]}, verbose=1)

        # self.learner = GridSearchCV(network, verbose=1, param_grid={
        #     'learning_rate': [0.05, 0.01, 0.005, 0.001],
        #     'hidden0__units': [4, 8, 12],
        #     'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})

    # Utility function to report best scores
    def report(self, n_top=3):
        grid_scores = self.learner.grid_scores_
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  score.mean_validation_score,
                  np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")

    def _train(self, x_train, y_train):
        print('Starting GPU training')
        if not isinstance(x_train, np.ndarray):
            x_train = np.array(x_train)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
        self.learner = self.learner.fit(x_train, y_train)

        try:
            self.report(3)  # report best params
        except:
            pass

    def _predict(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return self.learner.predict(x)

    def _predict_proba(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return self.learner.predict_proba(x)

    def __str__(self):
        return 'NeuralNet (units %d)' % self.units

