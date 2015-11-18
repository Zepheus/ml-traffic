from learn import AbstractLearner
from sknn.mlp import Classifier, Layer, Convolution
from sklearn.grid_search import GridSearchCV
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
            n_stable=10,
            verbose=True)
        # self.learner = GridSearchCV(network, verbose=1, param_grid={
        #     'learning_rate': [0.05, 0.01, 0.005, 0.001],
        #     'hidden0__units': [4, 8, 12],
        #     'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})

    def _train(self, x_train, y_train):
        print('Starting GPU training')
        if not isinstance(x_train, np.ndarray):
            x_train = np.array(x_train)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
        self.learner = self.learner.fit(x_train, y_train)

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

