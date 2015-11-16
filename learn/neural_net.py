from learn import AbstractLearner
from sknn.mlp import Classifier, Layer, Convolution
import numpy as np

class NeuralNet(AbstractLearner):

    def __init__(self):
        self.learner = Classifier(
            layers=[
                # Convolution("Rectifier", channels=10, pool_shape=(2,2), kernel_shape=(3, 3)),
                Layer('Rectifier', units=600),
                Layer('Softmax')],
            learning_rate=0.01,
            learning_rule='momentum',
            learning_momentum=0.9,
            batch_size=25,
            valid_size=0.1,
            # valid_set=(X_test, y_test),
            n_stable=10,
            n_iter=10,
            verbose=True)

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
        return 'NeuralNet'

