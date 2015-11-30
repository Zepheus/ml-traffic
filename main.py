from learn import NormalSVCTrainer, LogisticRegressionTrainer
from cross_validation import *
from prediction import *

features = [HsvFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(3, 3), resize=96),
                 HaarFeature(n_haars=5)]

features2 = [HsvFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(3, 3), resize=96), RegionRatio()]

def createLinearTrainer(param):
    return lambda: NormalSVCTrainer(kernel='linear', scale=True, penalty=param, cache=1000)

def createRbfTrainer(reg, gamma):
    return lambda: NormalSVCTrainer(kernel='rbf', scale=True, penalty=reg, gamma=gamma, cache=1000)

def createLogisticTrainer(x):
    return lambda: LogisticRegressionTrainer(regularization=x)


train_and_predict(createLogisticTrainer(181), features, ['data/train'], ['data/test'])
