from learn import NormalSVCTrainer, LogisticRegressionTrainer, NeuralNet
from cross_validation import *
from prediction import *

features = [HsvFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(3, 3), resize=96),
                 HaarFeature(n_haars=5)]

features2 = [HsvFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(3, 3), resize=96), RegionRatio()]

def createLinearTrainer(param):
    return lambda: NormalSVCTrainer(kernel='linear', scale=True, penalty=param, cache=1000)

def createRbfTrainer(reg, gamma):
    return lambda: NormalSVCTrainer(kernel='rbf', scale=True, penalty=reg, gamma=gamma, cache=1000)

def createNeuralNetwork(x):
    return lambda: NeuralNet(num_units=x)

def createLogisticTrainer(x):
    return lambda: LogisticRegressionTrainer(regularization=x)


train_and_predict(createNeuralNetwork(100), features, ['data/train'], ['data/test'])

#cross_grid_search(['data/train'], SVC(C=1.0), features,
#                  [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}])


#def createRbfTrainer(gamma, c):
#    return lambda: LinearSVCTrainer(kernel='rbf', scale=True, penalty=c)


#trainFolds(["data/train"], createLogisticTrainer(181), features, augment=False, folds=3)
#trainFolds(["data/train/"], createNeuralNetwork(200), features) # mean error_ratio is 0.083940 (std: 0.008505)
#trainFolds(["data/train"], [createLinearTrainer(x) for x in [0.01, 0.1, 0.2]], features) # mean error_ratio is 0.083940 (std: 0.008505)
#estimateMetas(['data/train'], lambda: LogisticRegressionTrainer(181.0))
