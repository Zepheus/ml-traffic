from learn import *
from MetaParameterEstimators import *
from cross_validation import *
import numpy as np
from prediction import *
from sklearn.svm import SVC


features = [HsvFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(3, 3), resize=96),
                DetectSymmetry(blocksize=3, size=96), RegionRatio()]

#train_and_predict(lambda: NormalSVCTrainer(kernel='linear', cache=1000, penalty=0.85, scale=False),
#                    features, 0, ['data/train'], ['data/test'])

#cross_grid_search(['data/train'], SVC(C=1.0), features,
#                  [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}])


#def createRbfTrainer(gamma, c):
#    return lambda: LinearSVCTrainer(kernel='rbf', scale=True, penalty=c)

def createLinearTrainer(param):
    return lambda: NormalSVCTrainer(kernel='linear', scale=True, penalty=param, cache=1000)

def createRbfTrainer(reg, gamma):
    return lambda: NormalSVCTrainer(kernel='rbf', scale=True, penalty=reg, gamma=gamma, cache=1000)

trainFolds(["data/train"], [createRbfTrainer(reg, gamma) for reg in [1000, 100] for gamma in [1e-5, 1e-6]], features) # mean error_ratio is 0.083940 (std: 0.008505)
#trainFolds(["data/train"], [createLinearTrainer(x) for x in [0.01, 0.1, 0.2]], features) # mean error_ratio is 0.083940 (std: 0.008505)
#estimateMetas(['data/train'], lambda: LogisticRegressionTrainer(181.0))
