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

def createTrainer(param):
    return lambda: LinearSVCTrainer(scale=True, penalty=param)

trainFolds(["data/train"], [createTrainer(x) for x in [1, 10, 100, 1000]], features) # mean error_ratio is 0.083940 (std: 0.008505)
#estimateMetas(['data/train'], lambda: LogisticRegressionTrainer(181.0))
