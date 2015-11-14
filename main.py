from learn import *
from MetaParameterEstimators import *
from cross_validation import *
from prediction import *

#train_and_predict(lambda: NormalSVCTrainer(kernel='linear', cache=1000, penalty=0.85, scale=False),
#                  [HsvFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(3, 3), resize=96),
#                    DetectSymmetry(blocksize=3, size=96), RegionRatio()], 0,
#                  ['data/train'], ['data/test'])

trainFolds(["data/train"], lambda: NormalSVCTrainer(kernel='linear', cache=1000, penalty=0.85, scale=True)) # mean error_ratio is 0.083940 (std: 0.008505)
#estimateMetas(['data/train'], lambda: LogisticRegressionTrainer(181.0))
