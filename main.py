from learn import *
from MetaParameterEstimators import *
from cross_validation import *
from prediction import *

#train_and_predict(lambda: LogisticRegressionTrainer(181),
                  #[HogFeature(orientations=5, pixels_per_cell=(8, 8),cells_per_block=(3, 3), resize=96), HsvFeature()],
                  #0, ['data/train'], ['data/test'])

trainFolds(["data/train"], lambda: LogisticRegressionTrainer(181.0))  # Estimated 181 through CV
#estimateMetas(['data/train'], lambda: LogisticRegressionTrainer(181.0))
