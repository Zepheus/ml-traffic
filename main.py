from learn import LogisticRegressionTrainer
from cross_validation import *
from prediction import *

features = [ColorFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(3, 3), resize=96),
            HaarFeature(n_haars=5)]

def create_logistic_trainer(x):
    return lambda: LogisticRegressionTrainer(regularization=x)


train_and_predict(create_logistic_trainer(181), features, ['data/train'],
                  ['data/test'], augment=False)
