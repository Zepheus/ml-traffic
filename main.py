# Linear imports:
from learn import LogisticRegressionTrainer
from cross_validation import *
from prediction import *

# Neural imports:
from neural import train_and_predict_ensemble, \
    build_rgb_cnn_2, build_rgb_cnn, build_grayscale_cnn, train_single_with_warmup, cross_validate_neural

# Non-linear (convnets)
train_and_predict_ensemble(['data/train'],  ['data/test'],
                           networks=(build_rgb_cnn, build_grayscale_cnn),
                           learning_rates=(0.005, 0.005),
                           grays=(False, True),
                           input_sizes=(45, 45),
                           weights=(0.6, 0.4),
                           epochs=(300, 200),
                           augment=True)

# Example of evaluating an ensemble 50/50
# cross_validate_neural(['data/train'],
#                networks=(build_rgb_cnn, build_grayscale_cnn),
#                epochs=(300, 200),
#                input_sizes=(45, 45),
#                weights=(0.6, 0.4),
#                learning_rates=(0.005, 0.005),
#                grays=(False, True),
#                num_folds=5, augment=True)

# train_and_predict_ensemble(['data/train'],  ['data/test'],
#                            networks=(build_rgb_cnn, build_rgb_cnn_2, build_grayscale_cnn),
#                            learning_rates=(0.005, 0.005, 0.005),
#                            grays=(False, False, True),
#                            input_sizes=(45, 48, 45),
#                            weights=(0.5, 0.25, 0.25),
#                            epochs=(250, 200, 200),
#                            augment=True)

# Example of evaluating one model, explicitly takes tuples:
# Trains default RGB network:
# cross_validate_neural(['data/train'],
#                networks=(build_rgb_cnn, ),
#                epochs=(10, ),
#                input_sizes=(45, ),
#                weights=(1, ),
#                learning_rates=(0.005, ),
#                grays=(False, ),
#                num_folds=2, augment=True)

# Trains grayscale network:
# cross_validate_neural(['data/train'],
#                networks=(build_grayscale_cnn, ),
#                grays=(True, ),
#                epochs=(150, ),
#                input_sizes=(45, ),
#                weights=(1, ),
#                learning_rates=(0.005, ),
#                num_folds=2, augment=True)

# Trains 2nd RGB network
# cross_validate_neural(['data/train'],
#                networks=(build_rgb_cnn_2, ),
#                grays=(False, ),
#                epochs=(150, ),
#                input_sizes=(45, ),
#                weights=(1, ),
#                learning_rates=(0.005, ),
#                num_folds=2, augment=True)

# train_single_with_warmup(['data/train'],  ['data/test'],
#                         build_rgb_cnn, 400, flip=200, input_size=45, learning_rate=0.005, gray=False, augment=True)


# Linear model:
# features = [ColorFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(3, 3), resize=96),
#             HaarFeature(n_haars=5), DetectCircle(sigma=1.8)]
#
# train_and_predict(lambda: LogisticRegressionTrainer(181), features, ['data/train'],
#                   ['data/test'], augment=False)
