from features import *
from image_loader import *
from cross_validation import cross_validate
from learn import LogisticRegressionTrainer
from visualize import ScatterPlot
import inspect


def estimate_meta(directories, trainer, range_values, label, static_features):
    images = load(directories, True, permute=True)
    results = []
    for v, feature in range_values:
        print("Optimizing %s: %f" % (feature.key(), v))
        feature_calculator = [feature] + static_features
        error_rate = cross_validate(images, feature_calculator, trainer, k=10, verbose=False)
        print("Error rate: %f" % error_rate)
        results.append([v, error_rate])

        for i in images:
            i.reset(feature)

    plot = ScatterPlot(ylabel='error rate', xlabel='param')
    name = inspect.stack()[1][3]
    plot.save([label], [results], "result_graphs/" + name)


def estimate_hog_orientations_parameters(directories, trainer):
    estimate_meta(directories, trainer, [(ori, HogFeature(orientations=ori, pixels_per_cell=(8, 8), cells_per_block=(1, 1))) for ori in range(1, 15, 2)], "orientations",
                  [HsvFeature(), DetectCircle(), DetectSymmetry(), RegionRatio()])


def estimate_hog_resize_parameters(directories, trainer):
    estimate_meta(directories, trainer, [(size, HogFeature(orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), resize=size)) for size in range(32, 200, 16)], "size",
                  [HsvFeature(), DetectCircle(), DetectSymmetry(), RegionRatio()])


def estimate_hog_pixels_per_cell_parameters(directories, trainer):
    estimate_meta(directories, trainer,
                  [(v, HogFeature(orientations=5, pixels_per_cell=(v, v), cells_per_block=(1, 1))) for v in [2, 4, 8, 16, 32, 64, 96]],
                 "pixels per cell", [HsvFeature(), DetectCircle(), DetectSymmetry(), RegionRatio()])


def estimate_hog_cells_per_block_parameters(directories, trainer):
    estimate_meta(directories, trainer,
                  [(v, HogFeature(cells_per_block=(v, v), orientations=5, pixels_per_cell=(8, 8))) for v in range(1, 10)], "cells per block",
                  [HsvFeature(), DetectCircle(), DetectSymmetry(), RegionRatio()])


def estimate_detect_circle_parameters(directories, trainer):
    estimate_meta(directories, trainer, [(v, DetectCircle(sigma=v)) for v in np.arange(0.1, 5, 0.4)],
                 "sigma", [HsvFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(1, 1)), DetectSymmetry(), RegionRatio()])


def estimate_detect_symmetry_large_block_parameters(directories, trainer):
    estimate_meta(directories, trainer, [(v, DetectSymmetry(size=v)) for v in range(10, 96, 10)],
             "large_size", [HsvFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(3, 3)), RegionRatio(), DetectCircle()])


def estimate_detect_symmetry_small_block_parameters(directories, trainer):
    estimate_meta(directories, trainer, [(v, DetectSymmetry(size=96, block_size=v)) for v in range(2, 32, 5)],
             "small_size", [HsvFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(3, 3)), RegionRatio(), DetectCircle()])


def estimate_metas(directories, trainer):
    meta_estimators = [ #estimateHogResizeParameters,
                        #estimateHogOrientationsParameters,
                        #estimateHogCellsPerBlockParameters,
                        #estimateDetectCircleParameters,
                        estimate_detect_symmetry_small_block_parameters,
                        #estimateDetectSymmetryLargeBlockParameters,
                        #estimateHogPixelsPerCellParameters
                       ]

    for estimator in meta_estimators:
        estimator(directories, trainer)


def create_logistic_trainer(x):
    return lambda: LogisticRegressionTrainer(regularization=x)

estimate_metas(['data/train'], create_logistic_trainer(181))
