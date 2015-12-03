from features import *
from image_loader import *
from cross_validation import cross_validate
from visualize import ScatterPlot
import inspect


def estimateMeta(directories, trainer, rangeValues, label, staticFeatures):
    images = load(directories, True, permute=True)
    results = []
    for v, feature in rangeValues:
        print("Optimizing %s: %f" % (feature.key(), v))
        feature_calculator = [feature] + staticFeatures
        error_rate = cross_validate(images, feature_calculator, trainer, k=10,
                                          use_super_class=False, number_of_pca_components=0, verbose=False)
        print("Error rate: %f" % error_rate)
        results.append([v, error_rate])

        for i in images:
            i.reset(feature)

    plot = ScatterPlot(ylabel='error rate', xlabel='param')
    name = inspect.stack()[1][3]
    plot.save([label], [results], "result_graphs/" + name)


def estimateHogOrientationsParameters(directories, trainer):
    estimateMeta(directories, trainer, [(ori, HogFeature(orientations=ori, pixels_per_cell=(8, 8), cells_per_block=(1, 1))) for ori in range(1, 15, 2)], "orientations",
                 [ColorFeature(), DetectCircle(), DetectSymmetry(), RegionRatio()])

def estimateHogResizeParameters(directories, trainer):
    estimateMeta(directories, trainer, [(size, HogFeature(orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), resize=size)) for size in range(32, 200, 16)], "size",
                 [ColorFeature(), DetectCircle(), DetectSymmetry(), RegionRatio()])


def estimateHogPixelsPerCellParameters(directories, trainer):
    estimateMeta(directories, trainer,
                 [(v, HogFeature(orientations=5, pixels_per_cell=(v, v), cells_per_block=(1, 1))) for v in [2, 4, 8, 16, 32, 64, 96]],
                 "pixels per cell", [ColorFeature(), DetectCircle(), DetectSymmetry(), RegionRatio()])


def estimateHogCellsPerBlockParameters(directories, trainer):
    estimateMeta(directories, trainer,
                 [(v, HogFeature(cells_per_block=(v, v), orientations=5, pixels_per_cell=(8, 8))) for v in range(1, 10)], "cells per block",
                 [ColorFeature(), DetectCircle(), DetectSymmetry(), RegionRatio()])


def estimateDetectCircleParameters(directories, trainer):
    estimateMeta(directories, trainer, [(v, DetectCircle(sigma=v)) for v in np.arange(0.1, 5, 0.4)],
                 "sigma", [ColorFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(1, 1)), DetectSymmetry(), RegionRatio()])

def estimateDetectSymmetryLargeBlockParameters(directories, trainer):
    estimateMeta(directories, trainer, [(v, DetectSymmetry(size=v)) for v in range(10, 96, 10)],
             "large_size", [ColorFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(3, 3)), RegionRatio(), DetectCircle()])

def estimateDetectSymmetrySmallBlockParameters(directories, trainer):
    estimateMeta(directories, trainer, [(v, DetectSymmetry(size=96, blocksize=v)) for v in range(2, 32, 5)],
             "small_size", [ColorFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(3, 3)), RegionRatio(), DetectCircle()])

def estimateColorCenterParameters(directories, trainer):
    estimateMeta(directories, trainer, [(v, ColorCenter(size=v)) for v in range(1, 20)], "scale size",
                 [ColorFeature(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(1, 1)), DetectSymmetry(), RegionRatio(), DetectCircle()])


def estimateMetas(directories, trainer):
    meta_estimators = [ #estimateHogResizeParameters,
                        #estimateHogOrientationsParameters,
                        #estimateHogCellsPerBlockParameters,
                        #estimateDetectCircleParameters,
                        estimateDetectSymmetrySmallBlockParameters,
                        #estimateDetectSymmetryLargeBlockParameters,
                        #estimateHogPixelsPerCellParameters
                       ]

    for estimator in meta_estimators:
        estimator(directories, trainer)