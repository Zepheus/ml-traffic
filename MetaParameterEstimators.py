# Own packages
from learn import *
from features import *
from image_loader import *
from cross_validation import cross_validate

def estimatePcaParameters(directories):
    images = load(directories, True, permute=True)
    combiner = FeatureCombiner([HsvFeature(),HogFeature(),DetectCircle(),ColorCenter()])  # Feature selection
    trainer = GaussianNaiveBayes  # Learning algorithm, make sure this is a function and not an object
    results = []
    for pca_comp in range(100):

        if (pca_comp >= len(images)):
            break
        results.append([pca_comp ,cross_validate(images, combiner, trainer, k=10, use_super_class=False, number_of_pca_components=pca_comp,verbose=False)]) # use 10 folds, no pca

    from visualize import ScatterPlot
    plot = ScatterPlot()
    plot.show(["pca"],[results])

def estimateHogOrientationsParameters(directories):
    images = load(directories, True, permute=True)
    trainer = GaussianNaiveBayes  # Learning algorithm, make sure this is a function and not an object
    results = []
    for ori in range(1,20):
        combiner = FeatureCombiner([HogFeature(orientations=ori)])
        results.append([ori ,cross_validate(images, combiner, trainer, k=10, use_super_class=False, number_of_pca_components=0,verbose=False)]) # use 10 folds, no pca

    from visualize import ScatterPlot
    plot = ScatterPlot()
    plot.show(["orientations"],[results])

def estimateHogPixelsPerCellParameters(directories):
    images = load(directories, True, permute=True)
    trainer = GaussianNaiveBayes  # Learning algorithm, make sure this is a function and not an object
    results = []
    for pixels in [2**x for x in range(1,7)]:
        combiner = FeatureCombiner([HogFeature(pixels_per_cell=(pixels,pixels))])
        results.append([pixels ,cross_validate(images, combiner, trainer, k=10, use_super_class=False, number_of_pca_components=0,verbose=False)]) # use 10 folds, no pca

    from visualize import ScatterPlot
    plot = ScatterPlot()
    plot.show(["pixels_per_cell"],[results])
