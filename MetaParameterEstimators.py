# Own packages
from learn import *
from features import *
from image_loader import *
from cross_validation import cross_validate
import inspect

def estimateMeta(directories,trainer,rangeValues,label):
    images = load(directories, True, permute=True)
    results = []
    for v,feature in rangeValues:
        print("value: %f" % v)
        results.append([v ,cross_validate(images, feature, trainer, k=10,
                                          use_super_class=False, number_of_pca_components=0,verbose=False)])
        for i in images:
            i.reset(feature)

    from visualize import ScatterPlot
    plot = ScatterPlot()
    name = inspect.stack()[1][3]
    plot.save([label],[results],"result_graphs/" + name)

def estimateHogOrientationsParameters(directories,trainer):
    estimateMeta(directories,trainer,[(ori,HogFeature(orientations=ori)) for ori in range(1,20)],"orientations")

def estimateHogPixelsPerCellParameters(directories,trainer):
    estimateMeta(directories,trainer,[(v,HogFeature(pixels_per_cell=(v,v))) for v in [2,4,5,10,20,50,100]],"pixels per cell")

def estimateHogCellsPerBlockParameters(directories,trainer):
    estimateMeta(directories,trainer,[(v,HogFeature(cells_per_block=(v,v))) for v in [1,2,3,4,5,6,7,8,9,10]],"pixels per cell")

def estimateDetectCircleParameters(directories,trainer):
    features = [HogFeature()]
    estimateMeta(directories,trainer,[(v,features.append(DetectCircle(sigma=v))) for v in np.arange(0.1,5,0.1)],"sigma")

def estimateColorCenterParameters(directories,trainer):
    estimateMeta(directories,trainer,[(v,ColorCenter(size=v)) for v in range(20)],"scale size")