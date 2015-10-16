import os
from skimage import io

from preps import *
from features import *
from visualize import *

def testRatioTransform():
    preperation = RatioTransform()
    visual = ImagePlot()

    #im = io.imread('data/train/rectangles_up/B21/00969_01994.png')
    im = io.imread('data/train/blue_circles/D1b/02583_06123.png')

    im = preperation.process(im)
    visual.show(im)

def testDetectCircles(directories):
    feature = DetectCircle()
    transform = RatioTransform()
    plot = BoxPlot()

    values = []
    names = []
    for directory in directories:
        for dirpath, dirnames, _ in os.walk(directory):
            if not dirnames:
                images = io.imread_collection(os.path.join(dirpath, '*.png'))
                localvalues = []
                for (image, fn) in zip(images, images.files):
                    #imageTemp = transform.process(image)
                    featureValue = feature.process(image)

                    localvalues.append(featureValue)
                    print('%s, %s, %f' % (os.path.basename(dirpath), fn, featureValue))
                values.append(localvalues)
                names.append('%s/%s' % (os.path.basename(os.path.dirname(dirpath)) ,os.path.basename(dirpath)))

    plot.show(names,values)

def testHog(directories):
    feature = HogFeature()
    plot = BoxPlot()

    values = []
    names = []
    for directory in directories:
        for dirpath, dirnames, _ in os.walk(directory):
            if not dirnames:
                images = io.imread_collection(os.path.join(dirpath, '*.png'))
                localvalues = []
                for (image, fn) in zip(images, images.files):
                    featureValue = feature.process(image)

                    localvalues.append(featureValue[0])
                    print('%s, %s, %f' % (os.path.basename(dirpath), fn, featureValue[0]))
                values.append(localvalues)
                names.append('%s/%s' % (os.path.basename(os.path.dirname(dirpath)) ,os.path.basename(dirpath)))

    plot.show(names,values)

def testHsv(directories):
    feature = HsvFeature()
    featureCircle = DetectCircle()
    plot = ScatterPlot('hue','circleness')

    values = []
    names = []
    for directory in directories:
        for dirpath, dirnames, _ in os.walk(directory):
            if not dirnames:
                images = io.imread_collection(os.path.join(dirpath, '*.png'))
                localvalues = []
                for (image, fn) in zip(images, images.files):
                    featureValue = feature.process(image)
                    circleness = featureCircle.process(image)

                    localvalues.append([featureValue[0],circleness])
                    print('%s, %s' % (os.path.basename(dirpath), fn))
                values.append(localvalues)
                names.append('%s/%s' % (os.path.basename(os.path.dirname(dirpath)) ,os.path.basename(dirpath)))

    plot.show(names,values)


#im = io.imread('data/train/blue_circles/D1b/02583_06123.png')

#f = HogFeature(visualize=True)
#fd,result = f.process(im)
#visual = ImagePlot()
#visual.show(result)
testHsv(['data/train/rectangles_up/B21','data/train/blue_circles/D10','data/train/stop'])