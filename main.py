import os
from sys import stdin

# Scientific packages
import numpy as np
from skimage import io

# Own packages
from learn import *
from preps import *
from features import *
from visualize import *
from Load import *

def testRatioTransform():
    preperation = RatioTransform()
    visual = ImagePlot()

    #im = io.imread('data/train/rectangles_up/B21/00969_01994.png')
    im = io.imread('data/train/blue_circles/D1b/02583_06123.png')

    im = preperation.process(im)
    visual.show(im)

def trainAll(directories):
    combiner = FeatureCombiner([HsvFeature(), DetectCircle()]) # Feature selection
    trainer = KNN() # Learning algorithm
    feature_by_class = {}
    for directory in directories:
        for dirpath, dirnames, _ in os.walk(directory):
                if not dirnames:
                    images = io.imread_collection(os.path.join(dirpath, '*.png'))
                    for (image, fn) in zip(images, images.files):
                        print('Training %s' % (fn))
                        features = combiner.process(image)
                        classification = os.path.basename(os.path.dirname(dirpath))
                        if classification in feature_by_class:
                            feature_by_class[classification].append(features)
                        else:
                            feature_by_class[classification] = [features]

    tuples = list(feature_by_class.items())
    features = np.concatenate([np.array(x[1], dtype=np.float64) for x in tuples])
    classes = np.hstack([np.repeat([i], len(x[1])) for i, x in enumerate(tuples)])

    trainer.train(features, classes)
    print('Trained data!')
    while True:
        print('Enter filename:')
        file = stdin.readline().rstrip()
        try:
            im = io.imread(file)
            features = combiner.process(im)
            prediction = trainer.predict(features)
            print('Predicted class: %s' % (tuples[prediction][0]))
        except:
            print('Failed processing file.')

def trainFolds(directories):
    images = load(directories, permute=True)
    combiner = FeatureCombiner([HsvFeature(), DetectCircle(), HogFeature()]) # Feature selection
    trainer = LinearSVCTrainer() # Learning algorithm
    ratios = folds(images, combiner, trainer, 3, True)
    print('average errorRatio is %f' % np.mean(ratios))


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
#testHsv(['data/train/rectangles_up/B21','data/train/blue_circles/D10','data/train/stop'])

#train_test()
#trainAll(['data/train/diamonds', 'data/train/forbidden'])
trainFolds(['data/train/rectangles_up/B21', 'data/train/blue_circles/D10', 'data/train/stop'])