import os
import numpy as np
from skimage import io
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
import sys


class LabelledImage():
    def __init__(self, image, label, filename, superLabel=''):
        self.image = image
        self.label = label
        self.filename = filename
        self.superLabel = superLabel


def load(directories, permute=True):
    values = []
    for directory in directories:
        for dirpath, dirnames, _ in os.walk(directory):
            if not dirnames:
                images = io.imread_collection(os.path.join(dirpath, '*.png'))
                label = os.path.basename(dirpath)
                superLabel = os.path.basename(os.path.dirname(dirpath))
                for (image, fn) in zip(images, images.files):
                    values.append(LabelledImage(image, label, fn, superLabel))

    return np.random.permutation(values) if permute else values


def folds(images, feature, trainerFunc, k=1, useSuperClass=True, numPCA=0):
    kf = KFold(len(images), n_folds=k)
    errorRatios = []
    i = 1
    for trainIndices, testIndices in kf:
        print('-------- calculating fold %d --------' % i)
        trainData = [images[i] for i in trainIndices]
        testData = [images[i] for i in testIndices]
        trainer = trainerFunc()
        if numPCA > 0:
            (tuples, pca) = train(trainData, feature, trainer, useSuperClass, numPCA)
        else:
            (tuples, pca) = (train(trainData, feature, trainer, useSuperClass, numPCA), None) # Do not use PCA

        errorRatio, errors = test(testData, feature, trainer, tuples, useSuperClass, pca)
        print('    error ratio for fold %d is %f' % (i, errorRatio))
        errorRatios.append(errorRatio)
        i = i + 1
    print('-------- folds done --------\n')
    return errorRatios


def train(train, feature, trainer, useSuperClass, numPCA):
    feature_by_class = {}
    sys.stdout.write('')
    pca = None
    for idx, image in enumerate(train):
        features = feature.process(image.image)
        classification = image.superLabel if useSuperClass else image.label

        if classification in feature_by_class:
            feature_by_class[classification].append(features)
        else:
            feature_by_class[classification] = [features]
        sys.stdout.write('\r    feature calculation [%d %%]' % int(100.0 * float(idx) / len(train)))
        sys.stdout.flush()
    sys.stdout.write('\r    feature calculation [100 %]\n')

    tuples = list(feature_by_class.items())
    features = np.concatenate([np.array(x[1], dtype=np.float64) for x in tuples])
    classes = np.hstack([np.repeat([i], len(x[1])) for i, x in enumerate(tuples)])

    if numPCA > 0:
        pca = PCA(n_components=min(numPCA, len(features[0])))
        pca.fit(features)
        features = pca.transform(features)

    trainer.train(features, classes)
    print('    training complete!!')
    if numPCA > 0:
        return (tuples, pca)
    else:
        return tuples


def test(test, feature, trainer, classes, useSuperClass, pca):
    errors = []
    for idx, image in enumerate(test):
        sys.stdout.write('\r    test calculation [%d %%]' % (int(100.0 * float(idx) / len(test))))
        features = feature.process(image.image)
        if pca is not None:
            features = pca.transform(features)  # perform PCA

        prediction = classes[trainer.predict(features)[0]][0]
        groundTruth = image.superLabel if useSuperClass else image.label
        if (prediction != groundTruth):
            print('\r    [ERROR] for image %s I predicted "%s" but the sign actually was "%s"' % (
                image.filename, prediction, groundTruth))
            errors.append([image, prediction, groundTruth])
    sys.stdout.write('\r    test calculation [100 %]\n')

    return float(len(errors)) / len(test), errors
