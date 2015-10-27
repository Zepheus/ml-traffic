import os
import sys
import numpy as np
from skimage import io


class LabelledImage:
    def __init__(self, image, filename, label='Unknown', superLabel='Unknown'):
        self.image = image
        self.filename = filename
        self.label = label
        self.super_label = superLabel
        self.features = {}

    def isSet(self,feature):
        if feature.key() == "FeatureCombiner": return False

        return self.features.__contains__(feature.key()) and self.features[feature.key()] is not None

    def set(self,feature,value):
        if feature.key() == "FeatureCombiner": raise "cannot set value for combiner"

        if not self.isSet(feature):
            self.features[feature.key()] = value

    def get(self,feature):
        if feature.key() == "FeatureCombiner": raise "cannot get value for combiner"

        return self.features[feature.key()]

    def getFeatureVector(self):
        return np.concatenate([value for key,value in sorted(self.features.items())])

    def reset(self,feature):
        if feature.key() == "FeatureCombiner":
            for f in feature.extractors:
                self.reset(f)
        else:
            self.features[feature.key()] = None

    def __str__(self):
        return self.filename


def load(directories, is_train_data, permute=True):
    print('Loading images. Train: %r' % is_train_data)
    values = []
    for directory in directories:
        for dirpath, dirnames, _ in os.walk(directory):
            if not dirnames:
                images = io.imread_collection(os.path.join(dirpath, '*.png'))
                if is_train_data:
                    label = os.path.basename(dirpath)
                    super_label = os.path.basename(os.path.dirname(dirpath))
                for (image, fn) in zip(images, images.files):
                    if is_train_data:
                        values.append(LabelledImage(image, fn, label, super_label))
                    else:
                        values.append(LabelledImage(image, fn))
    return np.random.permutation(values) if permute else values


def feature_extraction(images, feature,verbose=True):
    if verbose:
        sys.stdout.write('')
    for idx, image in enumerate(images):
        if verbose:
            sys.stdout.write('\r    feature calculation [%d %%] (%s)'
                         % (int(100.0 * float(idx) / len(images)), image.filename))
            sys.stdout.flush()
        if feature.key() == "FeatureCombiner":
            feature.process(image)
        else:
            if not image.isSet(feature):
                image.set(feature,feature.process(image.image))
    if verbose:
        sys.stdout.write('\r    feature calculation [100 %]\n')