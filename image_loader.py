import os
import sys
import numpy as np
from skimage import io
from preps import RotateTransform, GaussianTransform, MirrorTransform, SqueezeTransform
from random import shuffle


class LabelledImage(object):
    def __init__(self, filename, label='Unknown'):
        self._image = None
        self.filename = filename
        self.label = label
        self.features = {}
        self.preps = {}

    @property
    def image(self):
        if self._image is None:
            self._image = io.imread(self.filename)
        return self._image

    def disposeImage(self):
        if self._image is not None:
            del self._image
            self._image = None

    @image.setter
    def image(self, value):
        self._image = value

    def isSet(self, feature):
        if feature.key() == "FeatureCombiner": return False

        return self.features.__contains__(feature.key()) and self.features[feature.key()] is not None

    def set(self, feature, value):
        if feature.key() == "FeatureCombiner": raise "cannot set value for combiner"

        if not self.isSet(feature):
            self.features[feature.key()] = value

    def get(self, feature):
        if feature.key() == "FeatureCombiner": raise "cannot get value for combiner"

        return self.features[feature.key()]

    def getFeatureVector(self):
        return np.hstack([self.features[key] for key in sorted(self.features)])

    def reset(self, feature):
        if isinstance(feature, list):
            for f in feature:
                self.reset(f)
        else:
            del self.features[feature.key()]

    def calcFeature(self, feature):
        if isinstance(feature, list):
            for feature_single in feature:
                self.calcFeature(feature_single)  # recursively calculate features
        else:
            if feature.key() not in self.features:
                self.features[feature.key()] = feature.process(self)

            return self.features[feature.key()]

    def prep(self, prep):
        # if prep.key() not in self.preps:
        #    self.preps[prep.key()] = prep.process(self.image)
        return prep.process(self.image)
        # return self.preps[prep.key()]

    def __str__(self):
        return self.filename


def augment_images(images, transforms):

    augmented = []
    for img in images:
        for idx, transform in enumerate(rotators):
            transformed = transform.process(img.image)
            newImage = LabelledImage("%s_aug_%d" % (img.filename, idx), img.label)
            newImage.image = transformed # fix for lazy loading
            augmented.append(newImage)
    return images + augmented


def load(directories, is_train_data, permute=True):
    print('Loading images. Train: %r' % is_train_data)
    values = []
    for directory in directories:
        for dirpath, dirnames, filenames in os.walk(directory):
            if filenames:
                # images = io.imread_collection(os.path.join(dirpath, '*.png'))
                if is_train_data:
                    label = os.path.basename(dirpath)
                for fn in filenames:
                    if is_train_data:
                        values.append(LabelledImage(os.path.join(dirpath, fn), label))
                    else:
                        values.append(LabelledImage(os.path.join(dirpath, fn)))
    print('Loaded %d images.' % len(values))
    if permute:
        shuffle(values)
        print('Shuffled images')
    return values


def print_update(idx, total, name):
    sys.stdout.write('\r    feature calculation [%d %%] (%s)'
                     % (int(100.0 * float(idx) / total), name))
    sys.stdout.flush()


def feature_extraction(images, feature, verbose=True):
    if verbose:
        sys.stdout.write('')
    for idx, image in enumerate(images):
        if verbose:
            sys.stdout.write('\r    feature calculation [%d %%] (%s)'
                             % (int(100.0 * float(idx) / len(images)), image.filename))
            sys.stdout.flush()
        image.calcFeature(feature)
    if verbose:
        sys.stdout.write('\r    feature calculation [100 %]\n')
