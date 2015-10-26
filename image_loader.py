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
        self.features = None

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


def feature_extraction(images, feature_combiner,verbose=True):
    if verbose:
        sys.stdout.write('')
    for idx, image in enumerate(images):
        if verbose:
            sys.stdout.write('\r    feature calculation [%d %%] (%s)'
                         % (int(100.0 * float(idx) / len(images)), image.filename))
            sys.stdout.flush()
        if not image.features:
            image.features = feature_combiner.process(image.image)
    if verbose:
        sys.stdout.write('\r    feature calculation [100 %]\n')