import os
import numpy as np
from skimage import io

class LabelledImage:
    def __init__(self, image, label, filename):
        self.image = image
        self.label = label
        self.filename = filename

def load(directories, permute=True):
    values = []
    for directory in directories:
        for dirpath, dirnames, _ in os.walk(directory):
            if not dirnames:
                images = io.imread_collection(os.path.join(dirpath, '*.png'))
                label = os.path.basename(dirpath)
                superLabel = os.path.basename(os.path.dirname(dirpath))
                for (image, fn) in zip(images, images.files):
                    values.append(LabelledImage(image, label, fn))

    return np.random.permutation(values) if permute else values
