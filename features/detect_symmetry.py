import numpy as np

from preps import BWTransform
from preps import ResizeTransform

from features import AbstractFeature

class DetectSymmetry(AbstractFeature):

    def __init__(self, size=32, threshold=0.1, blocksize=5):
        self.threshold = threshold
        self.transform = BWTransform()
        self.blocksize = blocksize
        self.resize = ResizeTransform(size)


    def process(self, im):
        resized = self.resize.process(im)
        bw = self.transform.process(resized)

        # Vertical residue
        vertflipped = np.fliplr(bw) # flip horizontally
        vertresidu = np.subtract(bw, vertflipped)
        idx = vertresidu[:, :] < self.threshold
        vertresidu[idx] = 0
        vertFeatures = ResizeTransform(self.blocksize).process(vertresidu).flatten()

        # Horizontal residue
        horflipped = np.transpose(np.fliplr(np.transpose(bw)))
        horresidu = np.subtract(bw, horflipped)
        idx = horresidu[:, :] < self.threshold
        horresidu[idx] = 0
        horFeatures = ResizeTransform(self.blocksize).process(horresidu).flatten()
        return np.concatenate((vertFeatures, horFeatures))

