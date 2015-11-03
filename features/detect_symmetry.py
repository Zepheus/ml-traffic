import numpy as np

from preps import BWTransform,ResizeTransform,PrepCombiner

from features import AbstractFeature

class DetectSymmetry(AbstractFeature):

    def __init__(self, size=96, threshold=0.1, blocksize=2):
        self.threshold = threshold
        self.blocksize = blocksize
        self.transform = PrepCombiner([ResizeTransform(size),BWTransform()])


    def process(self, im):
        bw = im.prep(self.transform)

        # Vertical residue
        vertflipped = np.fliplr(bw) # flip horizontally
        vertresidu = np.subtract(bw, vertflipped)
        #idx = vertresidu[:, :] < self.threshold
        #vertresidu[idx] = 0
        vertFeatures = ResizeTransform(self.blocksize).process(vertresidu).flatten()

        # Horizontal residue
        horflipped = np.transpose(np.fliplr(np.transpose(bw)))
        horresidu = np.subtract(bw, horflipped)
        #idx = horresidu[:, :] < self.threshold
        #horresidu[idx] = 0
        horFeatures = ResizeTransform(self.blocksize).process(horresidu).flatten()
        return np.concatenate((vertFeatures, horFeatures))

