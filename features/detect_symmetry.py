import numpy as np

from preps import BWTransform, ResizeTransform, PrepCombiner
from features import AbstractFeature


# Detect symmetry within the image based on the residual image
class DetectSymmetry(AbstractFeature):
    # Initialise the feature with a size value indicating the size of the
    # image after resizing (default 96) and a block size indicating the size
    # of the residual image that will be used as feature
    def __init__(self, size=96, block_size=2):
        self.block_size = block_size
        self.transform = PrepCombiner([ResizeTransform(size), BWTransform()])

    # Process the feature to extract the residual image indicating the symmetry.
    # The image is flipped horizontal/vertical and subtracted from the original
    # image, yielding the residual image.
    def process(self, im):
        bw = im.prep(self.transform)

        # Vertical residual
        flipped_vert = np.fliplr(bw)
        residual_vert = np.subtract(bw, flipped_vert)
        feature_vert = ResizeTransform(self.block_size).process(residual_vert).flatten()

        # Horizontal residual
        flipped_hor = np.transpose(np.fliplr(np.transpose(bw)))
        residual_hor = np.subtract(bw, flipped_hor)
        feature_hor = ResizeTransform(self.block_size).process(residual_hor).flatten()
        return np.concatenate((feature_vert, feature_hor))

