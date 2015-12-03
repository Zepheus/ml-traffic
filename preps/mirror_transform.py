from preps import AbstractPrep
import numpy as np


# this preprocessor will mirror the image
class MirrorTransform(AbstractPrep):

    def process(self, im):
        return np.fliplr(im)
