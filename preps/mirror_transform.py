from preps import AbstractPrep
import numpy as np


class MirrorTransform(AbstractPrep):

    def process(self,im):
        return np.fliplr(im)
