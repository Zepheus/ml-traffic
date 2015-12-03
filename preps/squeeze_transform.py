from preps import AbstractPrep
from skimage.transform import resize


# This preprocessor will resize the image such that the width
# of the resulting image is a fraction of the original (fraction is determined by "ratio" parameter).
class SqueezeTransform(AbstractPrep):

    def __init__(self, ratio=0.8):
        self.ratio = ratio

    def process(self, im):
        (w, h, _) = im.shape
        w = int(w*self.ratio)
        return resize(im, [w, h])

