from preps import AbstractPrep
from skimage.transform import resize


class SqueezeTransform(AbstractPrep):

    def __init__(self, ratio=0.75):
        self.ratio = ratio

    def process(self, im):
        (w, h, _) = im.shape
        w *= self.ratio
        return resize(im, [w, h])

