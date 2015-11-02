from features import AbstractFeature
from skimage.transform import *
from skimage import exposure

from preps import HsvTransform

class ColorHistogram(AbstractFeature):

    def __init__(self, numbins=256):
        self.numbins = numbins
        self.hsv_transform = HsvTransform()

    def process(self, im):
        hsvImg = im.prep(self.hsv_transform)
        return exposure.histogram(hsvImg[:, :, 0], nbins=self.numbins) # only on hue
