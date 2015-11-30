from preps import AbstractPrep
from skimage.filters import *

# This preprocessor will blur the image using a gaussian filter.
class GaussianTransform(AbstractPrep):

    def __init__(self, sigma=2.0, multichannel=False):
        self.sigma = sigma
        self.multichannel = multichannel

    def process(self, im):
        return gaussian_filter(im, sigma=self.sigma, multichannel=self.multichannel)
