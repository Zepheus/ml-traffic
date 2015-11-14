from preps import AbstractPrep
from skimage.filters import *


class GaussianTransform(AbstractPrep):

    def __init__(self, sigma=2.0):
        self.sigma = sigma

    def process(self, im):
        return gaussian_filter(im, sigma=self.sigma, multichannel=False)
