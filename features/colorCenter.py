from features import AbstractFeature
from preps import Segmentation
from skimage.transform import *


class ColorCenter(AbstractFeature):

    def __init__(self, size=10):
        self.segmentation = Segmentation()
        self.size = size

    def process(self, im):
        segmented = im.prep(self.segmentation)
        small = resize(segmented,(self.size,self.size),preserve_range=True)

        return small.ravel()

