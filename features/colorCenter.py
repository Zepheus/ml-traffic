from features import AbstractFeature
from preps import HsvTransform
from preps import RatioTransform
from preps import Segmentation
from skimage.transform import *


class ColorCenter(AbstractFeature):

    def __init__(self):
        self.transform = HsvTransform()
        self.segmentation = Segmentation()
        self.ratioTransform = RatioTransform()

    def process(self,im):
        segmented = self.segmentation.process(im)
        small = resize(segmented,(10,10),preserve_range=True)

        return small.ravel()

