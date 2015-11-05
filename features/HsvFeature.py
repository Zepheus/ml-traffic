from features import AbstractFeature
from preps import HsvTransform, PrepCombiner
import numpy as np


class HsvFeature(AbstractFeature):
    def __init__(self):
        self.transform = PrepCombiner([HsvTransform()])

    def process(self, im):
        hsvIm = im.prep(self.transform)
        hAvg = np.mean(hsvIm[:, :, 0])

      #  hWhiteValue = hsvIm[:, :, 2] > 0.6
      #  hWhiteSaturation = hsvIm[:, :, 1] < 0.25
      #  whiteMask = np.logical_and(hWhiteValue, hWhiteSaturation)
      #  hWhiteAvg = np.mean(whiteMask)

       # hB = im.image[:, :, 2] < 60
       # hG = im.image[:, :, 1] < 60
       # hR = im.image[:, :, 0] < 60
       # blackMask = np.logical_and(hR, np.logical_and(hB, hG))
       # hBlackAvg = np.mean(blackMask)


        return [hAvg]  # saturation and value are useless
        # return [hAvg,sAvg,vAvg]
