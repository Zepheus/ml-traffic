from features import AbstractFeature
from preps import HsvTransform, PrepCombiner
import numpy as np

# Class to extract hsv (hue, saturation and value) features
class HsvFeature(AbstractFeature):
    # Initialise the hsv feature extractor
    def __init__(self):
        self.transform = PrepCombiner([HsvTransform()])

    # Process the hsv feature extractor to extract the hsv features
    # Only the hue value is considered as both saturation and value contain
    # less information
    def process(self, im):
        hsvIm = im.prep(self.transform)
        hAvg = np.mean(hsvIm[:, :, 0])

        return [hAvg]  # saturation and value are useless
