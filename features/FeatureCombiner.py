import numpy as np
from features import AbstractFeature


class FeatureCombiner(AbstractFeature):
    def __init__(self, feature_extractors):
        self.extractors = feature_extractors

    def process(self, im):
        for extractor in self.extractors:
            im.calcFeature(extractor)
