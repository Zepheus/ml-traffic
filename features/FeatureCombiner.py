import numpy as np
from features import AbstractFeature


class FeatureCombiner(AbstractFeature):
    def __init__(self, feature_extractors):
        self.extractors = feature_extractors

    def process(self, im):
        features = []
        for extractor in self.extractors:
            if not im.isSet(extractor):
                im.set(extractor, extractor.process(im.image))
            feature = im.get(extractor)
            if isinstance(feature, np.ndarray):  # fix for numpy array -> normal array
                feature = feature.tolist()
            elif not isinstance(feature, list):  # fix for scalar -> single array
                feature = [feature]

            features.append(feature)
        return [x for y in features for x in y]  # flatten list
