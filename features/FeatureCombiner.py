import numpy as np

class FeatureCombiner:

    def __init__(self, feature_extractors):
        self.extractors = feature_extractors

    def process(self, im):
        features = []
        for extractor in self.extractors:
            feature = extractor.process(im)
            if isinstance(feature, np.ndarray): # fix for numpy array -> normal array
                feature = feature.tolist()
            elif not isinstance(feature, list): # fix for scalar -> single array
                feature = [feature]

            features.append(feature)
        return [x for y in features for x in y] # flatten list
