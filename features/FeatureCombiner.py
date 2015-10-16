class FeatureCombiner:

    def __init__(self, feature_extractors):
        self.extractors = feature_extractors

    def process(self, im):
        features = []
        for extractor in self.extractors:
            feature = extractor.process(im)
            if not isinstance(feature, list):
                feature = [feature]

            features.append(feature)
        return [x for y in features for x in y] # flatten list
