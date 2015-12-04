from features import AbstractFeature


# Class to calculate the ratio feature (i.e. image ratio as a feature)
class RegionRatio(AbstractFeature):
    # Process the image ratio feature
    def process(self, im):
        (height, width, _) = im.image.shape
        return [width / height]
