from features import AbstractFeature


class RegionRatio(AbstractFeature):

    def process(self, im):
        (height, width, _) = im.image.shape
        return [width / height]
