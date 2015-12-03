from preps import AbstractPrep
from skimage import color


# This preprocessor extracts the hue information from an image
class HsvTransform(AbstractPrep):
    def process(self, im):
        return color.rgb2hsv(im)
