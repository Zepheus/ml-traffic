from preps import AbstractPrep
from skimage import color

class HsvTransform(AbstractPrep):

    def process(self,im):
        return color.rgb2hsv(im)
