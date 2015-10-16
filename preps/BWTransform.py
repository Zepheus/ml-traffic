from preps import AbstractPrep
from skimage import color

class BWTransform(AbstractPrep):

    def process(self,im):
        return color.rgb2gray(im)
