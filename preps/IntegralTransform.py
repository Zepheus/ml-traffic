from preps import AbstractPrep
from skimage.transform import integral_image

class IntegralTransform(AbstractPrep):


    def process(self, im):
        return integral_image(im)
