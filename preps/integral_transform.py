from preps import AbstractPrep
from skimage.transform import integral_image


# This preprocessor calculates the integral from an image.
# In an integral image every pixel contains the summation of the area above-left.
class IntegralTransform(AbstractPrep):

    def process(self, im):
        return integral_image(im)
