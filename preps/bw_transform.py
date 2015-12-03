from preps import AbstractPrep
from skimage import color

# this preprocessor will transform the image to a grayscale image.
class BWTransform(AbstractPrep):

    def process(self, im):
        return color.rgb2gray(im)
