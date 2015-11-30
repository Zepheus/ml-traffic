from preps import AbstractPrep
from skimage import exposure


# This preprocesser equalizes the histogram of the image.
class EqualizerTransform(AbstractPrep):

    def process(self, im):
        return exposure.equalize_hist(im)
