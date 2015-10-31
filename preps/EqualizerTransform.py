from preps import AbstractPrep
from skimage import exposure

class EqualizerTransform(AbstractPrep):


    def process(self, im):
        return exposure.equalize_hist(im)