from skimage.segmentation import slic

from preps import AbstractPrep


class Segmentation(AbstractPrep):

    def process(self,im):
        return slic(im,n_segments=5,max_iter=5)
