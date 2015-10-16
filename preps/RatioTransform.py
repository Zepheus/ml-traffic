from preps import AbstractPrep
from skimage.transform import resize

class RatioTransform(AbstractPrep):

    def process(self,im):
        if im.shape[0] != im.shape[1]:
            size = min(im.shape[0],im.shape[1])
            return resize(im,[size,size])

        return im
