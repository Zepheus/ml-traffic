from preps import AbstractPrep
from skimage.transform import resize

class ResizeTransform(AbstractPrep):

    def __init__(self, size=100):
        self.size = size

    def process(self, im):
        return resize(im, [self.size, self.size])

