from features import AbstractFeature
from skimage.feature import hog
from preps import BWTransform
from preps import ResizeTransform

class HogFeature(AbstractFeature):

    def __init__(self,visualize = False):
        self.transform = BWTransform()
        self.resize = ResizeTransform(100)
        self.visualize = visualize

    def process(self,im):
        grey = self.transform.process(im)
        greyscaled = self.resize.process(grey)
        if (not self.visualize):
            fd = hog(greyscaled,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
            return fd
        else:
            fd,result = hog(greyscaled,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=True)
            return fd,result

