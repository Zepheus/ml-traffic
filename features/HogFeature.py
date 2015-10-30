from features import AbstractFeature
from skimage.feature import hog
from preps import BWTransform
from preps import ResizeTransform


class HogFeature(AbstractFeature):
    def __init__(self, orientations=4, pixels_per_cell=(10, 10), cells_per_block=(2, 2)):
        self.transform = BWTransform()
        self.resize = ResizeTransform(100)
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def process(self, im):
        grey = self.transform.process(im)
        greyscaled = self.resize.process(grey)
        fd = hog(greyscaled, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                 cells_per_block=self.cells_per_block, visualise=False)
        return fd
