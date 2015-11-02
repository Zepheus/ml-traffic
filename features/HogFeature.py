from features import AbstractFeature
from skimage.feature import hog
from preps import ResizeTransform,BWTransform,PrepCombiner


class HogFeature(AbstractFeature):
    def __init__(self, orientations=5, pixels_per_cell=(8, 8), cells_per_block=(1, 1)):
        self.transform = PrepCombiner([BWTransform(),ResizeTransform(100)])
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def process(self, im):
        greyscaled = im.prep(self.transform)
        fd = hog(greyscaled, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                 cells_per_block=self.cells_per_block, visualise=False)
        return fd
