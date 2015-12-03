from features import AbstractFeature
from skimage.feature import hog
from preps import ResizeTransform, BWTransform, PrepCombiner


# Calculate hog features for the image
class HogFeature(AbstractFeature):
    # Initialise the feature extractor with a number of orientations (default 5), values indicating
    # the number of pixels per cell (default 8x8), values indicating the number of cells per block
    # (default 3x3) required for the hog feature and a value designating the size of the image after
    # resizing (default 96)
    def __init__(self, orientations=5, pixels_per_cell=(8, 8), cells_per_block=(3, 3), resize=96):
        self.transform = PrepCombiner([BWTransform(), ResizeTransform(resize)])
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    # Process the hog feature
    def process(self, im):
        greyscaled = im.prep(self.transform)
        fd = hog(greyscaled, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                 cells_per_block=self.cells_per_block, visualise=False)
        return fd
